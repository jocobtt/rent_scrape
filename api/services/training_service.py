"""
Training service for handling model retraining operations.
"""

import logging
import os
import joblib
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any

from utils.scrape_data import ScrapeData, TOKYO_WARD_CODES, build_suumo_url

logger = logging.getLogger(__name__)


class TrainingService:
    """Service class for handling model training and retraining."""

    def __init__(self):
        self.models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        self.dataset_versions_dir = Path(os.path.dirname(__file__)).parent / "dataset_versions"
        self.dataset_versions_dir.mkdir(parents=True, exist_ok=True)
    
    def retrain_models(self, url: str, wait_time_min: int, wait_time_max: int, 
                      pages: tuple, models_to_retrain: List[str]) -> Dict[str, Any]:
        """
        Retrain specified models with fresh data.
        
        Args:
            url: URL to scrape data from
            wait_time_min: Minimum wait time between requests
            wait_time_max: Maximum wait time between requests
            pages: Tuple of (start_page, end_page)
            models_to_retrain: List of model names to retrain
            
        Returns:
            Dict containing training results
        """
        try:
            # Scrape and prepare data
            scraper = ScrapeData(url, wait_time_min, wait_time_max, pages)
            scraped_data = scraper.scrape()
            cleaned_data = scraper.clean_data(scraped_data)
            quality = self._check_data_quality(scraper, cleaned_data)

            # Save temporary training data (needed by challenger_model.load_data)
            temp_file = "temp_training_data.csv"
            cleaned_data.to_csv(temp_file, index=False)

            # Persist to versioned store BEFORE training so data is never lost,
            # even for scrapes that fail the quality gate (useful for postmortem)
            versioned_path = self._persist_scraped_dataset(cleaned_data, url)

            if quality["blocking"]:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                return {
                    "status": "error",
                    "message": "Data quality gate failed; scraped data was not used for training",
                    "retrained_models": {},
                    "dataset_version_path": str(versioned_path),
                    "data_quality": quality,
                }

            retrained_models = {}

            for model_name in models_to_retrain:
                if model_name == "challenger":
                    retrained_models["challenger"] = self._retrain_challenger_model(
                        temp_file, source_url=url
                    )
                elif model_name == "passed":
                    retrained_models["passed"] = self._retrain_passed_model(temp_file)
                else:
                    logger.warning(f"Unknown model type: {model_name}")

            # Remove temp file; the versioned copy in dataset_versions/ is kept
            if os.path.exists(temp_file):
                os.remove(temp_file)

            return {
                "message": f"Successfully retrained models: {list(retrained_models.keys())}",
                "retrained_models": retrained_models,
                "status": "success",
                "dataset_version_path": str(versioned_path),
                "data_quality": quality,
            }

        except Exception as e:
            logger.error(f"Training service error: {str(e)}")
            raise
    
    def retrain_with_wards(
        self,
        ward_codes: list | None = None,
        pages_per_ward: int = 10,
        wait_time_min: int = 2,
        wait_time_max: int = 5,
        models_to_retrain: list | None = None,
    ) -> dict:
        """
        Retrain models using multi-ward scraping for a broader, more
        representative Tokyo dataset.

        Args:
            ward_codes:       List of Suumo sc codes.  Defaults to all 23
                              Tokyo special wards.
            pages_per_ward:   Pages to scrape per ward (~30 listings/page).
            wait_time_min/max: Politeness delay between requests (seconds).
            models_to_retrain: Model names to retrain.  Defaults to
                               ["challenger", "passed"].

        Returns:
            Same structure as retrain_models().
        """
        if models_to_retrain is None:
            models_to_retrain = ["challenger", "passed"]
        if ward_codes is None:
            ward_codes = TOKYO_WARD_CODES

        try:
            raw = ScrapeData.scrape_wards(
                ward_codes=ward_codes,
                pages_per_ward=pages_per_ward,
                wait_time_min=wait_time_min,
                wait_time_max=wait_time_max,
            )
            if raw.empty:
                return {
                    "status": "error",
                    "message": "No listings collected from any ward",
                    "retrained_models": {},
                }

            # Use any ward's URL for provenance metadata
            source_url = build_suumo_url() + f"&wards={','.join(ward_codes)}"
            scraper = ScrapeData(build_suumo_url(), pages=(1, 2))
            cleaned = scraper.clean_data(raw)
            quality = self._check_data_quality(scraper, cleaned)

            temp_file = "temp_training_data.csv"
            cleaned.to_csv(temp_file, index=False)
            versioned_path = self._persist_scraped_dataset(cleaned, source_url)

            if quality["blocking"]:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                return {
                    "status": "error",
                    "message": "Data quality gate failed; scraped data was not used for training",
                    "retrained_models": {},
                    "wards_scraped": len(ward_codes),
                    "total_listings": len(cleaned),
                    "dataset_version_path": str(versioned_path),
                    "data_quality": quality,
                }

            retrained_models = {}
            for model_name in models_to_retrain:
                if model_name == "challenger":
                    retrained_models["challenger"] = self._retrain_challenger_model(
                        temp_file, source_url=source_url
                    )
                elif model_name == "passed":
                    retrained_models["passed"] = self._retrain_passed_model(temp_file)
                else:
                    logger.warning("Unknown model type: %s", model_name)

            if os.path.exists(temp_file):
                os.remove(temp_file)

            return {
                "message": f"Successfully retrained models: {list(retrained_models.keys())}",
                "retrained_models": retrained_models,
                "status": "success",
                "wards_scraped": len(ward_codes),
                "total_listings": len(cleaned),
                "dataset_version_path": str(versioned_path),
                "data_quality": quality,
            }

        except Exception as e:
            logger.error("retrain_with_wards error: %s", e)
            raise

    def _check_data_quality(self, scraper: ScrapeData, cleaned_data) -> Dict[str, Any]:
        """
        Run the scraper's data quality checks against freshly cleaned data
        and log the results. `quality["blocking"]` signals data too broken
        (near-empty, mostly-zero prices, etc.) to safely train on.
        """
        quality = scraper.validate_scraped_data(cleaned_data)
        if quality["warnings"]:
            logger.warning("Data quality warnings: %s", quality["warnings"])
        if quality["blocking"]:
            logger.error("Data quality gate failed: %s", quality["critical"])
        return quality

    def _persist_scraped_dataset(self, df, source_url: str) -> Path:
        """
        Persist a scraped DataFrame to content-addressed storage before training.

        Called before training starts so the data is never lost even if training fails.

        Args:
            df:         Cleaned scraped DataFrame.
            source_url: The URL that was scraped (stored for provenance).

        Returns:
            Absolute path to the versioned data.csv, or temp path on failure.
        """
        try:
            from services.data_lineage_service import DataLineageService
            lineage_svc = DataLineageService()
            date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            source_desc = f"Suumo scrape {date_str} url={source_url}"
            manifest = lineage_svc.save_versioned_dataset(df, source_description=source_desc, kind="scrape")
            logger.info("Scraped dataset persisted: %s", manifest["hash_short"])
            return Path(manifest["data_path"])
        except Exception as e:
            logger.warning("Dataset persistence failed (non-fatal): %s", e)
            return Path("temp_training_data.csv")

    def _retrain_challenger_model(self, data_file: str, source_url: str = ""):
        """Retrain the challenger model."""
        try:
            from models.challenger_model import load_data, train_and_log_model

            X_train, X_test, y_train, y_test, df_for_lineage = load_data(data_file)
            params = {
                'task': 'train',
                'boosting_type': 'gbdt',
                'objective': 'regression',
                'metric': ['l1', 'l2'],
                'learning_rate': 0.005,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.7,
                'bagging_freq': 10,
                'verbose': 0,
                "max_depth": 8,
                "num_leaves": 128,
                "max_bin": 512,
                "num_iterations": 100000,
            }

            model = train_and_log_model(
                X_train, X_test, y_train, y_test, params,
                df_for_lineage=df_for_lineage,
                dataset_source_path=data_file,
                source_type="scraped" if source_url else "local",
            )
            return model
            
        except Exception as e:
            logger.error(f"Challenger model retraining failed: {str(e)}")
            raise
    
    def _retrain_passed_model(self, data_file: str = None):
        """Retrain the passed model."""
        try:
            from models.reg_model import load_data, train_model, evaluate_model, log_mlflow

            X_train, X_test, y_train, y_test, df_for_lineage = load_data(data_file)
            models = train_model(X_train, y_train, alpha=1.0)

            # Find the best model
            best_mse = float('inf')
            best_model = None
            best_model_name = ""

            for model_name, model in models.items():
                mse = evaluate_model(model, X_test, y_test)
                if mse < best_mse:
                    best_mse = mse
                    best_model = model
                    best_model_name = model_name

            # Log to MLflow if model is good enough
            if best_mse < 4:  # Quality threshold
                log_mlflow(
                    best_model_name,
                    best_model,
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    mlflow_name="reg_passed_model",
                    best_mse=best_mse,
                    df_for_lineage=df_for_lineage,
                )

            # Save the best model
            model_path = os.path.join(self.models_dir, "passed-model.joblib")
            joblib.dump(best_model, model_path)

            logger.info(f"Retrained passed model: {best_model_name} with MSE: {best_mse}")

            return best_model

        except Exception as e:
            logger.error(f"Passed model retraining failed: {str(e)}")
            raise