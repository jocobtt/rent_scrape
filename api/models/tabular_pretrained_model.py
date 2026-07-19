"""
Pretrained Tabular Deep Learning Models for Tokyo Rent Prediction

This module integrates state-of-the-art pretrained models specifically designed
for tabular data:

1. TabPFN - Prior-Fitted Networks (zero-shot learning for tabular data)
2. TabNet - Google's interpretable tabular learning architecture

These models are integrated with the MLOps pipeline including:
- MLflow tracking and model registry
- Auto-promotion based on metrics
- Model evaluation and validation

Follows the same pattern as challenger_model.py and reg_model.py.
"""

import os
import sys
import logging
import joblib
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.pyfunc
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from datasets import load_dataset
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.auto_promotion_service import AutoPromotionService

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set MLflow tracking URI
mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
mlflow.set_tracking_uri(mlflow_uri)
logger.info(f"MLflow tracking URI: {mlflow_uri}")


def load_data():
    """Load and preprocess the Tokyo rent dataset"""
    try:
        logger.info("Loading dataset from HuggingFace...")
        ds = load_dataset("jbrazzy/tokyo_rent", split="train")
        df = ds.to_pandas()
    except Exception as e:
        logger.warning(f"Failed to load from HuggingFace: {e}")
        logger.info("Loading from local CSV...")
        df = pd.read_csv("../data/tokyo_model.csv")

    # Preprocessing
    logger.info("Preprocessing data...")
    df['ku_name'] = df['ku_name'].astype('category')
    df = pd.get_dummies(df, columns=['ku_name'], prefix='ku')

    df['apartment_type'] = df['apartment_type'].astype('category')
    df = pd.get_dummies(df, columns=['apartment_type'], prefix='apt')

    df['house_type'] = df['house_type'].astype('category')
    df = pd.get_dummies(df, columns=['house_type'], prefix='house')

    if 'address' in df.columns:
        df = df.drop(['address'], axis=1)

    df_for_lineage = df.copy()
    X = df.drop('rent_price', axis=1)
    y = df['rent_price']

    logger.info(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, df_for_lineage


def calculate_mape(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Avoid division by zero
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def train_tabpfn_model():
    """
    Train TabPFN model (Prior-Fitted Networks for tabular data)

    TabPFN is pretrained on millions of synthetic tabular datasets and can
    perform zero-shot or few-shot learning on new tabular tasks.

    Note: TabPFN has limitations:
    - Max 1000 training samples (uses subset if larger)
    - Max 100 features (will select most important if more)
    - Best for smaller datasets
    """
    logger.info("Starting TabPFN model training")
    logger.info("="*70)

    try:
        from tabpfn import TabPFNRegressor

        # Load data
        X_train, X_test, y_train, y_test, df_for_lineage = load_data()

        # TabPFN has limitations on dataset size
        MAX_SAMPLES = 1000
        MAX_FEATURES = 100

        # Handle sample limit
        if len(X_train) > MAX_SAMPLES:
            logger.warning(f"TabPFN limited to {MAX_SAMPLES} samples. Using subset.")
            X_train_sample = X_train.iloc[:MAX_SAMPLES]
            y_train_sample = y_train.iloc[:MAX_SAMPLES]
        else:
            X_train_sample = X_train
            y_train_sample = y_train

        # Handle feature limit
        if X_train.shape[1] > MAX_FEATURES:
            logger.warning(f"TabPFN limited to {MAX_FEATURES} features. Selecting top features.")

            # Use correlation with target to select features
            correlations = X_train.corrwith(y_train).abs().sort_values(ascending=False)
            top_features = correlations.head(MAX_FEATURES).index.tolist()

            X_train_sample = X_train_sample[top_features]
            X_test_sample = X_test[top_features]

            logger.info(f"Selected top {len(top_features)} features")
        else:
            X_test_sample = X_test

        with mlflow.start_run(run_name="tabpfn_model") as run:
            run_id = run.info.run_id

            # Dataset lineage logging (non-fatal)
            try:
                import sys as _sys, os as _os
                _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
                from services.data_lineage_service import DataLineageService
                DataLineageService().log_dataset_to_run(
                    df=df_for_lineage, run_id=run_id,
                    source_description="HuggingFace jbrazzy/tokyo_rent split=train",
                    source_type="huggingface",
                    dataset_name="tokyo_rent_training",
                    context="training",
                )
            except Exception as _e:
                logger.warning(f"Dataset lineage logging failed (non-fatal): {_e}")

            # Log parameters
            mlflow.log_param("model_type", "TabPFN")
            mlflow.log_param("framework", "TabPFN")
            mlflow.log_param("training_samples", len(X_train_sample))
            mlflow.log_param("test_samples", len(X_test))
            mlflow.log_param("n_features", X_train_sample.shape[1])
            mlflow.log_param("pretrained", True)

            # Initialize and train TabPFN
            logger.info("Training TabPFN model (pretrained, minimal training needed)...")
            model = TabPFNRegressor(
                device='cpu',  # Use 'cuda' if GPU available
                N_ensemble_configurations=32  # Number of ensemble members
            )

            model.fit(X_train_sample.values, y_train_sample.values)

            # Make predictions
            logger.info("Making predictions...")
            y_pred = model.predict(X_test_sample.values)

            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            mape = calculate_mape(y_test, y_pred)

            # Log metrics
            mlflow.log_metric("test_rmse", rmse)
            mlflow.log_metric("test_mae", mae)
            mlflow.log_metric("test_r2", r2)
            mlflow.log_metric("test_mse", mse)
            mlflow.log_metric("test_mape", mape)

            logger.info(f"TabPFN Test Metrics:")
            logger.info(f"  RMSE: {rmse:.2f}")
            logger.info(f"  MAE:  {mae:.2f}")
            logger.info(f"  R²:   {r2:.4f}")
            logger.info(f"  MAPE: {mape:.2f}%")

            # Save model
            model_path = "tabpfn_model.pkl"
            joblib.dump(model, model_path)
            mlflow.log_artifact(model_path)

            # Register model
            mlflow.sklearn.log_model(
                model,
                "model",
                registered_model_name="tokyo_rent_tabpfn"
            )

            # Add description
            model_description = f"""
            TabPFN (Prior-Fitted Networks) for Tokyo Rent Prediction

            Pretrained model optimized for tabular data with zero-shot learning capability.

            Performance Metrics:
            - RMSE: {rmse:.2f}
            - MAE: {mae:.2f}
            - R²: {r2:.4f}
            - MAPE: {mape:.2f}%

            Training: {len(X_train_sample)} samples, {X_train_sample.shape[1]} features
            Test: {len(X_test)} samples
            """

            client = mlflow.tracking.MlflowClient()
            versions = client.search_model_versions(f"name='tokyo_rent_tabpfn'")
            if versions:
                latest_version = max([int(v.version) for v in versions])
                client.update_model_version(
                    name="tokyo_rent_tabpfn",
                    version=latest_version,
                    description=model_description.strip()
                )

            logger.info(f"Model registered: tokyo_rent_tabpfn")

            return {
                "run_id": run_id,
                "model_type": "TabPFN",
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
                "mape": mape
            }

    except ImportError:
        logger.error("TabPFN not installed. Install with: pip install tabpfn")
        logger.info("Skipping TabPFN training")
        return None
    except Exception as e:
        logger.error(f"Error training TabPFN: {e}", exc_info=True)
        return None


def train_tabnet_model():
    """
    Train TabNet model (Google's interpretable tabular architecture)

    TabNet provides:
    - Sequential attention mechanism for feature selection
    - Interpretability through attention masks
    - Strong performance on tabular data
    - Can handle both classification and regression
    """
    logger.info("Starting TabNet model training")
    logger.info("="*70)

    try:
        from pytorch_tabnet.tab_model import TabNetRegressor
        import torch

        # Load data
        X_train, X_test, y_train, y_test, df_for_lineage = load_data()
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )

        with mlflow.start_run(run_name="tabnet_model") as run:
            run_id = run.info.run_id

            # Dataset lineage logging (non-fatal)
            try:
                import sys as _sys, os as _os
                _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
                from services.data_lineage_service import DataLineageService
                DataLineageService().log_dataset_to_run(
                    df=df_for_lineage, run_id=run_id,
                    source_description="HuggingFace jbrazzy/tokyo_rent split=train",
                    source_type="huggingface",
                    dataset_name="tokyo_rent_training",
                    context="training",
                )
            except Exception as _e:
                logger.warning(f"Dataset lineage logging failed (non-fatal): {_e}")

            # Log parameters
            mlflow.log_param("model_type", "TabNet")
            mlflow.log_param("framework", "PyTorch-TabNet")
            mlflow.log_param("training_samples", len(X_train_split))
            mlflow.log_param("validation_samples", len(X_val))
            mlflow.log_param("test_samples", len(X_test))
            mlflow.log_param("n_features", X_train.shape[1])

            # TabNet hyperparameters
            tabnet_params = {
                "n_d": 64,  # Width of decision prediction layer
                "n_a": 64,  # Width of attention embedding
                "n_steps": 5,  # Number of steps in the architecture
                "gamma": 1.5,  # Coefficient for feature reusage
                "n_independent": 2,  # Number of independent GLU layers
                "n_shared": 2,  # Number of shared GLU layers
                "lambda_sparse": 1e-4,  # Sparsity regularization
                "optimizer_fn": torch.optim.Adam,
                "optimizer_params": dict(lr=2e-2),
                "scheduler_params": {
                    "step_size": 50,
                    "gamma": 0.9
                },
                "scheduler_fn": torch.optim.lr_scheduler.StepLR,
                "mask_type": "entmax",  # "sparsemax" or "entmax"
                "verbose": 1,
                "seed": 42
            }

            mlflow.log_params(tabnet_params)

            # Initialize TabNet
            logger.info("Training TabNet model...")
            model = TabNetRegressor(**tabnet_params)

            # Train model
            model.fit(
                X_train=X_train_split.values,
                y_train=y_train_split.values.reshape(-1, 1),
                eval_set=[(X_val.values, y_val.values.reshape(-1, 1))],
                eval_name=['valid'],
                eval_metric=['rmse', 'mae'],
                max_epochs=200,
                patience=20,
                batch_size=256,
                virtual_batch_size=128,
                num_workers=0,
                drop_last=False
            )

            # Make predictions
            logger.info("Making predictions...")
            y_pred = model.predict(X_test.values).flatten()

            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            mape = calculate_mape(y_test, y_pred)

            # Log metrics
            mlflow.log_metric("test_rmse", rmse)
            mlflow.log_metric("test_mae", mae)
            mlflow.log_metric("test_r2", r2)
            mlflow.log_metric("test_mse", mse)
            mlflow.log_metric("test_mape", mape)

            logger.info(f"TabNet Test Metrics:")
            logger.info(f"  RMSE: {rmse:.2f}")
            logger.info(f"  MAE:  {mae:.2f}")
            logger.info(f"  R²:   {r2:.4f}")
            logger.info(f"  MAPE: {mape:.2f}%")

            # Get feature importance
            feature_importances = model.feature_importances_
            feature_importance_dict = {
                col: float(imp) for col, imp in
                zip(X_train.columns, feature_importances)
            }

            # Log feature importance
            import json
            importance_path = "tabnet_feature_importance.json"
            with open(importance_path, 'w') as f:
                json.dump(feature_importance_dict, f, indent=2)
            mlflow.log_artifact(importance_path)

            # Save model
            model_path = "tabnet_model.pkl"
            joblib.dump(model, model_path)
            mlflow.log_artifact(model_path)

            # Register model
            mlflow.sklearn.log_model(
                model,
                "model",
                registered_model_name="tokyo_rent_tabnet"
            )

            # Add description
            model_description = f"""
            TabNet (Google) for Tokyo Rent Prediction

            Interpretable deep learning architecture with sequential attention mechanism.

            Performance Metrics:
            - RMSE: {rmse:.2f}
            - MAE: {mae:.2f}
            - R²: {r2:.4f}
            - MAPE: {mape:.2f}%

            Architecture:
            - Decision width (n_d): {tabnet_params['n_d']}
            - Attention width (n_a): {tabnet_params['n_a']}
            - Steps: {tabnet_params['n_steps']}

            Training: {len(X_train_split)} samples
            Validation: {len(X_val)} samples
            Test: {len(X_test)} samples
            """

            client = mlflow.tracking.MlflowClient()
            versions = client.search_model_versions(f"name='tokyo_rent_tabnet'")
            if versions:
                latest_version = max([int(v.version) for v in versions])
                client.update_model_version(
                    name="tokyo_rent_tabnet",
                    version=latest_version,
                    description=model_description.strip()
                )

            logger.info(f"Model registered: tokyo_rent_tabnet")

            return {
                "run_id": run_id,
                "model_type": "TabNet",
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
                "mape": mape,
                "feature_importances": feature_importance_dict
            }

    except ImportError:
        logger.error("TabNet not installed. Install with: pip install pytorch-tabnet")
        logger.info("Skipping TabNet training")
        return None
    except Exception as e:
        logger.error(f"Error training TabNet: {e}", exc_info=True)
        return None


def check_promotion(model_name: str):
    """Check if model should be auto-promoted"""
    logger.info("="*70)
    logger.info(f"CHECKING AUTO-PROMOTION FOR {model_name.upper()}")
    logger.info("="*70)

    try:
        # Add criteria for pretrained models if not exists
        promotion_service = AutoPromotionService(tracking_uri=mlflow_uri)

        # Add TabPFN criteria if training TabPFN
        if model_name == "tokyo_rent_tabpfn" and model_name not in promotion_service.criteria:
            from services.auto_promotion_service import PromotionCriteria
            promotion_service.set_criteria(
                "tokyo_rent_tabpfn",
                PromotionCriteria(
                    model_name="tokyo_rent_tabpfn",
                    max_rmse=15000,  # TabPFN should achieve excellent RMSE
                    min_r2=0.90,
                    max_mape=8.0,
                    min_improvement_percent=2.0,
                    require_better_than_production=True
                )
            )

        # Add TabNet criteria if training TabNet
        if model_name == "tokyo_rent_tabnet" and model_name not in promotion_service.criteria:
            from services.auto_promotion_service import PromotionCriteria
            promotion_service.set_criteria(
                "tokyo_rent_tabnet",
                PromotionCriteria(
                    model_name="tokyo_rent_tabnet",
                    max_rmse=16000,  # TabNet should achieve very good RMSE
                    min_r2=0.89,
                    max_mape=8.5,
                    min_improvement_percent=2.0,
                    require_better_than_production=True
                )
            )

        promotion_result = promotion_service.check_model_for_promotion(
            model_name=model_name,
            dry_run=False
        )

        if promotion_result["promoted"]:
            logger.info(f"✅ MODEL PROMOTED TO PRODUCTION!")
            logger.info(f"   Version: {promotion_result['version']}")
            logger.info(f"   Reason: {promotion_result['reason']}")
        else:
            logger.info(f"⚠️  Model not promoted")
            logger.info(f"   Reason: {promotion_result['reason']}")
            if "failed_criteria" in promotion_result:
                logger.info(f"   Failed criteria: {promotion_result['failed_criteria']}")

    except Exception as e:
        logger.error(f"Auto-promotion check failed: {e}")


def main():
    """Main training function"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Train pretrained tabular models for Tokyo rent prediction"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["tabpfn", "tabnet", "all"],
        help="Which model to train"
    )

    args = parser.parse_args()

    logger.info("="*70)
    logger.info("PRETRAINED TABULAR MODEL TRAINING")
    logger.info("="*70)

    results = {}

    # Train TabPFN
    if args.model in ["tabpfn", "all"]:
        logger.info("\n" + "="*70)
        logger.info("TRAINING TABPFN MODEL")
        logger.info("="*70)

        tabpfn_result = train_tabpfn_model()
        if tabpfn_result:
            results["TabPFN"] = tabpfn_result
            check_promotion("tokyo_rent_tabpfn")

    # Train TabNet
    if args.model in ["tabnet", "all"]:
        logger.info("\n" + "="*70)
        logger.info("TRAINING TABNET MODEL")
        logger.info("="*70)

        tabnet_result = train_tabnet_model()
        if tabnet_result:
            results["TabNet"] = tabnet_result
            check_promotion("tokyo_rent_tabnet")

    # Summary
    logger.info("\n" + "="*70)
    logger.info("TRAINING SUMMARY")
    logger.info("="*70)

    if results:
        for model_name, result in results.items():
            logger.info(f"\n{model_name}:")
            logger.info(f"  RMSE: {result['rmse']:.2f}")
            logger.info(f"  MAE:  {result['mae']:.2f}")
            logger.info(f"  R²:   {result['r2']:.4f}")
            logger.info(f"  MAPE: {result['mape']:.2f}%")
            logger.info(f"  Run ID: {result['run_id']}")

        # Find best model
        best_model = min(results.items(), key=lambda x: x[1]['rmse'])
        logger.info(f"\n🏆 BEST MODEL: {best_model[0]} (RMSE: {best_model[1]['rmse']:.2f})")
    else:
        logger.warning("No models were successfully trained.")
        logger.info("\nTo install required packages:")
        logger.info("  TabPFN: pip install tabpfn")
        logger.info("  TabNet: pip install pytorch-tabnet")

    logger.info("="*70)


if __name__ == "__main__":
    main()
