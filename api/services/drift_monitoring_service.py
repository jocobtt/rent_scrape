"""
Drift Monitoring Service using Evidently AI

This service provides drift detection and monitoring capabilities for the Tokyo Rent Predictor models.
It tracks:
- Data drift: Changes in input feature distributions
- Prediction drift: Changes in model output distributions
- Model performance: Degradation in prediction quality
- Data quality: Missing values, out-of-range values, etc.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import pandas as pd
from evidently import Report
from evidently.core.datasets import DataDefinition, Regression, Dataset
from evidently.presets import DataDriftPreset, DataSummaryPreset, RegressionPreset
from evidently.metrics import (
    ValueDrift,
    DriftedColumnsCount,
    DatasetMissingValueCount,
    RMSE,
    MAE,
    R2Score,
)
from evidently.tests import lte

logger = logging.getLogger(__name__)

# Evidently's ValueDrift reports a *distance* per column (normed Wasserstein, Jensen-Shannon, ...) and
# flags drift when it EXCEEDS the method's threshold (0.1 by default). Larger = more drift.
DEFAULT_DRIFT_THRESHOLD = 0.1
# Deposit / key money are optional request fields, so they are legitimately missing in prediction logs
OPTIONAL_COLUMNS = ("rei_price", "shikikin")
# Evidently's per-column distance threshold is fixed (0.1), so on small windows ordinary sampling noise
# crosses it: on data drawn from the reference distribution, >=30% of features were flagged in 88% of
# 60-row windows, 48% of 100-row, 4% of 200-row and 0% of 400-row ones. Below this many rows "drift" is
# not a decision anyone should act on (hourly auto-retrain would retrain on noise).
MIN_DRIFT_SAMPLES = int(os.getenv("DRIFT_MIN_SAMPLES", "200"))


class DriftMonitoringService:
    """Service for monitoring data and prediction drift using Evidently AI"""

    def __init__(
        self,
        reference_data: pd.DataFrame,
        reports_dir: str = "./monitoring_reports",
        feature_columns: Optional[List[str]] = None,
        target_column: str = "rent_price",
        prediction_column: str = "prediction",
    ):
        self.reference_data = reference_data
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        self.target_column = target_column
        self.prediction_column = prediction_column

        if feature_columns is None:
            self.feature_columns = [
                "sqr_m",
                "rei_price",
                "shikikin",
                "maintenence_price",
                "year_built",
                "floor",
                "eki_walk",
            ]
        else:
            self.feature_columns = feature_columns

        logger.info(
            f"Initialized DriftMonitoringService with {len(self.reference_data)} reference samples"
        )

    # ------------------------------------------------------------------
    # Dataset helpers
    # ------------------------------------------------------------------

    def _usable_columns(self, current: pd.DataFrame) -> List[str]:
        """Feature columns that can be compared: present in both frames and not entirely empty in either.
        (Evidently raises on an all-NaN column, e.g. when every logged request omitted deposit / key money.)"""
        ref = self.reference_data
        return [c for c in self.feature_columns
                if c in ref.columns and c in current.columns and ref[c].notna().any() and current[c].notna().any()]

    def _make_feature_dataset(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> Dataset:
        """Wrap a DataFrame in a feature-only DataDefinition (no regression task)."""
        available = [c for c in (columns if columns is not None else self.feature_columns) if c in df.columns]
        data_def = DataDefinition(numerical_columns=available)
        return Dataset.from_pandas(df[available], data_definition=data_def)

    def _make_regression_dataset(self, df: pd.DataFrame) -> Dataset:
        """Wrap a DataFrame in a DataDefinition that includes the regression task.

        If the prediction column is absent (e.g. reference data without predictions),
        it is filled with the target column values so the schema is consistent.
        """
        available_features = [c for c in self.feature_columns if c in df.columns]
        df_work = df.copy()

        # Ensure target column is present
        if self.target_column not in df_work.columns:
            df_work[self.target_column] = float("nan")

        # Ensure prediction column is present (fill from target if absent)
        if self.prediction_column not in df_work.columns:
            df_work[self.prediction_column] = df_work[self.target_column]

        all_cols = available_features + [self.target_column, self.prediction_column]
        data_def = DataDefinition(
            numerical_columns=all_cols,
            regression=[
                Regression(
                    name="default",
                    target=self.target_column,
                    prediction=self.prediction_column,
                )
            ],
        )
        return Dataset.from_pandas(df_work[all_cols], data_definition=data_def)

    # ------------------------------------------------------------------
    # Reports
    # ------------------------------------------------------------------

    def generate_data_drift_report(
        self,
        current_data: pd.DataFrame,
        save_html: bool = True,
        save_json: bool = True,
        drift_share_threshold: float = 0.5,
    ) -> Tuple[object, Dict]:
        """Generate a data drift report. The dataset counts as drifted when at least
        `drift_share_threshold` of the compared features drifted.

        Returns:
            (snapshot, summary_dict)
        """
        logger.info("Generating data drift report...")

        columns = self._usable_columns(current_data)
        ref_ds = self._make_feature_dataset(self.reference_data, columns)
        cur_ds = self._make_feature_dataset(current_data, columns)

        report = Report(metrics=[DataDriftPreset()])
        snapshot = report.run(reference_data=ref_ds, current_data=cur_ds)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if save_html:
            html_path = self.reports_dir / f"data_drift_report_{timestamp}.html"
            snapshot.save_html(str(html_path))
            logger.info(f"Saved HTML report to {html_path}")

        if save_json:
            json_path = self.reports_dir / f"data_drift_report_{timestamp}.json"
            snapshot.save_json(str(json_path))
            logger.info(f"Saved JSON report to {json_path}")

        summary = self._extract_drift_summary(snapshot, drift_share_threshold)
        summary["current_rows"] = len(current_data)
        summary["reliable"] = len(current_data) >= MIN_DRIFT_SAMPLES
        if not summary["reliable"]:
            summary["dataset_drift"] = False  # per-column numbers are still reported, as information only
        logger.info(f"Data drift summary: {summary}")

        return snapshot, summary

    def generate_data_quality_report(
        self,
        current_data: pd.DataFrame,
        save_html: bool = True,
        save_json: bool = True,
    ) -> Tuple[object, Dict]:
        """Generate a data quality report.

        Returns:
            (snapshot, metrics_dict)
        """
        logger.info("Generating data quality report...")

        ref_ds = self._make_feature_dataset(self.reference_data)
        cur_ds = self._make_feature_dataset(current_data)

        report = Report(metrics=[DataSummaryPreset(), DatasetMissingValueCount()])
        snapshot = report.run(reference_data=ref_ds, current_data=cur_ds)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if save_html:
            html_path = self.reports_dir / f"data_quality_report_{timestamp}.html"
            snapshot.save_html(str(html_path))
            logger.info(f"Saved HTML report to {html_path}")

        if save_json:
            json_path = self.reports_dir / f"data_quality_report_{timestamp}.json"
            snapshot.save_json(str(json_path))
            logger.info(f"Saved JSON report to {json_path}")

        missing_count = self._extract_missing_count(snapshot)
        return snapshot, {"missing_value_count": missing_count}

    def generate_model_performance_report(
        self,
        current_data: pd.DataFrame,
        save_html: bool = True,
        save_json: bool = True,
    ) -> Tuple[object, Dict]:
        """Generate a model performance report (requires actual target values).

        Returns:
            (snapshot, performance_metrics_dict)
        """
        logger.info("Generating model performance report...")

        if self.target_column not in current_data.columns:
            raise ValueError(
                f"Target column '{self.target_column}' not found in current_data. "
                "Cannot generate performance report without actual values."
            )
        if self.prediction_column not in current_data.columns:
            raise ValueError(
                f"Prediction column '{self.prediction_column}' not found in current_data."
            )

        ref_ds = self._make_regression_dataset(self.reference_data)
        cur_ds = self._make_regression_dataset(current_data)

        report = Report(metrics=[RMSE(), MAE(), R2Score()])
        snapshot = report.run(reference_data=ref_ds, current_data=cur_ds)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if save_html:
            html_path = self.reports_dir / f"model_performance_report_{timestamp}.html"
            snapshot.save_html(str(html_path))
            logger.info(f"Saved HTML report to {html_path}")

        if save_json:
            json_path = self.reports_dir / f"model_performance_report_{timestamp}.json"
            snapshot.save_json(str(json_path))
            logger.info(f"Saved JSON report to {json_path}")

        perf = self._extract_performance_metrics(snapshot)
        return snapshot, perf

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def run_drift_tests(
        self,
        current_data: pd.DataFrame,
        drift_share_threshold: float = 0.3,
    ) -> Tuple[object, bool]:
        """Run automated drift tests and return pass/fail status.

        Tests pass when:
        - Drifted column count <= drift_share_threshold * total_feature_count
        - Missing value count == 0

        Returns:
            (snapshot, tests_passed)
        """
        logger.info("Running drift tests...")

        columns = self._usable_columns(current_data)
        ref_ds = self._make_feature_dataset(self.reference_data, columns)
        cur_ds = self._make_feature_dataset(current_data, columns)

        report = Report(metrics=[ValueDrift(column=col) for col in columns])
        snapshot = report.run(reference_data=ref_ds, current_data=cur_ds)

        drift_summary = self._extract_drift_summary(snapshot, drift_share_threshold)
        max_drifted = int(drift_share_threshold * len(self.feature_columns))
        required = [c for c in self.feature_columns if c in current_data.columns and c not in OPTIONAL_COLUMNS]
        missing_ok = not bool(current_data[required].isna().any().any())
        tests_passed = drift_summary["drifted_features"] <= max_drifted and missing_ok

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot.save_json(str(self.reports_dir / f"drift_tests_{timestamp}.json"))
        snapshot.save_html(str(self.reports_dir / f"drift_tests_{timestamp}.html"))

        logger.info(f"Drift tests {'PASSED' if tests_passed else 'FAILED'}")
        return snapshot, tests_passed

    # ------------------------------------------------------------------
    # Column-level drift
    # ------------------------------------------------------------------

    def check_column_drift(
        self,
        current_data: pd.DataFrame,
        column_name: str,
    ) -> Dict:
        """Check drift for a specific column."""
        logger.info(f"Checking drift for column: {column_name}")

        available = [column_name] + [
            c for c in self.feature_columns if c in current_data.columns and c != column_name
        ]
        data_def = DataDefinition(numerical_columns=available)
        ref_df = self.reference_data[[c for c in available if c in self.reference_data.columns]]
        cur_df = current_data[[c for c in available if c in current_data.columns]]
        ref_ds = Dataset.from_pandas(ref_df, data_definition=data_def)
        cur_ds = Dataset.from_pandas(cur_df, data_definition=data_def)

        report = Report(metrics=[ValueDrift(column=column_name)])
        snapshot = report.run(reference_data=ref_ds, current_data=cur_ds)

        drift_info = {"column": column_name, "drift_detected": False, "drift_score": None}

        try:
            for entry in snapshot.dict().get("metrics", []):
                if entry.get("config", {}).get("column") == column_name:
                    score = entry.get("value")
                    threshold = entry["config"].get("threshold") or DEFAULT_DRIFT_THRESHOLD
                    drift_info["drift_score"] = score
                    drift_info["threshold"] = threshold
                    drift_info["drift_detected"] = score is not None and score > threshold
                    break
        except Exception as e:
            logger.warning(f"Could not extract column drift metrics: {e}")

        return drift_info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_drift_summary(self, snapshot, drift_share_threshold: float = 0.5) -> Dict:
        """Count drifted features in a drift snapshot.

        Each ValueDrift result carries a distance and the threshold of the method Evidently chose for
        that column; a column drifted when distance > threshold.
        """
        summary = {
            "total_features": len(self.feature_columns),
            "drifted_features": 0,
            "drift_share": 0.0,
            "drifted_columns": [],
            "dataset_drift": False,
        }
        try:
            compared, drifted = [], []
            for entry in snapshot.dict().get("metrics", []):
                config = entry.get("config", {})
                value = entry.get("value")
                if not str(config.get("type", "")).endswith("ValueDrift") or value is None:
                    continue
                compared.append(config.get("column"))
                if value > (config.get("threshold") or DEFAULT_DRIFT_THRESHOLD):
                    drifted.append(config.get("column"))
            if compared:
                summary["total_features"] = len(compared)
                summary["drifted_features"] = len(drifted)
                summary["drift_share"] = len(drifted) / len(compared)
                summary["drifted_columns"] = drifted
                summary["dataset_drift"] = summary["drift_share"] >= drift_share_threshold
        except Exception as e:
            logger.warning(f"Could not extract drift summary: {e}")
        return summary

    def _extract_missing_count(self, snapshot) -> int:
        """Extract total missing value count from a quality snapshot."""
        try:
            for result in snapshot.metric_results.values():
                dn = getattr(result, "display_name", "")
                if "missing" in dn.lower():
                    return int(getattr(result, "value", 0) or 0)
        except Exception:
            pass
        return 0

    def _extract_performance_metrics(self, snapshot) -> Dict:
        """Extract regression performance metrics from a performance snapshot."""
        perf = {"rmse": None, "mae": None, "r2": None}
        try:
            for result in snapshot.metric_results.values():
                dn = getattr(result, "display_name", "")
                val = getattr(result, "value", None)
                dn_lower = dn.lower()
                if "rmse" in dn_lower:
                    perf["rmse"] = val
                elif "mean absolute" in dn_lower or "mae" in dn_lower:
                    # MAE returns MeanStdValue (has .mean, not .value)
                    perf["mae"] = getattr(result, "mean", val)
                elif "r2" in dn_lower or "r^2" in dn_lower:
                    perf["r2"] = val
        except Exception as e:
            logger.warning(f"Could not extract performance metrics: {e}")
        return perf

    def get_latest_report_path(self, report_type: str = "data_drift") -> Optional[Path]:
        """Get the path to the most recent report of a given type."""
        pattern = f"{report_type}_report_*.html"
        reports = list(self.reports_dir.glob(pattern))
        if not reports:
            return None
        return max(reports, key=lambda p: p.stat().st_mtime)


def load_reference_data(data_path: Optional[str] = None) -> pd.DataFrame:
    """Load reference/baseline data for drift monitoring."""
    from services.data_cleaning import load_training_frame
    # explicit path > DATASET_PATH env var > newest scraped dataset
    df = load_training_frame(data_path)
    # Scraped data calls it nearest_eki_walk; the API / prediction logs use eki_walk
    if "eki_walk" not in df.columns and "nearest_eki_walk" in df.columns:
        df = df.rename(columns={"nearest_eki_walk": "eki_walk"})

    feature_columns = [
        "sqr_m",
        "rei_price",
        "shikikin",
        "maintenence_price",
        "year_built",
        "floor",
        "eki_walk",
    ]

    columns_to_keep = feature_columns + (["rent_price"] if "rent_price" in df.columns else [])
    available = [c for c in columns_to_keep if c in df.columns]
    df = df[available]

    logger.info(f"Loaded {len(df)} reference samples with {len(available)} columns")
    return df
