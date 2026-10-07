"""
Monitoring Reports Generator

This module provides high-level functions for generating and managing
Evidently monitoring reports for the Tokyo Rent Predictor.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta

import pandas as pd

from .drift_monitoring_service import DriftMonitoringService, load_reference_data
from .data_capture_service import DataCaptureService

logger = logging.getLogger(__name__)


class MonitoringReportsManager:
    """Manager for generating and coordinating monitoring reports"""

    def __init__(
        self,
        reference_data_path: Optional[str] = None,
        reports_dir: str = "./monitoring_reports",
        predictions_dir: str = "./prediction_logs",
    ):
        """
        Initialize the monitoring reports manager

        Args:
            reference_data_path: Path to reference data CSV (optional)
            reports_dir: Directory to save reports
            predictions_dir: Directory where predictions are logged
        """
        self.reports_dir = Path(reports_dir)
        self.predictions_dir = Path(predictions_dir)

        # Load reference data
        logger.info("Loading reference data for drift monitoring...")
        self.reference_data = load_reference_data(reference_data_path)

        # Initialize services
        self.drift_service = DriftMonitoringService(
            reference_data=self.reference_data,
            reports_dir=reports_dir,
        )

        self.data_capture = DataCaptureService(storage_dir=predictions_dir)

        logger.info("MonitoringReportsManager initialized")

    def generate_all_reports(
        self,
        days_back: int = 7,
        model_type: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Generate all available monitoring reports

        Args:
            days_back: Number of days of prediction data to analyze
            model_type: Filter by model type (optional)

        Returns:
            Dictionary with paths to generated reports
        """
        logger.info(f"Generating all monitoring reports for last {days_back} days")

        # Get current predictions
        start_date = (datetime.now() - timedelta(days=days_back)).isoformat()
        current_data = self.data_capture.get_recent_predictions(
            model_type=model_type,
            start_date=start_date,
        )

        if current_data.empty:
            logger.warning("No prediction data available for report generation")
            return {"error": "No prediction data available"}

        # Prepare data for Evidently
        current_data = self._prepare_data_for_evidently(current_data)

        report_paths = {}

        # Generate data drift report
        try:
            logger.info("Generating data drift report...")
            drift_report, _ = self.drift_service.generate_data_drift_report(
                current_data=current_data,
                save_html=True,
                save_json=True,
            )
            drift_path = self.drift_service.get_latest_report_path("data_drift")
            report_paths["data_drift"] = str(drift_path) if drift_path else None
        except Exception as e:
            logger.error(f"Error generating data drift report: {e}")
            report_paths["data_drift"] = None

        # Generate data quality report
        try:
            logger.info("Generating data quality report...")
            quality_report, _ = self.drift_service.generate_data_quality_report(
                current_data=current_data,
                save_html=True,
                save_json=True,
            )
            quality_path = self.drift_service.get_latest_report_path("data_quality")
            report_paths["data_quality"] = str(quality_path) if quality_path else None
        except Exception as e:
            logger.error(f"Error generating data quality report: {e}")
            report_paths["data_quality"] = None

        # Generate model performance report (only if we have actual values)
        predictions_with_actuals = self.data_capture.get_predictions_with_actuals()

        if not predictions_with_actuals.empty:
            try:
                logger.info("Generating model performance report...")
                predictions_with_actuals = self._prepare_data_for_evidently(
                    predictions_with_actuals
                )
                perf_report, _ = self.drift_service.generate_model_performance_report(
                    current_data=predictions_with_actuals,
                    save_html=True,
                    save_json=True,
                )
                perf_path = self.drift_service.get_latest_report_path(
                    "model_performance"
                )
                report_paths["model_performance"] = (
                    str(perf_path) if perf_path else None
                )
            except Exception as e:
                logger.error(f"Error generating model performance report: {e}")
                report_paths["model_performance"] = None
        else:
            logger.info(
                "No predictions with actual values available for performance report"
            )
            report_paths["model_performance"] = None

        logger.info(f"Report generation complete. Generated {len(report_paths)} reports")
        return report_paths

    def run_drift_detection(
        self,
        days_back: int = 7,
        model_type: Optional[str] = None,
        drift_threshold: float = 0.3,
    ) -> Dict:
        """
        Run drift detection and return results

        Args:
            days_back: Number of days of data to analyze
            model_type: Filter by model type
            drift_threshold: Maximum allowed share of drifted features

        Returns:
            Dictionary with drift detection results
        """
        logger.info("Running drift detection...")

        # Get current predictions
        start_date = (datetime.now() - timedelta(days=days_back)).isoformat()
        current_data = self.data_capture.get_recent_predictions(
            model_type=model_type,
            start_date=start_date,
        )

        if current_data.empty:
            return {
                "status": "error",
                "message": "No prediction data available",
            }

        # Prepare data
        current_data = self._prepare_data_for_evidently(current_data)

        # Run drift tests
        test_suite, tests_passed = self.drift_service.run_drift_tests(
            current_data=current_data,
            drift_share_threshold=drift_threshold,
        )

        # Generate drift report
        _, drift_summary = self.drift_service.generate_data_drift_report(
            current_data=current_data,
            save_html=True,
            save_json=False,
        )

        result = {
            "status": "success",
            "tests_passed": tests_passed,
            "drift_detected": drift_summary.get("dataset_drift", False),
            "summary": drift_summary,
            "report_path": str(
                self.drift_service.get_latest_report_path("data_drift")
            ),
            "timestamp": datetime.now().isoformat(),
        }

        logger.info(f"Drift detection complete: {result}")
        return result

    def compare_models_drift(
        self,
        days_back: int = 7,
    ) -> Dict:
        """
        Compare drift between challenger and passed models

        Args:
            days_back: Number of days to analyze

        Returns:
            Dictionary with comparison results
        """
        logger.info("Comparing drift between models...")

        start_date = (datetime.now() - timedelta(days=days_back)).isoformat()

        # Get predictions for each model
        passed_data = self.data_capture.get_recent_predictions(
            model_type="passed",
            start_date=start_date,
        )

        challenger_data = self.data_capture.get_recent_predictions(
            model_type="challenger",
            start_date=start_date,
        )

        results = {
            "passed_model": {"predictions": len(passed_data), "drift": None},
            "challenger_model": {"predictions": len(challenger_data), "drift": None},
        }

        # Check drift for passed model
        if not passed_data.empty:
            passed_data = self._prepare_data_for_evidently(passed_data)
            _, passed_summary = self.drift_service.generate_data_drift_report(
                current_data=passed_data,
                save_html=False,
                save_json=False,
            )
            results["passed_model"]["drift"] = passed_summary

        # Check drift for challenger model
        if not challenger_data.empty:
            challenger_data = self._prepare_data_for_evidently(challenger_data)
            _, challenger_summary = self.drift_service.generate_data_drift_report(
                current_data=challenger_data,
                save_html=False,
                save_json=False,
            )
            results["challenger_model"]["drift"] = challenger_summary

        return results

    def get_monitoring_summary(self) -> Dict:
        """
        Get a summary of monitoring status and recent reports

        Returns:
            Dictionary with monitoring summary
        """
        logger.info("Getting monitoring summary...")

        # Get prediction statistics
        stats = self.data_capture.get_statistics()

        # Get recent reports
        recent_reports = {
            "data_drift": self._get_report_info("data_drift"),
            "data_quality": self._get_report_info("data_quality"),
            "model_performance": self._get_report_info("model_performance"),
            "drift_tests": self._get_report_info("drift_tests"),
        }

        summary = {
            "prediction_statistics": stats,
            "recent_reports": recent_reports,
            "timestamp": datetime.now().isoformat(),
        }

        return summary

    def _prepare_data_for_evidently(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare prediction data for Evidently reports"""
        # Ensure we have the required columns
        feature_columns = [
            "sqr_m",
            "rei_price",
            "shikikin",
            "maintenence_price",
            "year_built",
            "floor",
            "eki_walk",
            "prediction",
        ]

        # Add actual value column if it exists
        if "actual_value" in df.columns and df["actual_value"].notna().any():
            # Rename actual_value to rent_price for Evidently
            df_copy = df.copy()
            df_copy["rent_price"] = df_copy["actual_value"]
            feature_columns.append("rent_price")
        else:
            df_copy = df.copy()

        # Select only the columns we need
        available_columns = [col for col in feature_columns if col in df_copy.columns]
        df_prepared = df_copy[available_columns]

        return df_prepared

    def _get_report_info(self, report_type: str) -> Optional[Dict]:
        """Get information about the most recent report of a type"""
        report_path = self.drift_service.get_latest_report_path(report_type)

        if not report_path or not report_path.exists():
            return None

        stat = report_path.stat()
        return {
            "path": str(report_path),
            "created": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "size_kb": stat.st_size / 1024,
        }


def create_monitoring_manager(
    reference_data_path: Optional[str] = None,
) -> MonitoringReportsManager:
    """
    Factory function to create a MonitoringReportsManager

    Args:
        reference_data_path: Optional path to reference data

    Returns:
        MonitoringReportsManager instance
    """
    return MonitoringReportsManager(reference_data_path=reference_data_path)
