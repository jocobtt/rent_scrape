"""
Monitoring Dashboard Generator

This module provides functionality to create interactive dashboards
for monitoring model performance and drift over time.
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime, timedelta

import pandas as pd
from evidently.ui.workspace import Workspace
from evidently.ui.dashboards import DashboardPanelCounter, DashboardPanelPlot, PanelValue
from evidently.renderers.html_widgets import WidgetSize

from .data_capture_service import DataCaptureService
from .drift_monitoring_service import DriftMonitoringService, load_reference_data

logger = logging.getLogger(__name__)


class MonitoringDashboard:
    """Create and manage monitoring dashboards"""

    def __init__(
        self,
        workspace_path: str = "./monitoring_workspace",
        predictions_dir: str = "./prediction_logs",
        reference_data_path: Optional[str] = None,
    ):
        """
        Initialize the monitoring dashboard

        Args:
            workspace_path: Path to Evidently workspace
            predictions_dir: Directory with prediction logs
            reference_data_path: Path to reference data (optional)
        """
        self.workspace_path = Path(workspace_path)
        self.predictions_dir = Path(predictions_dir)

        # Initialize services
        self.data_capture = DataCaptureService(storage_dir=str(predictions_dir))

        # Load reference data
        logger.info("Loading reference data for dashboard...")
        self.reference_data = load_reference_data(reference_data_path)

        self.drift_service = DriftMonitoringService(
            reference_data=self.reference_data,
        )

        logger.info("MonitoringDashboard initialized")

    def create_workspace(self, project_name: str = "Tokyo Rent Predictor") -> Workspace:
        """
        Create or load an Evidently workspace

        Args:
            project_name: Name of the project

        Returns:
            Workspace object
        """
        try:
            # Create workspace directory
            self.workspace_path.mkdir(parents=True, exist_ok=True)

            # Initialize workspace
            workspace = Workspace.create(str(self.workspace_path))

            logger.info(f"Created Evidently workspace at {self.workspace_path}")
            return workspace

        except Exception as e:
            logger.error(f"Error creating workspace: {e}")
            raise

    def generate_time_series_dashboard(
        self,
        days_back: int = 30,
        model_type: Optional[str] = None,
    ) -> Dict:
        """
        Generate a time-series dashboard showing drift over time

        Args:
            days_back: Number of days to include
            model_type: Filter by model type (optional)

        Returns:
            Dictionary with dashboard information
        """
        logger.info(f"Generating time-series dashboard for last {days_back} days")

        # Get predictions
        start_date = (datetime.now() - timedelta(days=days_back)).isoformat()
        predictions = self.data_capture.get_recent_predictions(
            model_type=model_type,
            start_date=start_date,
        )

        if predictions.empty:
            logger.warning("No predictions available for dashboard")
            return {"status": "error", "message": "No predictions available"}

        # Group by day for time-series analysis
        predictions["date"] = pd.to_datetime(predictions["timestamp"]).dt.date
        daily_groups = predictions.groupby("date")

        # Collect drift metrics per day
        drift_metrics = []

        for date, group in daily_groups:
            if len(group) < 10:  # Skip days with too few predictions
                continue

            try:
                # Prepare data
                group_prepared = self._prepare_data(group)

                # Check drift for this day
                _, metrics = self.drift_service.generate_data_drift_report(
                    current_data=group_prepared,
                    save_html=False,
                    save_json=False,
                )

                summary = self.drift_service._extract_drift_summary(metrics)
                drift_metrics.append({
                    "date": str(date),
                    "predictions": len(group),
                    "drift_detected": summary["dataset_drift"],
                    "drifted_features": summary["drifted_features"],
                    "drift_share": summary["drift_share"],
                })

            except Exception as e:
                logger.warning(f"Could not compute drift for {date}: {e}")

        if not drift_metrics:
            return {"status": "error", "message": "Not enough data for dashboard"}

        # Create summary
        dashboard_info = {
            "status": "success",
            "time_period": {
                "start": str(predictions["date"].min()),
                "end": str(predictions["date"].max()),
                "days": days_back,
            },
            "total_predictions": len(predictions),
            "daily_metrics": drift_metrics,
            "summary": {
                "days_with_drift": sum(1 for m in drift_metrics if m["drift_detected"]),
                "total_days": len(drift_metrics),
                "avg_drift_share": sum(m["drift_share"] for m in drift_metrics) / len(drift_metrics),
            },
        }

        # Save dashboard data
        dashboard_file = self.workspace_path / f"dashboard_data_{datetime.now().strftime('%Y%m%d')}.json"
        self.workspace_path.mkdir(parents=True, exist_ok=True)

        import json
        with open(dashboard_file, "w") as f:
            json.dump(dashboard_info, f, indent=2)

        logger.info(f"Dashboard data saved to {dashboard_file}")
        dashboard_info["dashboard_file"] = str(dashboard_file)

        return dashboard_info

    def generate_model_comparison_dashboard(
        self,
        days_back: int = 7,
    ) -> Dict:
        """
        Generate a dashboard comparing challenger vs passed model

        Args:
            days_back: Number of days to analyze

        Returns:
            Dictionary with comparison data
        """
        logger.info("Generating model comparison dashboard")

        start_date = (datetime.now() - timedelta(days=days_back)).isoformat()

        # Get predictions for both models
        passed_predictions = self.data_capture.get_recent_predictions(
            model_type="passed",
            start_date=start_date,
        )

        challenger_predictions = self.data_capture.get_recent_predictions(
            model_type="challenger",
            start_date=start_date,
        )

        comparison = {
            "status": "success",
            "time_period": days_back,
            "passed_model": self._compute_model_stats(passed_predictions),
            "challenger_model": self._compute_model_stats(challenger_predictions),
        }

        # Save comparison data
        comparison_file = (
            self.workspace_path / f"model_comparison_{datetime.now().strftime('%Y%m%d')}.json"
        )
        self.workspace_path.mkdir(parents=True, exist_ok=True)

        import json
        with open(comparison_file, "w") as f:
            json.dump(comparison, f, indent=2)

        comparison["comparison_file"] = str(comparison_file)
        logger.info(f"Model comparison saved to {comparison_file}")

        return comparison

    def _compute_model_stats(self, predictions: pd.DataFrame) -> Dict:
        """Compute statistics for a model's predictions"""
        if predictions.empty:
            return {
                "total_predictions": 0,
                "predictions_with_actuals": 0,
            }

        stats = {
            "total_predictions": len(predictions),
            "predictions_with_actuals": predictions["actual_value"].notna().sum(),
            "prediction_stats": {
                "mean": float(predictions["prediction"].mean()),
                "std": float(predictions["prediction"].std()),
                "min": float(predictions["prediction"].min()),
                "max": float(predictions["prediction"].max()),
            },
        }

        # Add performance metrics if we have actual values
        if stats["predictions_with_actuals"] > 0:
            actuals_df = predictions[predictions["actual_value"].notna()]
            errors = actuals_df["actual_value"] - actuals_df["prediction"]

            stats["performance"] = {
                "mae": float(errors.abs().mean()),
                "rmse": float((errors ** 2).mean() ** 0.5),
                "mape": float((errors.abs() / actuals_df["actual_value"]).mean() * 100),
            }

        return stats

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare prediction data for Evidently"""
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

        available_columns = [col for col in feature_columns if col in df.columns]
        return df[available_columns].copy()

    def get_dashboard_summary(self) -> Dict:
        """Get a summary of available dashboard data"""
        dashboard_files = list(self.workspace_path.glob("dashboard_data_*.json"))
        comparison_files = list(self.workspace_path.glob("model_comparison_*.json"))

        return {
            "workspace_path": str(self.workspace_path),
            "dashboard_files": [str(f) for f in dashboard_files],
            "comparison_files": [str(f) for f in comparison_files],
            "total_dashboards": len(dashboard_files),
            "total_comparisons": len(comparison_files),
        }


def create_monitoring_dashboard(
    workspace_path: str = "./monitoring_workspace",
) -> MonitoringDashboard:
    """
    Factory function to create a MonitoringDashboard

    Args:
        workspace_path: Path to workspace directory

    Returns:
        MonitoringDashboard instance
    """
    return MonitoringDashboard(workspace_path=workspace_path)
