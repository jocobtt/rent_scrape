"""
Automated Retraining Service

This service monitors for conditions that trigger model retraining:
1. Data drift detected
2. Model performance degradation
3. Scheduled retraining intervals
4. Manual triggers via API

After retraining, automatically checks if new models should be promoted.
"""

import logging
import threading
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

from .monitoring_reports import MonitoringReportsManager
from .auto_promotion_service import AutoPromotionService
from .training_service import TrainingService

logger = logging.getLogger(__name__)


class AutoRetrainService:
    """Service for automated model retraining and promotion"""

    # Class-level lock — shared across all instances in the same process.
    # Non-blocking acquire means concurrent requests skip rather than queue.
    _retrain_lock = threading.Lock()

    def __init__(
        self,
        drift_threshold: float = 0.3,
        performance_degradation_threshold: float = 0.15,
        min_days_between_retrains: int = 1,
    ):
        """
        Initialize auto-retrain service

        Args:
            drift_threshold: Share of drifted features to trigger retrain (0.0-1.0)
            performance_degradation_threshold: % degradation to trigger retrain (0.0-1.0)
            min_days_between_retrains: Minimum days between retraining attempts
        """
        self.monitoring_manager = MonitoringReportsManager()
        self.auto_promotion = AutoPromotionService()
        self.training_service = TrainingService()

        self.drift_threshold = drift_threshold
        self.performance_degradation_threshold = performance_degradation_threshold
        self.min_days_between_retrains = min_days_between_retrains

        self.last_retrain = {}  # Track last retrain time per model

        logger.info("AutoRetrainService initialized")

    def check_and_retrain(
        self,
        days_back: int = 7,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        Check conditions and retrain if needed

        Args:
            days_back: Days of data to analyze
            dry_run: If True, only check but don't retrain

        Returns:
            Dictionary with check and retrain results
        """
        logger.info("Running automated retrain check...")

        results = {
            "timestamp": datetime.now().isoformat(),
            "dry_run": dry_run,
            "checks_performed": {},
            "retraining_triggered": False,
            "models_retrained": [],
            "models_promoted": [],
        }

        # Check drift
        drift_check = self._check_drift(days_back)
        results["checks_performed"]["drift"] = drift_check

        # Check performance degradation
        performance_check = self._check_performance_degradation(days_back)
        results["checks_performed"]["performance"] = performance_check

        # Determine if retraining needed
        should_retrain = (
            drift_check["requires_retraining"]
            or performance_check["requires_retraining"]
        )

        if should_retrain:
            logger.warning(
                f"Retraining triggered! "
                f"Drift: {drift_check['drift_detected']}, "
                f"Performance degraded: {performance_check['performance_degraded']}"
            )

            if not dry_run:
                # Trigger retraining
                retrain_result = self._retrain_models()
                results["retraining_triggered"] = True
                results["models_retrained"] = retrain_result["models_retrained"]

                # Check if new models should be promoted
                promotion_result = self.auto_promotion.check_and_promote_new_models(
                    dry_run=False
                )
                results["models_promoted"] = promotion_result["models_promoted"]
            else:
                results["retraining_triggered"] = True
                results["message"] = "Dry run - would have retrained models"
        else:
            logger.info("No retraining needed - all systems healthy")
            results["message"] = "No retraining needed"

        return results

    def _check_drift(self, days_back: int = 7) -> Dict[str, Any]:
        """Check for data drift"""
        try:
            drift_result = self.monitoring_manager.run_drift_detection(
                days_back=days_back,
                drift_threshold=self.drift_threshold,
            )

            drift_detected = drift_result.get("drift_detected", False)
            drift_share = drift_result.get("summary", {}).get("drift_share", 0.0)

            requires_retraining = (
                drift_detected and drift_share >= self.drift_threshold
            )

            return {
                "drift_detected": drift_detected,
                "drift_share": drift_share,
                "threshold": self.drift_threshold,
                "requires_retraining": requires_retraining,
                "details": drift_result,
            }

        except Exception as e:
            logger.error(f"Error checking drift: {e}")
            return {
                "drift_detected": False,
                "requires_retraining": False,
                "error": str(e),
            }

    def _check_performance_degradation(self, days_back: int = 7) -> Dict[str, Any]:
        """Check if model performance has degraded"""
        try:
            # Get predictions with actual values
            from .data_capture_service import DataCaptureService

            data_capture = DataCaptureService()
            predictions_with_actuals = data_capture.get_predictions_with_actuals()

            if predictions_with_actuals.empty:
                return {
                    "performance_degraded": False,
                    "requires_retraining": False,
                    "message": "No predictions with actuals to evaluate",
                }

            # Filter to recent predictions
            cutoff_date = datetime.now() - timedelta(days=days_back)
            predictions_with_actuals["timestamp"] = pd.to_datetime(
                predictions_with_actuals["timestamp"]
            )
            recent = predictions_with_actuals[
                predictions_with_actuals["timestamp"] >= cutoff_date
            ]

            if recent.empty:
                return {
                    "performance_degraded": False,
                    "requires_retraining": False,
                    "message": "No recent predictions with actuals",
                }

            # Calculate recent MAE
            import pandas as pd

            recent["error"] = abs(recent["actual_value"] - recent["prediction"])
            recent_mae = recent["error"].mean()

            # Compare with expected performance (from training)
            # For simplicity, use hardcoded thresholds
            # In production, you'd compare with historical performance
            expected_mae = {
                "passed": 19000,
                "challenger": 17500,
            }

            performance_degraded = False
            for model_type, threshold in expected_mae.items():
                model_data = recent[recent["model_type"] == model_type]
                if not model_data.empty:
                    model_mae = model_data["error"].mean()
                    degradation = (model_mae - threshold) / threshold

                    if degradation > self.performance_degradation_threshold:
                        performance_degraded = True
                        logger.warning(
                            f"{model_type} model degraded: "
                            f"MAE {model_mae:.2f} vs expected {threshold:.2f} "
                            f"({degradation:.1%} degradation)"
                        )

            return {
                "performance_degraded": performance_degraded,
                "recent_mae": recent_mae,
                "samples_evaluated": len(recent),
                "requires_retraining": performance_degraded,
            }

        except Exception as e:
            logger.error(f"Error checking performance: {e}")
            return {
                "performance_degraded": False,
                "requires_retraining": False,
                "error": str(e),
            }

    def _retrain_models(self) -> Dict[str, Any]:
        """Trigger model retraining. Skips gracefully if a retrain is already running."""
        if not AutoRetrainService._retrain_lock.acquire(blocking=False):
            logger.warning("Retrain already in progress — skipping concurrent request")
            return {
                "status": "skipped",
                "reason": "retrain_already_in_progress",
                "timestamp": datetime.now().isoformat(),
                "models_retrained": [],
                "errors": [],
            }

        try:
            return self._run_retraining()
        finally:
            AutoRetrainService._retrain_lock.release()

    def _run_retraining(self) -> Dict[str, Any]:
        """Internal: execute the actual retrain subprocesses (called only when lock is held)."""
        logger.info("Starting model retraining...")

        results = {
            "timestamp": datetime.now().isoformat(),
            "models_retrained": [],
            "errors": [],
        }

        # Retrain challenger model
        try:
            logger.info("Retraining challenger model...")
            # Note: This assumes you have a retrain endpoint or method
            # You may need to adapt this based on your setup
            import subprocess

            result = subprocess.run(
                ["python", "models/challenger_model.py"],
                cwd="/Users/jacobbraswell/apps/rent_scrape/api",
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout
            )

            if result.returncode == 0:
                results["models_retrained"].append("challenger")
                logger.info("✅ Challenger model retrained successfully")
                self.last_retrain["challenger"] = datetime.now()
            else:
                error_msg = f"Challenger retrain failed: {result.stderr}"
                results["errors"].append(error_msg)
                logger.error(error_msg)

        except Exception as e:
            error_msg = f"Error retraining challenger: {e}"
            results["errors"].append(error_msg)
            logger.error(error_msg)

        # Retrain passed model
        try:
            logger.info("Retraining passed model...")
            result = subprocess.run(
                ["python", "models/reg_model.py"],
                cwd="/Users/jacobbraswell/apps/rent_scrape/api",
                capture_output=True,
                text=True,
                timeout=600,
            )

            if result.returncode == 0:
                results["models_retrained"].append("passed")
                logger.info("✅ Passed model retrained successfully")
                self.last_retrain["passed"] = datetime.now()
            else:
                error_msg = f"Passed retrain failed: {result.stderr}"
                results["errors"].append(error_msg)
                logger.error(error_msg)

        except Exception as e:
            error_msg = f"Error retraining passed: {e}"
            results["errors"].append(error_msg)
            logger.error(error_msg)

        return results

    def check_if_retrain_allowed(self, model_name: str) -> bool:
        """
        Check if enough time has passed since last retrain

        Args:
            model_name: Name of model

        Returns:
            True if retraining is allowed
        """
        if model_name not in self.last_retrain:
            return True

        time_since_last = datetime.now() - self.last_retrain[model_name]
        return time_since_last.days >= self.min_days_between_retrains

    def force_retrain(
        self,
        model_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Force immediate retraining

        Args:
            model_name: Specific model to retrain, or None for all

        Returns:
            Retraining results
        """
        logger.info(f"Force retraining: {model_name or 'all models'}")

        results = self._retrain_models()

        # Auto-promote if successful
        if results["models_retrained"]:
            promotion_result = self.auto_promotion.check_and_promote_new_models()
            results["promotion_result"] = promotion_result

        return results
