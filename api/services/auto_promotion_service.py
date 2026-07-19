"""
Automated Model Promotion Service

This service automatically promotes models to production based on:
1. Metric thresholds (RMSE, R², MAPE)
2. Quality gates
3. Comparison with current production model
4. Safety checks

Features:
- Metric-based promotion criteria
- Automatic rollback if new model worse
- Configurable promotion rules
- Dry-run mode for testing
- Integration with MLflow Model Registry
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

logger = logging.getLogger(__name__)


class PromotionCriteria:
    """Criteria for promoting a model to production"""

    def __init__(
        self,
        model_name: str,
        max_rmse: Optional[float] = None,
        min_r2: Optional[float] = None,
        max_mape: Optional[float] = None,
        max_mae: Optional[float] = None,
        min_improvement_percent: float = 0.0,
        require_better_than_production: bool = True,
    ):
        """
        Initialize promotion criteria

        Args:
            model_name: Name of model in registry
            max_rmse: Maximum allowed RMSE
            min_r2: Minimum required R² score
            max_mape: Maximum allowed MAPE
            max_mae: Maximum allowed MAE
            min_improvement_percent: Minimum % improvement over production (0-100)
            require_better_than_production: Must be better than current production
        """
        self.model_name = model_name
        self.max_rmse = max_rmse
        self.min_r2 = min_r2
        self.max_mape = max_mape
        self.max_mae = max_mae
        self.min_improvement_percent = min_improvement_percent
        self.require_better_than_production = require_better_than_production

    def to_dict(self) -> Dict:
        return {
            "model_name": self.model_name,
            "max_rmse": self.max_rmse,
            "min_r2": self.min_r2,
            "max_mape": self.max_mape,
            "max_mae": self.max_mae,
            "min_improvement_percent": self.min_improvement_percent,
            "require_better_than_production": self.require_better_than_production,
        }


class AutoPromotionService:
    """Service for automated model promotion"""

    def __init__(self, tracking_uri: Optional[str] = None):
        """
        Initialize auto-promotion service

        Args:
            tracking_uri: MLflow tracking URI (optional)
        """
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        self.client = MlflowClient()

        # Default promotion criteria for each model
        self.criteria = {
            "tokyo_rent_lgbm": PromotionCriteria(
                model_name="tokyo_rent_lgbm",
                max_rmse=17500,  # Must have RMSE < 17,500
                min_r2=0.88,  # Must have R² > 0.88
                max_mape=9.0,  # Must have MAPE < 9%
                min_improvement_percent=2.0,  # Must be 2% better than production
                require_better_than_production=True,
            ),
            "tokyo_passed_rent_model": PromotionCriteria(
                model_name="tokyo_passed_rent_model",
                max_rmse=19000,
                min_r2=0.85,
                max_mape=10.0,
                min_improvement_percent=2.0,
                require_better_than_production=True,
            ),
            "tokyo_rent_torch": PromotionCriteria(
                model_name="tokyo_rent_torch",
                max_rmse=16000,  # PyTorch models should achieve better RMSE
                min_r2=0.90,  # Higher R² threshold for deep learning
                max_mape=8.0,  # Stricter MAPE for neural networks
                min_improvement_percent=2.0,  # Must be 2% better than production
                require_better_than_production=True,
            ),
        }

        logger.info("AutoPromotionService initialized")

    def set_criteria(self, model_name: str, criteria: PromotionCriteria) -> None:
        """Set custom promotion criteria for a model"""
        self.criteria[model_name] = criteria
        logger.info(f"Updated promotion criteria for {model_name}")

    def check_and_promote_new_models(self, dry_run: bool = False) -> Dict[str, Any]:
        """
        Check all registered models and promote qualified ones

        Args:
            dry_run: If True, only check but don't promote

        Returns:
            Dictionary with promotion results
        """
        results = {
            "timestamp": datetime.now().isoformat(),
            "dry_run": dry_run,
            "models_checked": [],
            "models_promoted": [],
            "models_skipped": [],
        }

        for model_name in self.criteria.keys():
            logger.info(f"Checking {model_name} for promotion...")

            try:
                result = self.check_model_for_promotion(model_name, dry_run=dry_run)
                results["models_checked"].append(model_name)

                if result["promoted"]:
                    results["models_promoted"].append({
                        "model_name": model_name,
                        "version": result["version"],
                        "reason": result["reason"],
                        "metrics": result["metrics"],
                    })
                else:
                    results["models_skipped"].append({
                        "model_name": model_name,
                        "reason": result["reason"],
                    })

            except Exception as e:
                logger.error(f"Error checking {model_name}: {e}")
                results["models_skipped"].append({
                    "model_name": model_name,
                    "reason": f"Error: {str(e)}",
                })

        logger.info(
            f"Promotion check complete: {len(results['models_promoted'])} promoted, "
            f"{len(results['models_skipped'])} skipped"
        )

        return results

    def check_model_for_promotion(
        self,
        model_name: str,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        Check if a specific model should be promoted

        Args:
            model_name: Name of model in registry
            dry_run: If True, only check but don't promote

        Returns:
            Dictionary with check results
        """
        try:
            # Get criteria for this model
            if model_name not in self.criteria:
                return {
                    "promoted": False,
                    "reason": f"No promotion criteria defined for {model_name}",
                }

            criteria = self.criteria[model_name]

            # Get latest non-production version
            candidate_versions = self.client.get_latest_versions(
                model_name,
                stages=["None", "Staging"],
            )

            if not candidate_versions:
                return {
                    "promoted": False,
                    "reason": "No candidate versions in None or Staging stage",
                }

            candidate = candidate_versions[0]
            candidate_metrics = self._get_model_metrics(candidate.run_id)

            # Check if meets absolute criteria
            passes_absolute, absolute_reason = self._check_absolute_criteria(
                candidate_metrics, criteria
            )

            if not passes_absolute:
                return {
                    "promoted": False,
                    "reason": f"Failed absolute criteria: {absolute_reason}",
                    "version": candidate.version,
                    "metrics": candidate_metrics,
                }

            # Check if better than current production
            if criteria.require_better_than_production:
                production_versions = self.client.get_latest_versions(
                    model_name,
                    stages=["Production"],
                )

                if production_versions:
                    production = production_versions[0]
                    production_metrics = self._get_model_metrics(production.run_id)

                    is_better, comparison_reason = self._compare_with_production(
                        candidate_metrics,
                        production_metrics,
                        criteria,
                    )

                    if not is_better:
                        return {
                            "promoted": False,
                            "reason": f"Not better than production: {comparison_reason}",
                            "version": candidate.version,
                            "metrics": candidate_metrics,
                            "production_metrics": production_metrics,
                        }

            # All checks passed - promote!
            if not dry_run:
                self.client.transition_model_version_stage(
                    name=model_name,
                    version=candidate.version,
                    stage="Production",
                    archive_existing_versions=True,
                )
                logger.info(
                    f"✅ Promoted {model_name} v{candidate.version} to Production"
                )

            return {
                "promoted": True,
                "version": candidate.version,
                "reason": "Passed all promotion criteria",
                "metrics": candidate_metrics,
                "dry_run": dry_run,
            }

        except Exception as e:
            logger.error(f"Error in promotion check: {e}")
            raise

    def _get_model_metrics(self, run_id: str) -> Dict[str, float]:
        """Get metrics for a model run"""
        try:
            run = self.client.get_run(run_id)
            return dict(run.data.metrics)
        except Exception as e:
            logger.error(f"Error getting metrics for run {run_id}: {e}")
            return {}

    def _check_absolute_criteria(
        self,
        metrics: Dict[str, float],
        criteria: PromotionCriteria,
    ) -> tuple[bool, str]:
        """
        Check if metrics meet absolute thresholds

        Returns:
            (passes, reason)
        """
        # Check RMSE
        if criteria.max_rmse is not None:
            rmse = metrics.get("rmse")
            if rmse is None:
                return False, "RMSE metric not found"
            if rmse > criteria.max_rmse:
                return False, f"RMSE {rmse:.2f} > threshold {criteria.max_rmse}"

        # Check R²
        if criteria.min_r2 is not None:
            r2 = metrics.get("r2_score")
            if r2 is None:
                return False, "R² metric not found"
            if r2 < criteria.min_r2:
                return False, f"R² {r2:.4f} < threshold {criteria.min_r2}"

        # Check MAPE
        if criteria.max_mape is not None:
            mape = metrics.get("mape")
            if mape is None:
                return False, "MAPE metric not found"
            if mape > criteria.max_mape:
                return False, f"MAPE {mape:.2f}% > threshold {criteria.max_mape}%"

        # Check MAE
        if criteria.max_mae is not None:
            mae = metrics.get("mae")
            if mae is None:
                return False, "MAE metric not found"
            if mae > criteria.max_mae:
                return False, f"MAE {mae:.2f} > threshold {criteria.max_mae}"

        return True, "All absolute criteria passed"

    def _compare_with_production(
        self,
        candidate_metrics: Dict[str, float],
        production_metrics: Dict[str, float],
        criteria: PromotionCriteria,
    ) -> tuple[bool, str]:
        """
        Compare candidate with production model

        Returns:
            (is_better, reason)
        """
        # Primary metric: RMSE (lower is better)
        candidate_rmse = candidate_metrics.get("rmse", float("inf"))
        production_rmse = production_metrics.get("rmse", float("inf"))

        if candidate_rmse >= production_rmse:
            return False, f"RMSE not better ({candidate_rmse:.2f} vs {production_rmse:.2f})"

        # Calculate improvement percentage
        improvement_percent = ((production_rmse - candidate_rmse) / production_rmse) * 100

        if improvement_percent < criteria.min_improvement_percent:
            return False, (
                f"Improvement {improvement_percent:.2f}% < "
                f"required {criteria.min_improvement_percent}%"
            )

        # Check R² improvement (higher is better)
        candidate_r2 = candidate_metrics.get("r2_score", 0)
        production_r2 = production_metrics.get("r2_score", 0)

        if candidate_r2 < production_r2:
            return False, f"R² worse ({candidate_r2:.4f} vs {production_r2:.4f})"

        return True, f"Improved by {improvement_percent:.2f}%"

    def promote_specific_version(
        self,
        model_name: str,
        version: str,
        force: bool = False,
    ) -> Dict[str, Any]:
        """
        Promote a specific model version to production

        Args:
            model_name: Name of model
            version: Version to promote
            force: Skip safety checks if True

        Returns:
            Dictionary with promotion result
        """
        try:
            model_version = self.client.get_model_version(model_name, version)

            if not force:
                # Run safety checks
                metrics = self._get_model_metrics(model_version.run_id)

                if model_name in self.criteria:
                    criteria = self.criteria[model_name]
                    passes, reason = self._check_absolute_criteria(metrics, criteria)

                    if not passes:
                        return {
                            "promoted": False,
                            "reason": f"Safety check failed: {reason}",
                            "force_required": True,
                        }

            # Promote
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage="Production",
                archive_existing_versions=True,
            )

            logger.info(f"Promoted {model_name} v{version} to Production")

            return {
                "promoted": True,
                "model_name": model_name,
                "version": version,
                "forced": force,
            }

        except Exception as e:
            logger.error(f"Error promoting model: {e}")
            return {
                "promoted": False,
                "error": str(e),
            }

    def rollback_to_previous(self, model_name: str) -> Dict[str, Any]:
        """
        Rollback to previous production version

        Args:
            model_name: Name of model to rollback

        Returns:
            Dictionary with rollback result
        """
        try:
            # Get archived versions (recently demoted from production)
            archived_versions = self.client.get_latest_versions(
                model_name,
                stages=["Archived"],
            )

            if not archived_versions:
                return {
                    "rolled_back": False,
                    "reason": "No archived versions to rollback to",
                }

            # Get most recently archived (was production before)
            rollback_version = archived_versions[0]

            # Promote back to production
            self.client.transition_model_version_stage(
                name=model_name,
                version=rollback_version.version,
                stage="Production",
                archive_existing_versions=True,
            )

            logger.info(
                f"Rolled back {model_name} to v{rollback_version.version}"
            )

            return {
                "rolled_back": True,
                "model_name": model_name,
                "version": rollback_version.version,
            }

        except Exception as e:
            logger.error(f"Error during rollback: {e}")
            return {
                "rolled_back": False,
                "error": str(e),
            }

    def get_promotion_status(self, model_name: str) -> Dict[str, Any]:
        """
        Get detailed promotion status for a model

        Args:
            model_name: Name of model

        Returns:
            Dictionary with status information
        """
        try:
            # Get all versions
            all_versions = self.client.search_model_versions(f"name='{model_name}'")

            status = {
                "model_name": model_name,
                "total_versions": len(all_versions),
                "versions_by_stage": {},
                "latest_by_stage": {},
                "promotion_criteria": None,
            }

            # Group by stage
            for version in all_versions:
                stage = version.current_stage
                if stage not in status["versions_by_stage"]:
                    status["versions_by_stage"][stage] = []
                status["versions_by_stage"][stage].append(version.version)

            # Get latest by stage
            for stage in ["Production", "Staging", "None", "Archived"]:
                versions = self.client.get_latest_versions(model_name, stages=[stage])
                if versions:
                    version = versions[0]
                    metrics = self._get_model_metrics(version.run_id)
                    status["latest_by_stage"][stage] = {
                        "version": version.version,
                        "metrics": metrics,
                    }

            # Add criteria
            if model_name in self.criteria:
                status["promotion_criteria"] = self.criteria[model_name].to_dict()

            return status

        except Exception as e:
            logger.error(f"Error getting promotion status: {e}")
            return {
                "model_name": model_name,
                "error": str(e),
            }

    # ------------------------------------------------------------------ #
    #  Version Cleanup / Retention Policy                                  #
    # ------------------------------------------------------------------ #

    def cleanup_old_versions(
        self,
        model_name: str,
        max_versions_to_keep: int = 10,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        Delete archived versions beyond the retention limit.

        Always keeps:
          - All Production / Staging / None-stage versions
          - The ``max_versions_to_keep`` most recent Archived versions (by version number)

        Args:
            model_name:           Registered model name in MLflow
            max_versions_to_keep: Number of Archived versions to retain (default: 10)
            dry_run:              If True, report what *would* be deleted without deleting

        Returns:
            {
              "model_name":         str,
              "dry_run":            bool,
              "total_versions":     int,
              "protected_versions": [version_str, ...],
              "archived_kept":      [version_str, ...],
              "deleted":            [version_str, ...],
            }
        """
        try:
            all_versions = self.client.search_model_versions(f"name='{model_name}'")
            # Sort descending by version number (newest first)
            all_versions = sorted(all_versions, key=lambda v: int(v.version), reverse=True)

            archived  = [v for v in all_versions if v.current_stage == "Archived"]
            protected = [v for v in all_versions if v.current_stage in ("Production", "Staging", "None")]

            to_keep   = archived[:max_versions_to_keep]
            to_delete = archived[max_versions_to_keep:]

            deleted = []
            for v in to_delete:
                if not dry_run:
                    self.client.delete_model_version(name=model_name, version=v.version)
                    logger.info(f"[Cleanup] Deleted {model_name} v{v.version} (Archived)")
                else:
                    logger.info(f"[Cleanup][DRY RUN] Would delete {model_name} v{v.version}")
                deleted.append(v.version)

            return {
                "model_name":         model_name,
                "dry_run":            dry_run,
                "total_versions":     len(all_versions),
                "protected_versions": [v.version for v in protected],
                "archived_kept":      [v.version for v in to_keep],
                "deleted":            deleted,
            }

        except Exception as e:
            logger.error(f"[Cleanup] Error cleaning up {model_name}: {e}")
            return {"model_name": model_name, "error": str(e), "dry_run": dry_run}

    def cleanup_all_models(
        self,
        max_versions_to_keep: int = 10,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        Run cleanup_old_versions for every model tracked by the promotion service.

        Returns:
            {
              "dry_run":       bool,
              "total_deleted": int,
              "results":       {model_name: cleanup_result, ...}
            }
        """
        results = {}
        for model_name in self.criteria:
            results[model_name] = self.cleanup_old_versions(
                model_name, max_versions_to_keep=max_versions_to_keep, dry_run=dry_run
            )
        total_deleted = sum(len(r.get("deleted", [])) for r in results.values())
        return {
            "dry_run":       dry_run,
            "total_deleted": total_deleted,
            "results":       results,
        }
