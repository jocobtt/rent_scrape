"""
Model SLA Service

Continuously monitors production models against minimum performance floors.
Automatically demotes a model to Staging if it falls below its SLA thresholds.

SLA thresholds are intentionally looser than promotion criteria — they represent
the minimum acceptable performance, not the bar a new model must clear.
"""

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

logger = logging.getLogger(__name__)


@dataclass
class SLACriteria:
    """Minimum acceptable performance floors for a production model."""
    max_rmse: float
    min_r2: float
    max_mape: float
    demotion_stage: str = "Staging"


class ModelSLAService:
    """
    Monitors production models against SLA floors and demotes failing models.

    Usage:
        sla = ModelSLAService()
        result = sla.check_all_models()   # check + auto-demote all models
        result = sla.check_model("tokyo_rent_lgbm")  # check a single model
    """

    # SLA floors — intentionally ~15% looser than promotion thresholds
    criteria: Dict[str, SLACriteria] = {
        "tokyo_rent_lgbm":         SLACriteria(max_rmse=20_000, min_r2=0.82, max_mape=12.0),
        "tokyo_passed_rent_model": SLACriteria(max_rmse=23_000, min_r2=0.78, max_mape=14.0),
        "tokyo_rent_tabpfn":       SLACriteria(max_rmse=18_000, min_r2=0.85, max_mape=11.0),
        "tokyo_rent_tabnet":       SLACriteria(max_rmse=19_000, min_r2=0.84, max_mape=11.5),
        "tokyo_rent_torch":        SLACriteria(max_rmse=19_000, min_r2=0.85, max_mape=11.0),
    }

    def __init__(self):
        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
        mlflow.set_tracking_uri(mlflow_uri)
        self.client = MlflowClient()

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def check_all_models(self) -> Dict:
        """
        Check SLA for all tracked models. Demotes any that are failing.

        Returns:
            {
              "checked_at":     str (ISO-8601),
              "models_checked": int,
              "models_passing": int,
              "models_demoted": int,
              "results":        {model_name: check_result, ...}
            }
        """
        results = {}
        for model_name in self.criteria:
            try:
                results[model_name] = self.check_model(model_name)
            except Exception as e:
                logger.error(f"[SLA] Unexpected error checking {model_name}: {e}")
                results[model_name] = {"error": str(e), "sla_passing": True, "demoted": False}

        return {
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "models_checked": len(results),
            "models_passing": sum(1 for r in results.values() if r.get("sla_passing", True)),
            "models_demoted": sum(1 for r in results.values() if r.get("demoted", False)),
            "results": results,
        }

    def check_model(self, model_name: str) -> Dict:
        """
        Check SLA for a single model. Demotes it to Staging if failing.

        Returns:
            {
              "model_name":         str,
              "production_version": str | None,
              "metrics":            {rmse, r2_score, mape},
              "sla_criteria":       {max_rmse, min_r2, max_mape},
              "sla_passing":        bool,
              "reason":             str,
              "demoted":            bool,
              "demoted_to":         str (only if demoted),
            }
        """
        criteria = self.criteria.get(model_name)
        if not criteria:
            return {
                "model_name": model_name,
                "sla_passing": True,
                "reason": f"No SLA criteria configured for '{model_name}' — skipping",
                "demoted": False,
            }

        metrics = self._get_production_metrics(model_name)
        if metrics is None:
            return {
                "model_name": model_name,
                "production_version": None,
                "sla_passing": True,
                "reason": "No production model found — skipping",
                "demoted": False,
            }

        passing, reason = self._check_sla(metrics, criteria)

        result: Dict = {
            "model_name": model_name,
            "production_version": str(metrics.get("version")),
            "metrics": {
                "rmse":     metrics.get("rmse"),
                "r2_score": metrics.get("r2_score"),
                "mape":     metrics.get("mape"),
            },
            "sla_criteria": {
                "max_rmse": criteria.max_rmse,
                "min_r2":   criteria.min_r2,
                "max_mape": criteria.max_mape,
            },
            "sla_passing": passing,
            "reason": reason,
            "demoted": False,
        }

        if not passing:
            logger.warning(f"[SLA] {model_name} v{metrics['version']} FAILED: {reason}")
            try:
                self.client.transition_model_version_stage(
                    name=model_name,
                    version=str(metrics["version"]),
                    stage=criteria.demotion_stage,
                    archive_existing_versions=False,
                )
                result["demoted"] = True
                result["demoted_to"] = criteria.demotion_stage
                logger.warning(
                    f"[SLA] Demoted {model_name} v{metrics['version']} → {criteria.demotion_stage}"
                )
            except MlflowException as e:
                logger.error(f"[SLA] Demotion failed for {model_name}: {e}")
                result["demotion_error"] = str(e)

        return result

    # ------------------------------------------------------------------ #
    #  Internals                                                           #
    # ------------------------------------------------------------------ #

    def _get_production_metrics(self, model_name: str) -> Optional[Dict]:
        """Fetch the current production version's metrics from MLflow."""
        try:
            versions = self.client.get_latest_versions(model_name, stages=["Production"])
            if not versions:
                return None
            v = versions[0]
            run = self.client.get_run(v.run_id)
            return {"version": v.version, "run_id": v.run_id, **run.data.metrics}
        except MlflowException as e:
            logger.error(f"[SLA] Could not fetch production metrics for {model_name}: {e}")
            return None
        except Exception as e:
            logger.error(f"[SLA] Unexpected error fetching metrics for {model_name}: {e}")
            return None

    def _check_sla(self, metrics: Dict, criteria: SLACriteria) -> Tuple[bool, str]:
        """Return (passing, reason) for the given metrics against the SLA floor."""
        rmse = metrics.get("rmse")
        r2   = metrics.get("r2_score")
        mape = metrics.get("mape")

        if rmse is not None and rmse > criteria.max_rmse:
            return False, f"RMSE {rmse:,.0f} exceeds SLA max {criteria.max_rmse:,.0f}"
        if r2 is not None and r2 < criteria.min_r2:
            return False, f"R² {r2:.4f} below SLA min {criteria.min_r2:.4f}"
        if mape is not None and mape > criteria.max_mape:
            return False, f"MAPE {mape:.2f}% exceeds SLA max {criteria.max_mape:.2f}%"

        missing = []
        if rmse is None:
            missing.append("rmse")
        if r2 is None:
            missing.append("r2_score")
        if mape is None:
            missing.append("mape")
        if missing:
            return True, f"SLA passed (metrics not logged: {', '.join(missing)})"

        return True, "All SLA checks passed"
