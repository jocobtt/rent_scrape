"""Serve what the registry says is Production.

Training logs each variant's three files (`<variant>-model.joblib`, `-features.json`, `-geo.json`) to its
MLflow run under `bundle/`, and stamps the run id into the features file (services/artifact_bundle.py).
`sync_production_models` compares that run id with the registry's Production version and, when they
differ, downloads the Production run's bundle into `models/`, so promotion, rollback and manual
promotion change what the API predicts instead of only a stage label.

Failure modes keep serving the files already on disk: registry unreachable, no Production version
(a demoted model keeps its last good files; use rollback to switch), a Production run logged before
bundles existed, or a downloaded bundle whose hashes do not match.
"""
import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Optional, Union

# A dead registry must not stall the API for minutes (MLflow retries with long backoff by default)
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "5")

from services import artifact_bundle  # noqa: E402

logger = logging.getLogger(__name__)

# serving variant -> registered model
VARIANT_MODELS = {
    "challenger": "tokyo_rent_lgbm",
    "passed": "tokyo_passed_rent_model",
    "challenger-noterms": "tokyo_rent_lgbm_noterms",
    "passed-noterms": "tokyo_passed_rent_noterms",
}
BUNDLE_DIR = "bundle"


def local_run_id(models_dir: Union[str, Path], variant: str) -> Optional[str]:
    features = Path(models_dir) / f"{variant}-features.json"
    if not features.exists():
        return None
    with open(features, encoding="utf-8") as f:
        return (json.load(f).get("bundle") or {}).get("run_id")


def _files(variant: str):
    return [f"{variant}-model.joblib", f"{variant}-geo.json", f"{variant}-features.json"]  # features last


def _install(downloaded: Path, models_dir: Path, variant: str) -> None:
    """Copy the bundle into place file by file (temp name + rename), features file last so a reader that
    sees the new features also sees the new model. A stale local geo file the new bundle lacks is removed."""
    for name in _files(variant):
        src, dst = downloaded / name, models_dir / name
        if src.exists():
            tmp = dst.with_name(dst.name + ".tmp")
            shutil.copyfile(src, tmp)
            os.replace(tmp, dst)
        elif name.endswith("-geo.json") and dst.exists():
            dst.unlink()


def sync_variant(client, models_dir: Union[str, Path], variant: str) -> Dict[str, object]:
    models_dir = Path(models_dir)
    name = VARIANT_MODELS[variant]
    try:
        versions = client.get_latest_versions(name, stages=["Production"])
    except Exception as e:
        return {"action": "skipped", "reason": f"registry unavailable: {e}"}
    if not versions:
        return {"action": "skipped", "reason": "no Production version; keeping local files"}
    prod = versions[0]
    result = {"version": prod.version, "run_id": prod.run_id}
    if local_run_id(models_dir, variant) == prod.run_id:
        return {**result, "action": "unchanged"}

    with tempfile.TemporaryDirectory() as tmp:
        try:
            got = client.download_artifacts(prod.run_id, BUNDLE_DIR, tmp)
        except Exception as e:
            return {**result, "action": "skipped",
                    "reason": f"Production run has no downloadable bundle ({e}); keeping local files"}
        downloaded = Path(got)
        if not (downloaded / f"{variant}-model.joblib").exists() or not (downloaded / f"{variant}-features.json").exists():
            return {**result, "action": "skipped", "reason": "Production bundle is incomplete; keeping local files"}
        bad = artifact_bundle.problems(downloaded, variant)
        recorded = json.load(open(downloaded / f"{variant}-features.json", encoding="utf-8")).get("bundle") or {}
        if bad or not recorded:
            return {**result, "action": "skipped",
                    "reason": "Production bundle failed verification: " + ("; ".join(bad) or "not stamped")}
        _install(downloaded, models_dir, variant)
    logger.info("Serving %s v%s (run %s) for variant %s", name, prod.version, prod.run_id, variant)
    return {**result, "action": "updated"}


def sync_production_models(models_dir: Union[str, Path], client=None) -> Dict[str, Dict[str, object]]:
    """Bring `models_dir` in line with the registry's Production versions; per-variant report."""
    if client is None:
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
    return {variant: sync_variant(client, models_dir, variant) for variant in VARIANT_MODELS}


def serving_status(models_dir: Union[str, Path], client=None) -> Dict[str, Dict[str, object]]:
    """Per variant: the run being served from disk and the registry's Production run, and whether they match."""
    if client is None:
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
    out = {}
    for variant, name in VARIANT_MODELS.items():
        entry = {"registered_model": name, "serving_run_id": local_run_id(models_dir, variant),
                 "bundle_problems": artifact_bundle.problems(models_dir, variant)}
        try:
            versions = client.get_latest_versions(name, stages=["Production"])
            entry["production_version"] = versions[0].version if versions else None
            entry["production_run_id"] = versions[0].run_id if versions else None
            entry["in_sync"] = bool(versions) and versions[0].run_id == entry["serving_run_id"]
        except Exception as e:
            entry["registry_error"] = str(e)
        out[variant] = entry
    return out
