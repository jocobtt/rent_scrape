"""Loading the served models, keeping them in step with the registry, and the middleware that does both."""
import asyncio
import os
import threading
import time
from typing import Any, Dict

from fastapi import Request

from app_state import S
from config.logging_config import get_logger
from services import PredictionService
from services.ensemble_prediction_service import EnsemblePredictionService
from services.tabular_models import load_deep_models
from services.torch_prediction_service import PyTorchPredictionService

logger = get_logger(__name__)

def _build_prediction_service() -> PredictionService:
    return PredictionService(
        S.app.state.passed_model, S.app.state.challenger_model,
        passed_noterms_model=getattr(S.app.state, "passed_noterms_model", None),
        challenger_noterms_model=getattr(S.app.state, "challenger_noterms_model", None),
    )


# ---- Serve the registry's Production versions (services/model_sync.py) -------------------------
# models/ is a cache of what the registry calls Production. The API re-checks it at most every
# MODEL_SYNC_INTERVAL_S seconds (promotions made by the scheduled tasks or a training script run in
# other processes) and straight away after a promote / rollback through this API. MODEL_SYNC=0 serves
# whatever is on disk.
MODEL_SYNC_INTERVAL_S = float(os.getenv("MODEL_SYNC_INTERVAL_S", "300"))
_model_sync_lock = threading.Lock()
_model_sync_last = [0.0]
_SERVED_MODELS = (("challenger_model", "challenger"), ("passed_model", "passed"),
                  ("challenger_noterms_model", "challenger-noterms"), ("passed_noterms_model", "passed-noterms"))


def _reload_models() -> None:
    """Reload every served model from models/ and rebuild the prediction services, swapping them in only
    once they all built (a bundle that fails verification leaves the running models untouched)."""
    loaded = {}
    for attr, name in _SERVED_MODELS:
        try:
            loaded[attr] = S.model_loader.load_model(name)
        except Exception as e:
            logger.info(f"No {name} model after sync: {e}")
            loaded[attr] = None
    if loaded["challenger_model"] is None or loaded["passed_model"] is None:
        raise RuntimeError("shared challenger / passed model missing after sync")
    new_service = PredictionService(
        loaded["passed_model"], loaded["challenger_model"],
        passed_noterms_model=loaded["passed_noterms_model"],
        challenger_noterms_model=loaded["challenger_noterms_model"],
    )
    new_ensemble = EnsemblePredictionService(loaded["passed_model"], loaded["challenger_model"])
    deep = load_deep_models(S.model_loader.models_dir)
    new_torch = PyTorchPredictionService(loaded["passed_model"], loaded["challenger_model"], deep) if deep else None
    for attr, model in loaded.items():
        setattr(S.app.state, attr, model)
    S.app.state.torch_models = deep
    S.prediction_service, S.ensemble_prediction_service, S.torch_prediction_service = new_service, new_ensemble, new_torch
    logger.info("Reloaded served models from the registry's Production versions")


def refresh_models_from_registry(force: bool = False) -> Dict[str, Any]:
    """Sync models/ with the registry and reload what changed. Never raises (serving continues)."""
    if os.getenv("MODEL_SYNC", "1") == "0":
        return {"status": "disabled"}
    if not _model_sync_lock.acquire(blocking=False):
        return {"status": "busy"}
    try:
        if not force and time.monotonic() - _model_sync_last[0] < MODEL_SYNC_INTERVAL_S:
            return {"status": "skipped"}
        _model_sync_last[0] = time.monotonic()
        from services.model_sync import sync_production_models
        report = sync_production_models(S.model_loader.models_dir)
        updated = [v for v, r in report.items() if r.get("action") == "updated"]
        if updated:
            _reload_models()
        return {"status": "ok", "updated": updated, "variants": report}
    except Exception as e:
        logger.error(f"Model sync failed; still serving the previous models: {e}")
        return {"status": "error", "error": str(e)}
    finally:
        _model_sync_lock.release()


async def load_model_middleware(request: Request, call_next):
    
    if not hasattr(S.app.state, 'challenger_model'):
        try:
            S.app.state.challenger_model = S.model_loader.load_model("challenger")
        except Exception as e:
            logger.warning(f"Failed to load challenger model: {e}")
            S.app.state.challenger_model = None
    
    if not hasattr(S.app.state, 'passed_model'):
        try:
            S.app.state.passed_model = S.model_loader.load_model("passed")
        except Exception as e:
            logger.warning(f"Failed to load passed model: {e}")
            S.app.state.passed_model = None
    
    # Dedicated variants trained without deposit / key money (optional; used when a request omits both)
    for attr, name in (("challenger_noterms_model", "challenger-noterms"), ("passed_noterms_model", "passed-noterms")):
        if not hasattr(S.app.state, attr):
            try:
                setattr(S.app.state, attr, S.model_loader.load_model(name))
            except Exception as e:
                logger.info(f"No {name} model, falling back to the shared model: {e}")
                setattr(S.app.state, attr, None)

    # Deep models (TabPFN, TabNet) installed from the registry (services/tabular_models.py)
    if not hasattr(S.app.state, 'torch_models'):
        S.app.state.torch_models = load_deep_models(S.model_loader.models_dir)

    # Initialize prediction service if models are loaded
    if (S.app.state.challenger_model is not None and 
        S.app.state.passed_model is not None and 
        S.prediction_service is None):
        S.prediction_service = _build_prediction_service()
        S.ensemble_prediction_service = EnsemblePredictionService(
            S.app.state.passed_model, 
            S.app.state.challenger_model
        )
        
        # Initialize PyTorch prediction service if PyTorch models are available
        if S.app.state.torch_models and S.torch_prediction_service is None:
            S.torch_prediction_service = PyTorchPredictionService(
                S.app.state.passed_model, 
                S.app.state.challenger_model,
                S.app.state.torch_models
            )
            logger.info(f"Initialized PyTorchPredictionService with {len(S.app.state.torch_models)} models")
    
    # Pick up promotions made elsewhere, off the request path
    if time.monotonic() - _model_sync_last[0] >= MODEL_SYNC_INTERVAL_S and os.getenv("MODEL_SYNC", "1") != "0":
        asyncio.get_running_loop().run_in_executor(None, refresh_models_from_registry)

    response = await call_next(request)
    return response
