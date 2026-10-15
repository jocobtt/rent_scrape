"""Model info and metadata, promotion / rollback / serving status, SLA and version cleanup."""
from typing import Optional

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Request, status
from starlette.concurrency import run_in_threadpool

import model_runtime
from app_state import S
from config.logging_config import get_logger
from deps import DEFAULT_RATE_LIMIT, limiter, require_admin

logger = get_logger(__name__)
router = APIRouter()


@router.get("/model_info")
async def get_model_info():
    """
    Get information about the loaded models.
    """
    try:
        passed_info = {"loaded": False, "file_exists": False}
        challenger_info = {"loaded": False, "file_exists": False}

        # Check if models are loaded
        if hasattr(S.app.state, 'passed_model') and S.app.state.passed_model is not None:
            passed_info.update(S.model_loader.get_model_info(S.app.state.passed_model))

        if hasattr(S.app.state, 'challenger_model') and S.app.state.challenger_model is not None:
            challenger_info.update(S.model_loader.get_model_info(S.app.state.challenger_model))

        # Check if model files exist
        passed_info["file_exists"] = S.model_loader.model_exists("passed")
        challenger_info["file_exists"] = S.model_loader.model_exists("challenger")

        info = {
            "models": {
                "passed_model": passed_info,
                "challenger_model": challenger_info
            },
            "services": {
                "prediction_service_available": S.prediction_service is not None,
                "ensemble_prediction_service_available": S.ensemble_prediction_service is not None,
                "training_service_available": S.training_service is not None,
                "model_loader_available": S.model_loader is not None
            },
            "api_version": "1.0.0",
            "supported_endpoints": [
                "/predict",
                "/challenger_predict",
                "/compare_predictions",
                "/ensemble/predict",
                "/ensemble/compare",
                "/ensemble/confidence",
                "/ensemble/recommendations",
                "/ensemble/batch",
                "/ensemble/methods",
                "/ensemble/train",
                "/retrain",
                "/health",
                "/model_info",
                "/model/metadata",
                "/monitoring/summary",
                "/monitoring/drift",
                "/monitoring/reports/generate",
                "/monitoring/compare_models"
            ]
        }
        return info
    except Exception as e:
        logger.error(f"Model info error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model info: {str(e)}"
        )


@router.get("/model/metadata")
async def get_model_metadata(
    model_name: Optional[str] = None,
    version: Optional[str] = None
):
    """
    Get metadata for ML models from MLflow model registry.

    Query Parameters:
        model_name: Optional. Specific model name (e.g., "tokyo_rent_lgbm" or "tokyo_passed_rent_model")
                   If not provided, returns metadata for the active production model.
        version: Optional. Specific version number. If not provided, returns latest production version.

    Returns:
        dict: Model metadata including metrics, parameters, and feature importance
    """
    try:
        # If no model name specified, return the challenger (LightGBM) model as primary
        if model_name is None:
            model_name = "tokyo_rent_lgbm"

        # Get metadata from MLflow
        if version:
            metadata = S.mlflow_service.get_model_metadata_by_version(model_name, version)
        else:
            metadata = S.mlflow_service.get_production_model_metadata(model_name)

        if metadata is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model '{model_name}' not found in MLflow registry. Make sure the model has been trained and registered."
            )

        return metadata

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model metadata: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model metadata: {str(e)}"
        )


@router.get("/model/metadata/all")
async def get_all_models_metadata():
    """
    Get metadata for all active models (both challenger and passed).

    Returns:
        dict: Metadata for all models including challenger (LightGBM) and passed (Linear)
    """
    try:
        all_metadata = S.mlflow_service.get_all_active_models_metadata()

        # Check if any models were found
        if all_metadata["challenger"] is None and all_metadata["passed"] is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No models found in MLflow registry. Train and register models first."
            )

        return {
            "models": all_metadata,
            "timestamp": pd.Timestamp.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting all models metadata: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get models metadata: {str(e)}"
        )


# Model Promotion Endpoints

@router.post("/models/{model_name}/promote/{version}", tags=["Model Management"], dependencies=[Depends(require_admin)])
@limiter.limit(DEFAULT_RATE_LIMIT)
async def promote_model_version(request: Request, model_name: str, version: str, force: bool = False):
    """Manually promote a specific model version to production"""
    try:
        result = S.auto_promotion_service.promote_specific_version(model_name, version, force)
        if not result["promoted"]:
            return {"status": "not_promoted", "message": result.get("reason"), "force_required": result.get("force_required", False)}
        serving = await run_in_threadpool(model_runtime.refresh_models_from_registry, True)
        return {"status": "success", "message": f"Promoted {model_name} v{version} to Production", "result": result,
                "serving": serving}
    except Exception as e:
        logger.error(f"Error promoting model: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/models/check-promotion", dependencies=[Depends(require_admin)])
@limiter.limit(DEFAULT_RATE_LIMIT)
async def check_models_for_promotion(request: Request, dry_run: bool = True):
    """Check all models and promote qualified ones"""
    try:
        results = S.auto_promotion_service.check_and_promote_new_models(dry_run=dry_run)
        serving = None if dry_run else await run_in_threadpool(model_runtime.refresh_models_from_registry, True)
        return {"status": "success", "results": results, "serving": serving}
    except Exception as e:
        logger.error(f"Error checking models: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/models/serving", tags=["Model Management"])
async def models_serving_status():
    """Which run each variant is serving from disk vs. the registry's Production run."""
    from services.model_sync import serving_status
    return {"status": "success", "variants": await run_in_threadpool(serving_status, S.model_loader.models_dir)}


@router.get("/models/{model_name}/promotion-status")
async def get_promotion_status(model_name: str):
    """Get detailed promotion status for a model"""
    try:
        status_info = S.auto_promotion_service.get_promotion_status(model_name)
        return {"status": "success", "info": status_info}
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/models/{model_name}/rollback", dependencies=[Depends(require_admin)])
@limiter.limit(DEFAULT_RATE_LIMIT)
async def rollback_model(request: Request, model_name: str):
    """Rollback to previous production version"""
    try:
        result = S.auto_promotion_service.rollback_to_previous(model_name)
        if not result["rolled_back"]:
            return {"status": "not_rolled_back", "message": result.get("reason")}
        serving = await run_in_threadpool(model_runtime.refresh_models_from_registry, True)
        return {"status": "success", "message": f"Rolled back {model_name} to v{result['version']}", "result": result,
                "serving": serving}
    except Exception as e:
        logger.error(f"Error during rollback: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# ============================================================================
# Lifecycle Endpoints — SLA monitoring and version cleanup
# ============================================================================

@router.get("/models/sla-status", tags=["Lifecycle"])
@limiter.limit(DEFAULT_RATE_LIMIT)
async def get_all_models_sla_status(request: Request):
    """
    Check SLA floors for all tracked production models.

    Returns per-model SLA status including current metrics, thresholds,
    pass/fail verdict, and whether the model was demoted this check.
    """
    try:
        return S.sla_service.check_all_models()
    except Exception as e:
        logger.error(f"Error running SLA check: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/models/{model_name}/sla-status", tags=["Lifecycle"])
@limiter.limit(DEFAULT_RATE_LIMIT)
async def get_model_sla_status(request: Request, model_name: str):
    """
    Check SLA floor for a single production model.

    Demotes the model to Staging automatically if it is failing its SLA.
    Returns 404 if no SLA criteria are configured for the model.
    """
    try:
        result = S.sla_service.check_model(model_name)
        if "error" in result and "No SLA criteria" in result.get("error", ""):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=result["error"]
            )
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error running SLA check for {model_name}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/models/cleanup", tags=["Lifecycle"], dependencies=[Depends(require_admin)])
@limiter.limit(DEFAULT_RATE_LIMIT)
async def cleanup_all_model_versions(
    request: Request,
    max_versions_to_keep: int = 10,
    dry_run: bool = True,
):
    """
    Delete old Archived model versions across all tracked models.

    Keeps all Production/Staging versions plus the `max_versions_to_keep` most
    recent Archived versions per model. Permanently deletes the rest from MLflow.

    **Default is dry_run=true** — pass `?dry_run=false` to execute deletions.
    """
    try:
        return S.auto_promotion_service.cleanup_all_models(
            max_versions_to_keep=max_versions_to_keep,
            dry_run=dry_run,
        )
    except Exception as e:
        logger.error(f"Error running version cleanup: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/models/{model_name}/cleanup", tags=["Lifecycle"], dependencies=[Depends(require_admin)])
@limiter.limit(DEFAULT_RATE_LIMIT)
async def cleanup_model_versions(
    request: Request,
    model_name: str,
    max_versions_to_keep: int = 10,
    dry_run: bool = True,
):
    """
    Delete old Archived versions for a specific model.

    Keeps all Production/Staging versions plus the `max_versions_to_keep` most
    recent Archived versions. Permanently deletes the rest from MLflow.

    **Default is dry_run=true** — pass `?dry_run=false` to execute deletions.
    """
    try:
        return S.auto_promotion_service.cleanup_old_versions(
            model_name=model_name,
            max_versions_to_keep=max_versions_to_keep,
            dry_run=dry_run,
        )
    except Exception as e:
        logger.error(f"Error running cleanup for {model_name}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
