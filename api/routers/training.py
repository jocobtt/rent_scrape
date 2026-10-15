"""Retraining: manual, drift-triggered and per-ward."""
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status

import model_runtime
from app_state import S
from config.logging_config import get_logger
from deps import TRAINING_RATE_LIMIT, limiter, require_admin
from schemas import RetrainData, WardRetrainData
from services.ensemble_prediction_service import EnsemblePredictionService

logger = get_logger(__name__)
router = APIRouter()


@router.post("/retrain", tags=["Training"], dependencies=[Depends(require_admin)])
@limiter.limit(TRAINING_RATE_LIMIT)
async def retrain(request: Request, data: RetrainData):
    """
    Retrain specified models with fresh data.
    
    Args:
        data: RetrainData containing scraping parameters and models to retrain
        
    Returns:
        dict: Dictionary containing retraining results
    """
    try:
        
        # Use training service to retrain models
        result = S.training_service.retrain_models(
            str(data.url),
            data.wait_time_min,
            data.wait_time_max, 
            data.pages, 
            data.model_to_retrain
        )
        
        # Update the loaded models in app state
        for model_name in data.model_to_retrain:
            if model_name == "challenger" and "challenger" in result["retrained_models"]:
                S.app.state.challenger_model = result["retrained_models"]["challenger"]
            elif model_name == "passed" and "passed" in result["retrained_models"]:
                S.app.state.passed_model = result["retrained_models"]["passed"]
        
        # Reinitialize prediction service with updated models
        if (S.app.state.challenger_model is not None and 
            S.app.state.passed_model is not None):
            S.prediction_service = model_runtime._build_prediction_service()
            S.ensemble_prediction_service = EnsemblePredictionService(
                S.app.state.passed_model, 
                S.app.state.challenger_model
            )
        
        return result
        
    except Exception as e:
        logger.error(f"Error during retraining: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Retraining failed: {str(e)}"
        )


@router.post("/retrain/check-and-execute", tags=["Training"], dependencies=[Depends(require_admin)])
@limiter.limit(TRAINING_RATE_LIMIT)
async def check_and_retrain(request: Request, days_back: int = 7, dry_run: bool = True):
    """Check conditions and automatically retrain if needed"""
    try:
        result = S.auto_retrain_service.check_and_retrain(days_back=days_back, dry_run=dry_run)
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Error in check and retrain: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/retrain/force", dependencies=[Depends(require_admin)])
@limiter.limit(TRAINING_RATE_LIMIT)
async def force_retrain(request: Request, model_name: Optional[str] = None):
    """Force immediate retraining"""
    try:
        result = S.auto_retrain_service.force_retrain(model_name=model_name)
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Error in force retrain: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/retrain/wards", tags=["Training"], dependencies=[Depends(require_admin)])
@limiter.limit(TRAINING_RATE_LIMIT)
async def retrain_with_wards(request: Request, data: WardRetrainData):
    """
    Scrape multiple Tokyo wards and retrain models on the combined dataset.

    Covers up to all 23 special wards for a broader, more representative
    sample than a single-URL scrape.  Each ward contributes up to
    `pages_per_ward × 30` listings.

    Omit `ward_codes` to scrape every ward automatically.
    """
    try:
        result = S.training_service.retrain_with_wards(
            ward_codes=data.ward_codes,
            pages_per_ward=data.pages_per_ward,
            wait_time_min=data.wait_time_min,
            wait_time_max=data.wait_time_max,
            models_to_retrain=data.model_to_retrain,
        )

        if result.get("status") == "success":
            for model_name in (data.model_to_retrain or []):
                if model_name == "challenger" and "challenger" in result.get("retrained_models", {}):
                    S.app.state.challenger_model = result["retrained_models"]["challenger"]
                elif model_name == "passed" and "passed" in result.get("retrained_models", {}):
                    S.app.state.passed_model = result["retrained_models"]["passed"]

        return result
    except Exception as e:
        logger.error(f"Error during ward-based retrain: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ward retraining failed: {str(e)}",
        )
