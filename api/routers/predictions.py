"""Core rent predictions (production model, challenger, side-by-side)."""
from fastapi import APIRouter, HTTPException, Request, status

from app_state import S
from config.logging_config import get_logger
from deps import PREDICTION_RATE_LIMIT, limiter
from schemas import InputData

logger = get_logger(__name__)
router = APIRouter()


@router.post("/predict", tags=["Predictions"])
@limiter.limit(PREDICTION_RATE_LIMIT)
async def predict_main(request: Request, data: InputData, log: bool = True):
    """
    Predict rent price using the passed (production) model.

    Args:
        data: InputData containing apartment features
        log: set false for exploratory calls (the form's live estimate) so they don't
            enter the drift-monitoring log

    Returns:
        dict: Dictionary containing the prediction
    """
    try:
        if S.prediction_service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Prediction service not available - models not loaded"
            )

        result = S.prediction_service.predict_with_passed_model(data.dict())

        # Log prediction for drift monitoring
        if log:
            try:
                result["prediction_id"] = S.data_capture_service.log_prediction(
                    model_type="passed",
                    input_features=data.dict(),
                    prediction=result["prediction"],
                    variant=result.get("variant"),
                )
            except Exception as log_error:
                logger.warning(f"Failed to log prediction: {log_error}")

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@router.post("/challenger_predict", tags=["Predictions"])
@limiter.limit(PREDICTION_RATE_LIMIT)
async def challenger_predict(request: Request, data: InputData, log: bool = True):
    """
    Predict rent price using the challenger model (LightGBM).

    Args:
        data: InputData containing apartment features
        log: set false for exploratory calls so they don't enter the drift-monitoring log

    Returns:
        dict: Dictionary containing the prediction
    """
    try:
        if S.prediction_service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Prediction service not available - models not loaded"
            )

        result = S.prediction_service.predict_with_challenger_model(data.dict())

        # Log prediction for drift monitoring
        if log:
            try:
                result["prediction_id"] = S.data_capture_service.log_prediction(
                    model_type="challenger",
                    input_features=data.dict(),
                    prediction=result["prediction"],
                    variant=result.get("variant"),
                )
            except Exception as log_error:
                logger.warning(f"Failed to log prediction: {log_error}")

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Challenger prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Challenger prediction failed: {str(e)}"
        )


@router.post("/compare_predictions", tags=["Predictions"])
@limiter.limit(PREDICTION_RATE_LIMIT)
async def compare_predictions(request: Request, data: InputData):
    """
    Compare predictions from both models side by side.
    
    Args:
        data: InputData containing apartment features
        
    Returns:
        dict: Dictionary containing predictions from both models
    """
    try:
        if S.prediction_service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Prediction service not available - models not loaded"
            )
        
        return S.prediction_service.compare_predictions(data.dict())

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Comparison prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Comparison prediction failed: {str(e)}"
        )
