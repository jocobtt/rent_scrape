"""Ensemble predictions (weighted, stacking, voting), confidence and batch."""
import time

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app_state import S
from config.logging_config import PerformanceLogger, get_logger, log_prediction
from deps import DEFAULT_RATE_LIMIT, PREDICTION_RATE_LIMIT, limiter, require_admin
from middleware.logging_middleware import get_request_id
from schemas import BatchPredictRequest, EnsemblePredictRequest, InputData, _require_terms

logger = get_logger(__name__)
router = APIRouter()


# Ensemble Prediction Endpoints

@router.post("/ensemble/predict", tags=["Ensemble Predictions"])
@limiter.limit(PREDICTION_RATE_LIMIT)
async def predict_with_ensemble(request: Request, data: EnsemblePredictRequest):
    """
    Make predictions using ensemble methods that combine multiple models.
    
    Available ensemble methods:
    - weighted_average: Combines predictions using optimized weights
    - stacking: Uses meta-model to combine base model predictions  
    - voting: Simple average of all model predictions
    - dynamic_weights: Adjusts weights based on input characteristics
    
    Args:
        data: EnsemblePredictRequest containing input data and ensemble method
        
    Returns:
        dict: Prediction result with ensemble metadata
    """
    try:
        if S.ensemble_prediction_service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Ensemble prediction service not available - models not loaded"
            )
        
        input_dict = _require_terms(data.input_data)
        started = time.perf_counter()
        result = S.ensemble_prediction_service.predict_with_ensemble(input_dict, data.method)
        duration_ms = (time.perf_counter() - started) * 1000

        # Log prediction for monitoring
        log_prediction(logger, f"ensemble_{data.method}", input_dict, result.get("prediction"), duration_ms,
                       get_request_id(request))
        S.data_capture_service.log_prediction(
            model_type=f"ensemble_{data.method}", input_features=input_dict,
            prediction=result.get("prediction"), source="api")

        return result
        
    except HTTPException:
        raise  # e.g. the 422 from _require_terms
    except ValueError as e:
        logger.warning(f"Invalid ensemble method: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Ensemble prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ensemble prediction failed: {str(e)}"
        )


@router.post("/ensemble/compare", tags=["Ensemble Predictions"])
@limiter.limit(PREDICTION_RATE_LIMIT)
async def compare_all_predictions(request: Request, data: InputData):
    """
    Compare predictions from all available models including ensemble methods.
    
    Returns predictions from:
    - Base models (passed, challenger)
    - All available ensemble methods
    - Statistical analysis of prediction agreement
    
    Args:
        data: InputData containing apartment features
        
    Returns:
        dict: Comprehensive comparison of all model predictions
    """
    try:
        if S.ensemble_prediction_service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Ensemble prediction service not available - models not loaded"
            )
        
        result = S.ensemble_prediction_service.compare_all_predictions(_require_terms(data))
        
        return result
        
    except HTTPException:
        raise  # e.g. the 422 from _require_terms
    except Exception as e:
        logger.error(f"Ensemble comparison error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ensemble comparison failed: {str(e)}"
        )


@router.post("/ensemble/confidence", tags=["Ensemble Predictions"])
@limiter.limit(PREDICTION_RATE_LIMIT)
async def get_prediction_confidence(request: Request, data: InputData):
    """
    Calculate prediction confidence based on model agreement.
    
    Analyzes how much models agree on predictions and provides confidence metrics:
    - Coefficient of variation across models
    - Confidence level (very_high, high, medium, low, very_low)
    - Prediction range and statistics
    - Model agreement percentages
    
    Args:
        data: InputData containing apartment features
        
    Returns:
        dict: Confidence analysis and model agreement metrics
    """
    try:
        if S.ensemble_prediction_service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Ensemble prediction service not available - models not loaded"
            )
        
        result = S.ensemble_prediction_service.get_prediction_confidence(_require_terms(data))
        
        return result
        
    except HTTPException:
        raise  # e.g. the 422 from _require_terms
    except Exception as e:
        logger.error(f"Confidence calculation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Confidence calculation failed: {str(e)}"
        )


@router.post("/ensemble/recommendations", tags=["Ensemble Predictions"])
@limiter.limit(PREDICTION_RATE_LIMIT)
async def get_model_recommendations(request: Request, data: InputData):
    """
    Get model recommendations based on input characteristics.
    
    Analyzes apartment features to recommend the best model/ensemble method:
    - Property type analysis (high-end, large, old/new)
    - Model suitability recommendations
    - Confidence analysis for each recommendation
    
    Args:
        data: InputData containing apartment features
        
    Returns:
        dict: Model recommendations with reasoning and confidence analysis
    """
    try:
        if S.ensemble_prediction_service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Ensemble prediction service not available - models not loaded"
            )
        
        result = S.ensemble_prediction_service.get_model_recommendations(_require_terms(data))
        
        return result
        
    except HTTPException:
        raise  # e.g. the 422 from _require_terms
    except Exception as e:
        logger.error(f"Model recommendations error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model recommendations failed: {str(e)}"
        )


@router.post("/ensemble/batch", tags=["Ensemble Predictions"])
@limiter.limit("10/minute")  # Lower limit for batch operations
async def batch_predict(request: Request, data: BatchPredictRequest):
    """
    Make batch predictions using specified method.
    
    Supports batch processing for efficiency when predicting many properties.
    Can use any available prediction method including ensemble methods.
    
    Args:
        data: BatchPredictRequest containing list of inputs and method
        
    Returns:
        dict: List of prediction results for each input
    """
    try:
        if S.ensemble_prediction_service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Ensemble prediction service not available - models not loaded"
            )
        
        # Limit batch size for performance
        if len(data.input_data_list) > 100:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Batch size limited to 100 predictions per request"
            )
        
        # Convert input data to list of dicts
        input_dicts = [_require_terms(item) for item in data.input_data_list]
        
        result = S.ensemble_prediction_service.batch_predict(input_dicts, data.method)
        
        return {
            "batch_size": len(input_dicts),
            "method": data.method,
            "results": result,
            "success_count": len([r for r in result if r.get("status") == "success"]),
            "error_count": len([r for r in result if r.get("status") == "failed"])
        }
        
    except HTTPException:
        raise  # e.g. the 422 from _require_terms
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


@router.get("/ensemble/methods", tags=["Ensemble Predictions"])
@limiter.limit(DEFAULT_RATE_LIMIT)
async def get_available_ensemble_methods(request: Request):
    """
    Get information about all available prediction methods.
    
    Returns:
        dict: Available base models, ensemble methods, and their details
    """
    try:
        if S.ensemble_prediction_service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Ensemble prediction service not available - models not loaded"
            )
        
        methods = S.ensemble_prediction_service.get_available_methods()
        
        # Add method descriptions
        methods["descriptions"] = {
            "weighted_average": "Optimally weighted combination of model predictions",
            "stacking": "Meta-model that learns how to combine base model predictions",
            "voting": "Simple average of all model predictions",
            "dynamic_weights": "Adaptive weighting based on input characteristics"
        }
        
        return methods

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting ensemble methods: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get ensemble methods: {str(e)}"
        )


@router.post("/ensemble/train", tags=["Ensemble Predictions"], dependencies=[Depends(require_admin)])
@limiter.limit("1/hour")  # Very limited for training operations
async def train_ensemble_models(request: Request):
    """
    Train ensemble models using current base models.
    
    This endpoint will:
    1. Load fresh training data
    2. Train all ensemble methods
    3. Evaluate and compare performance
    4. Save the best performing ensemble models
    5. Log results to MLflow
    
    Returns:
        dict: Training results and performance comparison
    """
    try:
        from models.ensemble_model import train_ensemble_models

        # Train all ensemble methods
        started = time.perf_counter()
        with PerformanceLogger("ensemble_training", logger):
            results = train_ensemble_models()
        training_seconds = time.perf_counter() - started
        
        if not results:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="No ensemble models were successfully trained"
            )
        
        # Prepare response
        response = {
            "training_duration": training_seconds,
            "models_trained": list(results.keys()),
            "results": {}
        }
        
        for method, result in results.items():
            response["results"][method] = {
                "rmse": result["metrics"]["rmse"],
                "r2": result["metrics"]["r2"],
                "mae": result["metrics"]["mae"],
                "improvement_vs_best_base": result["improvement_vs_best_base"]
            }
        
        # Find best method
        best_method = min(results.keys(), key=lambda k: results[k]["metrics"]["rmse"])
        response["best_method"] = best_method
        response["best_rmse"] = results[best_method]["metrics"]["rmse"]
        
        logger.info(f"Ensemble training completed in {training_seconds:.2f}s. Best method: {best_method}")
        
        return response
        
    except Exception as e:
        logger.error(f"Ensemble training error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ensemble training failed: {str(e)}"
        )
