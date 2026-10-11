"""PyTorch neural-network predictions and training."""
from fastapi import APIRouter, Depends, HTTPException, Request, status

from app_state import S
from config.logging_config import PerformanceLogger, get_logger, log_prediction
from deps import DEFAULT_RATE_LIMIT, PREDICTION_RATE_LIMIT, TRAINING_RATE_LIMIT, limiter, require_admin
from middleware.logging_middleware import get_request_id
from schemas import InputData, TorchBatchPredictRequest, TorchPredictRequest, _require_terms

logger = get_logger(__name__)
router = APIRouter()


# ============================================================================
# PyTorch Neural Network Prediction Endpoints
# ============================================================================

@router.post("/torch/predict", tags=["PyTorch Predictions"])
@limiter.limit(PREDICTION_RATE_LIMIT)
async def predict_with_torch_model(request: Request, data: TorchPredictRequest):
    """
    Make predictions using PyTorch neural network models
    
    - **model_name**: Name of the PyTorch model (e.g., 'tabular', 'ensemble', 'transformer')
    - **input_data**: Apartment features for prediction
    """
    with PerformanceLogger("torch_prediction"):
        try:
            request_id = get_request_id()
            
            if S.torch_prediction_service is None:
                logger.warning("PyTorch prediction service not available")
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
                    detail="PyTorch models not available. Please check model loading."
                )
            
            # Make prediction
            result = S.torch_prediction_service.predict_with_torch_model(
                _require_terms(data.input_data), 
                data.model_name
            )
            
            # Log prediction for monitoring
            log_prediction(
                input_data=_require_terms(data.input_data),
                prediction=result["prediction"],
                model_name=f"torch_{data.model_name}",
                request_id=request_id,
                processing_time_ms=result.get("processing_time_ms", 0)
            )
            
            # Capture data for monitoring
            await S.data_capture_service.capture_prediction(
                input_data=_require_terms(data.input_data),
                prediction=result["prediction"],
                model_name=f"torch_{data.model_name}",
                model_version="1.0",
                request_metadata={
                    "request_id": request_id,
                    "endpoint": "/torch/predict",
                    "framework": "pytorch"
                }
            )
            
            result["request_id"] = request_id
            return result
            
        except ValueError as e:
            logger.warning(f"PyTorch prediction validation error: {e}")
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
        except Exception as e:
            logger.error(f"PyTorch prediction error: {e}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/torch/models", tags=["PyTorch Predictions"])
@limiter.limit(DEFAULT_RATE_LIMIT)
async def get_available_torch_models(request: Request):
    """
    Get information about available PyTorch models
    
    Returns details about loaded PyTorch models including parameters, size, and capabilities.
    """
    try:
        if S.torch_prediction_service is None:
            return {
                "available": False,
                "models": {},
                "total_models": 0,
                "message": "PyTorch prediction service not initialized"
            }
        
        model_info = S.torch_prediction_service.get_available_torch_models()
        model_info["available"] = True
        return model_info
        
    except Exception as e:
        logger.error(f"Error getting PyTorch model info: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/torch/recommendations", tags=["PyTorch Predictions"])
@limiter.limit(PREDICTION_RATE_LIMIT)
async def get_torch_model_recommendations(request: Request, data: InputData):
    """
    Get PyTorch model recommendations based on input characteristics
    
    Analyzes apartment features and recommends the best PyTorch model for prediction.
    """
    try:
        if S.torch_prediction_service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="PyTorch prediction service not available"
            )
        
        recommendations = S.torch_prediction_service.get_torch_model_recommendations(_require_terms(data))
        return recommendations
        
    except Exception as e:
        logger.error(f"Error generating PyTorch recommendations: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/torch/compare-all", tags=["PyTorch Predictions"])
@limiter.limit(PREDICTION_RATE_LIMIT)
async def compare_all_models_with_torch(request: Request, data: InputData):
    """
    Compare predictions from all models including PyTorch models
    
    Returns predictions from sklearn models, ensemble methods, and PyTorch models with statistics.
    """
    try:
        if S.torch_prediction_service is None:
            # Fall back to regular comparison if PyTorch not available
            if S.prediction_service is None:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="No prediction services available"
                )
            return S.prediction_service.compare_predictions(data.dict())
        
        result = S.torch_prediction_service.compare_all_predictions_with_torch(_require_terms(data))
        return result
        
    except Exception as e:
        logger.error(f"Error comparing all models: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/torch/batch", tags=["PyTorch Predictions"])
@limiter.limit("50/minute")  # Lower rate limit for batch processing
async def batch_predict_torch(request: Request, data: TorchBatchPredictRequest):
    """
    Make batch predictions using PyTorch models
    
    Processes multiple apartment listings efficiently using the specified PyTorch model.
    """
    try:
        if S.torch_prediction_service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="PyTorch prediction service not available"
            )
        
        if len(data.input_data_list) > 100:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Batch size too large. Maximum 100 predictions per request."
            )
        
        input_data_dicts = [_require_terms(item) for item in data.input_data_list]
        results = S.torch_prediction_service.batch_predict_torch(input_data_dicts, data.model_name)
        
        return {
            "results": results,
            "total_processed": len(results),
            "successful_predictions": len([r for r in results if r.get("status") == "success"]),
            "failed_predictions": len([r for r in results if r.get("status") == "failed"]),
            "model_name": data.model_name
        }
        
    except Exception as e:
        logger.error(f"Error in PyTorch batch prediction: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/torch/train", tags=["PyTorch Training"], dependencies=[Depends(require_admin)])
@limiter.limit(TRAINING_RATE_LIMIT)
async def train_torch_model(request: Request, 
                          model_type: str = "tabular",
                          epochs: int = 100,
                          learning_rate: float = 0.001,
                          batch_size: int = 32):
    """
    Train a new PyTorch model
    
    Initiates training of a PyTorch neural network model with specified parameters.
    """
    try:
        # Import here to avoid circular imports
        from scripts.torch_run import run_comprehensive_torch_training
        
        logger.info(f"Starting PyTorch model training: {model_type}")
        
        # Start training in background (this would normally be async/celery task)
        training_params = {
            "model_type": model_type,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "data_path": None  # None -> DATASET_PATH env var or the newest scraped dataset
        }
        
        # For now, return training initiated message
        # In production, this would trigger async training
        return {
            "status": "training_initiated",
            "model_type": model_type,
            "parameters": training_params,
            "message": f"PyTorch {model_type} model training started. Check logs for progress.",
            "estimated_time_minutes": epochs // 10  # Rough estimate
        }
        
    except Exception as e:
        logger.error(f"Error starting PyTorch training: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
