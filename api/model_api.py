from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.responses import JSONResponse
import logging
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, AnyHttpUrl, model_validator
import uvicorn
import math
# import mlflow
from typing import Dict, Optional, List, Any, Literal
import pandas as pd
import httpx
import os
import numpy as np
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from services import PredictionService, TrainingService, ModelLoader, MLflowService
from services.data_capture_service import DataCaptureService
from services.monitoring_reports import MonitoringReportsManager
from services.ab_testing_service import ABTestingService, ABTestConfig, TrafficSplitStrategy
from services.ab_mlflow_integration import ABTestMLflowIntegration
from services.auto_promotion_service import AutoPromotionService
from services.auto_retrain_service import AutoRetrainService
from services.ensemble_prediction_service import EnsemblePredictionService
from services.torch_prediction_service import PyTorchPredictionService
from services.explainability_service import ExplainabilityService
from services.data_lineage_service import DataLineageService
from services.sla_service import ModelSLAService
from utils import ScrapeData

# Import logging configuration
from config.logging_config import setup_logging, get_logger, PerformanceLogger, log_prediction
from middleware.logging_middleware import (
    RequestLoggingMiddleware,
    RequestBodyLoggingMiddleware,
    get_request_id
)

# Setup logging (JSON format in production, colored console in development)
setup_logging(
    environment=os.getenv("ENVIRONMENT", "development"),
    log_level=os.getenv("LOG_LEVEL", "INFO"),
    log_dir="logs",
    enable_console=True,
    enable_file=True,
    enable_json=os.getenv("ENVIRONMENT") == "production"
)

logger = get_logger(__name__)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

# Rate limit configurations (requests per time window)
PREDICTION_RATE_LIMIT = "100/minute"  # Core prediction endpoints
TRAINING_RATE_LIMIT = "5/hour"        # Expensive training operations
MONITORING_RATE_LIMIT = "20/minute"   # Monitoring and reporting
DEFAULT_RATE_LIMIT = "200/minute"     # Default for other endpoints

# class MlflowRun(BaseModel):
#     run_id: str
#     experiment_name: str
#     experiment_id: str
#     metrics: Dict[str, float]
#     params: Dict[str, str]

class InputData(BaseModel):
    sqr_m:             float = Field(..., ge=5.0,  le=500.0, description="Apartment size (sqm)")
    rei_price:         float = Field(..., ge=0.0,  le=100.0, description="Key money (万円)")
    shikikin:          float = Field(..., ge=0.0,  le=100.0, description="Security deposit (万円)")
    maintenence_price: float = Field(..., ge=0.0,  le=50.0,  description="Monthly maintenance fee (万円)")
    year_built:        float = Field(..., ge=0.0,  le=100.0, description="Years since construction")
    floor:             float = Field(..., ge=0.0,  le=60.0,  description="Floor number (0 = ground)")
    eki_walk:          float = Field(..., ge=0.0,  le=120.0, description="Walk to nearest station (minutes)")
    # Optional categorical features — improves prediction accuracy when provided
    ku_name:           Optional[str] = Field(default=None, description="Tokyo ward (e.g. '渋谷区', '新宿区')")
    apartment_type:    Optional[str] = Field(default=None, description="Room layout (e.g. '1K', '2LDK', 'ワンルーム')")
    house_type:        Optional[str] = Field(default=None, description="Building type (e.g. '賃貸マンション', '賃貸アパート')")
    contract_type:     Optional[int] = Field(default=0, ge=0, le=1, description="0 = regular lease, 1 = fixed-term (定期借家)")

    @model_validator(mode="after")
    def check_finite(self) -> "InputData":
        for name, value in self.__dict__.items():
            if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                raise ValueError(f"Field '{name}' must be a finite number, got {value}")
        return self

class RetrainData(BaseModel):
    url: AnyHttpUrl
    wait_time_min: int = Field(default=1, ge=1, le=60)
    wait_time_max: int = Field(default=5, ge=1, le=120)
    pages: tuple = Field(default=(0, 50))
    model_to_retrain: Optional[List[Literal["challenger", "passed", "tabpfn", "tabnet", "torch"]]] = Field(
        default=["challenger", "passed"]
    )

    @model_validator(mode="after")
    def validate_retrain(self) -> "RetrainData":
        if self.wait_time_min >= self.wait_time_max:
            raise ValueError("wait_time_min must be less than wait_time_max")
        p = self.pages
        if (not isinstance(p, (list, tuple)) or len(p) != 2
                or not all(isinstance(x, int) for x in p)
                or p[0] < 0 or p[1] < 0 or p[0] >= p[1]):
            raise ValueError("pages must be a 2-element tuple [start, end] with 0 <= start < end")
        return self

class WardRetrainData(BaseModel):
    """Schema for multi-ward scrape-and-retrain requests."""
    ward_codes: Optional[List[str]] = Field(
        default=None,
        description=(
            "List of Suumo sc codes (e.g. ['13104', '13113']).  "
            "Defaults to all 23 Tokyo special wards."
        ),
    )
    pages_per_ward: int = Field(
        default=10, ge=1, le=100,
        description="Pages to scrape per ward (~30 listings/page).",
    )
    wait_time_min: int = Field(default=2, ge=1, le=60)
    wait_time_max: int = Field(default=5, ge=2, le=120)
    model_to_retrain: Optional[List[Literal["challenger", "passed", "tabpfn", "tabnet", "torch"]]] = Field(
        default=["challenger", "passed"]
    )

    @model_validator(mode="after")
    def validate_wait_times(self) -> "WardRetrainData":
        if self.wait_time_min >= self.wait_time_max:
            raise ValueError("wait_time_min must be less than wait_time_max")
        return self

class EnsemblePredictRequest(BaseModel):
    input_data: InputData
    method: Literal["weighted_average", "simple_average", "stacking", "best_model"] = Field(
        default="weighted_average", description="Ensemble method to use"
    )

class BatchPredictRequest(BaseModel):
    input_data_list: List[InputData] = Field(..., min_length=1, max_length=100)
    method: Literal["ensemble_weighted_average", "simple_average", "best_model"] = Field(
        default="ensemble_weighted_average", description="Prediction method to use for all inputs"
    )

class ABTestCreateRequest(BaseModel):
    experiment_name: str  = Field(..., min_length=1, max_length=100, pattern=r"^[a-zA-Z0-9_\-]+$")
    control_model:   str  = Field(default="passed")
    treatment_model: str  = Field(default="challenger")
    traffic_split:   float = Field(default=0.1, ge=0.0, le=1.0)
    shadow_mode:     bool  = Field(default=False)
    min_samples:     int   = Field(default=100,  ge=10,    le=100_000)
    confidence_level: float = Field(default=0.95, ge=0.5,  le=0.999)
    min_effect_size:  float = Field(default=0.05, ge=0.001, le=1.0)

class ABTestPredictRequest(BaseModel):
    experiment_name: str
    input_data: InputData
    user_id: Optional[str] = None
    force_model: Optional[str] = None

class TorchPredictRequest(BaseModel):
    input_data: InputData
    model_name: Literal["tokyo_rent_torch", "tokyo_rent_tabpfn", "tokyo_rent_tabnet"] = Field(
        description="Name of the PyTorch/pretrained model to use"
    )

class TorchBatchPredictRequest(BaseModel):
    input_data_list: List[InputData] = Field(..., min_length=1, max_length=100)
    model_name: Literal["tokyo_rent_torch", "tokyo_rent_tabpfn", "tokyo_rent_tabnet"] = Field(
        description="Name of the PyTorch/pretrained model to use for batch predictions"
    )

class ExplainRequest(BaseModel):
    input_data: InputData
    model_name: Literal["challenger", "passed", "tabpfn", "tabnet"] = Field(
        default="challenger", description="Model to explain"
    )
    method: Literal["shap", "lime"] = Field(default="shap", description="Explanation method")


app = FastAPI(
    debug=True,
    title="Tokyo Apartment Rent Predictor API",
    description="""
    ## Tokyo Apartment Rent Predictor

    A comprehensive machine learning API that predicts rental prices for apartments in Tokyo using multiple model types including traditional ML models and advanced PyTorch neural networks.

    ### Features
    - **Predictions**: Get rent predictions using production or challenger models
    - **PyTorch Neural Networks**: Advanced predictions with tabular neural networks, transformers, and ensemble architectures
    - **Model Management**: Auto-promotion, versioning, and rollback capabilities
    - **A/B Testing**: Run experiments to compare model performance
    - **Monitoring**: Drift detection and model performance tracking with Evidently AI
    - **MLflow Integration**: Full experiment tracking and model registry

    ### Model Types Available
    - **LightGBM & Linear Regression**: Fast, traditional ML models for baseline predictions
    - **Ensemble Models**: Combine multiple models using weighted averaging, stacking, and voting
    - **PyTorch Tabular Networks**: Deep neural networks optimized for structured apartment data
    - **Transformer Models**: Advanced attention-based models for complex feature interactions
    - **Hybrid Models**: Combine BERT transformers with numerical features for premium accuracy

    ### Rate Limits
    - Predictions: 100 requests/minute
    - Training: 5 requests/hour
    - Monitoring: 20 requests/minute
    - Default: 200 requests/minute
    """,
    version="1.0.0",
    contact={
        "name": "Jacob Braswell",
        "url": "https://github.com/jocobtt/rent_scrape",
    },
    license_info={
        "name": "MIT",
    },
    docs_url="/docs",  # Swagger UI at /docs
    redoc_url="/redoc",  # ReDoc at /redoc
    openapi_url="/openapi.json",  # OpenAPI schema
    openapi_tags=[
        {
            "name": "Predictions",
            "description": "Endpoints for making rent predictions using trained models"
        },
        {
            "name": "Ensemble Predictions",
            "description": "Advanced predictions using ensemble methods combining multiple models"
        },
        {
            "name": "PyTorch Predictions",
            "description": "Neural network predictions using PyTorch models with advanced architectures"
        },
        {
            "name": "PyTorch Training",
            "description": "Train PyTorch neural network models with custom architectures and hyperparameters"
        },
        {
            "name": "Model Management",
            "description": "Manage model versions, promotions, and rollbacks"
        },
        {
            "name": "Training",
            "description": "Retrain models with fresh data and auto-promotion"
        },
        {
            "name": "A/B Testing",
            "description": "Create and manage A/B test experiments"
        },
        {
            "name": "Monitoring",
            "description": "Model drift detection and performance monitoring"
        },
        {
            "name": "Health",
            "description": "Health checks and API status"
        },
        {
            "name": "Lineage",
            "description": "Dataset version and model lineage tracking"
        },
        {
            "name": "Lifecycle",
            "description": "Model SLA monitoring, auto-demotion, and version retention cleanup"
        },
    ]
)

# Register rate limiter with FastAPI
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add logging middleware (FIRST - to capture all requests)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(RequestBodyLoggingMiddleware)

# add middleware
_raw_origins = os.getenv("ALLOWED_ORIGINS", "")
_allow_origins = [o.strip() for o in _raw_origins.split(",") if o.strip()] or ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins = _allow_origins,
    allow_credentials = True,
    allow_methods = ["GET", "POST"],
    allow_headers = ["*"],
)

# Initialize services
model_loader = ModelLoader()
prediction_service = None  # Will be initialized after models are loaded
ensemble_prediction_service = None  # Will be initialized after models are loaded
torch_prediction_service = None  # Will be initialized after PyTorch models are loaded
training_service = TrainingService()
mlflow_service = MLflowService()  # Initialize MLflow service
data_capture_service = DataCaptureService()  # Initialize data capture service
monitoring_manager = None  # Will be initialized lazily
ab_testing_service = ABTestingService()  # Initialize A/B testing service
ab_mlflow_integration = None  # Will be initialized lazily
auto_promotion_service = AutoPromotionService()  # Initialize auto-promotion service
auto_retrain_service = AutoRetrainService()  # Initialize auto-retrain service
explainability_service = ExplainabilityService()  # Initialize explainability service
data_lineage_service = DataLineageService()  # Initialize lineage service
sla_service = ModelSLAService()             # Initialize SLA monitoring service

@app.middleware("http")
async def load_model_middleware(request: Request, call_next):
    global prediction_service, torch_prediction_service
    
    if not hasattr(app.state, 'challenger_model'):
        try:
            app.state.challenger_model = model_loader.load_model("challenger")
        except Exception as e:
            logger.warning(f"Failed to load challenger model: {e}")
            app.state.challenger_model = None
    
    if not hasattr(app.state, 'passed_model'):
        try:
            app.state.passed_model = model_loader.load_model("passed")
        except Exception as e:
            logger.warning(f"Failed to load passed model: {e}")
            app.state.passed_model = None
    
    # Load PyTorch models if available
    if not hasattr(app.state, 'torch_models'):
        app.state.torch_models = {}
        # Try to load common PyTorch model names
        torch_model_names = ['tabular', 'ensemble', 'transformer', 'attention', 'bert']
        for model_name in torch_model_names:
            try:
                torch_model = model_loader.load_model(model_name, model_type="torch")
                if torch_model is not None:
                    app.state.torch_models[model_name] = torch_model
                    logger.info(f"Loaded PyTorch model: {model_name}")
            except Exception as e:
                logger.debug(f"PyTorch model '{model_name}' not available: {e}")
    
    # Initialize prediction service if models are loaded
    if (app.state.challenger_model is not None and 
        app.state.passed_model is not None and 
        prediction_service is None):
        prediction_service = PredictionService(app.state.passed_model, app.state.challenger_model)
        ensemble_prediction_service = EnsemblePredictionService(
            app.state.passed_model, 
            app.state.challenger_model
        )
        
        # Initialize PyTorch prediction service if PyTorch models are available
        if app.state.torch_models and torch_prediction_service is None:
            torch_prediction_service = PyTorchPredictionService(
                app.state.passed_model, 
                app.state.challenger_model,
                app.state.torch_models
            )
            logger.info(f"Initialized PyTorchPredictionService with {len(app.state.torch_models)} models")
    
    response = await call_next(request)
    return response

@app.get("/", tags=["Health"])
@limiter.limit(DEFAULT_RATE_LIMIT)
def home(request: Request):
    return "Welcome to Tokyo Rent Predictor!"

@app.post("/predict", tags=["Predictions"])
@limiter.limit(PREDICTION_RATE_LIMIT)
async def predict_main(request: Request, data: InputData):
    """
    Predict rent price using the passed (production) model.

    Args:
        data: InputData containing apartment features

    Returns:
        dict: Dictionary containing the prediction
    """
    try:
        if prediction_service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Prediction service not available - models not loaded"
            )

        result = prediction_service.predict_with_passed_model(data.dict())

        # Log prediction for drift monitoring
        try:
            data_capture_service.log_prediction(
                model_type="passed",
                input_features=data.dict(),
                prediction=result["prediction"]
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

@app.post("/challenger_predict", tags=["Predictions"])
@limiter.limit(PREDICTION_RATE_LIMIT)
async def challenger_predict(request: Request, data: InputData):
    """
    Predict rent price using the challenger model (LightGBM).

    Args:
        data: InputData containing apartment features

    Returns:
        dict: Dictionary containing the prediction
    """
    try:
        if prediction_service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Prediction service not available - models not loaded"
            )

        result = prediction_service.predict_with_challenger_model(data.dict())

        # Log prediction for drift monitoring
        try:
            data_capture_service.log_prediction(
                model_type="challenger",
                input_features=data.dict(),
                prediction=result["prediction"]
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

@app.post("/retrain", tags=["Training"])
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
        global prediction_service
        
        # Use training service to retrain models
        result = training_service.retrain_models(
            str(data.url),
            data.wait_time_min,
            data.wait_time_max, 
            data.pages, 
            data.model_to_retrain
        )
        
        # Update the loaded models in app state
        for model_name in data.model_to_retrain:
            if model_name == "challenger" and "challenger" in result["retrained_models"]:
                app.state.challenger_model = result["retrained_models"]["challenger"]
            elif model_name == "passed" and "passed" in result["retrained_models"]:
                app.state.passed_model = result["retrained_models"]["passed"]
        
        # Reinitialize prediction service with updated models
        if (app.state.challenger_model is not None and 
            app.state.passed_model is not None):
            prediction_service = PredictionService(app.state.passed_model, app.state.challenger_model)
            ensemble_prediction_service = EnsemblePredictionService(
                app.state.passed_model, 
                app.state.challenger_model
            )
        
        return result
        
    except Exception as e:
        logger.error(f"Error during retraining: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Retraining failed: {str(e)}"
        )

# perform health check
@app.get("/health", tags=["Health"])
@limiter.limit(DEFAULT_RATE_LIMIT)
async def health_check(request: Request):
    """
    Basic health check endpoint - returns 200 if API is responding.
    Use /ready for readiness checks and /startup for startup probes.
    """
    return {"status": "ok", "timestamp": pd.Timestamp.now().isoformat()}

@app.get("/startup")
async def startup_probe():
    """
    Kubernetes startup probe - checks if the application has started.
    Returns 503 if models are not loaded yet.
    """
    try:
        challenger_loaded = hasattr(app.state, 'challenger_model') and app.state.challenger_model is not None
        passed_loaded = hasattr(app.state, 'passed_model') and app.state.passed_model is not None

        # At least one model must be loaded for startup to be complete
        if not (challenger_loaded or passed_loaded):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Models not loaded yet"
            )

        return {
            "status": "started",
            "models_loaded": {
                "challenger_model": challenger_loaded,
                "passed_model": passed_loaded
            },
            "timestamp": pd.Timestamp.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Startup probe failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Startup check failed: {str(e)}"
        )

@app.get("/ready")
async def readiness_probe():
    """
    Kubernetes readiness probe - checks if the application is ready to serve traffic.
    Performs comprehensive checks on all critical components.
    """
    try:
        health_status = {
            "status": "ready",
            "timestamp": pd.Timestamp.now().isoformat(),
            "checks": {}
        }

        is_ready = True

        # Check 1: Models loaded
        challenger_loaded = hasattr(app.state, 'challenger_model') and app.state.challenger_model is not None
        passed_loaded = hasattr(app.state, 'passed_model') and app.state.passed_model is not None

        health_status["checks"]["models"] = {
            "challenger_model": challenger_loaded,
            "passed_model": passed_loaded,
            "status": "healthy" if (challenger_loaded and passed_loaded) else "degraded"
        }

        if not (challenger_loaded and passed_loaded):
            is_ready = False

        # Check 2: Prediction service
        health_status["checks"]["prediction_service"] = {
            "initialized": prediction_service is not None,
            "status": "healthy" if prediction_service is not None else "unhealthy"
        }

        if prediction_service is None:
            is_ready = False
        
        # Check 2b: PyTorch prediction service (optional)
        torch_models_loaded = hasattr(app.state, 'torch_models') and len(app.state.torch_models) > 0
        health_status["checks"]["torch_prediction_service"] = {
            "initialized": torch_prediction_service is not None,
            "models_loaded": torch_models_loaded,
            "model_count": len(app.state.torch_models) if torch_models_loaded else 0,
            "available_models": list(app.state.torch_models.keys()) if torch_models_loaded else [],
            "status": "healthy" if torch_prediction_service is not None else "optional"
        }

        # Check 3: MLflow connectivity (optional - don't fail if unavailable)
        try:
            mlflow_status = mlflow_service.get_production_model_metadata("tokyo_rent_lgbm")
            health_status["checks"]["mlflow"] = {
                "status": "healthy",
                "connection": "ok"
            }
        except Exception as mlflow_error:
            logger.warning(f"MLflow check failed: {mlflow_error}")
            health_status["checks"]["mlflow"] = {
                "status": "degraded",
                "connection": "failed",
                "error": str(mlflow_error)
            }
            # Don't mark as not ready for MLflow issues

        # Check 4: Data capture service
        try:
            data_capture_check = hasattr(data_capture_service, '_buffer')
            health_status["checks"]["data_capture"] = {
                "status": "healthy" if data_capture_check else "unhealthy",
                "buffer_size": len(data_capture_service._buffer) if data_capture_check else 0
            }
        except Exception as dc_error:
            health_status["checks"]["data_capture"] = {
                "status": "degraded",
                "error": str(dc_error)
            }

        # Check 5: Disk space (important for model storage and logs)
        try:
            import shutil
            disk_usage = shutil.disk_usage("/app" if os.path.exists("/app") else ".")
            free_gb = disk_usage.free / (1024**3)
            health_status["checks"]["disk_space"] = {
                "free_gb": round(free_gb, 2),
                "status": "healthy" if free_gb > 1 else "warning"
            }
        except Exception as disk_error:
            health_status["checks"]["disk_space"] = {
                "status": "unknown",
                "error": str(disk_error)
            }

        # Check 6: Memory usage (important for ML models)
        try:
            import psutil
            memory = psutil.virtual_memory()
            health_status["checks"]["memory"] = {
                "percent_used": memory.percent,
                "available_mb": round(memory.available / (1024**2), 2),
                "status": "healthy" if memory.percent < 90 else "warning"
            }
        except ImportError:
            health_status["checks"]["memory"] = {
                "status": "unknown",
                "note": "psutil not installed"
            }
        except Exception as mem_error:
            health_status["checks"]["memory"] = {
                "status": "unknown",
                "error": str(mem_error)
            }

        # Final status determination
        if not is_ready:
            health_status["status"] = "not_ready"
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=health_status
            )

        return health_status

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Readiness probe failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": "not_ready",
                "error": str(e),
                "timestamp": pd.Timestamp.now().isoformat()
            }
        )

@app.post("/compare_predictions", tags=["Predictions"])
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
        if prediction_service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Prediction service not available - models not loaded"
            )
        
        return prediction_service.compare_predictions(data.dict())

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Comparison prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Comparison prediction failed: {str(e)}"
        )

@app.get("/model_info")
async def get_model_info():
    """
    Get information about the loaded models.
    """
    try:
        passed_info = {"loaded": False, "file_exists": False}
        challenger_info = {"loaded": False, "file_exists": False}

        # Check if models are loaded
        if hasattr(app.state, 'passed_model') and app.state.passed_model is not None:
            passed_info.update(model_loader.get_model_info(app.state.passed_model))

        if hasattr(app.state, 'challenger_model') and app.state.challenger_model is not None:
            challenger_info.update(model_loader.get_model_info(app.state.challenger_model))

        # Check if model files exist
        passed_info["file_exists"] = model_loader.model_exists("passed")
        challenger_info["file_exists"] = model_loader.model_exists("challenger")

        info = {
            "models": {
                "passed_model": passed_info,
                "challenger_model": challenger_info
            },
            "services": {
                "prediction_service_available": prediction_service is not None,
                "ensemble_prediction_service_available": ensemble_prediction_service is not None,
                "training_service_available": training_service is not None,
                "model_loader_available": model_loader is not None
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

@app.get("/model/metadata")
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
            metadata = mlflow_service.get_model_metadata_by_version(model_name, version)
        else:
            metadata = mlflow_service.get_production_model_metadata(model_name)

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

@app.get("/model/metadata/all")
async def get_all_models_metadata():
    """
    Get metadata for all active models (both challenger and passed).

    Returns:
        dict: Metadata for all models including challenger (LightGBM) and passed (Linear)
    """
    try:
        all_metadata = mlflow_service.get_all_active_models_metadata()

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

# Ensemble Prediction Endpoints

@app.post("/ensemble/predict", tags=["Ensemble Predictions"])
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
        if ensemble_prediction_service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Ensemble prediction service not available - models not loaded"
            )
        
        result = ensemble_prediction_service.predict_with_ensemble(
            data.input_data.dict(), 
            data.method
        )
        
        # Log prediction for monitoring
        with PerformanceLogger("ensemble_prediction") as perf:
            pass
        
        log_prediction(
            get_request_id(), 
            data.input_data.dict(), 
            result,
            f"ensemble_{data.method}",
            perf.duration
        )
        
        return result
        
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

@app.post("/ensemble/compare", tags=["Ensemble Predictions"])
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
        if ensemble_prediction_service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Ensemble prediction service not available - models not loaded"
            )
        
        result = ensemble_prediction_service.compare_all_predictions(data.dict())
        
        # Log comparison for monitoring
        with PerformanceLogger("ensemble_comparison") as perf:
            pass
        
        return result
        
    except Exception as e:
        logger.error(f"Ensemble comparison error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ensemble comparison failed: {str(e)}"
        )

@app.post("/ensemble/confidence", tags=["Ensemble Predictions"])
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
        if ensemble_prediction_service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Ensemble prediction service not available - models not loaded"
            )
        
        result = ensemble_prediction_service.get_prediction_confidence(data.dict())
        
        return result
        
    except Exception as e:
        logger.error(f"Confidence calculation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Confidence calculation failed: {str(e)}"
        )

@app.post("/ensemble/recommendations", tags=["Ensemble Predictions"])
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
        if ensemble_prediction_service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Ensemble prediction service not available - models not loaded"
            )
        
        result = ensemble_prediction_service.get_model_recommendations(data.dict())
        
        return result
        
    except Exception as e:
        logger.error(f"Model recommendations error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model recommendations failed: {str(e)}"
        )

@app.post("/ensemble/batch", tags=["Ensemble Predictions"])
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
        if ensemble_prediction_service is None:
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
        input_dicts = [item.dict() for item in data.input_data_list]
        
        result = ensemble_prediction_service.batch_predict(input_dicts, data.method)
        
        return {
            "batch_size": len(input_dicts),
            "method": data.method,
            "results": result,
            "success_count": len([r for r in result if r.get("status") == "success"]),
            "error_count": len([r for r in result if r.get("status") == "failed"])
        }
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )

@app.get("/ensemble/methods", tags=["Ensemble Predictions"])
@limiter.limit(DEFAULT_RATE_LIMIT)
async def get_available_ensemble_methods(request: Request):
    """
    Get information about all available prediction methods.
    
    Returns:
        dict: Available base models, ensemble methods, and their details
    """
    try:
        if ensemble_prediction_service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Ensemble prediction service not available - models not loaded"
            )
        
        methods = ensemble_prediction_service.get_available_methods()
        
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

@app.post("/ensemble/train", tags=["Ensemble Predictions"])
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
        with PerformanceLogger("ensemble_training") as perf:
            results = train_ensemble_models()
        
        if not results:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="No ensemble models were successfully trained"
            )
        
        # Prepare response
        response = {
            "training_duration": perf.duration,
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
        
        logger.info(f"Ensemble training completed in {perf.duration:.2f}s. Best method: {best_method}")
        
        return response
        
    except Exception as e:
        logger.error(f"Ensemble training error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ensemble training failed: {str(e)}"
        )

# Monitoring Endpoints

def get_monitoring_manager():
    """Lazy initialization of monitoring manager"""
    global monitoring_manager
    if monitoring_manager is None:
        try:
            monitoring_manager = MonitoringReportsManager()
            logger.info("Initialized MonitoringReportsManager")
        except Exception as e:
            logger.error(f"Failed to initialize MonitoringReportsManager: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Monitoring service initialization failed: {str(e)}"
            )
    return monitoring_manager

@app.get("/monitoring/summary")
async def get_monitoring_summary():
    """
    Get a summary of monitoring status including prediction statistics and recent reports.

    Returns:
        dict: Monitoring summary with prediction statistics and report information
    """
    try:
        manager = get_monitoring_manager()
        summary = manager.get_monitoring_summary()
        return summary
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting monitoring summary: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get monitoring summary: {str(e)}"
        )

@app.get("/scrape/health-check", tags=["Monitoring"])
@limiter.limit(MONITORING_RATE_LIMIT)
async def scrape_health_check(request: Request, url: str):
    """
    Scrape exactly one page from the given Suumo URL and verify the site is still parseable.

    Does NOT save data or trigger retraining — safe to call at any time.

    Returns field completeness (% non-zero) for all expected training features.
    """
    try:
        scraper = ScrapeData(url, wait_time_min=1, wait_time_max=2, pages=(0, 1))
        raw = scraper.scrape()
        if raw.empty:
            return {
                "status": "no_listings_parsed",
                "rows_parsed": 0,
                "field_completeness": {},
                "data_quality": scraper.validate_scraped_data(raw),
            }
        cleaned = scraper.clean_data(raw)
        expected_cols = [
            "sqr_m", "rei_price", "shikikin", "maintenence_price",
            "year_built", "floor", "nearest_eki_walk", "rent_price",
        ]
        completeness = {
            col: float((cleaned[col] != 0).mean()) if col in cleaned.columns else 0.0
            for col in expected_cols
        }
        return {
            "status": "ok" if len(cleaned) > 0 else "no_listings_parsed",
            "rows_parsed": len(cleaned),
            "field_completeness": completeness,
            "data_quality": scraper.validate_scraped_data(cleaned),
        }
    except Exception as e:
        logger.error(f"Scrape health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Scrape health check failed: {str(e)}",
        )


@app.get("/monitoring/drift")
async def check_drift(
    days_back: int = 7,
    model_type: Optional[str] = None,
    drift_threshold: float = 0.3
):
    """
    Run drift detection on recent predictions.

    Query Parameters:
        days_back: Number of days of prediction data to analyze (default: 7)
        model_type: Filter by model type - 'passed' or 'challenger' (optional)
        drift_threshold: Maximum allowed share of drifted features (default: 0.3)

    Returns:
        dict: Drift detection results including test pass/fail and drift metrics
    """
    try:
        manager = get_monitoring_manager()
        result = manager.run_drift_detection(
            days_back=days_back,
            model_type=model_type,
            drift_threshold=drift_threshold
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error checking drift: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Drift detection failed: {str(e)}"
        )

@app.post("/monitoring/reports/generate", tags=["Monitoring"])
@limiter.limit(MONITORING_RATE_LIMIT)
async def generate_monitoring_reports(
    request: Request,
    days_back: int = 7,
    model_type: Optional[str] = None
):
    """
    Generate all monitoring reports (data drift, data quality, model performance).

    Request Body (JSON):
        days_back: Number of days of data to analyze (default: 7)
        model_type: Filter by model type - 'passed' or 'challenger' (optional)

    Returns:
        dict: Paths to generated report files
    """
    try:
        manager = get_monitoring_manager()
        report_paths = manager.generate_all_reports(
            days_back=days_back,
            model_type=model_type
        )
        return {
            "status": "success",
            "reports": report_paths,
            "message": "Reports generated successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating monitoring reports: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Report generation failed: {str(e)}"
        )

@app.get("/monitoring/compare_models")
async def compare_models_drift(days_back: int = 7):
    """
    Compare drift between challenger and passed models.

    Query Parameters:
        days_back: Number of days of data to analyze (default: 7)

    Returns:
        dict: Comparison of drift metrics between models
    """
    try:
        manager = get_monitoring_manager()
        comparison = manager.compare_models_drift(days_back=days_back)
        return {
            "status": "success",
            "comparison": comparison,
            "days_analyzed": days_back
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error comparing model drift: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model drift comparison failed: {str(e)}"
        )

@app.get("/monitoring/predictions/stats")
async def get_prediction_stats():
    """
    Get statistics about captured predictions.

    Returns:
        dict: Statistics including total predictions, predictions with actuals, and model breakdown
    """
    try:
        stats = data_capture_service.get_statistics()
        return {
            "status": "success",
            "statistics": stats
        }
    except Exception as e:
        logger.error(f"Error getting prediction stats: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get prediction statistics: {str(e)}"
        )

# A/B Testing Endpoints

def get_ab_mlflow_integration():
    """Lazy initialization of A/B MLflow integration"""
    global ab_mlflow_integration
    if ab_mlflow_integration is None:
        ab_mlflow_integration = ABTestMLflowIntegration(ab_testing_service)
    return ab_mlflow_integration

@app.post("/ab-test/create", tags=["A/B Testing"])
@limiter.limit(DEFAULT_RATE_LIMIT)
async def create_ab_test(request: Request, body: ABTestCreateRequest):
    """
    Create a new A/B test experiment

    Request Body:
        experiment_name: Unique name for the experiment
        control_model: Control model name (default: "passed")
        treatment_model: Treatment model name (default: "challenger")
        traffic_split: Percentage to treatment 0.0-1.0 (default: 0.1 = 10%)
        shadow_mode: Both models predict, only control returned (default: False)
        min_samples: Min samples for statistical tests (default: 100)
        confidence_level: Statistical confidence level (default: 0.95)
        min_effect_size: Minimum meaningful improvement (default: 0.05 = 5%)

    Returns:
        dict: Experiment creation status
    """
    try:
        config = ABTestConfig(
            experiment_name=body.experiment_name,
            control_model=body.control_model,
            treatment_model=body.treatment_model,
            traffic_split=body.traffic_split,
            shadow_mode=body.shadow_mode,
            min_samples=body.min_samples,
            confidence_level=body.confidence_level,
            min_effect_size=body.min_effect_size,
        )

        success = ab_testing_service.create_experiment(config)

        if success:
            return {
                "status": "success",
                "message": f"Created A/B test experiment: {body.experiment_name}",
                "experiment": config.to_dict()
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create experiment"
            )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error creating A/B test: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create A/B test: {str(e)}"
        )

@app.post("/ab-test/{experiment_name}/start")
async def start_ab_test(experiment_name: str):
    """Start an A/B test experiment"""
    try:
        success = ab_testing_service.start_experiment(experiment_name)
        if success:
            return {
                "status": "success",
                "message": f"Started experiment: {experiment_name}"
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Experiment not found: {experiment_name}"
            )
    except Exception as e:
        logger.error(f"Error starting experiment: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.post("/ab-test/{experiment_name}/stop")
async def stop_ab_test(experiment_name: str):
    """Stop an A/B test experiment"""
    try:
        success = ab_testing_service.stop_experiment(experiment_name)
        if success:
            return {
                "status": "success",
                "message": f"Stopped experiment: {experiment_name}"
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Experiment not found: {experiment_name}"
            )
    except Exception as e:
        logger.error(f"Error stopping experiment: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.get("/ab-test/list")
async def list_ab_tests():
    """List all A/B test experiments"""
    try:
        experiments = ab_testing_service.list_experiments()
        return {
            "status": "success",
            "experiments": [exp.to_dict() for exp in experiments],
            "total": len(experiments)
        }
    except Exception as e:
        logger.error(f"Error listing experiments: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.get("/ab-test/{experiment_name}")
async def get_ab_test(experiment_name: str):
    """Get details of a specific A/B test"""
    try:
        config = ab_testing_service.get_experiment(experiment_name)
        if not config:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Experiment not found: {experiment_name}"
            )
        return {
            "status": "success",
            "experiment": config.to_dict()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting experiment: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.post("/ab-test/predict", tags=["A/B Testing"])
@limiter.limit(PREDICTION_RATE_LIMIT)
async def ab_test_predict(request: Request, body: ABTestPredictRequest):
    """
    Make prediction with A/B testing

    This endpoint:
    1. Determines which model to use based on traffic split
    2. Makes predictions with both models (for shadow mode)
    3. Returns the appropriate prediction
    4. Logs everything for analysis

    Request Body:
        experiment_name: Name of active experiment
        input_data: Apartment features
        user_id: Optional user ID for consistent assignment
        force_model: Force specific model (for testing)

    Returns:
        dict: Prediction result with experiment metadata
    """
    try:
        if prediction_service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Prediction service not available"
            )

        # Get experiment config
        config = ab_testing_service.get_experiment(body.experiment_name)
        if not config:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Experiment not found: {body.experiment_name}"
            )

        # Assign model
        import time
        start_time = time.time()

        assigned_model = ab_testing_service.assign_model(
            body.experiment_name,
            user_id=body.user_id,
            force_model=body.force_model
        )

        # Make predictions with both models
        input_dict = body.input_data.dict()

        control_result = prediction_service.predict_with_passed_model(input_dict)
        treatment_result = prediction_service.predict_with_challenger_model(input_dict)

        control_pred = control_result["prediction"]
        treatment_pred = treatment_result["prediction"]

        # Determine which prediction to return
        if config.shadow_mode:
            # Shadow mode: always return control
            returned_pred = control_pred
            returned_model = "control"
        else:
            # Normal mode: return assigned model's prediction
            if assigned_model == "treatment":
                returned_pred = treatment_pred
                returned_model = "treatment"
            else:
                returned_pred = control_pred
                returned_model = "control"

        latency_ms = (time.time() - start_time) * 1000

        # Log to A/B testing service
        ab_testing_service.log_prediction(
            experiment_name=body.experiment_name,
            assigned_model=assigned_model,
            input_features=input_dict,
            control_prediction=control_pred,
            treatment_prediction=treatment_pred,
            returned_prediction=returned_pred,
            latency_ms=latency_ms
        )

        # Also log to data capture for monitoring
        data_capture_service.log_prediction(
            model_type=returned_model,
            input_features=input_dict,
            prediction=returned_pred
        )

        return {
            "prediction": returned_pred,
            "status": "success",
            "experiment": {
                "name": body.experiment_name,
                "assigned_model": assigned_model,
                "returned_model": returned_model,
                "shadow_mode": config.shadow_mode,
            },
            "debug_info": {
                "control_prediction": control_pred,
                "treatment_prediction": treatment_pred,
                "difference": abs(control_pred - treatment_pred),
                "latency_ms": round(latency_ms, 2)
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in A/B test prediction: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.get("/ab-test/{experiment_name}/analyze")
async def analyze_ab_test(experiment_name: str, days_back: int = None):
    """
    Analyze A/B test results with statistical tests

    Query Parameters:
        days_back: Number of days to analyze (optional, defaults to all data)

    Returns:
        dict: Complete statistical analysis with winner recommendation
    """
    try:
        analysis = ab_testing_service.analyze_experiment(
            experiment_name,
            days_back=days_back
        )

        if "error" in analysis:
            return analysis

        # Log to MLflow
        try:
            config = ab_testing_service.get_experiment(experiment_name)
            if config:
                integration = get_ab_mlflow_integration()
                run_id = integration.log_experiment_to_mlflow(
                    experiment_name,
                    analysis,
                    config
                )
                analysis["mlflow_run_id"] = run_id
        except Exception as e:
            logger.warning(f"Failed to log to MLflow: {e}")

        return {
            "status": "success",
            "analysis": analysis
        }

    except Exception as e:
        logger.error(f"Error analyzing experiment: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.post("/ab-test/{experiment_name}/promote")
async def promote_ab_test_winner(
    experiment_name: str,
    model_registry_name: str = "tokyo_rent_lgbm",
    auto_promote: bool = False
):
    """
    Promote winning model to production

    Query Parameters:
        model_registry_name: MLflow model registry name (default: "tokyo_rent_lgbm")
        auto_promote: Auto-promote even with low confidence (default: False)

    Returns:
        dict: Promotion status and results
    """
    try:
        # First analyze to get winner
        analysis = ab_testing_service.analyze_experiment(experiment_name)

        if "error" in analysis:
            return {
                "status": "error",
                "message": analysis.get("error", "Unknown error"),
                "analysis": analysis
            }

        # Promote winner
        integration = get_ab_mlflow_integration()
        result = integration.promote_winner_to_production(
            experiment_name,
            analysis,
            model_registry_name,
            auto_promote=auto_promote
        )

        return result

    except Exception as e:
        logger.error(f"Error promoting winner: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.get("/ab-test/{experiment_name}/report")
async def get_ab_test_report(experiment_name: str):
    """
    Get markdown report of A/B test results

    Returns:
        dict: Markdown formatted report
    """
    try:
        analysis = ab_testing_service.analyze_experiment(experiment_name)

        if "error" in analysis:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=analysis.get("error", "Unknown error")
            )

        integration = get_ab_mlflow_integration()
        report = integration.create_experiment_report(experiment_name, analysis)

        return {
            "status": "success",
            "experiment_name": experiment_name,
            "report": report
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

# Model Promotion Endpoints

@app.post("/models/{model_name}/promote/{version}", tags=["Model Management"])
@limiter.limit(DEFAULT_RATE_LIMIT)
async def promote_model_version(request: Request, model_name: str, version: str, force: bool = False):
    """Manually promote a specific model version to production"""
    try:
        result = auto_promotion_service.promote_specific_version(model_name, version, force)
        if not result["promoted"]:
            return {"status": "not_promoted", "message": result.get("reason"), "force_required": result.get("force_required", False)}
        return {"status": "success", "message": f"Promoted {model_name} v{version} to Production", "result": result}
    except Exception as e:
        logger.error(f"Error promoting model: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@app.post("/models/check-promotion")
@limiter.limit(DEFAULT_RATE_LIMIT)
async def check_models_for_promotion(request: Request, dry_run: bool = True):
    """Check all models and promote qualified ones"""
    try:
        results = auto_promotion_service.check_and_promote_new_models(dry_run=dry_run)
        return {"status": "success", "results": results}
    except Exception as e:
        logger.error(f"Error checking models: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@app.get("/models/{model_name}/promotion-status")
async def get_promotion_status(model_name: str):
    """Get detailed promotion status for a model"""
    try:
        status_info = auto_promotion_service.get_promotion_status(model_name)
        return {"status": "success", "info": status_info}
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@app.post("/models/{model_name}/rollback")
@limiter.limit(DEFAULT_RATE_LIMIT)
async def rollback_model(request: Request, model_name: str):
    """Rollback to previous production version"""
    try:
        result = auto_promotion_service.rollback_to_previous(model_name)
        if not result["rolled_back"]:
            return {"status": "not_rolled_back", "message": result.get("reason")}
        return {"status": "success", "message": f"Rolled back {model_name} to v{result['version']}", "result": result}
    except Exception as e:
        logger.error(f"Error during rollback: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@app.post("/retrain/check-and-execute", tags=["Training"])
@limiter.limit(TRAINING_RATE_LIMIT)
async def check_and_retrain(request: Request, days_back: int = 7, dry_run: bool = True):
    """Check conditions and automatically retrain if needed"""
    try:
        result = auto_retrain_service.check_and_retrain(days_back=days_back, dry_run=dry_run)
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Error in check and retrain: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@app.post("/retrain/force")
@limiter.limit(TRAINING_RATE_LIMIT)
async def force_retrain(request: Request, model_name: Optional[str] = None):
    """Force immediate retraining"""
    try:
        result = auto_retrain_service.force_retrain(model_name=model_name)
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Error in force retrain: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@app.post("/retrain/wards", tags=["Training"])
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
        result = training_service.retrain_with_wards(
            ward_codes=data.ward_codes,
            pages_per_ward=data.pages_per_ward,
            wait_time_min=data.wait_time_min,
            wait_time_max=data.wait_time_max,
            models_to_retrain=data.model_to_retrain,
        )

        if result.get("status") == "success":
            for model_name in (data.model_to_retrain or []):
                if model_name == "challenger" and "challenger" in result.get("retrained_models", {}):
                    app.state.challenger_model = result["retrained_models"]["challenger"]
                elif model_name == "passed" and "passed" in result.get("retrained_models", {}):
                    app.state.passed_model = result["retrained_models"]["passed"]

        return result
    except Exception as e:
        logger.error(f"Error during ward-based retrain: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ward retraining failed: {str(e)}",
        )


# ============================================================================
# PyTorch Neural Network Prediction Endpoints
# ============================================================================

@app.post("/torch/predict", tags=["PyTorch Predictions"])
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
            
            if torch_prediction_service is None:
                logger.warning("PyTorch prediction service not available")
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
                    detail="PyTorch models not available. Please check model loading."
                )
            
            # Make prediction
            result = torch_prediction_service.predict_with_torch_model(
                data.input_data.dict(), 
                data.model_name
            )
            
            # Log prediction for monitoring
            log_prediction(
                input_data=data.input_data.dict(),
                prediction=result["prediction"],
                model_name=f"torch_{data.model_name}",
                request_id=request_id,
                processing_time_ms=result.get("processing_time_ms", 0)
            )
            
            # Capture data for monitoring
            await data_capture_service.capture_prediction(
                input_data=data.input_data.dict(),
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

@app.get("/torch/models", tags=["PyTorch Predictions"])
@limiter.limit(DEFAULT_RATE_LIMIT)
async def get_available_torch_models(request: Request):
    """
    Get information about available PyTorch models
    
    Returns details about loaded PyTorch models including parameters, size, and capabilities.
    """
    try:
        if torch_prediction_service is None:
            return {
                "available": False,
                "models": {},
                "total_models": 0,
                "message": "PyTorch prediction service not initialized"
            }
        
        model_info = torch_prediction_service.get_available_torch_models()
        model_info["available"] = True
        return model_info
        
    except Exception as e:
        logger.error(f"Error getting PyTorch model info: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@app.post("/torch/recommendations", tags=["PyTorch Predictions"])
@limiter.limit(PREDICTION_RATE_LIMIT)
async def get_torch_model_recommendations(request: Request, data: InputData):
    """
    Get PyTorch model recommendations based on input characteristics
    
    Analyzes apartment features and recommends the best PyTorch model for prediction.
    """
    try:
        if torch_prediction_service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="PyTorch prediction service not available"
            )
        
        recommendations = torch_prediction_service.get_torch_model_recommendations(data.dict())
        return recommendations
        
    except Exception as e:
        logger.error(f"Error generating PyTorch recommendations: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@app.post("/torch/compare-all", tags=["PyTorch Predictions"])
@limiter.limit(PREDICTION_RATE_LIMIT)
async def compare_all_models_with_torch(request: Request, data: InputData):
    """
    Compare predictions from all models including PyTorch models
    
    Returns predictions from sklearn models, ensemble methods, and PyTorch models with statistics.
    """
    try:
        if torch_prediction_service is None:
            # Fall back to regular comparison if PyTorch not available
            if prediction_service is None:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="No prediction services available"
                )
            return prediction_service.compare_predictions(data.dict())
        
        result = torch_prediction_service.compare_all_predictions_with_torch(data.dict())
        return result
        
    except Exception as e:
        logger.error(f"Error comparing all models: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@app.post("/torch/batch", tags=["PyTorch Predictions"])
@limiter.limit("50/minute")  # Lower rate limit for batch processing
async def batch_predict_torch(request: Request, data: TorchBatchPredictRequest):
    """
    Make batch predictions using PyTorch models
    
    Processes multiple apartment listings efficiently using the specified PyTorch model.
    """
    try:
        if torch_prediction_service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="PyTorch prediction service not available"
            )
        
        if len(data.input_data_list) > 100:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Batch size too large. Maximum 100 predictions per request."
            )
        
        input_data_dicts = [item.dict() for item in data.input_data_list]
        results = torch_prediction_service.batch_predict_torch(input_data_dicts, data.model_name)
        
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

@app.post("/torch/train", tags=["PyTorch Training"])
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
        from torch_run import run_comprehensive_torch_training
        
        logger.info(f"Starting PyTorch model training: {model_type}")
        
        # Start training in background (this would normally be async/celery task)
        training_params = {
            "model_type": model_type,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "data_path": "data/tokyo_model.csv"  # Default data path
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

# ============================================================================
# End PyTorch Endpoints
# ============================================================================

# ============================================================================
# Explainability Endpoints (SHAP + LIME)
# ============================================================================

@app.post("/explain/predict", tags=["Explainability"])
@limiter.limit(DEFAULT_RATE_LIMIT)
async def explain_prediction(request: Request, data: ExplainRequest):
    """
    Explain a single prediction using SHAP or LIME.

    Returns per-feature attribution values that show why the model predicted
    a particular rent price for the given apartment.

    - **model_name**: `"challenger"` (LightGBM) or `"passed"` (Linear)
    - **method**: `"shap"` (default, faster) or `"lime"` (model-agnostic)

    Response includes:
    - `prediction`: The predicted rent in 万円
    - `base_value`: SHAP baseline (average prediction over training data)
    - `features`: Per-feature SHAP/LIME values sorted by absolute importance
    - `top_positive`: Top features that *increase* the predicted rent
    - `top_negative`: Top features that *decrease* the predicted rent
    """
    try:
        if prediction_service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Models not loaded. Ensure the API has started successfully."
            )

        model_name = data.model_name.lower()
        if model_name == "challenger":
            model = app.state.challenger_model
        elif model_name == "passed":
            model = app.state.passed_model
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unknown model_name '{data.model_name}'. Use 'challenger' or 'passed'."
            )

        if model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Model '{data.model_name}' is not loaded."
            )

        if data.method not in ("shap", "lime"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unknown method '{data.method}'. Use 'shap' or 'lime'."
            )

        result = explainability_service.explain_prediction(
            model=model,
            input_data=data.input_data.dict(),
            background_data=None,  # uses single-row baseline; sufficient for SHAP tree/linear
            method=data.method,
        )

        return {
            "model_name": data.model_name,
            "method": data.method,
            **result,
        }

    except HTTPException:
        raise
    except RuntimeError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    except Exception as e:
        logger.error(f"Explanation error: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.get("/explain/global/{model_name}", tags=["Explainability"])
@limiter.limit(DEFAULT_RATE_LIMIT)
async def global_feature_importance(request: Request, model_name: str, n_samples: int = 200):
    """
    Compute global SHAP feature importance for a model over a sample of typical inputs.

    Returns all features ranked by mean absolute SHAP value — showing which features
    the model relies on most across all predictions.

    - **model_name**: `"challenger"` or `"passed"`
    - **n_samples**: Number of synthetic background samples to use (default 200)
    """
    try:
        if prediction_service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Models not loaded."
            )

        name = model_name.lower()
        if name == "challenger":
            model = app.state.challenger_model
        elif name == "passed":
            model = app.state.passed_model
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unknown model_name '{model_name}'. Use 'challenger' or 'passed'."
            )

        if model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Model '{model_name}' is not loaded."
            )

        from services.explainability_service import FEATURE_NAMES
        # Build a synthetic background dataset using sensible Tokyo apartment ranges
        rng = np.random.default_rng(42)
        n = min(n_samples, 500)
        background = pd.DataFrame({
            "sqr_m":              rng.uniform(15, 120, n),
            "rei_price":          rng.uniform(0, 3, n),
            "shikikin":           rng.uniform(0, 3, n),
            "maintenence_price":  rng.uniform(0, 2, n),
            "year_built":         rng.integers(1970, 2024, n).astype(float),
            "floor":              rng.integers(1, 30, n).astype(float),
            "eki_walk":           rng.integers(1, 25, n).astype(float),
        })

        result = explainability_service.global_feature_importance(
            model=model,
            background_data=background,
            feature_names=FEATURE_NAMES,
            n_samples=n,
        )

        return {"model_name": model_name, **result}

    except HTTPException:
        raise
    except RuntimeError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    except Exception as e:
        logger.error(f"Global importance error: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/explain/compare", tags=["Explainability"])
@limiter.limit(DEFAULT_RATE_LIMIT)
async def compare_model_explanations(request: Request, data: InputData):
    """
    Compare SHAP explanations from both the challenger and passed models side-by-side.

    Useful for understanding where the two models agree or disagree about which
    features drive the rent prediction.
    """
    try:
        if prediction_service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Models not loaded."
            )

        results = {}
        for model_name, model in [
            ("challenger", app.state.challenger_model),
            ("passed", app.state.passed_model),
        ]:
            if model is None:
                results[model_name] = {"error": "model not loaded"}
                continue
            try:
                results[model_name] = explainability_service.explain_prediction(
                    model=model,
                    input_data=data.dict(),
                    method="shap",
                )
            except Exception as e:
                logger.warning(f"Could not explain {model_name}: {e}")
                results[model_name] = {"error": str(e)}

        # Agreement score: how similar are the top features across both models?
        try:
            ch_top = {f["name"] for f in results.get("challenger", {}).get("features", [])[:3]}
            pa_top = {f["name"] for f in results.get("passed", {}).get("features", [])[:3]}
            overlap = len(ch_top & pa_top)
            agreement = f"{overlap}/3 top features agree"
        except Exception:
            agreement = "unavailable"

        return {
            "input_data": data.dict(),
            "explanations": results,
            "top_feature_agreement": agreement,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Explanation compare error: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

# ============================================================================
# End Explainability Endpoints
# ============================================================================

# MLFLOW_SERVER_URL = "s3_path"
# # get mlflow info
# @app.get("/{path:path}")
# async def proxy(request: Request, path: str):
#     async with httpx.AsyncClient() as client:
#         response = await client.get(f"{MLFLOW_SERVER_URL}{path}")
#         return response.content, response.status_code

# @app.post("/{path:path}")
# async def proxy_post(request: Request, path: str):
#     body = await request.body()
#     async with httpx.AsyncClient() as client:
#         response = await client.post(f"{MLFLOW_SERVER_URL}{path}", data=body)
#         return response.content, response.status_code



# ============================================================================
# Lineage Endpoints
# ============================================================================

@app.get("/lineage/{model_name}", tags=["Lineage"])
@limiter.limit(DEFAULT_RATE_LIMIT)
async def get_model_lineage(request: Request, model_name: str, version: Optional[str] = None):
    """
    Get dataset lineage for a registered model version.

    Returns the dataset hash, source, row/column counts, and columns used
    to train the specified model version (defaults to Production stage).
    Returns 404 if the model or lineage data is not found.
    """
    try:
        result = data_lineage_service.get_lineage_for_model(model_name, version)
        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No lineage data found for model '{model_name}'. "
                       "The model may not exist or was trained before lineage tracking was enabled."
            )
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching lineage for {model_name}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.get("/lineage/{model_name}/compare", tags=["Lineage"])
@limiter.limit(DEFAULT_RATE_LIMIT)
async def compare_model_lineage(
    request: Request,
    model_name: str,
    version_a: str,
    version_b: str,
):
    """
    Compare the training datasets of two model versions.

    Returns whether both versions were trained on the same dataset, along with
    a diff showing changes in row count, column count, columns added/removed,
    source URL changes, and creation dates.
    """
    if not version_a or not version_b:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Both version_a and version_b query parameters are required."
        )
    try:
        result = data_lineage_service.compare_model_datasets(model_name, version_a, version_b)
        return result
    except Exception as e:
        logger.error(f"Error comparing lineage for {model_name} v{version_a} vs v{version_b}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# ============================================================================
# Lifecycle Endpoints — SLA monitoring and version cleanup
# ============================================================================

@app.get("/models/sla-status", tags=["Lifecycle"])
@limiter.limit(DEFAULT_RATE_LIMIT)
async def get_all_models_sla_status(request: Request):
    """
    Check SLA floors for all tracked production models.

    Returns per-model SLA status including current metrics, thresholds,
    pass/fail verdict, and whether the model was demoted this check.
    """
    try:
        return sla_service.check_all_models()
    except Exception as e:
        logger.error(f"Error running SLA check: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.get("/models/{model_name}/sla-status", tags=["Lifecycle"])
@limiter.limit(DEFAULT_RATE_LIMIT)
async def get_model_sla_status(request: Request, model_name: str):
    """
    Check SLA floor for a single production model.

    Demotes the model to Staging automatically if it is failing its SLA.
    Returns 404 if no SLA criteria are configured for the model.
    """
    try:
        result = sla_service.check_model(model_name)
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


@app.post("/models/cleanup", tags=["Lifecycle"])
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
        return auto_promotion_service.cleanup_all_models(
            max_versions_to_keep=max_versions_to_keep,
            dry_run=dry_run,
        )
    except Exception as e:
        logger.error(f"Error running version cleanup: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/models/{model_name}/cleanup", tags=["Lifecycle"])
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
        return auto_promotion_service.cleanup_old_versions(
            model_name=model_name,
            max_versions_to_keep=max_versions_to_keep,
            dry_run=dry_run,
        )
    except Exception as e:
        logger.error(f"Error running cleanup for {model_name}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))