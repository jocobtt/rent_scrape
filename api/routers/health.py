"""Liveness / readiness probes and the landing route."""
import os

import pandas as pd
from fastapi import APIRouter, HTTPException, Request, status

from app_state import S
from config.logging_config import get_logger
from deps import DEFAULT_RATE_LIMIT, limiter

logger = get_logger(__name__)
router = APIRouter()


@router.get("/", tags=["Health"])
@limiter.limit(DEFAULT_RATE_LIMIT)
def home(request: Request):
    return "Welcome to Tokyo Rent Predictor!"


# perform health check
@router.get("/health", tags=["Health"])
@limiter.limit(DEFAULT_RATE_LIMIT)
async def health_check(request: Request):
    """
    Basic health check endpoint - returns 200 if API is responding.
    Use /ready for readiness checks and /startup for startup probes.
    """
    return {"status": "ok", "timestamp": pd.Timestamp.now().isoformat()}


@router.get("/startup")
async def startup_probe():
    """
    Kubernetes startup probe - checks if the application has started.
    Returns 503 if models are not loaded yet.
    """
    try:
        challenger_loaded = hasattr(S.app.state, 'challenger_model') and S.app.state.challenger_model is not None
        passed_loaded = hasattr(S.app.state, 'passed_model') and S.app.state.passed_model is not None

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


@router.get("/ready")
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
        challenger_loaded = hasattr(S.app.state, 'challenger_model') and S.app.state.challenger_model is not None
        passed_loaded = hasattr(S.app.state, 'passed_model') and S.app.state.passed_model is not None

        health_status["checks"]["models"] = {
            "challenger_model": challenger_loaded,
            "passed_model": passed_loaded,
            "status": "healthy" if (challenger_loaded and passed_loaded) else "degraded"
        }

        if not (challenger_loaded and passed_loaded):
            is_ready = False

        # Check 2: Prediction service
        health_status["checks"]["prediction_service"] = {
            "initialized": S.prediction_service is not None,
            "status": "healthy" if S.prediction_service is not None else "unhealthy"
        }

        if S.prediction_service is None:
            is_ready = False
        
        # Check 2b: PyTorch prediction service (optional)
        torch_models_loaded = hasattr(S.app.state, 'torch_models') and len(S.app.state.torch_models) > 0
        health_status["checks"]["torch_prediction_service"] = {
            "initialized": S.torch_prediction_service is not None,
            "models_loaded": torch_models_loaded,
            "model_count": len(S.app.state.torch_models) if torch_models_loaded else 0,
            "available_models": list(S.app.state.torch_models.keys()) if torch_models_loaded else [],
            "status": "healthy" if S.torch_prediction_service is not None else "optional"
        }

        # Check 3: MLflow connectivity (optional - don't fail if unavailable)
        try:
            mlflow_status = S.mlflow_service.get_production_model_metadata("tokyo_rent_lgbm")
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
            data_capture_check = hasattr(S.data_capture_service, '_buffer')
            health_status["checks"]["data_capture"] = {
                "status": "healthy" if data_capture_check else "unhealthy",
                "buffer_size": len(S.data_capture_service._buffer) if data_capture_check else 0
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
