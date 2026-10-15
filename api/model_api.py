"""Tokyo Apartment Rent Predictor API: app assembly. Routes live in routers/, shared services in
app_state.py, model loading / registry sync in model_runtime.py, request models in schemas.py."""
import os

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

import model_runtime
from app_state import init_services
from config.logging_config import get_logger, setup_logging
from deps import limiter
from middleware.logging_middleware import RequestBodyLoggingMiddleware, RequestLoggingMiddleware
from routers import ab_testing, data, ensemble, explain, health, lineage, models, monitoring, predictions, torch_routes, training
from schemas import InputData, _require_terms  # noqa: F401  (re-exported: tests and callers import them from here)

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
# Fail closed: with ALLOWED_ORIGINS unset, no cross-origin requests are allowed.
_allow_origins = [o.strip() for o in _raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins = _allow_origins,
    allow_credentials = True,
    allow_methods = ["GET", "POST"],
    allow_headers = ["*"],
    expose_headers = ["X-Dataset-Created"],  # the dashboard shows how old the scrape is
)

# The listings CSV is a few MB of repetitive text; everything else is tiny and skipped
app.add_middleware(GZipMiddleware, minimum_size=1024)

# Initialize services (S.<service>; see app_state.py)
init_services(app)

# Loads the models on first request and picks up registry promotions (see model_runtime.py)
app.middleware("http")(model_runtime.load_model_middleware)

for _router in (health, data, predictions, training, models, ensemble, monitoring, ab_testing, torch_routes, explain, lineage):
    app.include_router(_router.router)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
