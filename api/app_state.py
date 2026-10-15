"""Process-wide service instances, shared by the routers and the model runtime.

Routers read `S.<service>` at call time (not at import), so tests and the model reloader can swap a
service by assigning to it: `monkeypatch.setattr(S, "prediction_service", fake)`.
"""
from services import MLflowService, ModelLoader, TrainingService
from services.ab_testing_service import ABTestingService
from services.auto_promotion_service import AutoPromotionService
from services.auto_retrain_service import AutoRetrainService
from services.data_capture_service import DataCaptureService
from services.data_lineage_service import DataLineageService
from services.explainability_service import ExplainabilityService
from services.sla_service import ModelSLAService


class Services:
    app = None                          # the FastAPI app (S.app.state holds the loaded models)
    model_loader = None
    prediction_service = None           # initialised once the models are loaded
    ensemble_prediction_service = None  # initialised once the models are loaded
    torch_prediction_service = None     # initialised once the PyTorch models are loaded
    training_service = None
    mlflow_service = None
    data_capture_service = None
    monitoring_manager = None           # initialised lazily (routers/monitoring.py)
    ab_testing_service = None
    ab_mlflow_integration = None        # initialised lazily (routers/ab_testing.py)
    auto_promotion_service = None
    auto_retrain_service = None
    explainability_service = None
    data_lineage_service = None
    sla_service = None


S = Services()


def init_services(app) -> Services:
    """Create the service instances (called once by model_api.py)."""
    S.app = app
    S.model_loader = ModelLoader()
    S.training_service = TrainingService()
    S.mlflow_service = MLflowService()
    S.data_capture_service = DataCaptureService()
    S.ab_testing_service = ABTestingService()
    S.auto_promotion_service = AutoPromotionService()
    S.auto_retrain_service = AutoRetrainService()
    S.explainability_service = ExplainabilityService()
    S.data_lineage_service = DataLineageService()
    S.sla_service = ModelSLAService()
    return S
