import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
import os

os.environ.setdefault("ADMIN_API_KEY", "test-admin-key")

import model_api


@pytest.fixture(scope="session")
def client():
    with TestClient(model_api.app) as c:
        yield c


@pytest.fixture
def admin_headers():
    return {"X-API-Key": os.environ["ADMIN_API_KEY"]}


@pytest.fixture
def valid_input():
    return {
        "sqr_m": 30.0,
        "rei_price": 2.0,
        "shikikin": 2.0,
        "maintenence_price": 1.0,
        "year_built": 10.0,
        "floor": 3.0,
        "eki_walk": 8.0,
    }


@pytest.fixture
def mock_prediction_svc(monkeypatch):
    svc = MagicMock()
    svc.predict_with_passed_model.return_value = {"prediction": 12.5, "model": "passed"}
    svc.predict_with_challenger_model.return_value = {"prediction": 13.0, "model": "challenger"}
    svc.compare_predictions.return_value = {
        "passed_prediction": 12.5,
        "challenger_prediction": 13.0,
        "difference": 0.5,
    }
    monkeypatch.setattr(model_api, "prediction_service", svc)
    return svc


@pytest.fixture
def no_models(monkeypatch):
    """Force all model-dependent services to None so 503 paths are reachable
    regardless of whether .joblib files are present on the local machine."""
    # Null out module-level service globals
    monkeypatch.setattr(model_api, "prediction_service", None)
    monkeypatch.setattr(model_api, "ensemble_prediction_service", None)

    # Null out app.state so the middleware won't re-initialize the services
    prev_challenger = getattr(model_api.app.state, "challenger_model", _MISSING)
    prev_passed = getattr(model_api.app.state, "passed_model", _MISSING)
    model_api.app.state.challenger_model = None
    model_api.app.state.passed_model = None

    yield

    if prev_challenger is _MISSING:
        try:
            del model_api.app.state.challenger_model
        except AttributeError:
            pass
    else:
        model_api.app.state.challenger_model = prev_challenger

    if prev_passed is _MISSING:
        try:
            del model_api.app.state.passed_model
        except AttributeError:
            pass
    else:
        model_api.app.state.passed_model = prev_passed


# Sentinel used by no_models fixture
_MISSING = object()
