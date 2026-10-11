"""The ensemble / torch prediction routes the frontend's Model and Compare pages call, with fake services
(no ensemble or torch models are trained in the repo, so these paths were never exercised)."""
import pytest

from app_state import S

BODY = {"sqr_m": 30.0, "rei_price": 1.0, "shikikin": 1.0, "maintenence_price": 0.5,
        "year_built": 10.0, "floor": 3.0, "eki_walk": 5.0}


class FakeEnsemble:
    def predict_with_ensemble(self, data, method):
        if method == "bogus":
            raise ValueError("Ensemble method 'bogus' not available")
        return {"prediction": 12.5, "model_type": f"ensemble_{method}", "status": "success"}


class FakeTorch:
    def predict_with_torch_model(self, data, model_name):
        return {"prediction": 11.0, "model_name": model_name, "status": "success", "processing_time_ms": 3}


@pytest.fixture
def services(client, monkeypatch, tmp_path):
    from deps import limiter
    client.get("/health")  # the first request makes the middleware build the real services; patch after that
    from services.data_capture_service import DataCaptureService
    monkeypatch.setattr(limiter, "enabled", False)
    capture = DataCaptureService(storage_dir=str(tmp_path))
    monkeypatch.setattr(S, "data_capture_service", capture)
    monkeypatch.setattr(S, "ensemble_prediction_service", FakeEnsemble())
    monkeypatch.setattr(S, "torch_prediction_service", FakeTorch())
    return capture


def test_ensemble_predict_returns_what_the_model_page_reads(client, services):
    r = client.post("/ensemble/predict", json={"input_data": BODY, "method": "weighted_average"})
    assert r.status_code == 200, r.text
    assert r.json()["prediction"] == 12.5 and r.json()["model_type"] == "ensemble_weighted_average"
    assert list(services.get_recent_predictions()["model_type"]) == ["ensemble_weighted_average"]


def test_unknown_ensemble_method_is_rejected_not_a_crash(client, services):
    assert client.post("/ensemble/predict", json={"input_data": BODY, "method": "bogus"}).status_code in (400, 422)


def test_ensemble_and_torch_need_the_terms(client, services):
    partial = {k: v for k, v in BODY.items() if k not in ("rei_price", "shikikin")}
    assert client.post("/ensemble/predict", json={"input_data": partial, "method": "stacking"}).status_code == 422
    assert client.post("/torch/predict", json={"input_data": partial, "model_name": "tokyo_rent_tabnet"}).status_code == 422


def test_torch_predict_returns_what_the_model_page_reads(client, services):
    r = client.post("/torch/predict", json={"input_data": BODY, "model_name": "tokyo_rent_tabnet"})
    assert r.status_code == 200, r.text
    assert r.json()["prediction"] == 11.0 and r.json()["model_name"] == "tokyo_rent_tabnet" and r.json()["request_id"]
    assert list(services.get_recent_predictions()["model_type"]) == ["torch_tokyo_rent_tabnet"]


def test_torch_predict_without_models_is_a_clean_503(client, services, monkeypatch):
    monkeypatch.setattr(S, "torch_prediction_service", None)
    r = client.post("/torch/predict", json={"input_data": BODY, "model_name": "tokyo_rent_tabnet"})
    assert r.status_code in (500, 503) and "not available" in r.json()["detail"]


def test_extra_fields_from_older_frontends_are_ignored(client, services, monkeypatch):
    """The frontend used to send `contract_type`; it is no longer a model input."""
    r = client.post("/predict", json={**BODY, "contract_type": 0})
    assert r.status_code == 200 and "contract_type" not in r.json()


@pytest.mark.parametrize("path", ["/ensemble/confidence", "/ensemble/recommendations", "/ensemble/compare",
                                  "/torch/compare-all", "/torch/recommendations"])
def test_routes_that_need_the_terms_say_422_when_they_are_missing(client, services, monkeypatch, path):
    from unittest.mock import MagicMock
    monkeypatch.setattr(S, "ensemble_prediction_service", MagicMock())
    monkeypatch.setattr(S, "torch_prediction_service", MagicMock())
    partial = {k: v for k, v in BODY.items() if k not in ("rei_price", "shikikin")}
    r = client.post(path, json=partial)
    assert r.status_code == 422 and "required for this endpoint" in r.json()["detail"], (path, r.status_code, r.text[:200])


def test_feature_importance_is_reported_as_shares_that_sum_to_one():
    from services.mlflow_service import MLflowService
    out = MLflowService._normalise_importance([{"name": "a", "importance": 300.0}, {"name": "b", "importance": 100.0}])
    assert [i["importance"] for i in out] == [0.75, 0.25] and out[0]["name"] == "a"
    assert MLflowService._normalise_importance([{"name": "a", "importance": 0.0}]) == [{"name": "a", "importance": 0.0}]
