"""One pass through the whole lifecycle on isolated state: serve -> capture -> label -> monitor -> retrain
check -> promotion check -> SLA, using the committed models and synthetic traffic. Each service has unit
tests; this checks the joins between them, where the bugs found so far were."""
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import model_api
import services.monitoring_reports as mr
from services.auto_retrain_service import AutoRetrainService
from services.data_capture_service import DataCaptureService
from services.monitoring_reports import MonitoringReportsManager
from services.sla_service import ModelSLAService


def listing(rng, shift=0.0, terms=True, address=False):
    body = {
        "sqr_m": float(np.clip(rng.normal(30 + 20 * shift, 6), 12, 120)),
        "maintenence_price": float(rng.uniform(0.2, 1.0)),
        "year_built": float(rng.uniform(1, 35) + 30 * shift),
        "floor": float(rng.integers(1, 10)),
        "eki_walk": float(rng.integers(1, 15)),
    }
    if terms:
        body.update(rei_price=float(rng.choice([0.0, 1.0, 2.0])), shikikin=float(rng.choice([0.0, 1.0, 2.0])))
    if address:
        body["address"] = "東京都渋谷区猿楽町1"
    return body


@pytest.fixture
def stack(client, tmp_path, monkeypatch):
    """Isolated capture / monitoring / A-B state; the real models and prediction service."""
    client.get("/health")  # let the middleware load the committed models
    monkeypatch.setattr(model_api.limiter, "enabled", False)
    capture = DataCaptureService(storage_dir=str(tmp_path / "logs"), buffer_size=1000)
    monkeypatch.setattr(model_api, "data_capture_service", capture)

    rng = np.random.default_rng(0)
    reference = pd.DataFrame([listing(rng) for _ in range(1500)])
    monkeypatch.setattr(mr, "load_reference_data", lambda path=None: reference)
    manager = MonitoringReportsManager(reports_dir=str(tmp_path / "reports"), predictions_dir=str(tmp_path / "logs"))
    manager.data_capture = capture
    monkeypatch.setattr(model_api, "monitoring_manager", manager)

    ab = model_api.ABTestingService(str(tmp_path / "exp"), str(tmp_path / "res"))
    monkeypatch.setattr(model_api, "ab_testing_service", ab)
    return SimpleNamespace(capture=capture, manager=manager, rng=rng, ab=ab)


def predict_many(client, stack, n, route="/predict", **kw):
    ids = []
    for _ in range(n):
        r = client.post(route, json=listing(stack.rng, **kw))
        assert r.status_code == 200, r.text
        ids.append(r.json())
    return ids


def test_serving_to_capture_to_labels(client, stack, admin_headers):
    results = predict_many(client, stack, 30) + predict_many(client, stack, 10, terms=False) \
        + predict_many(client, stack, 10, route="/challenger_predict")
    assert all(r["prediction"] > 0 for r in results)
    df = stack.capture.get_recent_predictions()
    assert len(df) == 50 and set(df["model_type"]) == {"passed", "challenger"}
    assert df["prediction"].notna().all() and df["rei_price"].isna().sum() == 10  # terms-less requests logged as such

    # responses carry the id a label refers to, and the log records which variant answered
    assert all(r["prediction_id"] for r in results) and len({r["prediction_id"] for r in results}) == 50
    assert set(df["variant"]) == {"shared", "no_terms"}

    # labelling through the API must not damage the rest of the log
    labels = [{"prediction_id": r["prediction_id"], "actual_value": r["prediction"] + 0.5} for r in results[:20]]
    r = client.post("/monitoring/ground-truth", headers=admin_headers,
                    json={"labels": labels + [{"prediction_id": "nope", "actual_value": 9.0}]})
    assert r.status_code == 200 and r.json()["updated"] == 20 and r.json()["not_found"] == ["nope"]
    after = stack.capture.get_recent_predictions()
    assert len(after) == 50, "labelling lost part of the day's log"
    assert after["actual_value"].notna().sum() == 20
    assert client.post("/monitoring/ground-truth", json={"labels": labels}).status_code in (401, 403)


def test_drift_is_quiet_on_normal_traffic_and_loud_on_a_shift(client, stack):
    predict_many(client, stack, 260)
    quiet = stack.manager.run_drift_detection(days_back=1, drift_threshold=0.3)
    assert quiet["status"] == "success" and not quiet["drift_detected"], quiet["summary"]

    stack.capture._buffer.clear()
    for f in stack.capture.storage_dir.glob("predictions_*.csv"):
        f.unlink()
    predict_many(client, stack, 260, shift=1.5)
    # two of the seven features were shifted, so 0.3 would (rightly) not call it dataset drift
    loud = stack.manager.run_drift_detection(days_back=1, drift_threshold=0.25)
    assert loud["drift_detected"] and {"sqr_m", "year_built"} <= set(loud["summary"]["drifted_columns"])


def test_traffic_without_terms_does_not_break_monitoring(client, stack):
    predict_many(client, stack, 230, terms=False, address=True)
    out = stack.manager.run_drift_detection(days_back=1, drift_threshold=0.3)
    assert out["status"] == "success"
    reports = stack.manager.generate_all_reports(days_back=1)
    assert reports["data_drift"] and "error" not in reports


class FakeRegistry:
    """Production runs with training MAE 1.0 for both models."""

    def __init__(self):
        self.transitions = []

    def get_latest_versions(self, name, stages):
        return [SimpleNamespace(version="1", run_id=f"run-{name}")] if "Production" in stages else []

    def get_run(self, run_id):
        return SimpleNamespace(data=SimpleNamespace(metrics={"rmse": 3.0, "r2_score": 0.95, "mape": 6.0, "mae": 1.0}))

    def transition_model_version_stage(self, **kw):
        self.transitions.append(kw)


def label_everything(stack, error):
    df = stack.capture.get_recent_predictions()
    for pid, pred in zip(df["prediction_id"], df["prediction"]):
        assert stack.capture.update_actual_value(pid, pred + error)


def retrain_service(stack, registry, monkeypatch):
    import services.auto_retrain_service as ars
    monkeypatch.setattr(ars, "TrainingService", lambda: None)
    monkeypatch.setattr(ars, "AutoPromotionService", lambda: model_api.auto_promotion_service)
    monkeypatch.setattr(model_api.auto_promotion_service, "client", registry)
    monkeypatch.setattr(ars, "MonitoringReportsManager", lambda: stack.manager)
    import services.data_capture_service as dcs
    monkeypatch.setattr(dcs, "DataCaptureService", lambda *a, **k: stack.capture)
    return AutoRetrainService(min_days_between_retrains=1)


def test_retrain_stays_quiet_when_healthy_and_fires_on_degradation(client, stack, monkeypatch):
    predict_many(client, stack, 230)
    registry = FakeRegistry()
    svc = retrain_service(stack, registry, monkeypatch)

    label_everything(stack, error=0.5)                     # typical error 0.5 vs training MAE 1.0
    healthy = svc.check_and_retrain(dry_run=True)
    assert not healthy["retraining_triggered"], healthy["checks_performed"]["performance"]
    assert "error" not in healthy["checks_performed"]["performance"]

    label_everything(stack, error=3.0)                     # 3x the training MAE
    degraded = svc.check_and_retrain(dry_run=True)
    perf = degraded["checks_performed"]["performance"]
    assert perf["performance_degraded"] and degraded["retraining_triggered"] and "error" not in perf


def test_promotion_and_sla_checks_run_against_a_registry(client, stack, admin_headers, monkeypatch):
    registry = FakeRegistry()
    monkeypatch.setattr(model_api.auto_promotion_service, "client", registry)
    r = client.post("/models/check-promotion?dry_run=true", headers=admin_headers)
    assert r.status_code == 200 and r.json()["status"] == "success"

    sla = ModelSLAService.__new__(ModelSLAService)
    sla.client = registry
    out = sla.check_all_models()
    assert out["models_demoted"] == 0 and out["models_passing"] == out["models_checked"]


def test_ab_flow_from_assignment_to_analysis(client, stack, admin_headers):
    r = client.post("/ab-test/create", headers=admin_headers, json={
        "experiment_name": "smoke", "traffic_split": 0.5, "min_samples": 20})
    assert r.status_code == 200, r.text
    assert client.post("/ab-test/smoke/start", headers=admin_headers).status_code == 200
    for _ in range(60):
        resp = client.post("/ab-test/predict", json={"experiment_name": "smoke", "input_data": listing(stack.rng),
                                                     "user_id": f"u{_}"})
        assert resp.status_code == 200, resp.text
    df = stack.ab.get_experiment_results("smoke")
    assert len(df) == 60 and set(df["assigned_model"]) <= {"control", "treatment"}
    assert client.get("/ab-test/smoke/analyze").status_code == 200


def test_invalid_experiment_name_is_a_client_error(client, stack, admin_headers):
    r = client.post("/ab-test/create", headers=admin_headers, json={"experiment_name": "../escape"})
    assert r.status_code in (400, 422), r.text


def test_a_quiet_api_never_looks_like_drift(client, stack):
    """60 requests from the reference distribution: noise, not a reason to retrain."""
    predict_many(client, stack, 60)
    out = stack.manager.run_drift_detection(days_back=1, drift_threshold=0.3)
    assert out["status"] == "insufficient_data" and not out["drift_detected"]
