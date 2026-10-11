"""Auto-retrain: drift / performance triggers, cooldown, the retrain lock, and which scripts run."""
from datetime import datetime, timedelta
from types import SimpleNamespace

import pandas as pd
import pytest

import services.auto_retrain_service as ars
import services.data_capture_service as dcs
from services.auto_retrain_service import RETRAIN_JOBS, AutoRetrainService


class FakeMonitoring:
    result = {"drift_detected": False, "summary": {"drift_share": 0.0}}

    def run_drift_detection(self, days_back, drift_threshold):
        return self.result


class FakePromotion:
    def __init__(self):
        self.calls = 0
        self.mae = {"tokyo_passed_rent_model": 2.0, "tokyo_rent_lgbm": 1.5}
        self.client = SimpleNamespace(get_latest_versions=self._versions)

    def _versions(self, name, stages):
        return [SimpleNamespace(run_id=name)] if name in self.mae else []

    def _get_model_metrics(self, run_id):
        return {"mae": self.mae[run_id]}

    def check_and_promote_new_models(self, dry_run=False):
        self.calls += 1
        return {"models_promoted": ["tokyo_rent_lgbm"]}


@pytest.fixture
def svc(monkeypatch):
    monkeypatch.setattr(ars, "MonitoringReportsManager", FakeMonitoring)
    monkeypatch.setattr(ars, "AutoPromotionService", FakePromotion)
    monkeypatch.setattr(ars, "TrainingService", lambda: None)
    FakeMonitoring.result = {"drift_detected": False, "summary": {"drift_share": 0.0}}
    return AutoRetrainService(min_days_between_retrains=1)


@pytest.fixture
def ran(monkeypatch):
    """Record the scripts run instead of launching training."""
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(argv[1:])
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(ars.subprocess, "run", fake_run)
    return calls


def capture(monkeypatch, rows):
    frame = pd.DataFrame(rows, columns=["timestamp", "model_type", "prediction", "actual_value"])
    monkeypatch.setattr(dcs, "DataCaptureService",
                        lambda *a, **k: SimpleNamespace(get_predictions_with_actuals=lambda: frame.copy()))


def recent(model_type, error, n=20, hours_ago=1):
    ts = datetime.now() - timedelta(hours=hours_ago)
    return [(ts, model_type, 10.0, 10.0 + error) for _ in range(n)]


# ------------------------------------------------------------------ drift trigger

def test_drift_needs_both_the_flag_and_enough_share(svc):
    FakeMonitoring.result = {"drift_detected": True, "summary": {"drift_share": 0.1}}
    assert not svc._check_drift()["requires_retraining"]
    FakeMonitoring.result = {"drift_detected": True, "summary": {"drift_share": 0.5}}
    assert svc._check_drift()["requires_retraining"]
    FakeMonitoring.result = {"drift_detected": False, "summary": {"drift_share": 0.9}}
    assert not svc._check_drift()["requires_retraining"]


def test_drift_check_error_does_not_trigger_retraining(svc, monkeypatch):
    monkeypatch.setattr(svc.monitoring_manager, "run_drift_detection", lambda **k: 1 / 0)
    out = svc._check_drift()
    assert not out["requires_retraining"] and "error" in out


# ------------------------------------------------------------------ performance trigger

def test_performance_is_compared_with_the_models_training_mae(svc, monkeypatch):
    capture(monkeypatch, recent("challenger", 3.0))      # baseline 1.5 -> 100% worse
    out = svc._check_performance_degradation()
    assert "error" not in out, out
    assert out["performance_degraded"] and out["requires_retraining"]
    assert out["models"]["challenger"]["degradation"] == pytest.approx(1.0)


def test_healthy_models_do_not_trigger(svc, monkeypatch):
    capture(monkeypatch, recent("challenger", 1.6) + recent("passed", 2.1))  # within 15% of 1.5 / 2.0
    out = svc._check_performance_degradation()
    assert not out["performance_degraded"] and set(out["models"]) == {"challenger", "passed"}


def test_degradation_threshold_is_relative_to_each_models_baseline(svc, monkeypatch):
    # an error of 2.2 is 10% over the passed baseline (2.0, fine) but 47% over the challenger's (1.5, degraded)
    capture(monkeypatch, recent("passed", 2.2))
    assert not svc._check_performance_degradation()["performance_degraded"]
    capture(monkeypatch, recent("challenger", 2.2))
    assert svc._check_performance_degradation()["performance_degraded"]


def test_models_without_a_production_baseline_are_not_judged(svc, monkeypatch):
    svc.auto_promotion.mae.clear()
    capture(monkeypatch, recent("challenger", 50.0))
    out = svc._check_performance_degradation()
    assert not out["performance_degraded"] and out["models"] == {}


def test_old_or_missing_actuals_are_ignored(svc, monkeypatch):
    capture(monkeypatch, [])
    assert "No predictions" in svc._check_performance_degradation()["message"]
    capture(monkeypatch, recent("challenger", 9.0, hours_ago=24 * 30))
    assert "No recent" in svc._check_performance_degradation()["message"]


# ------------------------------------------------------------------ check_and_retrain

def test_healthy_system_does_nothing(svc, ran, monkeypatch):
    capture(monkeypatch, [])
    out = svc.check_and_retrain()
    assert not out["retraining_triggered"] and ran == []


def test_dry_run_reports_but_does_not_train(svc, ran, monkeypatch):
    capture(monkeypatch, [])
    FakeMonitoring.result = {"drift_detected": True, "summary": {"drift_share": 0.8}}
    out = svc.check_and_retrain(dry_run=True)
    assert out["retraining_triggered"] and "Dry run" in out["message"] and ran == []


def test_drift_retrains_every_variant_then_promotes(svc, ran, monkeypatch):
    capture(monkeypatch, [])
    FakeMonitoring.result = {"drift_detected": True, "summary": {"drift_share": 0.8}}
    out = svc.check_and_retrain(dry_run=False)
    assert out["models_retrained"] == ["challenger", "challenger-noterms", "passed", "passed-noterms"]
    assert ["models/challenger_model.py", "--no-terms"] in ran and ["models/reg_model.py", "--no-terms"] in ran
    assert out["models_promoted"] == ["tokyo_rent_lgbm"] and svc.auto_promotion.calls == 1


def test_cooldown_blocks_a_second_retrain(svc, ran, monkeypatch):
    capture(monkeypatch, [])
    FakeMonitoring.result = {"drift_detected": True, "summary": {"drift_share": 0.8}}
    svc.check_and_retrain(dry_run=False)
    ran.clear()
    again = svc.check_and_retrain(dry_run=False)
    assert ran == [] and not again["retraining_triggered"] and "skipped" in again["message"]
    svc.last_retrain = {m: datetime.now() - timedelta(days=2) for m in RETRAIN_JOBS}
    assert svc.check_and_retrain(dry_run=False)["retraining_triggered"]


def test_nothing_is_promoted_when_every_retrain_failed(svc, monkeypatch):
    capture(monkeypatch, [])
    FakeMonitoring.result = {"drift_detected": True, "summary": {"drift_share": 0.8}}
    monkeypatch.setattr(ars.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=1, stderr="boom"))
    out = svc.check_and_retrain(dry_run=False)
    assert out["models_retrained"] == [] and out["errors"] and svc.auto_promotion.calls == 0
    assert svc.last_retrain == {}  # a failed run must not start the cooldown


def test_one_failing_script_does_not_stop_the_rest(svc, monkeypatch):
    def run(argv, **kw):
        ok = "--no-terms" not in argv
        return SimpleNamespace(returncode=0 if ok else 1, stderr="" if ok else "boom")

    monkeypatch.setattr(ars.subprocess, "run", run)
    out = svc._retrain_models()
    assert out["models_retrained"] == ["challenger", "passed"] and len(out["errors"]) == 2


def test_timeouts_are_reported_as_errors(svc, monkeypatch):
    def run(argv, **kw):
        raise ars.subprocess.TimeoutExpired(argv, 1)

    monkeypatch.setattr(ars.subprocess, "run", run)
    out = svc._retrain_models(["passed"])
    assert out["models_retrained"] == [] and len(out["errors"]) == 2


# ------------------------------------------------------------------ lock and force

def test_concurrent_retrain_is_skipped(svc, ran):
    assert AutoRetrainService._retrain_lock.acquire(blocking=False)
    try:
        out = svc._retrain_models()
    finally:
        AutoRetrainService._retrain_lock.release()
    assert out["status"] == "skipped" and ran == []
    assert AutoRetrainService._retrain_lock.acquire(blocking=False)  # released after a normal run too
    AutoRetrainService._retrain_lock.release()


def test_lock_is_released_after_a_crash(svc, monkeypatch):
    monkeypatch.setattr(svc, "_run_retraining", lambda families=None: 1 / 0)
    with pytest.raises(ZeroDivisionError):
        svc._retrain_models()
    assert AutoRetrainService._retrain_lock.acquire(blocking=False)
    AutoRetrainService._retrain_lock.release()


def test_force_retrain_honours_the_model_name(svc, ran):
    out = svc.force_retrain("passed")
    assert out["models_retrained"] == ["passed", "passed-noterms"]
    assert all("challenger" not in a[0] for a in ran)
    assert out["promotion_result"]["models_promoted"]
    bad = svc.force_retrain("nonsense")
    assert bad["models_retrained"] == [] and bad["errors"]
    assert svc.force_retrain()["models_retrained"] == ["challenger", "challenger-noterms", "passed", "passed-noterms"]


def test_cooldown_check_per_model(svc):
    assert svc.check_if_retrain_allowed("passed")
    svc.last_retrain["passed"] = datetime.now()
    assert not svc.check_if_retrain_allowed("passed") and svc.check_if_retrain_allowed("challenger")
