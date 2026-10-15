"""SLA floors: derived from the promotion gates, in the models' own units, demote only on a real breach."""
from types import SimpleNamespace

import pytest
from mlflow.exceptions import MlflowException

import services.sla_service as sla
from services.auto_promotion_service import PROMOTION_THRESHOLDS
from services.sla_service import ModelSLAService, SLACriteria, sla_from_promotion

NAME = "tokyo_rent_lgbm"


class FakeClient:
    def __init__(self):
        self.production = {}   # name -> (version, metrics)
        self.transitions = []
        self.fail_transition = False

    def get_latest_versions(self, name, stages):
        if name in self.production and "Production" in stages:
            return [SimpleNamespace(version=self.production[name][0], run_id=f"run-{name}")]
        return []

    def get_run(self, run_id):
        name = run_id[len("run-"):]
        return SimpleNamespace(data=SimpleNamespace(metrics=self.production[name][1]))

    def transition_model_version_stage(self, name, version, stage, archive_existing_versions):
        if self.fail_transition:
            raise MlflowException("boom")
        self.transitions.append((name, version, stage, archive_existing_versions))


@pytest.fixture
def svc(monkeypatch):
    client = FakeClient()
    monkeypatch.setattr(sla, "MlflowClient", lambda: client)
    s = ModelSLAService()
    s.fake = client
    return s


def test_every_promotion_gated_model_has_a_looser_sla():
    assert set(ModelSLAService.criteria) == set(PROMOTION_THRESHOLDS)
    for name, limits in PROMOTION_THRESHOLDS.items():
        c = ModelSLAService.criteria[name]
        assert c.max_rmse > limits["max_rmse"] and c.min_r2 < limits["min_r2"] and c.max_mape > limits["max_mape"], name


def test_sla_is_in_the_units_rmse_is_logged_in():
    """RMSE is logged in 万円 (a few units); a floor in the thousands could never be breached."""
    for c in ModelSLAService.criteria.values():
        assert c.max_rmse < 50


def test_sla_from_promotion_scaling():
    c = sla_from_promotion({"max_rmse": 4.0, "min_r2": 0.9, "max_mape": 8.0})
    assert (c.max_rmse, c.min_r2, c.max_mape) == (5.0, 0.82, 12.0)


def test_healthy_production_model_is_left_alone(svc):
    svc.fake.production[NAME] = ("3", {"rmse": 3.6, "r2_score": 0.955, "mape": 6.7})
    out = svc.check_model(NAME)
    assert out["sla_passing"] and not out["demoted"] and svc.fake.transitions == []


@pytest.mark.parametrize("metrics, fragment", [
    ({"rmse": 9.0, "r2_score": 0.95, "mape": 6.0}, "RMSE"),
    ({"rmse": 3.0, "r2_score": 0.60, "mape": 6.0}, "R²"),
    ({"rmse": 3.0, "r2_score": 0.95, "mape": 30.0}, "MAPE"),
])
def test_breach_demotes_to_staging_without_archiving_others(svc, metrics, fragment):
    svc.fake.production[NAME] = ("7", metrics)
    out = svc.check_model(NAME)
    assert not out["sla_passing"] and out["demoted"] and out["demoted_to"] == "Staging" and fragment in out["reason"]
    assert svc.fake.transitions == [(NAME, "7", "Staging", False)]


def test_a_model_that_passes_its_promotion_gate_does_not_breach_its_sla(svc):
    """Just inside the promotion gate must never be demoted straight after being promoted."""
    for name, limits in PROMOTION_THRESHOLDS.items():
        svc.fake.production[name] = ("1", {"rmse": limits["max_rmse"], "r2_score": limits["min_r2"],
                                           "mape": limits["max_mape"]})
        assert svc.check_model(name)["sla_passing"], name


def test_failed_demotion_is_reported_not_raised(svc):
    svc.fake.production[NAME] = ("2", {"rmse": 50.0, "r2_score": 0.9, "mape": 5.0})
    svc.fake.fail_transition = True
    out = svc.check_model(NAME)
    assert not out["sla_passing"] and not out["demoted"] and "demotion_error" in out


def test_missing_production_model_or_criteria_is_skipped(svc):
    assert svc.check_model(NAME)["reason"].startswith("No production model")
    assert "No SLA criteria" in svc.check_model("unknown")["reason"]


def test_unlogged_metrics_pass_with_a_note(svc):
    svc.fake.production[NAME] = ("1", {"rmse": 3.0})
    out = svc.check_model(NAME)
    assert out["sla_passing"] and "not logged" in out["reason"]


def test_check_all_models_summarises_and_survives_errors(svc, monkeypatch):
    svc.fake.production[NAME] = ("1", {"rmse": 99.0, "r2_score": 0.9, "mape": 5.0})
    real = svc.check_model
    monkeypatch.setattr(svc, "check_model", lambda n: 1 / 0 if n == "tokyo_rent_tabnet" else real(n))
    out = svc.check_all_models()
    assert out["models_checked"] == len(ModelSLAService.criteria) and out["models_demoted"] == 1
    assert out["results"]["tokyo_rent_tabnet"]["error"]
