"""Promotion gates: absolute thresholds, the >= 2% improvement rule, dry runs, forced and rolled-back promotions.

MLflow is replaced by an in-memory registry so the tests need no server.
"""
from types import SimpleNamespace

import pytest

import services.auto_promotion_service as aps
from services.auto_promotion_service import PROMOTION_THRESHOLDS, AutoPromotionService, PromotionCriteria

NAME = "tokyo_rent_lgbm"  # max RMSE 5.0, min R2 0.90, max MAPE 8.0


def good(**over):
    return {"rmse": 3.5, "r2_score": 0.95, "mape": 6.0, "mae": 1.5, **over}


class FakeClient:
    """Just enough of MlflowClient: versions per stage, metrics per run, a log of transitions."""

    def __init__(self):
        self.versions = {}  # (name, stage) -> [SimpleNamespace(version, run_id)]
        self.metrics = {}   # run_id -> {metric: value}
        self.transitions = []

    def add(self, stage, version, metrics, name=NAME):
        run_id = f"run{version}"
        self.versions.setdefault((name, stage), []).append(SimpleNamespace(version=str(version), run_id=run_id))
        self.metrics[run_id] = metrics

    def get_latest_versions(self, name, stages):
        return [v for s in stages for v in self.versions.get((name, s), [])]

    def get_run(self, run_id):
        return SimpleNamespace(data=SimpleNamespace(metrics=self.metrics[run_id]))

    def get_model_version(self, name, version):
        for (n, _), vs in self.versions.items():
            for v in vs:
                if n == name and v.version == str(version):
                    return v
        raise KeyError(version)

    def transition_model_version_stage(self, name, version, stage, archive_existing_versions=False):
        self.transitions.append((name, str(version), stage, archive_existing_versions))


@pytest.fixture
def svc(monkeypatch):
    client = FakeClient()
    monkeypatch.setattr(aps, "MlflowClient", lambda: client)
    s = AutoPromotionService()
    s.fake = client
    return s


# ------------------------------------------------------------------ configuration

def test_every_threshold_entry_becomes_a_criteria_with_the_two_percent_rule(svc):
    assert set(svc.criteria) == set(PROMOTION_THRESHOLDS)
    for c in svc.criteria.values():
        assert c.min_improvement_percent == 2.0 and c.require_better_than_production


def test_threshold_values_are_sane():
    for name, limits in PROMOTION_THRESHOLDS.items():
        assert 0 < limits["max_rmse"] < 20 and 0.5 < limits["min_r2"] < 1 and 0 < limits["max_mape"] < 50, name
    # the dedicated no-terms variants are looser than the shared models
    assert PROMOTION_THRESHOLDS["tokyo_rent_lgbm_noterms"]["max_rmse"] > PROMOTION_THRESHOLDS[NAME]["max_rmse"]


# ------------------------------------------------------------------ absolute gates

@pytest.mark.parametrize("metrics, fragment", [
    (good(rmse=5.01), "RMSE"),
    (good(r2_score=0.89), "R²"),
    (good(mape=8.5), "MAPE"),
])
def test_absolute_threshold_violations_fail(svc, metrics, fragment):
    ok, reason = svc._check_absolute_criteria(metrics, svc.criteria[NAME])
    assert not ok and fragment in reason


@pytest.mark.parametrize("missing", ["rmse", "r2_score", "mape"])
def test_missing_metric_fails_rather_than_passes(svc, missing):
    metrics = {k: v for k, v in good().items() if k != missing}
    ok, reason = svc._check_absolute_criteria(metrics, svc.criteria[NAME])
    assert not ok and "not found" in reason


def test_metrics_exactly_at_the_limits_pass(svc):
    ok, _ = svc._check_absolute_criteria({"rmse": 5.0, "r2_score": 0.90, "mape": 8.0}, svc.criteria[NAME])
    assert ok


def test_mae_limit_applies_only_when_configured(svc):
    crit = PromotionCriteria(NAME, max_mae=1.0)
    assert not svc._check_absolute_criteria(good(mae=1.2), crit)[0]
    assert svc._check_absolute_criteria(good(mae=0.9), crit)[0]


# ------------------------------------------------------------------ comparison with Production

def test_candidate_needs_at_least_two_percent_lower_rmse(svc):
    crit = svc.criteria[NAME]
    prod = good(rmse=4.00)
    assert svc._compare_with_production(good(rmse=3.90), prod, crit)[0]       # 2.5% better
    ok, reason = svc._compare_with_production(good(rmse=3.95), prod, crit)    # 1.25% better
    assert not ok and "Improvement" in reason
    ok, reason = svc._compare_with_production(good(rmse=4.00), prod, crit)    # identical
    assert not ok and "not better" in reason
    assert not svc._compare_with_production(good(rmse=4.20), prod, crit)[0]   # worse


def test_better_rmse_but_worse_r2_is_rejected(svc):
    ok, reason = svc._compare_with_production(good(rmse=3.0, r2_score=0.94), good(rmse=4.0, r2_score=0.95),
                                              svc.criteria[NAME])
    assert not ok and "R²" in reason


# ------------------------------------------------------------------ check_model_for_promotion

def test_unknown_model_and_no_candidate(svc):
    assert not svc.check_model_for_promotion("not_a_model")["promoted"]
    out = svc.check_model_for_promotion(NAME)
    assert not out["promoted"] and "No candidate" in out["reason"]


def test_first_good_candidate_is_promoted_when_there_is_no_production(svc):
    svc.fake.add("None", 1, good())
    out = svc.check_model_for_promotion(NAME)
    assert out["promoted"] and out["version"] == "1"
    assert svc.fake.transitions == [(NAME, "1", "Production", True)]  # archives the old Production


def test_candidate_failing_absolute_gate_is_not_promoted_and_not_compared(svc):
    svc.fake.add("None", 2, good(rmse=6.0))
    svc.fake.add("Production", 1, good(rmse=9.0))
    out = svc.check_model_for_promotion(NAME)
    assert not out["promoted"] and out["reason"].startswith("Failed absolute criteria")
    assert svc.fake.transitions == []


def test_candidate_must_beat_production(svc):
    svc.fake.add("Production", 1, good(rmse=3.59))
    svc.fake.add("None", 2, good(rmse=4.54))
    out = svc.check_model_for_promotion(NAME)
    assert not out["promoted"] and "Not better than production" in out["reason"]
    assert out["production_metrics"]["rmse"] == 3.59 and svc.fake.transitions == []


def test_better_candidate_replaces_production(svc):
    svc.fake.add("Production", 1, good(rmse=4.0))
    svc.fake.add("Staging", 2, good(rmse=3.5))
    assert svc.check_model_for_promotion(NAME)["promoted"]
    assert svc.fake.transitions == [(NAME, "2", "Production", True)]


def test_dry_run_reports_but_does_not_transition(svc):
    svc.fake.add("None", 1, good())
    out = svc.check_model_for_promotion(NAME, dry_run=True)
    assert out["promoted"] and out["dry_run"] and svc.fake.transitions == []


# ------------------------------------------------------------------ manual promotion and rollback

def test_manual_promotion_runs_the_absolute_gate_unless_forced(svc):
    svc.fake.add("None", 3, good(rmse=7.0))
    blocked = svc.promote_specific_version(NAME, "3")
    assert not blocked["promoted"] and blocked["force_required"] and svc.fake.transitions == []
    forced = svc.promote_specific_version(NAME, "3", force=True)
    assert forced["promoted"] and forced["forced"]
    assert svc.fake.transitions == [(NAME, "3", "Production", True)]


def test_manual_promotion_of_a_good_version_does_not_need_force(svc):
    svc.fake.add("None", 4, good())
    out = svc.promote_specific_version(NAME, "4")
    assert out["promoted"] and not out["forced"]


def test_manual_promotion_of_unknown_version_reports_error(svc):
    out = svc.promote_specific_version(NAME, "99")
    assert not out["promoted"] and "error" in out


def test_rollback_restores_the_archived_version(svc):
    assert not svc.rollback_to_previous(NAME)["rolled_back"]
    svc.fake.add("Archived", 1, good())
    out = svc.rollback_to_previous(NAME)
    assert out["rolled_back"] and out["version"] == "1"
    assert svc.fake.transitions == [(NAME, "1", "Production", True)]
