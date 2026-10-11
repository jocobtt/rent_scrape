"""A/B testing: validation, persistence, traffic assignment, result loading and the statistics."""
import subprocess
import sys
from datetime import datetime, timedelta

import numpy as np
import pytest

from services.ab_testing_service import (
    ABTestConfig, ABTestingService, ExperimentStatus, TrafficSplitStrategy, stable_bucket,
)


@pytest.fixture
def svc(tmp_path):
    return ABTestingService(str(tmp_path / "exp"), str(tmp_path / "res"))


def running(svc, name="exp1", split=0.5, strategy=TrafficSplitStrategy.PERCENTAGE, **kw):
    assert svc.create_experiment(ABTestConfig(name, traffic_split=split, split_strategy=strategy, **kw))
    assert svc.start_experiment(name)
    return name


# ------------------------------------------------------------------ creation and lifecycle

@pytest.mark.parametrize("bad", ["../escape", "a/b", "", "has space", ".hidden", "x" * 65])
def test_experiment_names_cannot_escape_the_directory(svc, bad):
    assert not svc.create_experiment(ABTestConfig(bad))


@pytest.mark.parametrize("kwargs", [{"traffic_split": -0.1}, {"traffic_split": 1.1}, {"min_samples": 5}])
def test_invalid_config_is_rejected(svc, kwargs):
    assert not svc.create_experiment(ABTestConfig("ok", **kwargs))
    assert svc.get_experiment("ok") is None


def test_lifecycle_and_persistence(svc, tmp_path):
    name = running(svc, split=0.3)
    assert svc.get_experiment(name).status == ExperimentStatus.RUNNING
    svc.stop_experiment(name)
    assert svc.get_experiment(name).status == ExperimentStatus.PAUSED
    svc.complete_experiment(name)
    assert svc.get_experiment(name).end_date is not None
    again = ABTestingService(str(tmp_path / "exp"), str(tmp_path / "res"))  # reload from disk
    assert again.get_experiment(name).status == ExperimentStatus.COMPLETED
    assert again.get_experiment(name).traffic_split == 0.3
    assert [e.experiment_name for e in again.list_experiments(ExperimentStatus.COMPLETED)] == [name]
    assert again.list_experiments(ExperimentStatus.RUNNING) == []


def test_unknown_experiment_operations_return_false(svc):
    assert not svc.start_experiment("nope") and not svc.stop_experiment("nope") and not svc.complete_experiment("nope")


# ------------------------------------------------------------------ assignment

def test_only_running_experiments_split_traffic(svc):
    assert svc.assign_model("missing") == "control"
    svc.create_experiment(ABTestConfig("draft", traffic_split=1.0))
    assert svc.assign_model("draft") == "control"  # never started
    name = running(svc, "live", split=1.0)
    assert svc.assign_model(name) == "treatment"
    svc.stop_experiment(name)
    assert svc.assign_model(name) == "control"


def test_expired_experiment_falls_back_to_control(svc):
    name = running(svc, split=1.0, end_date=datetime.now() - timedelta(hours=1))
    svc.get_experiment(name).end_date = datetime.now() - timedelta(hours=1)  # start_experiment keeps end_date
    assert svc.assign_model(name) == "control"


def test_percentage_split_matches_traffic_share(svc):
    np.random.seed(0)
    import random
    random.seed(0)
    name = running(svc, split=0.2)
    share = np.mean([svc.assign_model(name) == "treatment" for _ in range(5000)])
    assert 0.17 < share < 0.23
    for split, expected in ((0.0, "control"), (1.0, "treatment")):
        n = running(svc, f"edge{int(split)}", split=split)
        assert {svc.assign_model(n) for _ in range(200)} == {expected}


def test_force_model_overrides_the_split(svc):
    name = running(svc, split=0.0)
    assert svc.assign_model(name, force_model="treatment") == "treatment"


def test_hash_assignment_is_sticky_per_user_and_matches_split(svc):
    name = running(svc, split=0.3, strategy=TrafficSplitStrategy.RANDOM_HASH)
    users = [f"user{i}" for i in range(3000)]
    first = [svc.assign_model(name, user_id=u) for u in users]
    assert first == [svc.assign_model(name, user_id=u) for u in users]
    assert 0.27 < np.mean([a == "treatment" for a in first]) < 0.33
    assert svc.assign_model(name) == "control"  # no user id: cannot be sticky


def test_hash_bucket_is_stable_across_processes():
    """Python's built-in hash() is salted per process, which would reshuffle users on every restart
    (and between gunicorn workers). The bucket must not depend on PYTHONHASHSEED."""
    code = "from services.ab_testing_service import stable_bucket; print(stable_bucket('user-42'))"
    outs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                           env={"PYTHONHASHSEED": str(seed), "PATH": ""}, cwd=".").stdout.strip()
            for seed in (1, 2, 3)}
    assert outs == {str(stable_bucket("user-42"))}
    assert 0 <= stable_bucket("anything") < 100


# ------------------------------------------------------------------ result logging

def log(svc, name, assigned, control, treatment, actual, returned=None):
    return svc.log_prediction(name, assigned, {"sqr_m": 30}, control, treatment,
                              returned if returned is not None else (treatment if assigned == "treatment" else control),
                              actual_value=actual, latency_ms=10.0 if assigned == "control" else 15.0)


def test_logged_predictions_round_trip(svc):
    name = running(svc)
    assert log(svc, name, "control", 10.0, 11.0, 10.5)
    df = svc.get_experiment_results(name)
    assert len(df) == 1 and df.loc[0, "actual_value"] == 10.5
    assert svc.get_experiment_results("other").empty


def test_experiment_results_do_not_leak_between_prefixed_names(svc):
    a, b = running(svc, "price"), running(svc, "price_v2")
    log(svc, a, "control", 1.0, 1.0, 1.0)
    log(svc, b, "control", 2.0, 2.0, 2.0)
    log(svc, b, "treatment", 2.0, 2.0, 2.0)
    assert len(svc.get_experiment_results(a)) == 1
    assert len(svc.get_experiment_results(b)) == 2


# ------------------------------------------------------------------ analysis

def fill(svc, name, n, control_noise, treatment_noise, seed=0, treat_share=0.5):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        actual = rng.uniform(8, 20)
        c = actual + rng.normal(0, control_noise)
        t = actual + rng.normal(0, treatment_noise)
        log(svc, name, "treatment" if rng.random() < treat_share else "control", c, t, actual)


def test_analysis_needs_data_and_enough_samples(svc):
    name = running(svc, min_samples=50)
    assert svc.analyze_experiment("missing")["error"] == "Experiment not found"
    assert svc.analyze_experiment(name)["error"] == "No data available"
    fill(svc, name, 20, 1.0, 1.0)
    assert svc.analyze_experiment(name)["status"] == "insufficient_data"


def test_clearly_better_treatment_wins_with_high_confidence(svc):
    name = running(svc, min_samples=50)
    fill(svc, name, 600, control_noise=2.0, treatment_noise=0.5)
    res = svc.analyze_experiment(name)
    rec = res["recommendation"]
    assert rec["winner"] == "treatment" and rec["confidence"] == "high"
    assert rec["recommendation_action"] == "promote_treatment_to_production"
    assert res["statistical_tests"]["t_test"]["significant"]
    assert res["statistical_tests"]["mean_errors"]["difference"] > 0
    assert res["treatment_metrics"]["rmse"] < res["control_metrics"]["rmse"]
    assert res["latency_comparison"] == {"control_avg_ms": 10.0, "treatment_avg_ms": 15.0}


def test_clearly_worse_treatment_is_discarded(svc):
    name = running(svc, min_samples=50)
    fill(svc, name, 600, control_noise=0.5, treatment_noise=2.0)
    rec = svc.analyze_experiment(name)["recommendation"]
    assert rec["winner"] == "control" and rec["recommendation_action"] == "keep_control_discard_treatment"


def test_identical_models_do_not_produce_a_winner_for_treatment(svc):
    name = running(svc, min_samples=50)
    fill(svc, name, 400, control_noise=1.0, treatment_noise=1.0, seed=3)
    rec = svc.analyze_experiment(name)["recommendation"]
    assert rec["winner"] != "treatment" or rec["confidence"] != "high"


def test_identical_predictions_do_not_divide_by_zero(svc):
    name = running(svc, min_samples=20)
    for i in range(60):  # both arms exactly right: zero error, zero variance
        log(svc, name, "control" if i % 2 else "treatment", 10.0 + i % 5, 10.0 + i % 5, 10.0 + i % 5)
    res = svc.analyze_experiment(name)
    errors = res["statistical_tests"]["mean_errors"]
    assert np.isfinite(errors["percent_improvement"])
    assert np.isfinite(res["statistical_tests"]["effect_size"]["cohens_d"])


def test_too_few_samples_in_one_arm_skips_the_tests(svc):
    name = running(svc, min_samples=20)
    fill(svc, name, 100, 1.0, 1.0, treat_share=0.05)
    res = svc.analyze_experiment(name)
    assert "error" in res["statistical_tests"] and res["recommendation"]["winner"] is None


def test_winner_direction_follows_the_reported_test(svc):
    """A control arm that got easy rows must not be declared worse than the t-test says."""
    cfg = ABTestConfig("x")
    tests = {"t_test": {"significant": True}, "mean_errors": {"difference": -1.0, "percent_improvement": -50.0}}
    # treatment's MAE over ALL rows looks better, but within its own traffic it was worse
    out = svc._determine_winner({"mae": 2.0}, {"mae": 1.0}, tests, cfg.min_effect_size)
    assert out["winner"] == "control"


def test_effect_size_labels(svc):
    assert [svc._interpret_effect_size(d) for d in (0.1, 0.3, 0.6, 1.2)] == ["negligible", "small", "medium", "large"]
