"""Multi-seed promotion evidence: evaluation, logging, comparison, and its use by the promotion service."""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from services import multiseed_gate as ms
import services.auto_promotion_service as aps
from services.auto_promotion_service import AutoPromotionService


def metrics(mean, seeds=None, r2=0.95):
    seeds = seeds or [mean] * 5
    out = {"rmse": mean, "r2_score": r2, "mape": 6.0, "rmse_multiseed": float(np.mean(seeds)), "r2_multiseed": r2}
    out.update({f"rmse_seed_{i}": v for i, v in enumerate(seeds)})
    return out


# ------------------------------------------------------------------ compare

def test_without_multiseed_on_either_side_the_caller_falls_back():
    assert ms.compare({"rmse": 3.0}, metrics(4.0), 2.0) is None
    assert ms.compare(metrics(3.0), {"rmse": 4.0}, 2.0) is None


def test_clear_consistent_improvement_passes():
    ok, reason = ms.compare(metrics(3.0, [2.9, 3.1, 3.0, 3.0, 3.0]), metrics(3.5, [3.4, 3.6, 3.5, 3.5, 3.5]), 2.0)
    assert ok and "5/5" in reason


def test_small_mean_gain_fails_the_two_percent_rule():
    ok, reason = ms.compare(metrics(3.45), metrics(3.5), 2.0)
    assert not ok and "below 2" in reason


def test_a_mean_win_driven_by_one_seed_is_rejected():
    cand = [2.0, 3.6, 3.6, 3.6, 3.6]   # mean 3.28: one big win, four losses
    prod = [3.5, 3.5, 3.5, 3.5, 3.5]   # mean 3.5 -> 6% better on average
    ok, reason = ms.compare(metrics(0, cand), metrics(0, prod), 2.0)
    assert not ok and "consistently" in reason


def test_worse_r2_is_rejected_even_if_rmse_is_better():
    ok, reason = ms.compare(metrics(3.0, r2=0.90), metrics(3.5, r2=0.95), 2.0)
    assert not ok and "R²" in reason


def test_seed_win_share_only_counts_seeds_both_ran():
    cand, prod = metrics(3.0), metrics(3.5)
    for k in [k for k in prod if k.startswith("rmse_seed_") and k.endswith(("3", "4"))]:
        del prod[k]
    assert ms.compare(cand, prod, 2.0)[0]


# ------------------------------------------------------------------ evaluate_seeds / log

def test_evaluate_seeds_aggregates_and_logs():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=300), "b": rng.normal(size=300)})
    y = 3 * X["a"] - X["b"] + rng.normal(0, 0.1, 300)

    def split(seed):
        idx = np.random.default_rng(seed).permutation(300)
        tr, te = idx[:240], idx[240:]
        return X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te], None

    result = ms.evaluate_seeds(split, lambda: Ridge(alpha=0.1), seeds=(0, 1, 2))
    assert result["multiseed_n"] == 3 and result["rmse_multiseed"] == pytest.approx(
        np.mean([r["rmse"] for r in result["seeds"]]))
    assert result["r2_multiseed"] > 0.98
    logged = {}
    ms.log_metrics(result, lambda k, v: logged.__setitem__(k, v))
    assert {"rmse_multiseed", "rmse_multiseed_std", "multiseed_n", "rmse_seed_0", "rmse_seed_2"} <= set(logged)
    assert ms.seed_rmses(logged) == {s["seed"]: s["rmse"] for s in result["seeds"]}


def test_lightgbm_models_get_early_stopping_with_a_grouped_validation_slice():
    import lightgbm as lgb
    from services.per_sqm_model import PerSqmRegressor
    rng = np.random.default_rng(1)
    n = 600
    X = pd.DataFrame({"sqr_m": rng.uniform(15, 60, n), "age": rng.uniform(0, 40, n)})
    y = X["sqr_m"] * (3 - 0.02 * X["age"]) + rng.normal(0, 0.3, n)
    groups = pd.Series(np.arange(n) // 3)

    def split(seed):
        return X.iloc[:480], X.iloc[480:], y.iloc[:480], y.iloc[480:], groups.iloc[:480]

    result = ms.evaluate_seeds(split, lambda: PerSqmRegressor(lgb.LGBMRegressor(n_estimators=2000, verbose=-1)),
                               seeds=(0,))
    assert result["rmse_multiseed"] < 3.0


# ------------------------------------------------------------------ promotion service

class FakeClient:
    def __init__(self):
        self.versions, self.metrics, self.transitions = {}, {}, []

    def add(self, stage, version, m, name="tokyo_rent_lgbm"):
        self.versions.setdefault((name, stage), []).append(type("V", (), {"version": str(version), "run_id": f"r{version}"})())
        self.metrics[f"r{version}"] = m

    def get_latest_versions(self, name, stages):
        return [v for s in stages for v in self.versions.get((name, s), [])]

    def get_run(self, run_id):
        return type("R", (), {"data": type("D", (), {"metrics": self.metrics[run_id]})()})()

    def transition_model_version_stage(self, name, version, stage, archive_existing_versions=False):
        self.transitions.append((version, stage))


@pytest.fixture
def svc(monkeypatch):
    client = FakeClient()
    monkeypatch.setattr(aps, "MlflowClient", lambda: client)
    s = AutoPromotionService()
    s.fake = client
    return s


def test_noisy_single_split_no_longer_blocks_a_better_model(svc):
    """The seed-42 situation: candidate loses on the single split but wins consistently over seeds."""
    svc.fake.add("Production", 1, {**metrics(3.8, [3.9, 3.8, 3.7, 3.9, 3.7]), "rmse": 3.59})
    svc.fake.add("None", 2, {**metrics(3.4, [3.3, 3.4, 3.5, 3.4, 3.4]), "rmse": 4.54})
    out = svc.check_model_for_promotion("tokyo_rent_lgbm")
    assert out["promoted"], out  # 4.54 still clears the absolute RMSE <= 5.0 gate
    assert svc.fake.transitions == [("2", "Production")]


def test_single_split_luck_no_longer_promotes_a_worse_model(svc):
    svc.fake.add("Production", 1, {**metrics(3.4, [3.3, 3.4, 3.5, 3.4, 3.4]), "rmse": 4.5})
    svc.fake.add("None", 2, {**metrics(3.8, [3.9, 3.8, 3.7, 3.9, 3.7]), "rmse": 3.2})
    out = svc.check_model_for_promotion("tokyo_rent_lgbm")
    assert not out["promoted"] and "Multi-seed" in out["reason"] and svc.fake.transitions == []


def test_runs_without_multiseed_use_the_single_split_rule(svc):
    svc.fake.add("Production", 1, {"rmse": 4.0, "r2_score": 0.95, "mape": 6.0})
    svc.fake.add("None", 2, {"rmse": 3.5, "r2_score": 0.95, "mape": 6.0, **metrics(3.5)})
    assert svc.check_model_for_promotion("tokyo_rent_lgbm")["promoted"]
