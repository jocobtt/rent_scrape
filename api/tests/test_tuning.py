"""The tuning helpers, on small synthetic data (trial counts kept tiny)."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    n = 400
    X = pd.DataFrame({"sqr_m": rng.uniform(15, 80, n), "floor": rng.integers(1, 15, n).astype(float),
                      "eki_walk": rng.integers(1, 15, n).astype(float)})
    y = pd.Series(X.sqr_m * 0.3 - X.eki_walk * 0.1 + rng.normal(0, 0.5, n) + (X.sqr_m > 70) * 20, name="rent_price")
    groups = pd.Series(np.repeat(np.arange(n // 4), 4), index=X.index)
    return X, y, groups


def test_tune_linear_returns_fitted_pipeline(data):
    from models.reg_model import tune_linear
    X, y, groups = data
    model, name, cv = tune_linear(X, y, groups, n_trials=3)
    assert name.startswith("Tuned_") and cv > 0
    assert model.predict(X).shape == (len(X),)


def test_tune_lgbm_sets_iterations_and_keeps_base_params(data):
    from models.challenger_model import tune_params
    X, y, groups = data
    base = {"objective": "regression", "verbose": -1, "num_iterations": 600}
    tuned, best = tune_params(X, y, groups, base, n_trials=2, n_folds=2)
    assert tuned["objective"] == "regression" and tuned["num_iterations"] != 600
    assert {"learning_rate", "num_leaves", "lambda_l2"} <= set(tuned)


def test_sample_context_stratified_reaches_further_into_the_tail(data):
    from models.tabular_pretrained_model import sample_context
    X, y, _ = data
    _, y_rand = sample_context(X, y, 100, "random")
    _, y_strat = sample_context(X, y, 100, "stratified")
    assert len(y_rand) == len(y_strat) == 100
    assert (y_strat > y.quantile(0.9)).mean() > (y_rand > y.quantile(0.9)).mean()


def test_sample_context_returns_all_rows_when_small(data):
    from models.tabular_pretrained_model import sample_context
    X, y, _ = data
    Xs, ys = sample_context(X, y, 10_000, "stratified")
    assert len(Xs) == len(X)


def test_tune_tabnet_smoke(data):
    pytest.importorskip("pytorch_tabnet")
    import torch
    from models.tabular_pretrained_model import tune_tabnet
    X, y, _ = data
    base = {"n_independent": 1, "n_shared": 1, "optimizer_fn": torch.optim.Adam, "seed": 0}
    tuned, batch, val = tune_tabnet(base, X.iloc[:300], y.iloc[:300], X.iloc[300:], y.iloc[300:],
                                    n_trials=1, trial_epochs=2)
    assert tuned["n_d"] == tuned["n_a"] and batch in (256, 512, 1024) and val > 0


def test_torch_best_params_include_hidden_dims(data, monkeypatch):
    """Optuna returns hidden_1..4; the final model must receive them as `hidden_dims`."""
    from models.torch_model import TorchRentPredictor
    X, y, groups = data
    p = TorchRentPredictor(model_type="tabular", device="cpu")
    p.groups_ = groups
    monkeypatch.setattr(p, "train_model", lambda *a, **k: {"val_loss": [1.0]})
    monkeypatch.setenv("TORCH_TRIAL_EPOCHS", "1")
    best = p.hyperparameter_optimization(X, y, n_trials=1)
    assert best["hidden_dims"] == [best[f"hidden_{i}"] for i in range(1, 5)]
