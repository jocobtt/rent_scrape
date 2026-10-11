"""Promotion evidence from several held-out splits instead of one.

On this data the test RMSE of the same model swings by ±1 万円 depending on which buildings land in
the test split (seed 42 once made a better model look 25% worse), so a single split makes the "≥2%
better than Production" rule a coin flip. Training therefore also scores the final configuration on
`SEEDS` grouped splits and logs the mean, spread and per-seed values; `compare` is what the promotion
service uses when both the candidate and Production carry them.
"""
import logging
from typing import Callable, Dict, Iterable, Optional, Tuple

import numpy as np
from sklearn.base import clone

logger = logging.getLogger(__name__)

SEEDS = (0, 1, 2, 3, 4)
MIN_WIN_SHARE = 0.6  # candidate must beat Production on at least this share of the seeds


def _metrics(y_true, y_pred) -> Dict[str, float]:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float).ravel()
    err = y_true - y_pred
    nz = y_true != 0
    return {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "r2": float(1 - (err ** 2).sum() / ((y_true - y_true.mean()) ** 2).sum()),
        "mape": float(np.mean(np.abs(err[nz] / y_true[nz])) * 100),
    }


def _fit(model, X, y, groups, seed: int):
    """Fit with early stopping on a grouped validation slice when the model is a LightGBM one."""
    from services.per_sqm_model import PerSqmRegressor
    inner = model.estimator if isinstance(model, PerSqmRegressor) else model
    if type(inner).__name__ == "LGBMRegressor" and groups is not None:
        import lightgbm as lgb
        from services.data_split import grouped_train_test_split
        Xf, Xv, yf, yv, _ = grouped_train_test_split(X, y, groups, test_size=0.1, random_state=seed)
        return model.fit(Xf, yf, eval_set=[(Xv, yv)], eval_metric="l1",
                         callbacks=[lgb.early_stopping(50, verbose=False)])
    return model.fit(X, y)


def evaluate_seeds(load_split: Callable[[int], Tuple], make_model: Callable[[], object],
                   seeds: Iterable[int] = SEEDS) -> Dict[str, object]:
    """`load_split(seed)` -> (X_train, X_test, y_train, y_test, groups_train); `make_model()` -> a fresh
    unfitted estimator. Returns the per-seed metrics and their aggregate."""
    rows = []
    for s in seeds:
        X_tr, X_te, y_tr, y_te, g_tr = load_split(s)
        model = _fit(make_model(), X_tr, y_tr, g_tr, s)
        rows.append({"seed": s, **_metrics(y_te, model.predict(X_te))})
        logger.info("multiseed %s: rmse=%.3f", s, rows[-1]["rmse"])
    rmse = np.array([r["rmse"] for r in rows])
    return {
        "seeds": rows,
        "rmse_multiseed": float(rmse.mean()), "rmse_multiseed_std": float(rmse.std()),
        "r2_multiseed": float(np.mean([r["r2"] for r in rows])),
        "mape_multiseed": float(np.mean([r["mape"] for r in rows])),
        "multiseed_n": len(rows),
    }


def log_metrics(result: Dict[str, object], log: Callable[[str, float], None]) -> None:
    """Write an `evaluate_seeds` result through `log(key, value)` (mlflow.log_metric, or a run-bound client)."""
    for key in ("rmse_multiseed", "rmse_multiseed_std", "r2_multiseed", "mape_multiseed", "multiseed_n"):
        log(key, float(result[key]))
    for row in result["seeds"]:
        log(f"rmse_seed_{row['seed']}", float(row["rmse"]))


def seed_rmses(metrics: Dict[str, float]) -> Dict[int, float]:
    return {int(k.rsplit("_", 1)[1]): v for k, v in metrics.items() if k.startswith("rmse_seed_")}


def compare(candidate: Dict[str, float], production: Dict[str, float], min_improvement_percent: float,
            min_win_share: float = MIN_WIN_SHARE) -> Optional[Tuple[bool, str]]:
    """(better, reason) from the multi-seed metrics of two runs, or None if either lacks them.

    Better means: mean RMSE lower by at least `min_improvement_percent`, AND lower on at least
    `min_win_share` of the seeds both ran, AND no worse R² on average.
    """
    if "rmse_multiseed" not in candidate or "rmse_multiseed" not in production:
        return None
    c, p = candidate["rmse_multiseed"], production["rmse_multiseed"]
    improvement = (p - c) / p * 100
    c_seeds, p_seeds = seed_rmses(candidate), seed_rmses(production)
    common = sorted(set(c_seeds) & set(p_seeds))
    wins = sum(c_seeds[s] < p_seeds[s] for s in common)
    detail = f"mean RMSE {c:.2f} vs {p:.2f} over {len(common)} seeds ({improvement:+.1f}%), better on {wins}/{len(common)}"
    if improvement < min_improvement_percent:
        return False, f"Multi-seed improvement below {min_improvement_percent}%: {detail}"
    if common and wins / len(common) < min_win_share:
        return False, f"Not consistently better across seeds: {detail}"
    if candidate.get("r2_multiseed", 0) < production.get("r2_multiseed", 0):
        return False, f"Multi-seed R² worse: {candidate.get('r2_multiseed'):.4f} vs {production.get('r2_multiseed'):.4f}"
    return True, f"Multi-seed: {detail}"


def clone_model(model):
    return clone(model)
