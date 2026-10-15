"""
Ensembles of the shared LightGBM ("challenger") and linear ("passed") models.

An ensemble here combines the base models' *predictions*, so it needs no feature handling of
its own: serving asks each base model through `PredictionService` (which applies the right
preprocessor) and hands the numbers to `EnsembleModel.combine`.

Training fits the combiner on grouped out-of-fold predictions: fresh clones of the two served
models are fitted on all-but-one fold of buildings and predict the held-out fold, so every
row's prediction comes from a model that never saw its building. (Fitting on the served
models' own held-out rows would need their exact train/test split, which changes whenever the
data cleaning does.) Deposit and key money are present throughout: these are the models the
ensemble routes serve, and those routes require both.

    cd api && uv run python models/ensemble_model.py
"""

import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

# Make `services` importable when run as a script (python models/<name>.py)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent
METHODS = ("weighted_average", "stacking", "voting", "dynamic_weights")
BASE_MODELS = ("challenger", "passed")  # also the order of the prediction matrix columns
# `dynamic_weights` keeps a separate blend per predicted-rent band; a band needs this many rows
MIN_BAND_ROWS = 200
N_BANDS = 3


def ensemble_path(method: str) -> Path:
    return MODELS_DIR / f"ensemble_{method}_model.joblib"


def _blend_weights(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Non-negative weights summing to 1 that minimise squared error of P @ w."""
    k = P.shape[1]
    res = minimize(
        lambda w: float(np.mean((P @ w - y) ** 2)),
        x0=np.full(k, 1.0 / k), bounds=[(0.0, 1.0)] * k,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0}, method="SLSQP",
    )
    return res.x if res.success else np.full(k, 1.0 / k)


class EnsembleModel:
    """Combines base-model predictions with one of `METHODS`."""

    def __init__(self, method: str = "weighted_average", base_names: Optional[List[str]] = None):
        if method not in METHODS:
            raise ValueError(f"Unknown ensemble method: {method}")
        self.method = method
        self.base_names = list(base_names or BASE_MODELS)
        self.weights: Dict[str, float] = {n: 1.0 / len(self.base_names) for n in self.base_names}
        self.meta_model: Optional[Ridge] = None
        self.band_edges: Optional[np.ndarray] = None     # dynamic_weights: predicted-rent cut points
        self.band_weights: Optional[np.ndarray] = None   # one weight row per band
        self.meta: Dict = {}                             # provenance (training rows, base runs, ...)
        self.is_trained = False

    # -- helpers -----------------------------------------------------------
    def _matrix(self, preds: Dict[str, np.ndarray]) -> np.ndarray:
        missing = [n for n in self.base_names if n not in preds]
        if missing:
            raise ValueError(f"Missing base predictions: {missing}")
        return np.column_stack([np.asarray(preds[n], dtype=float) for n in self.base_names])

    # -- training ----------------------------------------------------------
    def fit(self, preds: Dict[str, np.ndarray], y) -> "EnsembleModel":
        """Fit on base predictions for rows the base models did not train on."""
        P, y = self._matrix(preds), np.asarray(y, dtype=float)
        if self.method == "weighted_average":
            self.weights = dict(zip(self.base_names, _blend_weights(P, y)))
        elif self.method == "stacking":
            self.meta_model = Ridge(alpha=1.0).fit(P, y)
        elif self.method == "dynamic_weights":
            self._fit_bands(P, y)
        self.is_trained = True
        return self

    def _fit_bands(self, P: np.ndarray, y: np.ndarray) -> None:
        """A blend per predicted-rent band: a linear model may serve cheap flats better and
        trees the expensive tail, so the weights follow the level of the base predictions."""
        level = P.mean(axis=1)
        self.band_edges = np.quantile(level, np.linspace(0, 1, N_BANDS + 1)[1:-1])
        overall = _blend_weights(P, y)
        band = np.searchsorted(self.band_edges, level)
        rows = []
        for b in range(N_BANDS):
            in_band = band == b
            rows.append(_blend_weights(P[in_band], y[in_band]) if in_band.sum() >= MIN_BAND_ROWS else overall)
        self.band_weights = np.vstack(rows)

    # -- prediction --------------------------------------------------------
    def combine(self, preds: Dict[str, np.ndarray]) -> np.ndarray:
        """Final prediction(s) from the base models' predictions."""
        if not self.is_trained:
            raise ValueError("Ensemble model must be trained before making predictions")
        P = self._matrix(preds)
        if self.method == "weighted_average":
            return P @ np.array([self.weights[n] for n in self.base_names])
        if self.method == "stacking":
            return self.meta_model.predict(P)
        if self.method == "voting":
            return P.mean(axis=1)
        band = np.searchsorted(self.band_edges, P.mean(axis=1))
        return np.einsum("ij,ij->i", P, self.band_weights[band])

    def evaluate(self, preds: Dict[str, np.ndarray], y) -> Dict[str, float]:
        out = self.combine(preds)
        return {
            "rmse": float(np.sqrt(mean_squared_error(y, out))),
            "mae": float(mean_absolute_error(y, out)),
            "r2": float(r2_score(y, out)),
            "mse": float(mean_squared_error(y, out)),
        }

    # -- persistence -------------------------------------------------------
    def save(self, filepath) -> None:
        joblib.dump({
            "method": self.method, "base_names": self.base_names, "weights": self.weights,
            "meta_model": self.meta_model, "band_edges": self.band_edges,
            "band_weights": self.band_weights, "meta": self.meta, "is_trained": self.is_trained,
        }, filepath)

    @classmethod
    def load(cls, filepath) -> "EnsembleModel":
        data = joblib.load(filepath)
        model = cls(method=data["method"], base_names=data["base_names"])
        model.weights = data["weights"]
        model.meta_model = data["meta_model"]
        model.band_edges = data["band_edges"]
        model.band_weights = data["band_weights"]
        model.meta = data.get("meta", {})
        model.is_trained = data["is_trained"]
        return model

    def describe(self) -> Dict:
        """What `/ensemble/methods` reports for this ensemble."""
        if self.method == "stacking" and self.meta_model is not None:
            weights = dict(zip(self.base_names, map(float, self.meta_model.coef_)))
        elif self.method == "dynamic_weights" and self.band_weights is not None:
            weights = {n: [float(w) for w in self.band_weights[:, i]] for i, n in enumerate(self.base_names)}
        else:
            weights = {n: float(w) for n, w in self.weights.items()}
        return {"weights": weights, "method_type": self.method, **self.meta}


# ---------------------------------------------------------------------------
# Training data: grouped out-of-fold predictions of fresh clones of the served models
# ---------------------------------------------------------------------------

def _encoded_frame(df: pd.DataFrame):
    """(X, y, groups, preprocessor) the way challenger_model / reg_model build them (no geo)."""
    import re
    from services.data_split import group_labels
    from services.feature_preprocessor import FeaturePreprocessor

    df = df.rename(columns=lambda c: re.sub("[^A-Za-z0-9_]+", "", c))
    encoded, _ = FeaturePreprocessor().fit_transform(df)
    return encoded.drop("rent_price", axis=1), encoded["rent_price"], group_labels(df)


def out_of_fold_predictions(df: pd.DataFrame, n_splits: int = 5, models_dir: Path = MODELS_DIR):
    """Predictions of the two base models for every row, each from a clone fitted without that
    row's building. Returns ({name: predictions}, y, groups)."""
    from services.feature_preprocessor import mask_optional_terms

    X, y, groups = _encoded_frame(df)
    if groups is None or groups.nunique() < n_splits:
        raise RuntimeError("Need building / address groups to build leakage-free ensemble training data")
    recipes = {name: joblib.load(models_dir / f"{name}-model.joblib") for name in BASE_MODELS}
    oof = {name: np.full(len(X), np.nan) for name in BASE_MODELS}

    for fold, (tr, va) in enumerate(GroupKFold(n_splits=n_splits).split(X, y, groups)):
        for name, recipe in recipes.items():
            # The shared LightGBM is trained with deposit / key money hidden on a share of rows;
            # the linear model sees them throughout (see challenger_model.load_data)
            X_tr = mask_optional_terms(X.iloc[tr]) if name == "challenger" else X.iloc[tr]
            model = clone(recipe).fit(X_tr, y.iloc[tr])
            oof[name][va] = model.predict(X.iloc[va])
        logger.info("Out-of-fold fold %d/%d done", fold + 1, n_splits)
    return oof, y.to_numpy(), groups.to_numpy()


def train_ensemble_models(ensemble_methods: Optional[List[str]] = None, dataset: Optional[str] = None,
                          models_dir: Path = MODELS_DIR):
    """Fit every ensemble method and save it to `models/ensemble_<method>_model.joblib`.

    Each method is scored on buildings its blend weights were not fitted on (a grouped half
    split of the out-of-fold predictions), then refitted on everything for serving.
    Returns {method: {"model", "metrics", "base_metrics", "improvement_vs_best_base"}}.
    """
    import mlflow
    from sklearn.model_selection import GroupShuffleSplit
    from services.data_cleaning import load_training_frame

    methods = list(ensemble_methods or METHODS)
    df = load_training_frame(dataset)
    oof, y, groups = out_of_fold_predictions(df, models_dir=models_dir)

    fit_idx, eval_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42).split(y, y, groups))
    take = lambda preds, idx: {n: p[idx] for n, p in preds.items()}  # noqa: E731
    base_metrics = {
        n: {"rmse": float(np.sqrt(mean_squared_error(y[eval_idx], p[eval_idx]))),
            "r2": float(r2_score(y[eval_idx], p[eval_idx]))}
        for n, p in oof.items()
    }
    best_base = min(m["rmse"] for m in base_metrics.values())

    meta = {"training_rows": int(len(y)), "folds": 5, "base_runs": _base_runs(models_dir)}
    results = {}
    with mlflow.start_run(run_name="ensemble_comparison"):
        mlflow.log_param("ensemble_methods", methods)
        mlflow.log_param("rows", len(y))
        for n, m in base_metrics.items():
            mlflow.log_metric(f"{n}_rmse", m["rmse"])

        for method in methods:
            held_out = EnsembleModel(method).fit(take(oof, fit_idx), y[fit_idx])
            metrics = held_out.evaluate(take(oof, eval_idx), y[eval_idx])
            final = EnsembleModel(method).fit(oof, y)  # served: all rows
            final.meta = {**meta, "eval_rmse": metrics["rmse"], "eval_r2": metrics["r2"]}
            path = Path(models_dir) / f"ensemble_{method}_model.joblib"
            final.save(path)

            with mlflow.start_run(run_name=f"ensemble_{method}", nested=True):
                mlflow.log_param("method", method)
                for k, v in metrics.items():
                    mlflow.log_metric(f"ensemble_{k}", v)
                mlflow.log_metric("improvement_vs_best_base", best_base - metrics["rmse"])
                mlflow.log_artifact(str(path))

            logger.info("%-17s RMSE %.3f  R² %.4f  (best base %.3f)", method, metrics["rmse"], metrics["r2"], best_base)
            results[method] = {
                "model": final, "metrics": metrics, "base_metrics": base_metrics,
                "improvement_vs_best_base": best_base - metrics["rmse"],
            }
    return results


def _base_runs(models_dir: Path) -> Dict[str, Optional[str]]:
    """MLflow run id of each base model's files (from the bundle stamp), for traceability."""
    from services.model_sync import local_run_id
    return {name: local_run_id(models_dir, name) for name in BASE_MODELS}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import mlflow
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001"))
    outcome = train_ensemble_models()
    for name, r in outcome.items():
        print(f"{name:17s} RMSE {r['metrics']['rmse']:.3f}  R² {r['metrics']['r2']:.4f}  "
              f"vs best base {r['improvement_vs_best_base']:+.3f}")
