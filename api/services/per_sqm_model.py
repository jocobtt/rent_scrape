"""Predict rent per m², serve rent.

Rent scales roughly with size, so a model on the raw rent spends most of its capacity (and its loss)
on large, expensive units. Fitting rent / sqr_m and multiplying back by the size column removes that
and, on held-out buildings, cut LightGBM RMSE by 15-25% (scripts/geo_residual_experiment.py). The
wrapper keeps the rest of the stack unchanged: `predict` returns rent in 万円, so metrics,
promotion, A/B tests and the prediction service need not know the model works in per-m² space.

Samples are weighted by sqr_m ** weight_power (2 by default) so the loss still tracks rent errors.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone


class PerSqmRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, estimator=None, weight_power: float = 2.0, size_col: str = "sqr_m"):
        self.estimator = estimator
        self.weight_power = weight_power
        self.size_col = size_col

    def _size(self, X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            return X[self.size_col].to_numpy(dtype=float)
        return np.asarray(X, dtype=float)[:, list(self.feature_names_in_).index(self.size_col)]

    def _weights(self, size: np.ndarray) -> np.ndarray:
        return size ** self.weight_power if self.weight_power else np.ones_like(size)

    def fit(self, X: pd.DataFrame, y, eval_set=None, **fit_params) -> "PerSqmRegressor":
        """`eval_set` is a list of (X, rent) pairs, converted to per-m² like the training data."""
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        size = self._size(X)
        kwargs = dict(fit_params)
        if eval_set is not None:
            sizes = [self._size(Xv) for Xv, _ in eval_set]
            kwargs["eval_set"] = [(Xv, np.asarray(yv, dtype=float) / s) for (Xv, yv), s in zip(eval_set, sizes)]
            kwargs["eval_sample_weight"] = [self._weights(s) for s in sizes]
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, np.asarray(y, dtype=float) / size, sample_weight=self._weights(size), **kwargs)
        return self

    def predict(self, X) -> np.ndarray:
        return self.estimator_.predict(X) * self._size(X)

    def predict_per_sqm(self, X) -> np.ndarray:
        return self.estimator_.predict(X)

    @property
    def feature_importances_(self) -> np.ndarray:
        return self.estimator_.feature_importances_

    @property
    def best_iteration_(self):
        return getattr(self.estimator_, "best_iteration_", None)

    @property
    def n_features_in_(self) -> int:
        return len(self.feature_names_in_)
