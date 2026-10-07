"""
Model Explainability Service

Provides SHAP and LIME explanations for all model types in the pipeline:
- SHAP TreeExplainer  → LightGBM (fast, exact)
- SHAP LinearExplainer → Linear / Ridge / Lasso (fast, exact)
- SHAP KernelExplainer → any model (slower, model-agnostic fallback)
- LIME                 → model-agnostic local explanations (alternative to SHAP)
- TabNet attention     → built-in interpretability from pytorch-tabnet

Usage from API:
    service = ExplainabilityService()
    result  = service.explain_prediction(model, input_data, background_data, method="shap")
    summary = service.global_feature_importance(model, background_data)
"""

import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# Feature names used across all models
FEATURE_NAMES = [
    "sqr_m",
    "rei_price",
    "shikikin",
    "maintenence_price",
    "year_built",
    "floor",
    "eki_walk",
]

FEATURE_LABELS = {
    "sqr_m": "Size (sqm)",
    "rei_price": "Key Money (万円)",
    "shikikin": "Security Deposit (万円)",
    "maintenence_price": "Maintenance Fee (万円)",
    "year_built": "Year Built",
    "floor": "Floor",
    "eki_walk": "Walk to Station (min)",
}


class ExplainabilityService:
    """Central service for SHAP and LIME model explanations."""

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def explain_prediction(
        self,
        model: Any,
        input_data: Dict[str, float],
        background_data: Optional[pd.DataFrame] = None,
        method: str = "shap",
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Explain a single prediction using SHAP or LIME.

        Args:
            model:           Trained model (LightGBM, sklearn, TabNet, etc.)
            input_data:      Dict of feature values for the prediction to explain
            background_data: Reference dataset used as baseline (required for some methods)
            method:          "shap" or "lime"
            feature_names:   Column names; defaults to FEATURE_NAMES

        Returns:
            {
              "method":        "shap" | "lime",
              "prediction":    float,
              "base_value":    float,           # SHAP baseline prediction
              "features":      [
                  {"name": str, "label": str, "value": float,
                   "shap_value": float, "impact": "positive"|"negative"|"neutral"},
                  ...
              ],
              "top_positive":  [...],           # Top 3 features increasing rent
              "top_negative":  [...],           # Top 3 features decreasing rent
              "confidence":    str,
            }
        """
        names = feature_names or FEATURE_NAMES

        # Build a one-row DataFrame aligned to the expected features
        row = self._build_input_row(input_data, names)

        if method == "lime":
            return self._explain_lime(model, row, background_data, names)
        else:
            return self._explain_shap(model, row, background_data, names)

    def global_feature_importance(
        self,
        model: Any,
        background_data: pd.DataFrame,
        feature_names: Optional[List[str]] = None,
        n_samples: int = 200,
    ) -> Dict[str, Any]:
        """
        Compute global SHAP feature importance over a background dataset.

        Args:
            model:           Trained model
            background_data: Dataset to compute importance over
            feature_names:   Column names; defaults to FEATURE_NAMES
            n_samples:       Max rows to use (keeps it fast)

        Returns:
            {
              "method":   "shap_global",
              "features": [
                  {"name": str, "label": str, "mean_abs_shap": float, "rank": int},
                  ...                           # sorted by importance desc
              ]
            }
        """
        names = feature_names or FEATURE_NAMES
        data = self._align_data(background_data, names).head(n_samples)

        try:
            explainer, shap_values = self._get_shap_explainer_and_values(model, data)
        except Exception as e:
            logger.error(f"Global SHAP failed: {e}")
            return self._fallback_feature_importance(model, names)

        mean_abs = np.abs(shap_values).mean(axis=0)
        order = np.argsort(mean_abs)[::-1]

        features = [
            {
                "name": names[i],
                "label": FEATURE_LABELS.get(names[i], names[i]),
                "mean_abs_shap": float(mean_abs[i]),
                "rank": int(rank + 1),
            }
            for rank, i in enumerate(order)
            if i < len(names)
        ]

        return {"method": "shap_global", "features": features}

    # ------------------------------------------------------------------ #
    #  SHAP internals                                                      #
    # ------------------------------------------------------------------ #

    def _explain_shap(
        self,
        model: Any,
        row: pd.DataFrame,
        background_data: Optional[pd.DataFrame],
        names: List[str],
    ) -> Dict[str, Any]:
        """Run SHAP explanation for a single row."""
        try:
            import shap
        except ImportError:
            logger.error("shap not installed – run: pip install shap")
            raise RuntimeError("shap package not available")

        try:
            explainer, shap_values_bg = self._get_shap_explainer_and_values(
                model, background_data if background_data is not None else row
            )

            # Explain the single input row
            sv = explainer(row)

            if hasattr(sv, "values"):
                values = sv.values.flatten()
                base_value = float(sv.base_values.mean()) if hasattr(sv, "base_values") else 0.0
            else:
                values = np.array(sv).flatten()
                base_value = 0.0

            # Predicted value
            try:
                prediction = float(model.predict(row.values)[0])
            except Exception:
                prediction = float(base_value + np.sum(values))

            return self._format_shap_result(values, base_value, prediction, row, names)

        except Exception as e:
            logger.error(f"SHAP explanation failed: {e}", exc_info=True)
            raise RuntimeError(f"SHAP explanation error: {e}")

    def _get_shap_explainer_and_values(self, model: Any, data: pd.DataFrame):
        """Pick the fastest available SHAP explainer for the model type."""
        import shap

        model_class = type(model).__name__.lower()

        # LightGBM – exact TreeExplainer
        if "lgbm" in model_class or "lightgbm" in model_class or "booster" in model_class:
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(data)
            return explainer, np.array(sv)

        # Any sklearn tree ensemble (RandomForest, GradientBoosting, etc.)
        if hasattr(model, "estimators_") or hasattr(model, "tree_"):
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(data)
            return explainer, np.array(sv)

        # Linear / Ridge / Lasso
        if hasattr(model, "coef_"):
            bg = shap.maskers.Independent(data, max_samples=100)
            explainer = shap.LinearExplainer(model, bg)
            sv = explainer.shap_values(data)
            return explainer, np.array(sv)

        # TabNet – use built-in explain, wrap as generic explainer
        if "tabnet" in model_class:
            return self._tabnet_shap_proxy(model, data)

        # Fallback: KernelExplainer (slow but universal)
        logger.warning("Using KernelExplainer (slower) – no exact explainer for %s", model_class)
        bg = shap.sample(data, min(50, len(data)))
        explainer = shap.KernelExplainer(model.predict, bg)
        sv = explainer.shap_values(data, nsamples=100)
        return explainer, np.array(sv)

    def _tabnet_shap_proxy(self, model: Any, data: pd.DataFrame):
        """
        Use TabNet's built-in explain() to get attention masks as SHAP-like values.
        Returns a dummy explainer and numpy array of shape (n_samples, n_features).
        """
        try:
            masks, _ = model.explain(data.values)
            # masks shape: (n_samples, n_features) – aggregate across steps
            if masks.ndim == 3:
                importance = masks.sum(axis=1)  # (n_samples, n_features)
            else:
                importance = masks

            class _DummyExplainer:
                def __call__(self, X):
                    return importance

            return _DummyExplainer(), importance

        except Exception as e:
            logger.error(f"TabNet explain failed: {e}")
            raise

    def _format_shap_result(
        self,
        values: np.ndarray,
        base_value: float,
        prediction: float,
        row: pd.DataFrame,
        names: List[str],
    ) -> Dict[str, Any]:
        """Format raw SHAP values into a clean response dict."""
        features = []
        for i, name in enumerate(names):
            if i >= len(values):
                break
            sv = float(values[i])
            fv = float(row[name].iloc[0]) if name in row.columns else 0.0
            features.append({
                "name": name,
                "label": FEATURE_LABELS.get(name, name),
                "value": fv,
                "shap_value": sv,
                "impact": "positive" if sv > 0.05 else ("negative" if sv < -0.05 else "neutral"),
            })

        features_sorted = sorted(features, key=lambda x: abs(x["shap_value"]), reverse=True)
        top_positive = [f for f in features_sorted if f["impact"] == "positive"][:3]
        top_negative = [f for f in features_sorted if f["impact"] == "negative"][:3]

        # Rough confidence based on spread of SHAP values
        spread = float(np.std(values)) if len(values) > 1 else 0.0
        confidence = "high" if spread < 1.0 else ("medium" if spread < 3.0 else "low")

        return {
            "method": "shap",
            "prediction": prediction,
            "base_value": base_value,
            "features": features_sorted,
            "top_positive": top_positive,
            "top_negative": top_negative,
            "confidence": confidence,
        }

    # ------------------------------------------------------------------ #
    #  LIME internals                                                      #
    # ------------------------------------------------------------------ #

    def _explain_lime(
        self,
        model: Any,
        row: pd.DataFrame,
        background_data: Optional[pd.DataFrame],
        names: List[str],
    ) -> Dict[str, Any]:
        """Run LIME explanation for a single row."""
        try:
            from lime import lime_tabular
        except ImportError:
            logger.error("lime not installed – run: pip install lime")
            raise RuntimeError("lime package not available")

        if background_data is None:
            raise ValueError("background_data is required for LIME explanations")

        bg = self._align_data(background_data, names)

        explainer = lime_tabular.LimeTabularExplainer(
            training_data=bg.values,
            feature_names=names,
            mode="regression",
            random_state=42,
        )

        def predict_fn(X):
            df = pd.DataFrame(X, columns=names)
            return model.predict(df.values)

        explanation = explainer.explain_instance(
            data_row=row[names].values.flatten(),
            predict_fn=predict_fn,
            num_features=len(names),
            num_samples=500,
        )

        try:
            prediction = float(model.predict(row.values)[0])
        except Exception:
            prediction = float(explanation.predicted_value)

        features = []
        for feat_name, weight in explanation.as_list():
            # LIME returns strings like "sqr_m <= 45.00"; extract actual name
            raw_name = feat_name.split(" ")[0].split(">")[0].split("<")[0].strip()
            matched_name = next((n for n in names if n in feat_name), raw_name)
            fv = float(row[matched_name].iloc[0]) if matched_name in row.columns else 0.0
            features.append({
                "name": matched_name,
                "label": FEATURE_LABELS.get(matched_name, matched_name),
                "value": fv,
                "lime_weight": float(weight),
                "impact": "positive" if weight > 0 else ("negative" if weight < 0 else "neutral"),
            })

        features_sorted = sorted(features, key=lambda x: abs(x["lime_weight"]), reverse=True)

        return {
            "method": "lime",
            "prediction": prediction,
            "features": features_sorted,
            "top_positive": [f for f in features_sorted if f["impact"] == "positive"][:3],
            "top_negative": [f for f in features_sorted if f["impact"] == "negative"][:3],
        }

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _build_input_row(self, input_data: Dict[str, float], names: List[str]) -> pd.DataFrame:
        """Convert input_data dict to a one-row DataFrame with correct columns."""
        row = {name: input_data.get(name, 0.0) for name in names}
        return pd.DataFrame([row])

    def _align_data(self, data: pd.DataFrame, names: List[str]) -> pd.DataFrame:
        """Return a DataFrame containing only the requested feature columns."""
        available = [c for c in names if c in data.columns]
        missing = [c for c in names if c not in data.columns]
        if missing:
            logger.warning("Background data missing columns: %s; filling with 0", missing)
        result = data[available].copy()
        for c in missing:
            result[c] = 0.0
        return result[names]

    def _fallback_feature_importance(
        self, model: Any, names: List[str]
    ) -> Dict[str, Any]:
        """Return model's built-in feature_importances_ when SHAP fails."""
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_).flatten()
        else:
            importances = np.ones(len(names))

        # Normalize to [0, 1]
        total = importances.sum() or 1.0
        importances = importances / total

        order = np.argsort(importances)[::-1]
        features = [
            {
                "name": names[i],
                "label": FEATURE_LABELS.get(names[i], names[i]),
                "mean_abs_shap": float(importances[i]),
                "rank": int(rank + 1),
            }
            for rank, i in enumerate(order)
            if i < len(names)
        ]
        return {"method": "builtin_importance", "features": features}
