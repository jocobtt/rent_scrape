"""
Prediction service for handling model predictions and comparisons.
"""

import logging
import math
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

_MODELS_DIR = Path(__file__).parent.parent / "models"

# Base numeric features, in a stable fallback order used when no preprocessor
# JSON is available (e.g. first boot before any training run).
_FALLBACK_FEATURES = [
    "sqr_m", "rei_price", "shikikin", "maintenence_price",
    "year_built", "floor", "nearest_eki_walk", "contract_type",
]


def _load_preprocessor(filename: str):
    """Return a FeaturePreprocessor loaded from models/{filename}, or None."""
    path = _MODELS_DIR / filename
    if not path.exists():
        logger.debug("Preprocessor file not found: %s (will use fallback features)", path)
        return None
    try:
        from services.feature_preprocessor import FeaturePreprocessor
        return FeaturePreprocessor.load(path)
    except Exception as exc:
        logger.warning("Could not load preprocessor %s: %s", path, exc)
        return None


class PredictionService:
    """Service class for handling model predictions."""

    def __init__(self, passed_model, challenger_model):
        self.passed_model = passed_model
        self.challenger_model = challenger_model
        self._challenger_prep = _load_preprocessor("challenger-features.json")
        self._passed_prep = _load_preprocessor("passed-features.json")

    def _sanitize_input(self, input_data: Dict) -> Dict:
        """Replace NaN/inf float values with 0.0."""
        sanitized = {}
        for key, val in input_data.items():
            if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                logger.warning("Input field '%s' is %s; replacing with 0.0", key, val)
                sanitized[key] = 0.0
            else:
                sanitized[key] = val
        return sanitized

    def _build_features(self, input_data: Dict, preprocessor, model_label: str) -> pd.DataFrame:
        """
        Build the feature DataFrame the model expects.

        If a preprocessor is available it applies the full one-hot encoding
        (ku_name, apartment_type, house_type) with the exact column order
        used during training.

        If no preprocessor exists (models trained before this service version)
        a minimal DataFrame of numeric base features is returned — the models
        will still produce valid predictions, just without the categorical signal.
        """
        if preprocessor is not None:
            return preprocessor.transform_single(input_data)

        logger.debug(
            "No preprocessor for %s — using numeric base features only.", model_label
        )
        row = {
            "sqr_m":             input_data.get("sqr_m", 0),
            "rei_price":         input_data.get("rei_price", 0),
            "shikikin":          input_data.get("shikikin", 0),
            "maintenence_price": input_data.get("maintenence_price", 0),
            "year_built":        input_data.get("year_built", 0),
            "floor":             input_data.get("floor", 0),
            # Handle both API name (eki_walk) and training name (nearest_eki_walk)
            "nearest_eki_walk":  input_data.get("eki_walk", input_data.get("nearest_eki_walk", 0)),
            "contract_type":     input_data.get("contract_type", 0),
        }
        return pd.DataFrame([row])

    def predict_with_passed_model(self, input_data: Dict) -> Dict[str, Any]:
        """Make a prediction using the passed (production) model."""
        try:
            input_data = self._sanitize_input(input_data)
            df = self._build_features(input_data, self._passed_prep, "passed")
            pred = self.passed_model.predict(df)
            return {
                "prediction": float(pred[0]) if isinstance(pred, np.ndarray) else float(pred),
                "model_type": "passed",
                "status": "success",
            }
        except Exception as e:
            logger.error("Passed model prediction error: %s", e)
            raise

    def predict_with_challenger_model(self, input_data: Dict) -> Dict[str, Any]:
        """Make a prediction using the challenger model."""
        try:
            input_data = self._sanitize_input(input_data)
            df = self._build_features(input_data, self._challenger_prep, "challenger")
            pred = self.challenger_model.predict(df)
            return {
                "prediction": float(pred[0]) if isinstance(pred, np.ndarray) else float(pred),
                "model_type": "challenger",
                "status": "success",
            }
        except Exception as e:
            logger.error("Challenger model prediction error: %s", e)
            raise

    def compare_predictions(self, input_data: Dict) -> Dict[str, Any]:
        """Compare predictions from both models."""
        try:
            input_data = self._sanitize_input(input_data)
            passed_result = self.predict_with_passed_model(input_data)
            challenger_result = self.predict_with_challenger_model(input_data)

            passed_pred = passed_result["prediction"]
            challenger_pred = challenger_result["prediction"]
            difference = abs(passed_pred - challenger_pred)
            pct_diff = (
                abs((passed_pred - challenger_pred) / passed_pred * 100)
                if passed_pred != 0 else 0
            )

            return {
                "predictions": {
                    "passed_model": passed_pred,
                    "challenger_model": challenger_pred,
                    "difference": difference,
                    "percentage_difference": pct_diff,
                },
                "input_data": input_data,
                "status": "success",
            }
        except Exception as e:
            logger.error("Comparison prediction error: %s", e)
            raise
