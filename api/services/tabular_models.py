"""Serve the pretrained tabular deep models (TabPFN, TabNet) behind /torch/*.

Each is a bundle written by `models/tabular_pretrained_model.py` and installed by
`services/model_sync.py`: `<variant>-model.joblib` (the fitted model) and `<variant>-features.json`
(the `FeaturePreprocessor` it was trained with). These models need every feature, deposit and key
money included, which is why the torch routes require both.
"""
import logging
from pathlib import Path
from typing import Any, Dict, Union

import joblib
import numpy as np

from services import artifact_bundle
from services.feature_preprocessor import FeaturePreprocessor
from services.model_sync import local_run_id

logger = logging.getLogger(__name__)

# serving variant -> (registered model, label); the registered name is the key the API and the
# Compare page use for the model
DEEP_VARIANTS = {
    "tabpfn": ("tokyo_rent_tabpfn", "TabPFN"),
    "tabnet": ("tokyo_rent_tabnet", "TabNet"),
}


class TabularModel:
    """A fitted TabPFN / TabNet plus the encoding that feeds it. `predict_single` is what
    `PyTorchPredictionService` calls."""

    framework = "PyTorch"
    device = "cpu"

    def __init__(self, variant: str, model: Any, preprocessor: FeaturePreprocessor, run_id: Union[str, None] = None):
        self.variant = variant
        self.model = None  # the torch service reads parameter counts off `.model`; not meaningful here
        self.fitted = model
        self.preprocessor = preprocessor
        self.run_id = run_id
        self.model_type = DEEP_VARIANTS[variant][1]

    @classmethod
    def load(cls, models_dir: Union[str, Path], variant: str) -> "TabularModel":
        models_dir = Path(models_dir)
        artifact_bundle.verify(models_dir, variant)  # raises on a model paired with another run's features
        features = models_dir / f"{variant}-features.json"
        preprocessor = FeaturePreprocessor.load(features)
        return cls(variant, joblib.load(models_dir / f"{variant}-model.joblib"), preprocessor,
                   local_run_id(models_dir, variant))

    def predict_single(self, input_data: Dict[str, Any]) -> float:
        # Neither model takes NaN: a missing deposit / key money would be the training median
        row = self.preprocessor.transform_single(input_data, impute_missing=True).astype(float)
        return float(np.ravel(self.fitted.predict(row.values))[0])

    def describe(self) -> Dict[str, Any]:
        return {
            "name": DEEP_VARIANTS[self.variant][0], "type": self.model_type, "framework": self.framework,
            "device": self.device, "features": len(self.preprocessor.feature_columns_),
            "run_id": self.run_id, "ready_for_inference": True,
        }


def load_deep_models(models_dir: Union[str, Path]) -> Dict[str, TabularModel]:
    """Every deep-model bundle in `models_dir` that loads, keyed by registered model name."""
    loaded = {}
    for variant, (name, _) in DEEP_VARIANTS.items():
        if not (Path(models_dir) / f"{variant}-model.joblib").exists():
            continue
        try:
            loaded[name] = TabularModel.load(models_dir, variant)
            logger.info("Loaded deep model %s", name)
        except Exception as e:
            logger.warning("Could not load deep model %s: %s", name, e)
    return loaded
