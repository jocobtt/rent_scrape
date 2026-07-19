"""
FeaturePreprocessor — consistent one-hot encoding between training and inference.

Encodes categorical columns (ku_name, apartment_type, house_type) using the
exact category sets seen during training and reindexes any prediction row to
match the saved feature-column order.

Typical training flow:
    preprocessor = FeaturePreprocessor()
    encoded_df, feature_cols = preprocessor.fit_transform(df)
    preprocessor.save("models/challenger-features.json")
    X = encoded_df.drop("rent_price", axis=1)
    y = encoded_df["rent_price"]

Typical inference flow:
    preprocessor = FeaturePreprocessor.load("models/challenger-features.json")
    X_pred = preprocessor.transform_single(input_dict)
    prediction = model.predict(X_pred)
"""

import json
import logging
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)

_CAT_COLS = ["ku_name", "apartment_type", "house_type"]

# API field name → training column name mapping.
# The InputData schema uses 'eki_walk'; the scraped/cleaned data uses
# 'nearest_eki_walk'. Both are handled transparently here.
_FIELD_ALIASES = {
    "eki_walk": "nearest_eki_walk",
}


class FeaturePreprocessor:
    """
    Fits on a training DataFrame to capture the one-hot column order, then
    replicates that exact encoding for single-row inference inputs.

    All categorical fields are optional at inference time — missing or
    unrecognised values produce all-zero dummy columns for that category
    (equivalent to "other / unknown"), which is safe for both tree models
    and linear models.
    """

    def fit_transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """
        One-hot encode categorical columns in *df* and remember the column order.

        Args:
            df: Raw training DataFrame as produced by scraper.clean_data().
                May contain 'rent_price', 'address', and all scraped columns.

        Returns:
            (encoded_df, feature_columns) where encoded_df retains 'rent_price'
            and feature_columns lists every column that is NOT 'rent_price'.
        """
        encoded = df.copy()
        encoded = encoded.drop(columns=["address"], errors="ignore")

        for col in _CAT_COLS:
            if col in encoded.columns:
                encoded[col] = encoded[col].astype(str).astype("category")

        present_cats = [c for c in _CAT_COLS if c in encoded.columns]
        encoded = pd.get_dummies(encoded, columns=present_cats)

        self.feature_columns_: list[str] = [c for c in encoded.columns if c != "rent_price"]
        logger.info(
            "FeaturePreprocessor fitted: %d features (%d from categoricals)",
            len(self.feature_columns_),
            len(self.feature_columns_) - (len(df.columns) - len(present_cats) - 2),  # rough count
        )
        return encoded, self.feature_columns_

    def transform_single(self, input_dict: dict) -> pd.DataFrame:
        """
        Build a single-row DataFrame with exactly the columns the model expects.

        Args:
            input_dict: Flat dict of field values from the API (e.g. from
                        InputData.model_dump()).  Categorical fields
                        (ku_name, apartment_type, house_type) are optional.

        Returns:
            Single-row DataFrame aligned to self.feature_columns_.
        """
        if not hasattr(self, "feature_columns_"):
            raise RuntimeError("FeaturePreprocessor has not been fitted or loaded.")

        row: dict = {col: 0 for col in self.feature_columns_}

        # Resolve aliases first (eki_walk → nearest_eki_walk)
        resolved = {}
        for k, v in input_dict.items():
            resolved[_FIELD_ALIASES.get(k, k)] = v

        # Copy numeric fields that appear directly in the feature list
        for key, val in resolved.items():
            if key in row and val is not None:
                row[key] = val

        # Set appropriate dummy flag for each categorical input
        for cat_col in _CAT_COLS:
            val = resolved.get(cat_col)
            if val:
                dummy = f"{cat_col}_{val}"
                if dummy in row:
                    row[dummy] = 1
                else:
                    logger.debug(
                        "Category value '%s' not seen in training; treated as 'other'.", dummy
                    )

        return pd.DataFrame([row])

    def save(self, path: str | Path) -> None:
        """Persist the fitted column list to a JSON file."""
        path = Path(path)
        state = {"feature_columns": self.feature_columns_}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        logger.info("FeaturePreprocessor saved → %s  (%d features)", path, len(self.feature_columns_))

    @classmethod
    def load(cls, path: str | Path) -> "FeaturePreprocessor":
        """Load a previously saved preprocessor from a JSON file."""
        with open(path, encoding="utf-8") as f:
            state = json.load(f)
        obj = cls()
        obj.feature_columns_ = state["feature_columns"]
        logger.info("FeaturePreprocessor loaded ← %s  (%d features)", path, len(obj.feature_columns_))
        return obj
