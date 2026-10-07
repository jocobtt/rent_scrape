"""Leakage-aware train/test splitting.

Listings are not independent: one building lists several near-identical units,
and a street block (the `address` field) shares most rent drivers. A random
split puts siblings on both sides and overstates accuracy. These helpers split
by group (building, falling back to address) so unseen buildings are held out.
"""
import logging
from typing import Optional, Tuple

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

logger = logging.getLogger(__name__)

META_COLS = ["address", "building_id", "scraped_at"]  # identifiers, never features
MIN_GROUPS = 10  # below this a group split is meaningless; fall back to random


def group_labels(df: pd.DataFrame) -> Optional[pd.Series]:
    """Best available grouping key: `building_id`, else `address`, else None."""
    for col in ("building_id", "address"):
        if col in df.columns and df[col].notna().any():
            return df[col].astype(str)
    return None


def grouped_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    groups: Optional[pd.Series],
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Optional[pd.Series]]:
    """Split so no group appears in both train and test.

    Returns (X_train, X_test, y_train, y_test, groups_train). `groups_train` is
    None when no usable grouping exists and a plain random split was used.
    """
    if groups is None or groups.nunique() < MIN_GROUPS:
        logger.warning(
            "No usable group column (need >= %d groups); falling back to a random split, "
            "which can leak near-duplicate listings into the test set.", MIN_GROUPS,
        )
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return X_tr, X_te, y_tr, y_te, None

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    return (
        X.iloc[train_idx], X.iloc[test_idx],
        y.iloc[train_idx], y.iloc[test_idx],
        groups.iloc[train_idx],
    )


GROUP_COL = "_group"


def attach_groups(lineage_df: pd.DataFrame, groups: Optional[pd.Series]) -> pd.DataFrame:
    """Carry group labels on the lineage frame (same index as X) so downstream
    training code can recover them without changing `load_data`'s return shape."""
    if groups is not None:
        lineage_df = lineage_df.copy()
        lineage_df[GROUP_COL] = groups.values
    return lineage_df


def groups_for(lineage_df: Optional[pd.DataFrame], X_subset: pd.DataFrame) -> Optional[pd.Series]:
    """Group labels for the rows of `X_subset` (e.g. X_train), or None."""
    if lineage_df is None or GROUP_COL not in lineage_df.columns:
        return None
    return lineage_df.loc[X_subset.index, GROUP_COL]


def split_meta(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """Return (df without identifier columns, group labels).

    Call this on the raw frame *before* encoding so loaders that do their own
    `get_dummies` neither leak identifiers into the features nor lose the groups.
    """
    groups = group_labels(df)
    return df.drop(columns=META_COLS, errors="ignore"), groups


def grouped_train_val_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    groups: Optional[pd.Series],
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
):
    """Grouped train/val/test split (val is carved out of train, also by group).

    Returns (X_train, X_val, X_test, y_train, y_val, y_test).
    """
    X_tr, X_te, y_tr, y_te, g_tr = grouped_train_test_split(X, y, groups, test_size, random_state)
    X_tr, X_val, y_tr, y_val, _ = grouped_train_test_split(X_tr, y_tr, g_tr, val_size, random_state)
    return X_tr, X_val, X_te, y_tr, y_val, y_te
