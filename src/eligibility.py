"""Hard feasibility constraints on the Opportunity label.

Derived generically from ``label_config['category_mapping']`` (no hard-coded
category lists):

* ``Existing-category growth: X`` requires a prior qualifying purchase in at least
  one included category that maps to ``X`` for growth.
* ``New category adoption: X`` requires at least one included category mapping to
  ``X`` for adoption that has *never* been purchased.
* ``Inactivity`` and ``Stable`` are always feasible.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.data import included_categories

GROWTH = "Existing-category growth"
ADOPTION = "New category adoption"
ALWAYS_FEASIBLE = ("Inactivity", "Stable")


def ever_purchased(tx: pd.DataFrame, cutoff: pd.Timestamp, ids: pd.Index, label_config: Mapping[str, Any]) -> pd.DataFrame:
    """Boolean (ids x included categories) matrix of qualifying purchases before ``cutoff``."""
    cats = included_categories(label_config)
    q = tx[(tx["_time"] < pd.Timestamp(cutoff)) & tx["is_qualifying"]]
    counts = q.groupby(["ID", "ItemCategoryLevel2"])["TransactionId"].nunique().unstack()
    return counts.reindex(index=ids, columns=cats).fillna(0).gt(0)


def mask_from_history(ever: pd.DataFrame, label_config: Mapping[str, Any]) -> np.ndarray:
    """Feasibility mask (n_customers x n_labels) in ``opportunity_labels`` order.

    Raises:
        ValueError: if a label cannot be reached by any included category.
    """
    mapping = label_config["category_mapping"]
    cats = list(ever.columns)
    columns: list[np.ndarray] = []
    for label in label_config["opportunity_labels"]:
        if label in ALWAYS_FEASIBLE:
            columns.append(np.ones(len(ever), dtype=bool))
            continue
        kind, target = label.split(": ", 1)
        routes = [c for c in cats if c in mapping and mapping[c].get(kind) == target]
        if kind == GROWTH:
            columns.append(ever[routes].any(axis=1).to_numpy() if routes else np.zeros(len(ever), dtype=bool))
        elif kind == ADOPTION:
            columns.append((~ever[routes]).any(axis=1).to_numpy() if routes else np.ones(len(ever), dtype=bool))
        else:
            raise ValueError(f"Unknown opportunity kind in {label!r}")
    return np.column_stack(columns)


def opportunity_mask(tx: pd.DataFrame, cutoff: pd.Timestamp, ids: pd.Index, label_config: Mapping[str, Any]) -> np.ndarray:
    """Convenience wrapper: history -> feasibility mask."""
    return mask_from_history(ever_purchased(tx, cutoff, ids, label_config), label_config)
