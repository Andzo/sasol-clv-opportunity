"""Ground-truth label generation (thin, validated wrapper around ``src.label_rules``)."""
from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from src import label_rules

LABEL_COLUMNS: tuple[str, ...] = (
    "CLV_fuel", "CLV_nonfuel", "Opportunity", "fuel_litres", "nonfuel_rand",
    "winning_source_category", "winning_increment_rand",
)


def eligible_ids(tx: pd.DataFrame, cutoff: pd.Timestamp, label_config: Mapping[str, Any]) -> pd.Index:
    """Customers with >= N distinct baskets strictly before ``cutoff`` (rule 1)."""
    ids = label_rules.eligible_ids(tx, pd.Timestamp(cutoff), label_config["minimum_historical_transactions"])
    return pd.Index(ids.astype(str), name="ID")


def build_labels(tx: pd.DataFrame, cutoff: pd.Timestamp, label_config: Mapping[str, Any]) -> pd.DataFrame:
    """Official labels for ``cutoff`` indexed by customer ID.

    Raises:
        ValueError: if the outcome window extends beyond the released history.
    """
    cutoff = pd.Timestamp(cutoff)
    end = cutoff + pd.DateOffset(months=3)
    if end > pd.Timestamp(label_config["public_history_end_exclusive"]):
        raise ValueError(f"Outcome window for {cutoff.date()} ends after the released history")
    raw = label_rules.raw_labels(tx, str(cutoff.date()), label_config)
    out = label_rules.transform(raw, label_config)
    for col in ("fuel_litres", "nonfuel_rand", "winning_source_category", "winning_increment_rand"):
        out[col] = raw[col].to_numpy()
    out["ID"] = out["ID"].astype(str)
    return out.set_index("ID")[list(LABEL_COLUMNS)]
