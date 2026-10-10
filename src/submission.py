"""Submission assembly and strict format validation."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

COLUMNS: tuple[str, ...] = ("ID", "CLV_fuel", "CLV_nonfuel", "Opportunity")


def build_submission(ids: Sequence[str], fuel: np.ndarray, nonfuel: np.ndarray, opportunity: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame({
        "ID": [str(i) for i in ids],
        "CLV_fuel": np.clip(np.asarray(fuel, dtype="float64"), 0.0, None),
        "CLV_nonfuel": np.clip(np.asarray(nonfuel, dtype="float64"), 0.0, None),
        "Opportunity": list(opportunity),
    })


def validate_submission(sub: pd.DataFrame, sample: pd.DataFrame, allowed_labels: Sequence[str]) -> None:
    """Raise ``ValueError`` describing the first format violation found."""
    if list(sub.columns) != list(COLUMNS):
        raise ValueError(f"Columns must be {list(COLUMNS)}, got {list(sub.columns)}")
    if len(sub) != len(sample):
        raise ValueError(f"Expected {len(sample)} rows, got {len(sub)}")
    if sub["ID"].duplicated().any():
        raise ValueError("Duplicate IDs in submission")
    if set(sub["ID"].astype(str)) != set(sample["ID"].astype(str)):
        raise ValueError("Submission IDs differ from SampleSubmission IDs")
    for col in ("CLV_fuel", "CLV_nonfuel"):
        values = pd.to_numeric(sub[col], errors="coerce")
        if values.isna().any() or not np.isfinite(values).all():
            raise ValueError(f"{col} contains missing or non-finite values")
        if (values < 0).any():
            raise ValueError(f"{col} contains negative values")
    bad = set(sub["Opportunity"]) - set(allowed_labels)
    if bad:
        raise ValueError(f"Invalid Opportunity labels: {sorted(bad)}")


def write_submission(sub: pd.DataFrame, path: Path) -> None:
    sub.to_csv(path, index=False, float_format="%.10f")
