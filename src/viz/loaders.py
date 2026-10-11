"""Data access and cohort loading utilities for the EDA suite."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import PipelineConfig
from src.data import load_transactions
from src.dataset import Snapshot, SnapshotBuilder, stack

logger = logging.getLogger("src.viz.loaders")


def get_builder_and_tx(cfg: PipelineConfig) -> tuple[SnapshotBuilder, pd.DataFrame]:
    """Retrieve transaction dataframe and snapshot builder."""
    tx = load_transactions(cfg)
    builder = SnapshotBuilder(cfg, tx)
    return builder, tx


def load_cached_predictions(cfg: PipelineConfig) -> dict[str, pd.DataFrame]:
    """Load cached model predictions from reports/eda/cache/ if available."""
    cache_dir = cfg.paths.reports_dir / "eda" / "cache"
    out: dict[str, pd.DataFrame] = {}
    for name in ("eval_predictions", "test_predictions", "cal_predictions"):
        path = cache_dir / f"{name}.parquet"
        if path.exists():
            out[name] = pd.read_parquet(path)
    return out


def load_rolling_holdouts(
    cfg: PipelineConfig,
    cutoffs: tuple[str, ...] = ("2025-03-01", "2025-06-01", "2025-09-01")
) -> list[Snapshot]:
    """Load multiple out-of-time evaluation snapshots with labels to assess temporal stability."""
    builder, _ = get_builder_and_tx(cfg)
    snapshots: list[Snapshot] = []
    for c in cutoffs:
        try:
            snap = builder.build(c, with_labels=True)
            snapshots.append(snap)
        except Exception as e:
            logger.warning("Could not build snapshot for cutoff %s: %s", c, e)
    return snapshots
