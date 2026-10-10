"""Snapshot assembly (features + labels + feasibility mask) with a content-addressed cache.

The cache key is a SHA-256 over the *source code* of the feature, label and
eligibility modules, the label configuration and the transaction fingerprint, so a
code change can never silently reuse stale features.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from src import eligibility, features, labels
from src.config import PipelineConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Snapshot:
    """All model inputs/targets for one cutoff."""

    cutoff: pd.Timestamp
    features: pd.DataFrame
    mask: np.ndarray
    labels: pd.DataFrame | None = None

    @property
    def ids(self) -> pd.Index:
        return self.features.index


@dataclass(frozen=True)
class Panel:
    """Several snapshots stacked row-wise (training matrix)."""

    X: pd.DataFrame
    labels: pd.DataFrame
    mask: np.ndarray
    cutoffs: np.ndarray


class SnapshotBuilder:
    """Builds and caches :class:`Snapshot` objects for arbitrary cutoffs."""

    def __init__(self, cfg: PipelineConfig, tx: pd.DataFrame) -> None:
        self.cfg = cfg
        self.tx = tx
        self.data_start: pd.Timestamp = tx["_time"].min()
        self.fingerprint = self._fingerprint()

    def _fingerprint(self) -> str:
        h = hashlib.sha256()
        for module in (features, labels, eligibility):
            h.update(inspect.getsource(module).encode())
        h.update(json.dumps(self.cfg.label_config, sort_keys=True).encode())
        h.update(f"{len(self.tx)}|{self.data_start}|{int(self.tx['_amount'].sum())}".encode())
        return h.hexdigest()[:12]

    def _cache_dir(self, cutoff: pd.Timestamp) -> Path:
        return self.cfg.paths.cache_dir / f"snapshot_{cutoff:%Y-%m-%d}_{self.fingerprint}"

    def build(self, cutoff: pd.Timestamp | str, with_labels: bool = True) -> Snapshot:
        """Return the snapshot for ``cutoff`` (from cache when available)."""
        cutoff = pd.Timestamp(cutoff)
        cache = self._cache_dir(cutoff)
        f_path, m_path, l_path = cache / "features.parquet", cache / "mask.npy", cache / "labels.parquet"
        if self.cfg.use_cache and f_path.exists() and m_path.exists() and (l_path.exists() or not with_labels):
            feats = pd.read_parquet(f_path)
            lab = pd.read_parquet(l_path) if with_labels else None
            return Snapshot(cutoff, feats, np.load(m_path), lab)

        lc = self.cfg.label_config
        ids = labels.eligible_ids(self.tx, cutoff, lc)
        feats = features.build_features(self.tx, cutoff, ids, lc, self.data_start)
        mask = eligibility.opportunity_mask(self.tx, cutoff, ids, lc)
        lab = labels.build_labels(self.tx, cutoff, lc).reindex(ids) if with_labels else None
        if lab is not None and lab["Opportunity"].isna().any():
            raise RuntimeError(f"Label/feature ID mismatch at {cutoff.date()}")

        cache.mkdir(parents=True, exist_ok=True)
        feats.to_parquet(f_path)
        np.save(m_path, mask)
        if lab is not None:
            lab.to_parquet(l_path)
        logger.info("Built snapshot %s: %d customers x %d features", cutoff.date(), len(ids), feats.shape[1])
        return Snapshot(cutoff, feats, mask, lab)


def stack(snapshots: Sequence[Snapshot]) -> Panel:
    """Concatenate labelled snapshots into a training panel."""
    if any(s.labels is None for s in snapshots):
        raise ValueError("All snapshots in a training panel must carry labels")
    X = pd.concat([s.features for s in snapshots], axis=0)
    lab = pd.concat([s.labels for s in snapshots], axis=0)
    mask = np.vstack([s.mask for s in snapshots])
    cutoffs = np.concatenate([np.full(len(s.features), s.cutoff) for s in snapshots])
    return Panel(X=X, labels=lab, mask=mask, cutoffs=cutoffs)
