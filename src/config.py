"""Central, immutable configuration for the Sasol CLV & Opportunity pipeline.

All tunable knobs (paths, temporal design, model families, seeds) live here so that
a run is fully described by a single :class:`PipelineConfig` instance.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Paths:
    """Filesystem layout of the project."""

    root: Path = PROJECT_ROOT

    @property
    def train_csv(self) -> Path:
        return self.root / "train.csv"

    @property
    def test_csv(self) -> Path:
        return self.root / "test.csv"

    @property
    def sample_submission(self) -> Path:
        return self.root / "SampleSubmission.csv"

    @property
    def label_config(self) -> Path:
        return self.root / "label_config.json"

    @property
    def cache_dir(self) -> Path:
        return self.root / "data_cache"

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"

    @property
    def submissions_dir(self) -> Path:
        return self.root / "submissions"

    @property
    def submission(self) -> Path:
        return self.submissions_dir / "submission_latest.csv"


@dataclass(frozen=True)
class ModelConfig:
    """Model-family selection and determinism controls."""

    regression_members: tuple[str, ...] = (
        "lgbm_mse", "lgbm_huber", "lgbm_tweedie",
        "xgb_mse", "xgb_huber",
        "catboost", "catboost_huber",
        "ridge", "extratrees", "mlp",
    )
    classification_members: tuple[str, ...] = ("lgbm", "catboost", "xgb")
    seed: int = 42
    n_jobs: int = -1
    use_gpu: bool = False


@dataclass(frozen=True)
class PipelineConfig:
    """Complete description of a pipeline run.

    Temporal design
    ---------------
    * Training snapshots are taken every ``cutoff_step_months`` starting at
      ``first_train_cutoff``; a snapshot is usable only if its full outcome window
      ``[c, c + horizon)`` lies before the boundary it is trained for (embargo).
    * ``calibration_cutoff`` (fold A) is used to fit post-processing parameters
      (blend weights, class decision weights).
    * ``evaluation_cutoff`` (fold B) is scored with parameters fitted on fold A,
      giving an honest out-of-time estimate.
    """

    paths: Paths = field(default_factory=Paths)
    models: ModelConfig = field(default_factory=ModelConfig)
    first_train_cutoff: str = "2024-07-01"
    cutoff_step_months: int = 1
    horizon_months: int = 3
    calibration_cutoff: str = "2025-06-01"
    evaluation_cutoff: str = "2025-09-01"
    test_cutoff: str = "2025-12-01"
    use_cache: bool = True

    @cached_property
    def label_config(self) -> dict[str, Any]:
        """Official label configuration (``label_config.json``)."""
        return json.loads(self.paths.label_config.read_text())

    @property
    def horizon(self) -> pd.DateOffset:
        return pd.DateOffset(months=self.horizon_months)

    def training_cutoffs(self, boundary: str | pd.Timestamp) -> list[pd.Timestamp]:
        """Snapshot cutoffs whose whole outcome window ends on/before ``boundary``.

        This is the embargo that prevents label windows from overlapping the
        period a model is evaluated on.
        """
        boundary = pd.Timestamp(boundary)
        cutoffs: list[pd.Timestamp] = []
        c = pd.Timestamp(self.first_train_cutoff)
        while c + self.horizon <= boundary:
            cutoffs.append(c)
            c = c + pd.DateOffset(months=self.cutoff_step_months)
        return cutoffs
