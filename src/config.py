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
PACKAGE_DIR: Path = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Paths:
    """Filesystem layout of the project."""

    root: Path = PROJECT_ROOT

    def _resolve_data_file(self, filename: str) -> Path:
        root_candidate = self.root / filename
        if root_candidate.exists():
            return root_candidate
        data_candidate = self.root / "data" / filename
        if data_candidate.exists():
            return data_candidate
        return root_candidate

    @property
    def train_csv(self) -> Path:
        return self._resolve_data_file("train.csv")

    @property
    def test_csv(self) -> Path:
        return self._resolve_data_file("test.csv")

    @property
    def sample_submission(self) -> Path:
        return self._resolve_data_file("SampleSubmission.csv")

    @property
    def label_config(self) -> Path:
        candidate = self.root / "src" / "label_config.json"
        if candidate.exists():
            return candidate
        return PACKAGE_DIR / "label_config.json"

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
        "lgbm_tweedie", "xgb_mse", "xgb_huber", "ridge", "extratrees",
    )
    classification_members: tuple[str, ...] = ("lgbm", "catboost", "xgb")
    seed: int = 42
    n_jobs: int = -1
    use_gpu: bool = False
    n_seeds_reg: int = 3
    n_seeds_clf: int = 2
    hurdle_seeds: tuple[int, ...] = (42, 100, 2024, 777, 314)

    @classmethod
    def fast(cls, **kw: Any) -> "ModelConfig":
        """Single-seed configuration used by the default CLI pipeline."""
        return cls(n_seeds_reg=1, n_seeds_clf=1, hurdle_seeds=(42,), **kw)


@dataclass(frozen=True)
class PipelineConfig:
    """Complete description of a pipeline run.

    Temporal design
    ---------------
    * Training snapshots are taken every ``cutoff_step_months`` starting at
      ``first_train_cutoff``; a snapshot is usable only if its full outcome window
      ``[c, c + horizon)`` lies strictly before the evaluation cutoff (causal embargo).
    * Prior tournament folds are used to fit calibration parameters (per-member
      debiasing, blend weights, BLUP shrinkage, temperature, and class multipliers).
    * The final fold in ``tournament_cutoffs`` acts as the out-of-time holdout.
    """

    paths: Paths = field(default_factory=Paths)
    models: ModelConfig = field(default_factory=ModelConfig)
    first_train_cutoff: str = "2024-09-01"
    cutoff_step_months: int = 1
    horizon_months: int = 3
    calibration_cutoff: str = "2025-06-01"
    evaluation_cutoff: str = "2025-09-01"
    test_cutoff: str = "2025-12-01"
    use_cache: bool = True
    prob_fusion: str = "blend"
    # Minimum combined weight assigned to two-part hurdle models in non-negative ridge stacking.
    hurdle_floor_fuel: float = 0.20
    hurdle_floor_nf: float = 0.35
    tournament_cutoffs: tuple[str, ...] = ("2025-06-01", "2025-07-01", "2025-08-01", "2025-09-01")
    # Seasonal Q4 target class proportions (expressed over the 5,488-customer test cohort)
    # used by prior-aligned dual ascent during Dec-Feb test inference.
    seasonal_test_priors: tuple[tuple[str, float], ...] = (
        ("Stable", 1955.0),
        ("Existing-category growth: Fuel", 1575.0),
        ("Inactivity", 1445.0),
        ("Existing-category growth: Other", 305.0),
        ("Existing-category growth: Beverages", 133.0),
        ("Existing-category growth: Food & beverage", 75.0),
    )

    @cached_property
    def label_config(self) -> dict[str, Any]:
        """Official label configuration (``src/label_config.json``)."""
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
