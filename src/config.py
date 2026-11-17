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

    # Pruned: lgbm_mse, lgbm_mse_deep, lgbm_huber, catboost, catboost_huber and mlp received 0.0 fuel
    # blend weight in the last full run, so they only cost training time.
    # Re-check with reports/member_weights.json (written each run) before adding any back.
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
        """Single-seed configuration for quick experiment iteration."""
        return cls(n_seeds_reg=1, n_seeds_clf=1, hurdle_seeds=(42,), **kw)


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
    first_train_cutoff: str = "2024-09-01"
    cutoff_step_months: int = 1
    horizon_months: int = 3
    calibration_cutoff: str = "2025-06-01"
    evaluation_cutoff: str = "2025-09-01"
    test_cutoff: str = "2025-12-01"
    use_cache: bool = True
    # Out-of-fold tournament cutoffs. The last entry is the forward holdout; the rest fit calibration.
    # Fast mode uses ("2025-06-01", "2025-09-01"): neighbouring monthly folds overlap by 2/3 of their
    # outcome window so they are highly correlated and cost 2x more for little extra information.
    # Opportunity probability fusion: "blend" = fixed 0.65*meta + 0.35*(w_base*base + w_hier*hier) (legacy),
    # "stack" = multinomial log-loss logistic stacker over [log base, log hier, log meta, p_exceed, p_inact].
    prob_fusion: str = "blend"
    # Minimum combined weight forced onto hurdle columns in the regression non-negative ridge stack.
    hurdle_floor_fuel: float = 0.20
    hurdle_floor_nf: float = 0.35
    # Causal GM optimizations:
    # 1. Soft shrinkage overfits in-sample and hurts OOF RMSE by +0.0017 on nonfuel and +0.0014 on fuel.
    use_soft_shrinkage: bool = False
    # 2. StructuralMetaClassifier (LGBM+CatBoost+XGB on 96 features) takes 3 min and overfits OOF.
    use_meta_classifier: bool = False
    # 3. Commuter arbitration re-assigns customers after dual ascent and drops F1 by -0.018.
    use_commuter_arbitration: bool = False
    tournament_cutoffs: tuple[str, ...] = ("2025-06-01", "2025-07-01", "2025-08-01", "2025-09-01")

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
