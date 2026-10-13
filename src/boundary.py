"""Specialized binary boundary classifier module for Fuel Growth vs. Stable.

Resolves the single largest source of Opportunity classification confusion in the
competition: distinguishing customers who achieve >= 25% fuel spend growth (>= R20)
from those whose spend remains stable, using physical volume and hurdle dynamics.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

# Fuel conversion constant from label_config.json: log1p(litres) / 2.352825386185113
FUEL_NORM_FACTOR: float = 2.352825386185113


def build_boundary_features(
    features_df: pd.DataFrame,
    pred_fuel: np.ndarray | pd.Series | None = None,
) -> pd.DataFrame:
    """Engineer specialized physical boundary features for Fuel Growth vs Stable.

    Parameters
    ----------
    features_df : pd.DataFrame
        Snapshot feature matrix.
    pred_fuel : array-like or None
        Normalized CLV fuel predictions from regression model, if available.
    """
    out = features_df.copy()

    # Reconstruct physical fuel litres and historical rand
    hist_lit_90d = out["fuel_litres_90d"].fillna(0.0) if "fuel_litres_90d" in out.columns else pd.Series(0.0, index=out.index)
    hist_lit_30d = out["fuel_litres_30d"].fillna(0.0) if "fuel_litres_30d" in out.columns else pd.Series(0.0, index=out.index)
    hist_lit_180d = out["fuel_litres_180d"].fillna(0.0) if "fuel_litres_180d" in out.columns else pd.Series(0.0, index=out.index)

    if "cat_q1_spend_Fuel" in out.columns:
        hist_fuel_rand = out["cat_q1_spend_Fuel"].fillna(0.0)
    elif "m3_fuel_rand" in out.columns:
        hist_fuel_rand = out["m3_fuel_rand"].fillna(0.0)
    else:
        hist_fuel_rand = pd.Series(0.0, index=out.index)

    # 1. Official label_rules growth hurdle: >= 25% growth AND >= R20
    hurdle_rand = np.maximum(20.0, 0.25 * hist_fuel_rand)
    out["bnd_hurdle_25pct_rand"] = hurdle_rand
    out["bnd_hurdle_to_spend_ratio"] = hurdle_rand / (hist_fuel_rand + 1.0)

    # 2. Litre momentum and frequency acceleration
    out["bnd_litres_m1_m3_accel"] = (hist_lit_30d * 3.0 + 1.0) / (hist_lit_90d + 1.0)
    txns_30d = out["fuel_txns_30d"].fillna(0.0) if "fuel_txns_30d" in out.columns else pd.Series(0.0, index=out.index)
    txns_90d = out["fuel_txns_90d"].fillna(0.0) if "fuel_txns_90d" in out.columns else pd.Series(0.0, index=out.index)
    out["bnd_fuel_freq_accel"] = (txns_30d * 3.0 + 1.0) / (txns_90d + 1.0)

    # 3. Spend reversion score (recent 90d vs previous 180d)
    out["bnd_fuel_reversion_score"] = (hist_lit_90d * 2.0 + 1.0) / (hist_lit_180d + 1.0)

    # 4. Total spend momentum
    spend_30d = out["total_spend_30d"].fillna(0.0) if "total_spend_30d" in out.columns else hist_lit_30d * 23.0
    spend_90d = out["total_spend_90d"].fillna(0.0) if "total_spend_90d" in out.columns else hist_lit_90d * 23.0
    out["bnd_total_spend_m1_m3_accel"] = (spend_30d * 3.0 + 1.0) / (spend_90d + 1.0)
    out["bnd_is_high_volume_ceiling"] = (hist_lit_90d > 200.0).astype(np.float64)

    return out


class BinaryBoundaryClassifier:
    """Ensemble of LightGBM, CatBoost, and XGBoost specialized for Fuel Growth vs Stable.

    Trained strictly on the binary label:
      1: Existing-category growth: Fuel
      0: Stable
    """

    def __init__(
        self,
        seed: int = 42,
        n_jobs: int = -1,
        use_gpu: bool = False,
        lgb_weight: float = 0.40,
        cb_weight: float = 0.35,
        xgb_weight: float = 0.25,
    ) -> None:
        self.seed = seed
        self.n_jobs = n_jobs
        self.use_gpu = use_gpu
        self.weights = np.array([lgb_weight, cb_weight, xgb_weight], dtype=np.float64)
        self.weights /= self.weights.sum()

        self.clf_lgb = LGBMClassifier(
            n_estimators=350,
            learning_rate=0.03,
            num_leaves=31,
            colsample_bytree=0.7,
            subsample=0.8,
            reg_lambda=5.0,
            random_state=seed,
            n_jobs=n_jobs,
            verbose=-1,
        )

        cb_kwargs: dict[str, Any] = {
            "iterations": 400,
            "learning_rate": 0.035,
            "depth": 6,
            "l2_leaf_reg": 5.0,
            "loss_function": "Logloss",
            "random_seed": seed,
            "verbose": 0,
            "allow_writing_files": False,
        }
        if use_gpu:
            cb_kwargs["task_type"] = "GPU"
        else:
            cb_kwargs["thread_count"] = n_jobs
        self.clf_cb = CatBoostClassifier(**cb_kwargs)

        xgb_kwargs: dict[str, Any] = {
            "objective": "binary:logistic",
            "n_estimators": 350,
            "learning_rate": 0.03,
            "max_depth": 5,
            "min_child_weight": 5.0,
            "subsample": 0.8,
            "colsample_bytree": 0.7,
            "reg_lambda": 5.0,
            "tree_method": "hist",
            "random_state": seed,
            "n_jobs": n_jobs,
            "eval_metric": "logloss",
        }
        if use_gpu:
            xgb_kwargs["device"] = "cuda"
        self.clf_xgb = XGBClassifier(**xgb_kwargs)

        self.feature_names_: list[str] = []

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "BinaryBoundaryClassifier":
        """Fit the three ensemble members on the binary boundary."""
        # Drop ID or object columns if present
        drop_cols = [c for c in X.columns if c in ("ID", "opp", "Opportunity") or X[c].dtype == object]
        X_num = X.drop(columns=drop_cols) if drop_cols else X
        self.feature_names_ = list(X_num.columns)

        y_arr = np.asarray(y, dtype=int)
        sw = np.asarray(sample_weight, dtype=np.float32) if sample_weight is not None else None

        logger.info("Fitting BinaryBoundaryClassifier (LightGBM, CatBoost, XGBoost) on N=%d instances x %d features...", len(X_num), len(self.feature_names_))
        self.clf_lgb.fit(X_num, y_arr, sample_weight=sw)
        self.clf_cb.fit(X_num, y_arr, sample_weight=sw)
        self.clf_xgb.fit(X_num, y_arr, sample_weight=sw)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict binary probabilities: shape (N, 2), columns [P(Stable), P(Fuel Growth)]."""
        drop_cols = [c for c in X.columns if c in ("ID", "opp", "Opportunity") or X[c].dtype == object]
        X_num = X.drop(columns=drop_cols) if drop_cols else X
        # Ensure column alignment
        X_aligned = X_num[self.feature_names_]

        p1_lgb = self.clf_lgb.predict_proba(X_aligned)[:, 1]
        p1_cb = self.clf_cb.predict_proba(X_aligned)[:, 1]
        p1_xgb = self.clf_xgb.predict_proba(X_aligned)[:, 1]

        p1 = self.weights[0] * p1_lgb + self.weights[1] * p1_cb + self.weights[2] * p1_xgb
        p1 = np.clip(p1, 1e-6, 1.0 - 1e-6)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])
