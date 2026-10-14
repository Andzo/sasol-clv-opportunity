"""Module 2: Exact L2 Two-Part Hurdle Regressor for v3 Architecture.

Eliminates Tweedie/Huber median bias on log-transformed targets through:
1. Stage 1: Calibrated Binary Classifier P(Z > 0 | X) with Isotonic calibration.
2. Stage 2: Positive-Spender Multi-Loss Conditional Expectation:
   - Regularized L2 (RMSE) GBDT with target winsorization.
   - Huber (alpha=0.85) regressor with parametric polynomial mean-correction.
3. Stage 3: Residual Meta-Regressor taking [P, mu_pos, P*mu_pos, Z_direct, baseline_m3, is_summer]
   to minimize unconditional population RMSE.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.linear_model import RidgeCV
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold

logger = logging.getLogger("regression_v3")


class PolynomialMeanCorrector:
    """Corrects Huber/median regression bias to conditional expectation via cubic polynomial."""

    def __init__(self, degree: int = 3) -> None:
        self.degree = degree
        self.coeffs: np.ndarray | None = None

    def fit(self, y_pred: np.ndarray, y_true: np.ndarray) -> "PolynomialMeanCorrector":
        pred_clipped = np.clip(y_pred, 1e-4, None)
        self.coeffs = np.polyfit(pred_clipped, y_true, deg=self.degree)
        return self

    def predict(self, y_pred: np.ndarray) -> np.ndarray:
        if self.coeffs is None:
            return y_pred
        pred_clipped = np.clip(y_pred, 1e-4, None)
        corrected = np.polyval(self.coeffs, pred_clipped)
        return np.clip(corrected, 0.0, None)


class ExactL2TwoPartHurdle:
    """Exact L2 Two-Part Hurdle Regressor for CLV Fuel & Nonfuel."""

    def __init__(
        self,
        target_name: str = "CLV_fuel",
        seed: int = 42,
        n_jobs: int = -1,
        huber_alpha: float = 0.85,
        winsor_pct: float = 99.5,
    ) -> None:
        self.target_name = target_name
        self.seed = seed
        self.n_jobs = n_jobs
        self.huber_alpha = huber_alpha
        self.winsor_pct = winsor_pct

        # Stage 1: Classifiers
        self.clf_lgb: LGBMClassifier | None = None
        self.clf_cb: CatBoostClassifier | None = None
        self.calibrator: IsotonicRegression | None = None

        # Stage 2: Positive Regressors
        self.reg_lgb_l2: LGBMRegressor | None = None
        self.reg_cb_l2: CatBoostRegressor | None = None
        self.reg_lgb_huber: LGBMRegressor | None = None
        self.corrector_huber: PolynomialMeanCorrector | None = None

        # Direct Single-Stage Regressor
        self.reg_direct_lgb: LGBMRegressor | None = None

        # Stage 3: Meta-Regressor
        self.meta_reg: RidgeCV | None = None

    def _prepare_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Sanitizes feature matrix for tree models."""
        X_clean = X.copy()
        cols_to_drop = [c for c in ["ID", "cutoff", "CLV_fuel", "CLV_nonfuel", "Opportunity"] if c in X_clean.columns]
        if cols_to_drop:
            X_clean = X_clean.drop(columns=cols_to_drop)
        num_cols = X_clean.select_dtypes(include=[np.number]).columns
        return X_clean[num_cols].fillna(0.0)

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
        is_summer: np.ndarray | None = None,
    ) -> "ExactL2TwoPartHurdle":
        """Fits 3-stage hurdle model with cross-validated calibration."""
        X_df = self._prepare_features(X)
        y_arr = np.asarray(y, dtype=np.float64)
        N = len(y_arr)

        if sample_weight is None:
            sample_weight = np.ones(N, dtype=np.float64)
        if is_summer is None:
            is_summer = np.zeros(N, dtype=np.float64)

        y_binary = (y_arr > 0).astype(int)
        pos_mask = y_arr > 0

        logger.info("[%s] Fitting Stage 1 (Activity Hurdles)...", self.target_name)
        kf = KFold(n_splits=5, shuffle=True, random_state=self.seed)
        oof_p = np.zeros(N, dtype=np.float64)

        for train_idx, val_idx in kf.split(X_df):
            clf = LGBMClassifier(
                n_estimators=250,
                learning_rate=0.05,
                max_depth=6,
                num_leaves=31,
                min_child_samples=50,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.seed,
                n_jobs=self.n_jobs,
                verbose=-1,
            )
            clf.fit(X_df.iloc[train_idx], y_binary[train_idx], sample_weight=sample_weight[train_idx])
            oof_p[val_idx] = clf.predict_proba(X_df.iloc[val_idx])[:, 1]

        self.calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self.calibrator.fit(oof_p, y_binary, sample_weight=sample_weight)

        self.clf_lgb = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            min_child_samples=50,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.seed,
            n_jobs=self.n_jobs,
            verbose=-1,
        )
        self.clf_lgb.fit(X_df, y_binary, sample_weight=sample_weight)

        self.clf_cb = CatBoostClassifier(
            iterations=300,
            learning_rate=0.06,
            depth=6,
            random_seed=self.seed,
            thread_count=self.n_jobs if self.n_jobs > 0 else 4,
            verbose=False,
        )
        self.clf_cb.fit(X_df, y_binary, sample_weight=sample_weight)

        logger.info("[%s] Fitting Stage 2 (Positive Conditional Expectation)...", self.target_name)
        X_pos = X_df.iloc[pos_mask]
        y_pos = y_arr[pos_mask]
        sw_pos = sample_weight[pos_mask]

        high_cap = np.percentile(y_pos, self.winsor_pct)
        y_pos_winsor = np.clip(y_pos, 0.0, high_cap)

        self.reg_lgb_l2 = LGBMRegressor(
            n_estimators=350,
            learning_rate=0.04,
            max_depth=6,
            num_leaves=31,
            reg_lambda=20.0,
            min_child_samples=50,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.seed,
            n_jobs=self.n_jobs,
            verbose=-1,
        )
        self.reg_lgb_l2.fit(X_pos, y_pos_winsor, sample_weight=sw_pos)

        self.reg_cb_l2 = CatBoostRegressor(
            iterations=350,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=15.0,
            random_seed=self.seed,
            thread_count=self.n_jobs if self.n_jobs > 0 else 4,
            verbose=False,
        )
        self.reg_cb_l2.fit(X_pos, y_pos_winsor, sample_weight=sw_pos)

        kf_pos = KFold(n_splits=5, shuffle=True, random_state=self.seed)
        oof_huber = np.zeros(len(y_pos), dtype=np.float64)

        for tr_idx, val_idx in kf_pos.split(X_pos):
            reg_h = LGBMRegressor(
                objective="huber",
                alpha=self.huber_alpha,
                n_estimators=300,
                learning_rate=0.04,
                max_depth=6,
                num_leaves=31,
                reg_lambda=15.0,
                min_child_samples=50,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.seed,
                n_jobs=self.n_jobs,
                verbose=-1,
            )
            reg_h.fit(X_pos.iloc[tr_idx], y_pos[tr_idx], sample_weight=sw_pos[tr_idx])
            oof_huber[val_idx] = reg_h.predict(X_pos.iloc[val_idx])

        self.corrector_huber = PolynomialMeanCorrector(degree=3)
        self.corrector_huber.fit(oof_huber, y_pos)

        self.reg_lgb_huber = LGBMRegressor(
            objective="huber",
            alpha=self.huber_alpha,
            n_estimators=350,
            learning_rate=0.04,
            max_depth=6,
            num_leaves=31,
            reg_lambda=15.0,
            min_child_samples=50,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.seed,
            n_jobs=self.n_jobs,
            verbose=-1,
        )
        self.reg_lgb_huber.fit(X_pos, y_pos, sample_weight=sw_pos)

        logger.info("[%s] Fitting Direct Single-Stage Regressor...", self.target_name)
        self.reg_direct_lgb = LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            reg_lambda=25.0,
            min_child_samples=60,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.seed,
            n_jobs=self.n_jobs,
            verbose=-1,
        )
        self.reg_direct_lgb.fit(X_df, y_arr, sample_weight=sample_weight)

        logger.info("[%s] Fitting Stage 3 (Residual Meta-Regression)...", self.target_name)
        meta_X = self._build_meta_features(X_df, is_summer)
        self.meta_reg = RidgeCV(alphas=np.logspace(-2, 4, 25), cv=5)
        self.meta_reg.fit(meta_X, y_arr, sample_weight=sample_weight)
        logger.info("[%s] Meta-regressor chosen alpha: %.4f", self.target_name, self.meta_reg.alpha_)

        return self

    def _build_meta_features(self, X_df: pd.DataFrame, is_summer: np.ndarray) -> np.ndarray:
        """Constructs meta-features from Stage 1, Stage 2, direct regressor, and physics baselines."""
        p_lgb = self.clf_lgb.predict_proba(X_df)[:, 1]
        p_cb = self.clf_cb.predict_proba(X_df)[:, 1]
        p_raw = 0.55 * p_lgb + 0.45 * p_cb
        p_calib = self.calibrator.predict(p_raw)

        mu_lgb = np.clip(self.reg_lgb_l2.predict(X_df), 0.0, None)
        mu_cb = np.clip(self.reg_cb_l2.predict(X_df), 0.0, None)
        mu_l2 = 0.55 * mu_lgb + 0.45 * mu_cb

        mu_huber_raw = np.clip(self.reg_lgb_huber.predict(X_df), 0.0, None)
        mu_huber_corr = self.corrector_huber.predict(mu_huber_raw)
        mu_pos = 0.5 * mu_l2 + 0.5 * mu_huber_corr

        y_hurdle = p_calib * mu_pos
        y_direct = np.clip(self.reg_direct_lgb.predict(X_df), 0.0, None)

        if self.target_name == "CLV_fuel":
            base_m3 = X_df["m3_fuel_litres"].values if "m3_fuel_litres" in X_df.columns else np.zeros(len(X_df))
            rec = X_df["days_since_last_fuel"].values if "days_since_last_fuel" in X_df.columns else np.full(len(X_df), 999.0)
        else:
            base_m3 = X_df["m3_nonfuel_rand"].values if "m3_nonfuel_rand" in X_df.columns else np.zeros(len(X_df))
            rec = X_df["days_since_last_nonfuel"].values if "days_since_last_nonfuel" in X_df.columns else np.full(len(X_df), 999.0)

        base_log_m3 = np.log1p(np.maximum(0.0, base_m3))
        rec_inv = 1.0 / (1.0 + np.clip(rec, 0.0, 365.0))

        feats = np.column_stack([
            p_calib,
            mu_pos,
            y_hurdle,
            y_direct,
            base_log_m3,
            rec_inv,
            is_summer,
            y_hurdle * is_summer,
            y_direct * is_summer,
        ])
        return feats

    def predict(self, X: pd.DataFrame, is_summer: np.ndarray | None = None) -> np.ndarray:
        """Generates unconditional expected target predictions."""
        X_df = self._prepare_features(X)
        N = len(X_df)
        if is_summer is None:
            is_summer = np.zeros(N, dtype=np.float64)

        meta_X = self._build_meta_features(X_df, is_summer)
        pred = self.meta_reg.predict(meta_X)

        return np.clip(pred, 0.0, None)
