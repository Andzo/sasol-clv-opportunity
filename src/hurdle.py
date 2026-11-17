"""Two-part hurdle model for zero-inflated target prediction.

Deconstructs zero-inflated customer spend into:
1. P(Y > 0 | X): binary store attachment classification.
2. E[Y | X, Y > 0]: conditional spend regression trained exclusively on positive instances.

The expected unconditional target is E[Y | X] = P(Y > 0 | X) * E[Y | X, Y > 0].
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

import scipy.special as sp
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from src.models import StoreAttachmentClassifier, make_regressor


def _align_features(model: Any, X: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
    """Safely aligns features in X to model.feature_names_in_ if X is a superset DataFrame."""
    if isinstance(X, pd.DataFrame) and hasattr(model, "feature_names_in_"):
        if set(model.feature_names_in_).issubset(X.columns) and len(X.columns) != len(model.feature_names_in_):
            return X[model.feature_names_in_]
    return X


def _compute_imr(p: np.ndarray) -> np.ndarray:
    """Heckman Inverse Mills Ratio: lambda(p) = phi(Phi^-1(p)) / p."""
    p_clip = np.clip(np.asarray(p, dtype=np.float64), 0.01, 0.99)
    u = sp.ndtri(p_clip)
    phi = np.exp(-0.5 * u ** 2) / np.sqrt(2.0 * np.pi)
    return np.clip(phi / p_clip, 0.0, 5.0)


def _with_imr(X: pd.DataFrame | np.ndarray, p: np.ndarray) -> pd.DataFrame:
    """Appends Heckman Inverse Mills Ratio feature column to eliminate selection bias."""
    imr = _compute_imr(p)
    if isinstance(X, pd.DataFrame):
        X_out = X.copy()
        X_out["_imr"] = imr
        return X_out
    else:
        return np.column_stack([X, imr])


def _extract_anchor_matrix(X: pd.DataFrame | np.ndarray, target_type: str = "fuel") -> np.ndarray:
    """Extracts continuous log-transformed and z-space baseline predictors for the parametric anchor model."""
    if isinstance(X, pd.DataFrame):
        if target_type == "fuel":
            raw_candidates = [
                "m1_fuel_litres", "m3_fuel_litres", "q2_fuel_litres", "m6_fuel_litres",
                "btyd_fuel_exp_baskets_m3", "mundlak_perm_fuel_litres",
            ]
            z_candidates = ["z_anchor_fuel_mean", "lag1_clv_fuel", "proj_m1_clv_fuel"]
        else:
            raw_candidates = [
                "m1_nonfuel_rand", "m3_nonfuel_rand", "q2_nonfuel_rand", "m6_nonfuel_rand",
                "btyd_exp_clv_spend_m3", "mundlak_perm_nf_rand",
            ]
            z_candidates = ["z_anchor_nf_mean", "lag1_clv_nonfuel", "proj_m1_clv_nonfuel"]

        parts = []
        present_raw = [c for c in raw_candidates if c in X.columns]
        if present_raw:
            vals = np.nan_to_num(np.asarray(X[present_raw].to_numpy(), dtype="float64"), nan=0.0)
            parts.append(np.log1p(np.maximum(vals, 0.0)))
        present_z = [c for c in z_candidates if c in X.columns]
        if present_z:
            z_vals = np.nan_to_num(np.asarray(X[present_z].to_numpy(), dtype="float64"), nan=0.0)
            parts.append(np.maximum(z_vals, 0.0))
        if parts:
            return np.column_stack(parts)
        vals = np.asarray(X.iloc[:, :min(6, X.shape[1])].to_numpy(), dtype="float64")
        return np.log1p(np.maximum(np.nan_to_num(vals, nan=0.0), 0.0))
    else:
        arr = np.asarray(X, dtype="float64")
        if arr.ndim == 1:
            arr.reshape(-1, 1)
        sub = arr[:, :min(6, arr.shape[1])]
        return np.log1p(np.maximum(np.nan_to_num(sub, nan=0.0), 0.0))


class TwoPartNonfuelHurdle:
    """Two-part hurdle model for nonfuel customer lifetime value."""

    def __init__(self, seed: int = 42, n_jobs: int = -1, use_gpu: bool = False) -> None:
        self.seed = seed
        self.n_jobs = n_jobs
        self.use_gpu = use_gpu
        self.classifier = StoreAttachmentClassifier(seed=seed, n_jobs=n_jobs, use_gpu=use_gpu)
        self.regressor = make_regressor("lgbm_huber", seed=seed, n_jobs=n_jobs, use_gpu=use_gpu)

    def fit(self, X: pd.DataFrame, y: np.ndarray, sample_weight: np.ndarray | None = None) -> "TwoPartNonfuelHurdle":
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = list(X.columns)
        y_arr = np.asarray(y, dtype="float64")
        pos_mask = y_arr > 0
        self.classifier.fit(X, y_arr, sample_weight=sample_weight)
        X_pos = X.iloc[pos_mask] if isinstance(X, pd.DataFrame) else X[pos_mask]
        y_pos = y_arr[pos_mask]
        if sample_weight is not None:
            sw_pos = np.asarray(sample_weight, dtype="float64")[pos_mask]
            self.regressor.fit(X_pos, y_pos, sample_weight=sw_pos)
        else:
            self.regressor.fit(X_pos, y_pos)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X = _align_features(self, X)
        p_store = self.classifier.predict_proba(X)
        pred_amount = np.clip(self.regressor.predict(X), 0.0, None)
        return p_store * pred_amount


class TwoPartFuelHurdle:
    """Two-part hurdle model for fuel customer lifetime value."""

    def __init__(self, seed: int = 42, n_jobs: int = -1, use_gpu: bool = False) -> None:
        self.seed = seed
        self.n_jobs = n_jobs
        self.use_gpu = use_gpu
        self.classifier = StoreAttachmentClassifier(seed=seed, n_jobs=n_jobs, use_gpu=use_gpu)
        self.regressor = make_regressor("lgbm_huber", seed=seed, n_jobs=n_jobs, use_gpu=use_gpu)

    def fit(self, X: pd.DataFrame, y: np.ndarray, sample_weight: np.ndarray | None = None) -> "TwoPartFuelHurdle":
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = list(X.columns)
        y_arr = np.asarray(y, dtype="float64")
        pos_mask = y_arr > 0
        self.classifier.fit(X, y_arr, sample_weight=sample_weight)
        X_pos = X.iloc[pos_mask] if isinstance(X, pd.DataFrame) else X[pos_mask]
        y_pos = y_arr[pos_mask]
        if sample_weight is not None:
            sw_pos = np.asarray(sample_weight, dtype="float64")[pos_mask]
            self.regressor.fit(X_pos, y_pos, sample_weight=sw_pos)
        else:
            self.regressor.fit(X_pos, y_pos)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X = _align_features(self, X)
        p_active = self.classifier.predict_proba(X)
        pred_amount = np.clip(self.regressor.predict(X), 0.0, None)
        return p_active * pred_amount


class BaggedTwoPartFuelHurdle:
    """Multi-seed multi-model bagged two-part hurdle model for fuel CLV."""

    def __init__(
        self,
        seeds: tuple[int, ...] = (42, 100, 2024, 777, 314),
        n_jobs: int = -1,
        use_gpu: bool = False,
        huber_alpha: float = 1.35,
        power_shrinkage: float = 1.0,
        zero_gate_threshold: float = 0.0,
    ) -> None:
        self.seeds = seeds
        self.n_jobs = n_jobs
        self.use_gpu = use_gpu
        self.huber_alpha = huber_alpha
        self.power_shrinkage = power_shrinkage
        self.zero_gate_threshold = zero_gate_threshold
        self.anchor_model = Ridge(alpha=50.0)
        self.propensity_scaler = StandardScaler()
        self.propensity_model = LogisticRegression(C=1.0, max_iter=200, random_state=seeds[0])
        self.classifier = StoreAttachmentClassifier(seed=seeds[0], n_jobs=n_jobs, use_gpu=use_gpu)
        self.regressors_mse: list[Any] = []
        self.regressors_lgb: list[Any] = []
        self.regressors_cb: list[Any] = []
        for s in seeds:
            self.regressors_mse.append(
                make_regressor("lgbm_mse", seed=s, n_jobs=n_jobs, use_gpu=use_gpu)
            )
            self.regressors_lgb.append(
                make_regressor("lgbm_huber", seed=s, n_jobs=n_jobs, use_gpu=use_gpu, huber_alpha=huber_alpha)
            )
            self.regressors_cb.append(make_regressor("catboost", seed=s, n_jobs=n_jobs, use_gpu=use_gpu))

    def fit(self, X: pd.DataFrame, y: np.ndarray, sample_weight: np.ndarray | None = None) -> "BaggedTwoPartFuelHurdle":
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = list(X.columns)
        y_arr = np.asarray(y, dtype="float64")
        pos_mask = y_arr > 0
        self.classifier.fit(X, y_arr, sample_weight=sample_weight)
        X_pos = X.iloc[pos_mask] if isinstance(X, pd.DataFrame) else X[pos_mask]
        y_pos = y_arr[pos_mask]
        sw_pos = np.asarray(sample_weight, dtype="float64")[pos_mask] if sample_weight is not None else None

        A_all = _extract_anchor_matrix(X, "fuel")
        A_scaled = self.propensity_scaler.fit_transform(A_all)
        y_bin = (y_arr > 0).astype(int)
        if len(np.unique(y_bin)) >= 2:
            if sample_weight is not None:
                self.propensity_model.fit(A_scaled, y_bin, sample_weight=np.asarray(sample_weight, dtype="float64"))
            else:
                self.propensity_model.fit(A_scaled, y_bin)

        A_pos = _extract_anchor_matrix(X_pos, "fuel")
        if hasattr(self.propensity_model, "classes_") and len(self.propensity_model.classes_) >= 2:
            p_pos = self.propensity_model.predict_proba(self.propensity_scaler.transform(A_pos))[:, 1]
        else:
            p_pos = np.full(len(X_pos), 0.5)
        X_pos_imr = _with_imr(X_pos, p_pos)

        if len(y_pos) >= 5:
            self.anchor_model.fit(A_pos, y_pos, sample_weight=sw_pos)
            z_anchor_pos = self.anchor_model.predict(A_pos)
        else:
            z_anchor_pos = np.zeros(len(y_pos), dtype="float64")
        y_innov_pos = y_pos - z_anchor_pos

        for reg_mse, reg_lgb, reg_cb in zip(self.regressors_mse, self.regressors_lgb, self.regressors_cb):
            if sw_pos is not None:
                reg_mse.fit(X_pos_imr, y_innov_pos, sample_weight=sw_pos)
                reg_lgb.fit(X_pos_imr, y_innov_pos, sample_weight=sw_pos)
                reg_cb.fit(X_pos_imr, y_innov_pos, sample_weight=sw_pos)
            else:
                reg_mse.fit(X_pos_imr, y_innov_pos)
                reg_lgb.fit(X_pos_imr, y_innov_pos)
                reg_cb.fit(X_pos_imr, y_innov_pos)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X = _align_features(self, X)
        p_active = self.classifier.predict_proba(X)
        if self.power_shrinkage != 1.0:
            p_active = p_active ** self.power_shrinkage
        if self.zero_gate_threshold > 0.0:
            p_active = np.where(p_active < self.zero_gate_threshold, 0.0, p_active)

        A = _extract_anchor_matrix(X, "fuel")
        z_anchor = self.anchor_model.predict(A) if hasattr(self.anchor_model, "coef_") else 0.0
        if hasattr(self.propensity_model, "classes_") and len(self.propensity_model.classes_) >= 2:
            p_imr = self.propensity_model.predict_proba(self.propensity_scaler.transform(A))[:, 1]
        else:
            p_imr = p_active
        X_imr = _with_imr(X, p_imr)

        pred_innov = np.zeros(len(X), dtype="float64")
        n_bags = len(self.seeds)
        for reg_mse, reg_lgb, reg_cb in zip(self.regressors_mse, self.regressors_lgb, self.regressors_cb):
            p_mse = reg_mse.predict(X_imr)
            p_lgb = reg_lgb.predict(X_imr)
            p_cb = reg_cb.predict(X_imr)
            pred_innov += (0.35 * p_mse + 0.30 * p_lgb + 0.35 * p_cb) / n_bags
        pred_pos = np.clip(z_anchor + pred_innov, 0.0, None)
        return p_active * pred_pos


class BaggedTwoPartNonfuelHurdle:
    """Multi-seed multi-model bagged two-part hurdle model for nonfuel CLV."""

    def __init__(
        self,
        seeds: tuple[int, ...] = (42, 100, 2024, 777, 314),
        n_jobs: int = -1,
        use_gpu: bool = False,
        huber_alpha: float = 1.35,
        power_shrinkage: float = 1.0,
        zero_gate_threshold: float = 0.0,
    ) -> None:
        self.seeds = seeds
        self.n_jobs = n_jobs
        self.use_gpu = use_gpu
        self.huber_alpha = huber_alpha
        self.power_shrinkage = power_shrinkage
        self.zero_gate_threshold = zero_gate_threshold
        self.anchor_model = Ridge(alpha=50.0)
        self.propensity_scaler = StandardScaler()
        self.propensity_model = LogisticRegression(C=1.0, max_iter=200, random_state=seeds[0])
        self.classifier = StoreAttachmentClassifier(seed=seeds[0], n_jobs=n_jobs, use_gpu=use_gpu)
        self.regressors_mse: list[Any] = []
        self.regressors_lgb: list[Any] = []
        self.regressors_cb: list[Any] = []
        for s in seeds:
            self.regressors_mse.append(
                make_regressor("lgbm_mse", seed=s, n_jobs=n_jobs, use_gpu=use_gpu)
            )
            self.regressors_lgb.append(
                make_regressor("lgbm_huber", seed=s, n_jobs=n_jobs, use_gpu=use_gpu, huber_alpha=huber_alpha)
            )
            self.regressors_cb.append(make_regressor("catboost", seed=s, n_jobs=n_jobs, use_gpu=use_gpu))

    def fit(self, X: pd.DataFrame, y: np.ndarray, sample_weight: np.ndarray | None = None) -> "BaggedTwoPartNonfuelHurdle":
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = list(X.columns)
        y_arr = np.asarray(y, dtype="float64")
        pos_mask = y_arr > 0
        self.classifier.fit(X, y_arr, sample_weight=sample_weight)
        X_pos = X.iloc[pos_mask] if isinstance(X, pd.DataFrame) else X[pos_mask]
        y_pos = y_arr[pos_mask]
        sw_pos = np.asarray(sample_weight, dtype="float64")[pos_mask] if sample_weight is not None else None

        A_all = _extract_anchor_matrix(X, "nonfuel")
        A_scaled = self.propensity_scaler.fit_transform(A_all)
        y_bin = (y_arr > 0).astype(int)
        if len(np.unique(y_bin)) >= 2:
            if sample_weight is not None:
                self.propensity_model.fit(A_scaled, y_bin, sample_weight=np.asarray(sample_weight, dtype="float64"))
            else:
                self.propensity_model.fit(A_scaled, y_bin)

        A_pos = _extract_anchor_matrix(X_pos, "nonfuel")
        if hasattr(self.propensity_model, "classes_") and len(self.propensity_model.classes_) >= 2:
            p_pos = self.propensity_model.predict_proba(self.propensity_scaler.transform(A_pos))[:, 1]
        else:
            p_pos = np.full(len(X_pos), 0.5)
        X_pos_imr = _with_imr(X_pos, p_pos)

        if len(y_pos) >= 5:
            self.anchor_model.fit(A_pos, y_pos, sample_weight=sw_pos)
            z_anchor_pos = self.anchor_model.predict(A_pos)
        else:
            z_anchor_pos = np.zeros(len(y_pos), dtype="float64")
        y_innov_pos = y_pos - z_anchor_pos

        for reg_mse, reg_lgb, reg_cb in zip(self.regressors_mse, self.regressors_lgb, self.regressors_cb):
            if sw_pos is not None:
                reg_mse.fit(X_pos_imr, y_innov_pos, sample_weight=sw_pos)
                reg_lgb.fit(X_pos_imr, y_innov_pos, sample_weight=sw_pos)
                reg_cb.fit(X_pos_imr, y_innov_pos, sample_weight=sw_pos)
            else:
                reg_mse.fit(X_pos_imr, y_innov_pos)
                reg_lgb.fit(X_pos_imr, y_innov_pos)
                reg_cb.fit(X_pos_imr, y_innov_pos)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X = _align_features(self, X)
        p_store = self.classifier.predict_proba(X)
        if self.power_shrinkage != 1.0:
            p_store = p_store ** self.power_shrinkage
        if self.zero_gate_threshold > 0.0:
            p_store = np.where(p_store < self.zero_gate_threshold, 0.0, p_store)

        A = _extract_anchor_matrix(X, "nonfuel")
        z_anchor = self.anchor_model.predict(A) if hasattr(self.anchor_model, "coef_") else 0.0
        if hasattr(self.propensity_model, "classes_") and len(self.propensity_model.classes_) >= 2:
            p_imr = self.propensity_model.predict_proba(self.propensity_scaler.transform(A))[:, 1]
        else:
            p_imr = p_store
        X_imr = _with_imr(X, p_imr)

        pred_innov = np.zeros(len(X), dtype="float64")
        n_bags = len(self.seeds)
        for reg_mse, reg_lgb, reg_cb in zip(self.regressors_mse, self.regressors_lgb, self.regressors_cb):
            p_mse = reg_mse.predict(X_imr)
            p_lgb = reg_lgb.predict(X_imr)
            p_cb = reg_cb.predict(X_imr)
            pred_innov += (0.35 * p_mse + 0.30 * p_lgb + 0.35 * p_cb) / n_bags
        pred_pos = np.clip(z_anchor + pred_innov, 0.0, None)
        return p_store * pred_pos


class MultiQuantileTwoPartHurdle:
    """Multi-Quantile Simpson Quadrature Two-Part Hurdle.

    Deconstructs spend into:
    Stage 1: P(Z > 0 | X) using StoreAttachmentClassifier.
    Stage 2: Conditional quantile regressors across tau in (0.10, 0.25, 0.50, 0.75, 0.90)
             plus a winsorized L2 regressor and Heckman IMR control feature.
    Integrates positive spend via Simpson/trapezoidal quadrature and Stage 2.5
    heteroscedastic skewness calibrator:
        mu_exact = c0 + c1*mu_quad + c2*mu_L2 + c3*sigma + c4*tau_skew.
    Exports conditional spread sigma_pos = max(0.14, (Q_90 - Q_10) / 2.5631).
    """

    QUANTILES: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 0.90)
    QUAD_WEIGHTS: tuple[float, ...] = (0.15, 0.225, 0.25, 0.225, 0.15)

    def __init__(
        self,
        seed: int = 42,
        n_jobs: int = -1,
        use_gpu: bool = False,
        target_type: str = "fuel",
    ) -> None:
        self.seed = seed
        self.n_jobs = n_jobs
        self.use_gpu = use_gpu
        self.target_type = target_type
        self.anchor_model = Ridge(alpha=50.0)
        self.propensity_scaler = StandardScaler()
        self.propensity_model = LogisticRegression(C=1.0, max_iter=200, random_state=seed)
        self.classifier = StoreAttachmentClassifier(seed=seed, n_jobs=n_jobs, use_gpu=use_gpu)

        from lightgbm import LGBMRegressor
        from catboost import CatBoostRegressor

        self.lgb_quantiles: list[Any] = []
        self.cb_quantiles: list[Any] = []
        for tau in self.QUANTILES:
            self.lgb_quantiles.append(
                LGBMRegressor(
                    objective="quantile", alpha=tau, n_estimators=150, learning_rate=0.04,
                    num_leaves=28, colsample_bytree=0.6, reg_lambda=8.0,
                    random_state=seed, n_jobs=n_jobs, deterministic=True, force_col_wise=True, verbose=-1,
                )
            )
            cb_k: dict[str, Any] = {
                "iterations": 150, "learning_rate": 0.05, "depth": 5, "l2_leaf_reg": 8.0,
                "loss_function": f"Quantile:alpha={tau}", "random_seed": seed, "verbose": 0, "allow_writing_files": False,
            }
            if use_gpu:
                cb_k["task_type"] = "GPU"
            else:
                cb_k["thread_count"] = n_jobs
            self.cb_quantiles.append(CatBoostRegressor(**cb_k))

        self.l2_regressor = LGBMRegressor(
            objective="regression", n_estimators=180, learning_rate=0.04,
            num_leaves=28, colsample_bytree=0.6, reg_lambda=8.0,
            random_state=seed, n_jobs=n_jobs, deterministic=True, force_col_wise=True, verbose=-1,
        )
        self.stage25_calibrator = Ridge(alpha=10.0)

    def fit(self, X: pd.DataFrame, y: np.ndarray, sample_weight: np.ndarray | None = None) -> "MultiQuantileTwoPartHurdle":
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = list(X.columns)
        y_arr = np.asarray(y, dtype="float64")
        self.classifier.fit(X, y_arr, sample_weight=sample_weight)

        pos_mask = y_arr > 0
        X_pos = X.iloc[pos_mask] if isinstance(X, pd.DataFrame) else X[pos_mask]
        y_pos = y_arr[pos_mask]
        sw_pos = np.asarray(sample_weight, dtype="float64")[pos_mask] if sample_weight is not None else None

        A_all = _extract_anchor_matrix(X, self.target_type)
        A_scaled = self.propensity_scaler.fit_transform(A_all)
        y_bin = (y_arr > 0).astype(int)
        if len(np.unique(y_bin)) >= 2:
            if sample_weight is not None:
                self.propensity_model.fit(A_scaled, y_bin, sample_weight=np.asarray(sample_weight, dtype="float64"))
            else:
                self.propensity_model.fit(A_scaled, y_bin)

        A_pos = _extract_anchor_matrix(X_pos, self.target_type)
        if hasattr(self.propensity_model, "classes_") and len(self.propensity_model.classes_) >= 2:
            p_pos = self.propensity_model.predict_proba(self.propensity_scaler.transform(A_pos))[:, 1]
        else:
            p_pos = np.full(len(X_pos), 0.5)
        X_pos_imr = _with_imr(X_pos, p_pos)

        if len(y_pos) >= 5:
            self.anchor_model.fit(A_pos, y_pos, sample_weight=sw_pos)
            z_anchor_pos = self.anchor_model.predict(A_pos)
        else:
            z_anchor_pos = np.zeros(len(y_pos), dtype="float64")
        y_innov_pos = y_pos - z_anchor_pos

        if len(y_pos) > 0:
            # Winsorize positive innovation at 99.5th percentile for L2 regressor
            cap = float(np.percentile(y_innov_pos, 99.5))
            y_innov_win = np.minimum(y_innov_pos, cap)
            if sw_pos is not None:
                self.l2_regressor.fit(X_pos_imr, y_innov_win, sample_weight=sw_pos)
            else:
                self.l2_regressor.fit(X_pos_imr, y_innov_win)

            for lgb_q, cb_q in zip(self.lgb_quantiles, self.cb_quantiles):
                if sw_pos is not None:
                    lgb_q.fit(X_pos_imr, y_innov_pos, sample_weight=sw_pos)
                    cb_q.fit(X_pos_imr, y_innov_pos, sample_weight=sw_pos)
                else:
                    lgb_q.fit(X_pos_imr, y_innov_pos)
                    cb_q.fit(X_pos_imr, y_innov_pos)

            # Fit Stage 2.5 heteroscedastic calibrator on in-sample predictions
            mu_l2_innov_pos = self.l2_regressor.predict(X_pos_imr)
            q_preds_pos: list[np.ndarray] = []
            for lgb_q, cb_q in zip(self.lgb_quantiles, self.cb_quantiles):
                p_lgb = lgb_q.predict(X_pos_imr)
                p_cb = cb_q.predict(X_pos_imr)
                q_preds_pos.append(np.clip(z_anchor_pos + (0.55 * p_lgb + 0.45 * p_cb), 0.0, None))

            mu_quad_pos = sum(w * q for w, q in zip(self.QUAD_WEIGHTS, q_preds_pos))
            mu_l2_full_pos = np.clip(z_anchor_pos + mu_l2_innov_pos, 0.0, None)
            sigma_pos_fit = np.maximum(0.14, (q_preds_pos[-1] - q_preds_pos[0]) / 2.5631)
            tau_skew_pos = (q_preds_pos[-1] + q_preds_pos[0]) / 2.0 - q_preds_pos[2]
            M_pos = np.column_stack([mu_l2_full_pos, mu_quad_pos, sigma_pos_fit, np.maximum(0.0, tau_skew_pos)])
            if len(y_pos) >= 5:
                self.stage25_calibrator.fit(M_pos, y_pos, sample_weight=sw_pos)
            else:
                self.stage25_calibrator.fit(M_pos, y_pos)

        return self

    def predict_proba_active(self, X: pd.DataFrame) -> np.ndarray:
        X = _align_features(self, X)
        return self.classifier.predict_proba(X)

    def predict_conditional_pos(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Returns conditional mean mu_pos and spread sigma_pos."""
        X = _align_features(self, X)
        A = _extract_anchor_matrix(X, self.target_type)
        if hasattr(self.propensity_model, "classes_") and len(self.propensity_model.classes_) >= 2:
            p_imr = self.propensity_model.predict_proba(self.propensity_scaler.transform(A))[:, 1]
        else:
            p_imr = self.predict_proba_active(X)
        X_imr = _with_imr(X, p_imr)

        z_anchor = self.anchor_model.predict(A) if hasattr(self.anchor_model, "coef_") else 0.0
        mu_l2_innov = self.l2_regressor.predict(X_imr)
        q_preds: list[np.ndarray] = []
        for lgb_q, cb_q in zip(self.lgb_quantiles, self.cb_quantiles):
            p_lgb = lgb_q.predict(X_imr)
            p_cb = cb_q.predict(X_imr)
            q_preds.append(np.clip(z_anchor + (0.55 * p_lgb + 0.45 * p_cb), 0.0, None))

        mu_quad = sum(w * q for w, q in zip(self.QUAD_WEIGHTS, q_preds))
        mu_l2_full = np.clip(z_anchor + mu_l2_innov, 0.0, None)
        sigma_pos = np.maximum(0.14, (q_preds[-1] - q_preds[0]) / 2.5631)
        tau_skew = (q_preds[-1] + q_preds[0]) / 2.0 - q_preds[2]

        M = np.column_stack([mu_l2_full, mu_quad, sigma_pos, np.maximum(0.0, tau_skew)])
        if hasattr(self.stage25_calibrator, "coef_"):
            mu_pos = np.clip(self.stage25_calibrator.predict(M), 0.0, None)
        else:
            mu_pos = np.clip(0.45 * mu_l2_full + 0.55 * mu_quad + 0.15 * np.maximum(0.0, tau_skew), 0.0, None)
        return mu_pos, sigma_pos

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X = _align_features(self, X)
        p_act = self.predict_proba_active(X)
        mu_pos, _ = self.predict_conditional_pos(X)
        return p_act * mu_pos


class PhysicalBasketTicketRegressor:
    """Raw-Scale Tweedie Physical Spend Regressor with Out-of-Fold Variance Estimation."""

    def __init__(
        self,
        scale: float = 2.352825386185113,
        seed: int = 42,
        n_jobs: int = -1,
        use_gpu: bool = False,
        target_type: str = "fuel",
    ) -> None:
        self.scale = scale
        self.seed = seed
        self.n_jobs = n_jobs
        self.use_gpu = use_gpu
        self.target_type = target_type
        self.classifier = StoreAttachmentClassifier(seed=seed, n_jobs=n_jobs, use_gpu=use_gpu)

        from lightgbm import LGBMRegressor

        # Tweedie regressor on winsorized raw physical spend
        self.basket_regressor = LGBMRegressor(
            objective="tweedie", tweedie_variance_power=1.35, n_estimators=180, learning_rate=0.035,
            num_leaves=28, colsample_bytree=0.6, reg_lambda=8.0,
            random_state=seed, n_jobs=n_jobs, deterministic=True, force_col_wise=True, verbose=-1,
        )
        # Variance regressor on squared out-of-fold residuals
        self.ticket_regressor = LGBMRegressor(
            objective="gamma", n_estimators=150, learning_rate=0.035,
            num_leaves=20, colsample_bytree=0.6, reg_lambda=12.0,
            random_state=seed, n_jobs=n_jobs, deterministic=True, force_col_wise=True, verbose=-1,
        )

    def fit(self, X: pd.DataFrame, y: np.ndarray, sample_weight: np.ndarray | None = None) -> "PhysicalBasketTicketRegressor":
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = list(X.columns)
        y_arr = np.asarray(y, dtype="float64")
        self.classifier.fit(X, y_arr, sample_weight=sample_weight)

        pos_mask = y_arr > 0
        X_pos = X.iloc[pos_mask] if isinstance(X, pd.DataFrame) else X[pos_mask]
        y_pos = y_arr[pos_mask]
        sw_pos = np.asarray(sample_weight, dtype="float64")[pos_mask] if sample_weight is not None else None

        if len(y_pos) > 0:
            y_raw = np.expm1(self.scale * y_pos)
            cap_raw = float(np.percentile(y_raw, 99.2))
            y_raw_capped = np.clip(y_raw, 0.0, cap_raw)

            if sw_pos is not None:
                self.basket_regressor.fit(X_pos, y_raw_capped, sample_weight=sw_pos)
            else:
                self.basket_regressor.fit(X_pos, y_raw_capped)

            # Estimate conditional variance from 3-fold OOF squared residuals
            from sklearn.model_selection import KFold
            oof_pred = np.zeros(len(y_pos), dtype="float64")
            n_splits = min(3, max(2, len(y_pos) // 10))
            if n_splits >= 2:
                kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
                from lightgbm import LGBMRegressor
                for tr_idx, val_idx in kf.split(X_pos):
                    X_tr = X_pos.iloc[tr_idx] if isinstance(X_pos, pd.DataFrame) else X_pos[tr_idx]
                    y_tr = y_raw_capped[tr_idx]
                    sw_tr = sw_pos[tr_idx] if sw_pos is not None else None
                    X_v = X_pos.iloc[val_idx] if isinstance(X_pos, pd.DataFrame) else X_pos[val_idx]
                    fold_model = LGBMRegressor(
                        objective="tweedie", tweedie_variance_power=1.35, n_estimators=100, learning_rate=0.04,
                        num_leaves=24, colsample_bytree=0.6, reg_lambda=8.0,
                        random_state=self.seed, n_jobs=self.n_jobs, deterministic=True, force_col_wise=True, verbose=-1,
                    )
                    fold_model.fit(X_tr, y_tr, sample_weight=sw_tr)
                    oof_pred[val_idx] = fold_model.predict(X_v)
            else:
                oof_pred = self.basket_regressor.predict(X_pos)

            sq_residuals = np.maximum((y_raw_capped - oof_pred) ** 2, 1e-4)
            if sw_pos is not None:
                self.ticket_regressor.fit(X_pos, sq_residuals, sample_weight=sw_pos)
            else:
                self.ticket_regressor.fit(X_pos, sq_residuals)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X = _align_features(self, X)
        p_act = self.classifier.predict_proba(X)
        y_raw_hat = np.clip(self.basket_regressor.predict(X), 0.0, None)
        sigma_raw_sq = np.clip(self.ticket_regressor.predict(X), 1e-4, None)
        denom = 1.0 + y_raw_hat
        jensen_corr = np.minimum(sigma_raw_sq / (2.0 * denom ** 2), 0.25 * np.log1p(y_raw_hat))
        z_pos = (1.0 / self.scale) * np.clip(np.log1p(y_raw_hat) - jensen_corr, 0.0, None)
        return p_act * z_pos

