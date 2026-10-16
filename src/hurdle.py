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

from src.models import StoreAttachmentClassifier, make_regressor


class TwoPartNonfuelHurdle:
    """Two-part hurdle model for nonfuel customer lifetime value."""

    def __init__(self, seed: int = 42, n_jobs: int = -1, use_gpu: bool = False) -> None:
        self.seed = seed
        self.n_jobs = n_jobs
        self.use_gpu = use_gpu
        self.classifier = StoreAttachmentClassifier(seed=seed, n_jobs=n_jobs, use_gpu=use_gpu)
        self.regressor = make_regressor("lgbm_huber", seed=seed, n_jobs=n_jobs, use_gpu=use_gpu)

    def fit(self, X: pd.DataFrame, y: np.ndarray, sample_weight: np.ndarray | None = None) -> "TwoPartNonfuelHurdle":
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
        p_active = self.classifier.predict_proba(X)
        pred_amount = np.clip(self.regressor.predict(X), 0.0, None)
        return p_active * pred_amount


class BaggedTwoPartFuelHurdle:
    """Multi-seed multi-model bagged two-part hurdle model for fuel CLV.

    Compresses stochastic decision-tree variance by bagging positive spenders
    across 5 distinct random seeds and dual model architectures (LightGBM Huber + CatBoost).
    """

    def __init__(
        self,
        seeds: tuple[int, ...] = (42, 100, 2024, 777, 314),
        n_jobs: int = -1,
        use_gpu: bool = False,
        huber_alpha: float = 0.85,
        power_shrinkage: float = 0.98,
        zero_gate_threshold: float = 0.0,
    ) -> None:
        self.seeds = seeds
        self.n_jobs = n_jobs
        self.use_gpu = use_gpu
        self.huber_alpha = huber_alpha
        self.power_shrinkage = power_shrinkage
        self.zero_gate_threshold = zero_gate_threshold
        self.classifier = StoreAttachmentClassifier(seed=seeds[0], n_jobs=n_jobs, use_gpu=use_gpu)
        self.regressors_lgb: list[Any] = []
        self.regressors_cb: list[Any] = []
        for s in seeds:
            self.regressors_lgb.append(
                make_regressor("lgbm_huber", seed=s, n_jobs=n_jobs, use_gpu=use_gpu, huber_alpha=huber_alpha)
            )
            self.regressors_cb.append(make_regressor("catboost", seed=s, n_jobs=n_jobs, use_gpu=use_gpu))

    def fit(self, X: pd.DataFrame, y: np.ndarray, sample_weight: np.ndarray | None = None) -> "BaggedTwoPartFuelHurdle":
        y_arr = np.asarray(y, dtype="float64")
        pos_mask = y_arr > 0
        self.classifier.fit(X, y_arr, sample_weight=sample_weight)
        X_pos = X.iloc[pos_mask] if isinstance(X, pd.DataFrame) else X[pos_mask]
        y_pos = y_arr[pos_mask]
        sw_pos = np.asarray(sample_weight, dtype="float64")[pos_mask] if sample_weight is not None else None
        for reg_lgb, reg_cb in zip(self.regressors_lgb, self.regressors_cb):
            if sw_pos is not None:
                reg_lgb.fit(X_pos, y_pos, sample_weight=sw_pos)
                reg_cb.fit(X_pos, y_pos, sample_weight=sw_pos)
            else:
                reg_lgb.fit(X_pos, y_pos)
                reg_cb.fit(X_pos, y_pos)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        p_active = self.classifier.predict_proba(X)
        if self.power_shrinkage != 1.0:
            p_active = p_active ** self.power_shrinkage
        if self.zero_gate_threshold > 0.0:
            p_active = np.where(p_active < self.zero_gate_threshold, 0.0, p_active)
        pred_pos = np.zeros(len(X), dtype="float64")
        n_bags = len(self.seeds)
        for reg_lgb, reg_cb in zip(self.regressors_lgb, self.regressors_cb):
            p_lgb = np.clip(reg_lgb.predict(X), 0.0, None)
            p_cb = np.clip(reg_cb.predict(X), 0.0, None)
            pred_pos += (0.55 * p_lgb + 0.45 * p_cb) / n_bags
        return p_active * pred_pos


class BaggedTwoPartNonfuelHurdle:
    """Multi-seed multi-model bagged two-part hurdle model for nonfuel CLV.

    Compresses high-dispersion store spend variance by bagging positive spenders
    across 5 distinct random seeds and dual model architectures (LightGBM Huber + CatBoost).
    """

    def __init__(
        self,
        seeds: tuple[int, ...] = (42, 100, 2024, 777, 314),
        n_jobs: int = -1,
        use_gpu: bool = False,
        huber_alpha: float = 0.85,
        power_shrinkage: float = 0.96,
        zero_gate_threshold: float = 0.04,
    ) -> None:
        self.seeds = seeds
        self.n_jobs = n_jobs
        self.use_gpu = use_gpu
        self.huber_alpha = huber_alpha
        self.power_shrinkage = power_shrinkage
        self.zero_gate_threshold = zero_gate_threshold
        self.classifier = StoreAttachmentClassifier(seed=seeds[0], n_jobs=n_jobs, use_gpu=use_gpu)
        self.regressors_lgb: list[Any] = []
        self.regressors_cb: list[Any] = []
        for s in seeds:
            self.regressors_lgb.append(
                make_regressor("lgbm_huber", seed=s, n_jobs=n_jobs, use_gpu=use_gpu, huber_alpha=huber_alpha)
            )
            self.regressors_cb.append(make_regressor("catboost", seed=s, n_jobs=n_jobs, use_gpu=use_gpu))

    def fit(self, X: pd.DataFrame, y: np.ndarray, sample_weight: np.ndarray | None = None) -> "BaggedTwoPartNonfuelHurdle":
        y_arr = np.asarray(y, dtype="float64")
        pos_mask = y_arr > 0
        self.classifier.fit(X, y_arr, sample_weight=sample_weight)
        X_pos = X.iloc[pos_mask] if isinstance(X, pd.DataFrame) else X[pos_mask]
        y_pos = y_arr[pos_mask]
        sw_pos = np.asarray(sample_weight, dtype="float64")[pos_mask] if sample_weight is not None else None
        for reg_lgb, reg_cb in zip(self.regressors_lgb, self.regressors_cb):
            if sw_pos is not None:
                reg_lgb.fit(X_pos, y_pos, sample_weight=sw_pos)
                reg_cb.fit(X_pos, y_pos, sample_weight=sw_pos)
            else:
                reg_lgb.fit(X_pos, y_pos)
                reg_cb.fit(X_pos, y_pos)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        p_store = self.classifier.predict_proba(X)
        if self.power_shrinkage != 1.0:
            p_store = p_store ** self.power_shrinkage
        if self.zero_gate_threshold > 0.0:
            p_store = np.where(p_store < self.zero_gate_threshold, 0.0, p_store)
        pred_pos = np.zeros(len(X), dtype="float64")
        n_bags = len(self.seeds)
        for reg_lgb, reg_cb in zip(self.regressors_lgb, self.regressors_cb):
            p_lgb = np.clip(reg_lgb.predict(X), 0.0, None)
            p_cb = np.clip(reg_cb.predict(X), 0.0, None)
            pred_pos += (0.55 * p_lgb + 0.45 * p_cb) / n_bags
        return p_store * pred_pos


class MultiQuantileTwoPartHurdle:
    """Multi-Quantile Simpson Quadrature Two-Part Hurdle.

    Deconstructs spend into:
    Stage 1: P(Z > 0 | X) using StoreAttachmentClassifier.
    Stage 2: Conditional quantile regressors across tau in (0.10, 0.25, 0.50, 0.75, 0.90)
             plus a winsorized L2 regressor.
    Integrates positive spend via Simpson/trapezoidal quadrature:
        mu_pos = 0.50 * mu_L2 + 0.50 * sum(w_k * Q_k) with w = (0.15, 0.225, 0.25, 0.225, 0.15).
    Exports conditional spread sigma_pos = max(0.14, (Q_90 - Q_10) / 2.5631).
    """

    QUANTILES: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 0.90)
    QUAD_WEIGHTS: tuple[float, ...] = (0.15, 0.225, 0.25, 0.225, 0.15)

    def __init__(
        self,
        seed: int = 42,
        n_jobs: int = -1,
        use_gpu: bool = False,
    ) -> None:
        self.seed = seed
        self.n_jobs = n_jobs
        self.use_gpu = use_gpu
        self.classifier = StoreAttachmentClassifier(seed=seed, n_jobs=n_jobs, use_gpu=use_gpu)

        from lightgbm import LGBMRegressor
        from catboost import CatBoostRegressor

        self.lgb_quantiles: list[Any] = []
        self.cb_quantiles: list[Any] = []
        for tau in self.QUANTILES:
            self.lgb_quantiles.append(
                LGBMRegressor(
                    objective="quantile", alpha=tau, n_estimators=300, learning_rate=0.03,
                    num_leaves=28, colsample_bytree=0.6, reg_lambda=8.0,
                    random_state=seed, n_jobs=n_jobs, deterministic=True, force_col_wise=True, verbose=-1,
                )
            )
            cb_k: dict[str, Any] = {
                "iterations": 300, "learning_rate": 0.04, "depth": 5, "l2_leaf_reg": 8.0,
                "loss_function": f"Quantile:alpha={tau}", "random_seed": seed, "verbose": 0, "allow_writing_files": False,
            }
            if use_gpu:
                cb_k["task_type"] = "GPU"
            else:
                cb_k["thread_count"] = n_jobs
            self.cb_quantiles.append(CatBoostRegressor(**cb_k))

        self.l2_regressor = LGBMRegressor(
            objective="regression", n_estimators=350, learning_rate=0.03,
            num_leaves=28, colsample_bytree=0.6, reg_lambda=8.0,
            random_state=seed, n_jobs=n_jobs, deterministic=True, force_col_wise=True, verbose=-1,
        )

    def fit(self, X: pd.DataFrame, y: np.ndarray, sample_weight: np.ndarray | None = None) -> "MultiQuantileTwoPartHurdle":
        y_arr = np.asarray(y, dtype="float64")
        self.classifier.fit(X, y_arr, sample_weight=sample_weight)

        pos_mask = y_arr > 0
        X_pos = X.iloc[pos_mask] if isinstance(X, pd.DataFrame) else X[pos_mask]
        y_pos = y_arr[pos_mask]
        sw_pos = np.asarray(sample_weight, dtype="float64")[pos_mask] if sample_weight is not None else None

        if len(y_pos) > 0:
            # Winsorize positive target at 99.5th percentile for L2 regressor
            cap = float(np.percentile(y_pos, 99.5))
            y_pos_win = np.minimum(y_pos, cap)
            if sw_pos is not None:
                self.l2_regressor.fit(X_pos, y_pos_win, sample_weight=sw_pos)
            else:
                self.l2_regressor.fit(X_pos, y_pos_win)

            for lgb_q, cb_q in zip(self.lgb_quantiles, self.cb_quantiles):
                if sw_pos is not None:
                    lgb_q.fit(X_pos, y_pos, sample_weight=sw_pos)
                    cb_q.fit(X_pos, y_pos, sample_weight=sw_pos)
                else:
                    lgb_q.fit(X_pos, y_pos)
                    cb_q.fit(X_pos, y_pos)
        return self

    def predict_proba_active(self, X: pd.DataFrame) -> np.ndarray:
        return self.classifier.predict_proba(X)

    def predict_conditional_pos(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Returns conditional mean mu_pos and spread sigma_pos."""
        mu_l2 = np.clip(self.l2_regressor.predict(X), 0.0, None)
        q_preds: list[np.ndarray] = []
        for lgb_q, cb_q in zip(self.lgb_quantiles, self.cb_quantiles):
            p_lgb = np.clip(lgb_q.predict(X), 0.0, None)
            p_cb = np.clip(cb_q.predict(X), 0.0, None)
            q_preds.append(0.55 * p_lgb + 0.45 * p_cb)

        # Trapezoidal quadrature across quantiles
        mu_quad = sum(w * q for w, q in zip(self.QUAD_WEIGHTS, q_preds))
        mu_pos = 0.50 * mu_l2 + 0.50 * mu_quad

        # Interquartile spread from 90th and 10th percentiles
        sigma_pos = np.maximum(0.14, (q_preds[-1] - q_preds[0]) / 2.5631)
        return mu_pos, sigma_pos

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        p_act = self.predict_proba_active(X)
        mu_pos, _ = self.predict_conditional_pos(X)
        return p_act * mu_pos


class PhysicalBasketTicketRegressor:
    """Raw-Scale Multiplicative Basket x Ticket Decomposition Regressor.

    Models raw units Y_raw = exp(s * Z) - 1 on positive spenders:
        B_hat = E[B_future | X, Y > 0] (Poisson/Tweedie)
        M_hat = E[Y_raw / B_future | X, Y > 0] (Gamma/Huber)
    Recombines via second-order Taylor log-normal expectation correction:
        Z_phys = P(Z > 0 | X) * (1 / s) * [ln(1 + Y_raw_hat) - sigma_raw^2 / (2 * (1 + Y_raw_hat)^2)]
    """

    def __init__(
        self,
        scale: float = 2.352825386185113,
        seed: int = 42,
        n_jobs: int = -1,
        use_gpu: bool = False,
    ) -> None:
        self.scale = scale
        self.seed = seed
        self.n_jobs = n_jobs
        self.use_gpu = use_gpu
        self.classifier = StoreAttachmentClassifier(seed=seed, n_jobs=n_jobs, use_gpu=use_gpu)

        from lightgbm import LGBMRegressor

        self.basket_regressor = LGBMRegressor(
            objective="poisson", n_estimators=300, learning_rate=0.03,
            num_leaves=28, colsample_bytree=0.6, reg_lambda=8.0,
            random_state=seed, n_jobs=n_jobs, deterministic=True, force_col_wise=True, verbose=-1,
        )
        self.ticket_regressor = LGBMRegressor(
            objective="gamma", n_estimators=300, learning_rate=0.03,
            num_leaves=28, colsample_bytree=0.6, reg_lambda=8.0,
            random_state=seed, n_jobs=n_jobs, deterministic=True, force_col_wise=True, verbose=-1,
        )

    def fit(self, X: pd.DataFrame, y: np.ndarray, sample_weight: np.ndarray | None = None) -> "PhysicalBasketTicketRegressor":
        y_arr = np.asarray(y, dtype="float64")
        self.classifier.fit(X, y_arr, sample_weight=sample_weight)

        pos_mask = y_arr > 0
        X_pos = X.iloc[pos_mask] if isinstance(X, pd.DataFrame) else X[pos_mask]
        y_pos = y_arr[pos_mask]
        sw_pos = np.asarray(sample_weight, dtype="float64")[pos_mask] if sample_weight is not None else None

        if len(y_pos) > 0:
            # Convert log target to raw physical scale
            y_raw = np.expm1(self.scale * y_pos)
            cap_raw = float(np.percentile(y_raw, 99.2))
            y_raw_capped = np.clip(y_raw, 0.0, cap_raw)

            # Target baskets proxy from m3_baskets or spend
            if "m3_baskets" in X_pos.columns:
                baskets_pos = np.clip(X_pos["m3_baskets"].to_numpy(), 1.0, 150.0)
            else:
                baskets_pos = np.ones(len(y_pos), dtype="float64")

            ticket_pos = np.clip(y_raw_capped / baskets_pos, 1.0, None)

            if sw_pos is not None:
                self.basket_regressor.fit(X_pos, baskets_pos, sample_weight=sw_pos)
                self.ticket_regressor.fit(X_pos, ticket_pos, sample_weight=sw_pos)
            else:
                self.basket_regressor.fit(X_pos, baskets_pos)
                self.ticket_regressor.fit(X_pos, ticket_pos)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        p_act = self.classifier.predict_proba(X)
        pred_b = np.clip(self.basket_regressor.predict(X), 0.5, None)
        pred_m = np.clip(self.ticket_regressor.predict(X), 1.0, None)
        y_raw_hat = pred_b * pred_m

        # Second-order Taylor log-normal expectation correction
        sigma_raw = 0.25 * y_raw_hat
        denom = 1.0 + y_raw_hat
        z_pos = (1.0 / self.scale) * np.clip(np.log1p(y_raw_hat) - (sigma_raw ** 2) / (2.0 * denom ** 2), 0.0, None)
        return p_act * z_pos

