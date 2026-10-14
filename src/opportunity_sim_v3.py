"""Module 3: Decomposed 24-Category Monte Carlo Opportunity Simulator.

Models the exact data-generating process defined in LABEL_RULES.md (Rules 5, 6, 7):
1. 24 sub-category eligibility & qualification models (Growth and Adoption).
2. Category-specific conditional Rand increment regressions.
3. 2,000-draw vectorized Monte Carlo simulation with Rule 7 argmax competition.
4. Outputs exact, Jensen's-inequality-aware class probabilities P_MC(Y_i = k | X_i).
"""
from __future__ import annotations

from decimal import Decimal
import logging
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor

from label_rules import SCALE

logger = logging.getLogger("opportunity_sim_v3")


class DecomposedCategoryModel:
    """Trains qualification and increment models for a single category."""

    def __init__(self, category: str, seed: int = 42, n_jobs: int = -1) -> None:
        self.category = category
        self.seed = seed
        self.n_jobs = n_jobs

        self.clf_grow: LGBMClassifier | None = None
        self.reg_grow: LGBMRegressor | None = None
        self.sigma_grow: float = 0.5
        self.base_rate_grow: float = 0.0

        self.clf_adopt: LGBMClassifier | None = None
        self.reg_adopt: LGBMRegressor | None = None
        self.sigma_adopt: float = 0.5
        self.base_rate_adopt: float = 0.0

    def fit(
        self,
        X: pd.DataFrame,
        h_ever: np.ndarray,
        y_grow: np.ndarray,
        delta_grow_rand: np.ndarray,
        y_adopt: np.ndarray,
        spend_adopt_rand: np.ndarray,
        is_active: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> "DecomposedCategoryModel":
        num_cols = X.select_dtypes(include=[np.number]).columns
        X_df = X[num_cols].fillna(0.0)

        if is_active is None:
            is_active = np.ones(len(h_ever), dtype=int)

        # 1. Growth model: strictly condition on (h_ever == 1) AND (is_active == 1)
        mask_grow_eligible = (h_ever == 1) & (is_active == 1)
        if mask_grow_eligible.sum() > 20 and y_grow[mask_grow_eligible].sum() >= 5:
            X_g = X_df.iloc[mask_grow_eligible]
            y_g = y_grow[mask_grow_eligible]
            sw_g = sample_weight[mask_grow_eligible] if sample_weight is not None else None
            self.base_rate_grow = float(y_g.mean())

            self.clf_grow = LGBMClassifier(
                n_estimators=150,
                learning_rate=0.06,
                max_depth=5,
                num_leaves=20,
                min_child_samples=20,
                random_state=self.seed,
                n_jobs=self.n_jobs,
                verbose=-1,
            )
            self.clf_grow.fit(X_g, y_g, sample_weight=sw_g)

            # Growth increment regression on positive growth
            pos_g = y_g == 1
            if pos_g.sum() >= 5:
                X_pos_g = X_g.iloc[pos_g]
                log_inc = np.log1p(np.maximum(0.0, delta_grow_rand[mask_grow_eligible][pos_g]))
                sw_pos_g = sw_g[pos_g] if sw_g is not None else None

                self.reg_grow = LGBMRegressor(
                    n_estimators=120,
                    learning_rate=0.06,
                    max_depth=5,
                    num_leaves=20,
                    min_child_samples=15,
                    random_state=self.seed,
                    n_jobs=self.n_jobs,
                    verbose=-1,
                )
                self.reg_grow.fit(X_pos_g, log_inc, sample_weight=sw_pos_g)
                resids = log_inc - self.reg_grow.predict(X_pos_g)
                self.sigma_grow = float(np.std(resids) * 1.35) if len(resids) > 1 else 0.5
        else:
            self.base_rate_grow = float(y_grow[mask_grow_eligible].mean()) if mask_grow_eligible.sum() > 0 else 0.0

        # 2. Adoption model: strictly condition on (h_ever == 0) AND (is_active == 1)
        mask_adopt_eligible = (h_ever == 0) & (is_active == 1)
        if mask_adopt_eligible.sum() > 20 and y_adopt[mask_adopt_eligible].sum() >= 5:
            X_a = X_df.iloc[mask_adopt_eligible]
            y_a = y_adopt[mask_adopt_eligible]
            sw_a = sample_weight[mask_adopt_eligible] if sample_weight is not None else None
            self.base_rate_adopt = float(y_a.mean())

            self.clf_adopt = LGBMClassifier(
                n_estimators=150,
                learning_rate=0.06,
                max_depth=5,
                num_leaves=20,
                min_child_samples=20,
                random_state=self.seed,
                n_jobs=self.n_jobs,
                verbose=-1,
            )
            self.clf_adopt.fit(X_a, y_a, sample_weight=sw_a)

            # Adoption spend regression on positive adoption
            pos_a = y_a == 1
            if pos_a.sum() >= 5:
                X_pos_a = X_a.iloc[pos_a]
                log_spend = np.log1p(np.maximum(0.0, spend_adopt_rand[mask_adopt_eligible][pos_a]))
                sw_pos_a = sw_a[pos_a] if sw_a is not None else None

                self.reg_adopt = LGBMRegressor(
                    n_estimators=120,
                    learning_rate=0.06,
                    max_depth=5,
                    num_leaves=20,
                    min_child_samples=15,
                    random_state=self.seed,
                    n_jobs=self.n_jobs,
                    verbose=-1,
                )
                self.reg_adopt.fit(X_pos_a, log_spend, sample_weight=sw_pos_a)
                resids = log_spend - self.reg_adopt.predict(X_pos_a)
                self.sigma_adopt = float(np.std(resids) * 1.35) if len(resids) > 1 else 0.5
        else:
            self.base_rate_adopt = float(y_adopt[mask_adopt_eligible].mean()) if mask_adopt_eligible.sum() > 0 else 0.0

        return self

    def predict_growth_params(self, X: pd.DataFrame, h_ever: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        N = len(X)
        p_grow = np.zeros(N, dtype=np.float64)
        mu_grow = np.full(N, 3.0, dtype=np.float64)  # ~ R20 default

        num_cols = X.select_dtypes(include=[np.number]).columns
        X_df = X[num_cols].fillna(0.0)

        mask = (h_ever == 1)
        if mask.any():
            if self.clf_grow is not None:
                p_grow[mask] = self.clf_grow.predict_proba(X_df.iloc[mask])[:, 1]
            else:
                p_grow[mask] = self.base_rate_grow

            if self.reg_grow is not None:
                mu_grow[mask] = np.clip(self.reg_grow.predict(X_df.iloc[mask]), np.log1p(20.0), None)

        return p_grow, mu_grow, max(0.2, self.sigma_grow)

    def predict_adoption_params(self, X: pd.DataFrame, h_ever: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        N = len(X)
        p_adopt = np.zeros(N, dtype=np.float64)
        mu_adopt = np.full(N, 2.5, dtype=np.float64)  # ~ R12 default

        num_cols = X.select_dtypes(include=[np.number]).columns
        X_df = X[num_cols].fillna(0.0)

        mask = (h_ever == 0)
        if mask.any():
            if self.clf_adopt is not None:
                p_adopt[mask] = self.clf_adopt.predict_proba(X_df.iloc[mask])[:, 1]
            else:
                p_adopt[mask] = self.base_rate_adopt

            if self.reg_adopt is not None:
                mu_adopt[mask] = np.clip(self.reg_adopt.predict(X_df.iloc[mask]), np.log1p(1.0), None)

        return p_adopt, mu_adopt, max(0.2, self.sigma_adopt)


class VectorizedMonteCarloSimulator:
    """Vectorized Monte Carlo Simulator reproducing Rule 7 argmax competition."""

    def __init__(self, label_config: dict[str, Any], seed: int = 42, n_draws: int = 2000) -> None:
        self.label_config = label_config
        self.seed = seed
        self.n_draws = n_draws

        self.mapping = label_config["category_mapping"]
        self.included_cats = sorted(c for c, v in self.mapping.items() if v["included"])
        self.opp_labels = label_config["opportunity_labels"]
        self.K = len(self.opp_labels)
        self.C = len(self.included_cats)

        self.cat_to_idx = {c: i for i, c in enumerate(self.included_cats)}
        self.label_to_idx = {l: i for i, l in enumerate(self.opp_labels)}

        # Build lookup tables for mapped classes
        self.grow_class_lookup = np.zeros(self.C, dtype=np.int32)
        self.adopt_class_lookup = np.zeros(self.C, dtype=np.int32)
        for i, c in enumerate(self.included_cats):
            g_target = self.mapping[c]["Existing-category growth"]
            a_target = self.mapping[c]["New category adoption"]
            g_label = f"Existing-category growth: {g_target}"
            a_label = f"New category adoption: {a_target}"
            self.grow_class_lookup[i] = self.label_to_idx.get(g_label, self.label_to_idx["Stable"])
            self.adopt_class_lookup[i] = self.label_to_idx.get(a_label, self.label_to_idx["Stable"])

        self.inactivity_idx = self.label_to_idx["Inactivity"]
        self.stable_idx = self.label_to_idx["Stable"]

        # Models
        self.clf_active: LGBMClassifier | None = None
        self.cat_models: dict[str, DecomposedCategoryModel] = {
            c: DecomposedCategoryModel(c, seed=seed, n_jobs=2) for c in self.included_cats
        }

    @staticmethod
    def extract_ground_truth_matrices(
        tx: pd.DataFrame,
        cutoff: pd.Timestamp,
        ids: Sequence[str],
        config: dict[str, Any],
    ) -> dict[str, Any]:
        """Extracts exact historical and future qualification matrices matching label_rules.py."""
        t = pd.Timestamp(cutoff)
        end = t + pd.DateOffset(months=3)
        mapping = config["category_mapping"]
        cats = sorted(c for c, v in mapping.items() if v["included"])

        p = tx[~tx["ItemName"].isin(config["adjustment_items"])]
        h = p[p["_time"] < t]
        recent = h[h["_time"] >= t - pd.DateOffset(months=3)]
        future = p[(p["_time"] >= t) & (p["_time"] < end)]

        def matrix(frame: pd.DataFrame, count: bool = False) -> pd.DataFrame:
            frame_sub = frame[frame["ItemCategoryLevel2"].isin(cats)]
            if count:
                frame_sub = frame_sub[(frame_sub["_quantity"] > 0) & (frame_sub["_amount"] > 0)]
            g = frame_sub.groupby(["ID", "ItemCategoryLevel2"])
            s = g["TransactionId"].nunique() if count else g["_amount"].sum()
            return s.unstack(fill_value=0).reindex(index=ids, columns=cats, fill_value=0).astype("int64")

        historical = matrix(h, True)
        future_purchases = matrix(future, True)
        previous_spend = matrix(recent)
        future_spend = matrix(future)
        delta = future_spend - previous_spend

        grow = historical.gt(0) & future_purchases.ge(config["growth_min_purchases"])
        grow &= delta.ge(int(Decimal(str(config["growth_min_rand"])) * SCALE))
        if config["growth_min_relative"] is not None:
            grow &= future_spend.gt(previous_spend.clip(lower=0) * (1 + config["growth_min_relative"]))

        adopt = historical.eq(0) & future_purchases.ge(config["adoption_min_purchases"]) & future_spend.gt(0)
        threshold = int(Decimal(str(config["adoption_min_rand"])) * SCALE)
        adopt &= future_spend.ge(threshold)

        is_active = future_purchases.sum(axis=1).gt(0).astype(int).values

        return {
            "h_ever": historical.gt(0).astype(int).values,
            "grow": grow.astype(int).values,
            "delta_grow_rand": (delta / SCALE).values,
            "adopt": adopt.astype(int).values,
            "spend_adopt_rand": (future_spend / SCALE).values,
            "is_active": is_active,
            "cats": cats,
        }

    def fit(
        self,
        X_train: pd.DataFrame,
        gt_train: dict[str, Any],
        sample_weight: np.ndarray | None = None,
    ) -> "VectorizedMonteCarloSimulator":
        """Fits activity classifier and 24 category models."""
        logger.info("Fitting global activity classifier P(Active | X)...")
        num_cols = X_train.select_dtypes(include=[np.number]).columns
        X_df = X_train[num_cols].fillna(0.0)

        self.clf_active = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            min_child_samples=50,
            random_state=self.seed,
            n_jobs=-1,
            verbose=-1,
        )
        self.clf_active.fit(X_df, gt_train["is_active"], sample_weight=sample_weight)

        logger.info("Fitting 24 sub-category models...")
        h_ever = gt_train["h_ever"]
        y_grow = gt_train["grow"]
        delta_g = gt_train["delta_grow_rand"]
        y_adopt = gt_train["adopt"]
        spend_a = gt_train["spend_adopt_rand"]

        for i, c in enumerate(self.included_cats):
            self.cat_models[c].fit(
                X_train,
                h_ever[:, i],
                y_grow[:, i],
                delta_g[:, i],
                y_adopt[:, i],
                spend_a[:, i],
                is_active=gt_train["is_active"],
                sample_weight=sample_weight,
            )
        logger.info("Fitted all 24 sub-category models successfully.")
        return self

    def simulate(
        self,
        X_test: pd.DataFrame,
        h_ever_test: np.ndarray,
        chunk_size: int = 200,
    ) -> np.ndarray:
        """Executes vectorized Monte Carlo simulation to compute P_MC(Y_i = k | X_i)."""
        N = len(X_test)
        M = self.n_draws
        C = self.C
        K = self.K

        # Predict activity probabilities
        num_cols = X_test.select_dtypes(include=[np.number]).columns
        X_df = X_test[num_cols].fillna(0.0)
        p_active = self.clf_active.predict_proba(X_df)[:, 1]  # shape (N,)

        # Predict parameters for all 24 categories
        p_grow_mat = np.zeros((N, C), dtype=np.float32)
        mu_grow_mat = np.zeros((N, C), dtype=np.float32)
        sig_grow_arr = np.zeros(C, dtype=np.float32)

        p_adopt_mat = np.zeros((N, C), dtype=np.float32)
        mu_adopt_mat = np.zeros((N, C), dtype=np.float32)
        sig_adopt_arr = np.zeros(C, dtype=np.float32)

        for i, c in enumerate(self.included_cats):
            pg, mg, sg = self.cat_models[c].predict_growth_params(X_test, h_ever_test[:, i])
            p_grow_mat[:, i] = pg
            mu_grow_mat[:, i] = mg
            sig_grow_arr[i] = sg

            pa, ma, sa = self.cat_models[c].predict_adoption_params(X_test, h_ever_test[:, i])
            p_adopt_mat[:, i] = pa
            mu_adopt_mat[:, i] = ma
            sig_adopt_arr[i] = sa

        logger.info("Running %d Monte Carlo draws for %d customers in chunks of %d...", M, N, chunk_size)
        rng = np.random.RandomState(self.seed)
        counts = np.zeros((N, K), dtype=np.int32)
        n_chunks = int(np.ceil(M / chunk_size))

        for chunk_idx in range(n_chunks):
            cur_draws = min(chunk_size, M - chunk_idx * chunk_size)
            if cur_draws <= 0:
                break

            # 1. Activity draw: shape (N, cur_draws)
            u_act = rng.rand(N, cur_draws) < p_active[:, None]

            # 2. Growth qualifications & values: shape (N, C, cur_draws)
            u_grow = rng.rand(N, C, cur_draws) < p_grow_mat[:, :, None]
            eps_grow = rng.randn(N, C, cur_draws)
            val_grow = np.exp(mu_grow_mat[:, :, None] + eps_grow * sig_grow_arr[None, :, None]) - 1.0
            gv = np.where(u_grow, np.maximum(20.0, val_grow), 0.0)

            # 3. Adoption qualifications & values: shape (N, C, cur_draws)
            u_adopt = rng.rand(N, C, cur_draws) < p_adopt_mat[:, :, None]
            eps_adopt = rng.randn(N, C, cur_draws)
            val_adopt = np.exp(mu_adopt_mat[:, :, None] + eps_adopt * sig_adopt_arr[None, :, None]) - 1.0
            av = np.where(u_adopt, np.maximum(1.0, val_adopt), 0.0)

            # 4. Rule 7 Max competition
            am = av.max(axis=1)  # (N, cur_draws)
            gm = gv.max(axis=1)  # (N, cur_draws)
            best_adopt_c = av.argmax(axis=1)  # (N, cur_draws)
            best_grow_c = gv.argmax(axis=1)  # (N, cur_draws)

            is_adopt = (am > 0) & (am >= gm)
            is_growth = (gm > 0) & ~is_adopt

            # Class index array for chunk
            class_idx = np.full((N, cur_draws), self.stable_idx, dtype=np.int32)

            # Map winners to classes
            class_idx[is_growth] = self.grow_class_lookup[best_grow_c[is_growth]]
            class_idx[is_adopt] = self.adopt_class_lookup[best_adopt_c[is_adopt]]

            # Inactivity overrides all
            class_idx[~u_act] = self.inactivity_idx

            # Accumulate counts
            for k in range(K):
                counts[:, k] += (class_idx == k).sum(axis=1)

        p_mc = counts.astype(np.float64) / float(M)
        logger.info("Monte Carlo simulation complete: P_MC matrix shape %s", p_mc.shape)
        return p_mc
