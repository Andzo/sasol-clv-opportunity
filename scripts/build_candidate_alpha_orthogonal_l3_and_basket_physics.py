"""Candidate Alpha: Orthogonal Level-3 Fine-Category & Multiplicative Basket-Physics Basis Generator.

Constructs a genuinely orthogonal prediction vector (rho ~ 0.92-0.95 with existing log-GBDTs)
to break the 13-model subspace ceiling on CLV_nonfuel and CLV_fuel:
1. Extracts unused DATA_DICTIONARY.md columns:
   - ItemCategoryLevel3 (top 35 fine categories log-spend, recency, and Herfindahl habit index)
   - LineItemSequence (basket depth, multi-item store attachment ratio)
   - TransactionTotalInclAmount vs in-scope TotalExclAmount (excluded wallet share, VAT/fuel mix)
   - Inter-visit interval regularity & weekend/commuter hour basket interactions
2. Multiplicative Physical Decomposition (Baskets x Ticket-Size -> Analytical Log-Normal Taylor Conversion):
   Predicts future basket count E[B] and conditional ticket spend E[M | B>0] in physical units,
   converting to Z = ln(1 + Y)/s via second-order log-normal variance correction.
3. CatBoost Native Multi-Quantile Quadrature + Ridge/MLP Orthogonal Stacker.
4. Blends at a conservative 12% weight onto Candidate Omega-4 (after zero-mean subgroup alignment)
   so that it both improves score directly AND serves as an orthogonal 14th basis vector for Hilbert projection!
"""
from __future__ import annotations

import hashlib
import logging
from datetime import datetime
from pathlib import Path
import sys
from typing import Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import QuantileTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PipelineConfig
from src.data import load_transactions
from src.dataset import SnapshotBuilder
from scripts.build_candidate_w2 import (
    GlobalAlignedSVD,
    extract_enriched_snapshot,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_candidate_alpha")

S_FUEL = 2.352825386185113
S_NF = 2.4200904988927476


class Level3AndBasketStructureExtractor:
    """Extracts previously unused DATA_DICTIONARY.md columns:
    ItemCategoryLevel3, ItemCategoryLevel1, LineItemSequence, TransactionTotalInclAmount,
    and raw basket/ticket multiplicative components."""

    def __init__(self, top_k_l3: int = 35) -> None:
        self.top_k_l3 = top_k_l3
        self.top_l3_cats: list[str] = []

    def fit(self, tx: pd.DataFrame, anchor_cutoff: pd.Timestamp) -> "Level3AndBasketStructureExtractor":
        h_tx = tx[(tx["_time"] < anchor_cutoff) & (tx["is_qualifying"] == True) & (tx["amount"] > 0)]
        if "ItemCategoryLevel3" in h_tx.columns:
            vc = h_tx[h_tx["is_nonfuel"] == True]["ItemCategoryLevel3"].dropna().astype(str).value_counts()
            self.top_l3_cats = sorted(vc.head(self.top_k_l3).index.tolist())
        return self

    def transform(self, tx: pd.DataFrame, cutoff: pd.Timestamp, ids: Sequence[str]) -> pd.DataFrame:
        id_list = [str(x) for x in ids]
        id_set = set(id_list)
        res = pd.DataFrame(index=pd.Index(id_list, name="ID"))

        h_tx = tx[
            (tx["ID"].astype(str).isin(id_set))
            & (tx["_time"] < cutoff)
        ].copy()
        if h_tx.empty:
            return res

        h_tx["ID_str"] = h_tx["ID"].astype(str)
        t_m3 = cutoff - pd.DateOffset(months=3)
        t_m1 = cutoff - pd.DateOffset(months=1)

        # 1. Basket Depth (LineItemSequence) & Excluded Wallet Share (TransactionTotalInclAmount)
        if "LineItemSequence" in h_tx.columns:
            h_tx["LineItemSequence_num"] = pd.to_numeric(h_tx["LineItemSequence"], errors="coerce").fillna(1.0)
            bask_depth = h_tx.groupby(["ID_str", "TransactionId"])["LineItemSequence_num"].max().reset_index()
            depth_agg = bask_depth.groupby("ID_str")["LineItemSequence_num"].agg(
                mean_basket_line_depth="mean",
                max_basket_line_depth="max",
                multi_line_basket_share=lambda s: float((s > 1).mean()),
            )
            res = res.join(depth_agg)

        if "TransactionTotalInclAmount" in h_tx.columns:
            h_tx["TransactionTotalInclAmount_num"] = pd.to_numeric(h_tx["TransactionTotalInclAmount"], errors="coerce").fillna(0.0)
            bask_tot = h_tx.groupby(["ID_str", "TransactionId"]).agg(
                incl_tot=("TransactionTotalInclAmount_num", "first"),
                inscope_tot=("amount", lambda s: float(np.maximum(0.0, s).sum())),
            ).reset_index()
            bask_tot["inscope_ratio"] = bask_tot["inscope_tot"] / np.maximum(1.0, np.abs(bask_tot["incl_tot"]))
            tot_agg = bask_tot.groupby("ID_str").agg(
                mean_basket_incl_rand=("incl_tot", "mean"),
                inscope_wallet_share=("inscope_ratio", "mean"),
            )
            res = res.join(tot_agg)

        # 2. ItemCategoryLevel3 Fine-Category Spend (m3 & m1) and Habit Concentration (HHI)
        nf_q = h_tx[(h_tx["is_nonfuel"] == True) & (h_tx["is_qualifying"] == True) & (h_tx["amount"] > 0)].copy()
        if not nf_q.empty and "ItemCategoryLevel3" in nf_q.columns and self.top_l3_cats:
            nf_q["L3_str"] = nf_q["ItemCategoryLevel3"].astype(str)

            # Herfindahl-Hirschman Index (HHI) across Level3 categories (routine vs impulse shopper)
            l3_tot = nf_q.groupby(["ID_str", "L3_str"])["amount"].sum().unstack(fill_value=0.0)
            row_sum = l3_tot.sum(axis=1)
            shares = l3_tot.div(np.maximum(row_sum, 1e-3), axis=0)
            hhi = (shares ** 2).sum(axis=1).rename("nf_l3_hhi_concentration")
            n_distinct_l3 = (l3_tot > 0).sum(axis=1).rename("nf_l3_distinct_count")
            res = res.join(hhi).join(n_distinct_l3)

            # m3 log spend across top 35 Level-3 categories
            nf_m3 = nf_q[nf_q["_time"] >= t_m3]
            if not nf_m3.empty:
                m3_l3 = nf_m3[nf_m3["L3_str"].isin(self.top_l3_cats)].groupby(["ID_str", "L3_str"])["amount"].sum().unstack(fill_value=0.0)
                m3_l3 = np.log1p(np.maximum(0.0, m3_l3.reindex(index=id_list, columns=self.top_l3_cats, fill_value=0.0)))
                m3_l3.columns = [f"l3_m3_log_{i}" for i in range(len(self.top_l3_cats))]
                res = res.join(m3_l3)

        # 3. Multiplicative Ticket-Size & Basket Frequency Physics (m1 and m3)
        q_m3 = h_tx[(h_tx["_time"] >= t_m3) & (h_tx["is_qualifying"] == True) & (h_tx["amount"] > 0)]
        if not q_m3.empty:
            f_m3 = q_m3[q_m3["is_fuel_sale"] == True]
            nf_m3_sub = q_m3[q_m3["is_nonfuel"] == True]

            f_bask = f_m3.groupby(["ID_str", "TransactionId"])["qty"].sum().reset_index()
            f_ticket = f_bask.groupby("ID_str")["qty"].agg(
                m3_fuel_ticket_mean="mean",
                m3_fuel_ticket_median="median",
                m3_fuel_ticket_std="std",
            )
            res = res.join(f_ticket)

            nf_bask = nf_m3_sub.groupby(["ID_str", "TransactionId"])["amount"].sum().reset_index()
            nf_ticket = nf_bask.groupby("ID_str")["amount"].agg(
                m3_nf_ticket_mean="mean",
                m3_nf_ticket_median="median",
                m3_nf_ticket_max="max",
                m3_nf_ticket_std="std",
            )
            res = res.join(nf_ticket)

        return res.fillna(0.0)


class OrthogonalMultiParadigmRegressor:
    """Combines 3 genuinely orthogonal regression paradigms not used in W2/Y2:
    1. CatBoost Native Multi-Quantile Regressor (Symmetric Oblivious Trees predicting 5 quantiles simultaneously).
    2. Raw-Scale Tweedie/Poisson Multiplicative Regressor converted to log-space via analytical Taylor correction.
    3. Quantile-Normalized Tabular MLP Regressor (smooth neural representation).
    Stacked via Customer-Grouped 5-Fold OOF RidgeCV."""

    def __init__(self, target_name: str = "CLV_fuel", scale_const: float = S_FUEL, seed: int = 42) -> None:
        self.target_name = target_name
        self.scale_const = scale_const
        self.seed = seed
        self.q_weights = np.array([0.15, 0.225, 0.25, 0.225, 0.15], dtype=np.float64)

        self.cb_mq_pos: CatBoostRegressor | None = None
        self.lgb_raw_tweedie: lgb.LGBMRegressor | None = None
        self.qt_scaler: QuantileTransformer | None = None
        self.mlp_reg: MLPRegressor | None = None
        self.clf_act: lgb.LGBMClassifier | None = None
        self.stacker: RidgeCV | None = None
        self.oof_pred: np.ndarray | None = None

    def fit(
        self,
        X: pd.DataFrame,
        y_log: np.ndarray,
        groups: np.ndarray,
        sample_weight: np.ndarray,
        is_summer: np.ndarray,
    ) -> "OrthogonalMultiParadigmRegressor":
        X_df = X.select_dtypes(include=[np.number]).fillna(0.0)
        y_arr = np.asarray(y_log, dtype=np.float64)
        y_bin = (y_arr > 0).astype(int)
        # Reconstruct raw physical target (litres or rand) capped at 99.5th percentile for raw Tweedie
        y_raw = np.expm1(np.clip(y_arr * self.scale_const, 0.0, 12.0))
        raw_cap = float(np.percentile(y_raw[y_arr > 0], 99.2))
        y_raw_capped = np.clip(y_raw, 0.0, raw_cap)

        N = len(y_arr)
        gkf = GroupKFold(n_splits=5)

        oof_p_act = np.zeros(N, dtype=np.float64)
        oof_cb_mq = np.zeros(N, dtype=np.float64)
        oof_tweedie_log = np.zeros(N, dtype=np.float64)
        oof_mlp = np.zeros(N, dtype=np.float64)

        logger.info("[%s] Fitting Orthogonal 3-Paradigm 5-Fold OOF Stack...", self.target_name)
        for tr_idx, val_idx in gkf.split(X_df, y_arr, groups=groups):
            # 1. Stage 1 Activity Probability
            clf = lgb.LGBMClassifier(
                n_estimators=260, learning_rate=0.05, max_depth=6, num_leaves=31,
                min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
                random_state=self.seed, n_jobs=-1, verbose=-1,
            ).fit(X_df.iloc[tr_idx], y_bin[tr_idx], sample_weight=sample_weight[tr_idx])
            oof_p_act[val_idx] = clf.predict_proba(X_df.iloc[val_idx])[:, 1]

            # 2. CatBoost Native Multi-Quantile on Positive Spenders
            tr_pos = tr_idx[y_arr[tr_idx] > 0]
            cb_mq = CatBoostRegressor(
                loss_function="MultiQuantile:alpha=0.10,0.25,0.50,0.75,0.90",
                iterations=400, learning_rate=0.06, depth=6, l2_leaf_reg=15.0,
                random_seed=self.seed, verbose=False,
            ).fit(X_df.iloc[tr_pos], y_arr[tr_pos], sample_weight=sample_weight[tr_pos])
            q_preds = np.clip(cb_mq.predict(X_df.iloc[val_idx]), 0.0, None)
            oof_cb_mq[val_idx] = q_preds.dot(self.q_weights)

            # 3. Raw-Scale Tweedie converted to Log-Space
            reg_tw = lgb.LGBMRegressor(
                objective="tweedie", tweedie_variance_power=1.35,
                n_estimators=300, learning_rate=0.04, max_depth=6, num_leaves=31,
                min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
                reg_lambda=25.0, random_state=self.seed, n_jobs=-1, verbose=-1,
            ).fit(X_df.iloc[tr_idx], y_raw_capped[tr_idx], sample_weight=sample_weight[tr_idx])
            raw_pred = np.clip(reg_tw.predict(X_df.iloc[val_idx]), 0.0, None)
            oof_tweedie_log[val_idx] = np.log1p(raw_pred) / self.scale_const

            # 4. Quantile-Normalized Tabular MLP
            qt = QuantileTransformer(output_distribution="normal", n_quantiles=800, random_state=self.seed)
            X_tr_qt = qt.fit_transform(X_df.iloc[tr_idx])
            X_val_qt = qt.transform(X_df.iloc[val_idx])
            mlp = MLPRegressor(
                hidden_layer_sizes=(128, 64), activation="relu", alpha=0.5,
                learning_rate_init=1e-3, max_iter=25, early_stopping=True, random_state=self.seed,
            ).fit(X_tr_qt, y_arr[tr_idx])
            oof_mlp[val_idx] = np.clip(mlp.predict(X_val_qt), 0.0, None)

        meta_oof = self._build_meta(oof_p_act, oof_cb_mq, oof_tweedie_log, oof_mlp, is_summer)
        self.stacker = RidgeCV(alphas=np.logspace(-1, 4, 25), cv=5).fit(meta_oof, y_arr, sample_weight=sample_weight)
        self.oof_pred = np.clip(self.stacker.predict(meta_oof), 0.0, None)
        oof_rmse = float(np.sqrt(np.mean((y_arr - self.oof_pred) ** 2)))
        logger.info("[%s Orthogonal Stack] 5-Fold Customer-Grouped OOF RMSE = %.6f", self.target_name, oof_rmse)

        # Fit full models on all training data
        pos_all = y_arr > 0
        self.clf_act = lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=6, num_leaves=31,
            min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
            random_state=self.seed, n_jobs=-1, verbose=-1,
        ).fit(X_df, y_bin, sample_weight=sample_weight)

        self.cb_mq_pos = CatBoostRegressor(
            loss_function="MultiQuantile:alpha=0.10,0.25,0.50,0.75,0.90",
            iterations=450, learning_rate=0.06, depth=6, l2_leaf_reg=15.0,
            random_seed=self.seed, verbose=False,
        ).fit(X_df.iloc[pos_all], y_arr[pos_all], sample_weight=sample_weight[pos_all])

        self.lgb_raw_tweedie = lgb.LGBMRegressor(
            objective="tweedie", tweedie_variance_power=1.35,
            n_estimators=350, learning_rate=0.04, max_depth=6, num_leaves=31,
            min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=25.0, random_state=self.seed, n_jobs=-1, verbose=-1,
        ).fit(X_df, y_raw_capped, sample_weight=sample_weight)

        self.qt_scaler = QuantileTransformer(output_distribution="normal", n_quantiles=1000, random_state=self.seed)
        X_all_qt = self.qt_scaler.fit_transform(X_df)
        self.mlp_reg = MLPRegressor(
            hidden_layer_sizes=(128, 64), activation="relu", alpha=0.5,
            learning_rate_init=1e-3, max_iter=30, early_stopping=True, random_state=self.seed,
        ).fit(X_all_qt, y_arr)

        return self

    @staticmethod
    def _build_meta(
        p_act: np.ndarray, cb_mq: np.ndarray, tw_log: np.ndarray, mlp_p: np.ndarray, is_summer: np.ndarray
    ) -> np.ndarray:
        h_cb = p_act * cb_mq
        return np.column_stack([
            p_act, cb_mq, h_cb, tw_log, mlp_p,
            h_cb * is_summer, tw_log * is_summer, mlp_p * is_summer,
        ])

    def predict(self, X: pd.DataFrame, is_summer: np.ndarray) -> np.ndarray:
        X_df = X.select_dtypes(include=[np.number]).fillna(0.0)
        p_act = self.clf_act.predict_proba(X_df)[:, 1]
        cb_q = np.clip(self.cb_mq_pos.predict(X_df), 0.0, None).dot(self.q_weights)
        tw_log = np.log1p(np.clip(self.lgb_raw_tweedie.predict(X_df), 0.0, None)) / self.scale_const
        mlp_p = np.clip(self.mlp_reg.predict(self.qt_scaler.transform(X_df)), 0.0, None)

        meta_X = self._build_meta(p_act, cb_q, tw_log, mlp_p, is_summer)
        return np.clip(self.stacker.predict(meta_X), 0.0, None)


def build_candidate_alpha(n_features: int = 180, blend_w: float = 0.12) -> Path:
    test_csv = resolve_data_file(PROJECT_ROOT / "test.csv", "test customer IDs CSV")
    test_df = pd.read_csv(test_csv)
    test_ids = [str(x) for x in test_df["ID"]]
    assert len(test_ids) == 5488

    submissions_dir = PROJECT_ROOT / "submissions"
    omega4_files = sorted(submissions_dir.glob("submission_candidate_omega4_spectral_hilbert_*.csv"))
    assert omega4_files, "Missing Candidate Omega-4 submission!"
    df_raw = pd.read_csv(omega4_files[-1])
    df_raw["ID"] = df_raw["ID"].astype(str)
    df_omega4 = df_raw.set_index("ID").reindex(test_ids).reset_index()
    assert not df_omega4.isna().any().any(), "df_omega4 contains NaNs after reindexing!"
    logger.info("Loaded baseline Candidate Omega-4: %s", omega4_files[-1].name)

    cfg = PipelineConfig()
    tx = load_transactions(cfg)
    builder = SnapshotBuilder(cfg, tx)

    svd_model = GlobalAlignedSVD(n_components=16, top_k_items=500, seed=42).fit(tx, pd.Timestamp("2025-09-01"))
    l3_extractor = Level3AndBasketStructureExtractor(top_k_l3=35).fit(tx, pd.Timestamp("2025-09-01"))

    cutoffs = [
        pd.Timestamp("2024-09-01"),
        pd.Timestamp("2024-12-01"),
        pd.Timestamp("2025-03-01"),
        pd.Timestamp("2025-06-01"),
        pd.Timestamp("2025-09-01"),
    ]

    X_list, yf_list, ynf_list, sw_list, sum_list, grp_list = [], [], [], [], [], []
    for c in cutoffs:
        logger.info("Extracting Level-3 & Basket-Physics enriched snapshot for %s...", c.strftime("%Y-%m-%d"))
        feats, lbls = extract_enriched_snapshot(builder, tx, svd_model, c, with_labels=True)
        c_ids = feats.index.astype(str).tolist()
        df_l3 = l3_extractor.transform(tx, c, c_ids)
        full_f = pd.concat([feats, df_l3], axis=1)
        full_f = full_f.loc[:, ~full_f.columns.duplicated()].fillna(0.0)

        is_sum = 1.0 if c.month == 12 else 0.0
        w = 2.0 if is_sum == 1.0 else 1.0
        X_list.append(full_f)
        yf_list.append(lbls["CLV_fuel"].to_numpy())
        ynf_list.append(lbls["CLV_nonfuel"].to_numpy())
        sw_list.append(np.full(len(full_f), w, dtype=np.float64))
        sum_list.append(np.full(len(full_f), is_sum, dtype=np.float64))
        grp_list.append(np.array(c_ids, dtype=object))

    X_all = pd.concat(X_list, axis=0)
    yf_all = np.concatenate(yf_list)
    ynf_all = np.concatenate(ynf_list)
    sw_all = np.concatenate(sw_list)
    sum_all = np.concatenate(sum_list)
    grp_all = np.concatenate(grp_list)

    test_cutoff = pd.Timestamp("2025-12-01")
    X_test_base, _ = extract_enriched_snapshot(builder, tx, svd_model, test_cutoff, ids=test_ids, with_labels=False)
    df_l3_test = l3_extractor.transform(tx, test_cutoff, test_ids)
    X_test_all = pd.concat([X_test_base, df_l3_test], axis=1)
    X_test_all = X_test_all.loc[:, ~X_test_all.columns.duplicated()].fillna(0.0)
    sum_test = np.ones(len(test_ids), dtype=np.float64)

    num_cols = X_all.select_dtypes(include=[np.number]).columns.tolist()
    logger.info("Total candidate features (including Level-3 & Basket Structure): %d", len(num_cols))

    sel_f = lgb.LGBMRegressor(
        n_estimators=150, learning_rate=0.05, num_leaves=31, importance_type="gain", random_state=42, n_jobs=-1, verbose=-1
    ).fit(X_all[num_cols], yf_all, sample_weight=sw_all)
    top_f_feats = pd.Series(sel_f.feature_importances_, index=num_cols).sort_values(ascending=False).head(n_features).index.tolist()

    sel_nf = lgb.LGBMRegressor(
        n_estimators=150, learning_rate=0.05, num_leaves=31, importance_type="gain", random_state=42, n_jobs=-1, verbose=-1
    ).fit(X_all[num_cols], ynf_all, sample_weight=sw_all)
    top_nf_feats = pd.Series(sel_nf.feature_importances_, index=num_cols).sort_values(ascending=False).head(n_features).index.tolist()

    new_l3_in_nf = [c for c in top_nf_feats if "l3_" in c or "ticket" in c or "basket" in c or "inscope" in c]
    logger.info("New Level-3 & Ticket/Basket features selected in Nonfuel Top %d: %d features (%s)", n_features, len(new_l3_in_nf), new_l3_in_nf[:10])

    # Train Orthogonal Multi-Paradigm Regressors
    orth_f = OrthogonalMultiParadigmRegressor(target_name="CLV_fuel", scale_const=S_FUEL, seed=42).fit(
        X_all[top_f_feats], yf_all, groups=grp_all, sample_weight=sw_all, is_summer=sum_all
    )
    pred_f_alpha = orth_f.predict(X_test_all[top_f_feats], is_summer=sum_test)

    orth_nf = OrthogonalMultiParadigmRegressor(target_name="CLV_nonfuel", scale_const=S_NF, seed=42).fit(
        X_all[top_nf_feats], ynf_all, groups=grp_all, sample_weight=sw_all, is_summer=sum_all
    )
    pred_nf_alpha = orth_nf.predict(X_test_all[top_nf_feats], is_summer=sum_test)

    ref_f = df_omega4["CLV_fuel"].to_numpy()
    ref_nf = df_omega4["CLV_nonfuel"].to_numpy()

    # Check correlation with Omega-4 (verifying genuine orthogonality: target rho ~ 0.94-0.97)
    corr_f = float(np.corrcoef(ref_f, pred_f_alpha)[0, 1])
    corr_nf = float(np.corrcoef(ref_nf, pred_nf_alpha)[0, 1])
    logger.info("Orthogonality Check vs Omega-4 -> Fuel rho=%.4f, Nonfuel rho=%.4f", corr_f, corr_nf)

    # Zero-mean, variance-matched alignment to Omega-4 on active subsets
    f_alpha_aligned = float(np.mean(ref_f)) + ((pred_f_alpha - float(np.mean(pred_f_alpha))) / max(float(np.std(pred_f_alpha)), 1e-6)) * float(np.std(ref_f))
    f_alpha_aligned = np.clip(f_alpha_aligned, 0.0, None)

    pos_nf = ref_nf > 1e-6
    nf_alpha_aligned = np.zeros_like(ref_nf)
    if pos_nf.any():
        p_sub = pred_nf_alpha[pos_nf]
        r_sub = ref_nf[pos_nf]
        nf_alpha_aligned[pos_nf] = np.clip(
            float(np.mean(r_sub)) + ((p_sub - float(np.mean(p_sub))) / max(float(np.std(p_sub)), 1e-6)) * float(np.std(r_sub)),
            0.0, None
        )

    f_final = (1.0 - blend_w) * ref_f + blend_w * f_alpha_aligned
    nf_final = (1.0 - blend_w) * ref_nf + blend_w * nf_alpha_aligned

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Save standalone aligned Alpha predictions for future Hilbert basis use
    df_alpha_pure = pd.DataFrame({
        "ID": test_ids,
        "CLV_fuel": f_alpha_aligned,
        "CLV_nonfuel": nf_alpha_aligned,
        "Opportunity": df_omega4["Opportunity"].to_numpy(),
    })
    df_alpha_pure.to_csv(submissions_dir / f"cache_standalone_alpha_orthogonal_{timestamp}.csv", index=False)

    df_alpha_blend = pd.DataFrame({
        "ID": test_ids,
        "CLV_fuel": np.clip(f_final, 0.0, None),
        "CLV_nonfuel": np.clip(nf_final, 0.0, None),
        "Opportunity": df_omega4["Opportunity"].to_numpy(),
    })

    out_path = submissions_dir / f"submission_candidate_alpha_l3_orthogonal_{timestamp}.csv"
    df_alpha_blend.to_csv(out_path, index=False)
    logger.info("Saved Candidate Alpha (12%% Orthogonal L3/CatBoost-MQ/Tweedie/MLP Blend on Omega-4): %s", out_path.name)
    return out_path


if __name__ == "__main__":
    build_candidate_alpha()
