"""Unit tests for feature engineering (src.features)."""
import numpy as np
import pandas as pd
from src.config import PipelineConfig
from src.data import annotate
from src.features import build_features


def test_build_features_schema_and_reproducibility():
    cfg = PipelineConfig()
    label_cfg = cfg.label_config

    # Synthetic historical transaction dataset
    rows = []
    base_time = pd.Timestamp("2024-06-01")
    for i in range(10):
        rows.append({
            "DateTimeZA": str(base_time + pd.Timedelta(days=i * 10)),
            "TransactionId": f"t_{i}",
            "LocationId": "loc_1",
            "ID": "cust_1",
            "LineItemType": "Sale_Fuel" if i % 2 == 0 else "Sale",
            "ItemName": "ULP95" if i % 2 == 0 else "Coffee",
            "ItemCategoryLevel2": "Fuel" if i % 2 == 0 else "Beverages",
            "Quantity": "20" if i % 2 == 0 else "1",
            "TotalExclAmount": "400" if i % 2 == 0 else "30",
            "_quantity": 20000000 if i % 2 == 0 else 1000000,
            "_amount": 400000000 if i % 2 == 0 else 30000000,
            "_time": base_time + pd.Timedelta(days=i * 10),
        })

    df = pd.DataFrame(rows)
    tx = annotate(df, label_cfg)
    cutoff = pd.Timestamp("2024-10-01")
    data_start = base_time
    ids = pd.Index(["cust_1"], name="ID")

    feats1 = build_features(tx, cutoff, ids, label_cfg, data_start)
    feats2 = build_features(tx, cutoff, ids, label_cfg, data_start)

    # 1. Exact reproducibility across repeated calls
    pd.testing.assert_frame_equal(feats1, feats2)

    # 2. Key newly added feature columns must exist
    new_cols = [
        "fuel_m1_m3_ratio",
        "nonfuel_m1_m3_ratio",
        "fuel_reversion_ratio",
        "store_attachment_rate",
        "exp_churn_hazard",
        "fuel_litres_trend_slope",
        "nonfuel_rand_trend_slope",
        "rel_fuel_price_m3",
        "m6_top_loc_mean_litres",
        "m6_top_loc_store_attach",
        "m6_top_loc_weekend_share",
        "m6_top_loc_log_volume",
        "is_holiday_quarter",
        "cutoff_quarter",
        "hist_december_active",
        "hist_december_spend_log",
        "btyd_overall_p_alive",
        "btyd_overall_lambda",
        "btyd_overall_exp_baskets_m3",
        "btyd_fuel_p_alive",
        "btyd_fuel_exp_baskets_m3",
        "btyd_nonfuel_p_alive",
        "btyd_nonfuel_exp_baskets_m3",
        "btyd_exp_monetary_spend",
        "btyd_exp_clv_spend_m3",
        "m6_mean_basket_items",
        "m6_discount_lines",
        "m6_discount_line_share",
        "m6_discount_savings_rand",
        "ever_discount_shopper",
        "fuel_volatility_cv",
        "fuel_monthly_std",
        "fuel_hurdle_zscore",
        "fuel_hurdle_rand",
        "fuel_attainability_q2",
        "fuel_attainability_life",
        "fuel_rebound_ratio",
        "fuel_headroom_rand",
        "nonfuel_volatility_cv",
        "nonfuel_monthly_std",
        "nonfuel_hurdle_zscore",
        "nonfuel_hurdle_rand",
        "nonfuel_attainability_q2",
        "nonfuel_attainability_life",
        "nonfuel_rebound_ratio",
        "nonfuel_headroom_rand",
        "z_diff_fuel_nonfuel",
        "dec24_surge_ratio",
        "m6_fuel_mean_litres",
        "m6_fuel_std_litres",
        "m6_commuter_share",
        "m6_dow_entropy",
    ]
    for col in new_cols:
        assert col in feats1.columns, f"Missing expected feature: {col}"

    assert len(feats1) == 1
    assert feats1.index.name == "ID"
    assert feats1.index[0] == "cust_1"


def test_btyd_monotonicity_and_bounds():
    """Verify BTYD customer base priors obey probability bounds and physical properties."""
    cfg = PipelineConfig()
    label_cfg = cfg.label_config
    base_time = pd.Timestamp("2024-01-01")

    # Cust A: Active fuel spender
    # Cust B: Completely dormant (only 1 purchase 6 months ago)
    # Cust C: Fuel only (never bought store)
    rows = []
    for day in range(0, 150, 15):
        rows.append({
            "DateTimeZA": str(base_time + pd.Timedelta(days=day)),
            "TransactionId": f"t_a_{day}",
            "LocationId": "loc_1",
            "ID": "cust_active",
            "LineItemType": "Sale_Fuel",
            "ItemName": "ULP95",
            "ItemCategoryLevel2": "Fuel",
            "Quantity": "25",
            "TotalExclAmount": "500",
            "_quantity": 25000000,
            "_amount": 500000000,
            "_time": base_time + pd.Timedelta(days=day),
        })

    rows.append({
        "DateTimeZA": str(base_time),
        "TransactionId": "t_dorm_1",
        "LocationId": "loc_1",
        "ID": "cust_dormant",
        "LineItemType": "Sale_Fuel",
        "ItemName": "ULP95",
        "ItemCategoryLevel2": "Fuel",
        "Quantity": "20",
        "TotalExclAmount": "400",
        "_quantity": 20000000,
        "_amount": 400000000,
        "_time": base_time,
    })

    df = pd.DataFrame(rows)
    tx = annotate(df, label_cfg)
    cutoff = pd.Timestamp("2024-07-01")
    ids = pd.Index(["cust_active", "cust_dormant"], name="ID")

    feats = build_features(tx, cutoff, ids, label_cfg, base_time)

    # Probabilities must be within [0, 1]
    for p_col in ["btyd_overall_p_alive", "btyd_fuel_p_alive", "btyd_nonfuel_p_alive"]:
        assert (feats[p_col] >= 0.0).all() and (feats[p_col] <= 1.0).all(), f"Invalid range in {p_col}"

    # Non-negative expectations
    for exp_col in ["btyd_overall_exp_baskets_m3", "btyd_fuel_exp_baskets_m3", "btyd_exp_clv_spend_m3"]:
        assert (feats[exp_col] >= 0.0).all(), f"Negative expectation in {exp_col}"

    # Active customer should have higher expected fuel transactions than dormant customer
    assert feats.loc["cust_active", "btyd_fuel_exp_baskets_m3"] > feats.loc["cust_dormant", "btyd_fuel_exp_baskets_m3"]
    # Neither customer bought nonfuel goods -> nonfuel p_alive should be 0.0
    assert feats.loc["cust_active", "btyd_nonfuel_p_alive"] == 0.0
    assert feats.loc["cust_dormant", "btyd_nonfuel_p_alive"] == 0.0
