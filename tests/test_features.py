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
        "fuel_litres_m1_m3_ratio",
        "nonfuel_m1_m3_ratio",
        "fuel_reversion_ratio",
        "fuel_growth_hurdle_rand",
        "store_attachment_rate",
        "forecourt_only",
        "exp_churn_hazard",
        "fuel_overdue_ratio",
        "fuel_rand_lag1",
        "fuel_rand_trend_slope",
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
    ]
    for col in new_cols:
        assert col in feats1.columns, f"Missing expected feature: {col}"

    assert len(feats1) == 1
    assert feats1.index.name == "ID"
    assert feats1.index[0] == "cust_1"
