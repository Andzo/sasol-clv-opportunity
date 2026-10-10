"""Vectorized feature engineering engine for Sasol CLV & Opportunity prediction."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import label_rules

PROJECT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_DIR / 'label_config.json'

with open(CONFIG_PATH, 'r') as f:
    CONFIG = json.load(f)

CATEGORY_MAPPING = CONFIG['category_mapping']
INCLUDED_CATS = sorted([c for c, v in CATEGORY_MAPPING.items() if v['included']])
ADJUSTMENT_ITEMS = set(CONFIG['adjustment_items'])
OPPORTUNITY_LABELS = CONFIG['opportunity_labels']
NON_FUEL_EXCLUDED = {'Fuel', 'Value added services', 'Print media', 'Carrier Bags'}


def extract_customer_features(transactions, cutoff_str, target_ids=None):
    """
    Extracts multi-window behavioral and category features for eligible customers
    strictly using transactions observed before cutoff_str.
    """
    t = pd.Timestamp(cutoff_str)
    
    # Filter history strictly before cutoff and exclude non-purchase adjustment rows
    history = transactions[transactions._time < t]
    valid_history = history[~history.ItemName.isin(ADJUSTMENT_ITEMS)].copy()

    # Determine eligible customers if not explicitly provided
    if target_ids is None:
        min_txns = CONFIG['minimum_historical_transactions']
        target_ids = label_rules.eligible_ids(transactions, t, min_txns)
    else:
        target_ids = pd.Index(target_ids, name='ID')

    if len(target_ids) == 0:
        return pd.DataFrame()

    cols = {}

    # Time boundaries for lookback windows
    t_14d = t - pd.DateOffset(days=14)
    t_30d = t - pd.DateOffset(days=30)
    t_60d = t - pd.DateOffset(days=60)
    t_90d = t - pd.DateOffset(days=90)      # Previous Quarter (Q1)
    t_180d = t - pd.DateOffset(days=180)    # Prior Quarter (Q2 is [t_180d, t_90d))

    # Amounts in regular floats for feature computation
    valid_history['amount_float'] = valid_history['_amount'] / label_rules.SCALE
    valid_history['qty_float'] = valid_history['_quantity'] / label_rules.SCALE

    # In-scope indicators
    is_fuel = (valid_history.ItemCategoryLevel2 == 'Fuel') & (valid_history.LineItemType == 'Sale_Fuel')
    is_nonfuel = ~valid_history.ItemCategoryLevel2.isin(NON_FUEL_EXCLUDED)
    is_positive = (valid_history._quantity > 0) & (valid_history._amount > 0)

    valid_history['is_fuel'] = is_fuel
    valid_history['is_nonfuel'] = is_nonfuel
    valid_history['is_positive'] = is_positive

    # --- A. RECENCY & TENURE FEATURES ---
    last_tx_time = valid_history.groupby('ID')._time.max()
    cols['days_since_last_txn'] = (t - last_tx_time).dt.total_seconds() / 86400.0

    first_tx_time = valid_history.groupby('ID')._time.min()
    cols['tenure_days'] = (t - first_tx_time).dt.total_seconds() / 86400.0

    last_fuel_time = valid_history[is_fuel & is_positive].groupby('ID')._time.max()
    cols['days_since_last_fuel'] = ((t - last_fuel_time).dt.total_seconds() / 86400.0).reindex(target_ids).fillna(999.0)

    last_nonfuel_time = valid_history[is_nonfuel & is_positive].groupby('ID')._time.max()
    cols['days_since_last_nonfuel'] = ((t - last_nonfuel_time).dt.total_seconds() / 86400.0).reindex(target_ids).fillna(999.0)

    # Inter-purchase interval statistics
    tx_dates = valid_history.groupby(['ID', 'TransactionId'])._time.first().reset_index()
    tx_dates = tx_dates.sort_values(['ID', '_time'])
    tx_dates['prev_time'] = tx_dates.groupby('ID')._time.shift(1)
    tx_dates['interval_days'] = (tx_dates._time - tx_dates.prev_time).dt.total_seconds() / 86400.0

    interval_stats = tx_dates.groupby('ID').interval_days.agg(['mean', 'std', 'max', 'median'])
    cols['interval_mean'] = interval_stats['mean'].reindex(target_ids).fillna(999.0)
    cols['interval_std'] = interval_stats['std'].reindex(target_ids).fillna(0.0)
    cols['interval_max'] = interval_stats['max'].reindex(target_ids).fillna(999.0)
    cols['interval_median'] = interval_stats['median'].reindex(target_ids).fillna(999.0)

    # --- B. FREQUENCY & BASKET COUNTS ACROSS WINDOWS ---
    tx_windows = [
        ('14d', valid_history._time >= t_14d),
        ('30d', valid_history._time >= t_30d),
        ('60d', valid_history._time >= t_60d),
        ('90d', valid_history._time >= t_90d),
        ('180d', valid_history._time >= t_180d),
        ('all', pd.Series(True, index=valid_history.index)),
    ]

    for name, mask in tx_windows:
        sub = valid_history[mask]
        cols[f'txns_{name}'] = sub.groupby('ID').TransactionId.nunique().reindex(target_ids, fill_value=0)
        sub_fuel = sub[sub.is_fuel & sub.is_positive]
        cols[f'fuel_txns_{name}'] = sub_fuel.groupby('ID').TransactionId.nunique().reindex(target_ids, fill_value=0)
        sub_nonfuel = sub[sub.is_nonfuel & sub.is_positive]
        cols[f'nonfuel_txns_{name}'] = sub_nonfuel.groupby('ID').TransactionId.nunique().reindex(target_ids, fill_value=0)

    cols['freq_velocity_30_90'] = cols['txns_30d'] / (cols['txns_90d'] / 3.0 + 0.1)
    cols['freq_velocity_14_60'] = cols['txns_14d'] / (cols['txns_60d'] / 4.28 + 0.1)

    # --- C. MONETARY & FUEL LITRES ACROSS WINDOWS ---
    for name, mask in tx_windows:
        sub = valid_history[mask]
        sub_fuel = sub[sub.is_fuel]
        cols[f'fuel_litres_{name}'] = sub_fuel.groupby('ID').qty_float.sum().reindex(target_ids, fill_value=0.0)
        sub_nonfuel = sub[sub.is_nonfuel]
        cols[f'nonfuel_spend_{name}'] = sub_nonfuel.groupby('ID').amount_float.sum().reindex(target_ids, fill_value=0.0)
        cols[f'total_spend_{name}'] = sub.groupby('ID').amount_float.sum().reindex(target_ids, fill_value=0.0)

    cols['avg_basket_spend_90d'] = cols['total_spend_90d'] / (cols['txns_90d'] + 0.001)
    cols['avg_basket_spend_all'] = cols['total_spend_all'] / (cols['txns_all'] + 0.001)
    cols['avg_fuel_litres_per_txn_90d'] = cols['fuel_litres_90d'] / (cols['fuel_txns_90d'] + 0.001)

    cols['fuel_spend_share_90d'] = (cols['total_spend_90d'] - cols['nonfuel_spend_90d']) / (cols['total_spend_90d'] + 1.0)
    cols['fuel_spend_share_all'] = (cols['total_spend_all'] - cols['nonfuel_spend_all']) / (cols['total_spend_all'] + 1.0)

    # --- D. QUARTER-OVER-QUARTER MOMENTUM (Q1 vs Q2) ---
    q2_mask = (valid_history._time >= t_180d) & (valid_history._time < t_90d)
    sub_q2 = valid_history[q2_mask]

    txns_q2 = sub_q2.groupby('ID').TransactionId.nunique().reindex(target_ids, fill_value=0)
    fuel_litres_q2 = sub_q2[sub_q2.is_fuel].groupby('ID').qty_float.sum().reindex(target_ids, fill_value=0.0)
    nonfuel_spend_q2 = sub_q2[sub_q2.is_nonfuel].groupby('ID').amount_float.sum().reindex(target_ids, fill_value=0.0)
    total_spend_q2 = sub_q2.groupby('ID').amount_float.sum().reindex(target_ids, fill_value=0.0)

    cols['txns_q2'] = txns_q2
    cols['fuel_litres_q2'] = fuel_litres_q2
    cols['nonfuel_spend_q2'] = nonfuel_spend_q2
    cols['total_spend_q2'] = total_spend_q2

    cols['txns_diff_q1_q2'] = cols['txns_90d'] - txns_q2
    cols['fuel_litres_diff_q1_q2'] = cols['fuel_litres_90d'] - fuel_litres_q2
    cols['nonfuel_spend_diff_q1_q2'] = cols['nonfuel_spend_90d'] - nonfuel_spend_q2
    cols['total_spend_diff_q1_q2'] = cols['total_spend_90d'] - total_spend_q2

    cols['fuel_litres_ratio_q1_q2'] = (cols['fuel_litres_90d'] + 0.1) / (fuel_litres_q2 + 0.1)
    cols['nonfuel_spend_ratio_q1_q2'] = (cols['nonfuel_spend_90d'] + 1.0) / (nonfuel_spend_q2 + 1.0)
    cols['total_spend_ratio_q1_q2'] = (cols['total_spend_90d'] + 1.0) / (total_spend_q2 + 1.0)

    # --- E. CATEGORY-LEVEL SPENDS, BASELINES & MASKS ---
    cat_items = valid_history[valid_history.ItemCategoryLevel2.isin(INCLUDED_CATS)]
    
    ever_bought_sub = cat_items[cat_items.is_positive]
    cat_ever_bought = ever_bought_sub.groupby(['ID', 'ItemCategoryLevel2']).TransactionId.nunique().unstack(fill_value=0).reindex(index=target_ids, columns=INCLUDED_CATS, fill_value=0).gt(0).astype(int)

    q1_sub = cat_items[cat_items._time >= t_90d]
    cat_q1_spend = q1_sub.groupby(['ID', 'ItemCategoryLevel2']).amount_float.sum().unstack(fill_value=0.0).reindex(index=target_ids, columns=INCLUDED_CATS, fill_value=0.0)

    cat_all_spend = cat_items.groupby(['ID', 'ItemCategoryLevel2']).amount_float.sum().unstack(fill_value=0.0).reindex(index=target_ids, columns=INCLUDED_CATS, fill_value=0.0)

    for cat in INCLUDED_CATS:
        safe_name = cat.replace(' ', '_').replace('&', 'and')
        cols[f'cat_ever_{safe_name}'] = cat_ever_bought[cat]
        cols[f'cat_q1_spend_{safe_name}'] = cat_q1_spend[cat]
        cols[f'cat_all_spend_{safe_name}'] = cat_all_spend[cat]

    cols['num_cats_ever_bought'] = cat_ever_bought.sum(axis=1)
    cols['num_cats_q1_bought'] = (cat_q1_spend > 0).sum(axis=1)
    cols['num_unadopted_cats'] = len(INCLUDED_CATS) - cols['num_cats_ever_bought']

    # --- F. BEHAVIORAL & STORE DIVERSITY ---
    cols['distinct_locations_all'] = valid_history.groupby('ID').LocationId.nunique().reindex(target_ids, fill_value=0)
    cols['distinct_locations_90d'] = valid_history[valid_history._time >= t_90d].groupby('ID').LocationId.nunique().reindex(target_ids, fill_value=0)

    basket_flags = valid_history[valid_history._time >= t_90d].groupby(['ID', 'TransactionId']).agg(
        has_fuel=('is_fuel', 'any'),
        has_nonfuel=('is_nonfuel', 'any')
    )
    basket_flags['is_cross'] = basket_flags.has_fuel & basket_flags.has_nonfuel
    cross_stats = basket_flags.groupby('ID').agg(
        cross_baskets=('is_cross', 'sum'),
        total_baskets=('has_fuel', 'count')
    )
    cols['cross_baskets_90d'] = cross_stats['cross_baskets'].reindex(target_ids, fill_value=0)
    cols['cross_ratio_90d'] = cols['cross_baskets_90d'] / (cross_stats['total_baskets'].reindex(target_ids, fill_value=1) + 0.001)

    valid_history['is_weekend'] = valid_history._time.dt.dayofweek.isin([5, 6])
    cols['weekend_txn_ratio'] = valid_history.groupby('ID').is_weekend.mean().reindex(target_ids, fill_value=0.0)

    valid_history['hour'] = valid_history._time.dt.hour
    hour_stats = valid_history.groupby('ID').hour.agg(['mean', 'std'])
    cols['mean_hour'] = hour_stats['mean'].reindex(target_ids, fill_value=12.0)
    cols['std_hour'] = hour_stats['std'].reindex(target_ids, fill_value=0.0).fillna(0.0)

    # Build DataFrame from dictionary of series
    feat_df = pd.DataFrame(cols, index=target_ids).reset_index()
    return feat_df


def compute_historical_eligibility_mask(transactions, cutoff_str, target_ids):
    """
    Computes a boolean eligibility mask [N_customers, 17_classes] where True indicates
    that the class is mathematically allowed for that customer according to label_rules.
    """
    t = pd.Timestamp(cutoff_str)
    history = transactions[transactions._time < t]
    valid_history = history[~history.ItemName.isin(ADJUSTMENT_ITEMS)].copy()

    is_positive = (valid_history._quantity > 0) & (valid_history._amount > 0)
    cat_items = valid_history[valid_history.ItemCategoryLevel2.isin(INCLUDED_CATS) & is_positive]

    target_idx = pd.Index(target_ids, name='ID')

    cat_ever = cat_items.groupby(['ID', 'ItemCategoryLevel2']).TransactionId.nunique().unstack(fill_value=0).reindex(index=target_idx, columns=INCLUDED_CATS, fill_value=0).gt(0)

    mask = pd.DataFrame(True, index=target_idx, columns=OPPORTUNITY_LABELS)

    # 1. Existing-category growth
    mask['Existing-category growth: Fuel'] = cat_ever['Fuel']
    mask['Existing-category growth: Beverages'] = cat_ever['Beverages']
    mask['Existing-category growth: Food & beverage'] = cat_ever['Food & beverage']
    growth_other_cats = [c for c, v in CATEGORY_MAPPING.items() if v['included'] and v['Existing-category growth'] == 'Other']
    mask['Existing-category growth: Other'] = cat_ever[growth_other_cats].any(axis=1)

    # 2. New category adoption
    specific_adopt_cats = [
        'Beverages', 'Braai & Ice', 'Breads & rolls', 'Confectionary',
        'Food & beverage', 'Groceries', 'Ice-cream', 'Lubricants',
        'Snacking', 'Tobacco'
    ]
    for c in specific_adopt_cats:
        label_name = f'New category adoption: {c}'
        mask[label_name] = ~cat_ever[c]

    adopt_other_cats = [c for c, v in CATEGORY_MAPPING.items() if v['included'] and v['New category adoption'] == 'Other']
    has_unadopted_other = (~cat_ever[adopt_other_cats]).any(axis=1)
    mask['New category adoption: Other'] = has_unadopted_other

    # Inactivity and Stable are always mathematically possible
    mask['Inactivity'] = True
    mask['Stable'] = True

    return mask.to_numpy()
