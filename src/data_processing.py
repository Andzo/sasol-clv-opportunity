"""Data processing and snapshot generation for Sasol CLV & Opportunity challenge."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import label_rules

PROJECT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_DIR / 'label_config.json'
PARQUET_PATH = PROJECT_DIR / 'train.parquet'
CSV_PATH = PROJECT_DIR / 'train.csv'


def load_config():
    with open(CONFIG_PATH, 'r') as f:
        return json.load(f)


def load_cleaned_transactions():
    """Loads transactions with precomputed integer scaled amounts and parsed datetime."""
    if PARQUET_PATH.exists():
        d = pd.read_parquet(PARQUET_PATH)
        if not pd.api.types.is_datetime64_any_dtype(d['DateTimeZA']):
            d['_time'] = pd.to_datetime(d['DateTimeZA'], format='mixed')
        else:
            d['_time'] = d['DateTimeZA']
        d['ItemCategoryLevel2'] = d['ItemCategoryLevel2'].fillna('NA')
    else:
        d = pd.read_csv(CSV_PATH, dtype=str, keep_default_na=False)
        d['_time'] = pd.to_datetime(d['DateTimeZA'], format='mixed')
        d['ItemCategoryLevel2'] = d['ItemCategoryLevel2'].fillna('NA')

    if '_quantity' not in d.columns or '_amount' not in d.columns:
        from decimal import Decimal
        SCALE = label_rules.SCALE
        for name, key in [('Quantity', '_quantity'), ('TotalExclAmount', '_amount')]:
            values = {}
            for v in d[name].unique():
                scaled = Decimal(str(v)) * SCALE
                values[v] = int(scaled)
            d[key] = d[name].map(values).astype('int64')

    return d


def get_snapshot_labels(transactions, cutoff_str, config=None):
    """Generates ground truth labels for a historical cutoff date."""
    if config is None:
        config = load_config()
    raw = label_rules.raw_labels(transactions, cutoff_str, config)
    transformed = label_rules.transform(raw, config)
    transformed['fuel_litres'] = raw['fuel_litres']
    transformed['nonfuel_rand'] = raw['nonfuel_rand']
    transformed['winning_source_category'] = raw['winning_source_category']
    transformed['winning_increment_rand'] = raw['winning_increment_rand']
    return transformed


def get_test_customer_ids(test_csv_path=None):
    """Loads test customer IDs."""
    if test_csv_path is None:
        test_csv_path = PROJECT_DIR / 'test.csv'
    df = pd.read_csv(test_csv_path)
    return df['ID'].astype(str).tolist()
