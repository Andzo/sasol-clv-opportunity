"""Transaction loading, schema validation and row-level annotation.

Parsing is delegated to :func:`label_rules.prepare` so that the exact integer
micro-unit amounts (``_quantity``/``_amount``) and timestamps used by the official
label generator are reused verbatim — guaranteeing feature/label parity.
"""
from __future__ import annotations

import hashlib
import logging
from typing import Any, Mapping

import pandas as pd

import label_rules
from src.config import PipelineConfig

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "v2"
REQUIRED_COLUMNS: tuple[str, ...] = (
    "DateTimeZA", "TransactionId", "LocationId", "ID", "LineItemType", "ItemName",
    "ItemCategoryLevel2", "Quantity", "TotalExclAmount",
)


def included_categories(label_config: Mapping[str, Any]) -> list[str]:
    """Sorted list of Level2 categories that count as purchase activity."""
    return sorted(c for c, v in label_config["category_mapping"].items() if v["included"])


def validate_raw_schema(raw: pd.DataFrame, label_config: Mapping[str, Any]) -> None:
    """Fail fast on schema drift that would silently corrupt targets or features.

    Raises:
        ValueError: on missing columns, empty IDs, or unmapped Level2 categories
            (e.g. the literal ``"NA"`` category being parsed as a missing value).
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"train.csv is missing required columns: {missing}")
    if (raw["ID"].astype(str).str.len() == 0).any():
        raise ValueError("Empty customer IDs found")
    if raw["ItemCategoryLevel2"].isna().any():
        raise ValueError("ItemCategoryLevel2 contains NaN — read the CSV with keep_default_na=False")
    unknown = set(raw["ItemCategoryLevel2"]) - set(label_config["category_mapping"])
    if unknown:
        raise ValueError(f"Unmapped ItemCategoryLevel2 values: {sorted(unknown)}")


def annotate(tx: pd.DataFrame, label_config: Mapping[str, Any]) -> pd.DataFrame:
    """Add vectorised boolean flags mirroring the label rules.

    Flags:
        is_adjustment: named adjustment item (never counts as activity).
        is_included:   Level2 category included in targets/opportunities.
        in_scope:      contributes to net spend (included and not an adjustment).
        is_qualifying: qualifying purchase (in scope, Quantity > 0 and Amount > 0).
        is_fuel_cat:   Level2 == "Fuel" (opportunity spend is in rand).
        is_fuel_sale:  Fuel & LineItemType == "Sale_Fuel" (litres regression).
        is_nonfuel:    included and not Fuel (non-fuel regression).
    """
    out = tx.copy()
    scale = label_rules.SCALE
    out["amount"] = out["_amount"] / scale
    out["qty"] = out["_quantity"] / scale
    out["is_adjustment"] = out["ItemName"].isin(label_config["adjustment_items"])
    out["is_included"] = out["ItemCategoryLevel2"].isin(included_categories(label_config))
    out["in_scope"] = out["is_included"] & ~out["is_adjustment"]
    out["is_positive"] = (out["_quantity"] > 0) & (out["_amount"] > 0)
    out["is_qualifying"] = out["in_scope"] & out["is_positive"]
    out["is_fuel_cat"] = out["ItemCategoryLevel2"].eq("Fuel")
    out["is_fuel_sale"] = out["is_fuel_cat"] & out["LineItemType"].eq("Sale_Fuel")
    out["is_nonfuel"] = out["is_included"] & ~out["is_fuel_cat"]
    out["ID"] = out["ID"].astype(str)
    return out


def _cache_key(cfg: PipelineConfig) -> str:
    stat = cfg.paths.train_csv.stat()
    payload = f"{SCHEMA_VERSION}|{stat.st_size}|{stat.st_mtime_ns}"
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def load_transactions(cfg: PipelineConfig) -> pd.DataFrame:
    """Load, validate, parse and annotate ``train.csv`` (with a fingerprinted cache).

    The cache key depends on the CSV size/mtime and ``SCHEMA_VERSION`` so a changed
    source file or parsing logic can never be served from a stale cache.
    """
    cache = cfg.paths.cache_dir / f"transactions_{_cache_key(cfg)}.parquet"
    if cfg.use_cache and cache.exists():
        logger.info("Loading cached transactions from %s", cache.name)
        return pd.read_parquet(cache)

    logger.info("Parsing %s", cfg.paths.train_csv.name)
    raw = pd.read_csv(cfg.paths.train_csv, dtype=str, keep_default_na=False)
    validate_raw_schema(raw, cfg.label_config)
    tx = annotate(label_rules.prepare(raw), cfg.label_config)
    cfg.paths.cache_dir.mkdir(parents=True, exist_ok=True)
    tx.to_parquet(cache, index=False)
    logger.info("Cached %d transactions to %s", len(tx), cache.name)
    return tx


def load_test_ids(cfg: PipelineConfig) -> list[str]:
    """Test customer IDs as strings, in file order."""
    return pd.read_csv(cfg.paths.test_csv, dtype=str)["ID"].tolist()
