"""Unit tests for src.data and src.labels modules."""
import pandas as pd
import pytest
from src.config import PipelineConfig
from src.data import annotate, included_categories, validate_raw_schema
from src.labels import eligible_ids


def test_included_categories():
    cfg = PipelineConfig()
    cats = included_categories(cfg.label_config)
    assert isinstance(cats, list)
    assert len(cats) > 0
    assert "Fuel" in cats
    assert "Beverages" in cats
    assert "Value added services" not in cats
    assert "Print media" not in cats
    assert "Carrier Bags" not in cats


def test_validate_raw_schema_detects_nan():
    cfg = PipelineConfig()
    df = pd.DataFrame({
        "DateTimeZA": ["2024-03-19"],
        "TransactionId": ["t1"],
        "LocationId": ["l1"],
        "ID": ["c1"],
        "LineItemType": ["Sale"],
        "ItemName": ["Item1"],
        "ItemCategoryLevel2": [float("nan")],
        "Quantity": ["1"],
        "TotalExclAmount": ["10"],
    })
    with pytest.raises(ValueError, match="ItemCategoryLevel2 contains NaN"):
        validate_raw_schema(df, cfg.label_config)


def test_annotate_flags():
    cfg = PipelineConfig()
    df = pd.DataFrame({
        "DateTimeZA": ["2024-03-19", "2024-03-19"],
        "TransactionId": ["t1", "t2"],
        "LocationId": ["l1", "l1"],
        "ID": ["c1", "c1"],
        "LineItemType": ["Sale_Fuel", "Sale"],
        "ItemName": ["ULP95", "TOTAL/ROUNDING DISCO"],
        "ItemCategoryLevel2": ["Fuel", "Beverages"],
        "Quantity": ["20", "1"],
        "TotalExclAmount": ["400", "-5"],
        "_quantity": [20000000, 1000000],
        "_amount": [400000000, -5000000],
        "_time": [pd.Timestamp("2024-03-19"), pd.Timestamp("2024-03-19")],
    })
    annotated = annotate(df, cfg.label_config)
    assert annotated.loc[0, "is_fuel_cat"] == True
    assert annotated.loc[0, "is_fuel_sale"] == True
    assert annotated.loc[0, "in_scope"] == True
    assert annotated.loc[0, "is_qualifying"] == True
    assert annotated.loc[1, "is_adjustment"] == True
    assert annotated.loc[1, "in_scope"] == False
    assert annotated.loc[1, "is_qualifying"] == False
