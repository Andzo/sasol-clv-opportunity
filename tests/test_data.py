"""Unit tests for src.data, src.labels, src.label_rules, and missing-data handling."""
import pandas as pd
import pytest
from src import label_rules
from src.config import Paths, PipelineConfig
from src.data import annotate, included_categories, load_test_ids, load_transactions, resolve_data_file, validate_raw_schema
from src.labels import eligible_ids
from src.main import main as cli_main


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


def test_label_config_resolves_inside_src(tmp_path):
    cfg = PipelineConfig(paths=Paths(root=tmp_path))
    assert cfg.paths.label_config.name == "label_config.json"
    assert cfg.paths.label_config.parent.name == "src"
    assert "opportunity_labels" in cfg.label_config
    assert label_rules.SCALE == 1_000_000


def test_missing_local_csv_raises_clear_file_not_found(tmp_path):
    cfg = PipelineConfig(paths=Paths(root=tmp_path))
    with pytest.raises(FileNotFoundError, match="Zindi"):
        load_transactions(cfg)
    with pytest.raises(FileNotFoundError, match="Zindi"):
        load_test_ids(cfg)


def test_resolve_data_file_supports_data_subdir_fallback(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    target = data_dir / "test.csv"
    target.write_text("ID\n1\n")
    resolved = resolve_data_file(tmp_path / "test.csv", "test customer IDs CSV")
    assert resolved == target
    cfg = PipelineConfig(paths=Paths(root=tmp_path))
    assert load_test_ids(cfg) == ["1"]


def test_cli_main_help_and_missing_data_exit_cleanly(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["src.main", "--help"])
    with pytest.raises(SystemExit) as exc:
        cli_main()
    assert exc.value.code == 0

    monkeypatch.setattr("src.main.PipelineConfig", lambda **kw: PipelineConfig(paths=Paths(root=tmp_path)))
    monkeypatch.setattr("sys.argv", ["src.main", "--fast"])
    rc = cli_main()
    captured = capsys.readouterr()
    assert rc == 1
    assert "Zindi" in captured.err
