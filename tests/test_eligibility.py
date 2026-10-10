"""Unit tests for eligibility masking."""
import numpy as np
import pandas as pd
from src.config import PipelineConfig
from src.eligibility import mask_from_history


def test_mask_from_history():
    cfg = PipelineConfig()
    label_cfg = cfg.label_config
    opp_labels = label_cfg["opportunity_labels"]
    
    # 2 customers: c1 has bought only Fuel, c2 has bought only Beverages
    cats = ["Beverages", "Fuel", "Food & beverage"]
    ever = pd.DataFrame(
        [[False, True, False],
         [True, False, False]],
        index=["c1", "c2"],
        columns=cats
    )
    
    mask = mask_from_history(ever, label_cfg)
    assert mask.shape == (2, len(opp_labels))
    assert mask.dtype == bool

    idx_fuel_growth = opp_labels.index("Existing-category growth: Fuel")
    idx_bev_growth = opp_labels.index("Existing-category growth: Beverages")
    idx_bev_adopt = opp_labels.index("New category adoption: Beverages")
    idx_inact = opp_labels.index("Inactivity")
    idx_stable = opp_labels.index("Stable")

    # c1: bought Fuel -> can grow Fuel, CANNOT adopt Fuel (Fuel maps to Other for adoption), cannot grow Beverages
    assert mask[0, idx_fuel_growth] == True
    assert mask[0, idx_bev_growth] == False
    assert mask[0, idx_bev_adopt] == True  # has NOT bought Beverages -> can adopt Beverages
    assert mask[0, idx_inact] == True
    assert mask[0, idx_stable] == True

    # c2: bought Beverages -> cannot grow Fuel, CAN grow Beverages, CANNOT adopt Beverages
    assert mask[1, idx_fuel_growth] == False
    assert mask[1, idx_bev_growth] == True
    assert mask[1, idx_bev_adopt] == False  # HAS bought Beverages -> CANNOT adopt
    assert mask[1, idx_inact] == True
    assert mask[1, idx_stable] == True
