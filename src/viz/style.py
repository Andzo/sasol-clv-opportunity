"""Styling, color palettes, and plotting standards for Sasol EDA suite."""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

# Primary color palette (Colorblind-friendly Wong/Okabe-Ito + extensions)
PALETTE = {
    "fuel": "#0072B2",        # Blue
    "nonfuel": "#D55E00",     # Vermillion
    "active": "#009E73",      # Green
    "inactive": "#CC79A7",    # Purple-Pink
    "neutral": "#56B4E9",     # Sky Blue
    "dark": "#222222",        # Near black
    "light": "#F0F0F0",       # Light gray
    "accent": "#E69F00",      # Yellow-Orange
}

# Fixed color map for the 17 authoritative Opportunity classes
OPPORTUNITY_COLORS: dict[str, str] = {
    "Inactivity": "#D55E00",                               # Vermillion (distinct warning)
    "Stable": "#56B4E9",                                   # Sky blue (neutral baseline)
    "Existing-category growth: Fuel": "#0072B2",           # Deep blue
    "Existing-category growth: Beverages": "#009E73",      # Bluish green
    "Existing-category growth: Food & beverage": "#E69F00",# Amber / Orange
    "Existing-category growth: Other": "#CC79A7",          # Purple / Pink
    "New category adoption: Beverages": "#882255",         # Wine
    "New category adoption: Food & beverage": "#44AA99",   # Teal
    "New category adoption: Confectionary": "#117733",     # Forest green
    "New category adoption: Snacking": "#999933",          # Olive
    "New category adoption: Tobacco": "#AA4499",           # Magenta
    "New category adoption: Breads & rolls": "#DDCC77",    # Sand
    "New category adoption: Groceries": "#661100",         # Dark red
    "New category adoption: Lubricants": "#332288",        # Indigo
    "New category adoption: Ice-cream": "#88CCEE",         # Light cyan
    "New category adoption: Braai & Ice": "#444444",       # Slate
    "New category adoption: Other": "#999999",             # Medium gray
}


def apply_style() -> None:
    """Apply consistent styling across matplotlib plots."""
    mpl.rcParams.update({
        "figure.dpi": 150,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#CCCCCC",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": "#EEEEEE",
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.family": "sans-serif",
        "font.size": 9.5,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "legend.fontsize": 8.5,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "#E0E0E0",
    })
