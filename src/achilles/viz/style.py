"""Shared plotting style so every figure looks like it belongs to one report."""
from __future__ import annotations

import matplotlib as mpl

# Palette (colour-blind safe-ish, calm).
SPEED_COLORS = {2.5: "#4C78A8", 3.5: "#F58518", 4.5: "#E45756"}
INK = "#222222"
MUTED = "#888888"
GOOD = "#54A24B"
WARN = "#E45756"
ACCENT = "#4C78A8"
ACCENT2 = "#F58518"


def apply_house_style() -> None:
    mpl.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.edgecolor": INK,
        "axes.linewidth": 0.9,
        "axes.grid": True,
        "grid.color": "#DDDDDD",
        "grid.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })
