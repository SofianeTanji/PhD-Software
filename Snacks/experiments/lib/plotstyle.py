"""Shared figure style for paper plots.

Single source of truth for the custom matplotlib style and the BuRd colormap so
that every figure (the existing heatmap/convergence plots and the new
hyperparameter-robustness plots) is visually identical.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Snacks brand color (matches plot_results._SOLVER_COLOR["RASSG-r"]).
SNACKS_COLOR = "#194F8C"

_STYLE_PATH = Path(__file__).resolve().parent.parent.parent / "custom_style.mplstyle"
_BURD_COLORS = [
    "#2166AC",
    "#4393C3",
    "#92C5DE",
    "#D1E5F0",
    "#F7F7F7",
    "#FDDBC7",
    "#F4A582",
    "#D6604D",
    "#B2182B",
]


def _register_burd() -> None:
    if "BuRd" in plt.colormaps():
        return
    cmap = LinearSegmentedColormap.from_list("BuRd", _BURD_COLORS)
    cmap.set_bad("#FFEE99")
    plt.colormaps.register(cmap)


def apply_style() -> None:
    """Register BuRd and apply the repo's custom matplotlib style."""
    _register_burd()
    plt.style.use(str(_STYLE_PATH))


def save_fig(fig, out: Path, stem: str, png: bool = False) -> Path:
    """Save a figure as PDF to ``out/{stem}.pdf`` (tight bbox); optionally PNG too."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{stem}.pdf"
    fig.savefig(path, bbox_inches="tight")
    print(f"Saved: {path}")
    if png:
        png_path = out / f"{stem}.png"
        fig.savefig(png_path, bbox_inches="tight")
        print(f"Saved: {png_path}")
    return path
