"""Shared style, data guard and output helpers for the paper figures (results/paper/figures/fig*.py).

Guard: every figure script imports this module first. It refuses to open any file whose path names a
sealed or retired set (confirm_v2, confirm3, matched-energy-confirm) or the v1 test splits, and
`check_parts` refuses a CombinedDataset spec that would load them. Figures 1-4 and 6 read committed
outputs only; figures 5 and 7 decode frozen checkpoints on exploration frames.
"""
from __future__ import annotations

import builtins
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

SEALED = ("confirm_v2", "confirm3", "matched_energy_confirm", "matched-energy-confirm")


def _check_path(path) -> None:
    try:
        p = os.fspath(path).replace("\\", "/").lower()
    except TypeError:
        return
    if any(s in p for s in SEALED):
        raise SystemExit(f"refused: {path} names a sealed or retired set")


_open, _np_load = builtins.open, np.load


def _guarded_open(file, *a, **k):
    _check_path(file)
    return _open(file, *a, **k)


def _guarded_np_load(file, *a, **k):
    _check_path(file)
    return _np_load(file, *a, **k)


builtins.open = _guarded_open
np.load = _guarded_np_load


def check_parts(parts) -> None:
    """Refuse a CombinedDataset spec that names a sealed set or a v1 split other than train/val."""
    for path, splits in parts:
        _check_path(path)
        if "data/generated/v1" in path.replace("\\", "/") and set(splits) - {"train", "val"}:
            raise SystemExit(f"refused: {path} {splits} would load the retired v1 test splits")


# ---- style: one font family, readable sizes, the validated palette (dataviz reference slots 1-3) ----
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"  # validated all-pairs on white (CVD ΔE ≥ 9.2)
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
NEG2, NEG1, POS1, POS2 = "#184f95", "#6da7ec", "#ec835a", "#c23b22"  # α = −2, −1, +1, +2 (blue ↔ red; 0 = ink)
FIELD_CMAP = "RdBu_r"  # diverging, white at zero
FULL_W, HALF_W = 6.5, 3.25  # inches (TMLR text width)


def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "legend.frameon": False,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.5, "axes.axisbelow": True, "lines.linewidth": 1.5, "lines.markersize": 4,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "figure.dpi": 150,
    })


PREVIEW = os.environ.get("FIG_PREVIEW_DIR")  # optional PNG previews for visual checks (not committed)


def save(fig, name: str) -> Path:
    """Vector PDF with neutral metadata (no author, no tool version), plus an optional PNG preview."""
    path = OUT / f"{name}.pdf"
    fig.savefig(path, metadata={"Title": name, "Author": None, "Subject": None, "Keywords": None,
                                "Creator": None, "Producer": None, "CreationDate": None})
    if PREVIEW:
        Path(PREVIEW).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(PREVIEW) / f"{name}.png", dpi=170)
    plt.close(fig)
    print("wrote", path.relative_to(REPO))
    return path
