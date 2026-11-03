"""Figure 6 (exploratory, E): the shape-noise dose with both decoders. Shape selectivity of the loudness-null
direction, seeds with both dials, and rendering error, against the calibrated dose (content shape R²).

Sources: results/build/stepB.json.gz @ 5234145 (sub-pixel decoder, τ = 0, 0.5, 1, 2), stepC.json.gz @ f0f6c1f
(resize-convolution, τ = 0.5), stepD.json.gz @ 6a3420d (resize-convolution, τ = 0.2); calibrated doses from
results/build/stepB_calibration.md and stepD_calibration.md (as in dials/build_report_figure.py). The curve
mixes decoders: τ = 0, 1 and 2 exist only with the sub-pixel decoder.

    python results/paper/figures/fig6_dose.py
"""
import gzip
import json

import numpy as np
from matplotlib.lines import Line2D

from _common import BLUE, FULL_W, INK, INK2, MUTED, ORANGE, REPO, plt, save, style

DOSE = {0.0: 0.966, 0.2: 0.611, 0.5: 0.294, 1.0: 0.181, 2.0: 0.066}  # τ -> calibrated content shape R²
TAU_LABEL = {0.966: "0", 0.611: "0.2", 0.294: "0.5", 0.181: "1", 0.066: "2"}


def load(name):
    return json.load(gzip.open(REPO / "results" / "build" / name, "rt", encoding="utf-8"))["results"]


def main():
    sets = [("sub-pixel decoder", load("stepB.json.gz"), BLUE, "o", -1),
            ("resize-convolution decoder", {**load("stepD.json.gz"), **load("stepC.json.gz")}, ORANGE, "s", +1)]
    style()
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 2.55), constrained_layout=True)
    for lab, res, col, mk, side in sets:
        taus = sorted(res, key=float)
        x = np.array([DOSE[float(t)] for t in taus])
        # (a) shape selectivity of the loudness-null direction
        for t, xx in zip(taus, x):
            ys = [r["shape"]["loudness-null"]["explore"]["sigma_shape"] for r in res[t]]
            axes[0].scatter([xx + side * 0.012] * len(ys), ys, color=col, marker=mk, s=12, alpha=0.6, linewidths=0, zorder=3)
        med = [np.median([r["shape"]["loudness-null"]["explore"]["sigma_shape"] for r in res[t]]) for t in taus]
        axes[0].plot(x, med, color=col, marker=mk, markersize=4, zorder=4)
        # (b) seeds with both dials, all criteria; rules dodged horizontally so lines never overlap
        for rule, ls, fill, off in (("loudness-null", "-", col, -0.02), ("PC2", (0, (3, 2)), "white", +0.02)):
            ys = [sum(bool(r["both"][rule]) for r in res[t]) for t in taus]
            axes[1].plot(x + off + side * 0.006, ys, color=col, ls=ls, marker=mk, markersize=4.5,
                         markerfacecolor=fill, markeredgecolor=col, zorder=4)
        # (c) rendering error
        for t, xx in zip(taus, x):
            ys = [r["rendering"]["omega"] for r in res[t]]
            axes[2].scatter([xx + side * 0.012] * len(ys), ys, color=col, marker=mk, s=12, alpha=0.6, linewidths=0, zorder=3)
        axes[2].plot(x, [np.median([r["rendering"]["omega"] for r in res[t]]) for t in taus], color=col, marker=mk,
                     markersize=4, zorder=4)
    axes[0].axhline(0.7, color=MUTED, lw=0.8, ls=(0, (3, 2)))
    axes[0].text(0.02, 0.715, "shape ≥ 0.7", fontsize=6.3, color=INK2, va="bottom", ha="right")
    axes[0].set_ylabel("σ_shape (loudness-null)")
    axes[0].set_title("(a) shape-knob selectivity", loc="left")
    axes[1].set_yticks([0, 1, 2, 3])
    axes[1].set_ylim(-0.3, 3.3)
    axes[1].set_ylabel("seeds of 3")
    axes[1].set_title("(b) seeds with both dials", loc="left")
    axes[1].text(0.5, 0.35, "sub-pixel: 0 of 3 under\nboth rules at every dose", fontsize=6.2, color=INK2,
                 ha="center", va="bottom")
    axes[2].axhline(0.0105, color=MUTED, lw=0.8, ls=(0, (1, 1.5)))
    axes[2].text(0.02, 0.0112, "VICReg ω autoencoder", fontsize=6.3, color=INK2, va="bottom", ha="right")
    axes[2].set_yscale("log")
    axes[2].set_ylabel("ω error (val, normalized MSE)")
    axes[2].set_title("(c) rendering cost", loc="left")
    for ax in axes:
        ax.set_xlim(1.02, -0.02)  # less shape in the content to the right
        ax.set_xlabel("content shape R² (calibrated dose)")
        top = ax.secondary_xaxis("top")
        top.set_xticks(list(TAU_LABEL), [f"τ={v}" if v == "0" else v for v in TAU_LABEL.values()], fontsize=6)
        top.tick_params(length=2, pad=1)
        top.spines["top"].set_visible(False)
    handles = [Line2D([], [], color=BLUE, marker="o", markersize=4, label="sub-pixel decoder"),
               Line2D([], [], color=ORANGE, marker="s", markersize=4, label="resize-convolution decoder"),
               Line2D([], [], color=INK2, marker="o", markersize=4, label="(b) loudness-null shape rule"),
               Line2D([], [], color=INK2, ls=(0, (3, 2)), marker="o", markerfacecolor="white", markersize=4,
                      label="(b) PC2 shape rule"),
               Line2D([], [], color=INK2, marker="o", ls="none", alpha=0.6, markersize=3, label="(a, c) seeds; line: median")]
    fig.legend(handles=handles, loc="outside lower center", ncol=5, fontsize=6.4, handletextpad=0.3, columnspacing=1.0)
    save(fig, "fig6_dose")


if __name__ == "__main__":
    main()
