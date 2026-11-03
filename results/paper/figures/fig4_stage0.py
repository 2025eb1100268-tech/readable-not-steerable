"""Figure 4 (exploratory, E): readable is not steerable. Effect and shape selectivity of every candidate direction
(PC1-4 of the pooled latent) and of the supervised shape direction in the 21 existing models, on the explore and
matched-energy sets.

Source: results/build/stage0.json.gz @ ae98ffa (results.<model>[seed].directions.<dir>.<set>). Effects below
1e-4 are drawn in a marked strip at the left edge. Two floors are drawn: 0.1, the Stage 0 threshold
(PREDICTIONS entry 1), and 0.2, the floor committed afterwards (entry 2).

    python results/paper/figures/fig4_stage0.py
"""
import gzip
import json
import zlib

import numpy as np
from matplotlib.lines import Line2D

from _common import AQUA, BLUE, FULL_W, INK, INK2, MUTED, ORANGE, REPO, plt, save, style

KINDS = {  # model -> (colour by family, marker by render target)
    "vorticity AE": (BLUE, "o"), "velocity AE": (BLUE, "s"), "streamfunction AE": (BLUE, "^"),
    "vorticity AE + VICReg": (ORANGE, "o"), "velocity AE + VICReg": (ORANGE, "s"),
    "streamfunction AE + VICReg": (ORANGE, "^"), "imaginer k=4": (AQUA, "D"),
}
CLIP = 1e-4


def main():
    r = json.load(gzip.open(REPO / "results" / "build" / "stage0.json.gz", "rt", encoding="utf-8"))["results"]
    style()
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.9), sharey=True, constrained_layout=True)
    n_clipped = {"explore": 0, "matched": 0}
    for ax, (st, title) in zip(axes, (("explore", "explore set"), ("matched", "matched-energy set"))):
        ax.axvspan(5e-5, 1.3e-4, color="#f0efec", zorder=0)
        ax.text(8e-5, 1.02, "< 10⁻⁴", ha="center", va="bottom", fontsize=6.3, color=INK2)
        for model, (col, mk) in KINDS.items():
            for run in r[model]:
                for name, dd in run["directions"].items():
                    m = dd[st]
                    eff, sig = m["effect"], m["sigma_shape"]
                    if not np.isfinite(sig):
                        continue
                    sup = name == "supervised s"
                    x = eff
                    if eff < CLIP:
                        j = zlib.crc32(f"{model}|{run['run']}|{name}".encode()) % 1000 / 1000  # deterministic jitter
                        x = 8e-5 * (0.8 + 0.4 * j)
                        n_clipped[st] += 1
                    ax.scatter(x, sig, s=16 if not sup else 22, marker=mk, zorder=3,
                               facecolors="white" if sup else col, edgecolors=col, linewidths=0.9 if sup else 0,
                               alpha=0.9 if sup else 0.7)
        ax.axvline(0.1, color=INK, lw=0.9, ls=(0, (4, 2)))
        ax.axvline(0.2, color=INK, lw=0.9, ls=(0, (1, 1.5)))
        for y, lab in ((0.2, "loudness ≤ 0.2"), (0.7, "shape ≥ 0.7")):
            ax.axhline(y, color=MUTED, lw=0.7, ls=(0, (3, 2)))
            ax.text(1.9, y + 0.01, lab, fontsize=6.2, color=INK2, va="bottom", ha="right")
        ax.set_xscale("log")
        ax.set_xlim(5e-5, 2.0)
        ax.set_ylim(-0.03, 1.08)
        ax.set_xlabel("effect |Δa| + |Δs| per unit α (log-units)")
        ax.set_title(title, loc="left")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("shape selectivity σ_shape")
    handles = [Line2D([], [], color=BLUE, marker="s", ls="none", label="autoencoder"),
               Line2D([], [], color=ORANGE, marker="s", ls="none", label="autoencoder + VICReg"),
               Line2D([], [], color=AQUA, marker="D", ls="none", label="imaginer k = 4"),
               Line2D([], [], color=INK2, marker="o", ls="none", label="renders ω"),
               Line2D([], [], color=INK2, marker="s", ls="none", label="renders u"),
               Line2D([], [], color=INK2, marker="^", ls="none", label="renders ψ"),
               Line2D([], [], color=INK2, marker="o", ls="none", label="PC1–4 (filled)"),
               Line2D([], [], markeredgecolor=INK2, markerfacecolor="white", marker="o", ls="none",
                      label="supervised shape direction (hollow)"),
               Line2D([], [], color=INK, lw=0.9, ls=(0, (4, 2)), label="effect floor at Stage 0 (0.1)"),
               Line2D([], [], color=INK, lw=0.9, ls=(0, (1, 1.5)), label="effect floor committed later (0.2)")]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, fontsize=6.5, handletextpad=0.3)
    print("clipped (< 1e-4):", n_clipped)
    save(fig, "fig4_stage0")


if __name__ == "__main__":
    main()
