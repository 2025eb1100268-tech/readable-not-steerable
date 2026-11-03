"""Figure 2 (exploratory, E): plain autoencoders that render the present match the imaginer's enstrophy
prominence; adding VICReg drops it to control level while raising linear decodability. Appendix figure
fig2b_e4: the energy-enstrophy mix E4 of the same models.

Source: results/explore/stage2/eval.json.gz @ 244f625 (summary.pooled; explore set, per seed).

    python results/paper/figures/fig2_autoencoders.py
"""
import gzip
import json

import numpy as np

from _common import AQUA, BLUE, FULL_W, HALF_W, INK, INK2, MUTED, ORANGE, REPO, plt, save, style

TARGETS = [("ψ", "streamfunction"), ("u", "velocity"), ("ω", "vorticity")]


def refline(ax, y, label, ls, color=MUTED):
    ax.axhline(y, color=color, lw=0.9, ls=ls, zorder=1)
    ax.text(2.48, y, f" {label}", fontsize=6.4, color=INK2, va="center", ha="left", clip_on=False)


def main():
    s = json.load(gzip.open(REPO / "results" / "explore" / "stage2" / "eval.json.gz", "rt", encoding="utf-8"))
    s = s["summary"]["pooled"]
    ref = {k: float(np.mean(s[k]["E1"])) for k in ("untrained", "imaginer k=4", "summarizer k=4")}
    refP = {k: float(np.mean(s[k]["S1"]["log_P"])) for k in ("untrained", "imaginer k=4", "summarizer k=4")}
    style()
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.6), constrained_layout=True)
    panels = [(axes[0], lambda g: s[g]["E1"], "E1: enstrophy |r| of the slowest directions", ref),
              (axes[1], lambda g: s[g]["S1"]["log_P"], "linear decodability: log palinstrophy R²", refP)]
    for ax, get, title, rf in panels:
        for i, (sym, _) in enumerate(TARGETS):
            for j, (suffix, col, lab) in enumerate((("", BLUE, "plain"), (" + VICReg", ORANGE, "+ VICReg"))):
                v = np.array(get(f"AE render {sym}{suffix}"))
                x = i + (-0.17 if j == 0 else 0.17)
                ax.scatter(x + np.array([-0.05, 0, 0.05]), v, s=10, color=col, alpha=0.6, linewidths=0, zorder=3)
                ax.plot([x - 0.11, x + 0.11], [v.mean()] * 2, color=col, lw=2, zorder=4,
                        label=f"autoencoder, {lab}" if i == 0 else None)
        refline(ax, rf["untrained"], "untrained", (0, (4, 2)))
        refline(ax, rf["imaginer k=4"], "imaginer k=4", (0, (1, 1.5)), INK2)
        refline(ax, rf["summarizer k=4"], "summarizer k=4", (0, (6, 2, 1, 2)), INK2)
        ax.set_xticks(range(3), [f"renders {sym}" for sym, _ in TARGETS])
        ax.set_xlim(-0.5, 2.45)
        ax.grid(axis="x", visible=False)
        ax.set_title(title, loc="left")
    axes[0].set_ylim(0.2, 0.95)
    axes[0].legend(loc="upper left", fontsize=6.6, handlelength=1.2)
    save(fig, "fig2_autoencoders")

    # appendix panel: E4 (energy : enstrophy mix), log scale
    fig, ax = plt.subplots(figsize=(HALF_W + 0.6, 2.4), constrained_layout=True)
    for i, (sym, _) in enumerate(TARGETS):
        for j, (suffix, col) in enumerate((("", BLUE), (" + VICReg", ORANGE))):
            v = np.array(s[f"AE render {sym}{suffix}"]["E4_ratio"])
            x = i + (-0.17 if j == 0 else 0.17)
            ax.scatter(x + np.array([-0.05, 0, 0.05]), v, s=10, color=col, alpha=0.6, linewidths=0, zorder=3)
            ax.plot([x - 0.11, x + 0.11], [v.mean()] * 2, color=col, lw=2, zorder=4,
                    label=("plain" if j == 0 else "+ VICReg") if i == 0 else None)
    ax.axhline(1.0, color=INK, lw=0.8, zorder=1)
    ax.text(2.48, 1.0, " a/b = 1 (log Z₀ only)", fontsize=6.4, color=INK2, va="center", clip_on=False)
    refline(ax, float(np.mean(s["untrained"]["E4_ratio"])), "untrained", (0, (4, 2)))
    ax.set_yscale("log")
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    ax.yaxis.set_major_locator(FixedLocator([0.5, 1, 2, 4, 6]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xticks(range(3), [f"renders {sym}" for sym, _ in TARGETS])
    ax.set_xlim(-0.5, 2.45)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("E4 = a/b (log scale)")
    ax.set_title("energy : enstrophy mix of the enstrophy-tracking direction", loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(0.38, 0.88), fontsize=6.6, handlelength=1.2)
    save(fig, "fig2b_e4")


if __name__ == "__main__":
    main()
