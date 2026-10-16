"""[EXPLORE] Figure for entry 6 step 3: render-target autoencoders with and without VICReg, next to
the k = 4 imaginer, the k = 4 summarizer and the untrained control (pooled latent, explore set).

    python -m explore.ae_figure --eval results/explore/stage2/eval.json.gz --out results/explore/stage2/autoencoders.png
"""
from __future__ import annotations

import argparse
import gzip
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common.run import REPO  # noqa: E402
from solver.validate import INK, INK2, SERIES, _style  # noqa: E402

TARGETS = (("ψ", "streamfunction"), ("u", "velocity"), ("ω", "vorticity"))
PANELS = [("E1: explore enstrophy |r| (slow directions)", lambda r: r["slow_explore"]["max_abs_r"]["log_Z"], None),
          ("E4: energy : enstrophy mix a/b (1 = log Z)", lambda r: r["direction_stats"]["E4_ratio_a_over_b"], "log"),
          ("Linear decodability: log P R² (S1 probe)", lambda r: r["probes"]["log_P"]["r2_partial"], None)]
REFS = (("untrained control", "untrained", INK2, (0, (3, 2))), ("imaginer k=4", "imaginer k=4", SERIES[2], (0, (1, 1.5))),
        ("summarizer k=4", "summarizer k=4", SERIES[3], (0, (5, 2))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    with gzip.open(REPO / a.eval, "rt", encoding="utf-8") as f:
        res = json.load(f)["results"]["pooled"]
    _style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    for ax, (title, get, scale) in zip(axes, PANELS):
        for lab, g, col, ls in REFS:
            ax.axhline(np.mean([get(r) for r in res[g]]), color=col, linewidth=1.3, linestyle=ls, label=lab)
        for i, (sym, _) in enumerate(TARGETS):
            for j, (suffix, col, name) in enumerate(((" ", SERIES[0], "plain"), (" + VICReg", SERIES[1], "+ VICReg"))):
                g = f"AE render {sym}" + ("" if j == 0 else " + VICReg")
                v = np.array([get(r) for r in res[g]])
                x = i + (j - 0.5) * 0.3
                ax.scatter(np.full(len(v), x), v, s=22, color=col, linewidths=0, zorder=3)
                ax.plot([x - 0.1, x + 0.1], [v.mean()] * 2, color=col, linewidth=2.4, zorder=4,
                        label=f"autoencoder, {name}" if i == 0 else None)
        ax.set_xticks(range(3))
        ax.set_xticklabels([f"renders {sym}" for sym, _ in TARGETS])
        if scale:
            ax.set_yscale(scale)
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
    axes[0].legend(fontsize=7.5, loc="lower right")
    fig.suptitle("[EXPLORE] Render without imagining: autoencoders by render target, with and without VICReg (exploratory)",
                 fontweight="bold", color=INK)
    fig.savefig(REPO / a.out, dpi=125)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
