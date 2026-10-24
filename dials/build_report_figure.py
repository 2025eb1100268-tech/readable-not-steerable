"""[BUILD] Build-report summary figure: the shape-noise dose with both decoders, against the calibrated dose
(content shape R^2). From results/build/stepB.json.gz (pixel-shuffle, tau 0 / 0.5 / 1 / 2), stepC.json.gz
(resize-convolution, tau 0.5) and stepD.json.gz (resize-convolution, tau 0.2). No new measurement.

    python -m dials.build_report_figure
"""
from __future__ import annotations

import gzip
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common.run import REPO  # noqa: E402
from solver.validate import INK, INK2, SERIES, _style  # noqa: E402

DOSE = {0.0: 0.966, 0.2: 0.611, 0.5: 0.294, 1.0: 0.181, 2.0: 0.066}


def load(name):
    return json.load(gzip.open(REPO / "results" / "build" / name, "rt", encoding="utf-8"))["results"]


def main():
    sets = {"pixel-shuffle decoder": load("stepB.json.gz"),
            "resize-convolution decoder": {**load("stepD.json.gz"), **load("stepC.json.gz")}}
    _style()
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), constrained_layout=True)
    for i, (lab, res) in enumerate(sets.items()):
        taus = sorted(res, key=float)
        x = [DOSE[float(t)] for t in taus]
        col, mk = SERIES[i], ("o", "s")[i]
        for t, xx in zip(taus, x):
            ys = [r["shape"]["loudness-null"]["explore"]["sigma_shape"] for r in res[t]]
            axes[0].scatter([xx] * len(ys), ys, color=col, marker=mk, s=24, alpha=0.8)
        axes[0].plot(x, [np.median([r["shape"]["loudness-null"]["explore"]["sigma_shape"] for r in res[t]]) for t in taus],
                     color=col, marker=mk, label=lab)
        for rule, ls in (("loudness-null", "-"), ("PC2", (0, (3, 2)))):
            axes[1].plot(x, [sum(r["both"][rule] for r in res[t]) for t in taus], color=col, marker=mk, linestyle=ls,
                         label=f"{lab}, shape rule {rule}")
        for t, xx in zip(taus, x):
            ys = [r["rendering"]["omega"] for r in res[t]]
            axes[2].scatter([xx] * len(ys), ys, color=col, marker=mk, s=24, alpha=0.8)
        axes[2].plot(x, [np.median([r["rendering"]["omega"] for r in res[t]]) for t in taus], color=col, marker=mk, label=lab)
    axes[0].axhline(0.7, color=INK2, linestyle=(0, (3, 2)), linewidth=1)
    axes[0].set_ylabel("σ_shape of the loudness-null direction (explore)")
    axes[0].set_title("shape selectivity of the shape knob", fontsize=10)
    axes[1].set_ylabel("seeds (of 3) with both dials, all criteria")
    axes[1].set_ylim(-0.2, 3.2)
    axes[1].set_title("both dials valid in the same seed", fontsize=10)
    axes[2].axhline(0.0105, color=INK2, linestyle=(0, (1, 2)), linewidth=1)
    axes[2].text(0.95, 0.0115, "VICReg ω autoencoder", fontsize=7, color=INK2)
    axes[2].set_yscale("log")
    axes[2].set_ylabel("ω val normalized MSE")
    axes[2].set_title("rendering cost", fontsize=10)
    for ax in axes:
        ax.set_xlabel("content shape R² (calibrated dose; less shape in content → right)")
        ax.invert_xaxis()
        ax.legend(fontsize=7)
    fig.suptitle("[BUILD] Build summary: the shape-noise dose with both decoders (exploratory)", fontweight="bold", color=INK)
    fig.savefig(REPO / "results" / "build" / "build_summary.png", dpi=115)
    plt.close(fig)


if __name__ == "__main__":
    main()
