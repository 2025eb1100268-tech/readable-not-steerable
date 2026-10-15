"""Summary figure for REPORT.md: every scored quantity on test-ID (raw readout), per model and
seed, with the untrained control's mean and the level the pre-registered rule needs.

    python -m probes.report_figure --results results/final/final_evaluation.json --out results/final/summary.png
"""
from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common.run import REPO  # noqa: E402
from solver.validate import INK, INK2, SERIES, _style  # noqa: E402

PANELS = [  # (title, getter, rule for the level the main model needs given the control mean c)
    ("S1 · log enstrophy", lambda r: r["probes"]["log_Z"]["r2_partial"], lambda c: min(c + 0.2, 1 - 0.5 * (1 - c))),
    ("S1 · log palinstrophy", lambda r: r["probes"]["log_P"]["r2_partial"], lambda c: min(c + 0.2, 1 - 0.5 * (1 - c))),
    ("S1 · log(1 + vortex count)", lambda r: r["probes"]["log1p_vortex_count"]["r2_partial"], lambda c: min(c + 0.2, 1 - 0.5 * (1 - c))),
    ("S1 · Okubo–Weiss fraction (mandatory)", lambda r: r["probes"]["ow_vort_frac"]["r2_partial"], lambda c: min(c + 0.2, 1 - 0.5 * (1 - c))),
    ("S1 · log dissipation", lambda r: r["probes"]["log_eps"]["r2_partial"], lambda c: min(c + 0.2, 1 - 0.5 * (1 - c))),
    ("S2 · log ν, frame-pair R²", lambda r: r["nu"]["pair_r2"], lambda c: max(0.6, c + 0.3)),
    ("S3 · enstrophy, slow-direction |r|", lambda r: r["slow"]["max_abs_r"]["log_Z"], lambda c: max(0.8, c + 0.1)),
]
LABELS = {"untrained": "untrained\n(control)", "jepa_spatial": "JEPA\n(main)", "pixel": "pixel\npredictor",
          "jepa_pooled": "pooled\nJEPA", "static_vicreg": "static\nVICReg", "nonphys_jepa_spatial": "non-phys.\nJEPA"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    with open(REPO / args.results, encoding="utf-8") as f:
        res = json.load(f)["results"]["raw"]
    groups = list(res)
    _style()
    fig, axes = plt.subplots(2, 4, figsize=(17, 8), constrained_layout=True)
    for ax, (title, get, need) in zip(axes.flat, PANELS):
        c = float(np.mean([get(r) for r in res["untrained"]]))
        ax.axhline(c, color=INK2, linewidth=1, linestyle=(0, (2, 2)), label="control mean")
        ax.axhline(need(c), color=SERIES[1], linewidth=1.4, label="needed by JEPA")
        for i, g in enumerate(groups):
            v = np.array([get(r) for r in res[g]])
            col = SERIES[0] if g == "jepa_spatial" else INK2
            ax.scatter(np.full(len(v), i) + np.array([-0.12, 0, 0.12])[: len(v)], v, s=22, color=col, zorder=3, linewidths=0)
            ax.plot([i - 0.25, i + 0.25], [v.mean()] * 2, color=INK if g != "jepa_spatial" else SERIES[0], linewidth=2.2, zorder=4)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([LABELS.get(g, g) for g in groups], fontsize=7.5)
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
    axes.flat[0].legend(loc="lower right", fontsize=7.5)
    last = axes.flat[-1]
    last.axis("off")
    last.text(0.0, 0.95, "Test-ID, raw readout, 3 seeds per model\n(dots = seeds, bar = mean).\n\n"
              "Blue = main JEPA. Orange line = level the\npre-registered rule needs (from the control\n"
              "mean); a target also needs its paired\nbootstrap CI above zero.\n\n"
              "Verdict: not supported (S1 fails on the\nmandatory Okubo–Weiss target; S2, S3 fail).",
              va="top", fontsize=9, color=INK)
    fig.savefig(REPO / args.out, dpi=130)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
