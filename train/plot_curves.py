"""Training curves and collapse diagnostics for Phase 2 runs.

    python -m train.plot_curves --runs <run> [<run> ...] --out results/phase2_training.png
"""
from __future__ import annotations

import argparse
import json
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common.run import REPO  # noqa: E402
from solver.validate import INK2, SERIES, _style  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    runs = {r: json.load(open(REPO / "results" / "phase2" / f"{r}.json", encoding="utf-8")) for r in args.runs}
    # one row per run family (run name without _lr<lr>_s<seed>), so e.g. nonphys_jepa_spatial is not merged into jepa_spatial
    family = {r: re.sub(r"_lr[^_]+_s\d+$", "", r) for r in runs}
    kinds = sorted(set(family.values()))
    _style()
    fig, ax = plt.subplots(len(kinds), 3, figsize=(13.5, 3.6 * len(kinds)), squeeze=False)
    for row, kind in enumerate(kinds):
        sel = {r: d for r, d in runs.items() if family[r] == kind}
        for i, (r, d) in enumerate(sel.items()):
            col = SERIES[i % len(SERIES)]
            lab = r.replace(f"{kind}_", "")
            steps = [c["step"] for c in d["curve"]]
            ax[row, 0].plot(steps, [c["pred"] for c in d["curve"]], color=col, linewidth=1.2, label=lab)
            ep = [e["epoch"] for e in d["evals"]]
            ax[row, 1].plot(ep, [e["metric"] for e in d["evals"]], color=col, marker="o", markersize=4, label=lab)
            ax[row, 2].plot(ep, [e["effective_rank_corr"] for e in d["evals"]], color=col, marker="o", markersize=4, label=lab)
        static = next(iter(sel.values()))["config"]["model"] == "static_vicreg"
        ax[row, 0].set_yscale("log")
        ax[row, 0].set_title(f"{kind}: training {'invariance' if static else 'prediction'} loss")
        ax[row, 0].set_xlabel("step")
        ax[row, 1].set_yscale("log")
        if not static:  # copy-baseline ratio: 1 = no better than copying frame t
            ax[row, 1].axhline(1.0, color=INK2, linewidth=0.8, linestyle=(0, (2, 2)))
        name = next(iter(sel.values()))["evals"][0]["metric_name"]
        ax[row, 1].set_title(f"{kind}: val {name}")
        ax[row, 1].set_xlabel("epoch")
        ax[row, 2].set_title(f"{kind}: latent eff. rank (corr.)")
        ax[row, 2].set_xlabel("epoch")
        ax[row, 2].set_ylim(0, 130)
        for a in ax[row]:
            a.legend(fontsize=7.5, loc="best")
    fig.tight_layout()
    fig.savefig(REPO / args.out, dpi=130)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
