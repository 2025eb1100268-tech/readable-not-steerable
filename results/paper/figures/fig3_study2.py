"""Figure 3 (Study 2, C): enstrophy prominence E1 per group on the sealed confirmation set, with the H1 threshold
and the H2/H4 drops marked.

Source: results/explore/stage3/confirm.json @ a88ddee only (per-seed E1: results.<group>[i].slow_confirm;
CIs: summary.E1_ci; hypothesis outcomes: summary.H1/H2/H4_secondary). No data access.

    python results/paper/figures/fig3_study2.py
"""
import json

import numpy as np

from _common import AQUA, BLUE, FULL_W, INK, INK2, MUTED, ORANGE, REPO, plt, save, style

FRESH = [("untrained (seeds 3-5)", "untrained", MUTED), ("AE velocity (fresh)", "velocity AE", BLUE),
         ("AE velocity + VICReg (fresh)", "+ VICReg", ORANGE), ("AE vorticity (fresh)", "vorticity AE", BLUE),
         ("AE vorticity + VICReg (fresh)", "+ VICReg", ORANGE)]
STAGE2 = [("untrained", "untrained", MUTED), ("AE velocity", "velocity AE", BLUE),
          ("AE velocity + VICReg", "+ VICReg", ORANGE), ("AE vorticity", "vorticity AE", BLUE),
          ("AE vorticity + VICReg", "+ VICReg", ORANGE), ("AE streamfunction", "streamfunction AE", BLUE),
          ("AE streamfunction + VICReg", "+ VICReg", ORANGE), ("imaginer k=4", "imaginer k = 4", BLUE),
          ("imaginer k=4 + VICReg", "+ VICReg", ORANGE), ("summarizer k=4", "summarizer k = 4", AQUA)]


def bracket(ax, x0, x1, y, text, ok):
    ax.plot([x0, x0, x1, x1], [y - 0.02, y, y, y - 0.02], color=INK2, lw=0.8)
    ax.text((x0 + x1) / 2, y + 0.012, f"{text} {'✓' if ok else '✗'}", ha="center", va="bottom", fontsize=6.4,
            color=INK if ok else INK)


def main():
    c = json.load(open(REPO / "results" / "explore" / "stage3" / "confirm.json", encoding="utf-8"))
    res, S = c["results"], c["summary"]
    style()
    fig, ax = plt.subplots(figsize=(FULL_W, 3.2), constrained_layout=True)
    xs, labels = [], []
    pos = {}
    x = 0.0
    for block in (FRESH, STAGE2):
        for g, lab, col in block:
            v = np.array([r["slow_confirm"]["max_abs_r"]["log_Z"] for r in res[g]])
            lo, hi = S["E1_ci"][g]
            ax.scatter(x + np.array([-0.12, 0, 0.12]), v, s=9, color=col, alpha=0.55, linewidths=0, zorder=3)
            ax.plot([x - 0.22, x + 0.22], [S["E1"][g]] * 2, color=col, lw=2, zorder=4)
            ax.plot([x + 0.3] * 2, [lo, hi], color=col, lw=1.0, zorder=4)
            pos[g] = x
            xs.append(x)
            labels.append(lab)
            x += 1
        x += 0.8
    # H1 threshold over the fresh block
    x0, x1 = pos["untrained (seeds 3-5)"] - 0.4, pos["AE vorticity + VICReg (fresh)"] + 0.4
    ax.plot([x0, x1], [0.70, 0.70], color=INK, lw=0.9, ls=(0, (4, 2)))
    ax.text(x0, 0.705, "H1 threshold 0.70", fontsize=6.4, color=INK, va="bottom")
    h2 = S["H2"]["per_pair"]
    for plain, vic, key in (("AE velocity (fresh)", "AE velocity + VICReg (fresh)", "AE velocity (fresh) − AE velocity + VICReg (fresh)"),
                            ("AE vorticity (fresh)", "AE vorticity + VICReg (fresh)", "AE vorticity (fresh) − AE vorticity + VICReg (fresh)")):
        e = h2[key]
        bracket(ax, pos[plain], pos[vic], 0.955, f"H2 +{e['gap']:.3f}", e["pass"])
    h4 = S["H4_secondary"]
    h4e = next(iter(h4["per_pair"].values())) if "per_pair" in h4 else h4
    bracket(ax, pos["imaginer k=4"], pos["imaginer k=4 + VICReg"], 0.955, f"H4 +{h4e['gap']:.3f}", bool(h4e.get("pass", h4.get("holds"))))
    ax.set_xticks(xs, labels, fontsize=6.4, rotation=35, ha="right", rotation_mode="anchor")
    ax.set_ylim(0.2, 1.02)
    ax.set_ylabel("E1: enstrophy |r|, sealed set")
    ax.grid(axis="x", visible=False)
    from matplotlib.transforms import blended_transform_factory
    tr = blended_transform_factory(ax.transData, ax.transAxes)
    ax.text((pos["untrained (seeds 3-5)"] + pos["AE vorticity + VICReg (fresh)"]) / 2, 1.02,
            "fresh seeds 3–5 (H1, H2: primary)", ha="center", va="bottom", fontsize=7, color=INK, transform=tr)
    ax.text((pos["untrained"] + pos["summarizer k=4"]) / 2, 1.02,
            "exploration seeds 0–2 (reported; H4 secondary)", ha="center", va="bottom", fontsize=7, color=INK, transform=tr)
    sep = (pos["AE vorticity + VICReg (fresh)"] + pos["untrained"]) / 2
    ax.axvline(sep, color=MUTED, lw=0.6)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=BLUE, lw=2, label="plain"), Line2D([], [], color=ORANGE, lw=2, label="+ VICReg"),
               Line2D([], [], color=AQUA, lw=2, label="summarizer (JEPA)"), Line2D([], [], color=MUTED, lw=2, label="untrained"),
               Line2D([], [], color=INK2, lw=1, label="95% CI (trajectory bootstrap)")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.36), ncol=5, fontsize=6.6, handlelength=1.4)
    save(fig, "fig3_study2")


if __name__ == "__main__":
    main()
