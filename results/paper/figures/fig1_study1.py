"""Figure 1 (Study 1, C): each model's gain over the untrained control on every pre-registered S1-S3 quantity,
per seed, with the stored 95% paired trajectory-bootstrap CIs and the gain each rule needed.

Source: results/final/final_evaluation.json @ e8d619e (per-seed values: results.raw; CIs: rules.raw). No
computation beyond differences of stored values.

    python results/paper/figures/fig1_study1.py
"""
import json

import numpy as np

from _common import BLUE, INK, INK2, MUTED, REPO, FULL_W, plt, save, style

MODELS = [("jepa_spatial", "JEPA"), ("pixel", "pixel"), ("jepa_pooled", "pooled"),
          ("static_vicreg", "static"), ("nonphys_jepa_spatial", "non-phys.")]
S1 = [("log_Z", "S1 · log enstrophy"), ("log_P", "S1 · log palinstrophy"),
      ("log1p_vortex_count", "S1 · log(1 + vortices)"), ("ow_vort_frac", "S1 · Okubo–Weiss frac.*"),
      ("log_eps", "S1 · log dissipation")]


def main():
    d = json.load(open(REPO / "results" / "final" / "final_evaluation.json", encoding="utf-8"))
    raw, rules = d["results"]["raw"], d["rules"]["raw"]
    val = {
        "S1": lambda run, t: run["probes"][t]["r2_partial"],
        "S2": lambda run, t: run["nu"]["pair_r2"],
        "S3": lambda run, t: run["slow"]["max_abs_r"]["log_Z"],
    }
    panels = [("S1", t, title) for t, title in S1] + [("S2", None, "S2 · log ν, frame pair"),
                                                       ("S3", None, "S3 · enstrophy |r|")]
    style()
    fig, axes = plt.subplots(2, 4, figsize=(FULL_W, 4.4), constrained_layout=True)
    for ax, (crit, t, title) in zip(axes.flat, panels):
        ctrl = np.array([val[crit](r, t) for r in raw["untrained"]])
        c = ctrl.mean()
        if crit == "S1":
            need = min(c + 0.20, 1 - 0.5 * (1 - c)) - c  # +0.20 or halving the unexplained variance
        elif crit == "S2":
            need = max(0.6, c + 0.3) - c
        else:
            need = max(0.8, c + 0.1) - c
        for i, (m, _) in enumerate(MODELS):
            seeds = np.array([val[crit](r, t) for r in raw[m]])
            gains = seeds - ctrl  # control seed s = model seed s's initialization
            col = BLUE if m == "jepa_spatial" else INK2
            ax.scatter(np.full(3, i) + np.array([-0.12, 0, 0.12]), gains, s=9, color=col, alpha=0.55, linewidths=0,
                       zorder=3)
            ax.plot([i - 0.22, i + 0.22], [gains.mean()] * 2, color=col, lw=1.8, solid_capstyle="round", zorder=4)
            if crit == "S1":
                ci = rules[m]["S1"]["targets"][t]["ci95"]
            elif crit == "S3":
                ci = rules[m]["S3"]["invariants"]["log_Z"]["ci95"]
            else:
                ci = None
            if ci is not None:
                ax.plot([i + 0.3, i + 0.3], ci, color=col, lw=1.0, zorder=4)
                ax.plot([i + 0.26, i + 0.34], [ci[0]] * 2, color=col, lw=1.0)
                ax.plot([i + 0.26, i + 0.34], [ci[1]] * 2, color=col, lw=1.0)
        ax.axhline(0, color=MUTED, lw=0.8, zorder=2)
        ax.axhline(need, color=INK, lw=0.9, ls=(0, (4, 2)), zorder=2)
        ax.set_title(f"{title}\ncontrol {c:.3f}", fontsize=7.5, loc="left")
        ax.set_xticks(range(len(MODELS)), [lab for _, lab in MODELS], fontsize=6.3, rotation=40, ha="right",
                      rotation_mode="anchor")
        ax.set_xlim(-0.5, len(MODELS) - 0.4)
        ax.grid(axis="x", visible=False)
    fig.supylabel("gain over the untrained control (model − control)", fontsize=8, color=INK)
    key = axes.flat[-1]
    key.axis("off")
    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], marker="o", ls="none", color=INK2, alpha=0.55, markersize=3.5, label="seed (paired)"),
        Line2D([], [], color=INK2, lw=1.8, label="seed mean"),
        Line2D([], [], color=INK2, lw=1.0, marker="_", markersize=5, label="95% CI of the gain"),
        Line2D([], [], color=INK, lw=0.9, ls=(0, (4, 2)), label="gain the rule needed"),
        Line2D([], [], color=MUTED, lw=0.8, label="control level"),
        Line2D([], [], color=BLUE, lw=1.8, label="main JEPA"),
    ]
    key.legend(handles=handles, loc="upper left", fontsize=6.8, handlelength=2.2, borderaxespad=0)
    key.text(0.0, 0.18, "* must be among the three S1\n  targets that count\nS2: no stored CI",
             fontsize=6.5, color=INK2, va="top", transform=key.transAxes)
    save(fig, "fig1_study1")


if __name__ == "__main__":
    main()
