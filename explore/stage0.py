"""[EXPLORE] Stage 0: spatial-map readouts of the six existing models, scored on the explore set.

    python -m explore.stage0 --config configs/explore/stage0.yaml

Exploratory (results/explore/BRIEF.md); nothing here is a finding. For every model, seed and
readout (pooled; map mean ⊕ std; map PCA-512): the S1 probes (time × family partialled), the ν
frame-pair probe and the slow-direction analysis, all with the unchanged probe code, fitted on v1
train/val as pre-registered and scored on explore. Next to every slow-direction result:
per-direction |r|, specificity against non-conserved statistics, and the partial r with log Z₀
given log E₀ and family. Comparisons use probes/criteria.py (paired trajectory bootstrap).
"""
from __future__ import annotations

import argparse
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common.run import REPO, append_log, command_line, environment, load_config, save_json, seed_everything  # noqa: E402
from data.combined import CombinedDataset  # noqa: E402
from data.dataset import S1_TARGETS  # noqa: E402
from explore.readouts import READOUTS, gpu_inputs, readouts_for  # noqa: E402
from probes import criteria  # noqa: E402
from probes.evaluate import PROBE_TARGETS, probe_nu, probe_targets, slow_directions  # noqa: E402
from solver.validate import INK, INK2, SERIES, _style  # noqa: E402

EVAL = "explore"
SPECIFICITY = {"log P (traj mean, not conserved)": "P", "log kurtosis (traj mean)": "kurtosis", "log max|ω| (traj mean)": "max_abs_w"}


def slow_extra(ds, slow: dict, eval_split: str = EVAL) -> dict:
    """Per-direction specificity and partial r for the slow directions scored on the eval set."""
    ev = ds.rows([eval_split], viscous=False)
    ids, inv = np.unique(ds.traj[ev], return_inverse=True)
    assert ids.tolist() == slow["traj"]
    counts = np.bincount(inv)
    proj = np.array(slow["proj_traj_means"])
    first = np.array([np.flatnonzero((ds.traj == t) & (ds.frame == 0))[0] for t in ids])
    abs_r = lambda y: [float(abs(np.corrcoef(proj[:, j], y)[0, 1])) for j in range(proj.shape[1])]  # noqa: E731
    spec = {name: abs_r(np.bincount(inv, weights=np.log(ds.f[key][ev])) / counts) for name, key in SPECIFICITY.items()}
    design = np.column_stack([np.ones(len(ids)), np.log(ds.f["E"][first]), ds.blob[first].astype(float)])
    resid = lambda y: y - design @ np.linalg.lstsq(design, y, rcond=None)[0]  # noqa: E731
    z_res = resid(np.log(ds.f["Z"][first]))
    partial = [float(abs(np.corrcoef(resid(proj[:, j]), z_res)[0, 1])) for j in range(proj.shape[1])]
    return {"specificity_abs_r": spec, "partial_r_logZ0_given_logE0_family": partial,
            "corr_logE0_logZ0": float(np.corrcoef(np.log(ds.f["E"][first]), np.log(ds.f["Z"][first]))[0, 1])}


def evaluate_run(ds, lat: np.ndarray, n_pca: int) -> dict:
    slow = slow_directions(ds, lat, ["train", "val"], EVAL, n_pca)
    return {"probes": probe_targets(ds, lat, EVAL), "nu": probe_nu(ds, lat, EVAL), "slow": slow, "slow_extra": slow_extra(ds, slow)}


def quantities(r: dict) -> dict:
    q = {f"{t} (R², time × family)": r["probes"][t]["r2_partial"] for t in PROBE_TARGETS}
    q.update({"log ν pair R²": r["nu"]["pair_r2"], "log ν single R²": r["nu"]["single_r2"]})
    for k in ("log_E", "log_Z", "log_C4"):
        q[f"slow max |r| {k}"] = r["slow"]["max_abs_r"][k]
    q["slow max partial r log Z₀ | log E₀, family"] = max(r["slow_extra"]["partial_r_logZ0_given_logE0_family"])
    for name, v in r["slow_extra"]["specificity_abs_r"].items():
        q[f"slow max |r| {name}"] = max(v)
    return q


def ms(v):
    return f"{np.mean(v):.3f} ± {np.std(v, ddof=1):.3f}"


def write_md(cfg, res, rules, readout_vs_pooled, out):
    groups = list(cfg["models"])
    L = ["# [EXPLORE] Stage 0: spatial-map readouts of the existing models, scored on explore", "",
         "Exploratory, not a finding (results/explore/BRIEF.md). Probes and slow directions fitted on v1 train/val "
         "(as pre-registered), scored on the explore set (1,006 trajectories, 378 inviscid). Mean ± std over 3 seeds.", ""]
    for ro in READOUTS:
        rows = {g: [quantities(r) for r in res[ro][g]] for g in groups}
        L += [f"## Readout: {ro}", "", "| quantity | " + " | ".join(groups) + " |", "|---|" + "---|" * len(groups)]
        for k in rows[groups[0]][0]:
            L.append(f"| {k.replace('|', chr(92) + '|')} | " + " | ".join(ms([q[k] for q in rows[g]]) for g in groups) + " |")
        L += ["", f"Each model vs the untrained control, {ro} readout (reported, not scored): S1 gain [95% CI]; ν pair; enstrophy |r| gap [CI].", "",
              "| model | " + " | ".join(S1_TARGETS) + " | ν pair (model vs control) | enstrophy \\|r\\| gap [CI] |", "|---|" + "---|" * (len(S1_TARGETS) + 2)]
        for g, v in rules[ro].items():
            z = v["S3"]["invariants"]["log_Z"]
            L.append(f"| {g} | " + " | ".join(f"{x['gain']:+.3f} [{x['ci95'][0]:+.3f}, {x['ci95'][1]:+.3f}]" for x in v["S1"]["targets"].values())
                     + f" | {v['S2']['r2_jepa']:.3f} vs {v['S2']['r2_control']:.3f} | {z['gap']:+.3f} [{z['ci95'][0]:+.2f}, {z['ci95'][1]:+.2f}] |")
        L.append("")
    L += ["## Map readouts minus the pooled readout, same model (paired; gain [95% CI])", "",
          "| model | readout | " + " | ".join(S1_TARGETS) + " | ν pair diff | enstrophy \\|r\\| diff [CI] |", "|---|---|" + "---|" * (len(S1_TARGETS) + 2)]
    for g, per in readout_vs_pooled.items():
        for ro, v in per.items():
            z = v["S3"]["invariants"]["log_Z"]
            L.append(f"| {g} | {ro} | " + " | ".join(f"{x['gain']:+.3f} [{x['ci95'][0]:+.3f}, {x['ci95'][1]:+.3f}]" for x in v["S1"]["targets"].values())
                     + f" | {v['S2']['r2_jepa'] - v['S2']['r2_control']:+.3f} | {z['gap']:+.3f} [{z['ci95'][0]:+.2f}, {z['ci95'][1]:+.2f}] |")
    L += ["", "## Every slow direction, every seed (|r| on explore inviscid runs; directions 1–3, slowest first)", "",
          "| model | readout | seed | log Z₀ | log E₀ | partial r log Z₀ \\| log E₀, family | traj-mean log P (not conserved) | held-out λ |",
          "|---|---|---|---|---|---|---|---|"]
    f3 = lambda v: " / ".join(f"{x:.2f}" for x in v)  # noqa: E731
    for ro in READOUTS:
        for g in groups:
            for r in res[ro][g]:
                s, e = r["slow"], r["slow_extra"]
                L.append(f"| {g} | {ro} | {r['run'].rsplit('_s', 1)[1]} | {f3(s['abs_r']['log_Z'])} | {f3(s['abs_r']['log_E'])} | "
                         f"{f3(e['partial_r_logZ0_given_logE0_family'])} | {f3(e['specificity_abs_r']['log P (traj mean, not conserved)'])} | "
                         f"{f3(s['lambda_heldout'][:3])} |")
    corr = res[READOUTS[0]][groups[0]][0]["slow_extra"]["corr_logE0_logZ0"]
    L += ["", f"corr(log E₀, log Z₀) over explore inviscid trajectories: {corr:.3f}.", ""]
    with open(REPO / out, "w", encoding="utf-8") as f:
        f.write("\n".join(L))


def plot(cfg, res, out):
    groups = list(cfg["models"])
    panels = [(f"S1 · {t}", lambda r, t=t: r["probes"][t]["r2_partial"]) for t in S1_TARGETS]
    panels += [("log ν, frame-pair R²", lambda r: r["nu"]["pair_r2"]), ("enstrophy, best-of-top-3 slow |r|", lambda r: r["slow"]["max_abs_r"]["log_Z"])]
    _style()
    fig, axes = plt.subplots(2, 4, figsize=(18, 8.4), constrained_layout=True)
    offs = {ro: o for ro, o in zip(READOUTS, (-0.25, 0.0, 0.25))}
    for ax, (title, get) in zip(axes.flat, panels):
        for ro, col in zip(READOUTS, SERIES):
            for i, g in enumerate(groups):
                v = np.array([get(r) for r in res[ro][g]])
                ax.scatter(np.full(len(v), i + offs[ro]), v, s=14, color=col, linewidths=0, zorder=3)
                ax.plot([i + offs[ro] - 0.1, i + offs[ro] + 0.1], [v.mean()] * 2, color=col, linewidth=2.2, zorder=4,
                        label=ro if i == 0 else None)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([g.replace("_", "\n", 1) for g in groups], fontsize=7)
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
    axes.flat[0].legend(fontsize=7.5, loc="lower right")
    last = axes.flat[-1]
    last.axis("off")
    last.text(0, 0.95, "[EXPLORE] exploratory, not a finding.\n\nScored on the explore set; probes and slow\n"
              "directions fitted on v1 train/val.\nDots = seeds, bars = means.\n\n"
              "pooled: spatial mean (pre-registered latent)\nmap_meanstd: per-channel spatial mean ⊕ std\n"
              "map_pca512: PCA-512 of the flattened map", va="top", fontsize=9, color=INK)
    fig.suptitle("[EXPLORE] Stage 0: does reading the spatial map change the picture?", fontweight="bold", color=INK2)
    fig.savefig(REPO / out, dpi=125)
    plt.close(fig)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = load_config(ap.parse_args().config)
    seed_everything(cfg["seed"])
    t0 = time.perf_counter()
    ds = CombinedDataset(cfg["data_parts"], cfg["cache_tag"])
    x = gpu_inputs(ds)
    fit_rows = ds.rows(["train"])
    res = {ro: {} for ro in READOUTS}
    for g, runs in cfg["models"].items():
        for ro in READOUTS:
            res[ro][g] = []
        for run in runs:
            lat = readouts_for(run, ds, x, fit_rows, cfg["map_pca_dims"])
            for ro in READOUTS:
                res[ro][g].append({"run": run, **evaluate_run(ds, lat[ro], cfg["n_pca"])})
            print(f"{run} done ({time.perf_counter() - t0:.0f} s)", flush=True)
    del x
    ctrl = cfg["control_group"]
    rules = {ro: {g: {"S1": criteria.s1(res[ro][g], res[ro][ctrl]), "S2": criteria.s2(res[ro][g], res[ro][ctrl]),
                      "S3": criteria.s3(res[ro][g], res[ro][ctrl])} for g in cfg["models"] if g != ctrl} for ro in READOUTS}
    vs_pooled = {g: {ro: {"S1": criteria.s1(res[ro][g], res["pooled"][g]), "S2": criteria.s2(res[ro][g], res["pooled"][g]),
                          "S3": criteria.s3(res[ro][g], res["pooled"][g])} for ro in READOUTS[1:]} for g in cfg["models"]}
    wall = time.perf_counter() - t0
    save_json({"environment": environment(), "config": cfg, "results": res, "vs_control": rules, "readout_vs_pooled": vs_pooled,
               "wall_seconds": wall}, cfg["output_json"])
    write_md(cfg, res, rules, vs_pooled, cfg["output_md"])
    plot(cfg, res, cfg["figure"])
    jz = {ro: np.mean([r["slow"]["max_abs_r"]["log_Z"] for r in res[ro]["jepa_spatial"]]) for ro in READOUTS}
    append_log("[EXPLORE] Stage 0: spatial-map readouts of the existing models (explore set)", cfg["_config_path"], command_line(), "ok",
               {"JEPA enstrophy |r| by readout": ", ".join(f"{ro} {v:.3f}" for ro, v in jz.items()),
                "results": f"{cfg['output_json']}, {cfg['output_md']}, {cfg['figure']}"}, wall, environment().get("gpu"))
    print(open(REPO / cfg["output_md"], encoding="utf-8").read()[:6000])


if __name__ == "__main__":
    main()
