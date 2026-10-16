"""[EXPLORE] Stage 2 evaluation: imaginer vs summarizer across horizons, plus the bridge, the
fake-physics imaginer and pixel + VICReg (results/explore/PREDICTIONS.md entries 3-4).

    python -m explore.stage2_eval --config configs/explore/stage2_eval.yaml [--decide-kstar]

Exploratory. Probes and slow directions fitted on v1 train/val as pre-registered; scored on explore
(and matched-energy for E5). Statistics (entry 3):
  E1  explore enstrophy |r|: best of top-3 slow directions with log Z0, seed mean
  E2  explore partial r with log Z0 given log E0 and family, best of top 3
  E3  for the enstrophy-tracking direction: partial r with log Z0 given trajectory-mean log P, and
      partial r with trajectory-mean log P given log Z0
  E4  that direction regressed on unstandardized log E0 and log(Z0/E0): ratio a/b and R^2
  E5  matched-energy best-of-top-3 |r| with log Z0
  secondary: S1 partialled R^2, log nu pair R^2 (explore); training diagnostics from the run files
--decide-kstar applies entry 3's rule to the imaginer's E1/E2 (pooled latent), writes
results/explore/stage2/kstar.json once and logs it; it refuses if the file already exists.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common.run import REPO, append_log, command_line, environment, load_config, save_json, seed_everything  # noqa: E402
from data.combined import CombinedDataset  # noqa: E402
from data.dataset import S1_TARGETS  # noqa: E402
from explore.readouts import _paths, gpu_inputs, readouts_for  # noqa: E402
from explore.stage0 import slow_extra  # noqa: E402
from explore.stage2_configs import HORIZONS, SEEDS, effective_lr, name  # noqa: E402
from probes import criteria  # noqa: E402
from probes.evaluate import probe_nu, probe_targets, slow_directions  # noqa: E402
from solver.validate import INK, INK2, SERIES, _style  # noqa: E402

S2 = REPO / "results" / "explore" / "stage2"
CONTROL = "untrained"
EXCLUDED: dict = {}  # run -> reason (divergence rule, entry 5)


def groups_from_files() -> dict:
    """{group label: [run names]} for every Stage 2 run whose checkpoint and results exist, with the
    effective lrs (entry 5). Pixel-family runs that diverged (final val loss / copy >= 1) are excluded
    and recorded in EXCLUDED."""
    lr = effective_lr()
    g = {CONTROL: [f"untrained_s{s}" for s in SEEDS]}
    for fam in ("imaginer", "summarizer"):
        for k in HORIZONS:
            g[f"{fam} k={k}"] = [name(fam, k, lr[fam], s) for s in SEEDS]
    g["bridge k=4"] = [name("bridge", 4, lr["imaginer"], s) for s in SEEDS]
    if (S2 / "kstar.json").exists():
        with open(S2 / "kstar.json", encoding="utf-8") as f:
            ks = json.load(f)["k_star"]
        g[f"fake-physics imaginer k={ks}"] = [name("fake_imaginer", ks, lr["imaginer"], s) for s in SEEDS]
        g[f"pixel + VICReg k={ks}"] = [name("pixel_vicreg", ks, lr["imaginer"], s) for s in SEEDS]
    def done(r):
        if r.startswith("untrained_s"):
            return True
        if not ((REPO / "results" / "checkpoints" / f"{r}.pt").exists() and (S2 / "runs" / f"{r}.json").exists()):
            return False
        with open(S2 / "runs" / f"{r}.json", encoding="utf-8") as f:
            res = json.load(f)
        if res["config"]["model"] in ("pixel", "pixel_vicreg") and res["final"]["metric"] >= 1:
            EXCLUDED[r] = f"diverged (final val loss / copy {res['final']['metric']:.3g} >= 1; entry 5 rule)"
            return False
        return True
    return {k: [r for r in v if done(r)] for k, v in g.items() if any(done(r) for r in v)}


def traj_mean_log_p(ds, split: str) -> np.ndarray:
    ev = ds.rows([split], viscous=False)
    _, inv = np.unique(ds.traj[ev], return_inverse=True)
    return np.bincount(inv, weights=np.log(ds.f["P"][ev])) / np.bincount(inv)


def partial(y, x, z):
    """corr(y, x | z): residualize y and x on [1, z]."""
    d = np.column_stack([np.ones(len(z)), z])
    res = lambda v: v - d @ np.linalg.lstsq(d, v, rcond=None)[0]  # noqa: E731
    return float(np.corrcoef(res(y), res(x))[0, 1])


def direction_stats(slow: dict, log_p: np.ndarray) -> dict:
    """E3 and E4 for the enstrophy-tracking direction (explore)."""
    proj, lz, le = np.array(slow["proj_traj_means"]), np.array(slow["targets"]["log_Z"]), np.array(slow["targets"]["log_E"])
    r = [np.corrcoef(proj[:, j], lz)[0, 1] for j in range(proj.shape[1])]
    j = int(np.argmax(np.abs(r)))
    y = proj[:, j] * np.sign(r[j])
    X = np.column_stack([np.ones(len(y)), le, lz - le])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    r2 = 1 - ((y - X @ coef) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return {"direction": j + 1, "abs_r_logZ0": float(abs(r[j])),
            "E3_partial_logZ0_given_logP": partial(y, lz, log_p), "E3_partial_logP_given_logZ0": partial(y, log_p, lz),
            "E4_a_logE0": float(coef[1]), "E4_b_logZ0_over_E0": float(coef[2]), "E4_ratio_a_over_b": float(coef[1] / coef[2]),
            "E4_r2": float(r2)}


def evaluate_run(ds, lat, n_pca, log_p) -> dict:
    out = {"probes": probe_targets(ds, lat, "explore"), "nu": probe_nu(ds, lat, "explore")}
    with warnings.catch_warnings():  # log E0 is constant on matched-energy: its correlations are NaN by construction
        warnings.simplefilter("ignore", RuntimeWarning)
        for s in ("explore", "matched"):
            slow = slow_directions(ds, lat, ["train", "val"], s, n_pca)
            out[f"slow_{s}"] = slow
            out[f"extra_{s}"] = slow_extra(ds, slow, s)
    out["direction_stats"] = direction_stats(out["slow_explore"], log_p)
    return out


def shim(runs, s):
    return [{"slow": r[f"slow_{s}"]} for r in runs]


def s3(a, b, s):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return criteria.s3(shim(a, s), shim(b, s))["invariants"]["log_Z"]


def boot_ci(runs, s, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    m = len(runs[0][f"slow_{s}"]["traj"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        v = [criteria._max_r(shim(runs, s), "log_Z", rng.integers(0, m, m)) for _ in range(n)]
    return [float(x) for x in np.percentile(v, [2.5, 97.5])]


def summarize(res: dict, ctrl: list) -> dict:
    out = {}
    for g, runs in res.items():
        e1 = [r["slow_explore"]["max_abs_r"]["log_Z"] for r in runs]
        e5 = [r["slow_matched"]["max_abs_r"]["log_Z"] for r in runs]
        e2 = [max(r["extra_explore"]["partial_r_logZ0_given_logE0_family"]) for r in runs]
        ds_ = [r["direction_stats"] for r in runs]
        out[g] = {"n_seeds": len(runs), "E1": e1, "E1_ci": boot_ci(runs, "explore"), "E2": e2, "E5": e5, "E5_ci": boot_ci(runs, "matched"),
                  "E3_partial_logZ0_given_logP": [d["E3_partial_logZ0_given_logP"] for d in ds_],
                  "E3_partial_logP_given_logZ0": [d["E3_partial_logP_given_logZ0"] for d in ds_],
                  "E4_ratio": [d["E4_ratio_a_over_b"] for d in ds_], "E4_r2": [d["E4_r2"] for d in ds_],
                  "S1": {t: [r["probes"][t]["r2_partial"] for r in runs] for t in S1_TARGETS},
                  "nu_pair": [r["nu"]["pair_r2"] for r in runs]}
        if g != CONTROL:
            out[g]["E1_vs_control"] = s3(runs, ctrl, "explore")
            out[g]["E5_vs_control"] = s3(runs, ctrl, "matched")
    return out


def ms(v):
    v = np.asarray(v, float)
    return f"{v.mean():.3f}" + (f" ± {v.std(ddof=1):.3f}" if len(v) > 1 else " (1 seed)")


def training_table(groups) -> list:
    L = ["| group | run | val loss / copy (at k) | steps | stop | collapsed | pred/true energy k ≥ 10 | copy/true energy k ≥ 10 | pred/target map variance |",
         "|---|---|---|---|---|---|---|---|---|"]
    for g, runs in groups.items():
        for run in runs:
            if run.startswith("untrained"):
                continue
            with open(S2 / "runs" / f"{run}.json", encoding="utf-8") as f:
                r = json.load(f)
            d = r.get("diagnostics", {})
            fmt = lambda k: f"{d[k]:.3f}" if k in d else "—"  # noqa: E731
            L.append(f"| {g} | {run} | {r['final']['metric']:.4f} | {r['steps']} | {r['stop_reason']} | {r['collapsed']} | "
                     f"{fmt('pred_over_true_energy_k_ge_10')} | {fmt('copy_over_true_energy_k_ge_10')} | {fmt('pred_over_target_map_variance')} |")
    analysed = {r for runs in groups.values() for r in runs}
    lr = effective_lr()
    L += ["", "Runs that are not analysed (every Stage 2 run is listed):", "",
          "| run | val loss / copy (at k) | steps | stop | reason |", "|---|---|---|---|---|"]
    for p in sorted((S2 / "runs").glob("*.json")):
        run = p.stem
        if run in analysed:
            continue
        with open(p, encoding="utf-8") as f:
            r = json.load(f)
        fam = run.split("_")[1]
        lr_run = r["config"]["lr"]
        if run in EXCLUDED:
            why = EXCLUDED[run]
        elif fam in ("imaginer", "bridge") and lr_run != lr["imaginer"]:
            why = (f"lr {lr_run:g} not used for the imaginer family (entry 5)"
                   + (f"; diverged (final {r['final']['metric']:.3g} >= 1)" if r["final"]["metric"] >= 1 else ""))
        elif fam == "summarizer" and lr_run != lr["summarizer"]:
            why = f"lr-selection run, lr {lr_run:g} not selected"
        else:
            why = "not part of a reported group"
        L.append(f"| {run} | {r['final']['metric']:.4g} | {r['steps']} | {r['stop_reason']} | {why} |")
    return L


def write_md(cfg, groups, summ, res, kstar, out):
    L = ["# [EXPLORE] Stage 2: imaginer vs summarizer across horizons", "",
         "Exploratory, not a finding (results/explore/BRIEF.md). Definitions fixed in results/explore/PREDICTIONS.md entries 3–4. "
         "Statistics say which conserved quantities are prominent in a latent, not that a model discovered conservation. "
         "Probes and slow directions fitted on v1 train/val; scored on explore (378 inviscid runs) and matched-energy (E5).", ""]
    if kstar:
        L += [f"k* = {kstar['k_star']} (decided by the pre-logged rule; see results/explore/stage2/kstar.json).", ""]
    for ro in cfg["readouts"]:
        sm = summ[ro]
        L += [f"## Primary statistics ({ro} latent){'' if ro == cfg['primary_readout'] else ' — secondary, unscored'}", "",
              "| group | seeds | E1 explore \\|r\\| [95% CI] | E1 − control [CI] | E2 partial r | E3 r(Z₀ \\| P̄) | E3 r(P̄ \\| Z₀) | E4 a/b | E4 R² | E5 matched \\|r\\| [CI] |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for g, v in sm.items():
            gap = v.get("E1_vs_control")
            L.append(f"| {g} | {v['n_seeds']} | {ms(v['E1'])} [{v['E1_ci'][0]:.2f}, {v['E1_ci'][1]:.2f}] | "
                     + (f"{gap['gap']:+.3f} [{gap['ci95'][0]:+.2f}, {gap['ci95'][1]:+.2f}]" if gap else "—")
                     + f" | {ms(v['E2'])} | {ms(v['E3_partial_logZ0_given_logP'])} | {ms(v['E3_partial_logP_given_logZ0'])} | "
                     f"{ms(v['E4_ratio'])} | {ms(v['E4_r2'])} | {ms(v['E5'])} [{v['E5_ci'][0]:.2f}, {v['E5_ci'][1]:.2f}] |")
        L += ["", "Imaginer − summarizer at each horizon (paired bootstrap):", "", "| k | E1 gap [95% CI] | E5 gap [95% CI] |", "|---|---|---|"]
        for k in HORIZONS:
            a, b = f"imaginer k={k}", f"summarizer k={k}"
            if a in res[ro] and b in res[ro] and len(res[ro][a]) and len(res[ro][b]):
                e1, e5 = s3(res[ro][a], res[ro][b], "explore"), s3(res[ro][a], res[ro][b], "matched")
                L.append(f"| {k} | {e1['gap']:+.3f} [{e1['ci95'][0]:+.2f}, {e1['ci95'][1]:+.2f}] | {e5['gap']:+.3f} [{e5['ci95'][0]:+.2f}, {e5['ci95'][1]:+.2f}] |")
        L += ["", f"Secondary ({ro}): S1 partialled R² and log ν pair R² on explore.", "",
              "| group | " + " | ".join(S1_TARGETS) + " | log ν pair |", "|---|" + "---|" * (len(S1_TARGETS) + 1)]
        for g, v in sm.items():
            L.append(f"| {g} | " + " | ".join(ms(v["S1"][t]) for t in S1_TARGETS) + f" | {ms(v['nu_pair'])} |")
        L.append("")
    L += ["## Training and prediction diagnostics", ""] + training_table(groups)
    f3 = lambda v: " / ".join(f"{x:.2f}" for x in v)  # noqa: E731
    ro = cfg["primary_readout"]
    L += ["", f"## Every slow direction, every seed ({ro} latent; directions 1–3, slowest first)", "",
          "| group | run | explore \\|r\\| log Z₀ | explore \\|r\\| log E₀ | explore partial r | explore traj-mean log P | matched \\|r\\| log Z₀ | held-out λ (explore) | tracking dir | E3 r(Z₀\\|P̄) / r(P̄\\|Z₀) | E4 a/b, R² |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for g, runs in res[ro].items():
        for r in runs:
            se, sm_, ex, dst = r["slow_explore"], r["slow_matched"], r["extra_explore"], r["direction_stats"]
            L.append(f"| {g} | {r['run']} | {f3(se['abs_r']['log_Z'])} | {f3(se['abs_r']['log_E'])} | {f3(ex['partial_r_logZ0_given_logE0_family'])} | "
                     f"{f3(ex['specificity_abs_r']['log P (traj mean, not conserved)'])} | {f3(sm_['abs_r']['log_Z'])} | {f3(se['lambda_heldout'][:3])} | "
                     f"{dst['direction']} | {dst['E3_partial_logZ0_given_logP']:+.2f} / {dst['E3_partial_logP_given_logZ0']:+.2f} | "
                     f"{dst['E4_ratio_a_over_b']:.2f}, {dst['E4_r2']:.2f} |")
    with open(REPO / out, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


def plot_primary(summ, ro, out):
    _style()
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
    panels = [("E1: explore enstrophy |r| (best of top 3)", "E1", "E1_ci"), ("E2: explore partial r, log Z₀ | log E₀, family", "E2", None),
              ("E5: matched-energy enstrophy |r|", "E5", "E5_ci")]
    sm = summ[ro]
    for ax, (title, key, ci) in zip(axes, panels):
        c = np.mean(sm[CONTROL][key])
        ax.axhline(c, color=INK2, linewidth=1.2, linestyle=(0, (3, 2)), label="untrained control")
        if ci:
            ax.axhspan(*sm[CONTROL][ci], color=INK2, alpha=0.08, linewidth=0)
        for fam, col in (("imaginer", SERIES[0]), ("summarizer", SERIES[1])):
            ks = [k for k in HORIZONS if f"{fam} k={k}" in sm]
            if not ks:
                continue
            mean = [np.mean(sm[f"{fam} k={k}"][key]) for k in ks]
            sd = [np.std(sm[f"{fam} k={k}"][key], ddof=1) if len(sm[f"{fam} k={k}"][key]) > 1 else 0 for k in ks]
            ax.errorbar(ks, mean, yerr=sd, color=col, marker="o", markersize=7, linewidth=2, capsize=3, label=f"{fam} (mean ± seed std)")
            if ci:
                ax.fill_between(ks, [sm[f"{fam} k={k}"][ci][0] for k in ks], [sm[f"{fam} k={k}"][ci][1] for k in ks], color=col, alpha=0.12, linewidth=0)
        for g, col, mk in (("bridge k=4", SERIES[2], "D"),):
            if g in sm:
                ax.scatter([4], [np.mean(sm[g][key])], color=col, marker=mk, s=60, zorder=5, label="bridge (original anchors)")
        for g in sm:
            if g.startswith(("fake-physics", "pixel + VICReg")):
                k = int(g.rsplit("=", 1)[1])
                col, mk = (SERIES[3], "s") if g.startswith("fake") else (SERIES[4], "^")
                ax.scatter([k], [np.mean(sm[g][key])], color=col, marker=mk, s=70, zorder=5, label=g.rsplit(" k=", 1)[0])
        ax.set_xscale("log", base=2)
        ax.set_xticks(HORIZONS)
        ax.set_xticklabels([str(k) for k in HORIZONS])
        ax.set_xlabel("prediction horizon k (frames of Δt = 0.04)")
        ax.set_ylim(0, 1)
        ax.set_title(title)
    axes[0].legend(fontsize=7.5, loc="lower left")
    fig.suptitle(f"[EXPLORE] Stage 2: which conserved quantities are prominent in the latent, by horizon ({ro} latent; exploratory)",
                 fontweight="bold", color=INK2)
    fig.savefig(REPO / out, dpi=125)
    plt.close(fig)


def plot_spectra(groups, out):
    _style()
    fig, axes = plt.subplots(1, len(HORIZONS) + 1, figsize=(19, 4.2), constrained_layout=True)
    for ax, k in zip(axes, HORIZONS):
        runs = groups.get(f"imaginer k={k}", [])
        if not runs:
            continue
        with open(S2 / "runs" / f"{runs[0]}.json", encoding="utf-8") as f:
            d = json.load(f)["diagnostics"]
        kk = np.arange(len(d["E_true"]))[1:]
        for key, col, lab in (("E_true", INK, "true frame t+k"), ("E_pred", SERIES[0], "imaginer prediction"), ("E_copy", INK2, "copy of frame t")):
            ax.loglog(kk, np.array(d[key])[1:], color=col, linewidth=1.6, label=lab, linestyle="-" if key != "E_copy" else (0, (3, 2)))
        ax.set_title(f"k = {k} ({runs[0].rsplit('_', 1)[1]}): pred/true k≥10 = {d['pred_over_true_energy_k_ge_10']:.2f}")
        ax.set_xlabel("k (shell)")
    axes[0].legend(fontsize=7.5)
    ax = axes[-1]
    ks, v = [], []
    for k in HORIZONS:
        for run in groups.get(f"summarizer k={k}", []):
            with open(S2 / "runs" / f"{run}.json", encoding="utf-8") as f:
                ks.append(k)
                v.append(json.load(f)["diagnostics"]["pred_over_target_map_variance"])
    ax.scatter(ks, v, color=SERIES[1], s=30)
    ax.set_xscale("log", base=2)
    ax.set_xticks(HORIZONS)
    ax.set_xticklabels([str(k) for k in HORIZONS])
    ax.set_title("summarizer: predicted / target map variance")
    fig.suptitle("[EXPLORE] Stage 2: how much the predictors picture vs average (val anchors; exploratory)", fontweight="bold", color=INK2)
    fig.savefig(REPO / out, dpi=120)
    plt.close(fig)


def decide_kstar(summ_primary) -> dict:
    path = S2 / "kstar.json"
    if path.exists():
        raise SystemExit("k* was already decided (results/explore/stage2/kstar.json); not re-deciding")
    e1 = {k: float(np.mean(summ_primary[f"imaginer k={k}"]["E1"])) for k in HORIZONS}
    e2 = {k: float(np.mean(summ_primary[f"imaginer k={k}"]["E2"])) for k in HORIZONS}
    seeds = {k: summ_primary[f"imaginer k={k}"]["n_seeds"] for k in HORIZONS}
    if not all(summ_primary.get(f"imaginer k={k}", {}).get("n_seeds", 0) >= 2 for k in HORIZONS):
        raise SystemExit("k* needs at least 2 non-diverged imaginer seeds per horizon (entry 5): stop and report")
    order = sorted(HORIZONS, key=lambda k: -e1[k])
    top, second = order[0], order[1]
    k_star = top if e1[top] - e1[second] > 0.02 else max((top, second), key=lambda k: e2[k])
    out = {"k_star": k_star, "rule": "argmax imaginer E1 (explore, pooled); top two within 0.02 -> larger E2 (PREDICTIONS entry 3)",
           "E1_by_k": e1, "E2_by_k": e2, "seeds_by_k": seeds}
    save_json(out, path)
    append_log("[EXPLORE] Stage 2 k* decision (before the fake-physics and pixel + VICReg runs)", "-", command_line(), f"k* = {k_star}",
               {"imaginer E1 by k": ", ".join(f"{k}: {v:.3f}" for k, v in e1.items()),
                "imaginer E2 by k": ", ".join(f"{k}: {v:.3f}" for k, v in e2.items())})
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--decide-kstar", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg["seed"])
    t0 = time.perf_counter()
    groups = groups_from_files()
    ds = CombinedDataset(cfg["data_parts"], cfg["cache_tag"])
    x = None  # inputs go to the GPU only if some latent is not cached yet
    log_p = traj_mean_log_p(ds, "explore")
    res = {ro: {g: [] for g in groups} for ro in cfg["readouts"]}
    for g, runs in groups.items():
        for run in runs:
            if x is None and not all(p.exists() for r, p in _paths(run, ds.cache_tag).items() if r in cfg["readouts"]):
                x = gpu_inputs(ds)
            lat = readouts_for(run, ds, x, ds.rows(["train"]), readouts=tuple(cfg["readouts"]))
            for ro in cfg["readouts"]:
                res[ro][g].append({"run": run, **evaluate_run(ds, lat[ro], cfg["n_pca"], log_p)})
            print(f"{g}: {run} done ({time.perf_counter() - t0:.0f} s)", flush=True)
    del x
    summ = {ro: summarize(res[ro], res[ro][CONTROL]) for ro in cfg["readouts"]}
    kstar = decide_kstar(summ[cfg["primary_readout"]]) if args.decide_kstar else None
    if kstar is None and (S2 / "kstar.json").exists():
        with open(S2 / "kstar.json", encoding="utf-8") as f:
            kstar = json.load(f)
    write_md(cfg, groups, summ, res, kstar, cfg["output_md"])
    plot_primary(summ, cfg["primary_readout"], cfg["figure_primary"])
    plot_spectra(groups, cfg["figure_spectra"])
    wall = time.perf_counter() - t0
    with gzip.open(REPO / cfg["output_json_gz"], "wt", encoding="utf-8") as f:
        json.dump({"environment": environment(), "config": cfg, "groups": groups, "summary": summ, "kstar": kstar, "results": res,
                   "wall_seconds": wall}, f, default=float)
    sm = summ[cfg["primary_readout"]]
    append_log("[EXPLORE] Stage 2 evaluation", cfg["_config_path"], command_line(), "ok",
               {"groups (runs)": ", ".join(f"{g} ({len(r)})" for g, r in groups.items()),
                "E1 imaginer by k": ", ".join(f"{k}: {np.mean(sm[f'imaginer k={k}']['E1']):.3f}" for k in HORIZONS if f"imaginer k={k}" in sm),
                "E1 summarizer by k": ", ".join(f"{k}: {np.mean(sm[f'summarizer k={k}']['E1']):.3f}" for k in HORIZONS if f"summarizer k={k}" in sm),
                "results": f"{cfg['output_md']}, {cfg['figure_primary']}, {cfg['figure_spectra']}, {cfg['output_json_gz']}"}, wall, environment().get("gpu"))
    print(open(REPO / cfg["output_md"], encoding="utf-8").read()[:5000])


if __name__ == "__main__":
    main()
