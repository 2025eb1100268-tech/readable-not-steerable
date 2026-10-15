"""[EXPLORE] Stage 1: does the pixel model's enstrophy lead survive? (existing checkpoints only)

    python -m explore.stage1 --config configs/explore/stage1.yaml

Exploratory (results/explore/BRIEF.md). The kill gate and the definitions were committed before
this ran (results/explore/PREDICTIONS.md, entries 1–2). Pooled latent (pre-registered); slow
directions fitted on v1 train+val inviscid runs exactly as pre-registered (probes/evaluate.py),
scored on explore and matched-energy inviscid runs.

  1a  per-direction and best-of-top-3 |r| with log Z₀ on explore; gate (a) = paired bootstrap of
      the pixel − control gap (probes/criteria.py s3)
  1b  partial r (log E₀, family) on explore, unscored; matched-energy |r| with log Z₀ = gate (b);
      robustness: gate (b) statistics with trajectory means over frames 0–15 only, unscored
  1c  regression of each model's enstrophy-tracking direction on scale-free descriptors at t = 0
Specificity against non-conserved statistics is reported next to every slow-direction result.
The map mean ⊕ std readout is reported as a secondary, unscored check.
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
import torch  # noqa: E402

from common.run import REPO, append_log, command_line, environment, load_config, seed_everything  # noqa: E402
from data.combined import CombinedDataset  # noqa: E402
from explore.readouts import gpu_inputs, readouts_for  # noqa: E402
from explore.stage0 import slow_extra  # noqa: E402
from probes import criteria  # noqa: E402
from probes import slow_features as sfeat  # noqa: E402
from probes.evaluate import slow_directions  # noqa: E402
from data.generate import build_ic  # noqa: E402
from solver.diagnostics import energy, energy_spectrum  # noqa: E402
from solver.ns2d import SpectralGrid  # noqa: E402
from solver.validate import INK, INK2, SERIES, _style  # noqa: E402

SETS = ("explore", "matched")
DESCRIPTORS = ("log E₀", "log(Z₀/E₀)", "high-k slope (k 8–20)", "log(P₀/Z₀)", "log kurtosis₀")


def slow_frames(ds, lat, eval_split: str, n_pca: int, max_frame: int | None = None) -> dict:
    """probes.evaluate.slow_directions with the eval trajectory means restricted to frames < max_frame
    (max_frame=None reproduces it exactly; checked in main)."""
    fit_rows = ds.rows(["train", "val"], viscous=False)
    ev_rows = ds.rows([eval_split], viscous=False)
    if max_frame is not None:
        ev_rows = ev_rows[ds.frame[ev_rows] < max_frame]
    model = sfeat.fit(lat[fit_rows], ds.traj[fit_rows], n_pca)
    proj = sfeat.project(model, lat[ev_rows])[:, :3]
    ids, inv = np.unique(ds.traj[ev_rows], return_inverse=True)
    counts = np.bincount(inv)
    means = np.stack([np.bincount(inv, weights=proj[:, j]) / counts for j in range(3)], 1)
    first = np.array([np.flatnonzero((ds.traj == t) & (ds.frame == 0))[0] for t in ids])
    return {"traj": ids.tolist(), "proj_traj_means": means.tolist(),
            "targets": {"log_E": np.log(ds.f["E"][first]).tolist(), "log_Z": np.log(ds.f["Z"][first]).tolist()}}


def descriptors(ds, eval_split: str, ic_cfg: dict) -> dict:
    """Scale-free descriptors at t = 0. Spectra come from the initial fields rebuilt exactly in
    float64 from their stored parameters (the float32 velocity's rounding floor would dominate the
    high-k tail of smooth fields); the rebuilt energy is checked against the stored E₀."""
    ev = ds.rows([eval_split], viscous=False)
    ids = np.unique(ds.traj[ev])
    first = np.array([np.flatnonzero((ds.traj == t) & (ds.frame == 0))[0] for t in ids])
    g = SpectralGrid(ds.velocity.shape[-1])
    plan = {t["traj"]: t for t in ds.trajectories}
    w_hat = torch.stack([build_ic(g, plan[int(t)], ic_cfg) * g.mask for t in ids])
    e_rebuilt = energy(g, w_hat).numpy()
    assert np.allclose(e_rebuilt, ds.f["E"][first], rtol=1e-9), "rebuilt initial fields do not match the stored E0"
    k, ek = energy_spectrum(g, w_hat)
    sel = ((k >= 8) & (k <= 20)).numpy()
    lk, le = np.log(k.numpy()[sel]), np.log(ek.numpy()[:, sel])
    slope = np.polyfit(lk, le.T, 1)[0]
    E, Z, P, K = (ds.f[q][first] for q in ("E", "Z", "P", "kurtosis"))
    return {"traj": ids.tolist(), "log E₀": np.log(E), "log(Z₀/E₀)": np.log(Z / E), "high-k slope (k 8–20)": slope,
            "log(P₀/Z₀)": np.log(P / Z), "log kurtosis₀": np.log(K)}


def regress_direction(slow: dict, desc: dict) -> dict:
    """The model's enstrophy-tracking direction (top-3 direction with the largest |r| with log Z₀),
    oriented to correlate positively with log Z₀, regressed on the standardized descriptors."""
    assert slow["traj"] == desc["traj"]
    proj, lz = np.array(slow["proj_traj_means"]), np.array(slow["targets"]["log_Z"])
    r = [np.corrcoef(proj[:, j], lz)[0, 1] for j in range(3)]
    j = int(np.argmax(np.abs(r)))
    y = proj[:, j] * np.sign(r[j])
    names = [d for d in DESCRIPTORS if np.std(desc[d]) > 1e-9 * (1 + np.abs(np.mean(desc[d])))]  # drop constants (log E₀ on matched)
    X = np.column_stack([(desc[d] - desc[d].mean()) / desc[d].std() for d in names])
    ys = (y - y.mean()) / y.std()
    coef, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(ys)), X]), ys, rcond=None)
    pred = np.column_stack([np.ones(len(ys)), X]) @ coef
    return {"direction": j + 1, "abs_r_logZ0": float(abs(r[j])), "r2": float(1 - ((ys - pred) ** 2).sum() / (ys**2).sum()),
            "std_coef": {d: float(c) for d, c in zip(names, coef[1:])},
            "r_with": {d: float(np.corrcoef(y, desc[d])[0, 1]) for d in names}}


def gate_stats(pix, ctrl):
    with warnings.catch_warnings():  # log E₀ is constant on matched-energy: its r is NaN by construction
        warnings.simplefilter("ignore", RuntimeWarning)
        return criteria.s3(pix, ctrl)["invariants"]["log_Z"]


def f3(v):
    return " / ".join(f"{x:.2f}" for x in v)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = load_config(ap.parse_args().config)
    seed_everything(cfg["seed"])
    t0 = time.perf_counter()
    ds = CombinedDataset(cfg["data_parts"], cfg["cache_tag"])
    x = gpu_inputs(ds)
    ic_cfg = {}
    for root, splits in cfg["data_parts"]:
        with open(REPO / root / "manifest.json", encoding="utf-8") as f:
            ic = json.load(f)["config"]["ic"]
        ic_cfg.update({s: ic for s in splits})
    desc = {s: descriptors(ds, s, ic_cfg[s]) for s in SETS}
    res = {ro: {s: {} for s in SETS + ("matched16",)} for ro in cfg["readouts"]}
    for g, runs in cfg["models"].items():
        for ro in cfg["readouts"]:
            for s in res[ro]:
                res[ro][s][g] = []
        for run in runs:
            lat = readouts_for(run, ds, x, ds.rows(["train"]))
            for ro in cfg["readouts"]:
                for s in SETS:
                    slow = slow_directions(ds, lat[ro], ["train", "val"], s, cfg["n_pca"])
                    mine = slow_frames(ds, lat[ro], s, cfg["n_pca"])
                    assert np.allclose(mine["proj_traj_means"], slow["proj_traj_means"]) and mine["traj"] == slow["traj"]
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        extra = slow_extra(ds, slow, s)
                    res[ro][s][g].append({"run": run, "slow": slow, "slow_extra": extra, "regression": regress_direction(slow, desc[s])})
                res[ro]["matched16"][g].append({"run": run, "slow": slow_frames(ds, lat[ro], "matched", cfg["n_pca"], max_frame=16)})
            print(f"{run} done ({time.perf_counter() - t0:.0f} s)", flush=True)
    del x

    # gates (pooled readout decides; other readouts reported only)
    gates = {}
    for ro in cfg["readouts"]:
        a = gate_stats(res[ro]["explore"]["pixel"], res[ro]["explore"]["untrained"])
        b = gate_stats(res[ro]["matched"]["pixel"], res[ro]["matched"]["untrained"])
        b16 = gate_stats(res[ro]["matched16"]["pixel"], res[ro]["matched16"]["untrained"])
        gates[ro] = {"a": {**a, "pass": bool(a["gap"] >= 0.1 and a["ci95"][0] > 0)},
                     "b": {**b, "pass": bool(b["r_jepa"] >= 0.5 and b["gap"] > 0 and b["ci95"][0] > 0)},
                     "b_first16_frames": {**b16, "pass": bool(b16["r_jepa"] >= 0.5 and b16["gap"] > 0 and b16["ci95"][0] > 0)}}
        gates[ro]["survives"] = gates[ro]["a"]["pass"] and gates[ro]["b"]["pass"]
    context = {ro: {s: {f"{a} vs {b}": gate_stats(res[ro][s][a], res[ro][s][b]) for a, b in
                        (("jepa_spatial", "untrained"), ("pixel", "jepa_spatial"))} for s in ("explore", "matched")} for ro in cfg["readouts"]}
    wall = time.perf_counter() - t0
    G = gates[cfg["deciding_readout"]]

    # markdown
    L = ["# [EXPLORE] Stage 1: does the pixel model's enstrophy lead survive?", "",
         "Exploratory, not a finding. Gate and definitions committed beforehand (results/explore/PREDICTIONS.md, entries 1–2). "
         "Slow directions fitted on v1 train+val inviscid runs as pre-registered; scored on explore (378 inviscid runs) and "
         "matched-energy (300 inviscid runs, all at E₀ = 0.2893). Wording rule: these statistics say which conserved quantities are "
         "prominent in a latent, not that a model discovered conservation.", "",
         f"## Kill gate ({cfg['deciding_readout']} latent, as committed)", "",
         "| condition | pixel | untrained | gap [95% CI] | rule | outcome |", "|---|---|---|---|---|---|"]
    for key, rule in (("a", "gap ≥ 0.1 and CI > 0 (explore)"), ("b", "pixel ≥ 0.5, gap > 0, CI > 0 (matched-energy)"),
                      ("b_first16_frames", "same as (b), frames 0–15 only (unscored robustness check)")):
        v = G[key]
        L.append(f"| ({key}) | {v['r_jepa']:.3f} | {v['r_control']:.3f} | {v['gap']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] | {rule} | "
                 f"{'pass' if v['pass'] else '**fail**'} |")
    L += ["", f"**Gate: the lead {'survives; Stage 2 may start (with your approval)' if G['survives'] else 'does NOT survive; stop, do not start Stage 2'}.**", ""]
    L += ["Context (unscored): best-of-top-3 enstrophy |r|, other pairs.", "", "| set | comparison | first | second | gap [95% CI] |", "|---|---|---|---|---|"]
    for s, per in context[cfg["deciding_readout"]].items():
        for name, v in per.items():
            L.append(f"| {s} | {name} | {v['r_jepa']:.3f} | {v['r_control']:.3f} | {v['gap']:+.3f} [{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}] |")
    for ro in cfg["readouts"]:
        L += ["", f"## Every direction, every seed ({ro} latent){'' if ro == cfg['deciding_readout'] else ' — secondary, unscored'}", "",
              "Directions 1–3, slowest first. |r| between the direction's trajectory-mean projection and each quantity.", "",
              "| set | model | seed | log Z₀ | log E₀ | partial r log Z₀ \\| log E₀, family | traj-mean log P (not conserved) | traj-mean log kurtosis | traj-mean log max\\|ω\\| | held-out λ |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for s in SETS:
            for g in cfg["models"]:
                for r in res[ro][s][g]:
                    sl, ex = r["slow"], r["slow_extra"]
                    sp = ex["specificity_abs_r"]
                    L.append(f"| {s} | {g} | {r['run'].rsplit('_s', 1)[1]} | {f3(sl['abs_r']['log_Z'])} | "
                             f"{f3(sl['abs_r']['log_E']) if s == 'explore' else '— (E₀ fixed)'} | {f3(ex['partial_r_logZ0_given_logE0_family'])} | "
                             f"{f3(sp['log P (traj mean, not conserved)'])} | {f3(sp['log kurtosis (traj mean)'])} | {f3(sp['log max|ω| (traj mean)'])} | "
                             f"{f3(sl['lambda_heldout'][:3])} |")
        if ro != cfg["deciding_readout"]:
            v = gates[ro]
            L += ["", f"Gate statistics with this readout (unscored): (a) gap {v['a']['gap']:+.3f} [{v['a']['ci95'][0]:+.3f}, {v['a']['ci95'][1]:+.3f}]; "
                  f"(b) pixel {v['b']['r_jepa']:.3f} vs {v['b']['r_control']:.3f}, gap {v['b']['gap']:+.3f} [{v['b']['ci95'][0]:+.3f}, {v['b']['ci95'][1]:+.3f}]."]
    L += ["", f"## 1c. What the enstrophy-tracking direction is ({cfg['deciding_readout']} latent)", "",
          "Per seed: the top-3 direction with the largest |r| with log Z₀, regressed on standardized descriptors at t = 0 "
          "(log E₀ is constant on matched-energy and dropped there). Note log Z₀ = log E₀ + log(Z₀/E₀) exactly.", "",
          "| set | model | seed | dir | \\|r\\| log Z₀ | R² | " + " | ".join(f"β {d}" for d in DESCRIPTORS) + " | " + " | ".join(f"r {d}" for d in DESCRIPTORS) + " |",
          "|---|---|---|---|---|---|" + "---|" * (2 * len(DESCRIPTORS))]
    for s in SETS:
        for g in cfg["models"]:
            for r in res[cfg["deciding_readout"]][s][g]:
                rg = r["regression"]
                L.append(f"| {s} | {g} | {r['run'].rsplit('_s', 1)[1]} | {rg['direction']} | {rg['abs_r_logZ0']:.2f} | {rg['r2']:.2f} | "
                         + " | ".join(f"{rg['std_coef'][d]:+.2f}" if d in rg["std_coef"] else "—" for d in DESCRIPTORS) + " | "
                         + " | ".join(f"{rg['r_with'][d]:+.2f}" if d in rg["r_with"] else "—" for d in DESCRIPTORS) + " |")
    ex0 = res[cfg["deciding_readout"]]["explore"]["untrained"][0]["slow_extra"]["corr_logE0_logZ0"]
    L += ["", f"corr(log E₀, log Z₀) over explore inviscid runs: {ex0:.3f}.", ""]
    with open(REPO / cfg["output_md"], "w", encoding="utf-8") as f:
        f.write("\n".join(L))

    # figure
    _style()
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), constrained_layout=True)
    ro = cfg["deciding_readout"]
    panels = [("explore: |r| with log Z₀", "explore", lambda r: r["slow"]["abs_r"]["log_Z"], 0.1),
              ("matched-energy: |r| with log Z₀", "matched", lambda r: r["slow"]["abs_r"]["log_Z"], 0.5),
              ("explore: partial r, log Z₀ | log E₀, family", "explore", lambda r: r["slow_extra"]["partial_r_logZ0_given_logE0_family"], None)]
    groups = list(cfg["models"])
    for ax, (title, s, get, line) in zip(axes, panels):
        for i, g in enumerate(groups):
            vals = np.array([get(r) for r in res[ro][s][g]])  # seeds x 3 directions
            for d, col in zip(range(3), SERIES):
                ax.scatter(np.full(len(vals), i + (d - 1) * 0.22), vals[:, d], s=22, color=col, linewidths=0, zorder=3,
                           label=f"direction {d + 1}" if i == 0 else None)
            ax.plot([i - 0.4, i + 0.4], [vals.max(1).mean()] * 2, color=INK, linewidth=2, zorder=4, label="best of top 3 (seed mean)" if i == 0 else None)
        if s == "matched":
            ax.axhline(line, color=INK2, linewidth=1, linestyle=(0, (2, 2)), label="gate (b): pixel ≥ 0.5")
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(groups)
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
    axes[0].legend(fontsize=7.5, loc="upper left")
    axes[1].legend(fontsize=7.5, loc="upper left")
    fig.suptitle(f"[EXPLORE] Stage 1: enstrophy in the slow directions (gate {'passed' if G['survives'] else 'FAILED'}; exploratory)",
                 fontweight="bold", color=INK2)
    fig.savefig(REPO / cfg["figure"], dpi=125)
    plt.close(fig)

    with gzip.open(REPO / cfg["output_json_gz"], "wt", encoding="utf-8") as f:
        json.dump({"environment": environment(), "config": cfg, "gates": gates, "context": context, "results": res,
                   "descriptors": {s: {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in d.items()} for s, d in desc.items()},
                   "wall_seconds": wall}, f, default=float)
    append_log("[EXPLORE] Stage 1: does the pixel model's enstrophy lead survive? (kill gate)", cfg["_config_path"], command_line(),
               f"gate {'passed: lead survives' if G['survives'] else 'FAILED: lead does not survive'}",
               {"(a) explore gap": f"{G['a']['gap']:+.3f} [{G['a']['ci95'][0]:+.3f}, {G['a']['ci95'][1]:+.3f}]",
                "(b) matched pixel / control": f"{G['b']['r_jepa']:.3f} / {G['b']['r_control']:.3f}, gap CI [{G['b']['ci95'][0]:+.3f}, {G['b']['ci95'][1]:+.3f}]",
                "(b) frames 0–15 (unscored)": f"{G['b_first16_frames']['r_jepa']:.3f} / {G['b_first16_frames']['r_control']:.3f}",
                "results": f"{cfg['output_md']}, {cfg['figure']}, {cfg['output_json_gz']}"}, wall, environment().get("gpu"))
    print("\n".join(L[:20]))


if __name__ == "__main__":
    main()
