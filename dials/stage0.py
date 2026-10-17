"""[BUILD] Stage 0: the steering harness on existing models (results/build/PREDICTIONS.md, entry 1).

    python -m dials.stage0 --config configs/build/stage0.yaml

Exploratory. For each model: the top-4 raw-PCA directions of the pooled latent (v1 train frames)
and the supervised ridge direction for s = log(Z/E) are steered on 256 explore and 256
matched-energy frames; every metric of dials/harness.py is reported for every direction, seed and
dataset; label-based readability (|r| of each direction's projection with a and s over all explore
frames) is reported for every model including the untrained control (which has no decoder).
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from common.run import REPO, append_log, command_line, environment, load_config, seed_everything  # noqa: E402
from data.combined import CombinedDataset  # noqa: E402
from dials.adapters import ExistingModel  # noqa: E402
from dials.harness import classify, envelope, pca_directions, steer, supervised_direction  # noqa: E402
from explore.readouts import readouts_for  # noqa: E402
from solver.ns2d import SpectralGrid  # noqa: E402
from solver.validate import INK, INK2, SERIES, _style  # noqa: E402
from train.train import Model  # noqa: E402

SETS = ("explore", "matched")


def frames(ds, split, n, k, seed):
    rows = ds.rows([split])
    nf = {t["traj"]: t["n_frames"] for t in ds.trajectories}
    n_frames = np.array([nf[int(t)] for t in ds.traj[rows]])
    ok = rows[(ds.frame[rows] >= 1) & (ds.frame[rows] + k <= n_frames - 1)]
    return np.sort(np.random.default_rng(seed).choice(ok, n, replace=False))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = load_config(ap.parse_args().config)
    seed_everything(cfg["seed"])
    t0 = time.perf_counter()
    ds = CombinedDataset(cfg["data_parts"], cfg["cache_tag"])
    g = SpectralGrid(ds.velocity.shape[-1], "cuda")
    to_gpu = lambda rows: torch.from_numpy(ds.velocity[rows]).cuda() / ds.input_scale  # noqa: E731
    tr, va = ds.rows(["train"]), ds.rows(["val"])
    fr = {s: frames(ds, s, cfg["n_frames"], 4, cfg["seed"]) for s in SETS}
    x = {s: (to_gpu(fr[s] - 1), to_gpu(fr[s])) for s in SETS}
    env = envelope(to_gpu(np.sort(np.random.default_rng(cfg["seed"]).choice(tr, cfg["n_envelope"], replace=False))), g)
    a_all, s_all = np.log(ds.f["E"]), np.log(ds.f["Z"] / ds.f["E"])
    ex_all = ds.rows(["explore"])
    crit = cfg["criteria"]
    res = {}
    for label, runs in cfg["groups"].items():
        res[label] = []
        for run in runs:
            lat = readouts_for(run, ds, None, tr, readouts=("pooled",))["pooled"].astype(np.float64)
            dirs, sig, expl = pca_directions(lat[tr])
            d_sup, sig_sup, alpha_sup = supervised_direction(lat[tr], s_all[tr], ds.frame[tr], lat[va], s_all[va], ds.frame[va])
            cand = {f"PC{i + 1}": (dirs[i], float(sig[i])) for i in range(len(dirs))}
            cand["supervised s"] = (d_sup, sig_sup)
            entry = {"run": run, "explained": expl.tolist(), "supervised_ridge_alpha": alpha_sup, "directions": {}}
            for name, (d, s_d) in cand.items():
                proj = lat[ex_all] @ d
                e = {"sigma": s_d, "readability": {"abs_r_a": float(abs(np.corrcoef(proj, a_all[ex_all])[0, 1])),
                                                   "abs_r_s": float(abs(np.corrcoef(proj, s_all[ex_all])[0, 1]))}}
                entry["directions"][name] = e
            if not run.startswith("untrained"):
                ckpt = torch.load(REPO / "results" / "checkpoints" / f"{run}.pt", map_location="cpu", weights_only=False)
                model = Model(ckpt["config"])
                model.load_state_dict(ckpt["model"])
                ad = ExistingModel(model.cuda())
                for name, (d, s_d) in cand.items():
                    e = entry["directions"][name]
                    for s in SETS:
                        m = steer(ad, x[s][0], x[s][1], d, s_d, g, env)
                        m.pop("heads", None)
                        e[s] = {**m, **classify(m, crit)}
                    e.update({k_: bool(e["explore"][k_] and e["matched"][k_]) for k_ in ("dial", "loudness", "shape")})
            res[label].append(entry)
            print(f"{label}: {run} done ({time.perf_counter() - t0:.0f} s)", flush=True)
    wall = time.perf_counter() - t0
    with gzip.open(REPO / cfg["output_json_gz"], "wt", encoding="utf-8") as f:
        json.dump({"environment": environment(), "config": cfg, "results": res, "wall_seconds": wall}, f, default=float)

    # ---- markdown -------------------------------------------------------------------------
    steerable = {k: v for k, v in res.items() if not v[0]["run"].startswith("untrained")}
    names = ["PC1", "PC2", "PC3", "PC4", "supervised s"]
    L = ["# [BUILD] Stage 0: steering the existing models (exploratory)", "",
         "Definitions: results/build/PREDICTIONS.md entry 1. Pooled-latent shifts applied at every map position; "
         "slopes are log-units per unit α (median over frames). A dial needs effect ≥ "
         f"{crit['min_effect']}, monotonicity ≥ {crit['min_monotonicity']}, cycle corr ≥ {crit['min_cycle_corr']}, "
         f"envelope ≥ {crit['min_envelope']} on both explore and matched-energy; loudness σ_shape ≤ {crit['max_sigma_loudness']}, "
         f"shape σ_shape ≥ {crit['min_sigma_shape']}.", "",
         "## Counts (all seeds)", "", "| model | candidate dials (PC1–4) | dials | loudness | shape | supervised s: dial? | supervised s: shape? |",
         "|---|---|---|---|---|---|---|"]
    tot = {"cand": 0, "dial": 0, "loud": 0, "shape": 0}
    for label, runs in steerable.items():
        c = [r["directions"][n] for r in runs for n in names[:4]]
        sup = [r["directions"]["supervised s"] for r in runs]
        row = (len(c), sum(e["dial"] for e in c), sum(e["loudness"] for e in c), sum(e["shape"] for e in c))
        tot["cand"] += row[0]; tot["dial"] += row[1]; tot["loud"] += row[2]; tot["shape"] += row[3]  # noqa: E702
        L.append(f"| {label} | {row[0]} | {row[1]} | {row[2]} | {row[3]} | {sum(e['dial'] for e in sup)}/{len(sup)} | "
                 f"{sum(e['shape'] for e in sup)}/{len(sup)} |")
    L.append(f"| **all** | {tot['cand']} | {tot['dial']} | {tot['loud']} | {tot['shape']} | | |")
    f3 = lambda v: f"{v:.3f}"  # noqa: E731
    for s in SETS:
        L += ["", f"## Every direction, every seed — {s}", "",
              "| model | run | dir | σ_d | Δa | Δs | Δlog P | effect | σ_shape | monot. | cycle corr / gain | drift | envelope min | shared pattern | dial / loud / shape |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for label, runs in steerable.items():
            for r in runs:
                for n in names:
                    e = r["directions"][n]
                    m = e[s]
                    L.append(f"| {label} | {r['run'].rsplit('_s', 1)[1]} | {n} | {e['sigma']:.3g} | {m['d_a']:+.3f} | {m['d_s']:+.3f} | "
                             f"{m['d_logP']:+.3f} | {f3(m['effect'])} | {f3(m['sigma_shape'])} | {f3(m['monotonicity'])} | "
                             f"{m['cycle_corr']:.2f} / {m['cycle_gain']:.2f} | {m['drift']:.2f} | {m['envelope_min']:.2f} | "
                             f"{m['shared_pattern_fraction']:.2f} | "
                             f"{'D' if m['dial'] else '·'}{'L' if m['loudness'] else '·'}{'S' if m['shape'] else '·'} |")
    L += ["", "## Readability (label-based, |r| over all explore frames), all models incl. the untrained control", "",
          "| model | run | " + " | ".join(f"{n}: \\|r\\| a / s" for n in names) + " | PC1–4 explained variance |",
          "|---|---|" + "---|" * (len(names) + 1)]
    for label, runs in res.items():
        for r in runs:
            L.append(f"| {label} | {r['run'].rsplit('_s', 1)[1]} | " + " | ".join(
                f"{r['directions'][n]['readability']['abs_r_a']:.2f} / {r['directions'][n]['readability']['abs_r_s']:.2f}" for n in names)
                + " | " + ", ".join(f"{v:.2f}" for v in r["explained"]) + " |")
    allm = [r["directions"][n][s] for runs in steerable.values() for r in runs for n in names[:4] for s in SETS]
    q = lambda key: np.percentile([m[key] for m in allm], [10, 50, 90])  # noqa: E731
    L += ["", "## Distributions over all candidate dials (both sets), for the threshold proposal", ""]
    for key in ("effect", "sigma_shape", "monotonicity", "cycle_corr", "cycle_gain", "envelope_min", "shared_pattern_fraction"):
        p = q(key)
        L.append(f"- {key}: p10 {p[0]:.3f}, median {p[1]:.3f}, p90 {p[2]:.3f}")
    with open(REPO / cfg["output_md"], "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")

    # ---- figure --------------------------------------------------------------------------------
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), constrained_layout=True)
    for ax, s in zip(axes, SETS):
        for i, (label, runs) in enumerate(steerable.items()):
            col = (SERIES + [INK2, INK])[i % 7]
            for r in runs:
                for n in names:
                    m = r["directions"][n][s]
                    ax.scatter(max(m["effect"], 1e-4), m["sigma_shape"], color=col, s=26 if n != "supervised s" else 60,
                               marker="o" if n != "supervised s" else "*", linewidths=0, alpha=0.85,
                               label=label if (r is runs[0] and n == "PC1") else None)
        ax.axvline(crit["min_effect"], color=INK2, linewidth=1, linestyle=(0, (3, 2)))
        for yv in (crit["max_sigma_loudness"], crit["min_sigma_shape"]):
            ax.axhline(yv, color=INK2, linewidth=0.8, linestyle=(0, (1, 2)))
        ax.set_xscale("log")
        ax.set_xlabel("effect |Δa| + |Δs| per unit α (log-units)")
        ax.set_ylabel("shape selectivity σ_shape")
        ax.set_title(f"{s}: each candidate dial (dots: PC1–4, stars: supervised s)")
    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle("[BUILD] Stage 0: how far does a pooled-latent shift move the decoded field? (exploratory)", fontweight="bold", color=INK)
    fig.savefig(REPO / cfg["figure"], dpi=125)
    plt.close(fig)
    append_log("[BUILD] Stage 0: steering harness on existing models", cfg["_config_path"], command_line(), "ok",
               {"candidate dials / dials / loudness / shape": f"{tot['cand']} / {tot['dial']} / {tot['loud']} / {tot['shape']}",
                "results": f"{cfg['output_md']}, {cfg['figure']}, {cfg['output_json_gz']}"}, wall, environment().get("gpu"))
    print("\n".join(L[:24]))


if __name__ == "__main__":
    main()
