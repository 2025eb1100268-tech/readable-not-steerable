"""[EXPLORE] Step 2 of PREDICTIONS entry 6: does a pixel predictor *use* its enstrophy direction?

    python -m explore.imagination_probe --config configs/explore/probe.yaml

For each imaginer: slow directions fitted as pre-registered on the pooled latent (v1 train+val
inviscid); its enstrophy-tracking and energy-tracking directions (top-3 direction with the largest
|r| with log Z0 / log E0 on explore inviscid trajectory means), each oriented so that + increases
that quantity. A direction's activation pattern is the regression of the pooled latent on the slow
coordinate over explore inviscid frames, scaled to one standard deviation of the coordinate. The
intervention adds delta x pattern to every position of both context maps (so the pooled latent of
each context frame moves by exactly delta x pattern), decodes the frame at t + k, and measures
log Z, log E and log P of the rendered field. Effect = least-squares slope over delta. Baseline:
random directions with the enstrophy pattern's norm. Exploratory; nothing here is a finding.
"""
from __future__ import annotations

import argparse
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from common.run import REPO, append_log, command_line, environment, load_config, save_json, seed_everything  # noqa: E402
from data.combined import CombinedDataset  # noqa: E402
from explore.readouts import _paths, gpu_inputs, readouts_for  # noqa: E402
from probes import slow_features as sfeat  # noqa: E402
from solver import diagnostics as dg  # noqa: E402
from solver.ns2d import SpectralGrid  # noqa: E402
from solver.validate import INK, INK2, SERIES, _style  # noqa: E402
from train.train import Model  # noqa: E402

QUANTITIES = ("log Z", "log E", "log P")


def directions(ds, lat, n_pca):
    """Enstrophy- and energy-tracking slow directions and their pooled-space patterns (1 std step)."""
    fit = ds.rows(["train", "val"], viscous=False)
    model = sfeat.fit(lat[fit], ds.traj[fit], n_pca)
    ev = ds.rows(["explore"], viscous=False)
    proj = sfeat.project(model, lat[ev])[:, :3]
    ids, inv = np.unique(ds.traj[ev], return_inverse=True)
    means = np.stack([np.bincount(inv, weights=proj[:, j]) / np.bincount(inv) for j in range(3)], 1)
    first = np.array([np.flatnonzero((ds.traj == t) & (ds.frame == 0))[0] for t in ids])
    out = {}
    for name, q in (("enstrophy", "Z"), ("energy", "E")):
        y = np.log(ds.f[q][first])
        r = np.array([np.corrcoef(means[:, j], y)[0, 1] for j in range(3)])
        j = int(np.argmax(np.abs(r)))
        p = proj[:, j] * np.sign(r[j])
        pc = p - p.mean()
        pattern = (lat[ev] - lat[ev].mean(0)).T @ pc / (pc @ pc) * p.std()
        out[name] = {"direction": j + 1, "abs_r": float(abs(r[j])), "pattern": pattern.astype(np.float32)}
    return out


@torch.no_grad()
def render_stats(model, x_prev, x_t, shift, scale, g) -> dict:
    maps = model.enc(torch.cat([x_prev, x_t]))
    maps = maps + shift[None, :, None, None]
    b = len(x_t)
    pred = model.head(torch.cat([maps[:b], maps[b:]], 1), torch.zeros(b, dtype=torch.long, device=x_t.device)).double() * scale
    w = g.vorticity_hat_from_velocity(pred[:, 0], pred[:, 1])
    return {"log Z": float(torch.log(dg.enstrophy(g, w)).mean()), "log E": float(torch.log(dg.energy(g, w)).mean()),
            "log P": float(torch.log(dg.palinstrophy(g, w)).mean())}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = load_config(ap.parse_args().config)
    seed_everything(cfg["seed"])
    t0 = time.perf_counter()
    ds = CombinedDataset(cfg["data_parts"], cfg["cache_tag"])
    g = SpectralGrid(ds.velocity.shape[-1], "cuda")
    rng = np.random.default_rng(cfg["seed"])
    x = None
    deltas = np.array(cfg["deltas"], float)
    results = {}
    for run in cfg["runs"]:
        if x is None and not all(p.exists() for r, p in _paths(run, ds.cache_tag).items() if r in ("pooled", "map_meanstd")):
            x = gpu_inputs(ds)
        lat = readouts_for(run, ds, x, ds.rows(["train"]), readouts=("pooled", "map_meanstd"))["pooled"]
        ckpt = torch.load(REPO / "results" / "checkpoints" / f"{run}.pt", map_location="cpu", weights_only=False)
        model = Model(ckpt["config"])
        model.load_state_dict(ckpt["model"])
        model = model.cuda().eval()
        k = ckpt["config"]["horizons"][0]
        dirs = directions(ds, lat, cfg["n_pca"])
        ev = ds.rows(["explore"], viscous=False)
        ok = ev[(ds.frame[ev] >= 1) & (ds.frame[ev] <= 23 - k)]
        anchors = np.sort(rng.choice(ok, cfg["n_anchors"], replace=False))
        to_gpu = lambda rows: torch.from_numpy(ds.velocity[rows]).cuda() / ds.input_scale  # noqa: E731
        x_prev, x_t = to_gpu(anchors - 1), to_gpu(anchors)
        norm = float(np.linalg.norm(dirs["enstrophy"]["pattern"]))
        rand = [v / np.linalg.norm(v) * norm for v in rng.standard_normal((cfg["n_random"], lat.shape[1]))]
        tests = {"enstrophy direction": dirs["enstrophy"]["pattern"], "energy direction": dirs["energy"]["pattern"],
                 **{f"random {i}": v for i, v in enumerate(rand)}}
        curves = {}
        for tname, vec in tests.items():
            v = torch.as_tensor(np.asarray(vec, np.float32), device="cuda")
            stats = [render_stats(model, x_prev, x_t, d * v, ds.input_scale, g) for d in deltas]
            curves[tname] = {q: [s[q] for s in stats] for q in QUANTITIES}
        slope = lambda ys: float(np.polyfit(deltas, ys, 1)[0])  # noqa: E731
        slopes = {t: {q: slope(c[q]) for q in QUANTITIES} for t, c in curves.items()}
        rnd = np.array([[abs(slopes[f"random {i}"][q]) for q in QUANTITIES] for i in range(cfg["n_random"])])
        sz, se = slopes["enstrophy direction"], slopes["energy direction"]
        res = {"k": k, "directions": {n: {kk: vv for kk, vv in d.items() if kk != "pattern"} for n, d in dirs.items()},
               "pattern_norms": {n: float(np.linalg.norm(d["pattern"])) for n, d in dirs.items()}, "curves": curves, "slopes": slopes,
               "random_abs_slope_p95": dict(zip(QUANTITIES, np.percentile(rnd, 95, axis=0).tolist())),
               "random_abs_slope_median": dict(zip(QUANTITIES, np.median(rnd, axis=0).tolist())),
               "P6.1": {"enstrophy dir: |dlogZ| >= 1.5 |dlogE|": bool(abs(sz["log Z"]) >= 1.5 * abs(sz["log E"])),
                        "enstrophy dir: |dlogZ| > random p95": bool(abs(sz["log Z"]) > np.percentile(rnd[:, 0], 95)),
                        "energy dir: |dlogE| > |dlogZ|": bool(abs(se["log E"]) > abs(se["log Z"]))}}
        res["P6.1"]["holds"] = all(res["P6.1"].values())
        results[run] = res
        print(f"{run}: enstrophy dir slopes {sz}, energy dir slopes {se}, P6.1 {res['P6.1']['holds']}", flush=True)
    wall = time.perf_counter() - t0
    save_json({"environment": environment(), "config": cfg, "results": results, "wall_seconds": wall}, cfg["output_json"])

    L = ["# [EXPLORE] Imagination probe: does the pixel predictor use its enstrophy direction?", "",
         "Exploratory (PREDICTIONS entry 6, step 2). Shift both context maps along a slow direction's pooled-latent pattern by δ "
         "standard deviations of that slow coordinate, decode the frame at t + k, and measure the rendered field. Slopes are per unit δ "
         f"(mean over {cfg['n_anchors']} explore inviscid anchors). Random baseline: {cfg['n_random']} random directions with the enstrophy "
         "pattern's norm.", "",
         "| run | k | direction (slow dir #, \\|r\\|) | d log Z / dδ | d log E / dδ | d log P / dδ |", "|---|---|---|---|---|---|"]
    for run, r in results.items():
        for t in ("enstrophy direction", "energy direction"):
            d = r["directions"][t.split()[0]]
            s = r["slopes"][t]
            L.append(f"| {run} | {r['k']} | {t} ({d['direction']}, {d['abs_r']:.2f}) | {s['log Z']:+.4f} | {s['log E']:+.4f} | {s['log P']:+.4f} |")
        L.append(f"| {run} | {r['k']} | random: median / 95th pct of \\|slope\\| | "
                 + " | ".join(f"{r['random_abs_slope_median'][q]:.4f} / {r['random_abs_slope_p95'][q]:.4f}" for q in QUANTITIES) + " |")
    L += ["", "P6.1 checks per run:", ""]
    for run, r in results.items():
        L.append(f"- {run}: " + "; ".join(f"{k_}: {'yes' if v else 'no'}" for k_, v in r["P6.1"].items()))
    with open(REPO / cfg["output_md"], "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")

    _style()
    fig, axes = plt.subplots(1, len(results), figsize=(5.6 * len(results), 4.4), constrained_layout=True, squeeze=False)
    for ax, (run, r) in zip(axes[0], results.items()):
        for q, col in zip(QUANTITIES, SERIES):
            for i in range(cfg["n_random"]):
                c = r["curves"][f"random {i}"][q]
                ax.plot(deltas, np.array(c) - c[2], color=col, alpha=0.12, linewidth=0.8)
            cz, ce = r["curves"]["enstrophy direction"][q], r["curves"]["energy direction"][q]
            ax.plot(deltas, np.array(cz) - cz[2], color=col, linewidth=2.2, marker="o", label=f"{q}: enstrophy direction")
            ax.plot(deltas, np.array(ce) - ce[2], color=col, linewidth=1.6, linestyle=(0, (4, 2)), label=f"{q}: energy direction")
        ax.axhline(0, color=INK2, linewidth=0.8)
        ax.set_title(f"{run.replace('s2_', '')}")
        ax.set_xlabel("shift δ (std of the slow coordinate)")
        ax.set_ylabel("change in the rendered field (log units)")
    axes[0][0].legend(fontsize=7)
    fig.suptitle("[EXPLORE] Imagination probe: rendered field vs latent shift (faint lines = random directions; exploratory)",
                 fontweight="bold", color=INK)
    fig.savefig(REPO / cfg["figure"], dpi=120)
    plt.close(fig)
    append_log("[EXPLORE] Imagination probe (entry 6, step 2)", cfg["_config_path"], command_line(), "ok",
               {"P6.1 holds per run": ", ".join(f"{run}: {r['P6.1']['holds']}" for run, r in results.items()),
                "results": f"{cfg['output_md']}, {cfg['figure']}, {cfg['output_json']}"}, wall, environment().get("gpu"))
    print("\n".join(L))


if __name__ == "__main__":
    main()
