"""Phase 2b: dataset for the non-physical dynamics control (PREREGISTRATION.md §4.5).

    python -m data.nonphysical --config configs/phase2b_nonphys_data.yaml

Same trajectories as the physical dataset (identical initial conditions, splits, frame counts and
snapshot interval); only the dynamics differ. Each Fourier mode of the vorticity evolves as
    omega_k(t) = omega_k(0) exp((-r g_k + i c_k) t)
with g_k > 0 and c_k random but fixed for the whole dataset (g symmetric, c antisymmetric in k so
fields stay real) and a hidden per-trajectory rate r (no r = 0 runs). Nothing quadratic and no
Casimir is conserved. The model input is scaled with the *physical* dataset's input scale, the
same scale used when this encoder is later evaluated on Navier–Stokes data.
"""
from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import torch

from common.run import REPO, append_log, command_line, environment, load_config, save_json, seed_everything
from data.generate import build_ic, sha256
from solver import diagnostics as dg
from solver.ns2d import SpectralGrid


def mode_rates(grid: SpectralGrid, cfg: dict, rng: np.random.Generator):
    """Per-mode damping multipliers g_k (symmetric) and phase speeds c_k (antisymmetric)."""
    n = grid.n
    lo, hi = cfg["g_range"]
    g_raw = np.exp(rng.uniform(math.log(lo), math.log(hi), size=(n, n)))
    s_raw = rng.uniform(-1.0, 1.0, size=(n, n))
    neg = lambda a: np.roll(np.flip(a, (0, 1)), 1, (0, 1))  # a(-k) on the fft grid  # noqa: E731
    g = 0.5 * (g_raw + neg(g_raw))
    s = 0.5 * (s_raw - neg(s_raw))
    kmag = np.sqrt(np.fft.fftfreq(n, 1 / n)[:, None] ** 2 + np.fft.fftfreq(n, 1 / n)[None, :] ** 2)
    c = cfg["c0"] * s * kmag
    half = n // 2 + 1
    to_t = lambda a: torch.as_tensor(a[:, :half], dtype=grid.dtype, device=grid.device)  # noqa: E731
    return to_t(g), to_t(c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = load_config(ap.parse_args().config)
    seed_everything(cfg["seed"])
    t0 = time.perf_counter()
    src = REPO / cfg["source_dir"]
    with open(src / "trajectories.json", encoding="utf-8") as f:
        plans = json.load(f)
    with open(src / "manifest.json", encoding="utf-8") as f:
        src_manifest = json.load(f)
    phys_cfg = src_manifest["config"]
    out_dir = REPO / cfg["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    g = SpectralGrid(phys_cfg["n"], cfg["device"])
    rng = np.random.default_rng(cfg["seed"])
    gk, ck = mode_rates(g, cfg, rng)
    lo, hi = cfg["r_range"]
    total = sum(p["n_frames"] for p in plans)
    vel = np.lib.format.open_memmap(out_dir / "velocity.npy", mode="w+", dtype=np.float32, shape=(total, 2, g.n, g.n))
    idx = {"traj": np.zeros(total, np.int32), "frame": np.zeros(total, np.int32), "time": np.zeros(total)}
    tg = {k: np.zeros(total) for k in ("E", "Z")}
    retention, change = [], []
    row = 0
    new_plans = []
    for p in plans:
        r = math.exp(rng.uniform(math.log(lo), math.log(hi)))
        w0 = build_ic(g, p, phys_cfg["ic"])
        lin = -r * gk + 1j * ck
        z0 = float(dg.enstrophy(g, w0))
        prev = None
        for fidx in range(p["n_frames"]):
            t = fidx * phys_cfg["dt_out"]
            wh = (w0 * torch.exp(lin * t)) * g.mask
            u, v = g.velocity(wh)
            uv = torch.stack([u, v]).float().cpu().numpy()
            vel[row] = uv
            idx["traj"][row], idx["frame"][row], idx["time"][row] = p["traj"], fidx, t
            tg["E"][row], tg["Z"][row] = float(dg.energy(g, wh)), float(dg.enstrophy(g, wh))
            if prev is not None:
                change.append(float(np.linalg.norm(uv - prev) / np.linalg.norm(prev)))
            prev = uv
            row += 1
        retention.append(tg["Z"][row - 1] / z0)
        q = {k: p[k] for k in ("traj", "family", "inviscid", "n_frames", "split", "omega_rms", "ic_seed") if k in p}
        q.update({k: p[k] for k in ("k0", "count") if k in p})
        q.update({"rate_r": r, "nu": 0.0, "Re": p["Re"], "physical_inviscid": p["inviscid"]})
        new_plans.append(q)
    vel.flush()
    nu_dummy = np.zeros(total)
    np.savez(out_dir / "frames.npz", **idx, nu=nu_dummy, **tg)
    with open(out_dir / "trajectories.json", "w", encoding="utf-8") as f:
        json.dump(new_plans, f, indent=1)
    wall = time.perf_counter() - t0
    manifest = {"config": cfg, "environment": environment(), "n_frames_total": int(total),
                "input_scale": src_manifest["input_scale"], "input_scale_source": "physical dataset (train velocity std)",
                "enstrophy_retention_last_frame": {"median": float(np.median(retention)), "p10": float(np.percentile(retention, 10))},
                "frame_change_median": float(np.median(change)), "physical_frame_change_median_reference": 0.038,
                "sha256": {n: sha256(out_dir / n) for n in ("velocity.npy", "frames.npz", "trajectories.json")}, "wall_seconds": wall}
    save_json(manifest, out_dir / "manifest.json")
    append_log("Phase 2b non-physical dataset", cfg["_config_path"], command_line(), "ok",
               {"enstrophy retained at last frame (median / p10)": f"{manifest['enstrophy_retention_last_frame']['median']:.2f} / "
                                                                 f"{manifest['enstrophy_retention_last_frame']['p10']:.2f}",
                "median frame-to-frame change": f"{manifest['frame_change_median']:.3f} (physical data: 0.038)",
                "sha256 velocity.npy": manifest["sha256"]["velocity.npy"][:16] + "…"}, wall, cfg["device"])
    print(json.dumps({k: manifest[k] for k in ("enstrophy_retention_last_frame", "frame_change_median")}, indent=1))


if __name__ == "__main__":
    main()
