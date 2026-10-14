"""Phase 1: generate trajectories, probe targets, splits and checksums.

    python -m data.generate --config configs/phase1_data.yaml

Writes to cfg["out_dir"]:
  velocity.npy      float32 (frames, 2, 64, 64), (u, v) — the only thing models ever see
  frames.npz        per-frame index (traj, frame, time) and every probe target
  trajectories.json per-trajectory parameters (family, IC params, omega_rms, nu, Re, ...) and split
  manifest.json     config, environment, input scale (train velocity std), SHA-256 checksums
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from common.run import (REPO, append_log, command_line, environment, load_config, pick_device,
                        save_json, seed_everything, set_threads)
from data.targets import frame_targets, integral_scale
from solver import diagnostics as dg
from solver.initial_conditions import gaussian_blobs, normalize_omega_rms, spectral_field
from solver.ns2d import NavierStokes2D, SpectralGrid


def plan_trajectories(cfg: dict) -> list[dict]:
    """Decide every trajectory's parameters up front (exact class balance, one RNG stream each)."""
    master = np.random.default_rng(cfg["seed"])
    n = cfg["n_traj"]
    n_inv = int(round(cfg["inviscid_fraction"] * n))
    inviscid = np.zeros(n, bool)
    inviscid[master.permutation(n)[:n_inv]] = True
    family = np.empty(n, object)
    for flag in (True, False):  # balance families within inviscid and within viscous
        idx = np.flatnonzero(inviscid == flag)
        idx = idx[master.permutation(len(idx))]
        n_blob = int(round(cfg["ic"]["blob_fraction"] * len(idx)))
        family[idx[:n_blob]] = "blobs"
        family[idx[n_blob:]] = "spectral"
    child_seeds = np.random.SeedSequence(cfg["seed"]).spawn(n)
    lo, hi = cfg["nu_range"]
    plans = []
    for i in range(n):
        rng = np.random.default_rng(child_seeds[i])
        p = {"traj": i, "family": family[i], "inviscid": bool(inviscid[i]),
             "omega_rms": float(math.exp(rng.uniform(*np.log(cfg["ic"]["omega_rms"])))),
             "nu": 0.0 if inviscid[i] else float(math.exp(rng.uniform(math.log(lo), math.log(hi)))),
             "ic_seed": int(rng.integers(0, 2**63 - 1))}
        if family[i] == "spectral":
            p["k0"] = float(rng.uniform(*cfg["ic"]["spectral_k0"]))
        else:
            p["count"] = int(rng.integers(cfg["ic"]["blob_count"][0], cfg["ic"]["blob_count"][1] + 1))
        plans.append(p)
    return plans


def build_ic(grid: SpectralGrid, p: dict, cic: dict) -> torch.Tensor:
    rng = np.random.default_rng(p["ic_seed"])
    if p["family"] == "spectral":
        w = spectral_field(grid, rng, p["k0"])
    else:
        centers = rng.uniform(0, 2 * math.pi, size=(p["count"], 2))
        r0, r1 = cic["blob_radius"]
        radii = np.exp(rng.uniform(math.log(r0), math.log(r1), size=p["count"]))
        w = gaussian_blobs(grid, centers, radii, rng.choice([-1.0, 1.0], size=p["count"]))
    return normalize_omega_rms(grid, w, p["omega_rms"])


def assign_splits(plans: list[dict], cs: dict) -> None:
    """OOD = viscous trajectories in an interior log-Re quantile band. The rest: train/val/test-ID by
    trajectory, stratified by family x inviscid x (Re tercile among viscous)."""
    rng = np.random.default_rng(cs["seed"])
    visc = [p for p in plans if not p["inviscid"]]
    log_re = np.array([math.log(p["Re"]) for p in visc])
    q_lo, q_hi = np.quantile(log_re, cs["ood_quantiles"])
    for p in plans:
        p["split"] = None
        if not p["inviscid"] and q_lo <= math.log(p["Re"]) <= q_hi:
            p["split"] = "test_ood"
    id_visc = np.array([math.log(p["Re"]) for p in visc if p["split"] is None])
    t1, t2 = np.quantile(id_visc, [1 / 3, 2 / 3])
    strata: dict = {}
    for p in plans:
        if p["split"] is None:
            if p["inviscid"]:
                band = "inviscid"
            else:
                lr = math.log(p["Re"])
                band = "re_low" if lr < t1 else ("re_mid" if lr < t2 else "re_high")
            strata.setdefault((p["family"], band), []).append(p)
    fr = cs["fractions"]
    for key in sorted(strata):
        group = strata[key]
        order = rng.permutation(len(group))
        n_tr = int(round(fr["train"] * len(group)))
        n_va = int(round(fr["val"] * len(group)))
        for rank, j in enumerate(order):
            group[j]["split"] = "train" if rank < n_tr else ("val" if rank < n_tr + n_va else "test_id")
    meta = {"ood_log_re_band": [float(q_lo), float(q_hi)], "ood_re_band": [math.exp(q_lo), math.exp(q_hi)],
            "id_viscous_re_terciles": [math.exp(t1), math.exp(t2)]}
    return meta


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = load_config(ap.parse_args().config)
    seed_everything(cfg["seed"])
    set_threads(cfg["cpu_threads"])
    device = pick_device(cfg["device"])
    out_dir = REPO / cfg["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    g = SpectralGrid(cfg["n"], device)
    plans = plan_trajectories(cfg)
    for p in plans:
        p["n_frames"] = cfg["frames_inviscid"] if p["inviscid"] else cfg["frames_viscous"]
    total = sum(p["n_frames"] for p in plans)
    offsets = np.cumsum([0] + [p["n_frames"] for p in plans])
    vel = np.lib.format.open_memmap(out_dir / "velocity.npy", mode="w+", dtype=np.float32,
                                    shape=(total, 2, cfg["n"], cfg["n"]))
    index = {"traj": np.zeros(total, np.int32), "frame": np.zeros(total, np.int32), "time": np.zeros(total)}
    targets: dict[str, np.ndarray] = {}

    # batches never mix frame counts
    for flag in (True, False):
        ids = [p["traj"] for p in plans if p["inviscid"] == flag]
        for b0 in range(0, len(ids), cfg["batch"]):
            batch = ids[b0:b0 + cfg["batch"]]
            w0 = torch.stack([build_ic(g, plans[i], cfg["ic"]) for i in batch])
            nu = torch.tensor([plans[i]["nu"] for i in batch], dtype=g.dtype, device=g.device)
            e0, z0 = dg.energy(g, w0), dg.enstrophy(g, w0)
            ell = integral_scale(g, w0)
            for j, i in enumerate(batch):
                p = plans[i]
                p.update({"E0": float(e0[j]), "Z0": float(z0[j]), "U0": math.sqrt(2 * float(e0[j])), "L0": float(ell[j])})
                p["Re"] = p["U0"] * p["L0"] / p["nu"] if p["nu"] > 0 else math.inf
            solver = NavierStokes2D(g, nu)
            nf = plans[batch[0]]["n_frames"]
            for f, (t, wh) in enumerate(solver.simulate(w0, nf, cfg["dt_out"], cfg["cfl"])):
                u, v = g.velocity(wh)
                uv = torch.stack([u, v], 1).float().cpu().numpy()
                tg = frame_targets(g, wh, nu)
                rows = offsets[batch] + f
                vel[rows] = uv
                index["traj"][rows] = batch
                index["frame"][rows] = f
                index["time"][rows] = t
                for k, val in tg.items():
                    if k not in targets:
                        targets[k] = np.zeros(total, val.dtype)
                    targets[k][rows] = val
            print(f"{'inviscid' if flag else 'viscous'} batch {b0 // cfg['batch']}: {len(batch)} traj, "
                  f"{time.perf_counter() - t_start:.0f} s")
    vel.flush()

    split_meta = assign_splits(plans, cfg["split"])
    split_of = np.array([p["split"] for p in plans])
    train_rows = np.flatnonzero(split_of[index["traj"]] == "train")
    sample = np.sort(np.random.default_rng(0).choice(train_rows, size=min(20000, len(train_rows)), replace=False))
    input_scale = float(np.asarray(vel[sample], dtype=np.float64).std())

    np.savez(out_dir / "frames.npz", **index, nu=np.array([p["nu"] for p in plans])[index["traj"]], **targets)
    with open(out_dir / "trajectories.json", "w", encoding="utf-8") as f:
        json.dump(plans, f, indent=1)
    wall = time.perf_counter() - t_start
    counts = {s: int((split_of == s).sum()) for s in ("train", "val", "test_id", "test_ood")}
    inv_rows = np.flatnonzero(np.array([p["inviscid"] for p in plans])[index["traj"]])
    def drift(name):
        x = targets[name][inv_rows].reshape(-1, cfg["frames_inviscid"])
        return float(np.abs(x / x[:, :1] - 1).max())
    manifest = {"config": cfg, "environment": environment(), "n_frames_total": int(total),
                "input_scale": input_scale, "split_counts": counts, **split_meta,
                "inviscid_max_drift": {"E": drift("E"), "Z": drift("Z"), "C4": drift("C4")},
                "inviscid_tail16_last_frame": {"median": float(np.median(targets["tail16"][inv_rows].reshape(-1, cfg["frames_inviscid"])[:, -1])),
                                               "share_above_0.05": float((targets["tail16"][inv_rows].reshape(-1, cfg["frames_inviscid"])[:, -1] > 0.05).mean())},
                "sha256": {name: sha256(out_dir / name) for name in ("velocity.npy", "frames.npz", "trajectories.json")},
                "wall_seconds": wall}
    save_json(manifest, out_dir / "manifest.json")
    append_log("Phase 1 data generation", cfg["_config_path"], command_line(), "ok",
               {"frames": total, "split counts (trajectories)": counts,
                "OOD Re band": f"{split_meta['ood_re_band'][0]:.0f}–{split_meta['ood_re_band'][1]:.0f}",
                "inviscid max drift E / Z / C4": f"{manifest['inviscid_max_drift']['E']:.1e} / {manifest['inviscid_max_drift']['Z']:.1e} / {manifest['inviscid_max_drift']['C4']:.3f}",
                "input scale (train velocity std)": f"{input_scale:.4f}",
                "sha256 velocity.npy": manifest["sha256"]["velocity.npy"][:16] + "…"}, wall, device)
    print(json.dumps({k: manifest[k] for k in ("split_counts", "ood_re_band", "inviscid_max_drift", "inviscid_tail16_last_frame", "input_scale")}, indent=1))


if __name__ == "__main__":
    main()
