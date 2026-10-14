"""Phase 0 pilot calibration (no model training; sets data-design numbers for Phase 1).

1. Inviscid window. 64² Galerkin-truncated Euler conserves E and Z exactly but thermalizes.
   For windows of increasing length, measure: share of runs past thermalization onset; the
   within/between-trajectory variance ratio of non-conserved quantities (if it is tiny, every
   trajectory-level quantity looks invariant); the Phase 4 slow-direction analysis run on
   untrained encoders (does a short window hand energy/enstrophy to any encoder?); and the
   divergence of 64² inviscid runs from 128² runs of the same initial condition.
2. Resolution. Viscous runs at 64² vs 128² for a ladder of nu: errors in Z, P, max|omega|,
   and k_max * eta, to set nu_min.
3. Frame change. Relative change of the velocity field versus time lag (prediction difficulty).

    python -m solver.pilot --config configs/phase0_pilot.yaml
"""
from __future__ import annotations

import argparse
import math
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from common.benchmark import provisional_encoder  # noqa: E402
from common.run import (append_log, command_line, environment, load_config, pick_device,  # noqa: E402
                        save_json, seed_everything, set_threads)
from probes import slow_features as sfeat  # noqa: E402
from solver import diagnostics as dg  # noqa: E402
from solver.initial_conditions import gaussian_blobs, normalize_omega_rms, spectral_field  # noqa: E402
from solver.ns2d import NavierStokes2D, SpectralGrid, resample_hat  # noqa: E402
from solver.validate import GRID, INK2, SERIES, _log_ticks, _style  # noqa: E402


def sample_ics(grid: SpectralGrid, n: int, cic: dict, seed: int):
    """Draw initial conditions from the candidate Phase 1 distribution, normalized by rms vorticity."""
    rng = np.random.default_rng(seed)
    fields, meta = [], []
    lo, hi = cic["omega_rms"]
    for _ in range(n):
        om = math.exp(rng.uniform(math.log(lo), math.log(hi)))
        if rng.random() < cic["blob_fraction"]:
            count = int(rng.integers(cic["blob_count"][0], cic["blob_count"][1] + 1))
            centers = rng.uniform(0, 2 * math.pi, size=(count, 2))
            r0, r1 = cic["blob_radius"]
            radii = np.exp(rng.uniform(math.log(r0), math.log(r1), size=count))
            w = gaussian_blobs(grid, centers, radii, rng.choice([-1.0, 1.0], size=count))
            meta.append({"family": "blobs", "count": count, "omega_rms": om})
        else:
            k0 = rng.uniform(*cic["spectral_k0"])
            w = spectral_field(grid, rng, k0)
            meta.append({"family": "spectral", "k0": k0, "omega_rms": om})
        fields.append(normalize_omega_rms(grid, w, om))
    return torch.stack(fields), meta


def ow_vortex_fraction(grid: SpectralGrid, w_hat: torch.Tensor, thresh: float = 0.2) -> torch.Tensor:
    """Okubo–Weiss: Q = s_n^2 + s_s^2 - omega^2; vorticity-dominated where Q < -thresh * std(Q)."""
    u_hat, v_hat = grid.velocity_hat(w_hat)
    ux, uy = grid.ifft(1j * grid.kx * u_hat), grid.ifft(1j * grid.ky * u_hat)
    vx, vy = grid.ifft(1j * grid.kx * v_hat), grid.ifft(1j * grid.ky * v_hat)
    q = (ux - vy) ** 2 + (vx + uy) ** 2 - (vx - uy) ** 2
    return (q < -thresh * q.std(dim=(-2, -1), keepdim=True)).to(grid.dtype).mean(dim=(-2, -1))


def frame_diagnostics(grid: SpectralGrid, w_hat: torch.Tensor, k_tail: float) -> dict:
    e, z, p = dg.energy(grid, w_hat), dg.enstrophy(grid, w_hat), dg.palinstrophy(grid, w_hat)
    return {"E": e, "Z": z, "P": p, "kurt": dg.kurtosis(grid, w_hat), "maxw": dg.max_abs_vorticity(grid, w_hat),
            "ow": ow_vortex_fraction(grid, w_hat), "tail": dg.enstrophy_fraction_above(grid, w_hat, k_tail),
            "C4": dg.casimir(grid, w_hat, 4)}


def run_frames(solver, w0, n_frames, dt_out, cfl, grid, k_tail, keep_velocity=False):
    diag, vel = {}, []
    for _, wh in solver.simulate(w0, n_frames, dt_out, cfl):
        for k, v in frame_diagnostics(grid, wh, k_tail).items():
            diag.setdefault(k, []).append(v.cpu().numpy())
        if keep_velocity:
            u, v = grid.velocity(wh)
            vel.append(torch.stack([u, v], 1).float().cpu().numpy())
    diag = {k: np.stack(v, 1) for k, v in diag.items()}  # (traj, frames)
    return diag, (np.stack(vel, 1) if keep_velocity else None)  # (traj, frames, 2, n, n)


def encode(velocity: np.ndarray, seed: int, scale: float, device: str, batch: int = 1024) -> np.ndarray:
    torch.manual_seed(seed)
    net = provisional_encoder().to(device).eval()
    flat = velocity.reshape(-1, *velocity.shape[2:])
    out = []
    with torch.no_grad():
        for i in range(0, len(flat), batch):
            x = torch.from_numpy(flat[i:i + batch]).to(device) / scale
            out.append(net(x).mean(dim=(2, 3)).cpu().numpy())
    return np.concatenate(out).reshape(velocity.shape[0], velocity.shape[1], -1)


def inviscid_study(cfg: dict, device: str) -> dict:
    ci = cfg["inviscid"]
    g = SpectralGrid(64, device)
    w0, meta = sample_ics(g, ci["n_traj"], cfg["ic"], ci["seed"])
    n_frames = int(round(ci["t_max"] / ci["dt_fine"])) + 1
    diag, vel = run_frames(NavierStokes2D(g, 0.0), w0, n_frames, ci["dt_fine"], cfg["cfl"], g, ci["k_tail"], True)
    t = np.arange(n_frames) * ci["dt_fine"]
    fam = np.array([m["family"] for m in meta])
    tau = 1.0 / np.array([m["omega_rms"] for m in meta])

    # onset of thermalization
    onset = {}
    for thr in ci["tail_thresholds"]:
        over = diag["tail"] > thr
        first = np.where(over.any(1), over.argmax(1), n_frames)
        t_on = np.where(first < n_frames, t[np.minimum(first, n_frames - 1)], np.inf)
        onset[str(thr)] = {"t": t_on.tolist(), "t_over_tau": (t_on / tau).tolist()}

    # 64^2 vs 128^2 divergence on a subset
    m = ci["n_compare_128"]
    g128 = SpectralGrid(128, device)
    s64 = NavierStokes2D(g, 0.0)
    s128 = NavierStokes2D(g128, 0.0)
    field_err, subgrid = [], []
    frames64 = s64.simulate(w0[:m], n_frames, ci["dt_fine"], cfg["cfl"])
    frames128 = s128.simulate(resample_hat(w0[:m], 64, 128), n_frames, ci["dt_fine"], cfg["cfl"])
    for (_, a), (_, b) in zip(frames64, frames128):
        b64 = resample_hat(b, 128, 64) * g.mask
        field_err.append((g.mean_square(a - b64) / g.mean_square(b64)).sqrt().cpu().numpy())
        subgrid.append(dg.enstrophy_fraction_above(g128, b, 21.0).cpu().numpy())
    field_err, subgrid = np.stack(field_err, 1), np.stack(subgrid, 1)

    # window analysis
    rng = np.random.default_rng(ci["seed"] + 1)
    order = rng.permutation(ci["n_traj"])
    n_fit = int(ci["fit_fraction"] * ci["n_traj"])
    fit_idx, ev_idx = np.sort(order[:n_fit]), np.sort(order[n_fit:])
    scale = float(vel[fit_idx].std())
    latents = [encode(vel, s, scale, device) for s in ci["encoder_seeds"]]
    quantities = {"log E": np.log(diag["E"]), "log Z": np.log(diag["Z"]), "log P": np.log(diag["P"]),
                  "log P/Z": np.log(diag["P"] / diag["Z"]), "kurtosis": diag["kurt"],
                  "log max|w|": np.log(diag["maxw"]), "OW fraction": diag["ow"]}
    windows = []
    for tw in ci["windows"]:
        nf = int(round(tw / ci["dt_fine"])) + 1
        traj_ids = np.repeat(np.arange(ci["n_traj"]), nf)
        row = {"T_w": tw, "frames_at_dt_fine": nf,
               "thermalized_share": {thr: float((diag["tail"][:, :nf] > float(thr)).any(1).mean()) for thr in map(str, ci["tail_thresholds"])},
               "within_between": {q: float(sfeat.within_between_ratio(v[:, :nf].reshape(-1), traj_ids)[0]) for q, v in quantities.items()}}
        targets_ev = {"log E (invariant)": np.log(diag["E"][:, 0]), "log Z (invariant)": np.log(diag["Z"][:, 0]),
                      "log P/Z (window mean)": quantities["log P/Z"][:, :nf].mean(1),
                      "kurtosis (window mean)": diag["kurt"][:, :nf].mean(1),
                      "OW fraction (window mean)": diag["ow"][:, :nf].mean(1)}
        per_seed = []
        for lat in latents:
            sub = lat[:, :nf]
            model = sfeat.fit(sub[fit_idx].reshape(-1, sub.shape[-1]), np.repeat(fit_idx, nf), ci["n_pca"])
            per_seed.append(sfeat.score(model, sub[ev_idx].reshape(-1, sub.shape[-1]), np.repeat(ev_idx, nf),
                                        {k: dict(enumerate(v)) for k, v in targets_ev.items()}))
        row["slow_max_abs_r"] = {k: float(np.mean([s["max_abs_r"][k] for s in per_seed])) for k in targets_ev}
        row["slow_lambda_heldout_top3"] = np.mean([s["lambda_heldout"] for s in per_seed], 0).tolist()
        windows.append(row)

    # frame-to-frame change vs lag
    lags = []
    norm = np.sqrt((vel**2).mean(axis=(2, 3, 4)))
    for lag in ci["lag_frames"]:
        d = np.sqrt(((vel[:, lag:] - vel[:, :-lag]) ** 2).mean(axis=(2, 3, 4))) / norm[:, :-lag]
        lags.append({"lag_time": lag * ci["dt_fine"], "median_rel_change": float(np.median(d)),
                     "p10": float(np.percentile(d, 10)), "p90": float(np.percentile(d, 90))})

    return {"t": t.tolist(), "family": fam.tolist(), "omega_rms": [mm["omega_rms"] for mm in meta],
            "tail_median_by_family": {f: np.median(diag["tail"][fam == f], 0).tolist() for f in ("blobs", "spectral")},
            "tail_p90_by_family": {f: np.percentile(diag["tail"][fam == f], 90, axis=0).tolist() for f in ("blobs", "spectral")},
            "drift_E_max": float(np.abs(diag["E"] / diag["E"][:, :1] - 1).max()),
            "drift_Z_max": float(np.abs(diag["Z"] / diag["Z"][:, :1] - 1).max()),
            "onset": onset, "field_err_64_vs_128": {"median": np.median(field_err, 0).tolist(), "p90": np.percentile(field_err, 90, 0).tolist()},
            "subgrid_enstrophy_128": {"median": np.median(subgrid, 0).tolist(), "p90": np.percentile(subgrid, 90, 0).tolist()},
            "windows": windows, "lags": lags,
            "between_traj_spread": {"E_ratio_max_min": float(diag["E"][:, 0].max() / diag["E"][:, 0].min()),
                                    "Z_ratio_max_min": float(diag["Z"][:, 0].max() / diag["Z"][:, 0].min()),
                                    "corr_logE_logZ": float(np.corrcoef(np.log(diag["E"][:, 0]), np.log(diag["Z"][:, 0]))[0, 1])}}


def resolution_study(cfg: dict, device: str) -> dict:
    cr = cfg["resolution"]
    g64, g128 = SpectralGrid(64, device), SpectralGrid(128, device)
    w0, meta = sample_ics(g64, cr["n_traj"], cfg["ic"], cr["seed"])
    n_frames = int(round(cr["t_end"] / cr["dt_out"])) + 1
    rows = []
    for nu in cr["nus"]:
        a = {k: [] for k in ("Z", "P", "maxw", "keta")}
        b = {k: [] for k in ("Z", "P", "maxw")}
        it64 = NavierStokes2D(g64, nu).simulate(w0, n_frames, cr["dt_out"], cfg["cfl"])
        it128 = NavierStokes2D(g128, nu).simulate(resample_hat(w0, 64, 128), n_frames, cr["dt_out"], cfg["cfl"])
        for (_, x), (_, y) in zip(it64, it128):
            a["Z"].append(dg.enstrophy(g64, x)); a["P"].append(dg.palinstrophy(g64, x))
            a["maxw"].append(dg.max_abs_vorticity(g64, x)); a["keta"].append(dg.kmax_eta(g64, x, nu))
            b["Z"].append(dg.enstrophy(g128, y)); b["P"].append(dg.palinstrophy(g128, y))
            b["maxw"].append(dg.max_abs_vorticity(g128, y))
        st = {k: torch.stack(v, 1).cpu().numpy() for k, v in a.items()}
        sb = {k: torch.stack(v, 1).cpu().numpy() for k, v in b.items()}
        row = {"nu": nu, "min_kmax_eta_p05": float(np.percentile(st["keta"].min(1), 5))}
        for q in ("Z", "P", "maxw"):
            err = (np.abs(st[q] - sb[q]) / sb[q]).max(1)
            row[f"{q}_err_p50"], row[f"{q}_err_p95"], row[f"{q}_err_max"] = (float(np.percentile(err, 50)),
                                                                                float(np.percentile(err, 95)), float(err.max()))
        rows.append(row)
        print("resolution", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
    return {"rows": rows, "t_end": cr["t_end"]}


def figure(inv: dict, res: dict, path: str, k_tail: float) -> None:
    _style()
    fig, ax = plt.subplots(2, 3, figsize=(13.5, 7.6))
    fig.suptitle("Phase 0 pilot: inviscid window, resolution limit, frame change (64², untrained encoders)",
                 fontsize=12, fontweight="bold")
    t = np.array(inv["t"])

    a = ax[0, 0]
    for i, fam in enumerate(("spectral", "blobs")):
        a.plot(t, inv["tail_median_by_family"][fam], color=SERIES[i], label=f"{fam}: median")
        a.plot(t, inv["tail_p90_by_family"][fam], color=SERIES[i], linestyle=(0, (4, 3)), linewidth=1.1, label=f"{fam}: 90th pct")
    a.axhline(0.05, color=INK2, linewidth=0.8, linestyle=(0, (2, 2)))
    a.set_title(f"(a) Inviscid 64²: enstrophy share at |k| > {k_tail:g}")
    a.set_xlabel("t")
    a.set_ylabel("enstrophy fraction in top shells")
    a.legend(loc="upper left", fontsize=8)

    b = ax[0, 1]
    tw = [w["T_w"] for w in inv["windows"]]
    names = ["log P/Z", "kurtosis", "OW fraction", "log max|w|", "log Z", "log E"]
    for i, q in enumerate(names):
        vals = np.maximum([w["within_between"][q] for w in inv["windows"]], 1e-16)
        style = (0, (4, 3)) if q in ("log Z", "log E") else "-"
        b.semilogy(tw, vals, color=SERIES[i % len(SERIES)] if i < 5 else INK2, linestyle=style, marker="o", markersize=4,
                   label=q + (" (invariant)" if q in ("log Z", "log E") else ""))
    b.set_title("(b) Within / between-trajectory variance")
    b.set_xlabel("inviscid window length T_w")
    b.set_ylabel("ratio (oracle quantities)")
    b.set_ylim(1e-17, 1e9)  # headroom for the legend
    b.legend(loc="upper left", fontsize=7.5, ncol=2)

    c = ax[0, 2]
    keys = list(inv["windows"][0]["slow_max_abs_r"].keys())
    for i, k in enumerate(keys):
        c.plot(tw, [w["slow_max_abs_r"][k] for w in inv["windows"]], color=SERIES[i], marker="o", markersize=4, label=k)
    c.axhline(0.8, color=INK2, linewidth=0.8, linestyle=(0, (2, 2)))
    c.set_ylim(0, 1.45)
    c.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    c.set_title("(c) Slow directions of UNTRAINED encoders")
    c.set_xlabel("inviscid window length T_w")
    c.set_ylabel("held-out max |r| over top-3 (3 seeds)")
    c.legend(loc="upper center", fontsize=7.5, ncol=2)

    d = ax[1, 0]
    d.plot(t, inv["field_err_64_vs_128"]["median"], color=SERIES[0], label="field error 64² vs 128²: median")
    d.plot(t, inv["field_err_64_vs_128"]["p90"], color=SERIES[0], linestyle=(0, (4, 3)), linewidth=1.1, label="field error: 90th pct")
    d.plot(t, inv["subgrid_enstrophy_128"]["median"], color=SERIES[1], label="128² enstrophy beyond k = 21: median")
    d.set_title("(d) Inviscid: 64² truncated vs 128² runs")
    d.set_xlabel("t")
    d.set_ylabel("relative L2 error / fraction")
    d.legend(loc="upper left", fontsize=8)

    e = ax[1, 1]
    nus = np.array([r["nu"] for r in res["rows"]])
    for i, (q, lab) in enumerate((("Z", "enstrophy"), ("P", "palinstrophy"), ("maxw", "max |ω|"))):
        e.loglog(nus, np.maximum([r[f"{q}_err_p95"] for r in res["rows"]], 1e-6), color=SERIES[i], marker="o", markersize=5, label=f"{lab}, 95th pct")
    e.axhline(0.05, color=INK2, linewidth=0.8, linestyle=(0, (2, 2)))
    _log_ticks(e.xaxis, [v for v in nus if v not in (0.0015, 0.003)])
    e.set_title("(e) Viscous: 64² vs 128² error (max over window)")
    e.set_xlabel("ν")
    e.set_ylabel("relative error")
    e.legend(loc="lower left", fontsize=8)

    f = ax[1, 2]
    lt = [r["lag_time"] for r in inv["lags"]]
    f.plot(lt, [r["median_rel_change"] for r in inv["lags"]], color=SERIES[0], marker="o", markersize=5, label="median")
    f.fill_between(lt, [r["p10"] for r in inv["lags"]], [r["p90"] for r in inv["lags"]], color=SERIES[0], alpha=0.15, linewidth=0, label="10–90th pct")
    f.set_title("(f) Velocity change between frames")
    f.set_xlabel("time lag")
    f.set_ylabel("‖u(t+lag) − u(t)‖ / ‖u(t)‖")
    f.legend(loc="upper left", fontsize=8)
    for axis in ax.flat:
        axis.grid(True, color=GRID)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--plot-only", action="store_true", help="redraw the figure from the saved JSON")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.plot_only:
        import json
        with open(cfg["output_json"], encoding="utf-8") as f:
            saved = json.load(f)
        figure(saved["inviscid"], saved["resolution"], cfg["output_figure"], cfg["inviscid"]["k_tail"])
        return
    seed_everything(cfg["seed"])
    set_threads(cfg["cpu_threads"])
    device = pick_device(cfg["device"])
    t0 = time.perf_counter()
    inv = inviscid_study(cfg, device)
    print("inviscid done", round(time.perf_counter() - t0, 1), "s")
    res = resolution_study(cfg, device)
    wall = time.perf_counter() - t0
    save_json({"environment": environment(), "config": cfg, "inviscid": inv, "resolution": res}, cfg["output_json"])
    figure(inv, res, cfg["output_figure"], cfg["inviscid"]["k_tail"])
    for w in inv["windows"]:
        print(f"T_w={w['T_w']}: thermalized {w['thermalized_share']}, "
              f"within/between {{{', '.join(f'{k}: {v:.3g}' for k, v in w['within_between'].items())}}}, "
              f"slow |r| {{{', '.join(f'{k}: {v:.2f}' for k, v in w['slow_max_abs_r'].items())}}}")
    append_log("Phase 0 pilot calibration", cfg["_config_path"], command_line(), "ok",
               {"windows (T_w: share >5% tail; untrained slow |r| logE/logZ)":
                "; ".join(f"{w['T_w']}: {w['thermalized_share']['0.05']:.2f}, {w['slow_max_abs_r']['log E (invariant)']:.2f}/{w['slow_max_abs_r']['log Z (invariant)']:.2f}"
                          for w in inv["windows"]),
                "resolution (nu: P err p95)": "; ".join(f"{r['nu']:g}: {r['P_err_p95']:.3f}" for r in res["rows"]),
                "results": cfg["output_json"], "figure": cfg["output_figure"]}, wall, device)


if __name__ == "__main__":
    main()
