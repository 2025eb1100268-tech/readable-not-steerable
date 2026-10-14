"""Phase 0: run every solver validation at full size, check stated tolerances, write the numbers
(results/phase0/solver_validation.json), the figure (results/solver_validation.png) and a
results/LOG.md entry. Exits non-zero if any check fails.

    python -m solver.validate --config configs/phase0_solver_validation.yaml
"""
from __future__ import annotations

import argparse
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402

from common.run import (append_log, command_line, environment, load_config, pick_device,  # noqa: E402
                        save_json, seed_everything, set_threads)
from solver import checks  # noqa: E402

# Reference categorical palette (fixed order), light surface, text inks.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def _style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.6,
        "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "legend.frameon": False,
    })


def _log_ticks(axis, values):
    """Label a log axis only at the given values (avoids colliding minor-tick labels)."""
    axis.set_major_locator(matplotlib.ticker.FixedLocator(list(values)))
    axis.set_major_formatter(matplotlib.ticker.FixedFormatter([f"{v:g}" for v in values]))
    axis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def run_all(cfg: dict, device: str) -> dict:
    out = {}
    out["analytic_rhs"] = checks.analytic_rhs(device=device, **cfg["analytic_rhs"])
    out["taylor_green"] = checks.taylor_green(device=device, **cfg["taylor_green"])
    out["lamb_oseen"] = checks.lamb_oseen(device=device, **cfg["lamb_oseen"])
    out["inviscid"] = checks.inviscid_conservation(device=device, **cfg["inviscid"])
    out["viscous_budget"] = checks.energy_budget(device=device, **cfg["viscous_budget"])
    out["forced_budget"] = checks.energy_budget(device=device, **cfg["forced_budget"])
    out["convergence"] = checks.convergence(device=device, **{**cfg["convergence"], "dts": tuple(cfg["convergence"]["dts"])})
    out["vortex_pair"] = checks.vortex_pair(device=device, **cfg["vortex_pair"])
    out["stored_frames"] = checks.stored_frames(device=device, **{**cfg["stored_frames"], "nus": tuple(cfg["stored_frames"]["nus"])})
    return out


def evaluate(r: dict, tol: dict) -> dict:
    sf = r["stored_frames"]
    visc = [(nu, fit) for nu, fit in zip(sf["nu_true"], sf["nu_fit"]) if nu > 0]
    inv = [fit for nu, fit in zip(sf["nu_true"], sf["nu_fit"]) if nu == 0]
    lo, hi = tol["vortex_pair_ratio"]
    return {
        "inviscid energy conserved": r["inviscid"]["max_rel_drift_E"] < tol["inviscid_energy_drift"],
        "inviscid enstrophy conserved": r["inviscid"]["max_rel_drift_Z"] < tol["inviscid_enstrophy_drift"],
        "dE/dt = -2 nu Z (pointwise)": r["viscous_budget"]["max_rel_err_pointwise"] < tol["budget_pointwise"],
        "dE/dt = -2 nu Z (integrated)": r["viscous_budget"]["max_rel_err_integrated"] < tol["budget_integrated"],
        "forced budget with drag": r["forced_budget"]["max_rel_err_pointwise"] < tol["budget_pointwise"],
        "Taylor-Green exact": r["taylor_green"]["max_rel_err"] < tol["taylor_green"],
        "Lamb-Oseen vs heat kernel": r["lamb_oseen"]["max_rel_err"] < tol["lamb_oseen"],
        "4th-order convergence": all(tol["order_min"] < p < tol["order_max"] for p in r["convergence"]["orders"]),
        "analytic nonlinear term": r["analytic_rhs"]["max_abs_err"] < tol["analytic_rhs_abs"],
        "vortex pair rotates CCW at point-vortex rate": r["vortex_pair"]["final_angle"] > 0 and lo < r["vortex_pair"]["ratio"] < hi,
        "nu recovered from stored frames": all(abs(f - n) / n < tol["nu_fit_rel"] for n, f in visc)
        and all(abs(f) < tol["nu_fit_abs_inviscid"] for f in inv),
        "stored-frame NS residual": max(sf["residual_rel"]) < tol["stored_residual"],
        "stored velocity divergence-free": sf["div_rel_stored"] < tol["stored_divergence"],
    }


def figure(r: dict, path: str) -> None:
    _style()
    fig, ax = plt.subplots(2, 3, figsize=(13.5, 7.6))
    fig.suptitle("Phase 0 solver validation (64², 2/3 dealiasing, IF-RK4, float64)", fontsize=12, fontweight="bold")

    # (a) inviscid invariants
    a = ax[0, 0]
    inv = r["inviscid"]
    t = np.array(inv["t"])
    for key, col, lab in (("drift_E", SERIES[0], "energy E"), ("drift_Z", SERIES[1], "enstrophy Z"),
                          ("drift_C4", SERIES[2], r"$\int\omega^4$ (not conserved by truncation)")):
        d = np.maximum(np.array(inv[key]), 1e-17)
        a.semilogy(t[1:], d[1:, 0], color=col, label=lab)
        a.semilogy(t[1:], d[1:, 1:], color=col, linewidth=0.8, alpha=0.45)
    a.set_title("(a) Inviscid: relative drift of invariants")
    a.set_xlabel("t")
    a.set_ylabel(r"$|Q(t)/Q(0) - 1|$ (4 initial conditions)")
    a.set_ylim(1e-12, 1e4)  # headroom for the legend
    a.legend(loc="upper left", fontsize=8)

    # (b) viscous energy budget
    b = ax[0, 1]
    vb = r["viscous_budget"]
    b.plot(vb["t"], -np.array(vb["dEdt"]), color=SERIES[0], label="−dE/dt (simulated)")
    b.plot(vb["t"], -np.array(vb["budget"]), color=SERIES[1], linestyle=(0, (4, 3)), label="2νZ")
    b.set_title("(b) Viscous decay: dE/dt = −2νZ (ν = 0.01)")
    b.set_xlabel("t")
    b.set_ylabel("energy dissipation rate")
    b.legend(loc="upper right", fontsize=8)
    b.text(0.03, 0.05, f"max pointwise rel. error {vb['max_rel_err_pointwise']:.1e}", transform=b.transAxes, color=INK2)

    # (c) exact / analytic solutions
    c = ax[0, 2]
    c.semilogy(r["taylor_green"]["t"][1:], np.maximum(r["taylor_green"]["err"][1:], 1e-17), color=SERIES[0],
               marker="o", markersize=3, label="Taylor–Green vs exp(−2νt)")
    c.semilogy(r["lamb_oseen"]["t"][1:], r["lamb_oseen"]["err"][1:], color=SERIES[1],
               marker="o", markersize=3, label="Lamb–Oseen vs periodized heat kernel")
    c.set_title("(c) Analytic vortex solutions")
    c.set_xlabel("t")
    c.set_ylabel("max relative error")
    c.legend(loc="center right", fontsize=8)

    # (d) convergence order
    d = ax[1, 0]
    dts, errs = np.array(r["convergence"]["dts"]), np.array(r["convergence"]["errors"])
    d.loglog(dts, errs, color=SERIES[0], marker="o", markersize=5, label="IF-RK4 error at t = 1")
    d.loglog(dts, errs[0] * (dts / dts[0]) ** 4, color=INK2, linestyle=(0, (2, 2)), linewidth=1.0, label="slope 4")
    _log_ticks(d.xaxis, dts)
    d.set_title("(d) Time-step convergence")
    d.set_xlabel("Δt")
    d.set_ylabel("max relative error vs Δt/8 reference")
    d.legend(loc="upper left", fontsize=8)
    d.text(0.97, 0.05, "observed orders " + ", ".join(f"{p:.2f}" for p in r["convergence"]["orders"]),
           transform=d.transAxes, ha="right", color=INK2)

    # (e) vortex pair rotation
    e = ax[1, 1]
    vp = r["vortex_pair"]
    e.plot(vp["t"], vp["angle"], color=SERIES[0], label="measured axis angle")
    e.plot(vp["t"], vp["predicted_angle"], color=SERIES[1], linestyle=(0, (4, 3)), label="point vortices Γt/(πd²)")
    e.set_title("(e) Co-rotating vortex pair (sign of advection)")
    e.set_xlabel("t")
    e.set_ylabel("rotation angle [rad], CCW > 0")
    e.legend(loc="upper left", fontsize=8)
    e.text(0.97, 0.05, f"ratio measured/predicted {vp['ratio']:.3f}", transform=e.transAxes, ha="right", color=INK2)

    # (f) nu recovered from stored float32 frames
    f = ax[1, 2]
    sf = r["stored_frames"]
    nus = np.array(sf["nu_true"])
    fits = np.array(sf["nu_fit"])
    pos = nus > 0
    f.loglog(nus[pos], nus[pos], color=INK2, linestyle=(0, (2, 2)), linewidth=1.0, label="ν recovered = ν true")
    f.loglog(nus[pos], fits[pos], color=SERIES[0], linestyle="none", marker="o", markersize=7, label="recovered from stored frames")
    _log_ticks(f.xaxis, nus[pos])
    _log_ticks(f.yaxis, nus[pos])
    f.set_title("(f) ν re-estimated from stored float32 (u, v)")
    f.set_xlabel("true ν")
    f.set_ylabel("ν from least-squares fit of the vorticity equation")
    f.legend(loc="upper left", fontsize=8)
    f.text(0.97, 0.05, f"ν = 0 run: fit {fits[~pos][0]:.1e}\nmax residual {max(sf['residual_rel']):.1e} of |u·∇ω|\n"
                       f"divergence {sf['div_rel_stored']:.1e} of |ω|", transform=f.transAxes, ha="right", color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = load_config(ap.parse_args().config)
    seed_everything(cfg["seed"])
    set_threads(cfg["cpu_threads"])
    device = pick_device(cfg["device"])
    t0 = time.perf_counter()
    results = run_all(cfg, device)
    verdict = evaluate(results, cfg["tolerances"])
    wall = time.perf_counter() - t0
    save_json({"environment": environment(), "config": cfg, "verdict": verdict, "results": results}, cfg["output_json"])
    figure(results, cfg["output_figure"])
    failed = [k for k, ok in verdict.items() if not ok]
    for k, ok in verdict.items():
        print(("PASS " if ok else "FAIL ") + k)
    inv, vb = results["inviscid"], results["viscous_budget"]
    append_log("Phase 0 solver validation", cfg["_config_path"], command_line(),
               "all checks pass" if not failed else "FAILED: " + "; ".join(failed),
               {"inviscid drift E / Z / int w^4 (T=8)": f"{inv['max_rel_drift_E']:.1e} / {inv['max_rel_drift_Z']:.1e} / {inv['max_rel_drift_C4']:.2f}",
                "dE/dt vs -2 nu Z": f"{vb['max_rel_err_pointwise']:.1e}",
                "convergence orders": ", ".join(f"{p:.2f}" for p in results["convergence"]["orders"]),
                "figure": cfg["output_figure"]}, wall, device)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
