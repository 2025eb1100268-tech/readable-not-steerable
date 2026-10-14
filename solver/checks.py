"""Solver validation experiments.

Each function runs one experiment and returns a dict: scalar metrics plus time series for the
figure. tests/test_solver.py runs them at small sizes and asserts tolerances; solver/validate.py
runs them at full size and draws results/solver_validation.png.

``stored_frames`` re-derives the vorticity equation from saved float32 velocity frames with an
independent NumPy implementation, so a sign or factor error in the solver's nonlinear term
cannot cancel out (the brief's other tests are all blind to the sign of advection).
"""
from __future__ import annotations

import math

import numpy as np
import torch

from . import diagnostics as dg
from .initial_conditions import gaussian_blobs, normalize_energy, spectral_field
from .ns2d import NavierStokes2D, SpectralGrid, nonlinear_hat


def _max_rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max() / b.abs().max())


def analytic_rhs(n: int = 32, device: str = "cpu") -> dict:
    """psi = sin x + sin 2y  =>  omega = sin x + 4 sin 2y,  -u.grad(omega) = +6 cos x cos 2y."""
    g = SpectralGrid(n, device)
    w = torch.sin(g.x) + 4.0 * torch.sin(2.0 * g.y)
    expected = 6.0 * torch.cos(g.x) * torch.cos(2.0 * g.y)
    got = g.ifft(nonlinear_hat(g, g.fft(w) * g.mask))
    return {"max_abs_err": float((got - expected).abs().max()), "scale": 6.0}


def taylor_green(n: int = 32, nu: float = 0.01, t_end: float = 2.0, n_out: int = 21,
                 cfl: float = 0.4, device: str = "cpu") -> dict:
    """omega = 2 sin x sin y is an exact NS solution: nonlinear term vanishes, decay exp(-2 nu t)."""
    g = SpectralGrid(n, device)
    w_hat = (g.fft(2.0 * torch.sin(g.x) * torch.sin(g.y)) * g.mask)[None]
    nl = float(g.ifft(nonlinear_hat(g, w_hat)).abs().max() / 2.0)
    solver = NavierStokes2D(g, nu)
    ts, errs = [], []
    for t, wh in solver.simulate(w_hat, n_out, t_end / (n_out - 1), cfl):
        exact = w_hat * math.exp(-2.0 * nu * t)
        ts.append(t)
        errs.append(_max_rel(g.ifft(wh), g.ifft(exact)))
    return {"max_rel_err": max(errs), "nonlinear_rel": nl, "t": ts, "err": errs}


def lamb_oseen(n: int = 64, sigma0: float = 0.3, nu: float = 0.01, t_end: float = 5.0,
               n_out: int = 26, cfl: float = 0.4, device: str = "cpu") -> dict:
    """Single Gaussian (Lamb–Oseen) vortex with uniform compensating background.

    On a torus the axisymmetric vortex is not exactly steady (periodic images), so the expected
    solution is the periodized heat-kernel solution omega_hat(k, t) = omega_hat(k, 0) exp(-nu k^2 t);
    the test bounds the deviation caused by the (small) image-induced nonlinear term.
    Peak vorticity of the isolated vortex decays as sigma0^2 / (sigma0^2 + 2 nu t).
    """
    g = SpectralGrid(n, device)
    w_hat0 = gaussian_blobs(g, [(math.pi, math.pi)], [sigma0], [1.0 / sigma0**2])[None]
    solver = NavierStokes2D(g, nu)
    ts, errs, peaks, peaks_exact = [], [], [], []
    for t, wh in solver.simulate(w_hat0, n_out, t_end / (n_out - 1), cfl):
        heat = w_hat0 * torch.exp(-nu * g.k2 * t)
        w, w_ref = g.ifft(wh), g.ifft(heat)
        ts.append(t)
        errs.append(_max_rel(w, w_ref))
        peaks.append(float(w.max()))
        peaks_exact.append(float(w_ref.max()))
    iso = [peaks[0] * sigma0**2 / (sigma0**2 + 2 * nu * t) for t in ts]
    return {"max_rel_err": max(errs), "t": ts, "err": errs, "peak": peaks,
            "peak_heat_kernel": peaks_exact, "peak_isolated_formula": iso}


def make_ics(g: SpectralGrid, seed: int, k0s=(3.0, 5.0), blob_sets: int = 2, e0: float = 0.5) -> torch.Tensor:
    """A small fixed set of test initial conditions: spectral fields plus blob fields."""
    rng = np.random.default_rng(seed)
    out = [spectral_field(g, rng, k0, e0) for k0 in k0s]
    for _ in range(blob_sets):
        count = int(rng.integers(4, 9))
        centers = rng.uniform(0, 2 * math.pi, size=(count, 2))
        radii = rng.uniform(0.3, 0.5, size=count)
        signs = rng.choice([-1.0, 1.0], size=count)
        out.append(normalize_energy(g, gaussian_blobs(g, centers, radii, signs), e0))
    return torch.stack(out)


def inviscid_conservation(n: int = 64, t_end: float = 4.0, n_out: int = 41, cfl: float = 0.3,
                          seed: int = 0, device: str = "cpu", k_tail: float | None = None) -> dict:
    g = SpectralGrid(n, device)
    w_hat0 = make_ics(g, seed)
    solver = NavierStokes2D(g, 0.0)
    k_tail = k_tail if k_tail is not None else 0.75 * g.kmax
    rec = {k: [] for k in ("t", "E", "Z", "C4", "P", "tail")}
    for t, wh in solver.simulate(w_hat0, n_out, t_end / (n_out - 1), cfl):
        rec["t"].append(t)
        rec["E"].append(dg.energy(g, wh).cpu().numpy())
        rec["Z"].append(dg.enstrophy(g, wh).cpu().numpy())
        rec["C4"].append(dg.casimir(g, wh, 4).cpu().numpy())
        rec["P"].append(dg.palinstrophy(g, wh).cpu().numpy())
        rec["tail"].append(dg.enstrophy_fraction_above(g, wh, k_tail).cpu().numpy())
    arr = {k: np.array(v) for k, v in rec.items()}
    drift = {k: np.abs(arr[k] / arr[k][0] - 1.0) for k in ("E", "Z", "C4")}
    return {"max_rel_drift_E": float(drift["E"].max()), "max_rel_drift_Z": float(drift["Z"].max()),
            "max_rel_drift_C4": float(drift["C4"].max()), "t": arr["t"].tolist(),
            "drift_E": drift["E"].tolist(), "drift_Z": drift["Z"].tolist(), "drift_C4": drift["C4"].tolist(),
            "P_ratio": (arr["P"] / arr["P"][0]).tolist(), "tail": arr["tail"].tolist(), "k_tail": k_tail}


def energy_budget(n: int = 64, nu: float = 0.01, alpha: float = 0.0, forcing_amp: float = 0.0,
                  forcing_k: int = 4, t_end: float = 2.0, h: float = 2e-3, cfl: float = 0.3,
                  seed: int = 1, device: str = "cpu") -> dict:
    """Compare dE/dt (central differences of the simulated E) with the budget
    -2 nu Z - 2 alpha E + <psi f>, pointwise in time and integrated over the run."""
    g = SpectralGrid(n, device)
    w_hat0 = make_ics(g, seed)
    f_hat = None
    if forcing_amp:
        f_hat = (g.fft(forcing_amp * torch.cos(forcing_k * g.y) * torch.ones_like(g.x)) * g.mask)
    solver = NavierStokes2D(g, nu, alpha=alpha, forcing_hat=f_hat)
    n_out = int(round(t_end / h)) + 1
    E, budget = [], []
    for _, wh in solver.simulate(w_hat0, n_out, h, cfl, dt_max=h):
        e = dg.energy(g, wh)
        b = -2.0 * nu * dg.enstrophy(g, wh) - 2.0 * alpha * e
        if f_hat is not None:
            b = b + dg.energy_input(g, wh, f_hat)
        E.append(e.cpu().numpy())
        budget.append(b.cpu().numpy())
    E, budget = np.array(E), np.array(budget)
    dEdt = (E[2:] - E[:-2]) / (2 * h)
    scale = np.abs(budget[1:-1]).max(axis=0)
    pointwise = np.abs(dEdt - budget[1:-1]) / scale
    integrated = h * (budget.sum(axis=0) - 0.5 * (budget[0] + budget[-1]))  # trapezoid rule
    total_err = np.abs((E[-1] - E[0]) - integrated) / np.abs(integrated)
    t = np.arange(n_out) * h
    return {"max_rel_err_pointwise": float(pointwise.max()), "max_rel_err_integrated": float(total_err.max()),
            "t": t[1:-1][::10].tolist(), "dEdt": dEdt[::10, 0].tolist(), "budget": budget[1:-1][::10, 0].tolist()}


def convergence(n: int = 32, nu: float = 0.02, t_end: float = 1.0, dts=(0.04, 0.02, 0.01, 0.005),
                seed: int = 2, device: str = "cpu") -> dict:
    """Fixed-step IF-RK4 runs against a dt_min/8 reference: error should fall ~16x per halving."""
    g = SpectralGrid(n, device)
    w_hat0 = make_ics(g, seed, k0s=(3.0,), blob_sets=1)
    solver = NavierStokes2D(g, nu)

    def run(dt):
        nsteps = max(1, int(round(t_end / dt)))
        wh = w_hat0.clone()
        for _ in range(nsteps):
            wh = solver.step(wh, t_end / nsteps)  # exact landing on t_end
        return g.ifft(wh)

    ref = run(min(dts) / 8)
    errs = [float((run(dt) - ref).abs().max() / ref.abs().max()) for dt in dts]
    orders = [math.log2(errs[i] / errs[i + 1]) for i in range(len(errs) - 1)]
    return {"dts": list(dts), "errors": errs, "orders": orders}


def vortex_pair(n: int = 64, sep: float = 1.25, sigma: float = 0.25, circulation: float = 2 * math.pi,
                t_end: float = 1.5, n_out: int = 31, cfl: float = 0.3, device: str = "cpu") -> dict:
    """Two equal positive Gaussian vortices co-rotate counter-clockwise at
    Omega = Gamma / (pi d^2) (point-vortex value). A flipped advection sign reverses the rotation."""
    g = SpectralGrid(n, device)
    c = math.pi
    amp = circulation / (2 * math.pi * sigma**2)
    w_hat0 = gaussian_blobs(g, [(c - sep / 2, c), (c + sep / 2, c)], [sigma, sigma], [amp, amp])[None]
    solver = NavierStokes2D(g, 0.0)
    dx = (g.x - c).expand(n, n)
    dy = (g.y - c).expand(n, n)
    near = (dx**2 + dy**2) < (1.5 * sep) ** 2
    ts, angles = [], []
    prev = 0.0
    for t, wh in solver.simulate(w_hat0, n_out, t_end / (n_out - 1), cfl):
        w = g.ifft(wh)[0]
        wt = torch.where(near & (w > 0.1 * w.max()), w, torch.zeros_like(w))
        sxx, syy, sxy = (wt * dx * dx).sum(), (wt * dy * dy).sum(), (wt * dx * dy).sum()
        theta = 0.5 * math.atan2(float(2 * sxy), float(sxx - syy))
        theta += math.pi * round((prev - theta) / math.pi)  # unwrap the 180-degree axis ambiguity
        prev = theta
        ts.append(t)
        angles.append(theta)
    omega = circulation / (math.pi * sep**2)
    return {"t": ts, "angle": angles, "predicted_angle": [omega * t for t in ts],
            "final_angle": angles[-1], "final_predicted": omega * ts[-1], "ratio": angles[-1] / (omega * ts[-1])}


def _np_wavenumbers(n: int):
    k = np.fft.fftfreq(n, d=1.0 / n)
    return k[None, :], k[:, None]  # kx (columns), ky (rows)


def stored_frames(n: int = 64, nus=(0.0, 0.002, 0.01, 0.05), dt_out: float = 0.01, n_frames: int = 41,
                  cfl: float = 0.3, seed: int = 3, device: str = "cpu") -> dict:
    """Simulate, store (u, v) as float32 exactly as the dataset will, then from the stored frames
    alone (independent NumPy code) measure: spectral divergence, the vorticity-equation residual
    with the true nu, and nu re-estimated by least squares."""
    g = SpectralGrid(n, device)
    rng = np.random.default_rng(seed)
    w_hat0 = torch.stack([spectral_field(g, rng, 4.0, 0.5) for _ in nus])
    solver = NavierStokes2D(g, torch.tensor(nus, dtype=g.dtype))
    frames = []
    for _, wh in solver.simulate(w_hat0, n_frames, dt_out, cfl, dt_max=dt_out):
        u, v = g.velocity(wh)
        frames.append(np.stack([u.cpu().numpy(), v.cpu().numpy()], axis=1).astype(np.float32))
    uv = np.stack(frames, axis=1).astype(np.float64)  # (traj, time, 2, n, n)

    kx, ky = _np_wavenumbers(n)
    kmax = n // 3
    mask = (np.abs(kx) <= kmax) & (np.abs(ky) <= kmax)
    uh, vh = np.fft.fft2(uv[:, :, 0]), np.fft.fft2(uv[:, :, 1])
    wh = 1j * kx * vh - 1j * ky * uh                      # omega = dv/dx - du/dy
    div = np.fft.ifft2(1j * kx * uh + 1j * ky * vh).real
    w = np.fft.ifft2(wh).real
    wx = np.fft.ifft2(1j * kx * wh).real
    wy = np.fft.ifft2(1j * ky * wh).real
    adv = np.fft.ifft2(np.fft.fft2(uv[:, :, 0] * wx + uv[:, :, 1] * wy) * mask).real   # u.grad(omega)
    lap = np.fft.ifft2(-(kx**2 + ky**2) * wh).real
    dwdt = (w[:, 2:] - w[:, :-2]) / (2 * dt_out)
    lhs = dwdt + adv[:, 1:-1]                              # should equal nu * lap(omega)
    lap_c = lap[:, 1:-1]
    out = {"nu_true": list(nus), "nu_fit": [], "residual_rel": [], "advection_rel_rms": []}
    for i, nu in enumerate(nus):
        out["nu_fit"].append(float((lhs[i] * lap_c[i]).sum() / (lap_c[i] ** 2).sum()))
        res = lhs[i] - nu * lap_c[i]
        out["residual_rel"].append(float(np.sqrt((res**2).mean() / (adv[i, 1:-1] ** 2).mean())))
    out["div_rel_stored"] = float(np.sqrt((div**2).mean() / (w**2).mean()))
    u64, v64 = g.velocity(w_hat0)
    div64 = g.ifft(1j * g.kx * g.fft(u64) + 1j * g.ky * g.fft(v64))
    out["div_rel_float64"] = float(div64.abs().max() / g.ifft(w_hat0).abs().max())
    return out
