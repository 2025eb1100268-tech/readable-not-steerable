"""Solver validation tests (brief Phase 0) plus the extra tests that catch a sign error.

Tolerances are stated here and in configs/phase0_solver_validation.yaml; measured values are
in results/phase0/solver_validation.json.
"""
import math

import numpy as np
import pytest
import torch

import solver.checks as checks
import solver.ns2d as ns2d
from solver import diagnostics as dg
from solver.initial_conditions import gaussian_blobs, random_blobs, spectral_field
from solver.ns2d import SpectralGrid


@pytest.fixture(autouse=True)
def _threads():
    torch.set_num_threads(4)


# --- brief's required tests -----------------------------------------------------------------

def test_inviscid_conserves_energy_and_enstrophy():
    r = checks.inviscid_conservation(n=64, t_end=2.0, n_out=21, cfl=0.3)
    assert r["max_rel_drift_E"] < 1e-6
    assert r["max_rel_drift_Z"] < 1e-5


def test_viscous_energy_dissipation_matches_2_nu_Z():
    r = checks.energy_budget(n=64, nu=0.01, t_end=1.0, h=2e-3)
    assert r["max_rel_err_pointwise"] < 1e-5
    assert r["max_rel_err_integrated"] < 1e-5


def test_forced_energy_budget_with_drag():
    r = checks.energy_budget(n=64, nu=0.005, alpha=0.1, forcing_amp=1.0, t_end=1.0, h=2e-3)
    assert r["max_rel_err_pointwise"] < 1e-5


def test_taylor_green_is_exact():
    r = checks.taylor_green(n=32, nu=0.01, t_end=2.0)
    assert r["nonlinear_rel"] < 1e-12
    assert r["max_rel_err"] < 1e-10


def test_lamb_oseen_follows_periodized_heat_kernel():
    r = checks.lamb_oseen(n=64, sigma0=0.3, nu=0.01, t_end=5.0, n_out=11)
    assert r["max_rel_err"] < 1e-3


def test_recovered_velocity_is_divergence_free():
    r = checks.stored_frames(n=32, nus=(0.01,), n_frames=5)
    assert r["div_rel_float64"] < 1e-12
    assert r["div_rel_stored"] < 1e-6  # float32 storage


# --- additional tests (sign, order, data path) ------------------------------------------------

def test_nonlinear_term_matches_analytic_value():
    assert checks.analytic_rhs(n=32)["max_abs_err"] < 1e-10


def test_time_stepping_is_fourth_order():
    r = checks.convergence(n=32, nu=0.02, t_end=0.5, dts=(0.04, 0.02, 0.01))
    assert all(3.8 < p < 4.3 for p in r["orders"])


def test_vortex_pair_corotates_counterclockwise():
    r = checks.vortex_pair(n=64, t_end=1.0, n_out=11)
    assert r["final_angle"] > 0
    assert 0.85 < r["ratio"] < 1.05


def test_nu_recovered_from_stored_frames():
    r = checks.stored_frames(n=64, nus=(0.0, 0.002, 0.02), n_frames=11)
    assert abs(r["nu_fit"][0]) < 1e-4
    for nu, fit in zip(r["nu_true"][1:], r["nu_fit"][1:]):
        assert abs(fit - nu) / nu < 0.02
    assert max(r["residual_rel"]) < 5e-3


def test_flipped_advection_sign_is_caught(monkeypatch):
    """The brief's tests cannot see the sign of advection; the analytic-RHS, vortex-pair and
    stored-frame tests can."""
    original = ns2d.nonlinear_hat
    flipped = lambda g, w: -original(g, w)  # noqa: E731
    monkeypatch.setattr(ns2d, "nonlinear_hat", flipped)
    monkeypatch.setattr(checks, "nonlinear_hat", flipped)
    # blind: still pass
    assert checks.inviscid_conservation(n=64, t_end=1.0, n_out=6)["max_rel_drift_E"] < 1e-6
    assert checks.energy_budget(n=64, nu=0.01, t_end=0.5, h=2e-3)["max_rel_err_pointwise"] < 1e-5
    assert checks.taylor_green(n=32)["max_rel_err"] < 1e-10
    assert checks.lamb_oseen(n=64, n_out=6)["max_rel_err"] < 1e-3
    # sensitive: now fail
    assert checks.analytic_rhs(n=32)["max_abs_err"] > 1.0
    assert checks.vortex_pair(n=64, t_end=1.0, n_out=11)["final_angle"] < 0
    assert min(checks.stored_frames(n=64, nus=(0.002,), n_frames=11)["residual_rel"]) > 0.5


def test_nonlinear_output_is_band_limited():
    g = SpectralGrid(32)
    w_hat = spectral_field(g, np.random.default_rng(0), 4.0)[None]
    out = ns2d.nonlinear_hat(g, w_hat)
    assert float((out * (1 - g.mask)).abs().max()) == 0.0


# --- diagnostics and initial conditions --------------------------------------------------------

def test_diagnostics_single_mode():
    """omega = cos 3x: E = 1/36, Z = 1/4, P = 9/4, <omega^4> = 3/8."""
    g = SpectralGrid(32)
    w_hat = g.fft(torch.cos(3 * g.x).expand(32, 32))[None]
    assert dg.energy(g, w_hat).item() == pytest.approx(1 / 36, rel=1e-12)
    assert dg.enstrophy(g, w_hat).item() == pytest.approx(0.25, rel=1e-12)
    assert dg.palinstrophy(g, w_hat).item() == pytest.approx(2.25, rel=1e-12)
    assert dg.casimir(g, w_hat, 4).item() == pytest.approx(0.375, rel=1e-12)
    assert dg.kurtosis(g, w_hat).item() == pytest.approx(1.5, rel=1e-12)
    _, spec = dg.energy_spectrum(g, w_hat)
    assert spec[0, 3].item() == pytest.approx(1 / 36, rel=1e-12)


def test_initial_conditions_zero_mean_and_exact_energy():
    g = SpectralGrid(64)
    rng = np.random.default_rng(1)
    for w_hat in (random_blobs(g, rng, e0=0.3), spectral_field(g, rng, 3.0, e0=0.7)):
        assert abs(w_hat[0, 0]) < 1e-12
        assert float((w_hat * (1 - g.mask)).abs().max()) == 0.0
    assert dg.energy(g, random_blobs(g, rng, e0=0.3)).item() == pytest.approx(0.3, rel=1e-12)
    assert dg.energy(g, spectral_field(g, rng, 3.0, e0=0.7)).item() == pytest.approx(0.7, rel=1e-12)


def test_spectral_field_peaks_at_k0():
    g = SpectralGrid(64)
    rng = np.random.default_rng(2)
    for k0 in (3.0, 6.0):
        fields = torch.stack([spectral_field(g, rng, k0) for _ in range(20)])
        _, spec = dg.energy_spectrum(g, fields)
        assert abs(int(spec.mean(0).argmax()) - k0) <= 1


def test_resample_roundtrip_and_values():
    """64 -> 128 upsampling keeps the same physical field; 128 -> 64 recovers the original."""
    g64, g128 = SpectralGrid(64), SpectralGrid(128)
    w_hat = spectral_field(g64, np.random.default_rng(3), 4.0)
    up = ns2d.resample_hat(w_hat, 64, 128)
    assert float((g128.ifft(up)[::2, ::2] - g64.ifft(w_hat)).abs().max()) < 1e-12
    back = ns2d.resample_hat(up, 128, 64) * g64.mask
    assert float((back - w_hat).abs().max()) < 1e-9 * float(w_hat.abs().max())


def test_blob_is_periodic_gaussian():
    """A blob centred near the corner wraps around: compare with the explicit image sum."""
    g = SpectralGrid(64)
    s, cx, cy = 0.4, 0.2, 6.1
    w = g.ifft(gaussian_blobs(g, [(cx, cy)], [s], [1.0]))
    xx, yy = g.x.expand(64, 64), g.y.expand(64, 64)
    direct = torch.zeros_like(w)
    for i in range(-2, 3):
        for j in range(-2, 3):
            direct += torch.exp(-((xx - cx + 2 * math.pi * i) ** 2 + (yy - cy + 2 * math.pi * j) ** 2) / (2 * s**2))
    direct -= direct.mean()
    assert float((w - direct).abs().max()) < 1e-6
