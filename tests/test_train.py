"""Training components: symmetry views for the static control, and every model's loss."""
import math

import numpy as np
import torch

from models.heads import vicreg_var_cov
from solver.initial_conditions import spectral_field
from solver.ns2d import SpectralGrid
from train.train import Model, d4_view


def _velocity_batch(b=3, n=64):
    g = SpectralGrid(n)
    rng = np.random.default_rng(0)
    w = torch.stack([spectral_field(g, rng, 4.0) for _ in range(b)])
    u, v = g.velocity(w)
    return g, torch.stack([u, v], 1).float()


def test_d4_views_preserve_incompressibility_energy_and_vorticity_magnitude():
    g, x = _velocity_batch()
    gen = torch.Generator().manual_seed(0)
    for _ in range(12):
        y = d4_view(x, gen).double()
        div = g.ifft(1j * g.kx * g.fft(y[:, 0]) + 1j * g.ky * g.fft(y[:, 1]))
        assert float(div.abs().max()) < 1e-4
        assert float((y**2).mean()) == __import__("pytest").approx(float((x.double() ** 2).mean()), rel=1e-6)
        w_x = g.ifft(g.vorticity_hat_from_velocity(x[:, 0].double(), x[:, 1].double()))
        w_y = g.ifft(g.vorticity_hat_from_velocity(y[:, 0], y[:, 1]))
        # vorticity is transported as a pseudo-scalar: same multiset of |omega| values
        assert np.allclose(np.sort(w_x.abs().flatten().numpy()), np.sort(w_y.abs().flatten().numpy()), atol=1e-4)


def test_quarter_turn_matches_analytic_rotation():
    """omega = sin(x) rotated 90° CCW becomes omega' (x, y) = omega(y, -x) = sin(y)."""
    g = SpectralGrid(32)
    w_hat = g.fft(torch.sin(g.x).expand(32, 32))[None]
    u, v = g.velocity(w_hat)
    x = torch.stack([u, v], 1)

    class FixedGen:  # rotate once, no shift, no reflection
        calls = iter([torch.tensor([0, 0]), torch.tensor([1]), torch.tensor([0])])

    import train.train as tt
    orig = torch.randint
    try:
        tt.torch.randint = lambda *a, **k: next(FixedGen.calls)
        y = d4_view(x, None)
    finally:
        tt.torch.randint = orig
    w_y = g.ifft(g.vorticity_hat_from_velocity(y[:, 0], y[:, 1]))[0]
    assert float((w_y - torch.sin(g.y).expand(32, 32)).abs().max()) < 1e-10


def test_vicreg_terms():
    var, _ = vicreg_var_cov(torch.randn(512, 16) * 3)
    assert float(var) == 0.0  # every std well above the hinge at 1
    _, cov = vicreg_var_cov(torch.randn(4096, 16))  # independent unit dims: off-diagonals ~ 1/sqrt(n)
    assert float(cov) < 0.05
    var_c, _ = vicreg_var_cov(torch.ones(512, 16))
    assert math.isclose(float(var_c), 1 - math.sqrt(1e-4), rel_tol=1e-6)


def test_every_model_loss_is_finite_and_backpropagates():
    _, x = _velocity_batch(b=4)
    w = {"pred": 25.0, "var": 25.0, "cov": 1.0}
    for kind in ("jepa_spatial", "jepa_pooled", "pixel", "static_vicreg"):
        cfg = {"model": kind, "seed": 0, "predictor_width": 64}
        m = Model(cfg)
        loss, _ = m.loss(x, x, x, torch.tensor([0, 1, 0, 1]), w, torch.Generator().manual_seed(0))
        loss.backward()
        assert math.isfinite(float(loss.detach()))
        assert all(p.grad is not None for p in m.enc.parameters())
