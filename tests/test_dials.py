"""[BUILD] Steering harness on synthetic dials with known answers, and field conversions."""
import math

import numpy as np
import pytest
import torch

from dials import fields
from dials.harness import steer
from solver.initial_conditions import spectral_field
from solver.ns2d import SpectralGrid

G = SpectralGrid(64)


class FakeDials:
    """Two-dim dial space on fixed fields: dim 0 scales the field by exp(delta) (loudness);
    dim 1 tilts the spectrum by k^delta and restores the energy (shape at fixed loudness).
    Re-encoding returns [a / 2, s] of the field, so both dials are read back exactly."""

    def __init__(self, b=4):
        rng = np.random.default_rng(0)
        self.w0 = torch.stack([spectral_field(G, rng, 4.0) for _ in range(b)])

    def decode(self, x_prev, x_t, delta):
        delta = delta.double()
        w = self.w0 * torch.exp(delta[0])
        tilt = torch.where(G.kmag > 0, G.kmag, torch.ones_like(G.kmag)) ** delta[1]
        e_before = fields.stats(w, G)["a"]
        w = w * tilt
        w = w * torch.exp((e_before - fields.stats(w, G)["a"]) / 2)[:, None, None]
        return [("vorticity", "vorticity", G.ifft(w)[:, None])]

    def reencode(self, vel):
        st = fields.stats(fields.vorticity_hat(vel, "velocity", G), G)
        return torch.stack([st["a"] / 2, st["s"]], 1)


WIDE = (torch.zeros(fields.K_MAX, dtype=torch.float64), torch.full((fields.K_MAX,), math.inf, dtype=torch.float64))


def test_loudness_dial_is_pure_loudness():
    f = FakeDials()
    m = steer(f, None, torch.zeros(4, 2, 64, 64), np.array([1.0, 0.0]), 1.0, G, WIDE)
    assert m["d_a"] == pytest.approx(2.0, rel=1e-6) and abs(m["d_s"]) < 1e-9
    assert m["sigma_shape"] < 1e-8 and m["monotonicity"] == pytest.approx(1.0)
    assert m["cycle_corr"] == pytest.approx(1.0, abs=1e-6) and m["cycle_gain"] == pytest.approx(1.0, rel=1e-6)
    assert m["envelope_min"] == 1.0


def test_shape_dial_moves_s_at_fixed_loudness():
    f = FakeDials()
    m = steer(f, None, torch.zeros(4, 2, 64, 64), np.array([0.0, 1.0]), 1.0, G, WIDE)
    assert abs(m["d_a"]) < 1e-9 and m["d_s"] > 0.1
    assert m["sigma_shape"] > 0.999 and m["monotonicity"] == pytest.approx(1.0)
    assert m["cycle_corr"] > 0.95  # s responds to the tilt monotonically but not linearly (Pearson ~0.98)


def test_conversions_on_an_analytic_flow():
    """psi = sin(x) cos(y): omega = 2 psi; Biot–Savart round trip; Z/E = k^2 = 2 for this single shell."""
    x, y = G.x.expand(64, 64), G.y.expand(64, 64)
    psi = torch.sin(x) * torch.cos(y)
    vel = torch.stack([-torch.sin(x) * torch.sin(y), -torch.cos(x) * torch.cos(y)])[None]
    w_from_vel = fields.vorticity_hat(vel, "velocity", G)
    for kind, f in (("vorticity", 2 * psi[None, None]), ("streamfunction", psi[None, None])):
        assert torch.allclose(fields.vorticity_hat(f, kind, G), w_from_vel, atol=1e-9)
    assert torch.allclose(fields.velocity_from_vorticity_hat(w_from_vel, G), vel, atol=1e-12)
    assert float(fields.stats(w_from_vel, G)["s"][0]) == pytest.approx(math.log(2.0), abs=1e-12)
    assert float(fields.relative_divergence(vel, G)[0]) < 1e-12
    assert float(fields.head_consistency(w_from_vel, fields.vorticity_hat(psi[None, None], "streamfunction", G), G)[0]) < 1e-12
