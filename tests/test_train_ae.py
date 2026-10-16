"""[EXPLORE] entry 6 autoencoders: field transforms, losses, defaults unchanged."""
import math

import numpy as np
import pytest
import torch

from solver.ns2d import SpectralGrid
from train.train import Model, ae_field


def test_field_transforms_match_an_analytic_flow():
    """psi = sin(x) cos(y): omega = -lap psi = 2 sin(x) cos(y); u = d psi/dy, v = -d psi/dx."""
    g = SpectralGrid(64, dtype=torch.float32)
    x, y = g.x.expand(64, 64), g.y.expand(64, 64)
    psi = torch.sin(x) * torch.cos(y)
    u, v = -torch.sin(x) * torch.sin(y), -torch.cos(x) * torch.cos(y)
    vel = torch.stack([u, v])[None].float()
    assert torch.allclose(ae_field(vel, "vorticity", g)[0, 0], 2 * psi, atol=1e-5)
    assert torch.allclose(ae_field(vel, "streamfunction", g)[0, 0], psi, atol=1e-5)
    assert torch.equal(ae_field(vel, "velocity", g), vel)


def _batch(b=4):
    from solver.initial_conditions import spectral_field

    g = SpectralGrid(64)
    rng = np.random.default_rng(0)
    u, v = g.velocity(torch.stack([spectral_field(g, rng, 4.0) for _ in range(b)]))
    return torch.stack([u, v], 1).float()


@pytest.mark.parametrize("kind", ["autoencoder", "autoencoder_vicreg"])
@pytest.mark.parametrize("target", ["streamfunction", "velocity", "vorticity"])
def test_autoencoder_losses(kind, target):
    x = _batch()
    m = Model({"model": kind, "seed": 0, "horizons": [0], "ae_target": target, "ae_target_scale": 1.0})
    assert m.head.inp.in_channels == m.enc.d and m.head.body[-1].out_channels == (2 if target == "velocity" else 1)
    loss, parts = m.loss(None, x, x, torch.zeros(4, dtype=torch.long), {"pred": 25.0, "var": 25.0, "cov": 1.0}, None)
    loss.backward()
    assert math.isfinite(float(loss.detach())) and all(p.grad is not None for p in m.enc.parameters())
    assert set(parts) == ({"pred"} if kind == "autoencoder" else {"pred", "var", "cov"})


def test_pixel_decoder_defaults_unchanged():
    m = Model({"model": "pixel", "seed": 0})
    assert m.head.inp.in_channels == 2 * m.enc.d and m.head.body[-1].out_channels == 2
