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


def _tight_envelope(f):
    sp = fields.shell_spectrum(f.w0, G)
    return sp.min(0).values * 0.5, sp.max(0).values * 2.0


def test_ideal_envelope_scores_ideal_dials_as_ideal():
    """Both fake dials are ideal dials (scale; tilt + restore E), so their envelope fraction never
    falls below the ideal-dial reference by more than one shell's quantization, under a finite envelope."""
    f = FakeDials()
    env = _tight_envelope(f)
    for d in ([1.0, 0.0], [0.0, 1.0]):
        m = steer(f, None, torch.zeros(4, 2, 64, 64), np.array(d), 0.5, G, env)
        assert m["envelope_min"] < 1.0  # the envelope binds somewhere
        assert m["envelope_rel_min"] > -0.05
        assert m["envelope_ideal_by_alpha"][4] == m["envelope_by_alpha"][4]  # alpha = 0: identical fields


def _split_model(mode, norm):
    from train.train import Model
    cfg = {"model": "split_ae", "seed": 0, "horizons": [0], "dial_mode": mode, "content_norm": norm,
           "ae_target_scales": {"vorticity": 2.6, "streamfunction": 1.2}}
    return Model(cfg).eval()


def test_split_models_shapes_film_identity_and_content_norm():
    x = torch.randn(3, 2, 64, 64)
    for mode, norm in (("spatial", "none"), ("film", "rms")):
        m = _split_model(mode, norm)
        with torch.no_grad():
            preds, content = m.head(m.enc(x))
        assert set(preds) == {"vorticity", "streamfunction"} and preds["vorticity"].shape == (3, 1, 64, 64)
        assert content.shape == (3, 120)
    m = _split_model("film", "rms")
    with torch.no_grad():
        mp = m.enc(x)
        _, c1 = m.head.split(mp)
        scaled = torch.cat([mp[:, :8], 7.0 * mp[:, 8:]], 1)
        _, c2 = m.head.split(scaled)
        assert torch.allclose(c1, c2, atol=1e-5)  # per-frame amplitude of content is removed
        out1 = m.head.render(torch.zeros(3, 8), c1)
        out2 = m.head.render(torch.randn(3, 8), c1)
        assert all(torch.allclose(out1[t], out2[t]) for t in out1)  # FiLM starts as the identity


def test_split_adapter_and_loss():
    from dials.adapters import SplitModel
    x = torch.randn(4, 2, 64, 64)
    for mode, norm in (("spatial", "none"), ("film", "rms")):
        m = _split_model(mode, norm)
        with torch.no_grad():
            for t, p in m.head(m.enc(x))[0].items():
                for name, kind, f in SplitModel(m).decode(None, x, torch.zeros(8)):
                    if name == t:
                        assert kind == t and torch.allclose(f, p * m.ae_scales[t], atol=1e-5)
        dial, full = SplitModel(m, "dial"), SplitModel(m, "full")
        assert dial.reencode(x).shape == (4, 8) and full.reencode(x).shape == (4, 128)
        delta = torch.zeros(128)
        delta[:8] = torch.randn(8)
        a, b = dial.decode(None, x, delta[:8]), full.decode(None, x, delta)
        assert all(torch.allclose(u[2], v[2], atol=1e-5) for u, v in zip(a, b))
        assert torch.allclose(full.reencode(x)[:, :8], dial.reencode(x))
        m.train()
        w = {"head": {"vorticity": 14.7, "streamfunction": 42.5}, "var": 25.0, "cov": 1.0}
        loss, parts = m.loss(None, x, None, torch.zeros(4, dtype=torch.long), w)
        loss.backward()
        assert torch.isfinite(loss) and {"mse_vorticity", "mse_streamfunction", "var", "cov"} <= set(parts)
        assert m.enc.net[0].weight.grad is not None


def _branch_model():
    from train.train import Model
    cfg = {"model": "branch_ae", "seed": 0, "horizons": [0], "ae_target_scales": {"vorticity": 2.6, "streamfunction": 1.2}}
    return Model(cfg).eval()


def test_branch_model_content_is_scale_free_and_fields_agree():
    from dials.adapters import BranchModel
    m = _branch_model()
    x = torch.randn(3, 2, 64, 64)
    with torch.no_grad():
        assert torch.allclose(m.enc(x), m.enc(5.0 * x), atol=1e-5)  # content cannot see the overall scale
        assert not torch.allclose(m.head.dial(x), m.head.dial(5.0 * x), atol=1e-3)  # the dial branch can
        preds, content = m.render(x)
    assert preds["vorticity"].shape == preds["streamfunction"].shape == (3, 1, 64, 64) and content.shape == (3, 120)
    w = fields.vorticity_hat(preds["vorticity"].double() * 2.6, "vorticity", G)
    w[:, 0, 0] = 0  # psi cannot carry omega's spatial mean (zero for real flows; the untrained decoder's is not)
    p = fields.vorticity_hat(preds["streamfunction"].double() * 1.2, "streamfunction", G)
    assert float(fields.head_consistency(w, p, G).max()) < 1e-4  # psi is derived from omega
    full = BranchModel(m, "full").reencode(x)
    assert full.shape == (3, 128) and torch.allclose(full[:, 8:], BranchModel(m, "full").reencode(5.0 * x)[:, 8:], atol=1e-5)


def test_branch_adapter_and_loss():
    from dials.adapters import BranchModel
    m = _branch_model()
    x = torch.randn(4, 2, 64, 64)
    with torch.no_grad():
        (name, kind, f), = BranchModel(m).decode(None, x, torch.zeros(8))
        assert name == kind == "vorticity" and torch.allclose(f, m.render(x)[0]["vorticity"] * 2.6, atol=1e-5)
        delta = torch.zeros(128)
        delta[:8] = torch.randn(8)
        a, b = BranchModel(m, "dial").decode(None, x, delta[:8]), BranchModel(m, "full").decode(None, x, delta)
        assert torch.allclose(a[0][2], b[0][2], atol=1e-5)
    m.train()
    w = {"head": {"vorticity": 14.7, "streamfunction": 42.5}, "var": 25.0, "cov": 1.0}
    loss, parts = m.loss(None, x, None, torch.zeros(4, dtype=torch.long), w)
    loss.backward()
    assert torch.isfinite(loss) and {"mse_vorticity", "mse_streamfunction"} <= set(parts)
    assert m.head.dial_enc.net[0].weight.grad is not None and m.enc.inner.net[0].weight.grad is not None


def test_spectral_whitening():
    from dials.whiten import K_WHITE, SpectralWhiten, _shell_power
    rng = np.random.default_rng(0)
    w_hat = torch.stack([spectral_field(G, rng, 4.0) for _ in range(3)])
    x = fields.velocity_from_vorticity_hat(w_hat, G)
    ref = [10.0 ** (-0.3 * k) for k in range(1, K_WHITE + 1)]
    wh = SpectralWhiten(ref, 1e12)  # effectively uncapped for this field
    out = wh(x).double()
    assert torch.allclose(out, wh(3.7 * x).double(), atol=1e-5 * float(out.abs().max()))  # scale-free
    p = _shell_power(torch.fft.rfft2(out), wh.shell, wh.w, wh.nk)
    assert torch.allclose(p[:, 1:K_WHITE + 1], torch.tensor(ref, dtype=torch.float64).expand(3, -1), rtol=1e-4)
    assert float(p[:, K_WHITE + 1:].sum()) < 1e-12 and float(p[:, 0].sum()) < 1e-12  # mean and k > 21 zeroed
    div = fields.relative_divergence(out, G)
    assert float(div.max()) < 1e-6  # same gain for both components: still incompressible
    low = SpectralWhiten(ref, 1.0)  # cap 1: no shell is amplified
    _, p_in = low.normalized_spectrum(x)
    p_low = _shell_power(torch.fft.rfft2(low(x).double()), wh.shell, wh.w, wh.nk)
    assert torch.all(p_low[:, 1:K_WHITE + 1] <= p_in[:, 1:K_WHITE + 1] * (1 + 1e-4) + 1e-12)  # + float32 output rounding


def test_effect_space_jacobian_matches_finite_differences():
    from dials.effect_space import effect_space
    from train.train import Model
    cfg = {"model": "branch_ae", "seed": 0, "horizons": [0], "ae_target_scales": {"vorticity": 2.6, "streamfunction": 1.2}}
    m = Model(cfg).eval()
    torch.manual_seed(0)
    for f in m.head.dec.film:  # give the dials an effect (FiLM starts as the identity)
        torch.nn.init.normal_(f.weight, std=0.05)
    x = torch.randn(4, 2, 64, 64)
    es = effect_space(m, x, 4)
    assert es["jacobian"].shape == (fields.K_MAX, 8) and np.all(np.diff(es["singular_values"]) <= 1e-12)
    with torch.no_grad():
        content, g0 = m.enc(x), m.head.dial(x)
        logspec = lambda g: torch.log(fields.shell_spectrum(fields.vorticity_hat(m.head.dec(content, g).double() * 2.6, "vorticity", G), G))  # noqa: E731
        e = torch.zeros(8)
        e[3] = 1e-3
        fd = ((logspec(g0 + e) - logspec(g0 - e)) / 2e-3).mean(0).numpy()
    assert np.allclose(es["jacobian"][:, 3], fd, rtol=2e-2, atol=2e-3)


def test_shape_noise():
    from dials.shape_noise import ShapeNoise, basis, log_shell_power
    from dials.whiten import K_WHITE, _shell_power
    rng = np.random.default_rng(1)
    x = torch.stack([fields.velocity_from_vorticity_hat(spectral_field(G, rng, 4.0)[None], G)[0] for _ in range(64)]).float()
    zero = ShapeNoise(0.0, [0.5, 0.3]).eval()
    out = zero(x).double()
    assert torch.allclose(out.pow(2).mean(dim=(1, 2, 3)), torch.ones(64, dtype=torch.float64), atol=1e-5)  # RMS 1
    p = _shell_power(torch.fft.rfft2(out), zero.shell, zero.w, zero.nk)
    assert float(p[:, K_WHITE + 1:].sum() / p.sum()) < 1e-12  # k > 21 zeroed at every tau, tau = 0 included
    noisy = ShapeNoise(2.0, [0.5, 0.3]).eval()
    a, b = noisy(x), noisy(3.0 * x)
    assert torch.allclose(a, b, atol=1e-4 * float(a.abs().max()))  # scale-free up to float32 rounding of 3 x
    assert torch.equal(a, noisy(x))  # fixed per frame in evaluation
    # the injected log-power distortion is exactly tau * (eps1 sigma1 phi1 + eps2 sigma2 phi2) on shells 1..21
    d = (log_shell_power(a.double()) - log_shell_power(zero(x).double())).numpy()
    phi1, phi2 = basis()
    eps = noisy.draws(ShapeNoise._rms(x.double())).numpy()
    expect = 2.0 * (eps[:, :1] * 0.5 * phi1 + eps[:, 1:] * 0.3 * phi2)
    resid = (d - expect)[:, :15]  # shells 16-21 of this synthetic field sit at the float32 floor of the output
    assert np.allclose(resid - resid.mean(1, keepdims=True), 0, atol=1e-3)  # up to the RMS renormalization constant
    assert abs(eps.mean()) < 0.4 and 0.6 < eps.std() < 1.4  # hash draws look standard normal
    noisy.train()
    assert not torch.allclose(noisy(x), noisy(x))  # fresh draws in training


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
