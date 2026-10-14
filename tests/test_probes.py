"""Probe machinery on synthetic data with known answers."""
import numpy as np
import pytest

from probes import slow_features as sf
from probes.ridge import RidgeProbe


def _data(n_traj=200, n_frames=20, d=16, seed=0):
    rng = np.random.default_rng(seed)
    traj = np.repeat(np.arange(n_traj), n_frames)
    frame = np.tile(np.arange(n_frames), n_traj)
    x = rng.standard_normal((len(traj), d))
    return rng, traj, frame, x


def test_plain_ridge_recovers_linear_target():
    rng, traj, frame, x = _data()
    w = rng.standard_normal(x.shape[1])
    y = 5.0 + x @ w + 0.1 * rng.standard_normal(len(x))
    tr, va = traj < 140, traj >= 140
    p = RidgeProbe().fit(x[tr], y[tr], frame[tr], x[va], y[va], frame[va])
    expected = 1 - 0.01 / (w @ w + 0.01)
    assert p.score(x[va], y[va], frame[va])["r2"] == pytest.approx(expected, abs=0.01)


def test_time_partialling_removes_pure_clock():
    """A latent that only encodes time gets plain R^2 > 0 but partial R^2 ~ 0; a latent that
    encodes the time-independent part gets the corresponding partial R^2."""
    rng, traj, frame, x = _data()
    clock = np.sin(frame / 3.0)               # nonlinear function of time
    signal = rng.standard_normal(len(x))
    y = 3 * clock + signal
    tr, va = traj < 140, traj >= 140
    lat_clock = np.stack([clock, clock**2], 1) + 1e-3 * rng.standard_normal((len(x), 2))
    p = RidgeProbe(partial=True, n_frame_index=20).fit(lat_clock[tr], y[tr], frame[tr], lat_clock[va], y[va], frame[va])
    s = p.score(lat_clock[va], y[va], frame[va])
    assert s["r2"] > 0.7 and abs(s["r2_partial"]) < 0.02
    lat_sig = np.stack([signal, rng.standard_normal(len(x))], 1)
    p = RidgeProbe(partial=True, n_frame_index=20).fit(lat_sig[tr], y[tr], frame[tr], lat_sig[va], y[va], frame[va])
    assert p.score(lat_sig[va], y[va], frame[va])["r2_partial"] > 0.95


def test_slow_features_find_planted_invariant():
    rng, traj, frame, x = _data(n_traj=300, n_frames=12, d=40)
    inv = rng.standard_normal(300)                     # one value per trajectory
    mix = rng.standard_normal(40)
    lat = x + np.outer(inv[traj], mix)                 # invariant hidden along a random direction
    fit_ids = traj < 200
    model = sf.fit(lat[fit_ids], traj[fit_ids], n_pca=32)
    out = sf.score(model, lat[~fit_ids], traj[~fit_ids], {"inv": dict(enumerate(inv))}, top=3)
    assert out["abs_r"]["inv"][0] > 0.95
    assert out["lambda_heldout"][0] < 0.05
