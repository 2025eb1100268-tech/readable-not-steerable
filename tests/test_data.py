"""Probe-target definitions and split policy (no leakage across splits)."""
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from data.generate import assign_splits
from data.targets import frame_targets, okubo_weiss, periodic_labels, vortex_count
from solver.initial_conditions import gaussian_blobs
from solver.ns2d import SpectralGrid

DATA = Path(__file__).resolve().parents[1] / "data" / "generated" / "v1"


def test_periodic_labels_merge_across_edges():
    m = np.zeros((8, 8), bool)
    m[0, 3] = m[7, 3] = True  # touch across the top/bottom edge
    m[4, 0] = m[4, 7] = True  # touch across the left/right edge
    m[2, 2] = True            # isolated
    lab = periodic_labels(m)
    assert len(np.unique(lab[lab > 0])) == 3


@pytest.mark.parametrize("n_vortices", [1, 3, 6])
def test_vortex_count_matches_separated_blobs(n_vortices):
    g = SpectralGrid(64)
    rng = np.random.default_rng(n_vortices)
    def torus_dist2(p, q):
        d = [min(abs(a - b) % (2 * math.pi), 2 * math.pi - abs(a - b) % (2 * math.pi)) for a, b in zip(p, q)]
        return d[0] ** 2 + d[1] ** 2

    centers = [(0.05, 3.0)]  # one straddles the periodic x-boundary
    for _ in range(10000):
        if len(centers) == n_vortices:
            break
        c = tuple(rng.uniform(0, 2 * math.pi, size=2))
        if all(torus_dist2(c, q) > 1.8**2 for q in centers):
            centers.append(c)
    assert len(centers) == n_vortices
    signs = rng.choice([-1.0, 1.0], size=n_vortices)
    w_hat = gaussian_blobs(g, centers, [0.25] * n_vortices, signs)[None]
    out = frame_targets(g, w_hat, torch.zeros(1, dtype=torch.float64))
    assert out["vortex_count"][0] == n_vortices


def test_okubo_weiss_vortex_core_is_vorticity_dominated():
    g = SpectralGrid(64)
    w_hat = gaussian_blobs(g, [(math.pi, math.pi)], [0.4], [1.0])[None]
    q = okubo_weiss(g, w_hat)[0]
    assert float(q[32, 32]) < 0  # centre of the vortex
    out = frame_targets(g, w_hat, torch.zeros(1, dtype=torch.float64))
    assert 0.0 < out["ow_vort_frac"][0] < 0.2


def test_dissipation_target_is_two_nu_z():
    g = SpectralGrid(32)
    w_hat = gaussian_blobs(g, [(1.0, 2.0), (4.0, 4.0)], [0.4, 0.5], [1.0, -1.0])[None]
    out = frame_targets(g, w_hat, torch.tensor([0.01], dtype=torch.float64))
    assert out["eps"][0] == pytest.approx(2 * 0.01 * out["Z"][0], rel=1e-12)


def test_nonphysical_dynamics_real_and_nonconservative():
    from data.nonphysical import mode_rates
    from solver import diagnostics as dg
    from solver.initial_conditions import spectral_field

    g = SpectralGrid(32)
    gk, ck = mode_rates(g, {"g_range": [0.25, 4.0], "c0": 0.5}, np.random.default_rng(0))
    w0 = spectral_field(g, np.random.default_rng(1), 4.0)
    wt = w0 * torch.exp((-0.2 * gk + 1j * ck) * 1.5) * g.mask
    # Hermitian-consistent half spectrum: a real-field round trip reproduces it
    assert float((g.fft(g.ifft(wt)) * g.mask - wt).abs().max()) < 1e-10 * float(wt.abs().max())
    # nothing quadratic is conserved (with r = 0 the per-mode amplitudes, hence E and Z, would be)
    for f in (dg.energy, dg.enstrophy):
        assert abs(float(f(g, wt)) / float(f(g, w0)) - 1) > 0.05


def _synthetic_plans(n=400, seed=0):
    rng = np.random.default_rng(seed)
    plans = []
    for i in range(n):
        inv = i % 4 == 0
        plans.append({"traj": i, "family": "blobs" if i % 2 else "spectral", "inviscid": inv,
                      "Re": math.inf if inv else float(np.exp(rng.uniform(2, 8)))})
    return plans


def test_split_policy_synthetic():
    plans = _synthetic_plans()
    meta = assign_splits(plans, {"seed": 1, "ood_quantiles": [0.45, 0.55],
                                 "fractions": {"train": 0.7, "val": 0.15, "test_id": 0.15}})
    splits = [p["split"] for p in plans]
    assert set(splits) == {"train", "val", "test_id", "test_ood"}
    lo, hi = meta["ood_log_re_band"]
    for p in plans:
        in_band = (not p["inviscid"]) and lo <= math.log(p["Re"]) <= hi
        assert (p["split"] == "test_ood") == in_band
    # inviscid runs appear in train, val and test_id (needed for Phase 4)
    assert {p["split"] for p in plans if p["inviscid"]} == {"train", "val", "test_id"}


@pytest.mark.skipif(not (DATA / "frames.npz").exists(), reason="dataset not generated")
def test_generated_dataset_has_no_split_leakage():
    with open(DATA / "trajectories.json", encoding="utf-8") as f:
        plans = json.load(f)
    split_of = np.array([p["split"] for p in plans])
    assert len({p["traj"] for p in plans}) == len(plans)
    frames = np.load(DATA / "frames.npz")
    traj = frames["traj"]
    # frames of one trajectory are contiguous, complete and therefore in exactly one split
    for p in plans[:: max(1, len(plans) // 100)]:
        rows = np.flatnonzero(traj == p["traj"])
        assert len(rows) == p["n_frames"] and np.all(np.diff(rows) == 1)
        assert np.array_equal(frames["frame"][rows], np.arange(p["n_frames"]))
    ood = [p for p in plans if p["split"] == "test_ood"]
    assert ood and all(not p["inviscid"] for p in ood)
    id_re = [math.log(p["Re"]) for p in plans if not p["inviscid"] and p["split"] != "test_ood"]
    lo, hi = min(math.log(p["Re"]) for p in ood), max(math.log(p["Re"]) for p in ood)
    assert not any(lo <= r <= hi for r in id_re)
    assert set(split_of) == {"train", "val", "test_id", "test_ood"}
