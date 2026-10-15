"""[EXPLORE] Stage 2 training options: plateau schedule, single horizon, pixel + VICReg, anchors."""
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from train.train import Model, Plateau, SplitData

V1 = Path(__file__).resolve().parents[1] / "data" / "generated" / "v1"
CKPT = Path(__file__).resolve().parents[1] / "results" / "checkpoints" / "jepa_spatial_lr0.001_s0.pt"


def test_plateau_controller_follows_entry_3_and_4():
    ctl = Plateau(min_rel_improve=0.005, patience=4, factor=0.3, max_reductions=2, start_step=2300)
    assert ctl.update(0.1, 230) == "burn-in" and ctl.best == math.inf  # early values never become the best
    assert ctl.update(1.0, 2300) == "improved" and ctl.best == 1.0
    assert ctl.update(0.996, 2530) == "no improvement"  # 0.4% < 0.5% threshold
    assert [ctl.update(1.0, 2530 + 230 * i) for i in range(1, 4)] == ["no improvement"] * 2 + ["reduce lr"]
    assert ctl.factor == pytest.approx(0.3) and ctl.reductions == 1
    assert ctl.update(0.9, 4000) == "improved"
    for _ in range(3):
        ctl.update(0.9, 4200)
    assert ctl.update(0.9, 4200) == "reduce lr" and ctl.factor == pytest.approx(0.09)
    assert [ctl.update(0.9, 5000) for _ in range(4)] == ["no improvement"] * 3 + ["stop"]


def _velocity_batch(b=4, n=64):
    from solver.initial_conditions import spectral_field
    from solver.ns2d import SpectralGrid

    g = SpectralGrid(n)
    rng = np.random.default_rng(0)
    u, v = g.velocity(torch.stack([spectral_field(g, rng, 4.0) for _ in range(b)]))
    return torch.stack([u, v], 1).float()


@pytest.mark.parametrize("kind", ["pixel", "pixel_vicreg", "jepa_spatial"])
def test_single_horizon_models_train(kind):
    x = _velocity_batch()
    m = Model({"model": kind, "seed": 0, "predictor_width": 64, "horizons": [32]})
    assert m.horizons == (32,) and m.head.k.table.shape[0] == 1
    loss, parts = m.loss(x, x, x, torch.zeros(4, dtype=torch.long), {"pred": 25.0, "var": 25.0, "cov": 1.0},
                         torch.Generator().manual_seed(0))
    loss.backward()
    assert math.isfinite(float(loss.detach())) and all(p.grad is not None for p in m.enc.parameters())
    if kind == "pixel_vicreg":
        assert set(parts) == {"pred", "var", "cov"}


def test_default_model_is_unchanged():
    """No horizons key: two horizon embeddings, as in Phase 2 (old checkpoints load)."""
    assert Model({"model": "pixel", "seed": 0}).head.k.table.shape[0] == 2


@pytest.mark.skipif(not CKPT.exists(), reason="Phase 2 checkpoint not present")
def test_phase2_checkpoint_still_loads():
    from probes.latents import load_probed_encoder

    assert load_probed_encoder("jepa_spatial_lr0.001_s0") is not None


@pytest.mark.skipif(not (V1 / "frames.npz").exists(), reason="dataset not generated")
def test_long_window_anchor_rule():
    from data.dataset import Dataset

    ds = Dataset(V1)
    sd = SplitData(ds, "val", "cpu", horizons=(32,), rule="long_windows", max_t=31, min_frames=64)
    rows = ds.rows(["val"])
    a = sd.anchors.numpy()
    frame, traj = ds.frame[rows], ds.traj[rows]
    n_frames = np.array([t["n_frames"] for t in ds.trajectories])[traj]
    assert set(np.unique(frame[a])) == set(range(1, 32))
    assert np.all(n_frames[a] == 64) and np.all(traj[a + 32] == traj[a]) and np.all(traj[a - 1] == traj[a])
    assert len(a) == 31 * len(np.unique(traj[n_frames == 64]))
    orig = SplitData(ds, "val", "cpu", horizons=(4,))  # the bridge's rule = Phase 2's rule
    assert np.array_equal(orig.anchors.numpy(), SplitData(ds, "val", "cpu").anchors.numpy())
