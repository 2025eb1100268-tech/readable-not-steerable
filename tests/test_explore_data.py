"""[EXPLORE] datasets: unchanged original ICs, band exclusion, matched energy, no overlap, seal.

None of these tests reads the retired v1 test splits (only v1 train/val rows and metadata).
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from data.generate import assign_single_split, build_ic, plan_matched_energy
from solver import diagnostics as dg
from solver.ns2d import SpectralGrid

GEN = Path(__file__).resolve().parents[1] / "data" / "generated"
V1, EXPLORE, CONFIRM, MATCHED = (GEN / d for d in ("v1", "explore_v2", "confirm_v2", "matched_energy_v2"))
CONFIRM3 = GEN / "confirm3"  # [BUILD]
need = lambda *ds: pytest.mark.skipif(not all((d / "manifest.json").exists() for d in ds), reason="dataset not generated")  # noqa: E731


def _json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _v1_trainval():
    return [p for p in _json(V1 / "trajectories.json") if p["split"] in ("train", "val")]


@need(V1)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="v1 was generated on CUDA")
def test_original_initial_conditions_unchanged():
    """The generator extensions leave v1's ICs as they were (frame 0 of stored train runs)."""
    cfg = _json(V1 / "manifest.json")["config"]
    g = SpectralGrid(cfg["n"], "cuda")
    plans = [p for p in _v1_trainval() if p["split"] == "train"]
    pick = [p for p in plans if p["family"] == "blobs"][:2] + [p for p in plans if p["family"] == "spectral"][:2]
    vel = np.load(V1 / "velocity.npy", mmap_mode="r")
    frames = np.load(V1 / "frames.npz")
    for p in pick:
        row = int(np.flatnonzero((frames["traj"] == p["traj"]) & (frames["frame"] == 0))[0])
        u, v = g.velocity(build_ic(g, p, cfg["ic"]) * g.mask)
        uv = torch.stack([u, v]).float().cpu().numpy()
        np.testing.assert_allclose(uv, vel[row], rtol=0, atol=1e-5 * float(np.abs(vel[row]).max()))


@need(V1)
def test_single_split_excludes_retired_band():
    lo, hi = _json(V1 / "manifest.json")["ood_re_band"]
    plans = [{"traj": 0, "inviscid": True, "Re": math.inf}, {"traj": 1, "inviscid": False, "Re": (lo + hi) / 2},
             {"traj": 2, "inviscid": False, "Re": lo / 2}, {"traj": 3, "inviscid": False, "Re": hi * 2}]
    meta = assign_single_split(plans, {"label": "explore", "exclude_re_band_from": "data/generated/v1"})
    assert [p["split"] for p in plans] == ["explore", "excluded_re_band", "explore", "explore"]
    assert meta["excluded_re_band"] == [lo, hi]


def test_matched_energy_planner_fixes_energy_and_bounds_enstrophy():
    g = SpectralGrid(64)
    cfg = {"seed": 1, "ic": {"matched_energy": {"e0": 0.2893, "slope": [3.0, 5.0], "z0": [2.0, 8.0]}}}
    plans = [{"traj": i, "family": "spectral", "inviscid": True} for i in range(12)]
    plan_matched_energy(plans, g, cfg)
    for p in plans:
        w = build_ic(g, p, cfg["ic"])
        assert float(dg.energy(g, w)) == pytest.approx(0.2893, rel=1e-12)
        assert 2.0 <= float(dg.enstrophy(g, w)) <= 8.0


@need(V1, EXPLORE, CONFIRM, MATCHED)
def test_new_datasets_share_no_seeds_or_initial_fields():
    v1 = _v1_trainval()
    vel = np.load(V1 / "velocity.npy", mmap_mode="r")
    frames = np.load(V1 / "frames.npz")
    first = {int(t): int(r) for r, t in zip(np.flatnonzero(frames["frame"] == 0), frames["traj"][frames["frame"] == 0])}
    sets = {"v1 train/val": ({p["ic_seed"] for p in v1},
                             {hashlib.sha256(np.ascontiguousarray(vel[first[p["traj"]]]).tobytes()).hexdigest() for p in v1})}
    new_sets = [("explore", EXPLORE), ("confirm", CONFIRM), ("matched", MATCHED)]
    if (CONFIRM3 / "ic_index.json").exists():  # [BUILD] confirm3, disjoint from everything before it
        new_sets.append(("confirm3", CONFIRM3))
    for name, d in new_sets:
        idx = _json(d / "ic_index.json")  # seeds and hashes only; the sealed sets are not loaded
        sets[name] = ({e["ic_seed"] for e in idx}, {e["ic_sha256"] for e in idx})
        assert len(sets[name][0]) == len(idx) and len(sets[name][1]) == len(idx)
    names = list(sets)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not sets[a][0] & sets[b][0], f"IC seeds overlap: {a} / {b}"
            assert not sets[a][1] & sets[b][1], f"initial fields overlap: {a} / {b}"


@need(EXPLORE)
def test_explore_composition():
    plans = _json(EXPLORE / "trajectories.json")
    lo, hi = _json(EXPLORE / "manifest.json")["excluded_re_band"]
    kept = [p for p in plans if p["split"] == "explore"]
    assert sum(p["inviscid"] for p in kept) >= 300
    assert all(p["inviscid"] or not (lo <= p["Re"] <= hi) for p in kept)
    assert all((not p["inviscid"]) and lo <= p["Re"] <= hi for p in plans if p["split"] == "excluded_re_band")


@need(MATCHED)
def test_matched_energy_spreads():
    plans = _json(MATCHED / "trajectories.json")
    e0, z0 = np.array([p["E0"] for p in plans]), np.array([p["Z0"] for p in plans])
    assert len(plans) == 300 and all(p["inviscid"] for p in plans)
    assert e0.std() / e0.mean() < 1e-9
    assert z0.min() >= 2.0 and z0.max() <= 8.0 and z0.max() / z0.min() >= 3.0


@need(V1, EXPLORE)
def test_combined_dataset_maps_rows_and_excludes_v1_test_splits():
    from data.combined import CombinedDataset
    from data.dataset import Dataset

    ds = CombinedDataset([("data/generated/v1", ["train", "val"]), ("data/generated/explore_v2", ["explore"])], "test")
    v1, ex = Dataset(V1), Dataset(EXPLORE)
    r1, r2 = v1.rows(["train", "val"]), ex.rows(["explore"])
    assert set(ds.split) == {"train", "val", "explore"} and len(ds.traj) == len(r1) + len(r2)
    pick = np.random.default_rng(0).choice(len(ds.traj), 64, replace=False)
    got = ds.velocity[pick]
    for g, i in zip(got, pick):
        np.testing.assert_array_equal(g, v1.velocity[r1[i]] if i < len(r1) else ex.velocity[r2[i - len(r1)]])
    assert len(np.unique(ds.traj)) == len(ds.trajectories)
    np.testing.assert_array_equal(ds.target("log_Z")[: len(r1)], v1.target("log_Z")[r1])


@need(CONFIRM, CONFIRM3)
def test_each_sealed_set_opens_only_under_its_own_preregistration(monkeypatch):
    """[BUILD] confirm3 names PREREGISTRATION_v3.md; unsealing v2 must not open it (manifests only)."""
    from data import seal

    m2, m3 = _json(CONFIRM / "manifest.json"), _json(CONFIRM3 / "manifest.json")
    assert m3["sealed"] and m3["sealed_for"] == "PREREGISTRATION_v3.md" and "sealed_for" not in m2
    monkeypatch.setattr(seal, "_UNSEALED", {"PREREGISTRATION_v2.md"})
    seal.check(CONFIRM, m2)
    with pytest.raises(seal.SealedDatasetError):
        seal.check(CONFIRM3, m3)
    monkeypatch.setattr(seal, "_UNSEALED", set())
    with pytest.raises(seal.SealedDatasetError):
        seal.check(CONFIRM, m2)
    with pytest.raises(seal.SealedDatasetError):
        seal.check(CONFIRM3, m3)
    monkeypatch.setattr(seal, "_UNSEALED", {"PREREGISTRATION_v3.md"})
    seal.check(CONFIRM3, m3)


@need(CONFIRM)
def test_confirmation_set_is_sealed():
    from data import seal
    from data.dataset import Dataset

    with pytest.raises(seal.SealedDatasetError):
        Dataset(CONFIRM)
    if not (GEN.parents[1] / "PREREGISTRATION_v2.md").exists():
        with pytest.raises(seal.SealedDatasetError):
            seal.unseal("test: must be refused before the v2 pre-registration exists")
