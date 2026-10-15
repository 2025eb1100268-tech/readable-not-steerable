"""[EXPLORE] Several datasets viewed as one, so the existing probe code (probes/evaluate.py) runs
unchanged: probes and slow directions are fitted on v1 train/val rows and scored on the new
evaluation sets. Only the listed splits of each dataset are kept (v1 test splits never enter).
Trajectory ids are offset per part so they stay unique.
"""
from __future__ import annotations

import numpy as np

from data.dataset import TRANSFORMS, Dataset


class _Velocity:
    """Read-only row view over several velocity memmaps (combined row index -> part, local row)."""

    def __init__(self, parts):
        self.parts = parts
        self.starts = np.cumsum([0] + [len(r) for _, r in parts])
        self.shape = (int(self.starts[-1]),) + tuple(parts[0][0].shape[1:])

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, rows):
        rows = np.atleast_1d(np.asarray(rows))
        out = np.empty((len(rows),) + self.shape[1:], np.float32)
        part = np.searchsorted(self.starts, rows, side="right") - 1
        for k, (vel, local) in enumerate(self.parts):
            sel = np.flatnonzero(part == k)
            if len(sel):
                src = local[rows[sel] - self.starts[k]]
                order = np.argsort(src)
                out[sel[order]] = vel[src[order]]
        return out


class CombinedDataset:
    def __init__(self, parts, cache_tag: str):
        """parts: [(dataset dir, [splits to keep]), ...]; cache_tag names the latent cache."""
        dss = [(Dataset(root), list(splits)) for root, splits in parts]
        scales = {ds.input_scale for ds, _ in dss}
        assert len(scales) == 1, f"datasets disagree on the input scale: {scales}"
        self.input_scale = scales.pop()
        self.cache_tag = cache_tag
        keep = [ds.rows(splits) for ds, splits in dss]
        offsets = np.cumsum([0] + [len(ds.trajectories) for ds, _ in dss])
        keys = set.intersection(*(set(ds.f) for ds, _ in dss))
        self.f = {k: np.concatenate([ds.f[k][r] for (ds, _), r in zip(dss, keep)]) for k in keys}
        self.f["traj"] = np.concatenate([ds.traj[r] + off for (ds, _), r, off in zip(dss, keep, offsets)])
        self.traj, self.frame = self.f["traj"], self.f["frame"]
        self.split = np.concatenate([ds.split[r] for (ds, _), r in zip(dss, keep)])
        self.inviscid = np.concatenate([ds.inviscid[r] for (ds, _), r in zip(dss, keep)])
        self.blob = np.concatenate([ds.blob[r] for (ds, _), r in zip(dss, keep)])
        self.source = np.concatenate([np.full(len(r), i) for i, r in enumerate(keep)])
        self.trajectories = [dict(t, traj=t["traj"] + off) for (ds, splits), off in zip(dss, offsets)
                             for t in ds.trajectories if t["split"] in splits]
        self.n_frame_max = max(ds.n_frame_max for ds, _ in dss)
        self.velocity = _Velocity([(ds.velocity, r) for (ds, _), r in zip(dss, keep)])

    def partial_index(self) -> np.ndarray:
        return self.frame + self.n_frame_max * self.blob.astype(np.int64)

    def rows(self, splits, viscous=None) -> np.ndarray:
        keep = np.isin(self.split, list(splits))
        if viscous is True:
            keep &= ~self.inviscid
        elif viscous is False:
            keep &= self.inviscid
        return np.flatnonzero(keep)

    def target(self, name: str) -> np.ndarray:
        return TRANSFORMS[name](self.f)
