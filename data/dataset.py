"""Read-only access to a generated dataset and the pre-registered probe-target transforms."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from common.run import REPO
from data import seal

# Pre-registered transforms. log for heavy-tailed positive quantities, log1p for the count.
TRANSFORMS = {
    "log_E": lambda f: np.log(f["E"]),
    "log_Z": lambda f: np.log(f["Z"]),
    "log_P": lambda f: np.log(f["P"]),
    "log_max_abs_w": lambda f: np.log(f["max_abs_w"]),
    "log_kurtosis": lambda f: np.log(f["kurtosis"]),
    "ow_vort_frac": lambda f: f["ow_vort_frac"],
    "ow_strain_frac": lambda f: f["ow_strain_frac"],
    "log1p_vortex_count": lambda f: np.log1p(f["vortex_count"].astype(float)),
    "log_eps": lambda f: np.log(np.where(f["eps"] > 0, f["eps"], np.nan)),   # viscous frames only
    "log_nu": lambda f: np.log(np.where(f["nu"] > 0, f["nu"], np.nan)),      # viscous frames only
    "time": lambda f: f["time"],
}
S1_TARGETS = ["log_Z", "log_P", "log1p_vortex_count", "ow_vort_frac", "log_eps"]
VISCOUS_ONLY = {"log_eps", "log_nu"}


class Dataset:
    def __init__(self, root: str | Path):
        root = Path(root)
        if not root.is_absolute():
            root = REPO / root
        self.root = root
        with open(root / "manifest.json", encoding="utf-8") as fh:
            self.manifest = json.load(fh)
        seal.check(root, self.manifest)  # [EXPLORE] the confirmation set refuses to load
        self.velocity = np.load(root / "velocity.npy", mmap_mode="r")
        with np.load(root / "frames.npz") as f:
            self.f = {k: f[k] for k in f.files}
        with open(root / "trajectories.json", encoding="utf-8") as fh:
            self.trajectories = json.load(fh)
        self.input_scale = float(self.manifest["input_scale"])
        self.traj_split = np.array([t["split"] for t in self.trajectories])
        self.traj_inviscid = np.array([t["inviscid"] for t in self.trajectories])
        self.traj = self.f["traj"]
        self.frame = self.f["frame"]
        self.split = self.traj_split[self.traj]
        self.inviscid = self.traj_inviscid[self.traj]
        self.blob = np.array([t["family"] == "blobs" for t in self.trajectories])[self.traj]
        self.n_frame_max = int(self.frame.max()) + 1

    def partial_index(self) -> np.ndarray:
        """Pre-registered S1 partialling index: frame index crossed with initial-condition family
        (amendment 2.1a). Values 0 .. 2 * n_frame_max - 1."""
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
