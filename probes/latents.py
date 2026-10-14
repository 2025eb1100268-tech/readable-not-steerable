"""Load probed encoders and extract pooled latents for every dataset row (cached on disk).

Readouts (PREREGISTRATION.md §5.1): 'raw' pooled latent (primary) and 'shift_avg', the pooled
latent averaged over the 64 sub-stride circular shifts (secondary; makes any encoder exactly
invariant to integer-pixel translations). Latents are computed in float32.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from common.run import REPO
from models.encoder import build_encoder

CACHE = REPO / "results" / "checkpoints" / "latents"
SHIFTS = [(dy, dx) for dy in range(8) for dx in range(8)]


def load_probed_encoder(run_name: str) -> torch.nn.Module:
    """Trained run -> its probed encoder (EMA target for JEPA variants). 'untrained_s<k>' -> the
    shared encoder at initialization with seed k (the untrained control)."""
    if run_name.startswith("untrained_s"):
        return build_encoder(int(run_name.rsplit("_s", 1)[1]))
    from train.train import Model

    ckpt = torch.load(REPO / "results" / "checkpoints" / f"{run_name}.pt", map_location="cpu", weights_only=False)
    model = Model(ckpt["config"])
    model.load_state_dict(ckpt["model"])
    return model.probed_encoder()


@torch.no_grad()
def extract(encoder: torch.nn.Module, velocity, rows: np.ndarray, scale: float, readout: str,
            device: str = "cuda", batch: int = 1024) -> np.ndarray:
    encoder = encoder.to(device).eval()
    out = np.zeros((len(rows), encoder.d), np.float32)
    for i in range(0, len(rows), batch):
        x = torch.from_numpy(np.ascontiguousarray(velocity[rows[i:i + batch]])).to(device) / scale
        if readout == "raw":
            z = encoder.pooled(x)
        elif readout == "shift_avg":
            z = sum(encoder.pooled(torch.roll(x, shifts=s, dims=(-2, -1))) for s in SHIFTS) / len(SHIFTS)
        else:
            raise ValueError(readout)
        out[i:i + len(z)] = z.float().cpu().numpy()
    return out


def latents_for(run_name: str, ds, readout: str, rows: np.ndarray | None = None, device: str = "cuda") -> np.ndarray:
    """Latents for all dataset rows (or the given rows; other rows are NaN), cached per run/readout."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{run_name}__{readout}.npy"
    n = len(ds.traj)
    if path.exists():
        lat = np.load(path)
        need = rows if rows is not None else np.arange(n)
        if not np.isnan(lat[need, 0]).any():
            return lat
    else:
        lat = np.full((n, 128), np.nan, np.float32)
    need = np.arange(n) if rows is None else rows
    todo = need[np.isnan(lat[need, 0])]
    if len(todo):
        lat[todo] = extract(load_probed_encoder(run_name), ds.velocity, todo, ds.input_scale, readout, device)
        np.save(path, lat)
    return lat
