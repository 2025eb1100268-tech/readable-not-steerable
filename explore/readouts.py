"""[EXPLORE] Readouts of an encoder's 8×8×128 spatial map for every row of a combined dataset.

  pooled       spatial mean (the pre-registered latent)
  map_meanstd  per-channel spatial mean ⊕ spatial std (256 dims)
  map_pca512   the flattened map (8192 dims) projected on its top-512 principal components,
               PCA fitted on the given fit rows (v1 train)

Inputs are held on the GPU once (``gpu_inputs``) and shared by all checkpoints. Results are cached
under results/checkpoints/latents/explore/ (gitignored). Encoders run in float32, as in Phase 3.
"""
from __future__ import annotations

import numpy as np
import torch

from common.run import REPO
from probes.latents import load_probed_encoder

READOUTS = ("pooled", "map_meanstd", "map_pca512")
CACHE = REPO / "results" / "checkpoints" / "latents" / "explore"


def gpu_inputs(ds, device: str = "cuda", chunk: int = 8192) -> torch.Tensor:
    n = len(ds.velocity)
    x = torch.empty((n,) + ds.velocity.shape[1:], dtype=torch.float32, device=device)
    for i in range(0, n, chunk):
        x[i:i + chunk] = torch.from_numpy(ds.velocity[np.arange(i, min(i + chunk, n))]).to(device)
    return x / ds.input_scale


def _paths(run: str, tag: str) -> dict:
    return {r: CACHE / f"{run}__{r}__{tag}.npy" for r in READOUTS}


@torch.no_grad()
def readouts_for(run: str, ds, x: torch.Tensor, fit_rows: np.ndarray, n_pca: int = 512, batch: int = 1024,
                 readouts=READOUTS) -> dict:
    paths = {r: p for r, p in _paths(run, ds.cache_tag).items() if r in readouts}
    if all(p.exists() for p in paths.values()):
        return {r: np.load(p) for r, p in paths.items()}
    enc = load_probed_encoder(run).to(x.device).eval()
    if "map_pca512" not in readouts:  # pooled and mean ⊕ std only (Stage 2)
        out = {"pooled": [], "map_meanstd": []}
        for i in range(0, len(x), batch):
            m = enc(x[i:i + batch])
            out["pooled"].append(m.mean(dim=(2, 3)).cpu())
            out["map_meanstd"].append(torch.cat([m.mean(dim=(2, 3)), m.std(dim=(2, 3), unbiased=False)], 1).cpu())
        res = {r: torch.cat(v).numpy().astype(np.float32) for r, v in out.items() if r in readouts}
        CACHE.mkdir(parents=True, exist_ok=True)
        for r, p in paths.items():
            np.save(p, res[r])
        return res
    fit = torch.as_tensor(np.sort(fit_rows), device=x.device)
    # PCA of the flattened map on the fit rows: mean, then centred covariance (float32 batches, float64 sum)
    mean = torch.zeros(enc.d * 64, dtype=torch.float64, device=x.device)
    for i in range(0, len(fit), batch):
        mean += enc(x[fit[i:i + batch]]).flatten(1).double().sum(0)
    mean = (mean / len(fit)).float()
    cov = torch.zeros(enc.d * 64, enc.d * 64, dtype=torch.float64, device=x.device)
    for i in range(0, len(fit), batch):
        m = enc(x[fit[i:i + batch]]).flatten(1) - mean
        cov += (m.T @ m).double()
    evals, evecs = torch.linalg.eigh((cov / len(fit)).float())
    basis = evecs[:, -n_pca:].flip(1)  # top components first
    out = {"pooled": [], "map_meanstd": [], "map_pca512": []}
    for i in range(0, len(x), batch):
        m = enc(x[i:i + batch])
        out["pooled"].append(m.mean(dim=(2, 3)).cpu())
        out["map_meanstd"].append(torch.cat([m.mean(dim=(2, 3)), m.std(dim=(2, 3), unbiased=False)], 1).cpu())
        out["map_pca512"].append(((m.flatten(1) - mean) @ basis).cpu())
    res = {r: torch.cat(v).numpy().astype(np.float32) for r, v in out.items()}
    CACHE.mkdir(parents=True, exist_ok=True)
    for r, p in paths.items():
        np.save(p, res[r])
    np.save(CACHE / f"{run}__map_pca512_explained__{ds.cache_tag}.npy",
            (evals.flip(0)[:n_pca] / evals.clamp(min=0).sum()).cpu().numpy())
    return res
