"""Unsupervised invariant discovery (Phase 4 protocol, also used by the Phase 0 pilot).

Find latent directions that vary little within a trajectory but a lot across trajectories:
solve  Sigma_within w = lambda Sigma_between w  and keep the lowest-lambda directions.

Protocol (pre-registered): z-score latents and PCA to ``n_pca`` dims, both fitted on the fitting
trajectories only; Ledoit–Wolf shrinkage of Sigma_within; directions fitted on train+val
trajectories and scored on held-out trajectories. No labels are used until scoring.
"""
from __future__ import annotations

import numpy as np
import scipy.linalg
from sklearn.covariance import ledoit_wolf


def _traj_means(x: np.ndarray, traj: np.ndarray):
    ids, inv = np.unique(traj, return_inverse=True)
    sums = np.zeros((len(ids), x.shape[1]))
    np.add.at(sums, inv, x)
    counts = np.bincount(inv).astype(float)
    return ids, inv, sums / counts[:, None]


def fit(latents: np.ndarray, traj: np.ndarray, n_pca: int = 32) -> dict:
    mu = latents.mean(0)
    sd = latents.std(0) + 1e-12
    zs = (latents - mu) / sd
    _, _, vt = np.linalg.svd(zs - zs.mean(0), full_matrices=False)
    k = min(n_pca, vt.shape[0])
    pca = vt[:k].T
    x = zs @ pca
    _, inv, means = _traj_means(x, traj)
    within = x - means[inv]
    sw, shrink = ledoit_wolf(within, assume_centered=True)
    sb = np.cov(means, rowvar=False)
    lam, w = scipy.linalg.eigh(sw, sb)  # ascending: slowest first
    return {"mu": mu, "sd": sd, "pca": pca, "w": w, "lambda_fit": lam, "shrinkage": float(shrink)}


def project(model: dict, latents: np.ndarray) -> np.ndarray:
    return ((latents - model["mu"]) / model["sd"]) @ model["pca"] @ model["w"]


def within_between_ratio(values: np.ndarray, traj: np.ndarray) -> np.ndarray:
    """Per column: mean within-trajectory variance / variance of trajectory means."""
    values = values.reshape(len(values), -1)
    _, inv, means = _traj_means(values, traj)
    within = ((values - means[inv]) ** 2).mean(0)
    return within / means.var(0)


def score(model: dict, latents: np.ndarray, traj: np.ndarray, targets: dict, top: int = 3) -> dict:
    """Held-out scoring: lambda of each top direction and |Pearson r| between the trajectory-mean
    projection and each trajectory-level target (one value per trajectory)."""
    proj = project(model, latents)[:, :top]
    ids, _, means = _traj_means(proj, traj)
    out = {"lambda_heldout": within_between_ratio(proj, traj).tolist(), "abs_r": {}, "max_abs_r": {}}
    for name, per_traj in targets.items():
        y = np.asarray([per_traj[i] for i in ids], dtype=float)
        rs = [abs(np.corrcoef(means[:, j], y)[0, 1]) for j in range(proj.shape[1])]
        out["abs_r"][name] = rs
        out["max_abs_r"][name] = max(rs)
    return out
