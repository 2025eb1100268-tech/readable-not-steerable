"""Per-frame probe targets, computed from the true float64 spectral state (never shown to models).

Definitions (pre-registered):
- E, Z, P: energy, enstrophy, palinstrophy (solver/ns2d.py conventions); C4 = <omega^4>.
- max|omega| on the 64² grid; vorticity kurtosis <omega^4>/<omega^2>^2 (exact, padded grid).
- Okubo–Weiss Q = s_n^2 + s_s^2 - omega^2 with s_n = u_x - v_y, s_s = v_x + u_y. With sigma the
  standard deviation of Q over the frame: vorticity-dominated where Q < -0.2 sigma,
  strain-dominated where Q > +0.2 sigma. "OW fraction" = vorticity-dominated area fraction.
- Vortex count: connected components of the vorticity-dominated mask, 4-connectivity, merged
  across the periodic boundaries, keeping components of at least 4 grid cells.
- Dissipation rate eps = 2 nu Z. Tail = enstrophy fraction at |k| > 16 (thermalization indicator).
"""
from __future__ import annotations

import numpy as np
import scipy.ndimage as ndi
import torch

from solver import diagnostics as dg
from solver.ns2d import SpectralGrid

OW_THRESHOLD = 0.2
MIN_VORTEX_AREA = 4
TAIL_K = 16.0


def okubo_weiss(grid: SpectralGrid, w_hat: torch.Tensor) -> torch.Tensor:
    u_hat, v_hat = grid.velocity_hat(w_hat)
    ux, uy = grid.ifft(1j * grid.kx * u_hat), grid.ifft(1j * grid.ky * u_hat)
    vx, vy = grid.ifft(1j * grid.kx * v_hat), grid.ifft(1j * grid.ky * v_hat)
    return (ux - vy) ** 2 + (vx + uy) ** 2 - (vx - uy) ** 2


def periodic_labels(mask: np.ndarray) -> np.ndarray:
    """4-connected component labels of a 2D boolean mask on a torus (0 = background)."""
    lab, n = ndi.label(mask)
    parent = np.arange(n + 1)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in list(zip(lab[0, :], lab[-1, :])) + list(zip(lab[:, 0], lab[:, -1])):
        if a and b:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    roots = np.array([find(i) for i in range(n + 1)])
    return roots[lab]


def vortex_count(mask: np.ndarray, min_area: int = MIN_VORTEX_AREA) -> int:
    lab = periodic_labels(mask)
    _, counts = np.unique(lab[lab > 0], return_counts=True)
    return int((counts >= min_area).sum())


def frame_targets(grid: SpectralGrid, w_hat: torch.Tensor, nu: torch.Tensor) -> dict:
    """All targets for a batch of frames (w_hat: (B, n, n//2+1); nu: (B,)). Returns numpy arrays."""
    z = dg.enstrophy(grid, w_hat)
    q = okubo_weiss(grid, w_hat)
    sigma = q.std(dim=(-2, -1), keepdim=True)
    vort = q < -OW_THRESHOLD * sigma
    strain = q > OW_THRESHOLD * sigma
    out = {
        "E": dg.energy(grid, w_hat),
        "Z": z,
        "P": dg.palinstrophy(grid, w_hat),
        "C4": dg.casimir(grid, w_hat, 4),
        "max_abs_w": dg.max_abs_vorticity(grid, w_hat),
        "kurtosis": dg.kurtosis(grid, w_hat),
        "ow_vort_frac": vort.to(grid.dtype).mean(dim=(-2, -1)),
        "ow_strain_frac": strain.to(grid.dtype).mean(dim=(-2, -1)),
        "eps": 2.0 * nu.to(grid.dtype) * z,
        "tail16": dg.enstrophy_fraction_above(grid, w_hat, TAIL_K),
    }
    out = {k: v.cpu().numpy() for k, v in out.items()}
    masks = vort.cpu().numpy()
    out["vortex_count"] = np.array([vortex_count(m) for m in masks], dtype=np.int32)
    return out


def integral_scale(grid: SpectralGrid, w_hat: torch.Tensor) -> torch.Tensor:
    """L = 2 pi * sum_k E(k)/k / sum_k E(k): energy-weighted mean wavelength."""
    k, spec = dg.energy_spectrum(grid, w_hat)
    k = k.to(spec.device, spec.dtype)
    inv_k = torch.where(k > 0, 1.0 / torch.clamp(k, min=1.0), torch.zeros_like(k))
    return 2 * torch.pi * (spec * inv_k).sum(-1) / spec.sum(-1)
