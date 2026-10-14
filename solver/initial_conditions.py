"""Random initial vorticity fields (band-limited, zero-mean, normalized to a target energy).

Family (a): periodic Gaussian vortex blobs, built analytically in Fourier space so they are
exactly periodic. Family (b): Gaussian random fields with a prescribed energy spectrum
E(k) ∝ k^4 exp(-2 (k/k0)^2) and random phases. Both families are scaled to an exact energy E0.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .diagnostics import energy
from .ns2d import SpectralGrid


def normalize_energy(grid: SpectralGrid, w_hat: torch.Tensor, e0: float) -> torch.Tensor:
    return w_hat * math.sqrt(e0 / float(energy(grid, w_hat)))


def normalize_omega_rms(grid: SpectralGrid, w_hat: torch.Tensor, omega_rms: float) -> torch.Tensor:
    """Scale to a target rms vorticity (sets the eddy time 1/omega_rms; Z0 = omega_rms^2 / 2)."""
    return w_hat * (omega_rms / math.sqrt(float(grid.mean_square(w_hat))))


def gaussian_blobs(grid: SpectralGrid, centers, radii, amplitudes) -> torch.Tensor:
    """Sum of periodized Gaussians A_j exp(-|x - x_j|^2 / (2 s_j^2)), with the mean removed.

    Fourier coefficient of one periodized blob: A s^2 / (2 pi) exp(-s^2 |k|^2 / 2) exp(-i k.x_j);
    rfft2 coefficients are n^2 times that.
    """
    w_hat = torch.zeros(grid.n, grid.n // 2 + 1, dtype=grid.cdtype, device=grid.device)
    for (cx, cy), s, a in zip(centers, radii, amplitudes):
        phase = torch.exp(-1j * (grid.kx * cx + grid.ky * cy))
        w_hat = w_hat + a * s**2 / (2 * math.pi) * torch.exp(-0.5 * s**2 * grid.k2) * phase
    w_hat = w_hat * grid.n**2
    w_hat[0, 0] = 0.0
    return w_hat * grid.mask


def random_blobs(grid: SpectralGrid, rng: np.random.Generator, count_range=(2, 12),
                 radius_range=(0.25, 0.6), e0: float = 0.5) -> torch.Tensor:
    count = int(rng.integers(count_range[0], count_range[1] + 1))
    centers = rng.uniform(0.0, 2 * math.pi, size=(count, 2))
    radii = np.exp(rng.uniform(math.log(radius_range[0]), math.log(radius_range[1]), size=count))
    signs = rng.choice([-1.0, 1.0], size=count)
    return normalize_energy(grid, gaussian_blobs(grid, centers, radii, signs), e0)


def spectral_field(grid: SpectralGrid, rng: np.random.Generator, k0: float, e0: float = 0.5) -> torch.Tensor:
    """Random-phase field with expected shell spectrum E(k) ∝ k^4 exp(-2 (k/k0)^2).

    White noise has a flat expected spectrum per mode; per-mode vorticity variance must scale as
    k E(k), so the noise is filtered by sqrt(k E(k)).
    """
    noise = torch.as_tensor(rng.standard_normal((grid.n, grid.n)), dtype=grid.dtype, device=grid.device)
    k = grid.kmag
    shape = k**4 * torch.exp(-2.0 * (k / k0) ** 2)
    w_hat = grid.fft(noise) * torch.sqrt(k * shape) * grid.mask
    w_hat[0, 0] = 0.0
    return normalize_energy(grid, w_hat, e0)
