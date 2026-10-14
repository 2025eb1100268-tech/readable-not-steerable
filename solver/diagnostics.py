"""Integral diagnostics of a spectral vorticity state (conventions in solver/ns2d.py).

All functions accept rfft2 coefficients of shape (..., n, n//2+1) and return one value per
leading index. Quantities with higher-than-quadratic powers (Casimirs, kurtosis) are evaluated
on a zero-padded grid so the result is the exact integral of the band-limited field.
"""
from __future__ import annotations

import torch

from .ns2d import SpectralGrid


def energy(grid: SpectralGrid, w_hat: torch.Tensor) -> torch.Tensor:
    return 0.5 * (((w_hat.real**2 + w_hat.imag**2) * grid.inv_k2) * grid.parseval_w).sum(dim=(-2, -1))


def enstrophy(grid: SpectralGrid, w_hat: torch.Tensor) -> torch.Tensor:
    return 0.5 * grid.mean_square(w_hat)


def palinstrophy(grid: SpectralGrid, w_hat: torch.Tensor) -> torch.Tensor:
    return 0.5 * (((w_hat.real**2 + w_hat.imag**2) * grid.k2) * grid.parseval_w).sum(dim=(-2, -1))


def energy_input(grid: SpectralGrid, w_hat: torch.Tensor, forcing_hat: torch.Tensor) -> torch.Tensor:
    """<psi f>, the rate at which vorticity forcing f injects kinetic energy."""
    psi = w_hat * grid.inv_k2
    return ((psi.real * forcing_hat.real + psi.imag * forcing_hat.imag) * grid.parseval_w).sum(dim=(-2, -1))


def padded_vorticity(grid: SpectralGrid, w_hat: torch.Tensor, pad: int = 2) -> torch.Tensor:
    """Vorticity on a (pad*n)^2 grid by zero-padding the spectrum (same field, finer sampling)."""
    n, m = grid.n, pad * grid.n
    out = torch.zeros(w_hat.shape[:-2] + (m, m // 2 + 1), dtype=w_hat.dtype, device=w_hat.device)
    h = n // 2
    out[..., :h, : h + 1] = w_hat[..., :h, : h + 1]
    out[..., m - h :, : h + 1] = w_hat[..., n - h :, : h + 1]
    return torch.fft.irfft2(out, s=(m, m)) * (m / n) ** 2


def casimir(grid: SpectralGrid, w_hat: torch.Tensor, power: int = 4) -> torch.Tensor:
    """<omega^p> of the band-limited field (exact for p <= 2 * pad on the padded grid)."""
    w = padded_vorticity(grid, w_hat, pad=2)
    return (w**power).mean(dim=(-2, -1))


def kurtosis(grid: SpectralGrid, w_hat: torch.Tensor) -> torch.Tensor:
    w = padded_vorticity(grid, w_hat, pad=2)
    m2 = (w**2).mean(dim=(-2, -1))
    return (w**4).mean(dim=(-2, -1)) / m2**2


def max_abs_vorticity(grid: SpectralGrid, w_hat: torch.Tensor) -> torch.Tensor:
    return grid.ifft(w_hat).abs().amax(dim=(-2, -1))


def enstrophy_fraction_above(grid: SpectralGrid, w_hat: torch.Tensor, k_cut: float) -> torch.Tensor:
    """Fraction of enstrophy in modes with |k| > k_cut (thermalization / resolution indicator)."""
    e = (w_hat.real**2 + w_hat.imag**2) * grid.parseval_w
    hi = (e * (grid.kmag > k_cut).to(e.dtype)).sum(dim=(-2, -1))
    return hi / e.sum(dim=(-2, -1))


def energy_spectrum(grid: SpectralGrid, w_hat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Shell-summed energy spectrum E(k) for integer shells k = 1 .. kmax*sqrt(2); sum_k E(k) = E."""
    e = 0.5 * (w_hat.real**2 + w_hat.imag**2) * grid.inv_k2 * grid.parseval_w
    shells = torch.round(grid.kmag).long()
    nk = int(shells.max()) + 1
    flat = e.reshape(-1, e.shape[-2] * e.shape[-1])
    out = torch.zeros(flat.shape[0], nk, dtype=e.dtype, device=e.device)
    out.scatter_add_(1, shells.reshape(1, -1).expand(flat.shape[0], -1), flat)
    return torch.arange(nk), out.reshape(e.shape[:-2] + (nk,))


def kmax_eta(grid: SpectralGrid, w_hat: torch.Tensor, nu) -> torch.Tensor:
    """k_max * eta with eta = (nu^3 / beta)^(1/6), beta = 2 nu P the enstrophy dissipation rate
    (2D enstrophy-cascade dissipation scale). >= 1 is the usual well-resolved criterion."""
    nu_t = torch.as_tensor(nu, dtype=grid.dtype, device=w_hat.device)
    beta = 2.0 * nu_t * palinstrophy(grid, w_hat)
    eta = (nu_t**3 / beta) ** (1.0 / 6.0)
    return grid.kmax * eta
