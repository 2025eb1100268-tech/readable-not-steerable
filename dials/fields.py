"""[BUILD] Field conversions and the amplitude/shape coordinates (results/build/BRIEF.md).

Every decoded field is turned into its vorticity spectrum, from which
  a = log E (loudness) and s = log(Z / E) (shape: energy-weighted mean of k^2),
plus log Z, log P, the shell energy spectrum, and (for velocity) the relative divergence.
Units drop out of every slope; fields are in the models' scaled units.
"""
from __future__ import annotations

import torch

from solver import diagnostics as dg
from solver.ns2d import SpectralGrid

KINDS = ("velocity", "vorticity", "streamfunction")
K_MAX = 21  # 2/3-rule shells kept in the envelope check


def vorticity_hat(field: torch.Tensor, kind: str, g: SpectralGrid) -> torch.Tensor:
    """field: (B, 2, n, n) velocity or (B, 1, n, n) / (B, n, n) scalar -> masked rfft2 of omega."""
    f = field.to(g.dtype)
    if kind == "velocity":
        return g.vorticity_hat_from_velocity(f[:, 0], f[:, 1])
    f = f[:, 0] if f.dim() == 4 else f
    f_hat = g.fft(f) * g.mask
    if kind == "vorticity":
        return f_hat
    if kind == "streamfunction":
        return f_hat * g.k2  # omega = -lap psi
    raise ValueError(kind)


def velocity_from_vorticity_hat(w_hat: torch.Tensor, g: SpectralGrid) -> torch.Tensor:
    """Biot–Savart (inverse Laplacian): (B, 2, n, n) velocity in the same units."""
    u, v = g.velocity(w_hat)
    return torch.stack([u, v], 1)


def stats(w_hat: torch.Tensor, g: SpectralGrid) -> dict:
    """Per-field log E, log Z, log P, a, s (each a (B,) tensor)."""
    log_e, log_z = torch.log(dg.energy(g, w_hat)), torch.log(dg.enstrophy(g, w_hat))
    return {"a": log_e, "s": log_z - log_e, "log Z": log_z, "log P": torch.log(dg.palinstrophy(g, w_hat))}


def shell_spectrum(w_hat: torch.Tensor, g: SpectralGrid) -> torch.Tensor:
    """(B, K_MAX) shell energy for k = 1..K_MAX."""
    _, e = dg.energy_spectrum(g, w_hat)
    return e[:, 1:K_MAX + 1]


def relative_divergence(vel: torch.Tensor, g: SpectralGrid) -> torch.Tensor:
    """||div u|| / ||grad u|| per field (0 for an exactly incompressible field)."""
    u_hat, v_hat = g.fft(vel[:, 0].to(g.dtype)), g.fft(vel[:, 1].to(g.dtype))
    div = (1j * g.kx * u_hat + 1j * g.ky * v_hat) * g.mask
    num = (div.abs() ** 2 * g.parseval_w).sum((-2, -1))
    den = ((u_hat.abs() ** 2 + v_hat.abs() ** 2) * g.k2 * g.mask * g.parseval_w).sum((-2, -1))
    return torch.sqrt(num / den)


def head_consistency(w_hat_omega_head: torch.Tensor, w_hat_psi_head: torch.Tensor, g: SpectralGrid) -> torch.Tensor:
    """||omega_head - (-lap psi_head)|| / ||omega_head|| per field (inputs already as vorticity spectra)."""
    num = ((w_hat_omega_head - w_hat_psi_head).abs() ** 2 * g.parseval_w).sum((-2, -1))
    den = (w_hat_omega_head.abs() ** 2 * g.parseval_w).sum((-2, -1))
    return torch.sqrt(num / den)
