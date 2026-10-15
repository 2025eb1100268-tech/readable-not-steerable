"""Heads on top of the shared encoder (PREREGISTRATION.md §4).

All convolutions use circular padding. The prediction horizon k (in {1, 4}, or one horizon for
the [EXPLORE] Stage 2 models) is fed as a learned vector selected by a one-hot matmul (deterministic on CUDA, unlike nn.Embedding's backward).
Upsampling uses PixelShuffle (deterministic, unlike nearest-neighbour interpolation's backward).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoder import _conv, kaiming_init

HORIZONS = (1, 4)


class HorizonEmbedding(nn.Module):
    def __init__(self, dim: int, n_horizons: int = len(HORIZONS)):
        super().__init__()
        self.n = n_horizons
        self.table = nn.Parameter(torch.zeros(n_horizons, dim))
        nn.init.normal_(self.table, std=0.02)

    def forward(self, k_index: torch.Tensor) -> torch.Tensor:
        """k_index: (B,) integer index into the model's horizon list (HORIZONS by default)."""
        return F.one_hot(k_index, self.n).to(self.table.dtype) @ self.table


class SpatialPredictor(nn.Module):
    """Main JEPA predictor: context maps of frames (t-1, t) -> target map of frame t+k."""

    def __init__(self, d: int = 128, width: int = 256, n_horizons: int = len(HORIZONS)):
        super().__init__()
        self.inp = _conv(2 * d, width)
        self.k = HorizonEmbedding(width, n_horizons)
        self.body = nn.Sequential(nn.GELU(), _conv(width, width), nn.GELU(), _conv(width, d))
        kaiming_init(self)

    def forward(self, ctx: torch.Tensor, k_index: torch.Tensor) -> torch.Tensor:
        h = self.inp(ctx) + self.k(k_index)[:, :, None, None].to(ctx.dtype)
        return self.body(h)


class PooledPredictor(nn.Module):
    """Ablation: pooled latents of (t-1, t) -> pooled target latent of t+k."""

    def __init__(self, d: int = 128, width: int = 512, k_dim: int = 16, n_horizons: int = len(HORIZONS)):
        super().__init__()
        self.k = HorizonEmbedding(k_dim, n_horizons)
        self.net = nn.Sequential(nn.Linear(2 * d + k_dim, width), nn.GELU(), nn.Linear(width, width),
                                 nn.GELU(), nn.Linear(width, d))

    def forward(self, z_prev: torch.Tensor, z_t: torch.Tensor, k_index: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z_prev, z_t, self.k(k_index).to(z_t.dtype)], dim=1))


class PixelDecoder(nn.Module):
    """Pixel predictor head: context maps of (t-1, t) -> velocity field (u, v) at t+k."""

    def __init__(self, d: int = 128, widths=(256, 128, 64, 32), n_horizons: int = len(HORIZONS)):
        super().__init__()
        self.inp = _conv(2 * d, widths[0])
        self.k = HorizonEmbedding(widths[0], n_horizons)
        layers = []
        for c_in, c_out in zip(widths[:-1], widths[1:]):
            layers += [nn.GELU(), _conv(c_in, 4 * c_out), nn.PixelShuffle(2)]
        layers += [nn.GELU(), nn.Conv2d(widths[-1], 2, 1)]
        self.body = nn.Sequential(*layers)
        kaiming_init(self)

    def forward(self, ctx: torch.Tensor, k_index: torch.Tensor) -> torch.Tensor:
        h = self.inp(ctx) + self.k(k_index)[:, :, None, None].to(ctx.dtype)
        return self.body(h)


def vicreg_var_cov(z: torch.Tensor, eps: float = 1e-4) -> tuple[torch.Tensor, torch.Tensor]:
    """VICReg variance hinge (target std 1) and off-diagonal covariance penalty, in float32."""
    z = z.float()
    z = z - z.mean(0)
    std = torch.sqrt(z.var(0) + eps)
    var_loss = F.relu(1.0 - std).mean()
    cov = (z.T @ z) / (z.shape[0] - 1)
    off = cov - torch.diag(torch.diag(cov))
    cov_loss = (off**2).sum() / z.shape[1]
    return var_loss, cov_loss
