"""Shared encoder (identical for JEPA, pixel predictor, controls and the untrained control).

Input (B, 2, 64, 64) velocity divided by one dataset-wide scale. Circular padding (periodic
domain), GELU, no normalization layers (per-sample normalization would erase global amplitude,
hence energy), three stride-2 stages to an 8x8 map with d channels. The pooled latent is the
spatial mean of that map (torch.mean, not AdaptiveAvgPool2d, whose CUDA backward is
nondeterministic). Kaiming-normal init (fan_in, ReLU gain), zero biases: the untrained control is
exactly this network at initialization.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _conv(c_in: int, c_out: int, stride: int = 1, k: int = 3) -> nn.Conv2d:
    return nn.Conv2d(c_in, c_out, k, stride=stride, padding=k // 2, padding_mode="circular")


class Encoder(nn.Module):
    def __init__(self, widths=(32, 64, 128, 128), d: int = 128):
        super().__init__()
        layers = [_conv(2, widths[0]), nn.GELU()]
        c_in = widths[0]
        for w in widths[1:]:
            layers += [_conv(c_in, w, stride=2), nn.GELU(), _conv(w, w), nn.GELU()]
            c_in = w
        layers.append(nn.Conv2d(c_in, d, 1))
        self.net = nn.Sequential(*layers)
        self.d = d
        kaiming_init(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Spatial feature map (B, d, 8, 8)."""
        return self.net(x)

    def pooled(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).mean(dim=(2, 3))


def kaiming_init(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            nn.init.zeros_(m.bias)


@torch.no_grad()
def encode_rows(encoder: nn.Module, velocity, rows, scale: float, device: str, batch: int = 2048) -> "np.ndarray":
    """Pooled latents (float32) for the given dataset rows; input divided by the dataset scale."""
    import numpy as np

    encoder = encoder.to(device).eval()
    out = []
    for i in range(0, len(rows), batch):
        x = torch.from_numpy(np.ascontiguousarray(velocity[rows[i:i + batch]])).to(device) / scale
        out.append(encoder.pooled(x).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, encoder.d), np.float32)


def build_encoder(seed: int, widths=(32, 64, 128, 128), d: int = 128) -> Encoder:
    """Deterministic construction: the same seed gives the same initial weights."""
    gen_state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    enc = Encoder(widths, d)
    torch.random.set_rng_state(gen_state)
    return enc
