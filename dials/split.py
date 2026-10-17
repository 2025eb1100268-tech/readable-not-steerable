"""[BUILD] Stage 1 split models (results/build/PREDICTIONS.md entry 3). Exploratory.

The shared encoder's 8x8x128 map is split by channel: [0:4] dial-omega, [4:8] dial-psi,
[8:128] content. Two render heads with the autoencoders' decoder design: vorticity (target omega)
and streamfunction (target psi); each dial group reaches one head only.

  M1-spec (dial_mode "spatial"): the vorticity head decodes the [dial-omega, content] channels,
      the streamfunction head the [dial-psi, content] channels (the brief as written).
  M1-G (dial_mode "film"): the dial channels are pooled to an 8-dim vector; each head decodes the
      content map alone, modulated after its input stage and after each upsampling stage by FiLM,
      h <- h * (1 + gamma) + beta, with gamma and beta linear in that head's 4 dial dims
      (zero-initialized, so every head starts unmodulated).
  content_norm "rms" (M1-G): the content map is divided by its per-sample RMS and multiplied by
      one learned scalar gain, the same for every sample: per-frame amplitude cannot pass through
      content, and VICReg's unit-std target on the pooled content stays reachable (with the RMS
      fixed at 1 it would force the map to be spatially constant).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from models.encoder import _conv, kaiming_init
from models.heads import HorizonEmbedding, PixelDecoder

TARGETS = ("vorticity", "streamfunction")


class FiLMDecoder(nn.Module):
    """PixelDecoder's layers (one horizon) with FiLM after the input stage and each upsampling stage."""

    def __init__(self, in_ch: int, n_dial: int, widths=(256, 128, 64, 32), out_ch: int = 1):
        super().__init__()
        self.inp = _conv(in_ch, widths[0])
        self.k = HorizonEmbedding(widths[0], 1)
        self.stages = nn.ModuleList(nn.Sequential(nn.GELU(), _conv(c_in, 4 * c_out), nn.PixelShuffle(2))
                                    for c_in, c_out in zip(widths[:-1], widths[1:]))
        self.out = nn.Sequential(nn.GELU(), nn.Conv2d(widths[-1], out_ch, 1))
        kaiming_init(self)
        self.film = nn.ModuleList(nn.Linear(n_dial, 2 * w) for w in widths)
        for f in self.film:
            nn.init.zeros_(f.weight)
            nn.init.zeros_(f.bias)

    def _mod(self, h: torch.Tensor, g: torch.Tensor, i: int) -> torch.Tensor:
        gamma, beta = self.film[i](g.to(h.dtype)).chunk(2, dim=1)
        return h * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]

    def forward(self, content: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        zeros = torch.zeros(len(content), dtype=torch.long, device=content.device)
        h = self._mod(self.inp(content) + self.k(zeros)[:, :, None, None].to(content.dtype), g, 0)
        for i, stage in enumerate(self.stages):
            h = self._mod(stage(h), g, i + 1)
        return self.out(h)


class SplitHeads(nn.Module):
    def __init__(self, cfg: dict, d: int = 128):
        super().__init__()
        self.mode = cfg["dial_mode"]
        self.norm = cfg.get("content_norm", "none")
        self.nd = int(cfg.get("n_dial_per_group", 4))
        self.nc = d - 2 * self.nd
        if self.mode == "spatial":
            self.dec = nn.ModuleDict({t: PixelDecoder(d, n_horizons=1, in_ch=self.nd + self.nc, out_ch=1) for t in TARGETS})
        elif self.mode == "film":
            self.dec = nn.ModuleDict({t: FiLMDecoder(self.nc, self.nd) for t in TARGETS})
        else:
            raise ValueError(self.mode)
        if self.norm == "rms":
            self.log_gain = nn.Parameter(torch.zeros(()))
        elif self.norm != "none":
            raise ValueError(self.norm)
        self.groups = {"vorticity": slice(0, self.nd), "streamfunction": slice(self.nd, 2 * self.nd)}

    def split(self, m: torch.Tensor):
        """Encoder map -> (dial map (B, 2nd, 8, 8), content map (B, nc, 8, 8) as the heads see it)."""
        dial, content = m[:, :2 * self.nd], m[:, 2 * self.nd:]
        if self.norm == "rms":
            rms = torch.sqrt(content.float().pow(2).mean(dim=(1, 2, 3), keepdim=True) + 1e-6)
            content = (content.float() / rms * self.log_gain.exp()).to(content.dtype)
        return dial, content

    def render(self, dial: torch.Tensor, content: torch.Tensor) -> dict:
        """dial: the dial map (spatial) or the pooled 8-dim dial vector (film) -> {target: (B, 1, n, n)}."""
        if self.mode == "spatial":
            zeros = torch.zeros(len(content), dtype=torch.long, device=content.device)
            return {t: self.dec[t](torch.cat([dial[:, self.groups[t]], content], 1), zeros) for t in TARGETS}
        return {t: self.dec[t](content, dial[:, self.groups[t]]) for t in TARGETS}

    def forward(self, m: torch.Tensor):
        """-> ({target: prediction}, pooled content (B, nc))."""
        dial, content = self.split(m)
        return self.render(dial if self.mode == "spatial" else dial.mean(dim=(2, 3)), content), content.mean(dim=(2, 3))
