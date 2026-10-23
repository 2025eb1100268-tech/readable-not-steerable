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

[BUILD] Stage 2, M2-SF (PREDICTIONS entry 4): separate branches, scale-free content, one field.
  The content encoder sees the input divided by its own RMS (ScaleFree: its output is exactly
  invariant to the input's overall scale); a second encoder of the same design sees the raw input
  and is pooled to the 8-dim dial vector. One FiLMDecoder renders omega from the content map, every
  stage modulated by all 8 dials; psi is derived from that omega by the inverse Laplacian, so the
  two rendered fields agree by construction.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from models.encoder import _conv, build_encoder, kaiming_init
from models.heads import HorizonEmbedding, PixelDecoder
from solver.ns2d import SpectralGrid

TARGETS = ("vorticity", "streamfunction")


class NearestUp2(nn.Module):
    """x2 nearest-neighbour upsampling by expand + reshape: its backward is a plain sum, deterministic on
    CUDA (F.interpolate's nearest backward is not)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        return x[:, :, :, None, :, None].expand(b, c, h, 2, w, 2).reshape(b, c, 2 * h, 2 * w)


class FiLMDecoder(nn.Module):
    """PixelDecoder's layers (one horizon) with FiLM after the input stage and each upsampling stage.
    upsample "pixelshuffle" (sub-pixel convolution, as PixelDecoder: GELU, 3x3 conv to 4 c_out, PixelShuffle 2)
    or [BUILD] Step C "resize" (resize-convolution: GELU, nearest x2, 3x3 conv c_in -> c_in, GELU, 3x3 conv
    c_in -> c_out; 79% of the sub-pixel decoder's parameters)."""

    def __init__(self, in_ch: int, n_dial: int, widths=(256, 128, 64, 32), out_ch: int = 1, upsample: str = "pixelshuffle"):
        super().__init__()
        self.inp = _conv(in_ch, widths[0])
        self.k = HorizonEmbedding(widths[0], 1)
        if upsample == "pixelshuffle":
            self.stages = nn.ModuleList(nn.Sequential(nn.GELU(), _conv(c_in, 4 * c_out), nn.PixelShuffle(2))
                                        for c_in, c_out in zip(widths[:-1], widths[1:]))
        elif upsample == "resize":
            self.stages = nn.ModuleList(nn.Sequential(nn.GELU(), NearestUp2(), _conv(c_in, c_in), nn.GELU(), _conv(c_in, c_out))
                                        for c_in, c_out in zip(widths[:-1], widths[1:]))
        else:
            raise ValueError(upsample)
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


class ScaleFree(nn.Module):
    """Divides each input by its RMS over channels and pixels before the wrapped encoder, so the
    output is exactly invariant to the input's overall scale (up to the 1e-12 guard)."""

    def __init__(self, enc: nn.Module):
        super().__init__()
        self.inner, self.d = enc, enc.d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        return self.inner(x / torch.sqrt(x.pow(2).mean(dim=(1, 2, 3), keepdim=True) + 1e-12))

    def pooled(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x).mean(dim=(2, 3))


class BranchHeads(nn.Module):
    """M2-SF: the dial encoder (raw input -> pooled n_dial vector) and one FiLM decoder rendering omega
    (in target units) from the content map; psi (in target units) is derived from that omega."""

    def __init__(self, cfg: dict, d_content: int):
        super().__init__()
        self.n_dial = int(cfg.get("n_dial", 8))
        self.dial_enc = build_encoder(cfg["seed"] + 500, d=self.n_dial)
        self.dec = FiLMDecoder(d_content, self.n_dial, upsample=cfg.get("decoder_upsample", "pixelshuffle"))
        self.scales = {t: float(v) for t, v in cfg["ae_target_scales"].items()}
        self._grid = None

    def grid(self, device) -> SpectralGrid:
        if self._grid is None or self._grid.device != torch.device(device):
            self._grid = SpectralGrid(64, device, dtype=torch.float32)
        return self._grid

    def dial(self, x: torch.Tensor) -> torch.Tensor:
        return self.dial_enc.pooled(x)

    def psi_from_omega(self, w: torch.Tensor) -> torch.Tensor:
        g = self.grid(w.device)
        w_hat = g.fft(w.float()[:, 0] * self.scales["vorticity"]) * g.mask
        return (g.ifft(g.streamfunction_hat(w_hat)) / self.scales["streamfunction"])[:, None]

    def render(self, dial: torch.Tensor, content: torch.Tensor) -> dict:
        w = self.dec(content, dial)
        return {"vorticity": w, "streamfunction": self.psi_from_omega(w)}

    def forward(self, content_map: torch.Tensor, x: torch.Tensor):
        """-> ({target: prediction}, pooled content (B, d_content))."""
        return self.render(self.dial(x), content_map), content_map.mean(dim=(2, 3))
