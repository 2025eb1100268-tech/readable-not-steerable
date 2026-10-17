"""[BUILD] Adapters: for each model family, what its dial subspace is and how a shift in it is
decoded. A shift delta (B, D) or (D,) in dial-subspace coordinates is applied as specified in
results/build/PREDICTIONS.md (entry 1 for existing models): added at every position of the 8x8 map,
so the pooled latent moves by exactly delta.

Interface (all tensors on the model's device, inputs scaled like training inputs):
  dial_latent(x_prev, x_t)  -> (B, D) coordinates in the dial subspace
  full_latent(x_prev, x_t)  -> (B, D_full) the whole pooled latent
  decode(x_prev, x_t, delta) -> list of (head name, field kind, field tensor); the first is primary
  reencode(vel)             -> (B, D) dial coordinates of a scaled velocity field (cycle consistency)
"""
from __future__ import annotations

import torch

from dials.split import TARGETS
from train.train import AE_KINDS, PIXEL_KINDS, SPLIT_KINDS, Model


def _rows(delta: torch.Tensor, b: int) -> torch.Tensor:
    return delta.expand(b, -1) if delta.dim() == 1 else delta


class ExistingModel:
    """Stage 2 autoencoders (one head, current frame) and the k-step pixel predictor (both context
    maps shifted). Dial subspace = full pooled latent."""

    def __init__(self, model: Model):
        self.m = model.eval()
        self.kind = model.kind
        if self.kind in AE_KINDS:
            self.out_kind, self.scale = model.ae_target, model.ae_scale
        elif self.kind in PIXEL_KINDS:
            self.out_kind, self.scale = "velocity", 1.0
        else:
            raise ValueError(f"no decoder for {self.kind}")
        self.d = model.enc.d

    @torch.no_grad()
    def dial_latent(self, x_prev, x_t):
        return self.m.enc.pooled(x_t)

    full_latent = dial_latent

    @torch.no_grad()
    def decode(self, x_prev, x_t, delta):
        b = len(x_t)
        shift = _rows(delta, b)[:, :, None, None]
        zeros = torch.zeros(b, dtype=torch.long, device=x_t.device)
        if self.kind in AE_KINDS:
            out = self.m.head(self.m.enc(x_t) + shift, zeros)
        else:
            maps = self.m.enc(torch.cat([x_prev, x_t])) + torch.cat([shift, shift])
            out = self.m.head(torch.cat([maps[:b], maps[b:]], 1), zeros)
        return [("output", self.out_kind, out.float() * self.scale)]

    @torch.no_grad()
    def reencode(self, vel):
        return self.m.enc.pooled(vel.float())


class SplitModel:
    """Stage 1 split models (dials/split.py; PREDICTIONS entry 3). space "dial": the 8 pooled dial
    dims (dial-omega then dial-psi); space "full": those 8 followed by the 120 pooled content dims as
    the heads see them (after the RMS normalization of M1-G). A shift is added at every map position
    (M1-spec dial channels, content) or to the pooled dial vector (M1-G). Heads: vorticity (primary),
    streamfunction."""

    def __init__(self, model: Model, space: str = "dial"):
        assert model.kind in SPLIT_KINDS and space in ("dial", "full")
        self.m, self.h, self.space = model.eval(), model.head, space
        self.nd2 = 2 * self.h.nd
        self.d = self.nd2 if space == "dial" else model.enc.d

    @torch.no_grad()
    def latent(self, x):
        dial, content = self.h.split(self.m.enc(x.float()))
        z = dial.mean(dim=(2, 3))
        return z if self.space == "dial" else torch.cat([z, content.mean(dim=(2, 3))], 1)

    def dial_latent(self, x_prev, x_t):
        return self.latent(x_t)[:, :self.nd2]

    def full_latent(self, x_prev, x_t):
        dial, content = self.h.split(self.m.enc(x_t.float()))
        return torch.cat([dial.mean(dim=(2, 3)), content.mean(dim=(2, 3))], 1)

    @torch.no_grad()
    def decode(self, x_prev, x_t, delta):
        dial, content = self.h.split(self.m.enc(x_t.float()))
        shift = _rows(delta, len(x_t))
        if self.space == "full":
            content = content + shift[:, self.nd2:, None, None]
            shift = shift[:, :self.nd2]
        dial = dial + shift[:, :, None, None] if self.h.mode == "spatial" else dial.mean(dim=(2, 3)) + shift
        out = self.h.render(dial, content)
        return [(t, t, out[t].float() * self.m.ae_scales[t]) for t in TARGETS]

    reencode = latent
