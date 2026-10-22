"""[BUILD] Step B: non-invertible shape noise on M2-SF-style content (results/build/PREDICTIONS.md entry 8). Exploratory.

Content input N_tau(x): RMS-normalize x -> multiply each shell k = 1..21 (round |k|; both velocity components
alike) by exp(L_k) with the log-amplitude gain
    L_k = tau * (eps1 * sigma1 * phi1(k) + eps2 * sigma2 * phi2(k)) / 2,
phi1 = centred log k (tilt), phi2 = centred phi1^2 (curvature), centred over k = 1..21; sigma1, sigma2 =
the train-set std of the data's own tilt / curvature coefficients (least-squares fit of each frame's
log shell power on [1, phi1, phi2]); eps ~ N(0, 1) per sample -> zero the mean and every shell k > 21
-> RMS-normalize again. At tau = 1 the injected tilt and curvature of the power spectrum have the
data's own spread. Training: fresh draws every step. Evaluation: per-frame draws fixed by a
deterministic hash of the RMS-normalized input (so a frame's noise is the same at every steering alpha).
Exactly blind to the input's overall scale. Float64, autocast off.

    python -m dials.shape_noise --config configs/build/stepB_calibration.yaml

computes sigma1, sigma2 and the calibrated dose (ridge R^2 of s from the content input's log shell
powers, fitted on v1 train frames, scored on v1 val) for each tau.
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import torch
import torch.nn as nn

from dials.whiten import K_WHITE, _shell_power
from solver.ns2d import SpectralGrid


def basis() -> tuple[np.ndarray, np.ndarray]:
    lk = np.log(np.arange(1, K_WHITE + 1, dtype=np.float64))
    phi1 = lk - lk.mean()
    phi2 = phi1**2 - (phi1**2).mean()
    return phi1, phi2


class ShapeNoise(nn.Module):
    def __init__(self, tau: float, sigmas, n: int = 64, hash_seed: int = 20261023):
        super().__init__()
        g = SpectralGrid(n, "cpu", dtype=torch.float64)
        self.register_buffer("shell", torch.round(g.kmag).long(), persistent=False)
        self.register_buffer("w", g.parseval_w.to(torch.float64), persistent=False)
        phi1, phi2 = basis()
        self.register_buffer("phi", torch.as_tensor(np.stack([phi1, phi2]), dtype=torch.float64), persistent=False)
        self.register_buffer("sig", torch.as_tensor(sigmas, dtype=torch.float64), persistent=False)
        gen = torch.Generator().manual_seed(hash_seed)
        self.register_buffer("hash_w", torch.randn(2, 2, n, n, generator=gen, dtype=torch.float64), persistent=False)
        self.tau, self.n = float(tau), n
        self.nk = int(g.kmag.round().max()) + 1

    @staticmethod
    def _rms(x):
        return x / torch.sqrt(x.pow(2).mean(dim=(1, 2, 3), keepdim=True) + 1e-300)

    def draws(self, xn: torch.Tensor) -> torch.Tensor:
        """(B, 2) standard-normal draws: fresh in training, hash-fixed per frame in evaluation."""
        if self.training:
            return torch.randn(len(xn), 2, device=xn.device, dtype=torch.float64)
        h = torch.einsum("bchw,jchw->bj", xn, self.hash_w)  # ~ N(0, 90^2): a random projection of the frame
        # slow wrap (x 0.1): continuous in the input away from the wrap points, so float rounding does not re-draw
        u = torch.frac(torch.abs(h) * 0.1 + 0.5).clamp(1e-7, 1 - 1e-7)
        return torch.erfinv(2 * u - 1) * 2**0.5

    def log_gain(self, eps: torch.Tensor) -> torch.Tensor:
        """(B, 2) draws -> (B, 21) log-amplitude gains for shells 1..21."""
        return self.tau * (eps * self.sig) @ self.phi / 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast("cuda", enabled=False):
            xn = self._rms(x.double())
            gain = torch.zeros(len(xn), self.nk, dtype=torch.float64, device=xn.device)
            gain[:, 1:K_WHITE + 1] = torch.exp(self.log_gain(self.draws(xn)))
            out = torch.fft.irfft2(torch.fft.rfft2(xn) * gain[:, self.shell][:, None], s=(self.n, self.n))
            return self._rms(out).float()


class Noised(nn.Module):
    """An encoder applied to N_tau(x)."""

    def __init__(self, enc: nn.Module, noise: ShapeNoise):
        super().__init__()
        self.inner, self.noise, self.d = enc, noise, enc.d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner(self.noise(x))

    def pooled(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x).mean(dim=(2, 3))


@torch.no_grad()
def log_shell_power(x: torch.Tensor) -> torch.Tensor:
    """(B, 21) log shell power (shells 1..21) of RMS-normalized velocity fields, float64."""
    g = SpectralGrid(x.shape[-1], x.device)
    shell = torch.round(g.kmag).long()
    xn = ShapeNoise._rms(x.double())
    p = _shell_power(torch.fft.rfft2(xn), shell, g.parseval_w, int(shell.max()) + 1)
    return torch.log(p[:, 1:K_WHITE + 1].clamp_min(1e-300))


@torch.no_grad()
def data_sigmas(x: torch.Tensor) -> list:
    """Train-set std of the tilt and curvature coefficients of the frames' own log shell power."""
    phi1, phi2 = basis()
    A = np.column_stack([np.ones(K_WHITE), phi1, phi2])
    coef, *_ = np.linalg.lstsq(A, log_shell_power(x).cpu().numpy().T, rcond=None)
    return [float(coef[1].std()), float(coef[2].std())]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    from common.run import REPO, append_log, command_line, environment, load_config
    from data.dataset import Dataset
    from dials.whiten import train_sample
    from probes.ridge import RidgeProbe
    from train.train import SplitData

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = load_config(ap.parse_args().config)
    t0 = time.perf_counter()
    sig = data_sigmas(train_sample(cfg["train_data_dir"]))
    ds = Dataset(cfg["train_data_dir"])
    tr, va = SplitData(ds, "train", "cuda"), SplitData(ds, "val", "cuda")
    rows_tr = torch.arange(0, tr.n, max(1, tr.n // cfg["n_fit"]), device="cuda")
    rows_va = torch.arange(0, va.n, max(1, va.n // cfg["n_score"]), device="cuda")
    f = ds.f
    s_all = np.log(f["Z"] / f["E"])
    s_tr, s_va = s_all[ds.rows(["train"])][rows_tr.cpu().numpy()], s_all[ds.rows(["val"])][rows_va.cpu().numpy()]
    res = {}
    for tau in cfg["taus"]:
        noise = ShapeNoise(tau, sig).cuda().eval()
        feats = []
        for xs in (tr.x[rows_tr], va.x[rows_va]):
            feats.append(torch.cat([log_shell_power(noise(xs[i:i + 512]).double()) for i in range(0, len(xs), 512)]).cpu().numpy())
        n_fit = len(feats[0])
        cut = int(0.8 * n_fit)  # ridge alpha chosen on a held-out 20% of the fit frames
        p = RidgeProbe().fit(feats[0][:cut], s_tr[:cut], np.zeros(cut, int), feats[0][cut:], s_tr[cut:], np.zeros(n_fit - cut, int))
        p = RidgeProbe(alphas=[p.alpha]).fit(feats[0], s_tr, np.zeros(n_fit, int), feats[0], s_tr, np.zeros(n_fit, int))
        res[tau] = p.score(feats[1], s_va, np.zeros(len(s_va), int))["r2"]
        print(f"tau {tau}: calibrated dose (ridge R^2 of s from content-input log shell powers, v1 val) = {res[tau]:.3f}", flush=True)
    L = ["# [BUILD] Step B: shape-noise calibration (before any Step B training; exploratory)", "",
         f"sigma1 (tilt) = {sig[0]:.4f}, sigma2 (curvature) = {sig[1]:.4f}: train-set std of the coefficients of each frame's "
         "log shell power on [1, phi1, phi2] (shells 1..21; the train frames train.py samples).", "",
         f"Calibrated dose: ridge from the content input's 21 log shell powers to s, fitted on {len(rows_tr)} v1 train frames "
         f"(alpha on a held-out 20%), R^2 on {len(rows_va)} v1 val frames, evaluation draws (hash-fixed per frame).", "",
         "| tau | content shape R^2 |", "|---|---|"] + [f"| {t:g} | {v:.3f} |" for t, v in res.items()]
    with open(REPO / cfg["output_md"], "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    append_log("[BUILD] Step B: shape-noise calibration (before training)", cfg["_config_path"], command_line(), "ok",
               {"sigma1 / sigma2": f"{sig[0]:.4f} / {sig[1]:.4f}",
                "content shape R^2 by tau": ", ".join(f"{t:g}: {v:.3f}" for t, v in res.items()), "results": cfg["output_md"]},
               time.perf_counter() - t0, environment().get("gpu"))


if __name__ == "__main__":
    main()
