"""[BUILD] Stage 2b: spectrally whitened content input for M2-SW (results/build/PREDICTIONS.md entry 5). Exploratory.

W(x): the velocity field divided by its RMS (over channels and pixels), then each integer shell
k = 1..21 (round |k|) rescaled, both components alike (incompressibility is kept), so its power
equals a fixed reference ref_k: the median over v1 train frames of the RMS-normalized shell power.
The per-shell gain is capped at `cap`; the mean (k = 0) and every shell k > 21 (partial rings: the
corner modes of the solver's square dealiasing mask) are set to zero. W(c x) = W(x) for any c > 0,
so the content encoder sees neither the loudness nor (where the cap does not bind) the spectral
shape. Computed in float64 with autocast off: a float32 FFT's roundoff (~1e-7 of the field norm)
would sit far above the float32 storage floor that sets the cap.

    python -m dials.whiten --config configs/build/stage2b_whitening.yaml

writes the pre-training check (cap justification, capped fractions, whitened spectra, example
inputs) to results/build/stage2b_whitening.{md,png}.
"""
from __future__ import annotations

import argparse
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from solver.ns2d import SpectralGrid  # noqa: E402

K_WHITE = 21


def _shell_power(xh: torch.Tensor, shell: torch.Tensor, w: torch.Tensor, nk: int) -> torch.Tensor:
    """xh: (B, 2, n, n/2+1) rfft2 of a velocity field -> (B, nk) shell power (both components)."""
    p = (xh.real**2 + xh.imag**2).sum(1) * w
    out = torch.zeros(len(xh), nk, dtype=p.dtype, device=p.device)
    out.scatter_add_(1, shell.reshape(1, -1).expand(len(xh), -1), p.reshape(len(xh), -1))
    return out


class SpectralWhiten(nn.Module):
    def __init__(self, ref, cap: float, n: int = 64):
        super().__init__()
        g = SpectralGrid(n, "cpu", dtype=torch.float64)
        shell = torch.round(g.kmag).long()
        self.register_buffer("shell", shell, persistent=False)
        self.register_buffer("w", g.parseval_w.to(torch.float64), persistent=False)
        self.register_buffer("ref", torch.as_tensor(ref, dtype=torch.float64), persistent=False)
        self.nk, self.n, self.cap = int(shell.max()) + 1, n, float(cap)
        assert len(self.ref) == K_WHITE

    def normalized_spectrum(self, x: torch.Tensor):
        """-> (rfft2 of the RMS-normalized field, its shell power (B, nk)), float64."""
        x = x.double()
        x = x / torch.sqrt(x.pow(2).mean(dim=(1, 2, 3), keepdim=True) + 1e-300)
        xh = torch.fft.rfft2(x)
        return xh, _shell_power(xh, self.shell, self.w, self.nk)

    def gains(self, power: torch.Tensor) -> torch.Tensor:
        """(B, nk) shell power -> (B, nk) gains: capped sqrt(ref / power) on shells 1..21, 0 elsewhere."""
        gain = torch.zeros_like(power)
        gain[:, 1:K_WHITE + 1] = torch.sqrt(self.ref / power[:, 1:K_WHITE + 1].clamp_min(1e-300)).clamp(max=self.cap)
        return gain

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast("cuda", enabled=False):
            xh, power = self.normalized_spectrum(x)
            out = torch.fft.irfft2(xh * self.gains(power)[:, self.shell][:, None], s=(self.n, self.n))
        return out.float()


class Whitened(nn.Module):
    """An encoder applied to W(x)."""

    def __init__(self, enc: nn.Module, whiten: SpectralWhiten):
        super().__init__()
        self.inner, self.whiten, self.d = enc, whiten, enc.d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner(self.whiten(x))

    def pooled(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x).mean(dim=(2, 3))


@torch.no_grad()
def whiten_reference(x: torch.Tensor) -> list:
    """Median over frames of the RMS-normalized shell power, shells 1..21 (x: scaled velocity, (B, 2, n, n))."""
    probe = SpectralWhiten([1.0] * K_WHITE, 1.0, x.shape[-1]).to(x.device)
    _, power = probe.normalized_spectrum(x)
    return power[:, 1:K_WHITE + 1].median(dim=0).values.cpu().tolist()


def train_sample(data_dir: str, device: str = "cuda") -> torch.Tensor:
    """The v1 train frames train/train.py uses for its per-target scales: every (n // 4096)-th train frame."""
    from data.dataset import Dataset
    from train.train import SplitData
    tr = SplitData(Dataset(data_dir), "train", device)
    return tr.x[torch.arange(0, tr.n, max(1, tr.n // 4096), device=device)]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    from common.run import REPO, append_log, command_line, environment, load_config
    from data.combined import CombinedDataset
    from solver.validate import INK, INK2, SERIES, _style

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = load_config(ap.parse_args().config)
    t0 = time.perf_counter()
    ref = whiten_reference(train_sample(cfg["train_data_dir"]))
    ds = CombinedDataset(cfg["data_parts"], cfg["cache_tag"])
    rows = np.sort(np.random.default_rng(cfg["seed"]).choice(ds.rows(["explore"]), cfg["n_frames"], replace=False))
    stored = ds.velocity[rows]  # float32 as stored
    x = torch.from_numpy(stored).cuda() / ds.input_scale
    wh = SpectralWhiten(ref, cfg["cap"]).cuda()
    xh, power = wh.normalized_spectrum(x)
    ref_t = wh.ref.cpu().numpy()
    P = power[:, 1:K_WHITE + 1].cpu().numpy()
    need = np.sqrt(ref_t / np.maximum(P, 1e-300))
    # float32 storage rounding: uniform in +-ulp/2 of each stored value, through the same normalization
    gen = torch.Generator(device="cuda").manual_seed(cfg["seed"])
    ulp = torch.from_numpy(np.spacing(np.abs(stored)).astype(np.float64)).cuda()
    rms = torch.sqrt((x.double()).pow(2).mean(dim=(1, 2, 3), keepdim=True))
    noise = (torch.rand(x.shape, generator=gen, device="cuda", dtype=torch.float64) - 0.5) * ulp / ds.input_scale / rms
    nk = _shell_power(torch.fft.rfft2(noise), wh.shell, wh.w, wh.nk)[:, 1:K_WHITE + 1].cpu().numpy()
    sim_p99 = np.percentile(nk, 99, axis=0)
    # empirical floor: power where the solver puts none (shells beyond the square mask's corners, k >= floor_from_shell),
    # per mode, scaled to each shell's mode count; 99th percentile over frames
    count = torch.zeros(wh.nk, dtype=torch.float64, device="cuda").scatter_add_(0, wh.shell.reshape(-1), torch.broadcast_to(wh.w, wh.shell.shape).reshape(-1)).cpu().numpy()
    k0 = cfg["floor_from_shell"]
    per_mode = np.median(power[:, k0:].cpu().numpy() / count[k0:], axis=1)
    noise_p99 = np.maximum(np.percentile(per_mode, 99) * count[1:K_WHITE + 1], sim_p99)
    nk = np.maximum(per_mode[:, None] * count[None, 1:K_WHITE + 1], nk)
    gmax_k = np.sqrt(cfg["max_noise_share_of_ref"] * ref_t / noise_p99)
    gain = np.minimum(need, cfg["cap"])
    capped = need > cfg["cap"]
    white = torch.fft.irfft2(xh * wh.gains(power)[:, wh.shell][:, None], s=(64, 64))
    Pw_all = _shell_power(torch.fft.rfft2(white), wh.shell, wh.w, wh.nk).cpu().numpy()
    Pw = Pw_all[:, 1:K_WHITE + 1]
    beyond = Pw_all[:, K_WHITE + 1:].sum(1) / Pw.sum(1)
    uncapped_dev = np.abs(Pw[~capped] / np.broadcast_to(ref_t, Pw.shape)[~capped] - 1).max()
    amp_noise_share = (gain**2 * nk).sum(1) / Pw.sum(1)  # amplified storage noise / whitened power, per frame
    scale_err = float((wh(3.7 * x) - wh(x)).abs().max() / wh(x).abs().max())
    k = np.arange(1, K_WHITE + 1)

    # ---- markdown ----
    L = ["# [BUILD] Stage 2b: whitening check for M2-SW's content input (before any Stage 2b training; exploratory)", "",
         f"{len(rows)} explore frames (seed {cfg['seed']}); reference = median RMS-normalized shell power of the "
         f"{cfg['train_data_dir']} train frames train/train.py samples. Cap G_max = {cfg['cap']:g}.", "",
         "## Cap justification",
         f"- Noise floor, empirical: the power the stored fields carry where the solver puts none (shells k ≥ {k0}, beyond the "
         f"square dealiasing mask's corners), per mode, scaled to each shell's mode count; 99th percentile over frames. "
         f"It lies above simulated float32 rounding (uniform ±½ ulp of each stored value) by a median factor of "
         f"{np.median(np.percentile(per_mode, 99) * count[1:K_WHITE + 1] / sim_p99):.0f}, so it is the one used (the larger of "
         f"the two per shell). Floor / ref_k: k=1 {noise_p99[0] / ref_t[0]:.1e}, k=10 {noise_p99[9] / ref_t[9]:.1e}, "
         f"k=15 {noise_p99[14] / ref_t[14]:.1e}, k=21 {noise_p99[20] / ref_t[20]:.1e}.",
         f"- Largest gain that keeps the amplified floor ≤ {cfg['max_noise_share_of_ref']:.0%} of ref_k: "
         f"min over k = {gmax_k.min():.0f} (k = {gmax_k.argmin() + 1}). The cap {cfg['cap']:g} keeps it ≤ "
         f"{(cfg['cap'] ** 2 * noise_p99 / ref_t).max():.2%} of ref_k in every shell.",
         f"- Required gains (to reach ref_k) over explore: median {np.median(need):.2f}; 99th percentile per shell up to "
         f"{np.percentile(need, 99, axis=0).max():.0f}; maximum {need.max():.1e} (shells already at the storage floor).", "",
         "## What the cap costs (shape leakage into content)",
         f"- Capped (frame, shell) pairs: {capped.mean():.2%}; frames with any capped shell: {capped.any(1).mean():.1%} "
         f"(the smoothest viscous frames, high k only). Per shell k = 10, 15, 18, 21: "
         + ", ".join(f"{capped.mean(0)[i]:.1%}" for i in (9, 14, 17, 20)) + ".", "",
         "## Artefact checks on the whitened inputs",
         f"- Power beyond k = 21 / power in shells 1–21: max over frames {beyond.max():.1e} (zeroed by construction).",
         f"- Uncapped shells: whitened power equals ref_k to within {uncapped_dev:.1e} (relative).",
         f"- Amplified noise floor / whitened power, per frame: median {np.median(amp_noise_share):.1e}, "
         f"max {amp_noise_share.max():.1e}.",
         f"- Scale invariance: max |W(3.7 x) − W(x)| / max |W(x)| = {scale_err:.1e}.", "",
         "ref_k (k = 1..21): " + ", ".join(f"{v:.3g}" for v in ref_t)]
    with open(REPO / cfg["output_md"], "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")

    # ---- figure ----
    _style()
    picks = {"smoothest (largest required gain)": int(np.argmax(need.max(1))), "typical (median required gain)":
             int(np.argsort(need.max(1))[len(rows) // 2]),
             "most power beyond k = 21": int(np.argmax(power[:, K_WHITE + 1:].sum(1).cpu().numpy() / power.sum(1).cpu().numpy()))}
    g = SpectralGrid(64, "cuda")
    fig = plt.figure(figsize=(16, 12), constrained_layout=True)
    gs = fig.add_gridspec(4, 4)
    ax = fig.add_subplot(gs[0, :2])
    for q, col in zip((50, 90, 99), (SERIES[0], SERIES[1], SERIES[2])):
        ax.semilogy(k, np.percentile(need, q, axis=0), color=col, label=f"required gain, {q}th percentile")
    ax.axhline(cfg["cap"], color=INK, linestyle=(0, (3, 2)), label=f"cap {cfg['cap']:g}")
    ax.semilogy(k, gmax_k, color=INK2, linestyle=(0, (1, 2)), label=f"gain at which the floor = {cfg['max_noise_share_of_ref']:.0%} of ref")
    ax.set_xlabel("shell k")
    ax.set_title("per-shell whitening gain over explore frames")
    ax.legend(fontsize=7)
    ax = fig.add_subplot(gs[0, 2:])
    ax.bar(k, capped.mean(0) * 100, color=SERIES[0])
    ax.set_xlabel("shell k")
    ax.set_ylabel("% of explore frames capped")
    ax.set_title("where the cap binds")
    for row, (lab, i) in enumerate(picks.items(), start=1):
        orig = torch.fft.irfft2(g.vorticity_hat_from_velocity(x[i:i + 1, 0].double(), x[i:i + 1, 1].double()), s=(64, 64))[0].cpu().numpy()
        wv = torch.fft.irfft2(g.vorticity_hat_from_velocity(white[i:i + 1, 0], white[i:i + 1, 1]), s=(64, 64))[0].cpu().numpy()
        for c, (fld, ttl) in enumerate(((orig, "vorticity of the input"), (wv, "vorticity of the whitened input"))):
            a = fig.add_subplot(gs[row, c])
            lim = float(np.abs(fld).max())
            a.imshow(fld, cmap="RdBu_r", vmin=-lim, vmax=lim)
            a.set_title(f"{lab}: {ttl}" if c == 0 else ttl, fontsize=8)
            a.set_xticks([])
            a.set_yticks([])
        a = fig.add_subplot(gs[row, 2:])
        kk = np.arange(1, wh.nk)
        a.loglog(kk, power[i, 1:].cpu().numpy(), color=SERIES[0], label="input (RMS-normalized)")
        a.loglog(kk, np.maximum(Pw_all[i, 1:], 1e-40), color=SERIES[1], label="whitened input")
        a.loglog(k, ref_t, color=INK, linestyle=(0, (3, 2)), label="reference")
        a.loglog(k, nk[i], color=INK2, linestyle=(0, (1, 2)), label="noise floor (empirical)")
        a.axvline(K_WHITE + 0.5, color=INK2, linewidth=0.8)
        a.set_ylim(max(1e-30, float(nk[i].min()) / 10), 10)
        a.set_xlabel("shell k")
        a.set_title(f"shell power; capped shells: {int(capped[i].sum())}", fontsize=8)
        a.legend(fontsize=7)
    fig.suptitle("[BUILD] Stage 2b: whitening check of M2-SW's content input (exploratory)", fontweight="bold", color=INK)
    fig.savefig(REPO / cfg["figure"], dpi=110)
    plt.close(fig)
    append_log("[BUILD] Stage 2b: whitening check (before training)", cfg["_config_path"], command_line(), "ok",
               {"cap / largest floor-safe gain": f"{cfg['cap']:g} / {gmax_k.min():.0f} (k = {gmax_k.argmin() + 1})",
                "capped pairs / frames": f"{capped.mean():.2%} / {capped.any(1).mean():.1%}",
                "amplified noise / whitened power (max frame)": f"{amp_noise_share.max():.1e}",
                "results": f"{cfg['output_md']}, {cfg['figure']}"}, time.perf_counter() - t0, environment().get("gpu"))
    print("\n".join(L))


if __name__ == "__main__":
    main()
