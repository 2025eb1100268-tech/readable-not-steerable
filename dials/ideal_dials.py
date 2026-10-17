"""[BUILD] Stage 0 diagnostic (exploratory): envelope scores of IDEAL dials on real frames.

    python -m dials.ideal_dials > results/build/stage0_ideal_dials.txt

No model. Loudness: scale vorticity so a = log E moves by slope*alpha. Shape: tilt the spectrum by
k^(t*alpha) and restore E. Same frames and envelope as Stage 0. Used to set the realism threshold."""
import sys

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
from common.run import load_config  # noqa: E402
from data.combined import CombinedDataset  # noqa: E402
from dials import fields  # noqa: E402
from dials.harness import ALPHAS, envelope  # noqa: E402
from dials.stage0 import frames  # noqa: E402
from solver.ns2d import SpectralGrid  # noqa: E402

cfg = load_config("configs/build/stage0.yaml")
ds = CombinedDataset(cfg["data_parts"], cfg["cache_tag"])
g = SpectralGrid(ds.velocity.shape[-1], "cuda")
to_gpu = lambda rows: torch.from_numpy(ds.velocity[rows]).cuda() / ds.input_scale  # noqa: E731
tr = ds.rows(["train"])
env_lo, env_hi = envelope(to_gpu(np.sort(np.random.default_rng(0).choice(tr, cfg["n_envelope"], replace=False))), g)
kk = torch.where(g.kmag > 0, g.kmag, torch.ones_like(g.kmag))


def inside(w):
    sp = fields.shell_spectrum(w, g)
    return ((sp >= env_lo) & (sp <= env_hi)).double().mean(1)  # per frame


for split in ("explore", "matched"):
    w0 = fields.vorticity_hat(to_gpu(frames(ds, split, 256, 4, 0)), "velocity", g)
    a0 = fields.stats(w0, g)["a"]
    print(f"\n== {split}: envelope fraction at alpha = 0: {float(inside(w0).mean()):.3f}")
    print("loudness slope (da per unit alpha) | envelope min over alpha | per alpha")
    for slope in (0.1, 0.2, 0.5, 1.0, 1.34):
        per = [float(inside(w0 * np.exp(slope * al / 2)).mean()) for al in ALPHAS]
        print(f"  {slope:4.2f} | {min(per):.3f} | " + " ".join(f"{p:.2f}" for p in per))
    print("shape: tilt t per unit alpha -> median ds per unit alpha | da | envelope min over alpha")
    for t in (0.05, 0.1, 0.2, 0.4):
        per, s_al = [], []
        for al in ALPHAS:
            w = w0 * kk ** (t * al)
            w = w * torch.exp((a0 - fields.stats(w, g)["a"]) / 2)[:, None, None]
            per.append(float(inside(w).mean()))
            s_al.append(fields.stats(w, g)["s"].cpu().numpy())
        s_al = np.stack(s_al)
        x = ALPHAS - ALPHAS.mean()
        ds_slope = float(np.median((x[:, None] * (s_al - s_al.mean(0))).sum(0) / (x**2).sum()))
        print(f"  t={t:4.2f} | ds {ds_slope:+.3f} | da 0 | {min(per):.3f} | " + " ".join(f"{p:.2f}" for p in per))
