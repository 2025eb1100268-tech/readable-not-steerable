"""[BUILD] Stage 2b supplementary figure: M2-SW's PC2 (the PCA shape-dial candidate) in every seed,
decoded at alpha = -2, -1, 0, +1, +2 on one explore and one matched-energy steering frame, with the
change field and the shell spectra. Directions and sigma_d are read from results/build/stage2b.json.gz.
Exploratory; no new measurement.

    python -m dials.stage2b_shape_figure
"""
from __future__ import annotations

import gzip
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from common.run import REPO, load_config  # noqa: E402
from data.combined import CombinedDataset  # noqa: E402
from dials import fields  # noqa: E402
from dials.adapters import adapter_for  # noqa: E402
from dials.stage0 import frames  # noqa: E402
from dials.stage1 import load  # noqa: E402
from solver.ns2d import SpectralGrid  # noqa: E402
from solver.validate import INK, SERIES, _style  # noqa: E402

ALPHAS = (-2.0, -1.0, 0.0, 1.0, 2.0)


@torch.no_grad()
def main():
    cfg = load_config(str(REPO / "configs" / "build" / "stage2b_eval.yaml"))
    res = json.load(gzip.open(REPO / cfg["output_json_gz"], "rt", encoding="utf-8"))["results"]["M2-SW"]
    ds = CombinedDataset(cfg["data_parts"], cfg["cache_tag"])
    g = SpectralGrid(64, "cuda")
    xs = {s: torch.from_numpy(ds.velocity[frames(ds, s, cfg["n_frames"], 4, cfg["seed"])[:1]]).cuda() / ds.input_scale
          for s in ("explore", "matched")}
    _style()
    rows = [(r, s) for r in res for s in xs]
    fig, axes = plt.subplots(len(rows), len(ALPHAS) + 2, figsize=(19, 2.9 * len(rows)), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1] * (len(ALPHAS) + 1) + [1.5]})
    for row, (r, s) in zip(axes, rows):
        e = r["directions"]["PC2"]
        ad = adapter_for(load(r["run"]), "dial")
        d = torch.as_tensor(e["direction"], dtype=torch.float32, device="cuda")
        out = [ad.decode(None, xs[s], float(a * e["sigma"]) * d)[0][2][0, 0].cpu().numpy() for a in ALPHAS]
        lim = float(np.abs(out[2]).max())
        m = e[s]
        for ax, f, a in zip(row, out, ALPHAS):
            ax.imshow(f, cmap="RdBu_r", vmin=-lim, vmax=lim)
            ax.set_title((f"seed {r['run'][-1]}, {s}: " if a == -2 else "") + f"α = {a:+g}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
        diff = out[-1] - out[2]
        dl = float(np.abs(diff).max())
        row[len(ALPHAS)].imshow(diff, cmap="RdBu_r", vmin=-dl, vmax=dl)
        row[len(ALPHAS)].set_title(f"(+2) − (0), ±{dl:.2g}", fontsize=8)
        row[len(ALPHAS)].set_xticks([])
        row[len(ALPHAS)].set_yticks([])
        ax = row[-1]
        k = np.arange(1, fields.K_MAX + 1)
        for f, a, col in zip(out, ALPHAS, (SERIES[0], SERIES[0], INK, SERIES[1], SERIES[1])):
            spec = fields.shell_spectrum(fields.vorticity_hat(torch.as_tensor(f)[None, None].cuda(), "vorticity", g), g)[0].cpu().numpy()
            ax.loglog(k, spec, color=col, alpha=1.0 if abs(a) in (0, 2) else 0.45, label=f"α = {a:+g}" if a in (-2, 0, 2) else None)
        ax.set_title(f"σ_shape {m['sigma_shape']:.2f}, Δa {m['d_a']:+.2f}, Δs {m['d_s']:+.2f} per α (median, {s})", fontsize=8)
        ax.legend(fontsize=7)
    fig.suptitle("[BUILD] Stage 2b: M2-SW's PC2 in every seed, vorticity field (exploratory)", fontweight="bold", color=INK)
    fig.savefig(REPO / "results" / "build" / "stage2b_shape_dials.png", dpi=105)
    plt.close(fig)


if __name__ == "__main__":
    main()
