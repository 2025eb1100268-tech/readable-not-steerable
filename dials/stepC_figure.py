"""[BUILD] Step C figure: both dials of each resize-convolution model at tau = 0.5 (the symmetry loudness
direction and the loudness-null shape direction), decoded at alpha = -2..+2 on one explore and one
matched-energy steering frame, with shell spectra. Directions and sigma_d from results/build/stepC.json.gz.
Exploratory; no new measurement.

    python -m dials.stepC_figure
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
    cfg = load_config(str(REPO / "configs" / "build" / "stepC_eval.yaml"))
    res = json.load(gzip.open(REPO / cfg["output_json_gz"], "rt", encoding="utf-8"))["results"]["0.5"]
    ds = CombinedDataset(cfg["data_parts"], cfg["cache_tag"])
    g = SpectralGrid(64, "cuda")
    xs = {s: torch.from_numpy(ds.velocity[frames(ds, s, cfg["n_frames"], 4, cfg["seed"])[:1]]).cuda() / ds.input_scale
          for s in ("explore", "matched")}
    rows = [(r, name, key, s) for r in res for name, key in (("loudness", None), ("shape (loudness-null)", "loudness-null")) for s in xs]
    _style()
    fig, axes = plt.subplots(len(rows), len(ALPHAS) + 1, figsize=(17, 2.55 * len(rows)), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1] * len(ALPHAS) + [1.5]})
    models = {}
    for row, (r, name, key, s) in zip(axes, rows):
        e = r["loudness"] if key is None else r["shape"][key]
        if r["run"] not in models:
            models[r["run"]] = adapter_for(load(r["run"]), "dial")
        d = torch.as_tensor(e["direction"], dtype=torch.float32, device="cuda")
        out = [models[r["run"]].decode(None, xs[s], float(a * e["sigma"]) * d)[0][2] for a in ALPHAS]
        lim = float(out[2].abs().max())
        for ax, f, a in zip(row, out, ALPHAS):
            ax.imshow(f[0, 0].cpu().numpy(), cmap="RdBu_r", vmin=-lim, vmax=lim)
            ax.set_title((f"seed {r['run'][-1]}, {name}, {s}: " if a == -2 else "") + f"α = {a:+g}", fontsize=7)
            ax.set_xticks([])
            ax.set_yticks([])
        k = np.arange(1, fields.K_MAX + 1)
        for f, a, col in zip(out, ALPHAS, (SERIES[0], SERIES[0], INK, SERIES[1], SERIES[1])):
            spec = fields.shell_spectrum(fields.vorticity_hat(f, "vorticity", g), g)[0].cpu().numpy()
            row[-1].loglog(k, spec, color=col, alpha=1.0 if a in (-2, 0, 2) else 0.45, label=f"α = {a:+g}" if a in (-2, 0, 2) else None)
        m = e[s]
        row[-1].set_title(f"Δa {m['d_a']:+.2f}, Δs {m['d_s']:+.2f} per α (median), σ_shape {m['sigma_shape']:.2f}", fontsize=7)
        row[-1].legend(fontsize=6)
    fig.suptitle("[BUILD] Step C: both dials of the resize-convolution models at τ = 0.5, vorticity field (exploratory)",
                 fontweight="bold", color=INK)
    fig.savefig(REPO / "results" / "build" / "stepC_dials.png", dpi=95)
    plt.close(fig)


if __name__ == "__main__":
    main()
