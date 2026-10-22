"""[BUILD] Calibration figure for the grid-artefact criterion (results/build/PREDICTIONS.md entry 6). Exploratory.

For M2-SW (seeds 0-2: PC2, and seed 1's ES3) and M2-SF with the omega-only loss (PC1), on the Stage 0 explore
steering frames: each frame's out-of-band power share (power outside the solver's square dealiasing band,
|kx| or |ky| > 21, over total power, of the decoded vorticity), maximised over alpha; the figure shows
the 4 worst frames per direction at their worst alpha and the frame closest to 1e-2. Calibrated on
seed 1's blocky output.

    python -m dials.artefact_calibration
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
from dials.adapters import adapter_for  # noqa: E402
from dials.fields import out_of_band_share  # noqa: E402
from dials.harness import ALPHAS  # noqa: E402
from dials.stage0 import frames  # noqa: E402
from dials.stage1 import load  # noqa: E402
from solver.ns2d import SpectralGrid  # noqa: E402
from solver.validate import INK, _style  # noqa: E402

CASES = [("M2-SW", 1, "PC2"), ("M2-SW", 1, "ES3"), ("M2-SW", 0, "PC2"), ("M2-SW", 2, "PC2"), ("M2-SF, omega-only loss", 0, "PC1")]


@torch.no_grad()
def main():
    cfg = load_config(str(REPO / "configs" / "build" / "stage2b_eval.yaml"))
    res = json.load(gzip.open(REPO / cfg["output_json_gz"], "rt", encoding="utf-8"))["results"]
    ds = CombinedDataset(cfg["data_parts"], cfg["cache_tag"])
    g = SpectralGrid(64, "cuda")
    x = torch.from_numpy(ds.velocity[frames(ds, "explore", cfg["n_frames"], 4, cfg["seed"])]).cuda() / ds.input_scale
    _style()
    fig, axes = plt.subplots(len(CASES), 5, figsize=(15, 3.1 * len(CASES)), constrained_layout=True)
    for row, (label, seed, name) in zip(axes, CASES):
        r = res[label][seed]
        e = r["directions"][name]
        ad = adapter_for(load(r["run"]), "dial")
        d = torch.as_tensor(e["direction"], dtype=torch.float32, device="cuda")
        fields_by_alpha = [ad.decode(None, x, float(a * e["sigma"]) * d)[0][2] for a in ALPHAS]
        share = np.stack([out_of_band_share(f, g).cpu().numpy() for f in fields_by_alpha])  # (alpha, frame)
        worst_alpha, worst = share.argmax(0), share.max(0)
        order = list(np.argsort(-worst)[:4]) + [int(np.argmin(np.abs(np.log(worst / 1e-2))))]
        for ax, i in zip(row, order):
            f = fields_by_alpha[worst_alpha[i]][i, 0].cpu().numpy()
            lim = float(np.abs(f).max())
            ax.imshow(f, cmap="RdBu_r", vmin=-lim, vmax=lim)
            ax.set_title(f"{label} s{seed} {name}, frame {i}, α {ALPHAS[worst_alpha[i]]:+g}\nout-of-band share {worst[i]:.1e}", fontsize=7)
            ax.set_xticks([])
            ax.set_yticks([])
        row[-1].set_title(row[-1].get_title() + " (closest to 1e-2)", fontsize=7)
    fig.suptitle("[BUILD] Artefact calibration: worst frames per direction (explore), each at its worst α (exploratory)",
                 fontweight="bold", color=INK)
    fig.savefig(REPO / "results" / "build" / "artefact_calibration.png", dpi=105)
    plt.close(fig)


if __name__ == "__main__":
    main()
