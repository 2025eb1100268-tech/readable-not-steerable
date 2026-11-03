"""Figure 7 (Study 3): both dials of the five fresh main models and of the controls, decoded on one exploration
frame, with energy spectra and the sealed-set verdict of each direction.

- Models: the frozen v3 checkpoints (results/checkpoints/b6_*.pt), each verified against its SHA-256 in
  configs/build/v3_frozen.json @ fc3b703 before loading.
- Directions, σ_d and pass/fail: the stored Stage 5 outputs, results/build/v3/confirm.json.gz @ 258d6af
  (committed outputs; the sealed sets themselves are spent and are not opened).
- Frame: the first of the build's 256 explore steering frames (seed 0), the frame used by the autopsy
  figure. Decoding follows the scoring: spatial mean removed (dials/v3_confirm.py ZeroMean).
- Outputs:
  - fig7_v3_dials.pdf: main seeds 100–104, plus one seed of each control (τ = 0 seed 200, untrained seed 300);
  - fig7b_v3_dials_all.pdf: all 11 models.

    python results/paper/figures/fig7_v3_dials.py
"""
import gzip
import hashlib
import json

import numpy as np

from _common import (FIELD_CMAP, FULL_W, INK, INK2, NEG1, NEG2, POS1, POS2, REPO, check_parts, plt, save,
                     style)

PARTS = [["data/generated/v1", ["train", "val"]], ["data/generated/explore_v2", ["explore"]],
         ["data/generated/matched_energy_v2", ["matched"]]]
CACHE_TAG = "v1tv_explore_matched"  # as configs/build/autopsy.yaml, so the frame is the same one
ALPHAS = (-2.0, -1.0, 0.0, 1.0, 2.0)
SHOW = (-2.0, 0.0, 2.0)
ACOL = {-2.0: NEG2, -1.0: NEG1, 0.0: INK, 1.0: POS1, 2.0: POS2}
DIRS = (("loudness", "loudness", "loudness"), ("shape (loudness-null)", "shape", "shape"))


def label(group, run):
    seed = run.rsplit("_s", 1)[1]
    return {"main": f"main\nseed {seed}", "tau0": f"τ = 0\ncontrol\nseed {seed}",
            "untrained": f"untrained\ncontrol\nseed {seed}"}[group]


def decode_all(runs, stored, x1, g, frozen):
    import torch
    from dials import fields
    from dials.adapters import adapter_for
    from dials.stage1 import load
    from dials.v3_confirm import ZeroMean
    out = {}
    for group, run in runs:
        ck = REPO / "results" / "checkpoints" / f"{run}.pt"
        h = hashlib.sha256(ck.read_bytes()).hexdigest()
        if h != frozen[run]["sha256"]:
            raise SystemExit(f"{run}: checkpoint SHA-256 does not match configs/build/v3_frozen.json")
        model = load(run)
        ad = ZeroMean(adapter_for(model, "dial"))
        e = stored[run]
        rows = {}
        for key, _, _ in DIRS:
            dd = e["directions"][key]
            d = torch.as_tensor(dd["direction"], dtype=torch.float32, device="cuda")
            with torch.no_grad():
                fs = {a: ad.decode(None, x1, float(a * dd["sigma"]) * d)[0][2] for a in ALPHAS}
            rows[key] = {"fields": {a: f[0, 0].cpu().numpy() for a, f in fs.items()},
                         "spectra": {a: fields.shell_spectrum(fields.vorticity_hat(f, "vorticity", g), g)[0].cpu().numpy()
                                     for a, f in fs.items()},
                         "valid": bool(dd[DIRS[0][2] if key == "loudness" else "shape"]),
                         "sigma": float(dd["sigma"])}
        out[run] = rows
        del model
        torch.cuda.empty_cache()
    return out


def draw(runs, dec, name):
    style()
    n = len(runs)
    H = 0.74 * n + 1.15
    fig = plt.figure(figsize=(FULL_W, H))
    gs = fig.add_gridspec(n, 10, width_ratios=[0.72, 1, 1, 1, 1.3, 0.16, 1, 1, 1, 1.3], wspace=0.08,
                          hspace=0.16, left=0.005, right=0.985, top=1 - 0.5 / H, bottom=0.78 / H)
    for i, (group, run) in enumerate(runs):
        ax = fig.add_subplot(gs[i, 0])
        ax.axis("off")
        ax.text(0.0, 0.5, label(group, run), fontsize=6.6, va="center", ha="left", color=INK, linespacing=1.15)
        for b, (key, short, _) in enumerate(DIRS):
            r = dec[run][key]
            c0 = 1 if b == 0 else 6
            vmax = max(np.percentile(np.abs(r["fields"][a]), 99.5) for a in SHOW)
            for j, a in enumerate(SHOW):
                axf = fig.add_subplot(gs[i, c0 + j])
                axf.imshow(r["fields"][a], cmap=FIELD_CMAP, vmin=-vmax, vmax=vmax, origin="lower", interpolation="nearest")
                axf.set_xticks([])
                axf.set_yticks([])
                for s in axf.spines.values():
                    s.set_visible(True)
                    s.set_color("#c3c2b7")
                    s.set_linewidth(0.5)
                if i == 0:
                    axf.set_title(f"{short}\nα = {a:+g}" if j == 1 else f"\nα = {a:+g}", fontsize=7)
                if j == 1:
                    ok = r["valid"]
                    axf.text(0.5, -0.06, ("✓ valid dial" if ok else "✗ not a dial"), transform=axf.transAxes,
                             ha="center", va="top", fontsize=6.2, color=INK if ok else "#c23b22", fontweight="bold")
            axs = fig.add_subplot(gs[i, c0 + 3])
            k = np.arange(1, len(r["spectra"][0.0]) + 1)
            for a in ALPHAS:
                axs.loglog(k, r["spectra"][a], color=ACOL[a], lw=1.0 if a else 1.3)
            axs.set_xlim(1, k[-1])
            axs.tick_params(labelsize=5.5, length=1.5, pad=1)
            axs.grid(False)
            axs.set_yticklabels([])
            axs.minorticks_off()
            if i == n - 1:
                axs.set_xlabel("wavenumber k", fontsize=6.3, labelpad=1)
            else:
                axs.set_xticklabels([])
            if i == 0:
                axs.set_title("energy\nspectrum", fontsize=7)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=ACOL[a], lw=1.3, label=f"α = {a:+g}") for a in ALPHAS]
    fig.legend(handles=handles, loc="lower right", ncol=5, fontsize=6.5, bbox_to_anchor=(0.99, 0.0),
               title="spectra (log–log): steering amplitude α in units of σ_d", title_fontsize=6.5)
    fig.text(0.005, 0.004, "Fields: decoded vorticity, spatial mean removed;\ncolour scale shared within each block of three.\n"
             "✓/✗: verdict on both sealed sets\n(stored Stage 5 outputs).", fontsize=6, color=INK2, va="bottom")
    save(fig, name)


def main():
    import torch
    from data.combined import CombinedDataset
    from dials.stage0 import frames
    from solver.ns2d import SpectralGrid
    check_parts(PARTS)
    frozen = json.load(open(REPO / "configs" / "build" / "v3_frozen.json", encoding="utf-8"))["runs"]
    stage5 = json.load(gzip.open(REPO / "results" / "build" / "v3" / "confirm.json.gz", "rt", encoding="utf-8"))["results"]
    stored = {e["run"]: e for rs in stage5.values() for e in rs}
    ds = CombinedDataset(PARTS, CACHE_TAG)
    row = frames(ds, "explore", 256, 4, 0)[:1]
    x1 = torch.from_numpy(ds.velocity[row]).cuda() / ds.input_scale
    g = SpectralGrid(64, "cuda")
    groups = [(grp, e["run"]) for grp, rs in stage5.items() for e in rs]
    dec = decode_all(groups, stored, x1, g, frozen)
    main_runs = [r for r in groups if r[0] == "main"] + [("tau0", "b6_tau0_s200"), ("untrained", "b6_untrained_s300")]
    draw(main_runs, dec, "fig7_v3_dials")
    draw(groups, dec, "fig7b_v3_dials_all")
    print("frame row", int(row[0]))


if __name__ == "__main__":
    main()
