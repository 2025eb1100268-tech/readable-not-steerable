"""Figure 5 (exploratory, E): grid artefacts from sub-pixel upsampling under strong steering, beside the
resize-convolution decoder at the same dose (τ = 0.5), seed, frame and α.

- Models: the Step B (sub-pixel) and Step C (resize-convolution) exploration checkpoints, b3_sn_t0p5_s* and
  b4_rc_t0p5_s* (results/checkpoints/).
- Directions and σ_d: results/build/stepB.json.gz @ 5234145 and stepC.json.gz @ f0f6c1f (each model's own
  direction).
- Frame: the sub-pixel direction's worst explore frame, per the stored per-frame out-of-band share. The
  build's 256 explore steering frames are drawn with seed 0.
- α: for each decoder, the step with the largest out-of-band share for that frame. The sign of a direction
  is arbitrary per model, so each decoder is shown at its own worst case.
- Shares are computed on the raw decoder output, as in the Step B/C scoring, recomputed with dials/fields.py
  and printed against the stored value. Fields are displayed with the spatial mean removed and a per-field
  colour scale, so that the texture is visible.

    python results/paper/figures/fig5_artefacts.py
"""
import gzip
import json

import numpy as np
from matplotlib.patches import Rectangle

from _common import FIELD_CMAP, FULL_W, INK, INK2, REPO, check_parts, plt, save, style

PARTS = [["data/generated/v1", ["train", "val"]], ["data/generated/explore_v2", ["explore"]],
         ["data/generated/matched_energy_v2", ["matched"]]]
CACHE_TAG = "v1tv_explore_matched"
ALPHAS = np.arange(-2.0, 2.01, 0.5)
CASES = [  # (label, direction key path, seed)
    ("shape knob (loudness-null), seed 2", ("shape", "loudness-null"), 2),
    ("loudness direction, seed 0", ("loudness",), 0),
]
CROP = 16


def get(entry, path):
    for k in path:
        entry = entry[k]
    return entry


def main():
    import torch
    from data.combined import CombinedDataset
    from dials import fields
    from dials.adapters import adapter_for
    from dials.stage0 import frames
    from dials.stage1 import load
    from solver.ns2d import SpectralGrid
    check_parts(PARTS)
    sp = {r["run"]: r for r in json.load(gzip.open(REPO / "results/build/stepB.json.gz", "rt", encoding="utf-8"))["results"]["0.5"]}
    rc = {r["run"]: r for r in json.load(gzip.open(REPO / "results/build/stepC.json.gz", "rt", encoding="utf-8"))["results"]["0.5"]}
    ds = CombinedDataset(PARTS, CACHE_TAG)
    rows = frames(ds, "explore", 256, 4, 0)
    g = SpectralGrid(64, "cuda")
    style()
    fig = plt.figure(figsize=(FULL_W, 3.6))
    gs = fig.add_gridspec(len(CASES), 5, width_ratios=[1, 1, 0.12, 1, 1], wspace=0.06, hspace=0.32,
                          left=0.07, right=0.995, top=0.86, bottom=0.03)
    for i, (lab, path, seed) in enumerate(CASES):
        runs = {"sub-pixel": (f"b3_sn_t0p5_s{seed}", sp), "resize-convolution": (f"b4_rc_t0p5_s{seed}", rc)}
        stored_sp = get(sp[runs["sub-pixel"][0]], path)
        per_frame = np.array(stored_sp["explore"]["oob_share_worst_per_frame"])
        fi = int(np.argmax(per_frame))
        x1 = torch.from_numpy(ds.velocity[rows[fi:fi + 1]]).cuda() / ds.input_scale
        out = {}
        for name, (run, store) in runs.items():
            model = load(run)
            ad = adapter_for(model, "dial")
            dd = get(store[run], path)
            d = torch.as_tensor(dd["direction"], dtype=torch.float32, device="cuda")
            with torch.no_grad():
                fs = {float(a): ad.decode(None, x1, float(a * dd["sigma"]) * d)[0][2] for a in ALPHAS}
            share = {a: float(fields.out_of_band_share(f, g)[0]) for a, f in fs.items()}
            out[name] = (fs, share)
            del model
            torch.cuda.empty_cache()
        # each decoder at its own worst α on this frame (the sign of a direction is arbitrary per model)
        a_sp = max(out["sub-pixel"][1], key=out["sub-pixel"][1].get)
        a_rc = max(out["resize-convolution"][1], key=out["resize-convolution"][1].get)
        print(f"{lab}: frame {fi}; stored worst share {per_frame[fi]:.4g}, recomputed sub-pixel "
              f"{out['sub-pixel'][1][a_sp]:.4g} at α {a_sp:+g}; resize worst {out['resize-convolution'][1][a_rc]:.4g} at α {a_rc:+g}")
        alpha = {"sub-pixel": a_sp, "resize-convolution": a_rc}
        # displayed with the spatial mean removed and a per-field colour scale; shares are from the raw output
        f_sp = out["sub-pixel"][0][a_sp][0, 0].cpu().numpy()
        f_rc = out["resize-convolution"][0][a_rc][0, 0].cpu().numpy()
        f_sp, f_rc = f_sp - f_sp.mean(), f_rc - f_rc.mean()
        # crop where the sub-pixel field has the most power at the grid scale (2-pixel checkerboard)
        hp = np.abs(f_sp - 0.25 * (np.roll(f_sp, 1, 0) + np.roll(f_sp, -1, 0) + np.roll(f_sp, 1, 1) + np.roll(f_sp, -1, 1)))
        box = np.array([[hp[np.ix_(np.arange(r, r + CROP) % 64, np.arange(c, c + CROP) % 64)].sum() for c in range(64)]
                        for r in range(64)])
        r0, c0 = np.unravel_index(np.argmax(box), box.shape)
        r0, c0 = min(r0, 64 - CROP), min(c0, 64 - CROP)
        for j, (name, f) in enumerate((("sub-pixel", f_sp), ("resize-convolution", f_rc))):
            vmax = np.percentile(np.abs(f), 99.5)
            share = out[name][1][alpha[name]]
            col = 0 if j == 0 else 3
            axf, axz = fig.add_subplot(gs[i, col]), fig.add_subplot(gs[i, col + 1])
            axf.imshow(f, cmap=FIELD_CMAP, vmin=-vmax, vmax=vmax, origin="lower", interpolation="nearest")
            axf.add_patch(Rectangle((c0 - 0.5, r0 - 0.5), CROP, CROP, fill=False, ec=INK, lw=0.8))
            axz.imshow(f[r0:r0 + CROP, c0:c0 + CROP], cmap=FIELD_CMAP, vmin=-vmax, vmax=vmax, origin="lower",
                       interpolation="nearest")
            for ax in (axf, axz):
                ax.set_xticks([])
                ax.set_yticks([])
                ax.grid(False)
                for s in ax.spines.values():
                    s.set_visible(True)
                    s.set_color("#c3c2b7")
                    s.set_linewidth(0.5)
            axz.text(0.03, 0.03, "zoom", transform=axz.transAxes, fontsize=5.8, color=INK2, va="bottom")
            exp = int(np.floor(np.log10(share))) if share > 0 else 0
            axf.set_title(f"{name}, α = {alpha[name]:+g}: out-of-band share {share / 10 ** exp:.1f}×10$^{{{exp}}}$",
                          fontsize=6.4, loc="left")
        fig.text(0.005, 1 - (i + 0.5) / len(CASES) * 0.85 - 0.06, lab.replace(", ", "\n"), rotation=90, ha="left",
                 va="center", fontsize=6.6, color=INK)
    fig.text(0.07, 0.995, "τ = 0.5, same seed and frame; each decoder at its worst α for that frame.\nA frame fails the "
             "artefact criterion if more than 10⁻² of its power lies outside the dealiasing band.", fontsize=6.4,
             color=INK2, va="top")
    save(fig, "fig5_artefacts")


if __name__ == "__main__":
    main()
