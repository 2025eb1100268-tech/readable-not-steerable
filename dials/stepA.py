"""[BUILD] Step A (no training): symmetry-based loudness identification on M2-SW, and the grid-artefact
criterion applied retroactively to Stage 2b (results/build/PREDICTIONS.md entry 6). Exploratory.

    python -m dials.stepA --config configs/build/stepA.yaml

Loudness by symmetry: in the 8-dim dial space, the unit direction d whose steered output is closest to
a pure rescaling of the unsteered output. Objective, over unit d (sigma_d = sqrt(d' C d), C the v1-train
covariance of the dial vector), alpha in {-2, -1.5, ..., 2} without 0, and a set of explore frames
disjoint from the steering frames:
    J(d) = sum ||y_a - c y_0||^2 / sum ||y_a - y_0||^2,   y_a = decoded vorticity at z + a sigma_d d,
c fitted per frame and alpha (least squares). J is the fraction of the steered change that a rescaling
of each field does not explain (normalized by the change, so a direction that changes nothing does not
win trivially). Adam on an unnormalized vector, restarts from PC1-8 and random directions; the
restart with the lowest J on all optimization frames is kept. No physics labels are used.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from common.run import REPO, append_log, command_line, environment, load_config, seed_everything  # noqa: E402
from data.combined import CombinedDataset  # noqa: E402
from dials import fields  # noqa: E402
from dials.adapters import adapter_for  # noqa: E402
from dials.harness import classify, envelope, pca_directions, steer  # noqa: E402
from dials.stage0 import SETS, frames  # noqa: E402
from dials.stage1 import load  # noqa: E402
from solver.ns2d import SpectralGrid  # noqa: E402
from solver.validate import INK, SERIES, _style  # noqa: E402

OPT_ALPHAS = torch.tensor([-2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0])


@torch.no_grad()
def dial_latents(model, ds, rows, chunk=4096):
    ad = adapter_for(model, "dial")
    return np.concatenate([ad.latent(torch.from_numpy(ds.velocity[rows[i:i + chunk]]).cuda() / ds.input_scale).double().cpu().numpy()
                           for i in range(0, len(rows), chunk)])


def rescaling_objective(model, content, dial0, y0, d, cov):
    """J(d) on the given frames (differentiable in d)."""
    d = d / d.norm()
    sigma = torch.sqrt(d @ cov @ d)
    num = den = 0.0
    for a in OPT_ALPHAS.to(d.device):
        ya = model.head.dec(content, dial0 + a * sigma * d).float()
        c = (ya * y0).sum((1, 2, 3)) / (y0 * y0).sum((1, 2, 3))
        num = num + ((ya - c[:, None, None, None] * y0) ** 2).sum()
        den = den + ((ya - y0) ** 2).sum()
    return num / den


def find_loudness(model, enc_opt, cov, inits, cfg):
    """enc_opt = (content, dial0, y0) of the optimization frames. Returns one result per restart."""
    content, dial0, y0 = enc_opt
    gen = torch.Generator(device="cpu").manual_seed(cfg["seed"])
    results = []
    for name, init in inits.items():
        v = torch.as_tensor(init, dtype=torch.float32, device="cuda").clone().requires_grad_(True)
        opt = torch.optim.Adam([v], lr=cfg["opt_lr"])
        for _ in range(cfg["opt_steps"]):
            idx = torch.randperm(len(content), generator=gen)[:cfg["opt_batch"]].cuda()
            loss = rescaling_objective(model, content[idx], dial0[idx], y0[idx], v, cov)
            opt.zero_grad()
            loss.backward()
            opt.step()
        d = (v / v.norm()).detach()
        results.append({"init": name, "direction": d.cpu().numpy(), "J": _full_j(model, content, dial0, y0, d, cov)})
    return results


@torch.no_grad()
def _full_j(model, content, dial0, y0, d, cov):
    d = d / d.norm()
    sigma = torch.sqrt(d @ cov @ d)
    num = den = 0.0
    for i in range(0, len(content), 64):
        for a in OPT_ALPHAS.cuda():
            ya = model.head.dec(content[i:i + 64], dial0[i:i + 64] + a * sigma * d).float()
            yb = y0[i:i + 64]
            c = (ya * yb).sum((1, 2, 3)) / (yb * yb).sum((1, 2, 3))
            num += float(((ya - c[:, None, None, None] * yb) ** 2).sum())
            den += float(((ya - yb) ** 2).sum())
    return num / den


def artefact_fraction(m, crit):
    return float(np.mean(np.asarray(m["oob_share_worst_per_frame"]) > crit["artefact_oob_share"]))


def label(e):
    return f"{'D' if e['dial'] else '·'}{'L' if e['loudness'] else '·'}{'S' if e['shape'] else '·'}"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = load_config(ap.parse_args().config)
    seed_everything(cfg["seed"])
    t0 = time.perf_counter()
    ds = CombinedDataset(cfg["data_parts"], cfg["cache_tag"])
    g = SpectralGrid(64, "cuda")
    to_gpu = lambda rows: torch.from_numpy(ds.velocity[rows]).cuda() / ds.input_scale  # noqa: E731
    tr = ds.rows(["train"])
    fr = {s: frames(ds, s, cfg["n_frames"], 4, cfg["seed"]) for s in SETS}
    x = {s: (to_gpu(fr[s] - 1), to_gpu(fr[s])) for s in SETS}
    env = envelope(to_gpu(np.sort(np.random.default_rng(cfg["seed"]).choice(tr, cfg["n_envelope"], replace=False))), g)
    rest = np.setdiff1d(ds.rows(["explore"]), fr["explore"])
    x_opt = to_gpu(np.sort(np.random.default_rng(cfg["opt_seed"]).choice(rest, cfg["n_opt_frames"], replace=False)))
    crit_old, crit = cfg["criteria_stage2b"], {**cfg["criteria_stage2b"], **cfg["artefact"]}

    def steer_all(model, d, sigma, space="dial"):
        ad = adapter_for(model, space)
        e = {"direction": np.asarray(d).tolist(), "sigma": float(sigma)}
        for s in SETS:
            m = steer(ad, x[s][0], x[s][1], d, sigma, g, env)
            m.pop("heads", None)
            e[s] = {**m, **classify(m, crit), "entry2_only": classify(m, crit_old), "artefact_fraction": artefact_fraction(m, crit)}
        for k_ in ("dial", "loudness", "shape"):
            e[k_] = bool(e["explore"][k_] and e["matched"][k_])
            e[k_ + "_entry2_only"] = bool(e["explore"]["entry2_only"][k_] and e["matched"]["entry2_only"][k_])
        return e

    # ---- retroactive artefact check (labelled: calibrated on M2-SW seed 1) ------------------------------
    s2b = json.load(gzip.open(REPO / cfg["stage2b_json"], "rt", encoding="utf-8"))["results"]
    retro = {}
    for grp, runs in s2b.items():
        for r in runs:
            model = load(r["run"])
            for n, e_old in r["directions"].items():
                if e_old["space"] != "dial":
                    continue
                e = steer_all(model, np.asarray(e_old["direction"]), e_old["sigma"])
                retro[(grp, r["run"], n)] = {"before": label(e_old), "after": label(e), "entry2_only_now": label(
                    {k_: e[k_ + "_entry2_only"] for k_ in ("dial", "loudness", "shape")}),
                    "artefact_fraction": {s: e[s]["artefact_fraction"] for s in SETS}, "result": e}
            print(f"retro {grp} {r['run']} ({time.perf_counter() - t0:.0f} s)", flush=True)
    for run in cfg["stage2_m2sf_runs"]:  # Stage 2's M2-SF loudness dials (PC1), recomputed: their JSON keeps no direction vectors
        model = load(run)
        dial = dial_latents(model, ds, tr)
        dirs, sig, _ = pca_directions(dial, n=8)
        e = steer_all(model, dirs[0], sig[0])
        retro[("M2-SF (Stage 2)", run, "PC1")] = {"before": "DL·", "after": label(e), "entry2_only_now": label(
            {k_: e[k_ + "_entry2_only"] for k_ in ("dial", "loudness", "shape")}),
            "artefact_fraction": {s: e[s]["artefact_fraction"] for s in SETS}, "result": e}
        print(f"retro Stage 2 {run} ({time.perf_counter() - t0:.0f} s)", flush=True)

    # ---- Step A: loudness by symmetry on M2-SW ---------------------------------------------------------
    stepa = []
    for run in cfg["m2sw_runs"]:
        model = load(run)
        dial = dial_latents(model, ds, tr)
        cov = torch.as_tensor(np.cov(dial.T), dtype=torch.float32, device="cuda")
        dirs, sig, _ = pca_directions(dial, n=8)
        rng = np.random.default_rng(cfg["seed"])
        inits = {**{f"PC{i + 1}": dirs[i] for i in range(8)}, **{f"random{j}": rng.standard_normal(8) for j in range(cfg["n_random_inits"])}}
        with torch.no_grad():
            content_o, dial_o = model.enc(x_opt), model.head.dial(x_opt)
            enc_opt = (content_o, dial_o, model.head.dec(content_o, dial_o).float())
        restarts = find_loudness(model, enc_opt, cov, inits, cfg)
        best = min(restarts, key=lambda r_: r_["J"])
        d = best["direction"].astype(np.float64)
        d /= np.linalg.norm(d)
        sigma = float((dial @ d).std())
        j_pcs = {f"PC{i + 1}": _full_j(model, *enc_opt, torch.as_tensor(dirs[i], dtype=torch.float32, device="cuda"), cov) for i in range(8)}
        loud = steer_all(model, d, sigma)
        shape_key = next(k for k in retro if k[1] == run and k[2] == "PC2")
        shape = retro[shape_key]["result"]
        both = bool(loud["loudness"] and shape["shape"])
        stepa.append({"run": run, "J_best": best["J"], "init_of_best": best["init"], "J_restarts": {r_["init"]: r_["J"] for r_ in restarts},
                      "J_PCs": j_pcs, "direction": d.tolist(), "sigma": sigma,
                      "cos2_with_PCs": [float((d @ dirs[i]) ** 2) for i in range(8)],
                      "cos2_PC1_PC2_plane": float((d @ dirs[0]) ** 2 + (d @ dirs[1]) ** 2),
                      "loudness_direction": loud, "PC2": shape, "both_dials": both,
                      "both_dials_entry2_only": bool(loud["loudness_entry2_only"] and shape["shape_entry2_only"])})
        print(f"Step A {run}: J {best['J']:.3f} (from {best['init']}), loudness {label(loud)}, PC2 {label(shape)} ({time.perf_counter() - t0:.0f} s)",
              flush=True)
        stepa[-1]["_strip"] = strip(model, d, sigma, x["explore"][1][:1], g)
    wall = time.perf_counter() - t0
    write(cfg, retro, stepa, crit, wall)


@torch.no_grad()
def strip(model, d, sigma, x1, g):
    ad = adapter_for(model, "dial")
    dt = torch.as_tensor(d, dtype=torch.float32, device="cuda")
    out = {"fields": [], "spectra": []}
    for a in (-2.0, -1.0, 0.0, 1.0, 2.0):
        f = ad.decode(None, x1, float(a * sigma) * dt)[0][2]
        out["fields"].append(f[0, 0].cpu().numpy())
        out["spectra"].append(fields.shell_spectrum(fields.vorticity_hat(f, "vorticity", g), g)[0].cpu().numpy())
    return out


def write(cfg, retro, stepa, crit, wall):
    strips = [r.pop("_strip") for r in stepa]
    n_both = sum(r["both_dials"] for r in stepa)
    f2 = lambda v: f"{v:.2f}"  # noqa: E731
    L = ["# [BUILD] Step A: loudness by symmetry on M2-SW, and the grid-artefact criterion (exploratory)", "",
         "Definitions and predictions: results/build/PREDICTIONS.md entry 6. Criteria: entry 2 plus the grid-artefact criterion "
         f"(a direction fails on a dataset if more than {crit['artefact_max_frame_fraction']:.0%} of frames put more than "
         f"{crit['artefact_oob_share']:g} of the decoded vorticity's power outside the dealiasing band at some α).", "",
         "## Step A summary", "",
         "| seed | J (fraction of change not explained by rescaling): best / PC1 / PC2 | best found from | cos² with PC1 / PC2 / PC1–PC2 plane | "
         "loudness direction: D L S (entry 2 only) | its σ_shape explore / matched | effect explore / matched | artefact frames explore / matched | "
         "PC2: D L S (entry 2 only) | PC2 artefact frames | both dials valid? (entry 2 only) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in stepa:
        lo, sh = r["loudness_direction"], r["PC2"]
        L.append(f"| {r['run'][-1]} | {r['J_best']:.3f} / {r['J_PCs']['PC1']:.3f} / {r['J_PCs']['PC2']:.3f} | {r['init_of_best']} | "
                 f"{r['cos2_with_PCs'][0]:.2f} / {r['cos2_with_PCs'][1]:.2f} / {r['cos2_PC1_PC2_plane']:.2f} | "
                 f"{label(lo)} ({label({k_: lo[k_ + '_entry2_only'] for k_ in ('dial', 'loudness', 'shape')})}) | "
                 f"{f2(lo['explore']['sigma_shape'])} / {f2(lo['matched']['sigma_shape'])} | {lo['explore']['effect']:.3f} / {lo['matched']['effect']:.3f} | "
                 f"{lo['explore']['artefact_fraction']:.1%} / {lo['matched']['artefact_fraction']:.1%} | "
                 f"{label(sh)} ({label({k_: sh[k_ + '_entry2_only'] for k_ in ('dial', 'loudness', 'shape')})}) | "
                 f"{sh['explore']['artefact_fraction']:.1%} / {sh['matched']['artefact_fraction']:.1%} | "
                 f"{'**yes**' if r['both_dials'] else 'no'} ({'yes' if r['both_dials_entry2_only'] else 'no'}) |")
    L += ["", f"Both dials valid in the same seed (all criteria): {n_both} of {len(stepa)} seeds → M2-SW "
          f"{'meets' if n_both >= 2 else 'does not meet'} the both-dials condition (≥ 2 of 3), subject to the rendering check "
          "(which fired for M2-SW in Stage 2b).", ""]
    for s in ("explore", "matched"):
        L += [f"### Loudness direction, every metric — {s}", "",
              "| seed | σ_d | Δa | Δs | Δlog P | effect | σ_shape | monot. | cycle corr / gain | env − ideal | shared | artefact frames | fails |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in stepa:
            m = r["loudness_direction"][s]
            L.append(f"| {r['run'][-1]} | {r['sigma']:.3g} | {m['d_a']:+.3f} | {m['d_s']:+.3f} | {m['d_logP']:+.3f} | {m['effect']:.3f} | "
                     f"{f2(m['sigma_shape'])} | {f2(m['monotonicity'])} | {f2(m['cycle_corr'])} / {f2(m['cycle_gain'])} | "
                     f"{m['envelope_rel_min']:+.2f} | {f2(m['shared_pattern_fraction'])} | {m['artefact_fraction']:.1%} | {', '.join(m['fails']) or '—'} |")
        L.append("")
    L += ["## Grid-artefact criterion applied retroactively (calibrated on M2-SW seed 1's PC2 at α = −2)", "",
          "Only directions that were dials before, or are dials now, are listed; every direction is in the JSON.", "",
          "| model | run | dir | before (Stage 2b) | after (+ artefact) | artefact frames explore / matched |", "|---|---|---|---|---|---|"]
    for (grp, run, n), v in retro.items():
        if "D" in v["before"] or "D" in v["after"]:
            L.append(f"| {grp} | {run} | {n} | {v['before']} | {v['after']} | {v['artefact_fraction']['explore']:.1%} / "
                     f"{v['artefact_fraction']['matched']:.1%} |")
    sw = [v for (grp, run, n), v in retro.items() if grp == "M2-SW" and n.startswith("PC")]
    n_shape_after = len({run for (grp, run, n), v in retro.items() if grp == "M2-SW" and n.startswith("PC") and v["after"].endswith("S")})
    L += ["", f"M2-SW seeds with a PCA shape dial after the artefact criterion: {n_shape_after} of 3 (before: 2 of 3)."]
    with open(REPO / cfg["output_md"], "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    with gzip.open(REPO / cfg["output_json_gz"], "wt", encoding="utf-8") as f:
        json.dump({"environment": environment(), "config": cfg, "stepA": stepa,
                   "retro": [{"group": k[0], "run": k[1], "direction": k[2], **v} for k, v in retro.items()], "wall_seconds": wall},
                  f, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))
    _style()
    fig, axes = plt.subplots(len(stepa), 6, figsize=(18, 3.0 * len(stepa)), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1, 1, 1, 1, 1, 1.5]})
    for row, r, st in zip(np.atleast_2d(axes), stepa, strips):
        lim = float(np.abs(st["fields"][2]).max())
        for ax, f, a in zip(row[:5], st["fields"], (-2, -1, 0, 1, 2)):
            ax.imshow(f, cmap="RdBu_r", vmin=-lim, vmax=lim)
            ax.set_title((f"seed {r['run'][-1]}, loudness direction: " if a == -2 else "") + f"α = {a:+d}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
        k = np.arange(1, fields.K_MAX + 1)
        for spec, a, col in zip(st["spectra"], (-2, -1, 0, 1, 2), (SERIES[0], SERIES[0], INK, SERIES[1], SERIES[1])):
            row[5].loglog(k, spec, color=col, alpha=1.0 if a in (-2, 0, 2) else 0.45, label=f"α = {a:+d}" if a in (-2, 0, 2) else None)
        m = r["loudness_direction"]["explore"]
        row[5].set_title(f"J {r['J_best']:.3f}; Δa {m['d_a']:+.2f}, Δs {m['d_s']:+.2f} per α (explore)", fontsize=8)
        row[5].legend(fontsize=7)
    fig.suptitle("[BUILD] Step A: M2-SW's loudness direction found by symmetry, one explore frame (exploratory)", fontweight="bold", color=INK)
    fig.savefig(REPO / cfg["figure"], dpi=110)
    plt.close(fig)
    append_log("[BUILD] Step A: loudness by symmetry (M2-SW) and the grid-artefact criterion retroactively", cfg["_config_path"],
               command_line(), "ok", {"seeds with both dials valid (all criteria)": f"{n_both} of {len(stepa)}",
                                      "M2-SW seeds with a PCA shape dial after the artefact criterion": f"{n_shape_after} of 3",
                                      "results": f"{cfg['output_md']}, {cfg['figure']}, {cfg['output_json_gz']}"}, wall, environment().get("gpu"))
    print("\n".join(L[:12 + len(stepa)]))


if __name__ == "__main__":
    main()
