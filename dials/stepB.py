"""[BUILD] Step B readouts: shape-noise dose (results/build/PREDICTIONS.md entry 8). Exploratory.

    python -m dials.stepB --config configs/build/stepB_eval.yaml

Per model: rendering (and the trigger); the loudness direction by symmetry (Step A); three shape rules:
PC2 of the dial vector (primary, frozen), the projected-out rule of Step A′, and loudness-null PCA;
each steered with the full harness on the standard explore / matched-energy frames under entry 2's
criteria plus the artefact criterion; both-dials per rule. Loudness-null PCA (entry 8): z-score the
dial vector (v1 train), g = the frame-mean finite-difference gradient of the decoded field's a = log E
with respect to the z-scored dial vector (central differences at +-0.5 z-units, i.e. half a natural
std, per coordinate, on the Step A optimization frames), restrict to the subspace orthogonal to g,
take the top PC of the projected data, map to a raw direction (s * u). Also applied to the M2-SW
checkpoints (analysis only). Readouts are plotted against the calibrated dose.
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
from dials.split import TARGETS  # noqa: E402
from dials.stage0 import SETS, frames  # noqa: E402
from dials.stage1 import load, references  # noqa: E402
from dials.stepA import _full_j, artefact_fraction, dial_latents, find_loudness, label  # noqa: E402
from dials.stepA_prime import shape_after_loudness  # noqa: E402
from solver.ns2d import SpectralGrid  # noqa: E402
from solver.validate import INK, INK2, SERIES, _style  # noqa: E402

RULES = ("PC2", "projected-out", "loudness-null")


@torch.no_grad()
def loudness_null(model, G: np.ndarray, x_opt: torch.Tensor, g, step: float = 0.5):
    """Entry 8: (raw unit direction, z-direction, frame-mean gradient g_z, spread of per-frame gradients)."""
    mu, s = G.mean(0), G.std(0)
    content, dial0 = model.enc(x_opt), model.head.dial(x_opt)
    a = lambda dial: fields.stats(fields.vorticity_hat(model.head.dec(content, dial).double() * model.ae_scales["vorticity"],  # noqa: E731
                                                       "vorticity", g), g)["a"]
    grads = []
    for j in range(G.shape[1]):
        e = torch.zeros(G.shape[1], device="cuda")
        e[j] = step * s[j]
        grads.append(((a(dial0 + e) - a(dial0 - e)) / (2 * step)).cpu().numpy())
    gf = np.stack(grads, 1)  # (frames, 8): d a / d z
    gm = gf.mean(0)
    cos = gf @ gm / (np.linalg.norm(gf, axis=1) * np.linalg.norm(gm))
    Z = (G - mu) / s
    gh = gm / np.linalg.norm(gm)
    Zp = Z - np.outer(Z @ gh, gh)
    u = np.linalg.svd(Zp - Zp.mean(0), full_matrices=False)[2][0]
    d = s * u
    return d / np.linalg.norm(d), u, gm, {"median_cos_frame_vs_mean": float(np.median(cos)), "p10_cos": float(np.percentile(cos, 10)),
                                          "rel_spread": float(np.median(np.linalg.norm(gf - gm, axis=1)) / np.linalg.norm(gm))}


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
    crit_old, crit = cfg["criteria"], {**cfg["criteria"], **cfg["artefact"]}
    refs = references(cfg)

    def steer_one(model, d, sigma):
        ad = adapter_for(model, "dial")
        e = {"direction": np.asarray(d).tolist(), "sigma": float(sigma)}
        for s in SETS:
            m = steer(ad, x[s][0], x[s][1], d, sigma, g, env)
            m.pop("heads", None)
            e[s] = {**m, **classify(m, crit), "entry2_only": classify(m, crit_old), "artefact_fraction": artefact_fraction(m, crit)}
        for k_ in ("dial", "loudness", "shape"):
            e[k_] = bool(e["explore"][k_] and e["matched"][k_])
            e[k_ + "_entry2_only"] = bool(e["explore"]["entry2_only"][k_] and e["matched"]["entry2_only"][k_])
        return e

    def analyse(run, with_loudness_search=True, d_l=None):
        model = load(run)
        G = dial_latents(model, ds, tr)
        pcs, sig, expl = pca_directions(G, n=8)
        out = {"run": run, "dial_explained": expl.tolist()}
        if with_loudness_search:
            cov = torch.as_tensor(np.cov(G.T), dtype=torch.float32, device="cuda")
            with torch.no_grad():
                c_o, d_o = model.enc(x_opt), model.head.dial(x_opt)
                enc_opt = (c_o, d_o, model.head.dec(c_o, d_o).float())
            rng = np.random.default_rng(cfg["seed"])
            inits = {**{f"PC{i + 1}": pcs[i] for i in range(8)}, **{f"random{j}": rng.standard_normal(8) for j in range(cfg["n_random_inits"])}}
            best = min(find_loudness(model, enc_opt, cov, inits, cfg), key=lambda r_: r_["J"])
            d_l = best["direction"].astype(np.float64) / np.linalg.norm(best["direction"])
            out.update({"J_loudness": best["J"], "J_PC1": _full_j(model, *enc_opt, torch.as_tensor(pcs[0], dtype=torch.float32, device="cuda"), cov)})
            out["loudness"] = steer_one(model, d_l, (G @ d_l).std())
        d_p, _, _ = shape_after_loudness(G, d_l)
        d_n, _, gm, spread = loudness_null(model, G, x_opt, g)
        out["loudness_gradient"] = {"g_z": gm.tolist(), **spread}
        out["shape"] = {"PC2": steer_one(model, pcs[1], sig[1]), "projected-out": steer_one(model, d_p, (G @ d_p).std()),
                        "loudness-null": steer_one(model, d_n, (G @ d_n).std())}
        out["cos2"] = {"null_vs_PC2": float((d_n @ pcs[1]) ** 2), "null_vs_projected": float((d_n @ d_p) ** 2),
                       "null_vs_loudness": float((d_n @ d_l) ** 2)}
        return out

    res = {}
    for tau, runs in cfg["groups"].items():
        res[tau] = []
        for run in runs:
            r = analyse(run)
            j = json.loads((REPO / cfg["runs_json"].format(run=run)).read_text())
            fin, dg = j["final"], j["diagnostics"]
            m, k10 = fin["metric_vorticity"], dg["vorticity_pred_over_true_power_k_ge_10"]
            r["rendering"] = {"omega": m, "k10": k10, "psi_derived": fin["metric_streamfunction"],
                              "trigger": [lbl for lbl, bad in (("> 2× VICReg AE", m > 2 * refs["vorticity"]["vicreg"]), ("> 0.02", m > 0.02),
                                                               ("< 50% small-scale power", k10 < 0.5)) if bad]}
            r["both"] = {rule: bool(r["loudness"]["loudness"] and r["shape"][rule]["shape"]) for rule in RULES}
            r["both_entry2_only"] = {rule: bool(r["loudness"]["loudness_entry2_only"] and r["shape"][rule]["shape_entry2_only"]) for rule in RULES}
            res[tau].append(r)
            print(f"tau {tau} {run}: loudness {label(r['loudness'])}, shape PC2 / proj / null: "
                  + " / ".join(label(r["shape"][k]) for k in RULES) + f", omega {m:.4f} ({time.perf_counter() - t0:.0f} s)", flush=True)
    sw = json.load(gzip.open(REPO / cfg["stepA_json"], "rt", encoding="utf-8"))["stepA"]
    m2sw = []
    for a in sw:  # M2-SW reference point (analysis only): Step A's loudness direction, the three shape rules
        r = analyse(a["run"], with_loudness_search=False, d_l=np.asarray(a["direction"]))
        r["loudness"] = a["loudness_direction"]
        r["both"] = {rule: bool(a["loudness_direction"]["loudness"] and r["shape"][rule]["shape"]) for rule in RULES}
        m2sw.append(r)
        print(f"M2-SW {a['run']}: shape PC2 / proj / null: " + " / ".join(label(r["shape"][k]) for k in RULES), flush=True)
    wall = time.perf_counter() - t0
    write(cfg, res, m2sw, wall)


def write(cfg, res, m2sw, wall):
    dose = {float(k): v for k, v in cfg["calibrated_dose"].items()}
    f2 = lambda v: f"{v:.2f}"  # noqa: E731
    L = ["# [BUILD] Step B: shape-noise dose on M2-SF-style content (exploratory)", "",
         "Definitions and predictions: results/build/PREDICTIONS.md entry 8. Criteria: entry 2 plus the grid-artefact criterion. "
         "Calibrated dose = ridge R² of s from the content input's log shell powers (results/build/stepB_calibration.md). "
         "Shape rule: PC2 (primary, frozen); the projected-out rule and loudness-null PCA are secondary, chosen after seeing results.", "",
         "## Per dose (counts over 3 seeds; all criteria)", "",
         "| τ | content shape R² | loudness dial (symmetry) | shape dial: PC2 / projected-out / loudness-null | both dials: PC2 / projected-out / loudness-null | "
         "median σ_shape explore: PC2 / projected-out / loudness-null | ω val MSE (median) | rendering triggers |", "|---|---|---|---|---|---|---|---|"]
    for tau, rs in res.items():
        t = float(tau)
        cnt = lambda f: sum(bool(f(r)) for r in rs)  # noqa: E731
        med = lambda rule: np.median([r["shape"][rule]["explore"]["sigma_shape"] for r in rs])  # noqa: E731
        L.append(f"| {t:g} | {dose[t]:.3f} | {cnt(lambda r: r['loudness']['loudness'])} | "
                 + " / ".join(str(cnt(lambda r, k=k: r["shape"][k]["shape"])) for k in RULES) + " | "
                 + " / ".join(str(cnt(lambda r, k=k: r["both"][k])) for k in RULES) + " | "
                 + " / ".join(f2(med(k)) for k in RULES) + f" | {np.median([r['rendering']['omega'] for r in rs]):.4f} | "
                 + f"{cnt(lambda r: r['rendering']['trigger'])} of 3 |")
    L += ["", "## Every model", "",
          "| τ | seed | ω MSE / k≥10 kept / triggers | loudness: D L S, σ_shape, effect (explore), artefact frames | "
          "PC2: D L S, σ_shape explore / matched | projected-out: D L S, σ_shape | loudness-null: D L S, σ_shape | cos² null vs PC2 | "
          "loudness gradient: median cos(frame, mean) |", "|---|---|---|---|---|---|---|---|---|"]
    for tau, rs in res.items():
        for r in rs:
            lo, sh = r["loudness"], r["shape"]
            fmt = lambda e: f"{label(e)}, {f2(e['explore']['sigma_shape'])} / {f2(e['matched']['sigma_shape'])}"  # noqa: E731
            rd = r["rendering"]
            L.append(f"| {float(tau):g} | {r['run'][-1]} | {rd['omega']:.4f} / {rd['k10']:.2f} / {', '.join(rd['trigger']) or 'none'} | "
                     f"{label(lo)}, {f2(lo['explore']['sigma_shape'])}, {lo['explore']['effect']:.2f}, "
                     f"{lo['explore']['artefact_fraction']:.1%} / {lo['matched']['artefact_fraction']:.1%} | {fmt(sh['PC2'])} | {fmt(sh['projected-out'])} | "
                     f"{fmt(sh['loudness-null'])} | {f2(r['cos2']['null_vs_PC2'])} | {f2(r['loudness_gradient']['median_cos_frame_vs_mean'])} |")
    L += ["", "## M2-SW reference (full whitening; analysis only; Step A's loudness direction)", "",
          "| seed | PC2 | projected-out | loudness-null | both dials: PC2 / projected-out / loudness-null | cos² null vs PC2 | loudness gradient cos |",
          "|---|---|---|---|---|---|---|"]
    for r in m2sw:
        sh = r["shape"]
        fmt = lambda e: f"{label(e)} ({f2(e['explore']['sigma_shape'])} / {f2(e['matched']['sigma_shape'])})"  # noqa: E731
        L.append(f"| {r['run'][-1]} | {fmt(sh['PC2'])} | {fmt(sh['projected-out'])} | {fmt(sh['loudness-null'])} | "
                 + " / ".join("yes" if r["both"][k] else "no" for k in RULES) + f" | {f2(r['cos2']['null_vs_PC2'])} | "
                 f"{f2(r['loudness_gradient']['median_cos_frame_vs_mean'])} |")
    for s in SETS:
        L += ["", f"## Shape-rule directions, every metric — {s}", "",
              "| τ | seed | rule | σ_d | Δa | Δs | effect | σ_shape | monot. | cycle corr / gain | env − ideal | shared | artefact frames | fails |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for tau, rs in list(res.items()) + [("M2-SW", m2sw)]:
            for r in rs:
                for rule in RULES + ("loudness",):
                    e = r["loudness"] if rule == "loudness" else r["shape"][rule]
                    if "explore" not in e or s not in e:
                        continue
                    m = e[s]
                    t = tau if tau == "M2-SW" else f"{float(tau):g}"
                    L.append(f"| {t} | {r['run'][-1]} | {rule} | {e['sigma']:.3g} | {m['d_a']:+.3f} | {m['d_s']:+.3f} | {m['effect']:.3f} | "
                             f"{f2(m['sigma_shape'])} | {f2(m['monotonicity'])} | {f2(m['cycle_corr'])} / {f2(m['cycle_gain'])} | "
                             f"{m['envelope_rel_min']:+.2f} | {f2(m['shared_pattern_fraction'])} | {m['artefact_fraction']:.1%} | {', '.join(m['fails']) or '—'} |")
    with open(REPO / cfg["output_md"], "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    with gzip.open(REPO / cfg["output_json_gz"], "wt", encoding="utf-8") as f:
        json.dump({"environment": environment(), "config": cfg, "results": res, "m2sw": m2sw, "wall_seconds": wall}, f, default=float)

    # ---- figure: every readout against the calibrated dose ----
    _style()
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.6), constrained_layout=True)
    xs = {tau: dose[float(tau)] for tau in res}
    for i, rule in enumerate(RULES):
        for tau, rs in res.items():
            ys = [r["shape"][rule]["explore"]["sigma_shape"] for r in rs]
            axes[0].scatter([xs[tau] + 0.006 * (i - 1)] * len(ys), ys, color=SERIES[i], s=22, label=rule if tau == list(res)[0] else None)
        axes[0].plot([xs[t] for t in res], [np.median([r["shape"][rule]["explore"]["sigma_shape"] for r in rs]) for rs in res.values()], color=SERIES[i])
    axes[0].axhline(cfg["criteria"]["min_sigma_shape"], color=INK2, linestyle=(0, (3, 2)), linewidth=1)
    axes[0].set_ylabel("σ_shape of the shape-rule direction (explore)")
    axes[0].legend(fontsize=7)
    for i, rule in enumerate(RULES):
        axes[1].plot([xs[t] for t in res], [sum(r["shape"][rule]["shape"] for r in rs) for rs in res.values()], marker="o", color=SERIES[i], label=f"shape dial: {rule}")
    axes[1].plot([xs[t] for t in res], [sum(r["loudness"]["loudness"] for r in rs) for rs in res.values()], marker="s", color=INK, label="loudness dial (symmetry)")
    axes[1].set_ylabel("seeds (of 3) with a valid dial (all criteria)")
    axes[1].set_ylim(-0.2, 3.2)
    axes[1].legend(fontsize=7)
    for tau, rs in res.items():
        axes[2].scatter([xs[tau]] * len(rs), [r["rendering"]["omega"] for r in rs], color=SERIES[0], s=22)
    axes[2].axhline(0.02, color=INK2, linestyle=(0, (3, 2)), linewidth=1)
    axes[2].set_ylabel("ω val normalized MSE")
    for tau, rs in res.items():
        axes[3].scatter([xs[tau]] * len(rs), [r["rendering"]["k10"] for r in rs], color=SERIES[1], s=22)
    axes[3].axhline(0.5, color=INK2, linestyle=(0, (3, 2)), linewidth=1)
    axes[3].set_ylabel("ω power kept at k ≥ 10")
    for ax in axes:
        ax.set_xlabel("content shape R² (calibrated dose; τ = " + ", ".join(f"{float(t):g}" for t in res) + ")")
        ax.invert_xaxis()
    fig.suptitle("[BUILD] Step B: readouts against the calibrated shape dose (less shape in content → right) (exploratory)",
                 fontweight="bold", color=INK)
    fig.savefig(REPO / cfg["figure"], dpi=115)
    plt.close(fig)
    append_log("[BUILD] Step B readouts (shape-noise dose)", cfg["_config_path"], command_line(), "ok",
               {f"tau {float(t):g} (content shape R² {dose[float(t)]:.3f})": "shape dial PC2 / proj / null: "
                + " / ".join(str(sum(r['shape'][k]['shape'] for r in rs)) for k in RULES)
                + f"; loudness {sum(r['loudness']['loudness'] for r in rs)}; both (PC2) {sum(r['both']['PC2'] for r in rs)}"
                for t, rs in res.items()} | {"results": f"{cfg['output_md']}, {cfg['figure']}, {cfg['output_json_gz']}"},
               wall, environment().get("gpu"))
    print("\n".join(L[:12]))


if __name__ == "__main__":
    main()
