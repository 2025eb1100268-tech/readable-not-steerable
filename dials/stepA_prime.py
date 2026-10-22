"""[BUILD] Step A′ (no training): shape dial = the top PC of the z-scored dial space after projecting out
the symmetry-loudness direction of Step A (results/build/PREDICTIONS.md entry 7). Exploratory.

    python -m dials.stepA_prime --config configs/build/stepA.yaml
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import time

import numpy as np
import torch

from common.run import REPO, append_log, command_line, environment, load_config, seed_everything
from data.combined import CombinedDataset
from dials.adapters import adapter_for
from dials.harness import classify, envelope, pca_directions, steer
from dials.stage0 import SETS, frames
from dials.stage1 import load
from dials.stepA import artefact_fraction, dial_latents, label
from solver.ns2d import SpectralGrid


def shape_after_loudness(G: np.ndarray, d_l: np.ndarray):
    """Entry 7 steps 1-5. Returns (raw unit direction d_S, z-space u, share of Z's variance along u)."""
    mu, s = G.mean(0), G.std(0)
    Z = (G - mu) / s
    l = (d_l / s) / np.linalg.norm(d_l / s)
    Zp = Z - np.outer(Z @ l, l)
    Zc = Zp - Zp.mean(0)
    _, sv, vt = np.linalg.svd(Zc, full_matrices=False)
    u = vt[0]
    d_s = s * u
    return d_s / np.linalg.norm(d_s), u, float(((Z - Z.mean(0)) @ u).var() / (Z - Z.mean(0)).var(0).sum())


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
    crit_old, crit = cfg["criteria_stage2b"], {**cfg["criteria_stage2b"], **cfg["artefact"]}
    stepa = {r["run"]: r for r in json.load(gzip.open(REPO / cfg["output_json_gz"], "rt", encoding="utf-8"))["stepA"]}
    rows = []
    for run in cfg["m2sw_runs"]:
        model = load(run)
        G = dial_latents(model, ds, tr)
        pcs, _, _ = pca_directions(G, n=8)
        a = stepa[run]
        d_l = np.asarray(a["direction"])
        d_s, u, share = shape_after_loudness(G, d_l)
        sigma = float((G @ d_s).std())
        ad = adapter_for(model, "dial")
        e = {"direction": d_s.tolist(), "sigma": sigma, "z_direction": u.tolist(), "z_variance_share": share,
             "cos2_PC2": float((d_s @ pcs[1]) ** 2), "cos2_loudness": float((d_s @ d_l) ** 2)}
        for s in SETS:
            m = steer(ad, x[s][0], x[s][1], d_s, sigma, g, env)
            m.pop("heads", None)
            e[s] = {**m, **classify(m, crit), "entry2_only": classify(m, crit_old), "artefact_fraction": artefact_fraction(m, crit)}
        for k_ in ("dial", "loudness", "shape"):
            e[k_] = bool(e["explore"][k_] and e["matched"][k_])
            e[k_ + "_entry2_only"] = bool(e["explore"]["entry2_only"][k_] and e["matched"]["entry2_only"][k_])
        loud, pc2 = a["loudness_direction"], a["PC2"]
        rows.append({"run": run, "new_rule": e,
                     "both_new_rule": bool(loud["loudness"] and e["shape"]),
                     "both_new_rule_entry2_only": bool(loud["loudness_entry2_only"] and e["shape_entry2_only"]),
                     "both_PC2_rule": bool(loud["loudness"] and pc2["shape"]),
                     "both_PC2_rule_entry2_only": bool(loud["loudness_entry2_only"] and pc2["shape_entry2_only"]),
                     "loudness": {k_: loud[k_] for k_ in ("dial", "loudness", "shape", "loudness_entry2_only")},
                     "PC2": {k_: pc2[k_] for k_ in ("dial", "loudness", "shape", "shape_entry2_only")}})
        print(f"{run}: new rule {label(e)}, cos2 PC2 {e['cos2_PC2']:.2f} ({time.perf_counter() - t0:.0f} s)", flush=True)
    wall = time.perf_counter() - t0
    f2 = lambda v: f"{v:.2f}"  # noqa: E731
    L = ["# [BUILD] Step A′: shape dial after projecting out the symmetry-loudness direction (M2-SW; exploratory)", "",
         "Procedure and predictions: results/build/PREDICTIONS.md entry 7. Criteria: entry 2 plus the grid-artefact criterion. "
         "D L S = dial / loudness / shape on both datasets (all criteria); in brackets, entry 2's criteria alone.", "",
         "| seed | new rule: D L S | σ_shape explore / matched | effect explore / matched | artefact frames | cos² with PC2 / with loudness | "
         "z-variance share | PC2 rule: D L S | loudness (Step A) | both dials, new rule | both dials, PC2 rule |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    yn = lambda b, b2: f"{'**yes**' if b else 'no'} ({'yes' if b2 else 'no'})"  # noqa: E731
    for r in rows:
        e = r["new_rule"]
        L.append(f"| {r['run'][-1]} | {label(e)} ({label({k_: e[k_ + '_entry2_only'] for k_ in ('dial', 'loudness', 'shape')})}) | "
                 f"{f2(e['explore']['sigma_shape'])} / {f2(e['matched']['sigma_shape'])} | {e['explore']['effect']:.3f} / {e['matched']['effect']:.3f} | "
                 f"{e['explore']['artefact_fraction']:.1%} / {e['matched']['artefact_fraction']:.1%} | {f2(e['cos2_PC2'])} / {f2(e['cos2_loudness'])} | "
                 f"{f2(e['z_variance_share'])} | {label(r['PC2'])} | {label(r['loudness'])} | "
                 f"{yn(r['both_new_rule'], r['both_new_rule_entry2_only'])} | {yn(r['both_PC2_rule'], r['both_PC2_rule_entry2_only'])} |")
    n_new = sum(r["both_new_rule"] for r in rows)
    n_pc2 = sum(r["both_PC2_rule"] for r in rows)
    L += ["", f"Both dials valid in the same seed (all criteria): new rule {n_new} of 3, PC2 rule {n_pc2} of 3. "
          f"Shape dial (all criteria): new rule {sum(r['new_rule']['shape'] for r in rows)} of 3, PC2 rule {sum(r['PC2']['shape'] for r in rows)} of 3.", ""]
    for s in SETS:
        L += [f"### New-rule direction, every metric — {s}", "",
              "| seed | σ_d | Δa | Δs | Δlog P | effect | σ_shape | monot. | cycle corr / gain | env − ideal | shared | artefact frames | fails |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            e, m = r["new_rule"], r["new_rule"][s]
            L.append(f"| {r['run'][-1]} | {e['sigma']:.3g} | {m['d_a']:+.3f} | {m['d_s']:+.3f} | {m['d_logP']:+.3f} | {m['effect']:.3f} | "
                     f"{f2(m['sigma_shape'])} | {f2(m['monotonicity'])} | {f2(m['cycle_corr'])} / {f2(m['cycle_gain'])} | "
                     f"{m['envelope_rel_min']:+.2f} | {f2(m['shared_pattern_fraction'])} | {m['artefact_fraction']:.1%} | {', '.join(m['fails']) or '—'} |")
        L.append("")
    with open(REPO / cfg["output_md_prime"], "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    with gzip.open(REPO / cfg["output_json_gz_prime"], "wt", encoding="utf-8") as f:
        json.dump({"environment": environment(), "config": cfg, "rows": rows, "wall_seconds": wall}, f, default=float)
    append_log("[BUILD] Step A′: shape dial after projecting out the symmetry-loudness direction (M2-SW)", cfg["_config_path"],
               command_line(), "ok", {"both dials (all criteria): new rule / PC2 rule": f"{n_new} / {n_pc2} of 3",
                                      "results": f"{cfg['output_md_prime']}, {cfg['output_json_gz_prime']}"}, wall, environment().get("gpu"))
    print("\n".join(L[:9]))


if __name__ == "__main__":
    main()
