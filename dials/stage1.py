"""[BUILD] Stage 1 readouts for the split models (results/build/PREDICTIONS.md entries 2-3). Exploratory.

    python -m dials.stage1 --config configs/build/stage1_eval.yaml

Per run: rendering of each head against the Stage 2 autoencoders (the rendering trigger), the dial
collapse check, and the candidate dials (PC1-8 of the 8-dim dial subspace; the investigator's
omega-psi contrast direction; the supervised s-direction in the dial subspace and in the full
latent), each steered on the Stage 0 explore and matched-energy frames and classified with the
entry 2 criteria; E1-style prominence (best of the top-3 slow directions, |r| with log Z0 and log E0)
in the dial, content and full latents; ridge probes from each latent. The Stage 2 autoencoders get
the same prominence and probes, for reference.

[BUILD] Stage 2 (entry 4) reuses this script for the branch model M2-SF (configs/build/stage2_eval.yaml):
one rendered field (no psi head, no head ratio) and no omega-psi contrast candidate (its dial groups
own no head); titles come from the config.

[BUILD] Stage 2b (entry 5): with effect_space: true, branch models also get the secondary
effect-space candidates ES1-4 (dials/effect_space.py), and the summary compares them with PCA.
Groups with fewer than 3 seeds get no "produces" verdict.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from common.run import REPO, append_log, command_line, environment, load_config, seed_everything  # noqa: E402
from data.combined import CombinedDataset  # noqa: E402
from dials import fields  # noqa: E402
from dials.adapters import adapter_for  # noqa: E402
from dials.effect_space import cos2, effect_space  # noqa: E402
from dials.harness import classify, envelope, pca_directions, steer, supervised_direction  # noqa: E402
from dials.split import TARGETS  # noqa: E402
from dials.stage0 import SETS, frames  # noqa: E402
from explore.readouts import readouts_for  # noqa: E402
from probes.evaluate import probe_nu, slow_directions  # noqa: E402
from probes.ridge import RidgeProbe  # noqa: E402
from solver.ns2d import SpectralGrid  # noqa: E402
from solver.validate import INK, INK2, SERIES, _style  # noqa: E402
from train.train import Model  # noqa: E402

PCS = [f"PC{i}" for i in range(1, 9)]
CONTRAST, SUP_DIAL, SUP_FULL = "contrast ω−ψ", "supervised s (dial)", "supervised s (full)"
CANDIDATES = PCS + [CONTRAST, SUP_DIAL, SUP_FULL]
ES = [f"ES{i}" for i in range(1, 5)]


def load(run: str) -> Model:
    ck = torch.load(REPO / "results" / "checkpoints" / f"{run}.pt", map_location="cpu", weights_only=False)
    m = Model(ck["config"])
    m.load_state_dict(ck["model"])
    return m.cuda().eval()


@torch.no_grad()
def all_latents(model: Model, ds, chunk: int = 4096) -> np.ndarray:
    """(N, 128) = [8 dial dims, content 120 as the decoder sees it] for every row."""
    ad = adapter_for(model, "full")
    out = []
    for i in range(0, len(ds.velocity), chunk):
        x = torch.from_numpy(ds.velocity[np.arange(i, min(i + chunk, len(ds.velocity)))]).cuda() / ds.input_scale
        out.append(ad.latent(x).double().cpu().numpy())
    return np.concatenate(out)


def contrast_direction(dial_tr: np.ndarray) -> tuple[np.ndarray, float]:
    """Entry 3 (investigator's prediction), label-free: u_w, u_p = top raw-PCA direction of the dial-omega
    and of the dial-psi group (train frames); u_p's sign makes the two projections correlate positively;
    c = [u_w / sd_w, -u_p / sd_p] / norm. Also returns r between the two group projections."""
    c = dial_tr - dial_tr.mean(0)
    u_w = np.linalg.svd(c[:, :4], full_matrices=False)[2][0]
    u_p = np.linalg.svd(c[:, 4:], full_matrices=False)[2][0]
    p_w, p_p = c[:, :4] @ u_w, c[:, 4:] @ u_p
    r = float(np.corrcoef(p_w, p_p)[0, 1])
    if r < 0:
        u_p, r = -u_p, -r
    d = np.concatenate([u_w / p_w.std(), -u_p / p_p.std()])
    return d / np.linalg.norm(d), r


def targets(ds) -> dict:
    l_e, l_z, l_p = (np.log(ds.f[k]) for k in ("E", "Z", "P"))
    return {"log E": l_e, "log Z": l_z, "log P": l_p, "s": l_z - l_e, "log(P/Z)": l_p - l_z}


def probes(ds, lat: np.ndarray, tg: dict) -> dict:
    """Ridge probes fitted on v1 train (alpha on v1 val), scored on explore: plain R^2 and the
    time-partialled R^2 of the Stage 2 protocol; log nu from single frames and from frame pairs."""
    tr, va, ev = ds.rows(["train"]), ds.rows(["val"]), ds.rows(["explore"])
    pidx, n_pidx = ds.partial_index(), 2 * ds.n_frame_max
    out = {}
    for name, y in tg.items():
        pp = RidgeProbe().fit(lat[tr], y[tr], ds.frame[tr], lat[va], y[va], ds.frame[va])
        p = RidgeProbe(partial=True, n_frame_index=n_pidx).fit(lat[tr], y[tr], pidx[tr], lat[va], y[va], pidx[va])
        out[name] = {"r2": pp.score(lat[ev], y[ev], ds.frame[ev])["r2"], "r2_partial": p.score(lat[ev], y[ev], pidx[ev])["r2_partial"]}
    nu = probe_nu(ds, lat, "explore")
    out["log ν"] = {"r2": nu["single_r2"], "r2_pair": nu["pair_r2"]}
    return out


def prominence(ds, lat: np.ndarray, n_pca: int) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sl = slow_directions(ds, lat, ["train", "val"], "explore", n_pca)
    return {"log Z": sl["max_abs_r"]["log_Z"], "log E": sl["max_abs_r"]["log_E"],
            "abs_r_log_Z": sl["abs_r"]["log_Z"], "abs_r_log_E": sl["abs_r"]["log_E"]}


def references(cfg) -> dict:
    """Stage 2 autoencoders' val normalized MSE and small-scale power, mean over their runs."""
    out = {}
    for t in TARGETS:
        for kind, label in (("plain", f"{t} AE"), ("vicreg", f"{t} AE + VICReg")):
            rs = [json.loads((REPO / cfg["reference_json"].format(run=r)).read_text()) for r in cfg["references"][label]]
            out.setdefault(t, {})[kind] = float(np.mean([r["final"]["metric"] for r in rs]))
            out[t][kind + "_k10"] = float(np.mean([r["diagnostics"]["pred_over_true_power_k_ge_10"] for r in rs]))
    return out


def rendering(run: str, cfg, refs: dict) -> dict:
    r = json.loads((REPO / cfg["runs_json"].format(run=run)).read_text())
    fin, dg = r["final"], r.get("diagnostics", {})
    out = {"wall_s": r["wall_seconds"], "steps": r["steps"], "stop_reason": r["stop_reason"]}
    for t in TARGETS:
        m, k10 = fin[f"metric_{t}"], dg.get(f"{t}_pred_over_true_power_k_ge_10", float("nan"))
        trig = [lbl for lbl, bad in (("> 2× VICReg AE", m > 2 * refs[t]["vicreg"]), ("> 0.02", m > 0.02),
                                     ("< 50% small-scale power", k10 < 0.5)) if bad]
        out[t] = {"metric": m, "vs_plain": m / refs[t]["plain"], "vs_vicreg": m / refs[t]["vicreg"], "k10": k10, "trigger": trig}
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = load_config(ap.parse_args().config)
    seed_everything(cfg["seed"])
    t0 = time.perf_counter()
    ds = CombinedDataset(cfg["data_parts"], cfg["cache_tag"])
    g = SpectralGrid(ds.velocity.shape[-1], "cuda")
    to_gpu = lambda rows: torch.from_numpy(ds.velocity[rows]).cuda() / ds.input_scale  # noqa: E731
    tr, va, ex = ds.rows(["train"]), ds.rows(["val"]), ds.rows(["explore"])
    fr = {s: frames(ds, s, cfg["n_frames"], 4, cfg["seed"]) for s in SETS}
    x = {s: (to_gpu(fr[s] - 1), to_gpu(fr[s])) for s in SETS}
    env = envelope(to_gpu(np.sort(np.random.default_rng(cfg["seed"]).choice(tr, cfg["n_envelope"], replace=False))), g)
    tg = targets(ds)
    a_all, s_all = tg["log E"], tg["s"]
    crit = cfg["criteria"]
    refs = references(cfg)

    res = {}
    for label, runs in cfg["groups"].items():
        res[label] = []
        for run in runs:
            model = load(run)
            lat = all_latents(model, ds)
            dial = lat[:, :8]
            entry = {"run": run, "rendering": rendering(run, cfg, refs)}
            if getattr(model.head, "norm", None) == "rms":
                entry["content_gain"] = float(model.head.log_gain.detach().exp())
            sd = lat[tr].std(0)
            entry["dial_std"] = sd[:8].tolist()
            entry["dial_std_over_content_median"] = float(sd[:8].max() / np.median(sd[8:]))
            entry["dial_collapsed"] = entry["dial_std_over_content_median"] < cfg["collapse_max_ratio"]
            dirs, sig, expl = pca_directions(dial[tr], n=8)
            groups = model.kind == "split_ae"
            c, r_groups = contrast_direction(dial[tr]) if groups else (np.zeros(8), float("nan"))
            entry.update({"dial_explained": expl.tolist(), "r_group_top_pcs": r_groups})
            d_sd, s_sd, _ = supervised_direction(dial[tr], s_all[tr], ds.frame[tr], dial[va], s_all[va], ds.frame[va])
            d_sf, s_sf, _ = supervised_direction(lat[tr], s_all[tr], ds.frame[tr], lat[va], s_all[va], ds.frame[va])
            cand = {PCS[i]: ("dial", dirs[i], float(sig[i])) for i in range(8)}
            if groups:
                cand[CONTRAST] = ("dial", c, float((dial[tr] @ c).std()))
            if cfg.get("effect_space") and model.kind == "branch_ae":  # secondary method (entry 5)
                es = effect_space(model, x["explore"][1], len(ES))
                for name, d in zip(ES, es["directions"]):
                    cand[name] = ("dial", d / np.linalg.norm(d), float((dial[tr] @ (d / np.linalg.norm(d))).std()))
                entry["effect_space"] = {"singular_values": es["singular_values"].tolist(), "left": es["left"].tolist(),
                                         "jacobian": es["jacobian"].tolist(),
                                         "cos2_with_PCs": [[cos2(cand[n][1], dirs[j]) for j in range(8)] for n in ES]}
            cand.update({SUP_DIAL: ("dial", d_sd, s_sd), SUP_FULL: ("full", d_sf, s_sf)})
            ads = {"dial": adapter_for(model, "dial"), "full": adapter_for(model, "full")}
            entry["directions"] = {}
            for name, (space, d, s_d) in cand.items():
                proj = (dial if space == "dial" else lat)[ex] @ d
                e = {"space": space, "sigma": s_d, "cos2_contrast": float((d[:8] @ c) ** 2), "direction": np.asarray(d).tolist(),
                     "readability": {"abs_r_a": float(abs(np.corrcoef(proj, a_all[ex])[0, 1])),
                                     "abs_r_s": float(abs(np.corrcoef(proj, s_all[ex])[0, 1]))}}
                for s in SETS:
                    m = steer(ads[space], x[s][0], x[s][1], d, s_d, g, env)
                    e[s] = {**m, **classify(m, crit)}
                e.update({k_: bool(e["explore"][k_] and e["matched"][k_]) for k_ in ("dial", "loudness", "shape")})
                entry["directions"][name] = e
            entry["prominence"] = {"dial": prominence(ds, dial, 8), "content": prominence(ds, lat[:, 8:], 32),
                                   "full": prominence(ds, lat, 32)}
            entry["probes"] = {"dial": probes(ds, dial, tg), "content": probes(ds, lat[:, 8:], tg), "full": probes(ds, lat, tg)}
            entry["_strip"] = strip_data(model, ads, cand, entry, x["explore"])
            res[label].append(entry)
            print(f"{label}: {run} done ({time.perf_counter() - t0:.0f} s)", flush=True)
            del model
            torch.cuda.empty_cache()
    ref_res = {}
    for label, runs in cfg["references"].items():
        ref_res[label] = []
        for run in runs:
            lat = readouts_for(run, ds, None, tr, readouts=("pooled",))["pooled"].astype(np.float64)
            ref_res[label].append({"run": run, "prominence": prominence(ds, lat, 32), "probes": probes(ds, lat, tg)})
        print(f"reference {label} done ({time.perf_counter() - t0:.0f} s)", flush=True)
    wall = time.perf_counter() - t0
    strips = {f"{label} {r['run'].rsplit('_s', 1)[1]}": r.pop("_strip") for label, runs in res.items() for r in runs}
    summary = summarize(res, cfg)
    with gzip.open(REPO / cfg["output_json_gz"], "wt", encoding="utf-8") as f:
        json.dump({"environment": environment(), "config": cfg, "results": res, "references": ref_res,
                   "reference_rendering": refs, "summary": summary, "wall_seconds": wall}, f, default=float)
    write_md(cfg, res, ref_res, refs, summary)
    figure(cfg, res)
    figure_strips(cfg, strips)
    append_log(cfg.get("log_title", "[BUILD] Stage 1 readouts (M1-spec, M1-G)"), cfg["_config_path"], command_line(), "ok",
               {**{f"{label}: seeds with a shape / loudness dial (PC1–8)": f"{v['seeds_shape']} / {v['seeds_loudness']} of {v['n_seeds']}"
                   for label, v in summary.items()},
                "results": f"{cfg['output_md']}, {cfg['figure']}, {cfg['figure_strips']}, {cfg['output_json_gz']}"},
               wall, environment().get("gpu"))


@torch.no_grad()
def strip_data(model, ads, cand, entry, x_ex) -> dict:
    """For the figure: the dial-subspace candidate with the largest |Δs| (explore, vorticity head),
    decoded on one explore frame at alpha = -2, 0, +2 (vorticity head), with shell spectra."""
    names = [n for n in cand if cand[n][0] == "dial"]
    best = max(names, key=lambda n: abs(entry["directions"][n]["explore"]["d_s"]))
    space, d, s_d = cand[best]
    d_t = torch.as_tensor(d, dtype=torch.float32, device="cuda")
    g = SpectralGrid(64, "cuda")
    out = {"direction": best, "fields": [], "spectra": []}
    for alpha in (-2.0, 0.0, 2.0):
        f = ads[space].decode(x_ex[0][:1], x_ex[1][:1], float(alpha * s_d) * d_t)[0][2]
        out["fields"].append(f[0, 0].cpu().numpy().tolist())
        out["spectra"].append(fields.shell_spectrum(fields.vorticity_hat(f, "vorticity", g), g)[0].cpu().numpy().tolist())
    m = entry["directions"][best]["explore"]
    out.update({"d_a": m["d_a"], "d_s": m["d_s"], "sigma_shape": m["sigma_shape"]})
    return out


def summarize(res: dict, cfg) -> dict:
    out = {}
    for label, runs in res.items():
        per_seed = []
        for r in runs:
            dirs = r["directions"]
            shape = [n for n in PCS if dirs[n]["shape"]]
            loud = [n for n in PCS if dirs[n]["loudness"]]
            strength = lambda n: min(dirs[n]["explore"]["effect"], dirs[n]["matched"]["effect"])  # noqa: E731
            best_shape = max(shape, key=strength) if shape else None
            es_shape = [n for n in ES if n in dirs and dirs[n]["shape"]]
            es_loud = [n for n in ES if n in dirs and dirs[n]["loudness"]]
            best_es = max(es_shape, key=strength) if es_shape else None
            both = (np.array(dirs[best_shape]["direction"]), np.array(dirs[best_es]["direction"])) if best_shape and best_es else None
            per_seed.append({"run": r["run"], "shape_PC1-8": shape, "shape_PC1-4": [n for n in shape if n in PCS[:4]],
                             "loudness_PC1-8": loud, "loudness_PC1-4": [n for n in loud if n in PCS[:4]],
                             "best_shape": best_shape,
                             "best_shape_cos2_contrast": dirs[best_shape]["cos2_contrast"] if best_shape else None,
                             "contrast": {k_: dirs[CONTRAST][k_] for k_ in ("dial", "loudness", "shape")} if CONTRAST in dirs else None,
                             "supervised_dial": {k_: dirs[SUP_DIAL][k_] for k_ in ("dial", "loudness", "shape")},
                             "supervised_full": {k_: dirs[SUP_FULL][k_] for k_ in ("dial", "loudness", "shape")},
                             "triggers": {t: r["rendering"][t]["trigger"] for t in TARGETS}, "dial_collapsed": r["dial_collapsed"],
                             **({"shape_ES": es_shape, "loudness_ES": es_loud, "best_shape_ES": best_es,
                                 "cos2_best_PC_vs_best_ES": cos2(*both) if both else None,
                                 "methods_agree": bool(bool(shape) == bool(es_shape) and (both is None or cos2(*both) >= 0.5))}
                                if "effect_space" in r else {})})
        has_contrast = all(p["contrast"] is not None for p in per_seed)
        n_shape = sum(bool(p["shape_PC1-8"]) for p in per_seed)
        with_best = [p for p in per_seed if p["best_shape"]]
        enough = len(per_seed) >= 3
        out[label] = {"n_seeds": len(per_seed), "seeds_shape": n_shape, "seeds_shape_PC1-4": sum(bool(p["shape_PC1-4"]) for p in per_seed),
                      "seeds_loudness": sum(bool(p["loudness_PC1-8"]) for p in per_seed),
                      "produces_shape_dial": n_shape >= cfg["criteria"]["min_seeds"] if enough else None,
                      "produces_loudness_dial": (sum(bool(p["loudness_PC1-8"]) for p in per_seed) >= cfg["criteria"]["min_seeds"]
                                                 if enough else None),
                      "contrast_prediction": (None if not (with_best and has_contrast) else
                                              sum(p["best_shape_cos2_contrast"] >= cfg["contrast_min_cos2"] for p in with_best) * 2 > len(with_best)),
                      "per_seed": per_seed}
        if all("shape_ES" in p for p in per_seed):
            n_es = sum(bool(p["shape_ES"]) for p in per_seed)
            out[label].update({"seeds_shape_ES": n_es, "seeds_loudness_ES": sum(bool(p["loudness_ES"]) for p in per_seed),
                               "produces_shape_dial_ES": n_es >= cfg["criteria"]["min_seeds"] if enough else None,
                               "methods_agree": all(p["methods_agree"] for p in per_seed)})
    return out


def write_md(cfg, res, ref_res, refs, summary):
    crit = cfg["criteria"]
    f2, f3 = (lambda v: f"{v:.2f}"), (lambda v: f"{v:.3f}")  # noqa: E731
    L = [cfg.get("title", "# [BUILD] Stage 1: M1-spec and M1-G (exploratory)"), "",
         cfg.get("intro", "Definitions and predictions: results/build/PREDICTIONS.md entries 2–3. M1-G's loudness dial exists by construction "
                 "(content is RMS-normalized): it is a design property, not a finding. The finding is whether a label-free shape dial emerges."), "",
         f"Dial: effect ≥ {crit['min_effect']}, monotonicity ≥ {crit['min_monotonicity']}, cycle corr ≥ {crit['min_cycle_corr']} and gain in "
         f"[{crit['cycle_gain'][0]}, {crit['cycle_gain'][1]}], shared pattern ≤ {crit['max_shared_pattern']}, envelope ≥ ideal-dial envelope − "
         f"{crit['envelope_rel_margin']} at every α, head inconsistency ≤ {crit['max_head_ratio']}× its α = 0 value; on both explore and "
         f"matched-energy; scored on the vorticity head. Loudness σ_shape ≤ {crit['max_sigma_loudness']}, shape ≥ {crit['min_sigma_shape']}. "
         f"A model produces a dial if ≥ {crit['min_seeds']} of 3 seeds have one among PC1–8 of the dial subspace.", "",
         "## Summary", "", "| model | seeds with a shape dial (PC1–8 / PC1–4) | seeds with a loudness dial (PC1–8) | produces a shape dial? | "
         "best shape dial loads on the ω−ψ contrast (cos² ≥ 0.5)? | contrast direction is a shape dial (seeds) | supervised s (dial / full) is a shape dial (seeds) |",
         "|---|---|---|---|---|---|---|"]
    for label, v in summary.items():
        ps = v["per_seed"]
        no_c = any(p["contrast"] is None for p in ps)
        cp = ("n/a (no contrast direction)" if no_c else "n/a (no shape dial)" if v["contrast_prediction"] is None
              else "yes" if v["contrast_prediction"] else "no")
        cs = "n/a" if no_c else sum(p["contrast"]["shape"] for p in ps)
        prod = "n/a (1 seed)" if v["produces_shape_dial"] is None else "**yes**" if v["produces_shape_dial"] else "no"
        L.append(f"| {label} | {v['seeds_shape']} / {v['seeds_shape_PC1-4']} of {v['n_seeds']} | {v['seeds_loudness']} of {v['n_seeds']} | "
                 f"{prod} | {cp} | {cs} | "
                 f"{sum(p['supervised_dial']['shape'] for p in ps)} / {sum(p['supervised_full']['shape'] for p in ps)} |")
    if any("seeds_shape_ES" in v for v in summary.values()):
        L += ["", "## Effect-space method (secondary): top-4 right-singular vectors of the frame-averaged Jacobian of the decoded "
              "log shell spectrum with respect to the dial vector", "",
              "| model | seed | singular values (top 4) | shape dials: PCA (PC1–8) / effect space (ES1–4) | loudness dials: PCA / effect space | "
              "cos² best PCA vs best ES shape dial | methods agree? | each ES's best-matching PC (cos²) |", "|---|---|---|---|---|---|---|---|"]
        for label, runs in res.items():
            for r, p_ in zip(runs, summary[label]["per_seed"]):
                if "effect_space" not in r:
                    continue
                esd = r["effect_space"]
                match = ", ".join(f"ES{i + 1}→PC{int(np.argmax(row)) + 1} ({max(row):.2f})" for i, row in enumerate(esd["cos2_with_PCs"]))
                c2b = "—" if p_["cos2_best_PC_vs_best_ES"] is None else f"{p_['cos2_best_PC_vs_best_ES']:.2f}"
                L.append(f"| {label} | {r['run'].rsplit('_s', 1)[1]} | {', '.join(f'{v:.3g}' for v in esd['singular_values'][:4])} | "
                         f"{', '.join(p_['shape_PC1-8']) or 'none'} / {', '.join(p_['shape_ES']) or 'none'} | "
                         f"{', '.join(p_['loudness_PC1-8']) or 'none'} / {', '.join(p_['loudness_ES']) or 'none'} | {c2b} | "
                         f"{'yes' if p_['methods_agree'] else 'no'} | {match} |")
        for label, v in summary.items():
            if "seeds_shape_ES" in v:
                prod = "n/a (1 seed)" if v["produces_shape_dial_ES"] is None else "yes" if v["produces_shape_dial_ES"] else "no"
                L.append(f"\n{label}: effect-space shape dial in {v['seeds_shape_ES']} of {v['n_seeds']} seeds (produces: {prod}); "
                         f"loudness in {v['seeds_loudness_ES']}; methods agree in every seed: {'yes' if v['methods_agree'] else 'no'}.")
    L += ["", "## Rendering (val normalized MSE; trigger: > 2× the VICReg AE, > 0.02, or < 50% of the true k ≥ 10 power)", "",
          "| model | seed | ω head MSE | ×plain AE | ×VICReg AE | ω k≥10 power kept | ψ head MSE | ×plain AE | ×VICReg AE | ψ k≥10 power | triggers | "
          "content gain | max dial std / median content std | r(top PC dial-ω, top PC dial-ψ) |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for label, runs in res.items():
        for r in runs:
            rd = r["rendering"]
            trig = "; ".join(f"{t}: {', '.join(rd[t]['trigger'])}" for t in TARGETS if rd[t]["trigger"]) or "none"
            if r["dial_collapsed"]:
                trig += "; dial collapse"
            L.append(f"| {label} | {r['run'].rsplit('_s', 1)[1]} | {rd['vorticity']['metric']:.5f} | {rd['vorticity']['vs_plain']:.1f} | "
                     f"{rd['vorticity']['vs_vicreg']:.2f} | {f2(rd['vorticity']['k10'])} | {rd['streamfunction']['metric']:.5f} | "
                     f"{rd['streamfunction']['vs_plain']:.1f} | {rd['streamfunction']['vs_vicreg']:.2f} | {f2(rd['streamfunction']['k10'])} | {trig} | "
                     f"{r.get('content_gain', float('nan')):.2f} | {r['dial_std_over_content_median']:.3f} | {r['r_group_top_pcs']:.2f} |")
    L.append("")
    L.append("Reference autoencoders (mean of 3 seeds): " + "; ".join(
        f"{t}: plain {refs[t]['plain']:.5f} (k≥10 {refs[t]['plain_k10']:.2f}), VICReg {refs[t]['vicreg']:.5f} (k≥10 {refs[t]['vicreg_k10']:.2f})" for t in TARGETS) + ".")
    for s in SETS:
        L += ["", f"## Every candidate dial — {s} (vorticity head; ψ head Δa / Δs alongside)", "",
              "| model | seed | dir | σ_d | Δa | Δs | Δlog P | effect | σ_shape | monot. | cycle corr / gain | env − ideal (min) | shared | head ratio | "
              "ψ head Δa / Δs | cos² contrast | D L S | fails |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for label, runs in res.items():
            for r in runs:
                for n, e in r["directions"].items():
                    m = e[s]
                    ps = m["heads"].get("streamfunction")
                    psi = f"{ps['d_a']:+.3f} / {ps['d_s']:+.3f}" if ps else "—"
                    hr = f2(m["head_consistency_ratio"]) if "head_consistency_ratio" in m else "—"
                    c2 = f2(e["cos2_contrast"]) if CONTRAST in r["directions"] else "—"
                    L.append(f"| {label} | {r['run'].rsplit('_s', 1)[1]} | {n} | {e['sigma']:.3g} | {m['d_a']:+.3f} | {m['d_s']:+.3f} | {m['d_logP']:+.3f} | "
                             f"{f3(m['effect'])} | {f2(m['sigma_shape'])} | {f2(m['monotonicity'])} | {f2(m['cycle_corr'])} / {f2(m['cycle_gain'])} | "
                             f"{m['envelope_rel_min']:+.2f} | {f2(m['shared_pattern_fraction'])} | {hr} | {psi} | {c2} | "
                             f"{'D' if m['dial'] else '·'}{'L' if m['loudness'] else '·'}{'S' if m['shape'] else '·'} | {', '.join(m['fails']) or '—'} |")
    L += ["", "## Prominence (E1-style: best of the top-3 slow directions, |r| with log Z₀ / log E₀, explore inviscid trajectories)", "",
          "| model | seed | dial (8) | content (120) | full (128) |", "|---|---|---|---|---|"]
    pz = lambda p: f"{p['log Z']:.2f} / {p['log E']:.2f}"  # noqa: E731
    for label, runs in res.items():
        for r in runs:
            pr = r["prominence"]
            L.append(f"| {label} | {r['run'].rsplit('_s', 1)[1]} | {pz(pr['dial'])} | {pz(pr['content'])} | {pz(pr['full'])} |")
    for label, runs in ref_res.items():
        for r in runs:
            L.append(f"| {label} (reference) | {r['run'].rsplit('_s', 1)[1]} | | | {pz(r['prominence'])} |")
    names = ["log E", "log Z", "log P", "s", "log(P/Z)"]
    L += ["", "## Ridge probes (fit v1 train, α on v1 val, scored on explore): R² (time-partialled R²); log ν: single frame (pair)", "",
          "| model | seed | latent | " + " | ".join(names) + " | log ν |", "|---|---|---|" + "---|" * (len(names) + 1)]

    def prow(label, seed, latent, p):
        return (f"| {label} | {seed} | {latent} | " + " | ".join(f"{p[n]['r2']:.3f} ({p[n]['r2_partial']:.3f})" for n in names)
                + f" | {p['log ν']['r2']:.3f} ({p['log ν']['r2_pair']:.3f}) |")
    for label, runs in res.items():
        for r in runs:
            for latent in ("content", "dial", "full"):
                L.append(prow(label, r["run"].rsplit("_s", 1)[1], latent, r["probes"][latent]))
    for label, runs in ref_res.items():
        for r in runs:
            L.append(prow(f"{label} (reference)", r["run"].rsplit("_s", 1)[1], "pooled", r["probes"]))
    with open(REPO / cfg["output_md"], "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L[:12]))


def figure(cfg, res):
    crit = cfg["criteria"]
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), constrained_layout=True)
    for ax, s in zip(axes, SETS):
        for i, (label, runs) in enumerate(res.items()):
            for r in runs:
                for n in r["directions"]:
                    m = r["directions"][n][s]
                    mk = "o" if n in PCS else "D" if n == CONTRAST else "s" if n in ES else "*"
                    ax.scatter(max(m["effect"], 1e-4), m["sigma_shape"], color=SERIES[i], marker=mk, s=60 if mk == "*" else 30,
                               edgecolors=INK if m["dial"] else "none", linewidths=1.2 if m["dial"] else 0, alpha=0.85,
                               label=label if (r is runs[0] and n == "PC1") else None)
        ax.axvline(crit["min_effect"], color=INK2, linewidth=1, linestyle=(0, (3, 2)))
        for yv in (crit["max_sigma_loudness"], crit["min_sigma_shape"]):
            ax.axhline(yv, color=INK2, linewidth=0.8, linestyle=(0, (1, 2)))
        ax.set_xscale("log")
        ax.set_xlabel("effect |Δa| + |Δs| per unit α (log-units)")
        ax.set_ylabel("shape selectivity σ_shape")
        ax.set_title(s)
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle(f"[BUILD] {cfg.get('stage_label', 'Stage 1')}: candidate dials (exploratory)\n"
                 "dots PC1–8 · squares effect-space ES1–4 · diamonds ω−ψ contrast · stars supervised s · outlined = passes every criterion on this set",
                 fontweight="bold", color=INK)
    fig.savefig(REPO / cfg["figure"], dpi=125)
    plt.close(fig)


def figure_strips(cfg, strips):
    _style()
    keys = list(strips)
    fig, axes = plt.subplots(len(keys), 5, figsize=(16, 3.0 * len(keys)), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1, 1, 1, 1, 1.4]})
    for row, key in zip(np.atleast_2d(axes), keys):
        st = strips[key]
        f = [np.array(v) for v in st["fields"]]
        lim = float(np.abs(f[1]).max())
        for ax, fld, ttl in zip(row[:3], f, ("α = −2", "α = 0", "α = +2")):
            ax.imshow(fld, cmap="RdBu_r", vmin=-lim, vmax=lim)
            ax.set_title(f"{key}: {st['direction']}, {ttl}" if ttl == "α = −2" else ttl, fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
        diff = f[2] - f[1]
        dl = float(np.abs(diff).max()) or 1.0
        row[3].imshow(diff, cmap="RdBu_r", vmin=-dl, vmax=dl)
        row[3].set_title(f"(+2) − (0), ±{dl:.2g}", fontsize=8)
        row[3].set_xticks([])
        row[3].set_yticks([])
        k = np.arange(1, len(st["spectra"][0]) + 1)
        for spec, col, lab in zip(st["spectra"], (SERIES[0], INK, SERIES[1]), ("α = −2", "α = 0", "α = +2")):
            row[4].loglog(k, spec, color=col, label=lab)
        row[4].set_title(f"Δa {st['d_a']:+.3f}, Δs {st['d_s']:+.3f} per α (median, explore)", fontsize=8)
        row[4].legend(fontsize=7)
    fig.suptitle(f"[BUILD] {cfg.get('stage_label', 'Stage 1')}: the largest-|Δs| dial-subspace candidate per run, vorticity field, "
                 "one explore frame (exploratory)",
                 fontweight="bold", color=INK)
    fig.savefig(REPO / cfg["figure_strips"], dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
