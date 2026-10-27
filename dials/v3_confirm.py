"""[BUILD] v3 Stage 5: the single confirmation run (PREREGISTRATION_v3.md).

    python -m dials.v3_confirm --config configs/build/v3_confirm.yaml --freeze      # after training: checkpoint checksums
    python -m dials.v3_confirm --config configs/build/v3_confirm.yaml               # the confirmation run (once)
    python -m dials.v3_confirm --config configs/build/v3_confirm.yaml --dry-config configs/build/v3_confirm_dry.yaml

--freeze (after all v3 models exist): records the SHA-256 of every v3 checkpoint, with each run's final
validation loss, in the config's frozen file; that file is committed before the confirmation run.

Confirmation run (exactly once; a rerun after a crash needs --rerun-reason, which is logged):
1. verify the frozen file is committed and unmodified, every checkpoint's SHA-256, and both sealed sets'
   file checksums; 2. unseal confirm3 and matched-energy-confirm (data.seal; requires the committed
   PREREGISTRATION_v3.md); 3. per model, identify the dials on exploration data only, exactly as in the
   build: the loudness direction by the symmetry method (dials/stepA.py), the shape direction by
   loudness-null PCA (dials/stepB.py), PC2 (secondary), with PCA, sigma_d and the identification frames
   from v1 train and explore; 4. score each direction with the steering harness on 256 confirm3 frames and
   256 matched-energy-confirm frames, the spatial mean of every decoded vorticity field removed before
   every measurement (the pre-removal share is recorded), under entry 2's criteria plus the artefact
   criterion; 5. rendering (reported, not a gate); 6. apply the pre-registered rules and write the result.

--dry-config (code check only): a stand-in config (exploration sets and existing models); no checksum
verification, no unseal, no log, outputs to the stand-in's paths.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from common.run import REPO, append_log, command_line, environment, load_config, seed_everything  # noqa: E402
from data import seal  # noqa: E402
from data.combined import CombinedDataset  # noqa: E402
from dials import fields  # noqa: E402
from dials.adapters import ExistingModel, adapter_for  # noqa: E402
from dials.harness import classify, envelope, pca_directions, steer  # noqa: E402
from dials.stage0 import frames  # noqa: E402
from dials.stage1 import load  # noqa: E402
from dials.stepA import _full_j, artefact_fraction, dial_latents, find_loudness, label  # noqa: E402
from dials.stepB import loudness_null  # noqa: E402
from solver.ns2d import SpectralGrid  # noqa: E402
from solver.validate import INK, SERIES, _style  # noqa: E402
from train.train import Model, ae_field  # noqa: E402

DIRECTIONS = ("loudness", "shape (loudness-null)", "PC2")


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


class ZeroMean:
    """The spatial mean of the decoded vorticity is removed before every measurement; the pre-removal
    fraction of frames with more than 10% of omega^2 in the mean is recorded per alpha."""

    def __init__(self, ad):
        self.ad, self.pre = ad, []

    def decode(self, x_prev, x_t, delta):
        outs = self.ad.decode(x_prev, x_t, delta)
        f = outs[0][2].double()
        self.pre.append(float(((f.mean(dim=(-2, -1)) ** 2 / f.pow(2).mean(dim=(-2, -1))) > 0.1).double().mean()))
        return [(n, k, v - v.mean(dim=(2, 3), keepdim=True)) for n, k, v in outs]

    def reencode(self, vel):
        return self.ad.reencode(vel)


@torch.no_grad()
def rendering_on(adapter, x_t, g) -> float:
    """Normalized MSE of the decoded vorticity (mean removed) against the true vorticity of the frames."""
    pred = adapter.decode(None, x_t, torch.zeros(adapter.d, device=x_t.device))[0][2].double()
    pred = pred - pred.mean(dim=(2, 3), keepdim=True)
    true = ae_field(x_t, "vorticity", SpectralGrid(64, x_t.device, dtype=torch.float32)).double()
    return float(((pred - true) ** 2).sum() / (true**2).sum())


def run_record(cfg, run) -> dict:
    for d in cfg["runs_dirs"]:
        if (REPO / d / f"{run}.json").exists():
            return json.loads((REPO / d / f"{run}.json").read_text(encoding="utf-8"))
    raise FileNotFoundError(f"no record for {run}")


def freeze(cfg):
    out = {"note": "[BUILD] v3 checkpoints, frozen after training and committed before the confirmation run", "runs": {}}
    for group, runs in cfg["groups"].items():
        for run in runs:
            rec = run_record(cfg, run)
            out["runs"][run] = {"group": group, "sha256": sha256(REPO / "results" / "checkpoints" / f"{run}.pt"),
                                "val_omega_normalized_mse": rec.get("final", {}).get("metric_vorticity")}
    (REPO / cfg["frozen_file"]).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


def verify(cfg):
    fz = REPO / cfg["frozen_file"]
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", cfg["frozen_file"]], cwd=REPO, capture_output=True).returncode == 0
    clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", cfg["frozen_file"]], cwd=REPO).returncode == 0
    if not (tracked and clean):
        raise SystemExit(f"{cfg['frozen_file']} must be committed and unmodified before the confirmation run")
    frozen = json.loads(fz.read_text(encoding="utf-8"))["runs"]
    for group, runs in cfg["groups"].items():
        for run in runs:
            got = sha256(REPO / "results" / "checkpoints" / f"{run}.pt")
            if got != frozen[run]["sha256"]:
                raise SystemExit(f"checkpoint {run} changed since it was frozen ({got[:12]}); aborting")
    for d, files in cfg["data_sha256"].items():
        for name, h in files.items():
            if sha256(REPO / d / name) != h:
                raise SystemExit(f"{d}/{name} changed since generation; aborting")
    return frozen


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--dry-config", default=None)
    ap.add_argument("--rerun-reason", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.freeze:
        return freeze(cfg)
    dry = args.dry_config is not None
    if dry:
        cfg = {**cfg, **load_config(args.dry_config)}
    seed_everything(cfg["seed"])
    t0 = time.perf_counter()
    out_json = REPO / cfg["output_json_gz"]
    frozen = {}
    if not dry:
        if out_json.exists() and not args.rerun_reason:
            raise SystemExit(f"{cfg['output_json_gz']} exists: the confirmation run is single-use (a rerun needs --rerun-reason)")
        frozen = verify(cfg)
        seal.unseal("v3 Stage 5 confirmation run" + (f" (rerun: {args.rerun_reason})" if args.rerun_reason else ""),
                    prereg="PREREGISTRATION_v3.md")
    ds = CombinedDataset(cfg["data_parts"], cfg["cache_tag"])
    g = SpectralGrid(64, "cuda")
    to_gpu = lambda rows: torch.from_numpy(ds.velocity[rows]).cuda() / ds.input_scale  # noqa: E731
    tr = ds.rows(["train"])
    env = envelope(to_gpu(np.sort(np.random.default_rng(cfg["seed"]).choice(tr, cfg["n_envelope"], replace=False))), g)
    excl = frames(ds, "explore", cfg["n_frames_explore_excluded"], 4, cfg["seed"])  # the build's steering frames
    x_opt = to_gpu(np.sort(np.random.default_rng(cfg["opt_seed"]).choice(np.setdiff1d(ds.rows(["explore"]), excl),
                                                                          cfg["n_opt_frames"], replace=False)))
    sets = {name: frames(ds, split, cfg["n_frames"], 4, cfg["seed"]) for name, split in cfg["score_sets"].items()}
    xs = {name: (to_gpu(r - 1), to_gpu(r)) for name, r in sets.items()}
    crit = {**cfg["criteria"], **cfg["artefact"]}
    first_set = next(iter(sets))

    res = {}
    for group, runs in cfg["groups"].items():
        res[group] = []
        for run in runs:
            model = load(run)
            G = dial_latents(model, ds, tr)
            pcs, sig, expl = pca_directions(G, n=8)
            cov = torch.as_tensor(np.cov(G.T), dtype=torch.float32, device="cuda")
            with torch.no_grad():
                c_o, d_o = model.enc(x_opt), model.head.dial(x_opt)
                enc_opt = (c_o, d_o, model.head.dec(c_o, d_o).float())
            rng = np.random.default_rng(cfg["seed"])
            inits = {**{f"PC{i + 1}": pcs[i] for i in range(8)}, **{f"random{j}": rng.standard_normal(8) for j in range(cfg["n_random_inits"])}}
            best = min(find_loudness(model, enc_opt, cov, inits, cfg), key=lambda r_: r_["J"])
            d_l = best["direction"].astype(np.float64) / np.linalg.norm(best["direction"])
            d_n, _, gm, spread = loudness_null(model, G, x_opt, g)
            cand = {"loudness": d_l, "shape (loudness-null)": d_n, "PC2": pcs[1]}
            entry = {"run": run, "J_loudness": best["J"], "loudness_gradient": {"g_z": gm.tolist(), **spread}, "directions": {}}
            for name, d in cand.items():
                e = {"direction": np.asarray(d).tolist(), "sigma": float((G @ d).std())}
                for s, (xp, xt) in xs.items():
                    zm = ZeroMean(adapter_for(model, "dial"))
                    m = steer(zm, xp, xt, d, e["sigma"], g, env)
                    m.pop("heads", None)
                    m["pre_removal_mean_gt10_by_alpha"] = zm.pre
                    e[s] = {**m, **classify(m, crit), "artefact_fraction": artefact_fraction(m, crit)}
                for k_ in ("dial", "loudness", "shape"):
                    e[k_] = bool(all(e[s][k_] for s in sets))
                entry["directions"][name] = e
            entry["loudness_dial"] = entry["directions"]["loudness"]["loudness"]
            entry["shape_dial"] = entry["directions"]["shape (loudness-null)"]["shape"]
            entry["both_dials"] = bool(entry["loudness_dial"] and entry["shape_dial"])
            entry["both_dials_PC2"] = bool(entry["loudness_dial"] and entry["directions"]["PC2"]["shape"])
            entry["rendering_" + first_set] = rendering_on(adapter_for(model, "dial"), xs[first_set][1], g)
            rec = run_record(cfg, run)
            entry["rendering_v1_val"] = rec.get("final", {}).get("metric_vorticity")
            entry["_strip"] = strip(model, cand, entry, xs[first_set][1][:1], g)
            res[group].append(entry)
            print(f"{group} {run}: loudness {label(entry['directions']['loudness'])}, shape {label(entry['directions']['shape (loudness-null)'])}, "
                  f"PC2 {label(entry['directions']['PC2'])} ({time.perf_counter() - t0:.0f} s)", flush=True)
            del model
            torch.cuda.empty_cache()
    refs = {}
    for run in cfg["vicreg_ae_reference"]:  # rendering reference: Stage 2 VICReg vorticity autoencoders on the same frames
        ck = torch.load(REPO / "results" / "checkpoints" / f"{run}.pt", map_location="cpu", weights_only=False)
        m = Model(ck["config"])
        m.load_state_dict(ck["model"])
        refs[run] = rendering_on(ExistingModel(m.cuda()), xs[first_set][1], g)
    verdict = decide(cfg, res)
    wall = time.perf_counter() - t0
    write(cfg, res, refs, verdict, frozen, wall, dry, args.rerun_reason, first_set)
    if not dry:
        append_log("[BUILD] v3 Stage 5 confirmation run (single use of confirm3 and matched-energy-confirm)"
                   + (f" — RERUN: {args.rerun_reason}" if args.rerun_reason else ""), cfg["_config_path"], command_line(), "ok",
                   {"verdict": verdict["text"], "results": f"{cfg['output_md']}, {cfg['output_json_gz']}"}, wall, environment().get("gpu"))


def decide(cfg, res) -> dict:
    r = cfg["rules"]
    n_main = sum(e["both_dials"] for e in res["main"])
    primary = "pass" if n_main >= r["pass_min"] else "partial" if n_main >= r["partial_min"] else "fail"
    n_tau0 = sum(e["shape_dial"] for e in res.get("tau0", []))
    n_untr = sum(e["shape_dial"] for e in res.get("untrained", []))
    ok_tau0, ok_untr = n_tau0 <= r["tau0_max_shape"], n_untr <= r["untrained_max_shape"]
    parts = [cfg["outcome_text"][primary].format(n=n_main, of=len(res["main"]))]
    if not ok_tau0:
        parts.append(cfg["control_failure_text"]["tau0"].format(n=n_tau0, of=len(res["tau0"])))
    if not ok_untr:
        parts.append(cfg["control_failure_text"]["untrained"].format(n=n_untr, of=len(res["untrained"])))
    if ok_tau0 and ok_untr:
        parts.append(cfg["controls_ok_text"].format(n_tau0=n_tau0, of_tau0=len(res.get("tau0", [])), n_untr=n_untr,
                                                    of_untr=len(res.get("untrained", []))))
    return {"primary": primary, "main_both_dials": n_main, "tau0_shape": n_tau0, "untrained_shape": n_untr,
            "tau0_ok": ok_tau0, "untrained_ok": ok_untr, "text": " ".join(parts)}


@torch.no_grad()
def strip(model, cand, entry, x1, g):
    ad = ZeroMean(adapter_for(model, "dial"))
    out = {}
    for name in ("loudness", "shape (loudness-null)"):
        e = entry["directions"][name]
        d = torch.as_tensor(cand[name], dtype=torch.float32, device="cuda")
        fs = [ad.decode(None, x1, float(a * e["sigma"]) * d)[0][2] for a in (-2.0, -1.0, 0.0, 1.0, 2.0)]
        out[name] = {"fields": [f[0, 0].cpu().numpy() for f in fs],
                     "spectra": [fields.shell_spectrum(fields.vorticity_hat(f, "vorticity", g), g)[0].cpu().numpy() for f in fs]}
    return out


def write(cfg, res, refs, verdict, frozen, wall, dry, rerun, first_set):
    sets = list(cfg["score_sets"])
    f2 = lambda v: f"{v:.2f}"  # noqa: E731
    head = "# [BUILD] v3 confirmation (PREREGISTRATION_v3.md)" + (" — DRY RUN (stand-in config; nothing sealed was opened)" if dry else "")
    L = [head, "", f"**Verdict:** {verdict['text']}", "",
         f"Main model: both dials valid (all criteria, both sets) in **{verdict['main_both_dials']} of {len(res['main'])}** seeds → "
         f"**{verdict['primary']}**. τ = 0 control: shape dial in {verdict['tau0_shape']} of {len(res.get('tau0', []))} seeds "
         f"({'within' if verdict['tau0_ok'] else 'outside'} its rule). Untrained control: shape dial in {verdict['untrained_shape']} of "
         f"{len(res.get('untrained', []))} seeds ({'within' if verdict['untrained_ok'] else 'outside'} its rule).", ""]
    if rerun:
        L += [f"RERUN, reason: {rerun}", ""]
    L += ["## Per model", "", "| group | run | loudness dial | shape dial (loudness-null) | both | PC2: shape dial / both (secondary) | "
          + " | ".join(f"σ_shape loudness / shape, {s}" for s in sets) + f" | rendering ω ({first_set}) | rendering ω (v1 val) |",
          "|---|---|---|---|---|---|" + "---|" * len(sets) + "---|---|"]
    yn = lambda b: "yes" if b else "no"  # noqa: E731
    for group, rs in res.items():
        for e in rs:
            dl, ds_, pc = (e["directions"][k] for k in DIRECTIONS)
            rv = e["rendering_v1_val"]
            L.append(f"| {group} | {e['run']} | {yn(e['loudness_dial'])} | {yn(e['shape_dial'])} | **{yn(e['both_dials'])}** | "
                     f"{yn(pc['shape'])} / {yn(e['both_dials_PC2'])} | "
                     + " | ".join(f"{f2(dl[s]['sigma_shape'])} / {f2(ds_[s]['sigma_shape'])}" for s in sets)
                     + f" | {e['rendering_' + first_set]:.4f} | {'—' if rv is None else f'{rv:.4f}'} |")
    L += ["", f"Rendering reference (not a gate): Stage 2 VICReg vorticity autoencoders on the same {first_set} frames: "
          + ", ".join(f"{k}: {v:.4f}" for k, v in refs.items()) + ".", ""]
    for s in sets:
        L += [f"## Every direction, every metric — {s} (spatial mean of decoded ω removed before measuring)", "",
              "| group | run | direction | σ_d | Δa | Δs | effect | σ_shape | monot. | cycle corr / gain | env − ideal | shared | artefact frames | "
              "pre-removal mean > 10% (max over α) | D L S | fails |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for group, rs in res.items():
            for e in rs:
                for name in DIRECTIONS:
                    d = e["directions"][name]
                    m = d[s]
                    L.append(f"| {group} | {e['run']} | {name} | {d['sigma']:.3g} | {m['d_a']:+.3f} | {m['d_s']:+.3f} | {m['effect']:.3f} | "
                             f"{f2(m['sigma_shape'])} | {f2(m['monotonicity'])} | {f2(m['cycle_corr'])} / {f2(m['cycle_gain'])} | "
                             f"{m['envelope_rel_min']:+.2f} | {f2(m['shared_pattern_fraction'])} | {m['artefact_fraction']:.1%} | "
                             f"{max(m['pre_removal_mean_gt10_by_alpha']):.1%} | "
                             f"{'D' if m['dial'] else '·'}{'L' if m['loudness'] else '·'}{'S' if m['shape'] else '·'} | {', '.join(m['fails']) or '—'} |")
        L.append("")
    if frozen:
        L += ["Checkpoints verified against the frozen file: " + ", ".join(f"{k} {v['sha256'][:12]}" for k, v in frozen.items()) + ".", ""]
    L.append(f"Wall-clock {wall / 60:.0f} min.")
    (REPO / cfg["output_md"]).parent.mkdir(parents=True, exist_ok=True)
    strips = {e["run"]: e.pop("_strip") for rs in res.values() for e in rs}
    with open(REPO / cfg["output_md"], "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    with gzip.open(REPO / cfg["output_json_gz"], "wt", encoding="utf-8") as f:
        json.dump({"environment": environment(), "config": cfg, "results": res, "rendering_reference": refs, "verdict": verdict,
                   "dry_run": dry, "rerun_reason": rerun, "wall_seconds": wall}, f, default=float)
    _style()
    main_runs = [e["run"] for e in res["main"]]
    fig, axes = plt.subplots(2 * len(main_runs), 6, figsize=(16, 2.4 * 2 * len(main_runs)), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1] * 5 + [1.5]})
    for i, run in enumerate(main_runs):
        for j, name in enumerate(("loudness", "shape (loudness-null)")):
            row, st = axes[2 * i + j], strips[run][name]
            lim = float(np.abs(st["fields"][2]).max())
            for ax, fld, a in zip(row[:5], st["fields"], (-2, -1, 0, 1, 2)):
                ax.imshow(fld, cmap="RdBu_r", vmin=-lim, vmax=lim)
                ax.set_title((f"{run}, {name}: " if a == -2 else "") + f"α = {a:+d}", fontsize=7)
                ax.set_xticks([])
                ax.set_yticks([])
            k = np.arange(1, fields.K_MAX + 1)
            for spec, a, col in zip(st["spectra"], (-2, -1, 0, 1, 2), (SERIES[0], SERIES[0], INK, SERIES[1], SERIES[1])):
                row[5].loglog(k, spec, color=col, alpha=1.0 if a in (-2, 0, 2) else 0.45)
    fig.suptitle(f"[BUILD] v3 confirmation: both dials of the main model on one {first_set} frame (mean removed)"
                 + (" — DRY RUN" if dry else ""), fontweight="bold", color=INK)
    fig.savefig(REPO / cfg["figure"], dpi=90)
    plt.close(fig)
    print("\n".join(L[:12]))


if __name__ == "__main__":
    main()
