"""[STAGE 3] The single confirmation run of the exploratory line (PREREGISTRATION_v2.md).

    python -m explore.stage3_confirm --config configs/explore/stage3_confirm.yaml

Run exactly once, after PREREGISTRATION_v2.md is committed; its output is final. A rerun (e.g.
after a crash) must be logged with the reason. Steps: verify every frozen checkpoint's SHA-256 and
the confirm set's checksums; unseal the confirm set (data.seal, which itself requires the committed
pre-registration); extract pooled latents; slow directions fitted on v1 train+val inviscid runs and
scored on the confirm inviscid runs; E1 and E4 exactly as in Stage 2 (explore/stage2_eval.py);
paired trajectory bootstrap (probes/criteria.py); apply the hypotheses; write results/explore/stage3/.

Diverged runs (pixel-family or autoencoder, final normalized val loss >= 1; PREREGISTRATION_v2 §5)
are excluded; a group with fewer than 2 remaining seeds makes its hypothesis "not evaluable".

--dry-run-on SPLIT --dry-config PATH (code check only): score on an unsealed split with a stand-in
config, skip the unseal, write to the config's dry-run paths, and do not log.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import warnings

import numpy as np

from common.run import REPO, append_log, command_line, environment, load_config, save_json, seed_everything
from data import seal
from data.combined import CombinedDataset
from explore.readouts import gpu_inputs, readouts_for
from explore.stage2_eval import direction_stats, s3
from probes import criteria
from probes.evaluate import probe_targets, slow_directions

RUN_DIRS = (REPO / "results" / "explore" / "stage2" / "runs", REPO / "results" / "explore" / "stage3" / "runs")


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def diverged(run: str) -> bool:
    if run.startswith("untrained_s"):
        return False
    for d in RUN_DIRS:
        if (d / f"{run}.json").exists():
            with open(d / f"{run}.json", encoding="utf-8") as f:
                r = json.load(f)
            return r["config"]["model"] in ("pixel", "pixel_vicreg", "autoencoder", "autoencoder_vicreg") and r["final"]["metric"] >= 1
    raise FileNotFoundError(f"no training record for {run}")


def traj_mean_log_p(ds, split):
    ev = ds.rows([split], viscous=False)
    _, inv = np.unique(ds.traj[ev], return_inverse=True)
    return np.bincount(inv, weights=np.log(ds.f["P"][ev])) / np.bincount(inv)


def boot_ci_single(runs, split, n_boot, seed):
    """95% CI of one group's seed-mean best-of-top-3 |r| (trajectory bootstrap)."""
    rng = np.random.default_rng(seed)
    m = len(runs[0][f"slow_{split}"]["traj"])
    shim = [{"slow": r[f"slow_{split}"]} for r in runs]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        v = [criteria._max_r(shim, "log_Z", rng.integers(0, m, m)) for _ in range(n_boot)]
    return [float(x) for x in np.percentile(v, [2.5, 97.5])]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry-run-on", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg["seed"])
    t0 = time.perf_counter()
    dry = args.dry_run_on is not None
    split = args.dry_run_on or cfg["eval_split"]

    for run, h in cfg["checkpoint_sha256"].items():  # frozen models
        got = sha256(REPO / "results" / "checkpoints" / f"{run}.pt")
        if got != h:
            raise SystemExit(f"checkpoint {run} changed since the pre-registration ({got[:12]} != {h[:12]}); aborting")
    if not dry:
        for name, h in cfg["confirm_sha256"].items():
            if sha256(REPO / cfg["confirm_dir"] / name) != h:
                raise SystemExit(f"confirm file {name} changed since generation; aborting")
        seal.unseal("Stage 3 confirmation run (PREREGISTRATION_v2.md)")
        parts = cfg["data_parts"]
    else:
        parts = [p for p in cfg["data_parts"] if p[0] != cfg["confirm_dir"]] + [[cfg["dry_run_dir"], [split]]]
    ds = CombinedDataset(parts, cfg["dry_run_cache_tag"] if dry else cfg["cache_tag"])
    x = gpu_inputs(ds)
    log_p = traj_mean_log_p(ds, split)

    res, excluded = {}, {}
    for g, runs in cfg["groups"].items():
        res[g] = []
        for run in runs:
            if diverged(run):
                excluded[run] = "diverged in training (final normalized val loss >= 1)"
                continue
            lat = readouts_for(run, ds, x, ds.rows(["train"]), readouts=("pooled",))["pooled"]
            slow = slow_directions(ds, lat, ["train", "val"], split, cfg["n_pca"])
            probes = probe_targets(ds, lat, split)
            res[g].append({"run": run, f"slow_{split}": slow, "direction_stats": direction_stats(slow, log_p),
                           "log_P_r2_partial": probes["log_P"]["r2_partial"], "log_Z_r2_partial": probes["log_Z"]["r2_partial"]})
            print(f"{g}: {run} done ({time.perf_counter() - t0:.0f} s)", flush=True)
    del x

    ok = lambda *gs: all(len(res[g]) >= 2 for g in gs)  # noqa: E731
    e1 = {g: float(np.mean([r[f"slow_{split}"]["max_abs_r"]["log_Z"] for r in runs])) if runs else float("nan") for g, runs in res.items()}
    gap = lambda a, b: s3(res[a], res[b], split)  # noqa: E731  (paired trajectory bootstrap of a - b, criteria.s3)
    H = cfg["hypotheses"]

    def rendering(groups, control):
        out = {}
        for g in groups:
            if not ok(g, control):
                out[g] = {"pass": False, "evaluable": False}
                continue
            d = gap(g, control)
            out[g] = {"E1": e1[g], "control": control, "gap_vs_control": d["gap"], "ci95": d["ci95"], "evaluable": True,
                      "pass": bool(e1[g] >= H["H1"]["min_e1"] and d["gap"] >= H["H1"]["min_gap"] and d["ci95"][0] > 0)}
        return out

    def drops(pairs, min_drop):
        out = {}
        for plain, reg in pairs:
            if not ok(plain, reg):
                out[f"{plain} − {reg}"] = {"pass": False, "evaluable": False}
                continue
            d = gap(plain, reg)
            out[f"{plain} − {reg}"] = {"gap": d["gap"], "ci95": d["ci95"], "evaluable": True,
                                       "pass": bool(d["gap"] >= min_drop and d["ci95"][0] > 0)}
        return out

    summary = {"E1": e1, "E1_ci": {g: boot_ci_single(runs, split, cfg["n_boot"], cfg["seed"]) for g, runs in res.items() if runs},
               "excluded": excluded}
    h1 = rendering(H["H1"]["groups"], H["H1"]["control"])
    h2 = drops(H["H2"]["pairs"], H["H2"]["min_drop"])
    ratio = {g: float(np.mean([r["direction_stats"]["E4_ratio_a_over_b"] for r in res[g]])) for g in res if res[g]}
    a3, b3 = H["H3"]["closer_to_one"], H["H3"]["farther"]
    h3 = {"ratio": {a3: ratio.get(a3), b3: ratio.get(b3)}, "evaluable": ok(a3, b3),
          "pass": bool(ok(a3, b3) and abs(math.log(ratio[a3])) < abs(math.log(ratio[b3])) and ratio[b3] > H["H3"]["min_ratio_farther"])}
    h4 = drops([H["H4"]["pair"]], H["H4"]["min_drop"])
    rep_h1 = rendering(H["replication"]["H1_groups"], H["replication"]["control"])
    rep_h2 = drops(H["replication"]["H2_pairs"], H["H2"]["min_drop"])
    summary.update({"H1": {"per_group": h1, "holds": all(v["pass"] for v in h1.values())},
                    "H2": {"per_pair": h2, "holds": all(v["pass"] for v in h2.values())},
                    "H3_secondary": h3, "H4_secondary": {"per_pair": h4, "holds": all(v["pass"] for v in h4.values())},
                    "replication_on_stage2_seeds": {"H1_rule": rep_h1, "H2_rule": rep_h2},
                    "E4_ratio": ratio,
                    "reported": {"imaginer k=4 − summarizer k=4": gap("imaginer k=4", "summarizer k=4") if ok("imaginer k=4", "summarizer k=4") else None,
                                 "log_P_r2_partial": {g: float(np.mean([r["log_P_r2_partial"] for r in runs])) for g, runs in res.items() if runs},
                                 "log_Z_r2_partial": {g: float(np.mean([r["log_Z_r2_partial"] for r in runs])) for g, runs in res.items() if runs}}})
    n_hold = summary["H1"]["holds"] + summary["H2"]["holds"]
    summary["verdict"] = "supported" if n_hold == 2 else "partially supported" if n_hold == 1 else "not supported"
    wall = time.perf_counter() - t0
    paths = cfg["dry_run_outputs"] if dry else cfg["outputs"]
    save_json({"environment": environment(), "config": cfg, "eval_split": split, "dry_run": dry, "summary": summary, "results": res,
               "wall_seconds": wall}, paths["json"])

    def line_h1(g, v):
        if not v["evaluable"]:
            return f"- {g}: not evaluable (fewer than 2 non-diverged seeds) → fail"
        return f"- {g} vs {v['control']}: E1 {v['E1']:.3f}, gap {v['gap_vs_control']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] → {'pass' if v['pass'] else 'fail'}"

    def line_drop(p, v):
        if not v["evaluable"]:
            return f"- {p}: not evaluable (fewer than 2 non-diverged seeds) → fail"
        return f"- {p}: {v['gap']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] → {'pass' if v['pass'] else 'fail'}"

    L = [f"# {'[DRY RUN on ' + split + ', stand-in groups] ' if dry else ''}Stage 3 confirmation (PREREGISTRATION_v2.md)", "",
         f"**Verdict: {summary['verdict']}** (H1 {'holds' if summary['H1']['holds'] else 'fails'}; "
         f"H2 {'holds' if summary['H2']['holds'] else 'fails'}).", "",
         f"Scored on `{split}` inviscid trajectories; slow directions fitted on v1 train+val inviscid runs; pooled latent. "
         "E1 = seed-mean best-of-top-3 |r| with log Z₀. CIs: 1,000 paired trajectory-bootstrap resamples.", "",
         "| group | seeds used | E1 [95% CI] | E4 a/b | log Z R² | log P R² |", "|---|---|---|---|---|---|"]
    for g, runs in res.items():
        if not runs:
            L.append(f"| {g} | 0 | — | — | — | — |")
            continue
        L.append(f"| {g} | {len(runs)} | {e1[g]:.3f} [{summary['E1_ci'][g][0]:.2f}, {summary['E1_ci'][g][1]:.2f}] | {ratio[g]:.2f} | "
                 f"{summary['reported']['log_Z_r2_partial'][g]:.3f} | {summary['reported']['log_P_r2_partial'][g]:.3f} |")
    L += ["", f"**H1 (primary)**: E1 ≥ {H['H1']['min_e1']} and ≥ control + {H['H1']['min_gap']}, CI > 0, for every group:", ""]
    L += [line_h1(g, v) for g, v in h1.items()]
    L += ["", f"**H2 (primary)**: plain − VICReg ≥ {H['H2']['min_drop']}, CI > 0, for every pair:", ""]
    L += [line_drop(p, v) for p, v in h2.items()]
    L += ["", f"**H3 (secondary)**: E4 a/b {a3} {ratio.get(a3, float('nan')):.2f} vs {b3} {ratio.get(b3, float('nan')):.2f} "
          f"→ {'pass' if h3['pass'] else 'fail'}", "",
          f"**H4 (secondary)**: plain imaginer − imaginer + VICReg ≥ {H['H4']['min_drop']}, CI > 0:", ""]
    L += [line_drop(p, v) for p, v in h4.items()]
    L += ["", "Reported (no verdict): the H1 and H2 rules applied to the Stage 2 seeds 0–2:", ""]
    L += [line_h1(g, v) for g, v in rep_h1.items()] + [line_drop(p, v) for p, v in rep_h2.items()]
    rep = summary["reported"]["imaginer k=4 − summarizer k=4"]
    if rep:
        L += ["", f"Reported: imaginer k=4 − summarizer k=4 E1 gap {rep['gap']:+.3f} [{rep['ci95'][0]:+.2f}, {rep['ci95'][1]:+.2f}]."]
    L += ["", "Excluded runs: " + ("; ".join(f"{k}: {v}" for k, v in excluded.items()) if excluded else "none") + ".", ""]
    with open(REPO / paths["md"], "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    if not dry:
        append_log("STAGE 3 confirmation run (single use of the sealed confirm set)", cfg["_config_path"], command_line(),
                   f"verdict: {summary['verdict']}", {"results": f"{paths['md']}, {paths['json']}"}, wall, environment().get("gpu"))
    print("\n".join(L))


if __name__ == "__main__":
    main()
