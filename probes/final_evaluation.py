"""The single test-set evaluation (PREREGISTRATION.md §8). Run exactly once, after §4–§6 are
frozen on val; its output is final. A rerun (e.g. after a crash) must be logged with the reason.

    python -m probes.final_evaluation --config configs/final_evaluation.yaml

For every model group and seed and both readouts: ridge probes scored on test-ID, the S2 probe,
OOD errors (§5.5), slow directions fitted on train+val inviscid runs and scored on test-ID
inviscid runs (§6), and MLP probes (raw readout). Then the pre-registered rules for the JEPA vs
the untrained control (raw readout = the verdict), and the same comparisons, reported only, for
the shift-averaged readout (attribution rule §2.6) and for every other model.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from common.run import append_log, command_line, environment, load_config, save_json, seed_everything
from data.dataset import S1_TARGETS, Dataset
from probes import criteria
from probes.evaluate import PROBE_TARGETS, probe_nu, probe_ood, probe_targets, slow_directions
from probes.latents import latents_for
from probes.mlp import mlp_probe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = load_config(ap.parse_args().config)
    seed_everything(cfg["seed"])
    t0 = time.perf_counter()
    ds = Dataset(cfg["data_dir"])
    results = {}
    for readout in cfg["readouts"]:
        results[readout] = {}
        for group, runs in cfg["models"].items():
            results[readout][group] = []
            for run in runs:
                lat = latents_for(run, ds, readout)
                r = {"run": run, "probes": probe_targets(ds, lat, "test_id"), "nu": probe_nu(ds, lat, "test_id"),
                     "slow": slow_directions(ds, lat, ["train", "val"], "test_id", cfg["n_pca"])}
                if readout == "raw":
                    r["ood"] = probe_ood(ds, lat)
                    if cfg["mlp"]:
                        r["mlp_r2_partial"] = {t: mlp_probe(ds, lat, t, "test_id", seed=cfg["seed"]) for t in S1_TARGETS}
                results[readout][group].append(r)
                print(f"{readout} {run} done ({time.perf_counter() - t0:.0f} s)", flush=True)

    ctrl, main_group = cfg["control_group"], cfg["main_group"]
    rules = {}
    for readout in cfg["readouts"]:
        rules[readout] = {}
        for group, runs in results[readout].items():
            if group == ctrl:
                continue
            a = criteria.s1(runs, results[readout][ctrl])
            b = criteria.s2(runs, results[readout][ctrl])
            c = criteria.s3(runs, results[readout][ctrl])
            rules[readout][group] = {"S1": a, "S2": b, "S3": c, "verdict": criteria.verdict(a, b, c)}
    verdict = rules["raw"][main_group]["verdict"]
    attribution = {}
    if "shift_avg" in rules:
        for t in S1_TARGETS:
            raw_c = rules["raw"][main_group]["S1"]["targets"][t]["counts"]
            sa_c = rules["shift_avg"][main_group]["S1"]["targets"][t]["counts"]
            attribution[t] = ("gain survives shift averaging" if raw_c and sa_c else
                              "gain attributed to shift-tolerance (§2.6)" if raw_c and not sa_c else "no counted gain")
    wall = time.perf_counter() - t0
    out = {"environment": environment(), "config": cfg, "verdict": verdict, "rules": rules,
           "shift_attribution": attribution, "results": results, "wall_seconds": wall}
    save_json(out, cfg["output_json"])

    def ms(vals):
        return f"{np.mean(vals):.3f} ± {np.std(vals, ddof=1):.3f}"

    lines = [f"# Final test evaluation\n\n**Verdict (pre-registered, raw readout, {main_group} vs {ctrl}): {verdict}**\n"]
    for readout in cfg["readouts"]:
        groups = list(cfg["models"])
        lines += [f"## Test-ID, readout: {readout}", "", "| quantity | " + " | ".join(groups) + " |", "|---|" + "---|" * len(groups)]
        rows = [(t, lambda r, t=t: r["probes"][t]["r2_partial"]) for t in PROBE_TARGETS]
        rows += [("log ν pair R² (S2)", lambda r: r["nu"]["pair_r2"]), ("log ν single-frame R²", lambda r: r["nu"]["single_r2"]),
                 ("time (plain R²)", lambda r: r["probes"]["time"]["r2_plain"])]
        rows += [(f"slow |r| {k}", lambda r, k=k: r["slow"]["max_abs_r"][k])
                 for k in ("log_E", "log_Z", "log_C4", "kurtosis (traj mean)", "log max|w| (traj mean)", "OW fraction (traj mean)")]
        if readout == "raw" and cfg["mlp"]:
            rows += [(f"MLP {t}", lambda r, t=t: r["mlp_r2_partial"][t]) for t in S1_TARGETS]
        for name, get in rows:
            lines.append(f"| {name} | " + " | ".join(ms([get(r) for r in results[readout][g]]) for g in groups) + " |")
        lines.append("")
        for g, v in rules[readout].items():
            s1, s2, s3 = v["S1"], v["S2"], v["S3"]
            lines.append(f"- **{g} vs {ctrl}**: S1 {'holds' if s1['holds'] else 'fails'} ({s1['n_counted']}/5; OW fraction "
                         f"{'counts' if s1['mandatory_counts'] else 'does not count'}); S2 {'holds' if s2['holds'] else 'fails'} "
                         f"({s2['r2_jepa']:.3f} vs {s2['r2_control']:.3f}); S3 {'holds' if s3['holds'] else 'fails'} "
                         f"(Z: {s3['invariants']['log_Z']['r_jepa']:.3f} vs {s3['invariants']['log_Z']['r_control']:.3f}, "
                         f"CI of gap {s3['invariants']['log_Z']['ci95'][0]:.2f}–{s3['invariants']['log_Z']['ci95'][1]:.2f}) → {v['verdict']}")
        lines.append("")
    with open(cfg["output_md"], "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    append_log("FINAL test evaluation (single test-set touch)", cfg["_config_path"], command_line(), f"verdict: {verdict}",
               {"results": cfg["output_json"], "tables": cfg["output_md"]}, wall, environment().get("gpu"))


if __name__ == "__main__":
    main()
