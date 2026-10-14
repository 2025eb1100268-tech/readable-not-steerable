"""Phase 3: probes on val only (preview). The test sets are not touched here.

    python -m probes.phase3_val --config configs/phase3_val.yaml

For every listed run (trained models and the untrained control, 3 seeds each) and readout:
ridge probes (S1 targets and others), the S2 frame-pair probe, MLP probes for the S1 targets,
and the slow-direction analysis fitted on train and scored on val. Applies the pre-registered
rules to val as a *preview only*: ridge alpha is chosen on the same val set it is scored on, so
these numbers are optimistic and decide nothing.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from common.run import append_log, command_line, environment, load_config, save_json, seed_everything
from data.dataset import S1_TARGETS, Dataset
from probes import criteria
from probes.evaluate import PROBE_TARGETS, probe_nu, probe_targets, slow_directions
from probes.latents import latents_for
from probes.mlp import mlp_probe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = load_config(ap.parse_args().config)
    seed_everything(cfg["seed"])
    t0 = time.perf_counter()
    ds = Dataset(cfg["data_dir"])
    rows = ds.rows(["train", "val"])
    results = {}
    for readout in cfg["readouts"]:
        results[readout] = {}
        for group, runs in cfg["models"].items():
            results[readout][group] = []
            for run in runs:
                lat = latents_for(run, ds, readout, rows=rows)
                r = {"run": run, "probes": probe_targets(ds, lat, "val"), "nu": probe_nu(ds, lat, "val"),
                     "slow": slow_directions(ds, lat, ["train"], "val", cfg["n_pca"])}
                if cfg["mlp"] and readout == "raw":
                    r["mlp_r2_partial"] = {t: mlp_probe(ds, lat, t, "val", seed=cfg["seed"]) for t in S1_TARGETS}
                results[readout][group].append(r)
                print(f"{readout} {run} done ({time.perf_counter() - t0:.0f} s)", flush=True)

    # summaries and val preview of the rules for each trained group vs the untrained control
    summary = {}
    for readout, groups in results.items():
        summary[readout] = {}
        for group, runs in groups.items():
            s = {t: [r["probes"][t]["r2_partial"] for r in runs] for t in PROBE_TARGETS}
            s["time (plain R²)"] = [r["probes"]["time"]["r2_plain"] for r in runs]
            s["log_nu pair (S2)"] = [r["nu"]["pair_r2"] for r in runs]
            s["log_nu single"] = [r["nu"]["single_r2"] for r in runs]
            for inv in ("log_E", "log_Z", "log_C4", "kurtosis (traj mean)", "OW fraction (traj mean)"):
                s[f"slow |r| {inv}"] = [r["slow"]["max_abs_r"][inv] for r in runs]
            if "mlp_r2_partial" in runs[0]:
                for t in S1_TARGETS:
                    s[f"MLP {t}"] = [r["mlp_r2_partial"][t] for r in runs]
            summary[readout][group] = {k: {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1))} for k, v in s.items()}
        ctrl = groups[cfg["control_group"]]
        summary[readout]["rules_val_preview"] = {}
        for group, runs in groups.items():
            if group == cfg["control_group"]:
                continue
            a, b, c = criteria.s1(runs, ctrl), criteria.s2(runs, ctrl), criteria.s3(runs, ctrl)
            summary[readout]["rules_val_preview"][group] = {"S1": a, "S2": b, "S3": c, "verdict_preview": criteria.verdict(a, b, c)}
    wall = time.perf_counter() - t0
    save_json({"environment": environment(), "config": cfg, "summary": summary, "results": results}, cfg["output_json"])

    lines = []
    for readout in cfg["readouts"]:
        groups = list(cfg["models"])
        lines += [f"## Readout: {readout}", "", "| quantity | " + " | ".join(groups) + " |", "|---|" + "---|" * len(groups)]
        for k in summary[readout][groups[0]]:
            lines.append(f"| {k} | " + " | ".join(f"{summary[readout][g][k]['mean']:.3f} ± {summary[readout][g][k]['std']:.3f}" for g in groups) + " |")
        lines.append("")
        for g, v in summary[readout]["rules_val_preview"].items():
            s1 = v["S1"]
            lines.append(f"- **{g} vs control (val preview)**: S1 {'holds' if s1['holds'] else 'fails'} "
                         f"({s1['n_counted']}/5 count; OW fraction {'counts' if s1['mandatory_counts'] else 'does not count'}), "
                         f"S2 {'holds' if v['S2']['holds'] else 'fails'} (pair R² {v['S2']['r2_jepa']:.3f} vs {v['S2']['r2_control']:.3f}), "
                         f"S3 {'holds' if v['S3']['holds'] else 'fails'} (Z |r| {v['S3']['invariants']['log_Z']['r_jepa']:.3f} vs "
                         f"{v['S3']['invariants']['log_Z']['r_control']:.3f}) → {v['verdict_preview']}")
        lines.append("")
    md = "\n".join(lines)
    with open(cfg["output_md"], "w", encoding="utf-8") as f:
        f.write("# Phase 3 val preview (decides nothing; alpha chosen on the same val set)\n\n" + md)
    print(md)
    append_log("Phase 3 probes, val preview", cfg["_config_path"], command_line(), "ok",
               {"preview verdicts (raw readout)": {g: v["verdict_preview"] for g, v in summary["raw"]["rules_val_preview"].items()},
                "results": cfg["output_json"]}, wall, environment().get("gpu"))


if __name__ == "__main__":
    main()
