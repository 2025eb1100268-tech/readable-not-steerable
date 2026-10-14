"""Label-free model selection (PREREGISTRATION.md §4.6 + amendment A1).

    python -m train.select --model jepa_spatial --runs jepa_spatial_lr0.0003_s0 jepa_spatial_lr0.001_s0

Discards collapsed runs (rule applies to collapse-prone objectives only), picks the lowest val
metric (loss / copy baseline; invariance loss for the static control), writes
results/phase2/selection_<model>.json and the seed-1 and seed-2 configs for the winner.
No probe and no label is read.
"""
from __future__ import annotations

import argparse
import json

import yaml

from common.run import REPO, append_log, command_line, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2])
    args = ap.parse_args()
    rows = []
    for name in args.runs:
        with open(REPO / "results" / "phase2" / f"{name}.json", encoding="utf-8") as f:
            r = json.load(f)
        rows.append({"run": name, "lr": r["config"]["lr"], "metric": r["final"]["metric"], "collapsed": r["collapsed"],
                     "effective_rank": r["final"]["effective_rank"], "latent_std_median": r["final"]["latent_std_median"]})
    alive = [r for r in rows if not r["collapsed"]]
    out = {"model": args.model, "candidates": rows}
    if not alive:
        out["selected"] = None
        save_json(out, REPO / "results" / "phase2" / f"selection_{args.model}.json")
        append_log(f"Phase 2 selection: {args.model}", "-", command_line(),
                   "FAILED: every configuration collapsed; stopping to report (pre-registered rule)", {"candidates": rows})
        raise SystemExit(f"all {args.model} configurations collapsed")
    best = min(alive, key=lambda r: r["metric"])
    out["selected"] = best
    save_json(out, REPO / "results" / "phase2" / f"selection_{args.model}.json")
    with open(REPO / "configs" / "phase2" / f"{best['run']}.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    written = []
    for s in args.seeds:
        c = dict(cfg, seed=s, run_name=cfg["run_name"].rsplit("_s", 1)[0] + f"_s{s}")
        path = REPO / "configs" / "phase2" / f"{c['run_name']}.yaml"
        path.write_text(f"# Phase 2 run, selected config (python -m train.train --config configs/phase2/{c['run_name']}.yaml)\n"
                        + yaml.safe_dump(c, sort_keys=False), encoding="utf-8")
        written.append(path.name)
    append_log(f"Phase 2 selection: {args.model}", "-", command_line(), f"selected {best['run']} (lr {best['lr']:g})",
               {"candidates (run: metric, collapsed)": "; ".join(f"{r['run']}: {r['metric']:.4f}, {r['collapsed']}" for r in rows),
                "configs written": ", ".join(written)})
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
