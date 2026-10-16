"""[EXPLORE] Stage 2 run configs and the label-free lr selection (results/explore/PREDICTIONS.md
entries 3-4). Every step writes YAML files under configs/explore/stage2/ and logs to results/LOG.md.

    python -m explore.stage2_configs selection            # 4 configs: imaginer / summarizer, k = 4, lr grid, seed 0
    python -m explore.stage2_configs select               # pick lr per family; write horizon x seed and bridge configs
    python -m explore.stage2_configs kstar --k K          # fake-physics imaginer and pixel + VICReg configs at k*
    python -m explore.stage2_configs override --family imaginer --lr 0.001   # PREDICTIONS entry 5
"""
from __future__ import annotations

import argparse
import json

import yaml

from common.run import REPO, append_log, command_line, save_json

CFG_DIR = REPO / "configs" / "explore" / "stage2"
RUNS = "results/explore/stage2/runs"
HORIZONS = (1, 4, 16, 32)
SEEDS = (0, 1, 2)
LR_GRID = (0.001, 0.003)
FAMILIES = {"imaginer": "pixel", "summarizer": "jepa_spatial"}

BASE = {
    "data_dir": "data/generated/v1", "weight_decay": 0.05, "batch": 256, "amp_bf16": True, "ema_start": 0.996,
    "loss_weights": {"pred": 25.0, "var": 25.0, "cov": 1.0}, "predictor_width": 256, "log_every": 100,
    "collapse": {"min_effective_rank": 8, "min_std_median": 0.1},
    "anchor_rule": "long_windows", "anchor_max_t": 31, "anchor_min_frames": 64,
    "schedule": "plateau", "warmup_steps": 575, "eval_every_steps": 230, "max_steps": 12000,
    "plateau": {"min_rel_improve": 0.005, "patience": 4, "factor": 0.3, "max_reductions": 2, "start_step": 2300},
    "diagnostics": True, "results_dir": RUNS, "log_title": "[EXPLORE] Stage 2 training",
}


def effective_lr() -> dict:
    """lr per family: results/explore/stage2/lr_final.json if present (entry 5), else the label-free selection."""
    s2 = REPO / "results" / "explore" / "stage2"
    if (s2 / "lr_final.json").exists():
        with open(s2 / "lr_final.json", encoding="utf-8") as f:
            return {fam: v["lr"] for fam, v in json.load(f).items()}
    with open(s2 / "lr_selection.json", encoding="utf-8") as f:
        return {fam: v["selected_lr"] for fam, v in json.load(f).items()}


def name(family: str, k: int, lr: float, seed: int) -> str:
    return f"s2_{family}_k{k}_lr{lr:g}_s{seed}"


def write(family: str, model: str, k: int, lr: float, seed: int, **extra) -> str:
    run = name(family, k, lr, seed)
    cfg = {**BASE, "model": model, "horizons": [k], "lr": lr, "seed": seed, "run_name": run, **extra}
    CFG_DIR.mkdir(parents=True, exist_ok=True)
    (CFG_DIR / f"{run}.yaml").write_text(f"# [EXPLORE] Stage 2 run (python -m train.train --config configs/explore/stage2/{run}.yaml)\n"
                                         + yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return run


def selection():
    runs = [write(f, m, 4, lr, 0) for f, m in FAMILIES.items() for lr in LR_GRID]
    print("\n".join(runs))


def select():
    """Label-free: lowest final val loss / copy baseline at k = 4 among non-collapsed seed-0 runs."""
    out, written = {}, []
    for family, model in FAMILIES.items():
        rows = []
        for lr in LR_GRID:
            with open(REPO / RUNS / f"{name(family, 4, lr, 0)}.json", encoding="utf-8") as f:
                r = json.load(f)
            rows.append({"run": r["config"]["run_name"], "lr": lr, "metric": r["final"]["metric"], "collapsed": r["collapsed"],
                         "steps": r["steps"], "stop_reason": r["stop_reason"]})
        alive = [r for r in rows if not r["collapsed"]]
        if not alive:
            raise SystemExit(f"every {family} configuration collapsed: stop and report")
        best = min(alive, key=lambda r: r["metric"])
        out[family] = {"candidates": rows, "selected_lr": best["lr"]}
        for k in HORIZONS:
            for s in SEEDS:
                if not (k == 4 and s == 0):
                    written.append(write(family, model, k, best["lr"], s))
    lr_img = out["imaginer"]["selected_lr"]
    written += [write("bridge", "pixel", 4, lr_img, s, anchor_rule="original") for s in SEEDS]
    save_json(out, REPO / "results" / "explore" / "stage2" / "lr_selection.json")
    append_log("[EXPLORE] Stage 2 lr selection (label-free, k = 4, seed 0)", "-", command_line(),
               ", ".join(f"{f}: lr {v['selected_lr']:g}" for f, v in out.items()),
               {"candidates (run: metric, collapsed)": "; ".join(f"{r['run']}: {r['metric']:.4f}, {r['collapsed']}"
                                                                 for v in out.values() for r in v["candidates"]),
                "configs written": len(written)})
    print("\n".join(written))


def override(family: str, lr: float):
    """[EXPLORE] entry 5: set the imaginer family's lr; write its horizon x seed and bridge configs and
    print the runs that have not been trained yet."""
    assert family == "imaginer", "entry 5 overrides the imaginer family only"
    with open(REPO / "results" / "explore" / "stage2" / "lr_selection.json", encoding="utf-8") as f:
        sel = json.load(f)
    final = {"imaginer": {"lr": lr, "source": "override (PREDICTIONS entry 5): 8 of 15 imaginer-family runs diverged at the selected lr 3e-3"},
             "summarizer": {"lr": sel["summarizer"]["selected_lr"], "source": "label-free selection (entry 3)"}}
    save_json(final, REPO / "results" / "explore" / "stage2" / "lr_final.json")
    runs = [write("imaginer", "pixel", k, lr, s) for k in HORIZONS for s in SEEDS]
    runs += [write("bridge", "pixel", 4, lr, s, anchor_rule="original") for s in SEEDS]
    todo = [r for r in runs if not (REPO / RUNS / f"{r}.json").exists()]
    append_log("[EXPLORE] Stage 2 imaginer lr override (PREDICTIONS entry 5)", "-", command_line(), f"imaginer family lr {lr:g}",
               {"configs written": len(runs), "runs to train": len(todo)})
    print("\n".join(todo))


def kstar(k: int):
    lr = effective_lr()["imaginer"]
    runs = [write("fake_imaginer", "pixel", k, lr, s, data_dir="data/generated/nonphys_v1") for s in SEEDS]
    runs += [write("pixel_vicreg", "pixel_vicreg", k, lr, s) for s in SEEDS]
    print("\n".join(runs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["selection", "select", "kstar", "override"])
    ap.add_argument("--k", type=int)
    ap.add_argument("--family")
    ap.add_argument("--lr", type=float)
    a = ap.parse_args()
    {"selection": selection, "select": select, "kstar": lambda: kstar(a.k), "override": lambda: override(a.family, a.lr)}[a.step]()


if __name__ == "__main__":
    main()
