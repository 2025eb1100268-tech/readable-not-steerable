"""[STAGE 3] Write configs/explore/stage3_confirm.yaml (agreed change 14, PREREGISTRATION_v2.md),
with every checkpoint's SHA-256, after the blind Stage 3 training and before the pre-registration is
committed. --dry writes a stand-in config instead (old models only, for a code check on explore).

    python -m explore.stage3_config [--dry PATH]
"""
from __future__ import annotations

import argparse
import hashlib
import json

import yaml

from common.run import REPO

AE = "s2_ae_{}_k0_lr0.001_s{}"


def groups(dry: bool) -> dict:
    fresh = (0, 1, 2) if dry else (3, 4, 5)  # dry run: stand-ins from Stage 2, never the blind models
    g = {"untrained (seeds 3-5)": [f"untrained_s{s}" for s in ((0, 1, 2) if dry else (3, 4, 5))]}
    for t in ("velocity", "vorticity"):
        g[f"AE {t} (fresh)"] = [AE.format(t, s) for s in fresh]
        g[f"AE {t} + VICReg (fresh)"] = [AE.format(t + "_vicreg", s) for s in fresh]
    g["untrained"] = [f"untrained_s{s}" for s in range(3)]
    for t in ("velocity", "vorticity", "streamfunction"):
        g[f"AE {t}"] = [AE.format(t, s) for s in range(3)]
        g[f"AE {t} + VICReg"] = [AE.format(t + "_vicreg", s) for s in range(3)]
    g["imaginer k=4"] = [f"s2_imaginer_k4_lr0.001_s{s}" for s in range(3)]
    g["imaginer k=4 + VICReg"] = ([f"s2_summarizer_k4_lr0.001_s{s}" for s in range(3)] if dry
                                  else [f"s2_pixel_vicreg_k4_lr0.001_s{s}" for s in range(3)])
    g["summarizer k=4"] = [f"s2_summarizer_k4_lr0.001_s{s}" for s in range(3)]
    return g


def sha(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 24), b""):
            h.update(c)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry")
    a = ap.parse_args()
    g = groups(bool(a.dry))
    ck = {r: sha(REPO / "results" / "checkpoints" / f"{r}.pt") for rs in g.values() for r in rs if not r.startswith("untrained")}
    with open(REPO / "data" / "generated" / "confirm_v2" / "manifest.json", encoding="utf-8") as f:
        conf = json.load(f)["sha256"]
    scratch = a.dry.rsplit("/", 1)[0] if a.dry else None
    cfg = {"seed": 0,
           "data_parts": [["data/generated/v1", ["train", "val"]], ["data/generated/confirm_v2", ["confirm"]]],
           "confirm_dir": "data/generated/confirm_v2", "eval_split": "confirm", "cache_tag": "v1tv_confirm",
           "confirm_sha256": conf, "n_pca": 32, "n_boot": 1000, "groups": g,
           "hypotheses": {
               "H1": {"groups": ["AE velocity (fresh)", "AE vorticity (fresh)"], "control": "untrained (seeds 3-5)", "min_e1": 0.70, "min_gap": 0.25},
               "H2": {"pairs": [["AE velocity (fresh)", "AE velocity + VICReg (fresh)"], ["AE vorticity (fresh)", "AE vorticity + VICReg (fresh)"]],
                      "min_drop": 0.20},
               "H3": {"closer_to_one": "AE vorticity", "farther": "AE streamfunction", "min_ratio_farther": 2.0},
               "H4": {"pair": ["imaginer k=4", "imaginer k=4 + VICReg"], "min_drop": 0.20},
               "replication": {"H1_groups": ["AE velocity", "AE vorticity"], "control": "untrained",
                               "H2_pairs": [["AE velocity", "AE velocity + VICReg"], ["AE vorticity", "AE vorticity + VICReg"]]}},
           "checkpoint_sha256": ck,
           "outputs": {"md": "results/explore/stage3/confirm.md", "json": "results/explore/stage3/confirm.json"},
           "dry_run_dir": "data/generated/explore_v2", "dry_run_cache_tag": "v1tv_explore_s3dry",
           "dry_run_outputs": {"md": f"{scratch}/stage3_dry.md", "json": f"{scratch}/stage3_dry.json"} if scratch else None}
    out = a.dry or str(REPO / "configs" / "explore" / "stage3_confirm.yaml")
    with open(out, "w", encoding="utf-8") as f:
        f.write("# STAGE 3 confirmation config (python -m explore.stage3_confirm --config configs/explore/stage3_confirm.yaml)\n"
                "# Frozen with PREREGISTRATION_v2.md: groups, thresholds, checkpoint and data checksums. Run once.\n"
                if not a.dry else "# DRY-RUN stand-in config (old models only; code check on explore)\n")
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    print(f"wrote {out}: {len(ck)} checkpoints")


if __name__ == "__main__":
    main()
