"""[BUILD] Stage 1 training configs (results/build/PREDICTIONS.md entry 3).

    python -m dials.stage1_configs

M1-spec and M1-G, seeds 0-2, with the autoencoder recipe of the Stage 2 VICReg autoencoders
(cosine decay, gradient clipping 1.0, 30 epochs, peak lr 1e-3, same data and anchors).
Head-loss rule: each head's MSE is weighted inversely to its VICReg-autoencoder counterpart's final
val normalized MSE (mean over seeds 0-2), with the geometric mean of the two weights equal to the
Stage 2 rendering weight 25, so at the counterparts' quality both heads contribute equally.
"""
from __future__ import annotations

import json
import math

import yaml

from common.run import REPO

BASE = REPO / "configs" / "explore" / "stage2" / "s2_ae_vorticity_vicreg_k0_lr0.001_s0.yaml"
REF = "results/explore/stage2/runs/s2_ae_{t}_vicreg_k0_lr0.001_s{s}.json"
MODELS = {"m1spec": {"dial_mode": "spatial", "content_norm": "none"},
          "m1g": {"dial_mode": "film", "content_norm": "rms"}}


def head_weights(w_pred: float = 25.0) -> tuple[dict, dict]:
    ref = {t: sum(json.loads((REPO / REF.format(t=t, s=s)).read_text())["final"]["metric"] for s in range(3)) / 3
           for t in ("vorticity", "streamfunction")}
    gm = math.sqrt(ref["vorticity"] * ref["streamfunction"])
    return {t: round(w_pred * gm / r, 4) for t, r in ref.items()}, ref


def main():
    base = yaml.safe_load(BASE.read_text(encoding="utf-8"))
    weights, ref = head_weights()
    out_dir = REPO / "configs" / "build" / "stage1"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, spec in MODELS.items():
        for seed in range(3):
            cfg = {k: v for k, v in base.items() if k not in ("model", "ae_target", "seed", "run_name", "loss_weights",
                                                               "results_dir", "log_title")}
            run = f"b1_{name}_s{seed}"
            cfg.update({"loss_weights": {"head": weights, "var": base["loss_weights"]["var"], "cov": base["loss_weights"]["cov"]},
                        "results_dir": "results/build/stage1/runs", "log_title": "[BUILD] Stage 1 training",
                        "model": "split_ae", **spec, "n_dial_per_group": 4, "seed": seed, "run_name": run})
            head = (f"# [BUILD] Stage 1 (python -m train.train --config configs/build/stage1/{run}.yaml); PREDICTIONS entry 3\n"
                    f"# head weights from VICReg-AE val normalized MSE {ref['vorticity']:.5f} (omega), "
                    f"{ref['streamfunction']:.5f} (psi): dials/stage1_configs.py\n")
            (out_dir / f"{run}.yaml").write_text(head + yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(weights, ref)


if __name__ == "__main__":
    main()
