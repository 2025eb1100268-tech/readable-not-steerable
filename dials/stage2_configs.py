"""[BUILD] Stage 2 training configs: M2-SF, seeds 0-2 (results/build/PREDICTIONS.md entry 4).

    python -m dials.stage2_configs

Same recipe, data and head-loss weights as Stage 1 (dials/stage1_configs.py); the psi term is
computed on the psi derived from the single rendered omega.
"""
from __future__ import annotations

import yaml

from common.run import REPO
from dials.stage1_configs import BASE, head_weights


def main():
    base = yaml.safe_load(BASE.read_text(encoding="utf-8"))
    weights, ref = head_weights()
    out_dir = REPO / "configs" / "build" / "stage2"
    out_dir.mkdir(parents=True, exist_ok=True)
    for seed in range(3):
        cfg = {k: v for k, v in base.items() if k not in ("model", "ae_target", "seed", "run_name", "loss_weights",
                                                           "results_dir", "log_title")}
        run = f"b2_m2sf_s{seed}"
        cfg.update({"loss_weights": {"head": weights, "var": base["loss_weights"]["var"], "cov": base["loss_weights"]["cov"]},
                    "results_dir": "results/build/stage2/runs", "log_title": "[BUILD] Stage 2 training (M2-SF)",
                    "model": "branch_ae", "content_dim": 120, "n_dial": 8, "seed": seed, "run_name": run})
        head = (f"# [BUILD] Stage 2 (python -m train.train --config configs/build/stage2/{run}.yaml); PREDICTIONS entry 4\n"
                f"# head weights as Stage 1 (VICReg-AE val normalized MSE {ref['vorticity']:.5f} omega, "
                f"{ref['streamfunction']:.5f} psi): dials/stage2_configs.py\n")
        (out_dir / f"{run}.yaml").write_text(head + yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(weights)


if __name__ == "__main__":
    main()
