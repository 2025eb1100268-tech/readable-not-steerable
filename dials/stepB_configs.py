"""[BUILD] Step B training configs (results/build/PREDICTIONS.md entry 8).

    python -m dials.stepB_configs

M2-SF-style models (scale-free content, dial encoder on the raw input, one FiLM decoder rendering omega)
with the shape-noise content input at tau in {0, 0.5, 1, 2}, seeds 0-2, omega-only loss (as Stage 2b).
"""
from __future__ import annotations

import yaml

from common.run import REPO
from dials.stage1_configs import BASE, head_weights

TAUS = {"0": 0.0, "0p5": 0.5, "1": 1.0, "2": 2.0}


def main():
    base = yaml.safe_load(BASE.read_text(encoding="utf-8"))
    weights, _ = head_weights()
    weights = {"vorticity": weights["vorticity"], "streamfunction": 0.0}
    out_dir = REPO / "configs" / "build" / "stepB"
    out_dir.mkdir(parents=True, exist_ok=True)
    for tag, tau in TAUS.items():
        for seed in range(3):
            run = f"b3_sn_t{tag}_s{seed}"
            cfg = {k: v for k, v in base.items() if k not in ("model", "ae_target", "seed", "run_name", "loss_weights",
                                                               "results_dir", "log_title")}
            cfg.update({"loss_weights": {"head": weights, "var": base["loss_weights"]["var"], "cov": base["loss_weights"]["cov"]},
                        "results_dir": "results/build/stepB/runs", "log_title": "[BUILD] Step B training",
                        "model": "branch_ae", "content_dim": 120, "n_dial": 8, "content_input": "shape_noise",
                        "shape_noise_tau": tau, "seed": seed, "run_name": run})
            head = (f"# [BUILD] Step B (python -m train.train --config configs/build/stepB/{run}.yaml); PREDICTIONS entry 8\n"
                    "# shape-noise content input, omega-only loss; dials/stepB_configs.py\n")
            (out_dir / f"{run}.yaml").write_text(head + yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


if __name__ == "__main__":
    main()
