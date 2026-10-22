"""[BUILD] Stage 2b training configs (results/build/PREDICTIONS.md entry 5).

    python -m dials.stage2b_configs

M2-SW seeds 0-2 (content encoder on the spectrally whitened input, cap 1000) and one M2-SF seed (0)
for attribution; both with the omega-only loss: the omega weight of Stages 1-2 (14.699) and psi
weight 0 (psi is derived from omega and reported, not trained).
"""
from __future__ import annotations

import yaml

from common.run import REPO
from dials.stage1_configs import BASE, head_weights

RUNS = {"b2b_m2sw_s0": ("whiten", 0), "b2b_m2sw_s1": ("whiten", 1), "b2b_m2sw_s2": ("whiten", 2), "b2b_m2sfw_s0": ("rms", 0)}


def main():
    base = yaml.safe_load(BASE.read_text(encoding="utf-8"))
    weights, _ = head_weights()
    weights = {"vorticity": weights["vorticity"], "streamfunction": 0.0}
    out_dir = REPO / "configs" / "build" / "stage2b"
    out_dir.mkdir(parents=True, exist_ok=True)
    for run, (content_input, seed) in RUNS.items():
        cfg = {k: v for k, v in base.items() if k not in ("model", "ae_target", "seed", "run_name", "loss_weights",
                                                           "results_dir", "log_title")}
        cfg.update({"loss_weights": {"head": weights, "var": base["loss_weights"]["var"], "cov": base["loss_weights"]["cov"]},
                    "results_dir": "results/build/stage2b/runs", "log_title": "[BUILD] Stage 2b training",
                    "model": "branch_ae", "content_dim": 120, "n_dial": 8, "content_input": content_input})
        if content_input == "whiten":
            cfg["whiten_cap"] = 1000
        cfg.update({"seed": seed, "run_name": run})
        head = (f"# [BUILD] Stage 2b (python -m train.train --config configs/build/stage2b/{run}.yaml); PREDICTIONS entry 5\n"
                "# omega-only loss (psi derived and reported); dials/stage2b_configs.py\n")
        (out_dir / f"{run}.yaml").write_text(head + yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(weights)


if __name__ == "__main__":
    main()
