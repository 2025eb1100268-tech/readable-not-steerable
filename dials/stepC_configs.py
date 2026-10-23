"""[BUILD] Step C training configs (results/build/PREDICTIONS.md entry 9).

    python -m dials.stepC_configs

Step B's tau = 0.5 runs with the resize-convolution decoder (decoder_upsample: resize), seeds 0-2;
everything else identical (shape-noise content input, omega-only loss, recipe).
"""
from __future__ import annotations

import yaml

from common.run import REPO


def main():
    out_dir = REPO / "configs" / "build" / "stepC"
    out_dir.mkdir(parents=True, exist_ok=True)
    for seed in range(3):
        src = REPO / "configs" / "build" / "stepB" / f"b3_sn_t0p5_s{seed}.yaml"
        cfg = yaml.safe_load(src.read_text(encoding="utf-8"))
        run = f"b4_rc_t0p5_s{seed}"
        cfg.update({"results_dir": "results/build/stepC/runs", "log_title": "[BUILD] Step C training", "decoder_upsample": "resize",
                    "run_name": run})
        head = (f"# [BUILD] Step C (python -m train.train --config configs/build/stepC/{run}.yaml); PREDICTIONS entry 9\n"
                f"# = configs/build/stepB/b3_sn_t0p5_s{seed}.yaml with the resize-convolution decoder; dials/stepC_configs.py\n")
        (out_dir / f"{run}.yaml").write_text(head + yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


if __name__ == "__main__":
    main()
