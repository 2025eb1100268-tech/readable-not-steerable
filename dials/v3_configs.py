"""[BUILD] v3 model configs (PREREGISTRATION_v3.md).

    python -m dials.v3_configs

Main model: Step D's design (configs/build/stepD/b5_rc_t0p2_s0.yaml: two-branch model, shape-noise content input
at tau = 0.2, resize-convolution decoder, omega-only loss, the autoencoder recipe), fresh seeds 100-104.
tau = 0 control: the same with shape_noise_tau 0, fresh seeds 200-202. Untrained control: the main
architecture at initialization with a randomly initialized FiLM (dials/v3_untrained.py), seeds 300-302.
"""
from __future__ import annotations

import yaml

from common.run import REPO

SOURCE = REPO / "configs" / "build" / "stepD" / "b5_rc_t0p2_s0.yaml"
GROUPS = {"main": (0.2, range(100, 105)), "tau0": (0.0, range(200, 203)), "untrained": (0.2, range(300, 303))}


def main():
    base = yaml.safe_load(SOURCE.read_text(encoding="utf-8"))
    out = REPO / "configs" / "build" / "v3"
    out.mkdir(parents=True, exist_ok=True)
    for group, (tau, seeds) in GROUPS.items():
        for seed in seeds:
            run = f"b6_{group}_s{seed}"
            cfg = {**base, "shape_noise_tau": tau, "seed": seed, "run_name": run, "results_dir": "results/build/v3/runs",
                   "log_title": f"[BUILD] v3 {'training' if group != 'untrained' else 'untrained control'} ({group})"}
            if group == "untrained":
                cfg.update({"untrained": True, "film_init": "pytorch_default_linear", "film_init_seed": seed + 2000})
            head = (f"# [BUILD] v3, group {group} (PREREGISTRATION_v3.md); dials/v3_configs.py from {SOURCE.relative_to(REPO).as_posix()}\n"
                    + (f"# python -m train.train --config configs/build/v3/{run}.yaml\n" if group != "untrained"
                       else f"# python -m dials.v3_untrained --config configs/build/v3/{run}.yaml\n"))
            (out / f"{run}.yaml").write_text(head + yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


if __name__ == "__main__":
    main()
