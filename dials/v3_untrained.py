"""[BUILD] v3 untrained control (PREREGISTRATION_v3.md): the main model's architecture at initialization, with
a randomly initialized FiLM.

    python -m dials.v3_untrained --config configs/build/v3/b6_untrained_s300.yaml

Everything is built exactly as train.train builds the main model for the same seed (encoders: Kaiming
normal, zero biases; decoder convolutions: Kaiming normal; target scales and shape-noise sigmas from
the same v1 train sample). The FiLM layers, which training starts at zero (so an untrained model's dials
would change nothing), are re-initialized with PyTorch's default nn.Linear initialization (weights and
biases uniform on +-1/sqrt(8), fan-in 8) under torch.manual_seed(film_init_seed). No training step is
taken. Writes the checkpoint and a record (no validation loss: the untrained control is not a rendering model).
"""
from __future__ import annotations

import argparse

import torch

from common.run import REPO, append_log, command_line, environment, load_config, save_json, seed_everything
from data.dataset import Dataset
from dials.shape_noise import data_sigmas
from dials.split import TARGETS
from solver.ns2d import SpectralGrid
from train.train import Model, SplitData, ae_field


def build(cfg: dict) -> Model:
    seed_everything(cfg["seed"])
    if "ae_target_scales" not in cfg or "shape_noise_sigmas" not in cfg:
        tr = SplitData(Dataset(cfg["data_dir"]), "train", "cuda")
        sample = tr.x[torch.arange(0, tr.n, max(1, tr.n // 4096), device="cuda")]
        cfg["ae_target_scales"] = {t: float(ae_field(sample, t, SpectralGrid(64, "cuda", dtype=torch.float32)).std()) for t in TARGETS}
        cfg["shape_noise_sigmas"] = data_sigmas(sample)
    model = Model(cfg)
    assert cfg["film_init"] == "pytorch_default_linear"
    torch.manual_seed(cfg["film_init_seed"])
    for f in model.head.dec.film:
        f.reset_parameters()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = load_config(ap.parse_args().config)
    model = build(cfg)
    ckpt = REPO / "results" / "checkpoints" / f"{cfg['run_name']}.pt"
    torch.save({"config": cfg, "model": model.state_dict()}, ckpt)
    out = REPO / cfg["results_dir"] / f"{cfg['run_name']}.json"
    save_json({"config": cfg, "environment": environment(), "untrained": True, "checkpoint": str(ckpt.relative_to(REPO).as_posix())}, out)
    append_log(f"{cfg['log_title']}: {cfg['run_name']}", cfg["_config_path"], command_line(), "ok",
               {"FiLM init": f"{cfg['film_init']} (seed {cfg['film_init_seed']})", "results": str(out.relative_to(REPO).as_posix())},
               0.0, environment().get("gpu"))


if __name__ == "__main__":
    main()
