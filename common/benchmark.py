"""Phase 0 hardware benchmark: encoder training cost, solver step cost, shift-averaged readout cost.

The encoder here is a provisional stand-in with the planned size (circular padding, GELU, no
normalization, Kaiming init, three stride-2 stages to an 8x8x128 map, global average pool);
Phase 2 defines the final architecture.

    python -m common.benchmark --config configs/phase0_benchmark.yaml
"""
from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn

from common.run import append_log, command_line, environment, load_config, save_json, seed_everything, set_threads
from solver.initial_conditions import spectral_field
from solver.ns2d import NavierStokes2D, SpectralGrid


def provisional_encoder(widths=(32, 64, 128, 128), d: int = 128) -> nn.Module:
    layers, c_in = [nn.Conv2d(2, widths[0], 3, padding=1, padding_mode="circular"), nn.GELU()], widths[0]
    for w in widths[1:]:
        layers += [nn.Conv2d(c_in, w, 3, stride=2, padding=1, padding_mode="circular"), nn.GELU(),
                   nn.Conv2d(w, w, 3, padding=1, padding_mode="circular"), nn.GELU()]
        c_in = w
    layers.append(nn.Conv2d(c_in, d, 1))
    net = nn.Sequential(*layers)
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            nn.init.zeros_(m.bias)
    return net


def _sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def time_encoder(device: str, batch: int, reps: int, amp: bool) -> dict:
    net = provisional_encoder().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    x = torch.randn(batch, 2, 64, 64, device=device)
    dtype = torch.bfloat16

    def train_step():
        with torch.autocast(device_type=device, dtype=dtype, enabled=amp):
            loss = net(x).mean(dim=(2, 3)).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    for _ in range(3):
        train_step()
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(reps):
        train_step()
    _sync(device)
    fb = (time.perf_counter() - t0) / reps / batch
    with torch.no_grad(), torch.autocast(device_type=device, dtype=dtype, enabled=amp):
        for _ in range(3):
            net(x)
        _sync(device)
        t0 = time.perf_counter()
        for _ in range(reps):
            net(x)
        _sync(device)
    fwd = (time.perf_counter() - t0) / reps / batch
    params = sum(p.numel() for p in net.parameters())
    return {"device": device, "amp_bf16": amp, "batch": batch, "params": params,
            "fwd_bwd_ms_per_frame": 1e3 * fb, "fwd_ms_per_frame": 1e3 * fwd}


def time_solver(device: str, n: int, batch: int, dtype: torch.dtype, steps: int) -> dict:
    g = SpectralGrid(n, device=device, dtype=dtype)
    import numpy as np
    rng = np.random.default_rng(0)
    w0 = spectral_field(g, rng, 4.0, 0.5)
    w_hat = w0.expand(batch, -1, -1).clone()
    solver = NavierStokes2D(g, torch.full((batch,), 1e-3, dtype=dtype))
    for _ in range(2):
        w_hat = solver.step(w_hat, 1e-3)
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(steps):
        w_hat = solver.step(w_hat, 1e-3)
    _sync(device)
    dt = (time.perf_counter() - t0) / steps
    return {"device": device, "n": n, "batch": batch, "dtype": str(dtype).replace("torch.", ""),
            "ms_per_step": 1e3 * dt, "us_per_trajectory_step": 1e6 * dt / batch}


def time_shift_average(device: str, batch: int) -> dict:
    """Forward cost of the latent averaged over all 64 sub-stride (8x8) circular shifts."""
    net = provisional_encoder().to(device).eval()
    x = torch.randn(batch, 2, 64, 64, device=device)
    with torch.no_grad():
        def run():
            acc = 0
            for sy in range(8):
                for sx in range(8):
                    acc = acc + net(torch.roll(x, shifts=(sy, sx), dims=(2, 3))).mean(dim=(2, 3))
            return acc / 64
        run()
        _sync(device)
        t0 = time.perf_counter()
        run()
        _sync(device)
    return {"device": device, "ms_per_frame_64_shifts": 1e3 * (time.perf_counter() - t0) / batch}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = load_config(ap.parse_args().config)
    seed_everything(cfg["seed"])
    set_threads(cfg["cpu_threads"])
    torch.backends.cuda.matmul.allow_tf32 = bool(cfg["allow_tf32"])
    torch.backends.cudnn.allow_tf32 = bool(cfg["allow_tf32"])
    t_start = time.perf_counter()
    out = {"environment": environment(), "encoder": [], "solver": [], "shift_average": []}
    for dev in cfg["encoder_devices"]:
        for amp in ([False, True] if dev == "cuda" else [False]):
            r = time_encoder(dev, cfg["encoder_batch"], cfg["encoder_reps"] if dev == "cuda" else 3, amp)
            out["encoder"].append(r)
            print("encoder", r)
    for s in cfg["solver_cases"]:
        r = time_solver(s["device"], s["n"], s["batch"], getattr(torch, s["dtype"]), s["steps"])
        out["solver"].append(r)
        print("solver", r)
    r = time_shift_average("cuda", cfg["encoder_batch"])
    out["shift_average"].append(r)
    print("shift-average", r)
    wall = time.perf_counter() - t_start
    save_json(out, cfg["output_json"])
    gpu = next(e for e in out["encoder"] if e["device"] == "cuda" and not e["amp_bf16"])
    gpu_amp = next(e for e in out["encoder"] if e["device"] == "cuda" and e["amp_bf16"])
    cpu = next(e for e in out["encoder"] if e["device"] == "cpu")
    append_log("Phase 0 hardware benchmark", cfg["_config_path"], command_line(),
               "ok", {"encoder fwd+bwd ms/frame (GPU fp32 / GPU bf16 / CPU 8 threads)":
                      f"{gpu['fwd_bwd_ms_per_frame']:.4f} / {gpu_amp['fwd_bwd_ms_per_frame']:.4f} / {cpu['fwd_bwd_ms_per_frame']:.2f}",
                      "solver": "; ".join(f"{s['device']} {s['n']}^2 B={s['batch']} {s['dtype']}: {s['ms_per_step']:.1f} ms/step"
                                          for s in out["solver"]),
                      "results": cfg["output_json"]}, wall, out["environment"].get("gpu", "cpu"))


if __name__ == "__main__":
    main()
