"""Phase 2 training (PREREGISTRATION.md §4). One YAML per run:

    python -m train.train --config configs/phase2/<run>.yaml

model: jepa_spatial (main) | pixel | jepa_pooled (ablation) | static_vicreg (control)
Writes results/phase2/<run>.json (curves, label-free val metrics, collapse diagnostics) and
results/checkpoints/<run>.pt (gitignored).
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # deterministic cuBLAS; set before CUDA init

import argparse  # noqa: E402
import copy  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from common.run import REPO, append_log, command_line, environment, load_config, save_json, seed_everything  # noqa: E402
from data.dataset import Dataset  # noqa: E402
from models.encoder import build_encoder  # noqa: E402
from models.heads import HORIZONS, PixelDecoder, PooledPredictor, SpatialPredictor, vicreg_var_cov  # noqa: E402

MAX_K = max(HORIZONS)


class SplitData:
    """All frames of one split on the GPU (scaled), plus valid anchor rows t with t-1 and t+4 in
    the same trajectory."""

    def __init__(self, ds: Dataset, split: str, device: str):
        rows = ds.rows([split])
        self.x = torch.from_numpy(np.asarray(ds.velocity[rows])).to(device) / ds.input_scale
        frame, traj = ds.frame[rows], ds.traj[rows]
        n_frames = np.array([t["n_frames"] for t in ds.trajectories])[traj]
        ok = (frame >= 1) & (frame + MAX_K <= n_frames - 1)
        self.anchors = torch.from_numpy(np.flatnonzero(ok)).to(device)
        self.n = len(rows)


def d4_view(x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Random circular shift + random element of the square's symmetry group, applied to a velocity
    field (B, 2, n, n) as a vector field: u'(x) = R u(R^-1 x). Same transform for the whole batch."""
    n = x.shape[-1]
    sy, sx = (int(v) for v in torch.randint(0, n, (2,), generator=gen))
    x = torch.roll(x, shifts=(sy, sx), dims=(-2, -1))
    rot = int(torch.randint(0, 4, (1,), generator=gen))
    for _ in range(rot):  # 90° CCW: (x, y) -> (-y, x);  u' = -v(y, -x), v' = u(y, -x)
        a = x.transpose(-2, -1)
        a = torch.roll(torch.flip(a, dims=[-1]), shifts=1, dims=-1)
        x = torch.stack([-a[:, 1], a[:, 0]], dim=1)
    if int(torch.randint(0, 2, (1,), generator=gen)):  # reflection y -> -y: u' = u(x, -y), v' = -v(x, -y)
        a = torch.roll(torch.flip(x, dims=[-2]), shifts=1, dims=-2)
        x = torch.stack([a[:, 0], -a[:, 1]], dim=1)
    return x


class Model(torch.nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.kind = cfg["model"]
        self.enc = build_encoder(cfg["seed"])
        torch.manual_seed(cfg["seed"] + 1000)
        d = self.enc.d
        if self.kind == "jepa_spatial":
            self.head = SpatialPredictor(d, cfg["predictor_width"])
        elif self.kind == "jepa_pooled":
            self.head = PooledPredictor(d, cfg["predictor_width"])
        elif self.kind == "pixel":
            self.head = PixelDecoder(d)
        elif self.kind == "static_vicreg":
            self.head = None
        else:
            raise ValueError(self.kind)
        self.target = copy.deepcopy(self.enc).requires_grad_(False) if self.kind.startswith("jepa") else None

    def probed_encoder(self):
        """The encoder whose pooled latent is probed: EMA target for JEPA variants."""
        return self.target if self.target is not None else self.enc

    def loss(self, x_prev, x_t, x_tk, k_idx, w, gen=None):
        b = x_t.shape[0]
        if self.kind == "static_vicreg":
            v1, v2 = d4_view(x_t, gen), d4_view(x_t, gen)
            z = self.enc.pooled(torch.cat([v1, v2]))
            z1, z2 = z[:b], z[b:]
            inv = F.mse_loss(z1.float(), z2.float())
            var1, cov1 = vicreg_var_cov(z1)
            var2, cov2 = vicreg_var_cov(z2)
            var, cov = (var1 + var2) / 2, (cov1 + cov2) / 2
            return w["pred"] * inv + w["var"] * var + w["cov"] * cov, {"pred": inv, "var": var, "cov": cov}
        maps = self.enc(torch.cat([x_prev, x_t]))
        m_prev, m_t = maps[:b], maps[b:]
        if self.kind == "pixel":
            pred = self.head(torch.cat([m_prev, m_t], 1), k_idx)
            mse = F.mse_loss(pred.float(), x_tk.float())
            return mse, {"pred": mse}
        z_t = m_t.mean(dim=(2, 3))
        with torch.no_grad():
            tgt_map = self.target(x_tk)
        if self.kind == "jepa_spatial":
            pred = self.head(torch.cat([m_prev, m_t], 1), k_idx)
            lp = F.mse_loss(pred.float(), tgt_map.float())
        else:
            pred = self.head(m_prev.mean(dim=(2, 3)), z_t, k_idx)
            lp = F.mse_loss(pred.float(), tgt_map.mean(dim=(2, 3)).float())
        var, cov = vicreg_var_cov(z_t)
        return w["pred"] * lp + w["var"] * var + w["cov"] * cov, {"pred": lp, "var": var, "cov": cov}

    @torch.no_grad()
    def ema_update(self, m: float):
        if self.target is not None:
            t_params, o_params = list(self.target.parameters()), list(self.enc.parameters())
            torch._foreach_mul_(t_params, m)
            torch._foreach_add_(t_params, o_params, alpha=1.0 - m)


@torch.no_grad()
def evaluate(model: Model, val: SplitData, k_val: torch.Tensor, batch: int, amp: bool) -> dict:
    """Label-free validation metric (prediction loss / copy baseline) and collapse diagnostics."""
    model.eval()
    num = den = 0.0
    gen = torch.Generator(device="cpu").manual_seed(12345)
    for i in range(0, len(val.anchors), batch):
        a = val.anchors[i:i + batch]
        ki = k_val[i:i + batch]
        k = torch.tensor(HORIZONS, device=a.device)[ki]
        x_prev, x_t, x_tk = val.x[a - 1], val.x[a], val.x[a + k]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            if model.kind == "pixel":
                maps = model.enc(torch.cat([x_prev, x_t]))
                pred = model.head(torch.cat([maps[:len(a)], maps[len(a):]], 1), ki).float()
                num += F.mse_loss(pred, x_tk, reduction="sum").item()
                den += F.mse_loss(x_t, x_tk, reduction="sum").item()
            elif model.kind == "static_vicreg":
                z1 = model.enc.pooled(d4_view(x_t, gen)).float()
                z2 = model.enc.pooled(d4_view(x_t, gen)).float()
                num += F.mse_loss(z1, z2, reduction="sum").item()
                den += z1.numel()
            else:
                maps = model.enc(torch.cat([x_prev, x_t]))
                tgt_t, tgt_tk = model.target(x_t).float(), model.target(x_tk).float()
                if model.kind == "jepa_spatial":
                    pred = model.head(torch.cat([maps[:len(a)], maps[len(a):]], 1), ki).float()
                    num += F.mse_loss(pred, tgt_tk, reduction="sum").item()
                    den += F.mse_loss(tgt_t, tgt_tk, reduction="sum").item()
                else:
                    pred = model.head(maps[:len(a)].mean(dim=(2, 3)), maps[len(a):].mean(dim=(2, 3)), ki).float()
                    num += F.mse_loss(pred, tgt_tk.mean(dim=(2, 3)), reduction="sum").item()
                    den += F.mse_loss(tgt_t.mean(dim=(2, 3)), tgt_tk.mean(dim=(2, 3)), reduction="sum").item()
    metric = num / den
    enc = model.probed_encoder()
    sub = torch.arange(0, val.n, 4, device=val.x.device)
    z = torch.cat([enc.pooled(val.x[sub[i:i + batch]]).float() for i in range(0, len(sub), batch)])
    std = z.std(0)

    def eff_rank(mat):
        ev = torch.linalg.eigvalsh(mat).clamp(min=0)
        p = ev / ev.sum()
        return float(torch.exp(-(p[p > 0] * p[p > 0].log()).sum()))

    model.train()
    return {"metric": metric, "metric_name": "loss / copy baseline" if model.kind != "static_vicreg" else "invariance loss",
            "latent_std_median": float(std.median()), "latent_std_min": float(std.min()),
            "effective_rank": eff_rank(torch.cov(z.T)), "effective_rank_corr": eff_rank(torch.corrcoef(z.T))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = load_config(ap.parse_args().config)
    seed_everything(cfg["seed"])
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    device = "cuda"
    t0 = time.perf_counter()
    ds = Dataset(cfg["data_dir"])
    train, val = SplitData(ds, "train", device), SplitData(ds, "val", device)
    model = Model(cfg).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    steps_per_epoch = math.ceil(len(train.anchors) / cfg["batch"])
    total = steps_per_epoch * cfg["epochs"]
    warm = max(1, int(cfg["warmup_frac"] * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    gen = torch.Generator(device=device).manual_seed(cfg["seed"])
    gen_cpu = torch.Generator(device="cpu").manual_seed(cfg["seed"])
    k_val = torch.from_numpy(np.random.default_rng(0).integers(0, len(HORIZONS), len(val.anchors))).to(device)
    horizons = torch.tensor(HORIZONS, device=device)
    w = cfg["loss_weights"]
    curve, evals, step = [], [], 0
    acc = {}
    for epoch in range(cfg["epochs"]):
        perm = train.anchors[torch.randperm(len(train.anchors), generator=gen, device=device)]
        for i in range(0, len(perm), cfg["batch"]):
            a = perm[i:i + cfg["batch"]]
            k_idx = torch.randint(0, len(HORIZONS), (len(a),), generator=gen, device=device)
            k = horizons[k_idx]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg["amp_bf16"]):
                loss, parts = model.loss(train.x[a - 1], train.x[a], train.x[a + k], k_idx, w, gen_cpu)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            m = 1.0 - (1.0 - cfg["ema_start"]) * 0.5 * (1 + math.cos(math.pi * step / total))
            model.ema_update(m)
            step += 1
            for key, v in {"loss": loss, **parts}.items():  # stays on the GPU: no per-step sync
                acc[key] = acc.get(key, 0.0) + v.detach()
            if step % cfg["log_every"] == 0:
                curve.append({"step": step, **{key: float(v) / cfg["log_every"] for key, v in acc.items()}})
                acc = {}
                if not math.isfinite(curve[-1]["loss"]):
                    raise FloatingPointError(f"non-finite loss at step {step}")
        if (epoch + 1) % cfg["eval_every_epochs"] == 0 or epoch + 1 == cfg["epochs"]:
            ev = evaluate(model, val, k_val, cfg["batch"], cfg["amp_bf16"])
            ev.update({"epoch": epoch + 1, "step": step, "elapsed_s": time.perf_counter() - t0})
            evals.append(ev)
            print(f"[{cfg['run_name']}] epoch {epoch + 1}/{cfg['epochs']}: train loss {curve[-1]['loss']:.4f}, "
                  f"val {ev['metric_name']} {ev['metric']:.4f}, latent std med {ev['latent_std_median']:.3f}, "
                  f"eff. rank {ev['effective_rank']:.1f} (corr {ev['effective_rank_corr']:.1f}), {ev['elapsed_s']:.0f} s", flush=True)
    wall = time.perf_counter() - t0
    final = evals[-1]
    # Amendment A1: the no-collapse rule applies only to objectives that can collapse (variance target)
    collapsed = cfg["model"] != "pixel" and (final["effective_rank"] < cfg["collapse"]["min_effective_rank"]
                                             or final["latent_std_median"] < cfg["collapse"]["min_std_median"])
    ckpt = REPO / "results" / "checkpoints" / f"{cfg['run_name']}.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": cfg, "model": model.state_dict()}, ckpt)
    out = {"config": cfg, "environment": environment(), "curve": curve, "evals": evals, "final": final,
           "collapsed": collapsed, "steps": step, "wall_seconds": wall,
           "peak_gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9, "checkpoint": str(ckpt.relative_to(REPO).as_posix())}
    save_json(out, REPO / "results" / "phase2" / f"{cfg['run_name']}.json")
    append_log(f"Phase 2 training: {cfg['run_name']}", cfg["_config_path"], command_line(),
               "collapsed (fails the pre-registered no-collapse rule)" if collapsed else "ok",
               {"final val " + final["metric_name"]: f"{final['metric']:.4f}",
                "effective rank / latent std median": f"{final['effective_rank']:.1f} / {final['latent_std_median']:.3f}",
                "steps": step, "peak GPU memory GB": f"{out['peak_gpu_mem_gb']:.2f}",
                "results": f"results/phase2/{cfg['run_name']}.json"}, wall, environment().get("gpu"))


if __name__ == "__main__":
    main()
