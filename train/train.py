"""Phase 2 training (PREREGISTRATION.md §4). One YAML per run:

    python -m train.train --config configs/phase2/<run>.yaml

model: jepa_spatial (main) | pixel | jepa_pooled (ablation) | static_vicreg (control)
       | pixel_vicreg ([EXPLORE] pixel loss + the JEPA's VICReg terms on the pooled latent)
Writes results/phase2/<run>.json (curves, label-free val metrics, collapse diagnostics) and
results/checkpoints/<run>.pt (gitignored).

[EXPLORE] Stage 2 options (absent = the Phase 2 behaviour; results/explore/PREDICTIONS.md entries
3-4): horizons (one k), anchor_rule long_windows (64-frame runs, t in [1, anchor_max_t]),
schedule plateau (warmup, constant lr, lr x factor on val plateaus, stop rule, step cap; JEPA EMA
momentum tied to the plateau lr), diagnostics (prediction spectra / map variance at the end),
results_dir and log_title.
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
from solver.diagnostics import energy_spectrum  # noqa: E402
from solver.ns2d import SpectralGrid  # noqa: E402

PIXEL_KINDS = ("pixel", "pixel_vicreg")


class SplitData:
    """All frames of one split on the GPU (scaled), plus the valid anchor rows t.

    rule "original": t - 1 and t + max(horizons) in the same trajectory (Phase 2).
    rule "long_windows" ([EXPLORE] Stage 2 common anchor set): trajectories with at least
    min_frames frames, t in [1, max_t], so every horizon sees identical inputs."""

    def __init__(self, ds: Dataset, split: str, device: str, horizons=HORIZONS, rule: str = "original",
                 max_t: int | None = None, min_frames: int = 64):
        rows = ds.rows([split])
        self.x = torch.from_numpy(np.asarray(ds.velocity[rows])).to(device) / ds.input_scale
        frame, traj = ds.frame[rows], ds.traj[rows]
        n_frames = np.array([t["n_frames"] for t in ds.trajectories])[traj]
        if rule == "original":
            ok = (frame >= 1) & (frame + max(horizons) <= n_frames - 1)
        elif rule == "long_windows":
            ok = (n_frames >= min_frames) & (frame >= 1) & (frame <= max_t)
            assert np.all(frame[ok] + max(horizons) <= n_frames[ok] - 1), "anchor_max_t + k exceeds the window"
        else:
            raise ValueError(rule)
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
        self.horizons = tuple(cfg.get("horizons", HORIZONS))
        nh = len(self.horizons)
        self.enc = build_encoder(cfg["seed"])
        torch.manual_seed(cfg["seed"] + 1000)
        d = self.enc.d
        if self.kind == "jepa_spatial":
            self.head = SpatialPredictor(d, cfg["predictor_width"], nh)
        elif self.kind == "jepa_pooled":
            self.head = PooledPredictor(d, cfg["predictor_width"], n_horizons=nh)
        elif self.kind in PIXEL_KINDS:
            self.head = PixelDecoder(d, n_horizons=nh)
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
        if self.kind in PIXEL_KINDS:
            pred = self.head(torch.cat([m_prev, m_t], 1), k_idx)
            mse = F.mse_loss(pred.float(), x_tk.float())
            if self.kind == "pixel":
                return mse, {"pred": mse}
            var, cov = vicreg_var_cov(m_t.mean(dim=(2, 3)))  # [EXPLORE] the JEPA's regularizer and weights
            return w["pred"] * mse + w["var"] * var + w["cov"] * cov, {"pred": mse, "var": var, "cov": cov}
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
        k = torch.tensor(model.horizons, device=a.device)[ki]
        x_prev, x_t, x_tk = val.x[a - 1], val.x[a], val.x[a + k]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            if model.kind in PIXEL_KINDS:
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


class Plateau:
    """[EXPLORE] Stage 2 schedule controller (results/explore/PREDICTIONS.md entries 3-4).

    Evaluations before start_step are ignored. Afterwards an evaluation improves if the metric is
    below best x (1 - min_rel_improve); after `patience` evaluations without improvement the lr is
    multiplied by `factor` (at most `max_reductions` times), and the next such run of evaluations
    stops training."""

    def __init__(self, min_rel_improve: float, patience: int, factor: float, max_reductions: int, start_step: int):
        self.min_rel, self.patience, self.mult = min_rel_improve, patience, factor
        self.max_reductions, self.start_step = max_reductions, start_step
        self.best, self.bad, self.reductions, self.factor = math.inf, 0, 0, 1.0

    def update(self, metric: float, step: int) -> str:
        if step < self.start_step:
            return "burn-in"
        if metric < self.best * (1 - self.min_rel):
            self.best, self.bad = metric, 0
            return "improved"
        self.bad += 1
        if self.bad < self.patience:
            return "no improvement"
        if self.reductions < self.max_reductions:
            self.reductions += 1
            self.factor *= self.mult
            self.bad = 0
            return "reduce lr"
        return "stop"


@torch.no_grad()
def prediction_diagnostics(model: Model, val: SplitData, k_val: torch.Tensor, amp: bool, n: int = 1024, chunk: int = 256) -> dict:
    """[EXPLORE] Stage 2: how much the prediction pictures vs averages, on n fixed val anchors.
    Pixel models: mean energy spectra of predicted, true and copied (frame t) fields at t + k.
    Spatial JEPA: variance of the predicted map relative to the target map (per channel, averaged)."""
    pixel = model.kind in PIXEL_KINDS
    if not (pixel or model.kind == "jepa_spatial"):
        return {}
    model.eval()
    hz = torch.tensor(model.horizons, device=val.x.device)
    g = SpectralGrid(val.x.shape[-1], val.x.device)
    acc: dict = {}
    count = 0
    for i in range(0, min(n, len(val.anchors)), chunk):
        a, ki = val.anchors[i:min(i + chunk, n)], k_val[i:min(i + chunk, n)]
        x_prev, x_t, x_tk = val.x[a - 1], val.x[a], val.x[a + hz[ki]]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            maps = model.enc(torch.cat([x_prev, x_t]))
            pred = model.head(torch.cat([maps[:len(a)], maps[len(a):]], 1), ki).float()
            tgt = None if pixel else model.target(x_tk).float()
        count += len(a)
        if pixel:
            for name, f in (("pred", pred), ("true", x_tk), ("copy", x_t)):
                _, e = energy_spectrum(g, g.vorticity_hat_from_velocity(f[:, 0].double(), f[:, 1].double()))
                acc[name] = acc.get(name, 0) + e.sum(0)
        else:
            for name, f in (("pred", pred), ("tgt", tgt)):
                acc[name + "_s1"] = acc.get(name + "_s1", 0) + f.double().sum((0, 2, 3))
                acc[name + "_s2"] = acc.get(name + "_s2", 0) + (f.double() ** 2).sum((0, 2, 3))
    model.train()
    if pixel:
        spec = {name: (acc[name] / count).cpu().numpy() for name in ("pred", "true", "copy")}
        out = {"n_anchors": count, "k": list(range(len(spec["true"]))), **{f"E_{name}": v.tolist() for name, v in spec.items()}}
        for k0 in (5, 10):
            out[f"pred_over_true_energy_k_ge_{k0}"] = float(spec["pred"][k0:].sum() / spec["true"][k0:].sum())
            out[f"copy_over_true_energy_k_ge_{k0}"] = float(spec["copy"][k0:].sum() / spec["true"][k0:].sum())
        return out
    m = count * 64
    var = {name: acc[name + "_s2"] / m - (acc[name + "_s1"] / m) ** 2 for name in ("pred", "tgt")}
    return {"n_anchors": count, "pred_over_target_map_variance": float((var["pred"] / var["tgt"]).mean())}


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
    hz_list = tuple(cfg.get("horizons", HORIZONS))
    anchor = {"horizons": hz_list, "rule": cfg.get("anchor_rule", "original"), "max_t": cfg.get("anchor_max_t"),
              "min_frames": cfg.get("anchor_min_frames", 64)}
    train, val = SplitData(ds, "train", device, **anchor), SplitData(ds, "val", device, **anchor)
    model = Model(cfg).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    plateau = cfg.get("schedule", "cosine") == "plateau"
    if plateau:  # [EXPLORE] Stage 2
        warm, total = cfg["warmup_steps"], cfg["max_steps"]
        ctl = Plateau(**cfg["plateau"])
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * ctl.factor)
    else:
        steps_per_epoch = math.ceil(len(train.anchors) / cfg["batch"])
        total = steps_per_epoch * cfg["epochs"]
        warm = max(1, int(cfg["warmup_frac"] * total))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    gen = torch.Generator(device=device).manual_seed(cfg["seed"])
    gen_cpu = torch.Generator(device="cpu").manual_seed(cfg["seed"])
    k_val = torch.from_numpy(np.random.default_rng(0).integers(0, len(hz_list), len(val.anchors))).to(device)
    horizons = torch.tensor(hz_list, device=device)
    w = cfg["loss_weights"]
    curve, evals, step = [], [], 0
    acc = {}
    stop_reason, epoch = None, 0

    def do_eval():
        ev = evaluate(model, val, k_val, cfg["batch"], cfg["amp_bf16"])
        ev.update({"epoch": epoch + (0 if plateau else 1), "step": step, "elapsed_s": time.perf_counter() - t0})
        if plateau:
            ev["lr_factor"] = ctl.factor
        evals.append(ev)
        return ev

    def report(ev, head):
        loss_txt = f"{curve[-1]['loss']:.4f}" if curve else "n/a"
        print(f"[{cfg['run_name']}] {head}: train loss {loss_txt}, "
              f"val {ev['metric_name']} {ev['metric']:.4f}, latent std med {ev['latent_std_median']:.3f}, "
              f"eff. rank {ev['effective_rank']:.1f} (corr {ev['effective_rank_corr']:.1f}), {ev['elapsed_s']:.0f} s", flush=True)

    while stop_reason is None:
        perm = train.anchors[torch.randperm(len(train.anchors), generator=gen, device=device)]
        for i in range(0, len(perm), cfg["batch"]):
            a = perm[i:i + cfg["batch"]]
            k_idx = torch.randint(0, len(hz_list), (len(a),), generator=gen, device=device)
            k = horizons[k_idx]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg["amp_bf16"]):
                loss, parts = model.loss(train.x[a - 1], train.x[a], train.x[a + k], k_idx, w, gen_cpu)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            if plateau:
                m = cfg["ema_start"] if step < warm else 1.0 - (1.0 - cfg["ema_start"]) * ctl.factor
            else:
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
            if plateau and step % cfg["eval_every_steps"] == 0:
                ev = do_eval()
                ev["plateau_action"] = ctl.update(ev["metric"], step)
                report(ev, f"step {step} ({ev['plateau_action']}, lr x{ctl.factor:g})")
                if ev["plateau_action"] == "stop":
                    stop_reason = "plateau"
                    break
            if plateau and step >= cfg["max_steps"]:
                stop_reason = "step cap"
                break
        if not plateau:
            if (epoch + 1) % cfg["eval_every_epochs"] == 0 or epoch + 1 == cfg["epochs"]:
                report(do_eval(), f"epoch {epoch + 1}/{cfg['epochs']}")
            if epoch + 1 == cfg["epochs"]:
                stop_reason = "epochs"
        epoch += 1
    if evals[-1]["step"] != step:
        report(do_eval(), f"final, step {step}")
    wall = time.perf_counter() - t0
    final = evals[-1]
    # Amendment A1: the no-collapse rule applies only to objectives that can collapse (variance target)
    collapsed = cfg["model"] != "pixel" and (final["effective_rank"] < cfg["collapse"]["min_effective_rank"]
                                             or final["latent_std_median"] < cfg["collapse"]["min_std_median"])
    diagnostics = prediction_diagnostics(model, val, k_val, cfg["amp_bf16"]) if cfg.get("diagnostics") else None
    ckpt = REPO / "results" / "checkpoints" / f"{cfg['run_name']}.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": cfg, "model": model.state_dict()}, ckpt)
    results_dir = cfg.get("results_dir", "results/phase2")
    out = {"config": cfg, "environment": environment(), "curve": curve, "evals": evals, "final": final,
           "collapsed": collapsed, "steps": step, "stop_reason": stop_reason, "wall_seconds": wall,
           "peak_gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9, "checkpoint": str(ckpt.relative_to(REPO).as_posix())}
    if diagnostics is not None:
        out["diagnostics"] = diagnostics
    save_json(out, REPO / results_dir / f"{cfg['run_name']}.json")
    details = {"final val " + final["metric_name"]: f"{final['metric']:.4f}",
               "effective rank / latent std median": f"{final['effective_rank']:.1f} / {final['latent_std_median']:.3f}",
               "steps": step, "peak GPU memory GB": f"{out['peak_gpu_mem_gb']:.2f}",
               "results": f"{results_dir}/{cfg['run_name']}.json"}
    if plateau:
        details["stop reason / lr reductions"] = f"{stop_reason} / {ctl.reductions}"
    if diagnostics:
        details["diagnostics"] = ", ".join(f"{key}: {v:.3f}" for key, v in diagnostics.items() if key.startswith(("pred_over", "copy_over")))
    append_log(f"{cfg.get('log_title', 'Phase 2 training')}: {cfg['run_name']}", cfg["_config_path"], command_line(),
               "collapsed (fails the pre-registered no-collapse rule)" if collapsed else "ok", details, wall, environment().get("gpu"))

if __name__ == "__main__":
    main()
