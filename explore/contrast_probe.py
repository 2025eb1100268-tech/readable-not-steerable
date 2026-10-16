"""[EXPLORE] Contrast probe (PREDICTIONS entry 7; POST HOC, designed after the imagination probe).

    python -m explore.contrast_probe --config configs/explore/contrast_probe.yaml

Does the decoder read enstrophy from the spatial variation ("contrast") of the encoder's 8x8 map
rather than from its spatial mean? Per channel, with mu the spatial mean over the 8x8 positions:
  contrast  m -> mu + (1 + eps)(m - mu)
  mean      m -> (1 + eps) mu + (m - mu)
  global    m -> (1 + eps) m
applied to every context map (both frames for an imaginer, the single map for an autoencoder),
then decode and measure log Z, log E and log P of the rendered field. Effect = least-squares slope
over eps. Exploratory and post hoc; nothing here is a finding.
"""
from __future__ import annotations

import argparse
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from common.run import REPO, append_log, command_line, environment, load_config, save_json, seed_everything  # noqa: E402
from data.combined import CombinedDataset  # noqa: E402
from solver import diagnostics as dg  # noqa: E402
from solver.ns2d import SpectralGrid  # noqa: E402
from solver.validate import INK, SERIES, _style  # noqa: E402
from train.train import AE_KINDS, Model  # noqa: E402

QUANTITIES = ("log Z", "log E", "log P")
INTERVENTIONS = {
    "contrast": lambda m, e: m.mean((2, 3), keepdim=True) + (1 + e) * (m - m.mean((2, 3), keepdim=True)),
    "mean": lambda m, e: (1 + e) * m.mean((2, 3), keepdim=True) + (m - m.mean((2, 3), keepdim=True)),
    "global": lambda m, e: (1 + e) * m,
}


def rendered_vorticity_hat(model, out: torch.Tensor, g: SpectralGrid) -> torch.Tensor:
    """Vorticity spectrum of the rendered field (units drop out of log slopes)."""
    out = out.double()
    target = model.ae_target if model.kind in AE_KINDS else "velocity"
    if target == "velocity":
        return g.vorticity_hat_from_velocity(out[:, 0], out[:, 1])
    f_hat = g.fft(out[:, 0]) * g.mask
    return f_hat if target == "vorticity" else f_hat * g.k2  # streamfunction: omega = -lap psi


@torch.no_grad()
def stats(model, x_prev, x_t, transform, g) -> dict:
    zeros = torch.zeros(len(x_t), dtype=torch.long, device=x_t.device)
    if model.kind in AE_KINDS:
        out = model.head(transform(model.enc(x_t)), zeros)
    else:
        maps = transform(model.enc(torch.cat([x_prev, x_t])))
        b = len(x_t)
        out = model.head(torch.cat([maps[:b], maps[b:]], 1), zeros)
    w = rendered_vorticity_hat(model, out, g)
    return {"log Z": float(torch.log(dg.enstrophy(g, w)).mean()), "log E": float(torch.log(dg.energy(g, w)).mean()),
            "log P": float(torch.log(dg.palinstrophy(g, w)).mean())}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = load_config(ap.parse_args().config)
    seed_everything(cfg["seed"])
    t0 = time.perf_counter()
    ds = CombinedDataset(cfg["data_parts"], cfg["cache_tag"])
    g = SpectralGrid(ds.velocity.shape[-1], "cuda")
    eps = np.array(cfg["eps"], float)
    ev = ds.rows(["explore"], viscous=False)
    results = {}
    for run in cfg["runs"]:
        ckpt = torch.load(REPO / "results" / "checkpoints" / f"{run}.pt", map_location="cpu", weights_only=False)
        model = Model(ckpt["config"])
        model.load_state_dict(ckpt["model"])
        model = model.cuda().eval()
        k = ckpt["config"]["horizons"][0]
        ok = ev[(ds.frame[ev] >= 1) & (ds.frame[ev] <= 23 - k)]
        anchors = np.sort(np.random.default_rng(cfg["seed"]).choice(ok, cfg["n_anchors"], replace=False))
        to_gpu = lambda rows: torch.from_numpy(ds.velocity[rows]).cuda() / ds.input_scale  # noqa: E731
        x_prev, x_t = to_gpu(anchors - 1), to_gpu(anchors)
        curves = {name: {q: [] for q in QUANTITIES} for name in INTERVENTIONS}
        for name, f in INTERVENTIONS.items():
            for e in eps:
                s = stats(model, x_prev, x_t, lambda m, e=e, f=f: f(m, e), g)
                for q in QUANTITIES:
                    curves[name][q].append(s[q])
        slopes = {name: {q: float(np.polyfit(eps, c[q], 1)[0]) for q in QUANTITIES} for name, c in curves.items()}
        c, mn = slopes["contrast"], slopes["mean"]
        checks = {"contrast: |dlogZ| >= 1.5 |dlogE|": bool(abs(c["log Z"]) >= 1.5 * abs(c["log E"])),
                  "mean: |dlogE| > |dlogZ|": bool(abs(mn["log E"]) > abs(mn["log Z"]))}
        results[run] = {"k": k, "kind": model.kind, "target": getattr(model, "ae_target", "velocity (t+k)"),
                        "curves": curves, "slopes": slopes, "checks": checks}
        print(f"{run}: contrast {c}, mean {mn}, checks {checks}", flush=True)
    wall = time.perf_counter() - t0
    save_json({"environment": environment(), "config": cfg, "results": results, "wall_seconds": wall}, cfg["output_json"])

    L = ["# [EXPLORE] Contrast probe (POST HOC): does the decoder read enstrophy from the map's spatial variation?", "",
         "PREDICTIONS entry 7, designed after the imagination probe. Per channel, μ = spatial mean of the 8×8 map. "
         "contrast: m → μ + (1+ε)(m − μ); mean: m → (1+ε)μ + (m − μ); global: m → (1+ε)m; applied to every context map. "
         f"Slopes of the rendered field's log Z, log E, log P per unit ε ({cfg['n_anchors']} explore inviscid anchors).", "",
         "| run | renders | intervention | d log Z / dε | d log E / dε | d log P / dε | log Z / log E slope ratio |",
         "|---|---|---|---|---|---|---|"]
    for run, r in results.items():
        for name, s in r["slopes"].items():
            ratio = s["log Z"] / s["log E"] if abs(s["log E"]) > 1e-12 else float("nan")
            L.append(f"| {run} | {r['target']} | {name} | {s['log Z']:+.3f} | {s['log E']:+.3f} | {s['log P']:+.3f} | {ratio:.2f} |")
    L += ["", "Checks (P7.1 on the imaginers' contrast row, P7.2 on their mean row, P7.3 = both for the vorticity autoencoder):", ""]
    for run, r in results.items():
        L.append(f"- {run}: " + "; ".join(f"{k_}: {'yes' if v else 'no'}" for k_, v in r["checks"].items()))
    with open(REPO / cfg["output_md"], "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")

    _style()
    fig, axes = plt.subplots(1, len(results), figsize=(3.6 * len(results), 3.8), constrained_layout=True, squeeze=False)
    for ax, (run, r) in zip(axes[0], results.items()):
        for q, col in zip(QUANTITIES, SERIES):
            for name, ls in (("contrast", "-"), ("mean", (0, (4, 2))), ("global", (0, (1, 2)))):
                y = np.array(r["curves"][name][q])
                ax.plot(eps, y - y[len(eps) // 2], color=col, linestyle=ls, linewidth=1.8 if name == "contrast" else 1.2,
                        label=f"{q}, {name}" if run == next(iter(results)) else None)
        ax.set_title(run.replace("s2_", "").replace("_lr0.001", ""), fontsize=8.5)
        ax.set_xlabel("ε")
    axes[0][0].set_ylabel("change in the rendered field (log units)")
    axes[0][0].legend(fontsize=6.5)
    fig.suptitle("[EXPLORE] Contrast probe (post hoc): solid = contrast, dashed = mean, dotted = global", fontweight="bold", color=INK)
    fig.savefig(REPO / cfg["figure"], dpi=120)
    plt.close(fig)
    append_log("[EXPLORE] Contrast probe (entry 7, post hoc)", cfg["_config_path"], command_line(), "ok",
               {"checks per run": "; ".join(f"{run}: {all(r['checks'].values())}" for run, r in results.items()),
                "results": f"{cfg['output_md']}, {cfg['figure']}, {cfg['output_json']}"}, wall, environment().get("gpu"))
    print("\n".join(L))


if __name__ == "__main__":
    main()
