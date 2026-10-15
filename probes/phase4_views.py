"""Phase 4 views (PREREGISTRATION.md §6, reported, unscored).

    python -m probes.phase4_views --config configs/phase4_views.yaml

- PCA and t-SNE views of val latents, coloured by enstrophy, vortex count and ν.
- Latent traversals along each model's top-3 unsupervised slow directions (§6 protocol: z-score,
  PCA, Ledoit–Wolf generalized eigenproblem on inviscid trajectories). Each step is a value of the
  slow coordinate (mean + s·std over val inviscid frames), read out through the k val inviscid
  frames whose coordinate is nearest it: their mean energy spectrum, vorticity PDF and the single
  nearest vorticity field. (Nearest neighbours in the full PCA space were tried first and
  rejected: the other directions dominate the distance, so several steps returned the same
  trajectory.)

Everything is fitted on train and shown on val; the test sets are not touched here (they are
used once, by probes/final_evaluation.py). One seed per model. The sign of each slow direction
is flipped, for display only, so that it correlates positively with log enstrophy.
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
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402

from common.run import REPO, append_log, command_line, environment, load_config, save_json, seed_everything  # noqa: E402
from data.dataset import Dataset  # noqa: E402
from probes import slow_features as sfeat  # noqa: E402
from probes.latents import latents_for  # noqa: E402
from solver.diagnostics import energy_spectrum  # noqa: E402
from solver.ns2d import SpectralGrid  # noqa: E402
from solver.validate import INK2, _style  # noqa: E402

# single-hue sequential ramp (light -> dark) on the repo's first series colour
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#cfe0f6", "#2a78d6", "#0b2a55"])
NEUTRAL = "#c9c8c3"


def embeddings(lat: np.ndarray, rows_fit: np.ndarray, rows_show: np.ndarray, n_pca: int, seed: int) -> dict:
    """z-score + PCA fitted on rows_fit; PCA (first two) and t-SNE (of the n_pca coords) of rows_show."""
    mu, sd = lat[rows_fit].mean(0), lat[rows_fit].std(0) + 1e-12
    z = (lat[rows_fit] - mu) / sd
    _, s, vt = np.linalg.svd(z - z.mean(0), full_matrices=False)
    x = ((lat[rows_show] - mu) / sd - z.mean(0)) @ vt[:n_pca].T
    tsne = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto", random_state=seed).fit_transform(x)
    return {"pca": x[:, :2], "tsne": tsne, "explained": (s[:2] ** 2 / (s**2).sum()).tolist()}


def vorticity(ds: Dataset, grid: SpectralGrid, rows: np.ndarray) -> torch.Tensor:
    vel = torch.from_numpy(np.ascontiguousarray(ds.velocity[np.sort(rows)])).to(torch.float64)
    return grid.vorticity_hat_from_velocity(vel[:, 0], vel[:, 1])


def traversals(ds: Dataset, lat: np.ndarray, grid: SpectralGrid, cfg: dict) -> list[dict]:
    """One traversal per top slow direction (slowest first)."""
    fit_rows, ev_rows = ds.rows(["train"], viscous=False), ds.rows(["val"], viscous=False)
    model = sfeat.fit(lat[fit_rows], ds.traj[fit_rows], cfg["n_pca"])
    proj = sfeat.project(model, lat[ev_rows])
    ids, inv = np.unique(ds.traj[ev_rows], return_inverse=True)
    first = np.array([np.flatnonzero((ds.traj == t) & (ds.frame == 0))[0] for t in ids])
    out = []
    for d in range(cfg["n_directions"]):
        p = proj[:, d]
        p = p if np.corrcoef(p, np.log(ds.f["Z"][ev_rows]))[0, 1] >= 0 else -p  # display orientation only
        p_traj = np.bincount(inv, weights=p) / np.bincount(inv)
        r = {q: float(abs(np.corrcoef(p_traj, np.log(ds.f[q][first]))[0, 1])) for q in ("E", "Z")}
        steps = []
        for s in cfg["traversal_steps"]:
            target = p.mean() + s * p.std()
            nn = np.argsort(np.abs(p - target))[: cfg["n_neighbours"]]
            rows = ev_rows[nn]
            w_hat = vorticity(ds, grid, rows)
            k, ek = energy_spectrum(grid, w_hat)
            w = grid.ifft(w_hat)
            steps.append({"s": s, "target_p": float(target), "nn_p_mean": float(p[nn].mean()),
                          "n_traj": int(len(np.unique(ds.traj[rows]))),
                          "nn_log_E": float(np.log(ds.f["E"][rows]).mean()), "nn_log_Z": float(np.log(ds.f["Z"][rows]).mean()),
                          "spectrum": ek.mean(0).numpy(), "k": k.numpy(), "w_values": w.numpy().ravel(),
                          "nearest_field": w[np.argsort(np.argsort(rows))[0]].numpy()})
        out.append({"direction": d + 1, "lambda_fit": float(model["lambda_fit"][d]),
                    "traj_abs_r_logE0": r["E"], "traj_abs_r_logZ0": r["Z"], "steps": steps})
    return out


def plot_embeddings(emb: dict, colour: dict, kind: str, out: str):
    names = list(emb)
    cols = list(colour)
    fig, ax = plt.subplots(len(names), len(cols), figsize=(3.3 * len(cols), 2.9 * len(names)), squeeze=False,
                           constrained_layout=True)
    for j, cname in enumerate(cols):
        c = colour[cname]
        lo, hi = np.nanpercentile(c, [1, 99])
        for i, name in enumerate(names):
            a, xy = ax[i, j], emb[name][kind]
            nan = np.isnan(c)
            if nan.any():
                a.scatter(xy[nan, 0], xy[nan, 1], s=2, color=NEUTRAL, linewidths=0, rasterized=True, label="inviscid (ν = 0)")
            sc = a.scatter(xy[~nan, 0], xy[~nan, 1], c=c[~nan], s=2, cmap=SEQ, vmin=lo, vmax=hi, linewidths=0, rasterized=True)
            a.set_xticks([])
            a.set_yticks([])
            a.grid(False)
            if kind == "pca":
                ev = emb[name]["explained"]
                a.set_xlabel(f"PC1 ({100 * ev[0]:.0f}%)", fontsize=7.5, color=INK2)
                a.set_ylabel(f"PC2 ({100 * ev[1]:.0f}%)", fontsize=7.5, color=INK2)
            if j == 0:
                a.set_ylabel(name + ("\n" + a.get_ylabel() if kind == "pca" else ""), fontsize=9)
            if i == 0:
                a.set_title(cname)
            if i == 0 and nan.any():
                a.legend(loc="upper right", fontsize=7, markerscale=4, handletextpad=0.2)
        fig.colorbar(sc, ax=ax[:, j], location="bottom", shrink=0.8, aspect=30, label=cname)
    fig.suptitle(f"Val latents, {'PCA (fitted on train)' if kind == 'pca' else 't-SNE of the 32 train-fitted PCA coordinates'}",
                 fontweight="bold")
    fig.savefig(REPO / out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_traversals(name: str, dirs: list[dict], out: str):
    n_steps = len(dirs[0]["steps"])
    fig = plt.figure(figsize=(18, 3.5 * len(dirs)), constrained_layout=True)
    gs = fig.add_gridspec(len(dirs), 2 + n_steps, width_ratios=[2.3, 2.3] + [1] * n_steps)
    colours = [SEQ(v) for v in np.linspace(0.3, 1.0, n_steps)]
    fig.suptitle(f"{name}: traversals along the top-3 slow directions (val inviscid frames nearest each step)",
                 fontweight="bold")
    for i, t in enumerate(dirs):
        a_spec, a_pdf = fig.add_subplot(gs[i, 0]), fig.add_subplot(gs[i, 1])
        lim = max(np.abs(st["w_values"]).max() for st in t["steps"])
        bins = np.linspace(-lim, lim, 61)
        for st, col in zip(t["steps"], colours):
            lab = f"s = {st['s']:+g}σ (log Z {st['nn_log_Z']:.2f})"
            a_spec.loglog(st["k"][1:], st["spectrum"][1:], color=col, label=lab)
            h, edges = np.histogram(st["w_values"], bins=bins, density=True)
            a_pdf.semilogy(0.5 * (edges[1:] + edges[:-1]), np.where(h > 0, h, np.nan), color=col, label=lab)
        a_spec.set_title(f"direction {t['direction']} (λ {t['lambda_fit']:.3f}): energy spectrum")
        a_spec.set_xlabel("k")
        a_spec.set_ylabel("E(k)")
        a_spec.legend(fontsize=7)
        a_pdf.set_title(f"vorticity PDF (traj. |r| with log E₀ {t['traj_abs_r_logE0']:.2f}, log Z₀ {t['traj_abs_r_logZ0']:.2f})")
        a_pdf.set_xlabel("ω")
        a_pdf.set_ylabel("PDF")
        vmax = max(np.abs(st["nearest_field"]).max() for st in t["steps"])
        for j, st in enumerate(t["steps"]):
            a = fig.add_subplot(gs[i, 2 + j])
            a.imshow(st["nearest_field"], cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="lower")
            a.set_xticks([])
            a.set_yticks([])
            a.grid(False)
            a.set_title(f"s = {st['s']:+g}σ", fontsize=8.5, fontweight="normal")
    fig.savefig(REPO / out, dpi=120)
    plt.close(fig)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = load_config(ap.parse_args().config)
    seed_everything(cfg["seed"])
    t0 = time.perf_counter()
    _style()
    ds = Dataset(cfg["data_dir"])
    grid = SpectralGrid(ds.velocity.shape[-1])
    rows_tr, rows_va = ds.rows(["train"]), ds.rows(["val"])
    sub = np.sort(np.random.default_rng(cfg["seed"]).choice(rows_va, cfg["n_points"], replace=False))
    colour = {"log enstrophy": ds.target("log_Z")[sub], "log(1 + vortex count)": ds.target("log1p_vortex_count")[sub],
              "log ν": ds.target("log_nu")[sub]}
    emb, trav = {}, {}
    for name, run in cfg["models"].items():
        lat = latents_for(run, ds, cfg["readout"], rows=np.concatenate([rows_tr, rows_va]))
        emb[name] = embeddings(lat, rows_tr, sub, cfg["n_pca"], cfg["seed"])
        if name in cfg["traversal_models"]:
            trav[name] = traversals(ds, lat, grid, cfg)
        print(f"{name} done ({time.perf_counter() - t0:.0f} s)", flush=True)
    plot_embeddings(emb, colour, "pca", cfg["out_pca"])
    plot_embeddings(emb, colour, "tsne", cfg["out_tsne"])
    figures = [cfg["out_pca"], cfg["out_tsne"]]
    for name, dirs in trav.items():
        figures.append(cfg["out_traversals"].format(model=name))
        plot_traversals(name, dirs, figures[-1])
    wall = time.perf_counter() - t0
    drop = ("spectrum", "k", "w_values", "nearest_field")
    summary = {name: [{**{k: v for k, v in t.items() if k != "steps"},
                       "steps": [{k: v for k, v in st.items() if k not in drop} for st in t["steps"]]} for t in dirs]
               for name, dirs in trav.items()}
    save_json({"environment": environment(), "config": cfg, "pca_explained": {n: e["explained"] for n, e in emb.items()},
               "traversals": summary, "wall_seconds": wall}, cfg["output_json"])
    for name, dirs in summary.items():
        for t in dirs:
            print(f"{name} dir {t['direction']}: traj |r| log E0 {t['traj_abs_r_logE0']:.2f}, log Z0 {t['traj_abs_r_logZ0']:.2f};",
                  "NN log Z per step:", [round(st["nn_log_Z"], 2) for st in t["steps"]],
                  "trajectories per step:", [st["n_traj"] for st in t["steps"]])
    append_log("Phase 4 views (val; unscored)", cfg["_config_path"], command_line(), "ok",
               {"figures": ", ".join(figures), "results": cfg["output_json"]}, wall, environment().get("gpu"))


if __name__ == "__main__":
    main()
