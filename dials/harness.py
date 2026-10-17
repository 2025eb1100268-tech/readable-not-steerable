"""[BUILD] Steering harness (definitions fixed in results/build/PREDICTIONS.md, entry 1).

For a unit direction d in an adapter's dial subspace and frames (x_prev, x_t): decode with the dial
latent shifted by alpha * sigma_d * d, alpha in {-2, -1.5, ..., 2}; from each decoded field compute
a = log E, s = log(Z/E), log Z, log P. Per direction: median-over-frames slopes per unit alpha,
effect = |da| + |ds|, sigma_shape = |ds| / (|da| + |ds|), monotonicity (median per-frame Spearman of
the dominant coordinate), and realism: cycle consistency (re-encode the decoded vorticity as
velocity), spectrum-envelope fraction (min over alpha), divergence for velocity outputs, and the
head-consistency error for two-head models.
"""
from __future__ import annotations

import numpy as np
import torch

from dials import fields
from probes.ridge import RidgeProbe

ALPHAS = np.round(np.arange(-2.0, 2.0001, 0.5), 2)


def pca_directions(lat_train: np.ndarray, n: int = 4):
    """Top-n principal directions of the raw (centred) latent; returns (n, D) unit rows, sigmas, explained."""
    c = lat_train - lat_train.mean(0)
    _, s, vt = np.linalg.svd(c.astype(np.float64), full_matrices=False)
    dirs = vt[:n]
    return dirs, (c @ dirs.T).std(0), (s[:n] ** 2 / (s**2).sum())


def supervised_direction(lat_tr, y_tr, f_tr, lat_va, y_va, f_va):
    """Ridge-probe direction for a target, mapped back to raw latent space and unit-normalized."""
    p = RidgeProbe().fit(lat_tr, y_tr, f_tr, lat_va, y_va, f_va)
    w = p.coef / p.sd
    d = w / np.linalg.norm(w)
    return d, float((lat_tr @ d).std()), float(p.alpha)


def envelope(train_vel: torch.Tensor, g, lo=1.0, hi=99.0):
    """Per-shell 1st/99th percentiles of the energy spectra of true training fields."""
    spec = fields.shell_spectrum(fields.vorticity_hat(train_vel, "velocity", g), g)
    return torch.quantile(spec, lo / 100, dim=0), torch.quantile(spec, hi / 100, dim=0)


def _slopes(y: np.ndarray) -> np.ndarray:
    """y: (n_alpha, n_frames) -> per-frame least-squares slope over ALPHAS."""
    a = ALPHAS - ALPHAS.mean()
    return (a[:, None] * (y - y.mean(0))).sum(0) / (a**2).sum()


def _spearman_vs_alpha(y: np.ndarray) -> np.ndarray:
    """Per-frame Spearman correlation between ALPHAS (increasing) and y."""
    r = np.argsort(np.argsort(y, axis=0), axis=0).astype(float)
    ra = np.arange(len(ALPHAS), dtype=float)[:, None]
    rc, ac = r - r.mean(0), ra - ra.mean()
    den = np.sqrt((rc**2).sum(0) * (ac**2).sum())
    return np.where(den > 0, (rc * ac).sum(0) / np.where(den > 0, den, 1), 0.0)


@torch.no_grad()
def steer(adapter, x_prev, x_t, d: np.ndarray, sigma: float, g, env) -> dict:
    dev = x_t.device
    d_t = torch.as_tensor(d, dtype=torch.float32, device=dev)
    env_lo, env_hi = env
    per_head: dict = {}
    z_rec, cons, div = [], [], []
    primary_field = {}
    for alpha in ALPHAS:
        outs = adapter.decode(x_prev, x_t, float(alpha * sigma) * d_t)
        if alpha in (-2.0, 0.0, 2.0):
            primary_field[float(alpha)] = outs[0][2].double().flatten(1)
        w_hats = {}
        for name, kind, f in outs:
            w = fields.vorticity_hat(f, kind, g)
            w_hats[name] = w
            st = fields.stats(w, g)
            inside = ((fields.shell_spectrum(w, g) >= env_lo) & (fields.shell_spectrum(w, g) <= env_hi)).double().mean(1)
            h = per_head.setdefault(name, {"kind": kind, "a": [], "s": [], "log Z": [], "log P": [], "inside": []})
            for k_ in ("a", "s", "log Z", "log P"):
                h[k_].append(st[k_].cpu().numpy())
            h["inside"].append(float(inside.mean()))
            if kind == "velocity" and name == outs[0][0]:
                div.append(float(fields.relative_divergence(f, g).median()))
        primary = w_hats[outs[0][0]]
        z_rec.append(adapter.reencode(fields.velocity_from_vorticity_hat(primary, g)).double().cpu().numpy())
        if "vorticity" in w_hats and "streamfunction" in w_hats:
            cons.append(float(fields.head_consistency(w_hats["vorticity"], w_hats["streamfunction"], g).median()))

    heads = {}
    for name, h in per_head.items():
        y = {k_: np.stack(h[k_]) for k_ in ("a", "s", "log Z", "log P")}
        sl = {k_: float(np.median(_slopes(v))) for k_, v in y.items()}
        dom = "a" if abs(sl["a"]) >= abs(sl["s"]) else "s"
        effect = abs(sl["a"]) + abs(sl["s"])
        heads[name] = {"kind": h["kind"], "d_a": sl["a"], "d_s": sl["s"], "d_logZ": sl["log Z"], "d_logP": sl["log P"],
                       "effect": effect, "sigma_shape": abs(sl["s"]) / effect if effect > 0 else float("nan"),
                       "monotonicity": float(np.median(np.abs(_spearman_vs_alpha(y[dom])))), "dominant": dom,
                       "envelope_min": float(min(h["inside"])), "envelope_by_alpha": h["inside"]}
    z = np.stack(z_rec)  # (n_alpha, B, D)
    i0 = int(np.flatnonzero(ALPHAS == 0)[0])
    dz = z - z[i0]
    dproj = dz @ d
    cmd = np.broadcast_to((ALPHAS * sigma)[:, None], dproj.shape)
    mask = ALPHAS != 0
    cyc_corr = float(np.corrcoef(cmd[mask].ravel(), dproj[mask].ravel())[0, 1])
    gain = float((cmd[mask] * dproj[mask]).sum() / (cmd[mask] ** 2).sum())
    ends = np.flatnonzero(np.abs(ALPHAS) == 2)
    ortho = dz[ends] - dproj[ends][..., None] * d
    drift = float(np.median(np.linalg.norm(ortho, axis=-1) / (2 * sigma)))
    # watermark test (added after Stage 0): is the change the same pattern for every frame?
    shared, align = [], []
    for end in (-2.0, 2.0):
        diff = primary_field[end] - primary_field[0.0]
        tot = float((diff**2).sum())
        shared.append(1.0 - float(((diff - diff.mean(0, keepdim=True)) ** 2).sum()) / tot if tot > 0 else float("nan"))
        align.append(float(torch.nn.functional.cosine_similarity(diff, primary_field[0.0], dim=1).abs().median()))
    first = next(iter(heads))
    out = {**heads[first], "head": first, "cycle_corr": cyc_corr, "cycle_gain": gain, "drift": drift,
           "shared_pattern_fraction": float(max(shared)), "change_field_alignment": float(min(align)), "heads": heads}
    if div:
        out["rel_divergence_max"] = float(max(div))
    if cons:
        out["head_consistency_max"] = float(max(cons))
    return out


def classify(m: dict, crit: dict) -> dict:
    """Dial criteria on one dataset's metrics (PREDICTIONS entry 1)."""
    dial = (m["effect"] >= crit["min_effect"] and m["monotonicity"] >= crit["min_monotonicity"]
            and m["cycle_corr"] >= crit["min_cycle_corr"] and m["envelope_min"] >= crit["min_envelope"])
    if "head_consistency_max" in m and crit.get("max_head_inconsistency") is not None:
        dial = dial and m["head_consistency_max"] <= crit["max_head_inconsistency"]
    return {"dial": bool(dial), "loudness": bool(dial and m["sigma_shape"] <= crit["max_sigma_loudness"]),
            "shape": bool(dial and m["sigma_shape"] >= crit["min_sigma_shape"])}
