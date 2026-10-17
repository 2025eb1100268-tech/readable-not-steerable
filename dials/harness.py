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


def _inside(w_hat: torch.Tensor, g, env) -> torch.Tensor:
    """Per-frame fraction of shells k = 1..K_MAX inside the envelope."""
    sp = fields.shell_spectrum(w_hat, g)
    return ((sp >= env[0]) & (sp <= env[1])).double().mean(1)


@torch.no_grad()
def ideal_envelope(w0: torch.Tensor, a: np.ndarray, s: np.ndarray, g, env, iters: int = 30) -> list:
    """[BUILD] entry 2. Envelope fraction (mean over frames) at each alpha of an IDEAL dial making the
    same measured change: per frame, the unsteered decoded field w0 is tilted by k^tau (tau by
    bisection, so s equals the steered field's s) and rescaled (so a equals the steered field's a).
    a, s: (n_alpha, B) of the steered fields. Same a and s, so any envelope shortfall relative to this
    reference comes from how the steered spectrum is shaped beyond (a, s)."""
    kk = torch.where(g.kmag > 0, g.kmag, torch.ones_like(g.kmag))
    out = []
    for i, alpha in enumerate(ALPHAS):
        if alpha == 0:
            out.append(float(_inside(w0, g, env).mean()))
            continue
        tgt_s = torch.as_tensor(s[i], dtype=g.dtype, device=w0.device)
        lo, hi = torch.full_like(tgt_s, -4.0), torch.full_like(tgt_s, 4.0)
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            below = fields.stats(w0 * kk ** mid[:, None, None], g)["s"] < tgt_s
            lo, hi = torch.where(below, mid, lo), torch.where(below, hi, mid)
        w = w0 * kk ** (0.5 * (lo + hi))[:, None, None]
        tgt_a = torch.as_tensor(a[i], dtype=g.dtype, device=w0.device)
        w = w * torch.exp((tgt_a - fields.stats(w, g)["a"]) / 2)[:, None, None]
        out.append(float(_inside(w, g, env).mean()))
    return out


def _cycle(z: np.ndarray, d: np.ndarray, sigma: float) -> dict:
    """z: (n_alpha, B, D) re-encoded latents -> correlation and gain along d, drift orthogonal to d."""
    i0 = int(np.flatnonzero(ALPHAS == 0)[0])
    dz = z - z[i0]
    dproj = dz @ d
    cmd = np.broadcast_to((ALPHAS * sigma)[:, None], dproj.shape)
    mask = ALPHAS != 0
    ends = np.flatnonzero(np.abs(ALPHAS) == 2)
    ortho = dz[ends] - dproj[ends][..., None] * d
    return {"cycle_corr": float(np.corrcoef(cmd[mask].ravel(), dproj[mask].ravel())[0, 1]),
            "cycle_gain": float((cmd[mask] * dproj[mask]).sum() / (cmd[mask] ** 2).sum()),
            "drift": float(np.median(np.linalg.norm(ortho, axis=-1) / (2 * sigma)))}


@torch.no_grad()
def steer(adapter, x_prev, x_t, d: np.ndarray, sigma: float, g, env) -> dict:
    dev = x_t.device
    d_t = torch.as_tensor(d, dtype=torch.float32, device=dev)
    per_head: dict = {}
    z_rec: dict = {}
    cons, div = [], []
    primary_field, w_at0 = {}, {}
    for alpha in ALPHAS:
        outs = adapter.decode(x_prev, x_t, float(alpha * sigma) * d_t)
        if alpha in (-2.0, 0.0, 2.0):
            primary_field[float(alpha)] = outs[0][2].double().flatten(1)
        w_hats = {}
        for name, kind, f in outs:
            w = fields.vorticity_hat(f, kind, g)
            w_hats[name] = w
            if alpha == 0:
                w_at0[name] = w
            st = fields.stats(w, g)
            h = per_head.setdefault(name, {"kind": kind, "a": [], "s": [], "log Z": [], "log P": [], "inside": []})
            for k_ in ("a", "s", "log Z", "log P"):
                h[k_].append(st[k_].cpu().numpy())
            h["inside"].append(float(_inside(w, g, env).mean()))
            if kind == "velocity" and name == outs[0][0]:
                div.append(float(fields.relative_divergence(f, g).median()))
            # cycle: each head's field, as velocity (inverse Laplacian), re-encoded
            z_rec.setdefault(name, []).append(adapter.reencode(fields.velocity_from_vorticity_hat(w, g)).double().cpu().numpy())
        if "vorticity" in w_hats and "streamfunction" in w_hats:
            cons.append(float(fields.head_consistency(w_hats["vorticity"], w_hats["streamfunction"], g).median()))

    heads = {}
    for name, h in per_head.items():
        y = {k_: np.stack(h[k_]) for k_ in ("a", "s", "log Z", "log P")}
        sl = {k_: float(np.median(_slopes(v))) for k_, v in y.items()}
        dom = "a" if abs(sl["a"]) >= abs(sl["s"]) else "s"
        effect = abs(sl["a"]) + abs(sl["s"])
        ideal = ideal_envelope(w_at0[name], y["a"], y["s"], g, env)
        heads[name] = {"kind": h["kind"], "d_a": sl["a"], "d_s": sl["s"], "d_logZ": sl["log Z"], "d_logP": sl["log P"],
                       "effect": effect, "sigma_shape": abs(sl["s"]) / effect if effect > 0 else float("nan"),
                       "monotonicity": float(np.median(np.abs(_spearman_vs_alpha(y[dom])))), "dominant": dom,
                       "envelope_min": float(min(h["inside"])), "envelope_by_alpha": h["inside"],
                       "envelope_ideal_by_alpha": ideal,
                       "envelope_rel_min": float(min(a_ - b_ for a_, b_ in zip(h["inside"], ideal))),
                       **_cycle(np.stack(z_rec[name]), d, sigma)}
    # watermark test (added after Stage 0): is the change the same pattern for every frame?
    shared, align = [], []
    for end in (-2.0, 2.0):
        diff = primary_field[end] - primary_field[0.0]
        tot = float((diff**2).sum())
        shared.append(1.0 - float(((diff - diff.mean(0, keepdim=True)) ** 2).sum()) / tot if tot > 0 else float("nan"))
        align.append(float(torch.nn.functional.cosine_similarity(diff, primary_field[0.0], dim=1).abs().median()))
    first = next(iter(heads))
    out = {**heads[first], "head": first,
           "shared_pattern_fraction": float(max(shared)), "change_field_alignment": float(min(align)), "heads": heads}
    if div:
        out["rel_divergence_max"] = float(max(div))
    if cons:
        i0 = int(np.flatnonzero(ALPHAS == 0)[0])
        out.update({"head_consistency_max": float(max(cons)), "head_consistency_at_0": cons[i0],
                    "head_consistency_ratio": float(max(cons) / cons[i0]), "head_consistency_by_alpha": cons})
    return out


def classify(m: dict, crit: dict) -> dict:
    """Dial criteria on one dataset's metrics: PREDICTIONS entry 1 (Stage 0 keys) or entry 2
    (Stage 1 on: envelope relative to an ideal dial, cycle gain, watermark, head-consistency ratio).
    A criterion is applied only when its key is in crit; "fails" lists the criteria not met."""
    checks = {"effect": m["effect"] >= crit["min_effect"],
              "monotonicity": m["monotonicity"] >= crit["min_monotonicity"],
              "cycle corr": m["cycle_corr"] >= crit["min_cycle_corr"]}
    if crit.get("min_envelope") is not None:
        checks["envelope"] = m["envelope_min"] >= crit["min_envelope"]
    if crit.get("envelope_rel_margin") is not None:
        checks["envelope vs ideal"] = m["envelope_rel_min"] >= -crit["envelope_rel_margin"]
    if crit.get("cycle_gain") is not None:
        checks["cycle gain"] = crit["cycle_gain"][0] <= m["cycle_gain"] <= crit["cycle_gain"][1]
    if crit.get("max_shared_pattern") is not None:
        checks["watermark"] = m["shared_pattern_fraction"] <= crit["max_shared_pattern"]
    if "head_consistency_max" in m and crit.get("max_head_inconsistency") is not None:
        checks["head consistency"] = m["head_consistency_max"] <= crit["max_head_inconsistency"]
    if "head_consistency_ratio" in m and crit.get("max_head_ratio") is not None:
        checks["head ratio"] = m["head_consistency_ratio"] <= crit["max_head_ratio"]
    dial = all(bool(v) for v in checks.values())
    return {"dial": dial, "loudness": bool(dial and m["sigma_shape"] <= crit["max_sigma_loudness"]),
            "shape": bool(dial and m["sigma_shape"] >= crit["min_sigma_shape"]),
            "fails": [k for k, v in checks.items() if not v]}
