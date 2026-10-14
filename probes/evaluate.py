"""Evaluation core shared by the Phase 3 val preview and the single final test run.

Everything here follows PREREGISTRATION.md §5–§6. Probes are always fitted on train with ridge
alpha chosen on val; ``eval_split`` only changes where they are scored. Per-trajectory squared
errors are kept so the paired trajectory bootstrap (§5.6) can be computed across models.
"""
from __future__ import annotations

import math

import numpy as np

from data.dataset import S1_TARGETS, VISCOUS_ONLY, Dataset
from probes import slow_features as sfeat
from probes.ridge import RidgeProbe

PROBE_TARGETS = S1_TARGETS + ["log_E", "log_max_abs_w", "log_kurtosis", "ow_strain_frac"]


def _per_traj(values: np.ndarray, traj: np.ndarray) -> dict:
    ids, inv = np.unique(traj, return_inverse=True)
    return {"traj": ids.tolist(), "sum": np.bincount(inv, weights=values).tolist()}


def probe_targets(ds: Dataset, lat: np.ndarray, eval_split: str) -> dict:
    rows_tr, rows_va, rows_ev = ds.rows(["train"]), ds.rows(["val"]), ds.rows([eval_split])
    pidx, n_pidx = ds.partial_index(), 2 * ds.n_frame_max
    out = {}
    for name in PROBE_TARGETS:
        y = ds.target(name)
        tr, va, ev = rows_tr, rows_va, rows_ev
        if name in VISCOUS_ONLY:
            tr, va, ev = (r[~ds.inviscid[r]] for r in (tr, va, ev))
        p = RidgeProbe(partial=True, n_frame_index=n_pidx).fit(lat[tr], y[tr], pidx[tr], lat[va], y[va], pidx[va])
        pred, base = p.predict(lat[ev], pidx[ev]), p._time(pidx[ev], len(ev))
        sse_m, sse_b = (y[ev] - pred) ** 2, (y[ev] - base) ** 2
        res = {"r2_partial": float(1 - sse_m.sum() / sse_b.sum()), "alpha": p.alpha, "alpha_at_edge": p.alpha_at_edge,
               "sse_model_per_traj": _per_traj(sse_m, ds.traj[ev]), "sse_base_per_traj": _per_traj(sse_b, ds.traj[ev])}
        pt = RidgeProbe(partial=True, n_frame_index=ds.n_frame_max).fit(lat[tr], y[tr], ds.frame[tr], lat[va], y[va], ds.frame[va])
        res["r2_partial_time_only"] = pt.score(lat[ev], y[ev], ds.frame[ev])["r2_partial"]
        pp = RidgeProbe().fit(lat[tr], y[tr], ds.frame[tr], lat[va], y[va], ds.frame[va])
        res["r2_plain"] = pp.score(lat[ev], y[ev], ds.frame[ev])["r2"]
        out[name] = res
    y = ds.target("time")
    pp = RidgeProbe().fit(lat[rows_tr], y[rows_tr], ds.frame[rows_tr], lat[rows_va], y[rows_va], ds.frame[rows_va])
    out["time"] = {"r2_plain": pp.score(lat[rows_ev], y[rows_ev], ds.frame[rows_ev])["r2"]}
    return out


def probe_nu(ds: Dataset, lat: np.ndarray, eval_split: str) -> dict:
    """S2: log nu from z-scored [z_{t-1}, z_t] on viscous frames (consecutive stored frames)."""
    y = ds.target("log_nu")
    sel = lambda split: ds.rows([split], viscous=True)[ds.frame[ds.rows([split], viscous=True)] >= 1]  # noqa: E731
    tr, va, ev = sel("train"), sel("val"), sel(eval_split)
    pair = lambda r: np.concatenate([lat[r - 1], lat[r]], 1)  # noqa: E731
    p = RidgeProbe().fit(pair(tr), y[tr], ds.frame[tr], pair(va), y[va], ds.frame[va])
    s = RidgeProbe().fit(lat[tr], y[tr], ds.frame[tr], lat[va], y[va], ds.frame[va])
    return {"pair_r2": p.score(pair(ev), y[ev], ds.frame[ev])["r2"], "single_r2": s.score(lat[ev], y[ev], ds.frame[ev])["r2"],
            "pair_alpha_at_edge": p.alpha_at_edge}


def probe_ood(ds: Dataset, lat: np.ndarray) -> dict:
    """§5.5: plain probes fitted on train; MSE on test-OOD vs neighbouring-band test-ID viscous
    trajectories and vs a constant predictor at the train mean."""
    visc = [t for t in ds.trajectories if not t["inviscid"]]
    log_re = np.array([math.log(t["Re"]) for t in visc])
    q = np.quantile(log_re, [0.35, 0.45, 0.55, 0.65])
    near = {t["traj"] for t in visc if t["split"] == "test_id" and (q[0] <= math.log(t["Re"]) < q[1] or q[2] < math.log(t["Re"]) <= q[3])}
    rows_near = np.flatnonzero(np.isin(ds.traj, list(near)))
    rows_tr, rows_va, rows_ood = ds.rows(["train"]), ds.rows(["val"]), ds.rows(["test_ood"])
    out = {"n_traj_neighbour_id": len(near)}
    for name in PROBE_TARGETS + ["log_nu"]:
        y = ds.target(name)
        tr, va = rows_tr, rows_va
        if name in VISCOUS_ONLY:
            tr, va = tr[~ds.inviscid[tr]], va[~ds.inviscid[va]]
        p = RidgeProbe().fit(lat[tr], y[tr], ds.frame[tr], lat[va], y[va], ds.frame[va])
        mse = lambda r: float(((y[r] - p.predict(lat[r], ds.frame[r])) ** 2).mean())  # noqa: E731
        const = lambda r: float(((y[r] - y[tr].mean()) ** 2).mean())  # noqa: E731
        out[name] = {"mse_ood": mse(rows_ood), "mse_neighbour_id": mse(rows_near),
                     "mse_const_ood": const(rows_ood), "mse_const_neighbour_id": const(rows_near)}
    return out


def slow_directions(ds: Dataset, lat: np.ndarray, fit_splits, eval_split: str, n_pca: int = 32) -> dict:
    """§6. Fit on inviscid trajectories of fit_splits, score on inviscid eval_split trajectories."""
    fit_rows = ds.rows(fit_splits, viscous=False)
    ev_rows = ds.rows([eval_split], viscous=False)
    model = sfeat.fit(lat[fit_rows], ds.traj[fit_rows], n_pca)
    proj = sfeat.project(model, lat[ev_rows])[:, :3]
    ids, inv = np.unique(ds.traj[ev_rows], return_inverse=True)
    counts = np.bincount(inv)
    means = np.stack([np.bincount(inv, weights=proj[:, j]) / counts for j in range(3)], 1)
    first = np.array([np.flatnonzero((ds.traj == t) & (ds.frame == 0))[0] for t in ids])
    traj_mean = lambda v: np.bincount(inv, weights=v[ev_rows]) / counts  # noqa: E731
    targets = {"log_E": np.log(ds.f["E"][first]), "log_Z": np.log(ds.f["Z"][first]),
               "log_C4": np.log(ds.f["C4"][first]),
               "kurtosis (traj mean)": traj_mean(ds.f["kurtosis"]), "log max|w| (traj mean)": traj_mean(np.log(ds.f["max_abs_w"])),
               "OW fraction (traj mean)": traj_mean(ds.f["ow_vort_frac"])}
    abs_r = {k: [float(abs(np.corrcoef(means[:, j], v)[0, 1])) for j in range(3)] for k, v in targets.items()}
    return {"traj": ids.tolist(), "proj_traj_means": means.tolist(), "targets": {k: v.tolist() for k, v in targets.items()},
            "abs_r": abs_r, "max_abs_r": {k: max(v) for k, v in abs_r.items()},
            "lambda_heldout": sfeat.within_between_ratio(proj, ds.traj[ev_rows]).tolist(),
            "lambda_fit": model["lambda_fit"][:3].tolist(), "shrinkage": model["shrinkage"]}
