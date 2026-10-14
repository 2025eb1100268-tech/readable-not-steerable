"""Pre-registered decision rules (PREREGISTRATION.md §1–§2), applied to outputs of
probes.evaluate for the JEPA and the untrained control (3 seeds each).

S1 (2.1, 2.1a): per target, seed-averaged time x family partialled R^2. Counts if
    (gain >= 0.20 or (1 - R2_J) <= 0.5 (1 - R2_C)) and the paired trajectory-bootstrap 95% CI of
    the gain excludes 0. Holds if >= 3 of 5 count and the Okubo–Weiss fraction counts.
S2 (2.2): seed-mean pair R^2 for log nu >= 0.6 and >= control + 0.3.
S3 (2.3): for E or Z: JEPA seed-mean max|r| over top 3 >= 0.8, control < 0.8, gap >= 0.1, and
    the trajectory-bootstrap 95% CI of the gap excludes 0.
"""
from __future__ import annotations

import numpy as np

from data.dataset import S1_TARGETS

MANDATORY = "ow_vort_frac"
N_BOOT = 1000


def _sse_matrix(runs: list[dict], target: str, ids: np.ndarray):
    """(seeds, n_traj) model SSE and (n_traj,) base SSE, aligned to ids."""
    model = []
    for r in runs:
        d = r["probes"][target]["sse_model_per_traj"]
        m = dict(zip(d["traj"], d["sum"]))
        model.append([m[i] for i in ids])
    b = runs[0]["probes"][target]["sse_base_per_traj"]
    bm = dict(zip(b["traj"], b["sum"]))
    return np.array(model), np.array([bm[i] for i in ids])


def s1(jepa: list[dict], control: list[dict], seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    out = {"targets": {}}
    for t in S1_TARGETS:
        ids = np.array(jepa[0]["probes"][t]["sse_model_per_traj"]["traj"])
        mj, base = _sse_matrix(jepa, t, ids)
        mc, base_c = _sse_matrix(control, t, ids)
        assert np.allclose(base, base_c)  # base depends only on the data
        r2 = lambda m, pick: float(np.mean(1 - m[:, pick].sum(1) / base[pick].sum()))  # noqa: E731
        all_ = np.arange(len(ids))
        r2j, r2c = r2(mj, all_), r2(mc, all_)
        boot = []
        for _ in range(N_BOOT):
            pick = rng.integers(0, len(ids), len(ids))
            boot.append(r2(mj, pick) - r2(mc, pick))
        lo, hi = np.percentile(boot, [2.5, 97.5])
        gain_ok = (r2j - r2c >= 0.20) or ((1 - r2j) <= 0.5 * (1 - r2c))
        out["targets"][t] = {"r2_jepa": r2j, "r2_control": r2c, "gain": r2j - r2c, "ci95": [float(lo), float(hi)],
                             "margin_or_halving": bool(gain_ok), "ci_excludes_zero": bool(lo > 0),
                             "counts": bool(gain_ok and lo > 0)}
    counted = [t for t, v in out["targets"].items() if v["counts"]]
    out["n_counted"] = len(counted)
    out["mandatory_counts"] = MANDATORY in counted
    out["holds"] = len(counted) >= 3 and out["mandatory_counts"]
    return out


def s2(jepa: list[dict], control: list[dict]) -> dict:
    rj = float(np.mean([r["nu"]["pair_r2"] for r in jepa]))
    rc = float(np.mean([r["nu"]["pair_r2"] for r in control]))
    return {"r2_jepa": rj, "r2_control": rc, "holds": bool(rj >= 0.6 and rj - rc >= 0.3),
            "single_frame_jepa": float(np.mean([r["nu"]["single_r2"] for r in jepa])),
            "single_frame_control": float(np.mean([r["nu"]["single_r2"] for r in control]))}


def _max_r(runs: list[dict], inv: str, pick: np.ndarray) -> float:
    vals = []
    for r in runs:
        m = np.array(r["slow"]["proj_traj_means"])[pick]
        y = np.array(r["slow"]["targets"][inv])[pick]
        vals.append(max(abs(np.corrcoef(m[:, j], y)[0, 1]) for j in range(m.shape[1])))
    return float(np.mean(vals))


def s3(jepa: list[dict], control: list[dict], seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    ids = jepa[0]["slow"]["traj"]
    assert all(r["slow"]["traj"] == ids for r in jepa + control)
    n = len(ids)
    out = {"invariants": {}}
    for inv in ("log_E", "log_Z"):
        all_ = np.arange(n)
        rj, rc = _max_r(jepa, inv, all_), _max_r(control, inv, all_)
        boot = []
        for _ in range(N_BOOT):
            pick = rng.integers(0, n, n)
            boot.append(_max_r(jepa, inv, pick) - _max_r(control, inv, pick))
        lo, hi = np.percentile(boot, [2.5, 97.5])
        ok = rj >= 0.8 and rc < 0.8 and rj - rc >= 0.1 and lo > 0
        out["invariants"][inv] = {"r_jepa": rj, "r_control": rc, "gap": rj - rc, "ci95": [float(lo), float(hi)], "holds": bool(ok)}
    out["holds"] = any(v["holds"] for v in out["invariants"].values())
    return out


def verdict(s1r: dict, s2r: dict, s3r: dict) -> str:
    if not s1r["holds"]:
        return "not supported"
    return "supported" if (s2r["holds"] and s3r["holds"]) else "partially supported"
