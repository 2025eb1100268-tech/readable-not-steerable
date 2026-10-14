"""Phase 1: untrained-encoder baselines on train/val only (test sets untouched).

These numbers calibrate the pre-registered criteria before any training:
- S1 targets, ridge on pooled latents, plain and time-partialled R^2 on val (fit on train),
  trajectory-bootstrap SD, and the R^2 a trained model would need under the approved rule
  (gain >= 0.20 OR unexplained variance halved).
- S2: log nu from z-scored concat [z_{t-1}, z_t] on viscous frames (+ single-frame, unscored).
- S3 stand-in: slow directions fitted on train inviscid trajectories, scored on val inviscid.

    python -m probes.untrained_baselines --config configs/phase1_untrained_baselines.yaml
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from common.run import append_log, command_line, environment, load_config, pick_device, save_json, seed_everything
from data.dataset import S1_TARGETS, TRANSFORMS, VISCOUS_ONLY, Dataset
from models.encoder import build_encoder, encode_rows
from probes import slow_features as sfeat
from probes.ridge import RidgeProbe, bootstrap_sd


def needed_r2(r_rand: float) -> float:
    """R^2 a trained model needs to count a target under the approved S1 rule (either branch)."""
    return min(r_rand + 0.20, 1.0 - 0.5 * (1.0 - r_rand))


def probe_target(ds, lat_of, name, rows_tr, rows_va, partial, index="time_x_family"):
    """index: 'time_x_family' (pre-registered S1 partialling, amendment 2.1a) or 'time'."""
    y = ds.target(name)
    if name in VISCOUS_ONLY:
        rows_tr = rows_tr[~ds.inviscid[rows_tr]]
        rows_va = rows_va[~ds.inviscid[rows_va]]
    idx = ds.partial_index() if index == "time_x_family" else ds.frame
    n_index = 2 * ds.n_frame_max if index == "time_x_family" else ds.n_frame_max
    p = RidgeProbe(partial=partial, n_frame_index=n_index).fit(lat_of(rows_tr), y[rows_tr], idx[rows_tr],
                                                               lat_of(rows_va), y[rows_va], idx[rows_va])
    s = p.score(lat_of(rows_va), y[rows_va], idx[rows_va])
    pred = p.predict(lat_of(rows_va), idx[rows_va])
    base = p._time(idx[rows_va], len(rows_va)) if partial else np.full(len(rows_va), y[rows_va].mean())
    s["boot_sd"] = bootstrap_sd(y[rows_va], pred, base, ds.traj[rows_va])
    s["alpha"], s["alpha_at_edge"] = p.alpha, p.alpha_at_edge
    return s


def nu_probes(ds, lat_of, rows_tr, rows_va):
    y = ds.target("log_nu")
    out = {}
    for split_rows, key in ((rows_tr, "tr"), (rows_va, "va")):
        r = split_rows[(~ds.inviscid[split_rows]) & (ds.frame[split_rows] >= 1)]
        out[key] = r
    pair = lambda r: np.concatenate([lat_of(r - 1), lat_of(r)], 1)  # noqa: E731
    p = RidgeProbe().fit(pair(out["tr"]), y[out["tr"]], ds.frame[out["tr"]], pair(out["va"]), y[out["va"]], ds.frame[out["va"]])
    res = {"pair_r2": p.score(pair(out["va"]), y[out["va"]], ds.frame[out["va"]])["r2"], "pair_alpha_at_edge": p.alpha_at_edge}
    p1 = RidgeProbe().fit(lat_of(out["tr"]), y[out["tr"]], ds.frame[out["tr"]], lat_of(out["va"]), y[out["va"]], ds.frame[out["va"]])
    res["single_r2"] = p1.score(lat_of(out["va"]), y[out["va"]], ds.frame[out["va"]])["r2"]
    return res


def slow_direction_baseline(ds, lat_of, n_pca, n_boot=500):
    fit_rows = ds.rows(["train"], viscous=False)
    ev_rows = ds.rows(["val"], viscous=False)
    model = sfeat.fit(lat_of(fit_rows), ds.traj[fit_rows], n_pca)
    first = {t: np.flatnonzero((ds.traj == t) & (ds.frame == 0))[0] for t in np.unique(ds.traj[ev_rows])}
    inv = {"log_E": np.log(ds.f["E"]), "log_Z": np.log(ds.f["Z"]), "log_C4 (approx.)": np.log(ds.f["C4"])}
    targets = {k: {t: v[r] for t, r in first.items()} for k, v in inv.items()}
    s = sfeat.score(model, lat_of(ev_rows), ds.traj[ev_rows], targets, top=3)
    # trajectory bootstrap of the max |r| over the top 3 directions
    proj = sfeat.project(model, lat_of(ev_rows))[:, :3]
    ids, invidx = np.unique(ds.traj[ev_rows], return_inverse=True)
    means = np.stack([np.bincount(invidx, weights=proj[:, j]) / np.bincount(invidx) for j in range(3)], 1)
    rng = np.random.default_rng(0)
    boot = {}
    for k in targets:
        yv = np.array([targets[k][t] for t in ids])
        vals = []
        for _ in range(n_boot):
            pick = rng.integers(0, len(ids), len(ids))
            vals.append(max(abs(np.corrcoef(means[pick, j], yv[pick])[0, 1]) for j in range(3)))
        boot[k] = [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
    s["max_abs_r_ci95"] = boot
    s["n_fit_traj"], s["n_eval_traj"] = int(len(np.unique(ds.traj[fit_rows]))), int(len(ids))
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = load_config(ap.parse_args().config)
    seed_everything(cfg["seed"])
    device = pick_device(cfg["device"])
    t0 = time.perf_counter()
    ds = Dataset(cfg["data_dir"])
    rows_tr, rows_va = ds.rows(["train"]), ds.rows(["val"])
    rows_all = np.concatenate([rows_tr, rows_va])
    vel = np.asarray(ds.velocity[np.sort(rows_all)])
    pos = np.full(len(ds.traj), -1)
    pos[np.sort(rows_all)] = np.arange(len(rows_all))
    # S2 pairs need the previous frame, which is always in the same split (same trajectory)

    names = S1_TARGETS + [n for n in TRANSFORMS if n not in S1_TARGETS and n not in ("log_nu",)]
    per_seed = []
    for seed in cfg["encoder_seeds"]:
        lat = encode_rows(build_encoder(seed), vel, np.arange(len(vel)), ds.input_scale, device)
        lat_of = lambda r, lat=lat: lat[pos[r]]  # noqa: E731
        res = {"seed": seed, "latent_std_over_mean": float(np.median(lat.std(0) / (np.abs(lat.mean(0)) + 1e-12)))}
        ev = np.linalg.eigvalsh(np.cov(lat[pos[rows_tr]], rowvar=False))
        ev = np.clip(ev, 0, None) / ev.sum()
        res["effective_rank"] = float(np.exp(-(ev[ev > 0] * np.log(ev[ev > 0])).sum()))
        res["targets"] = {}
        for name in names:
            partial = name != "time"
            res["targets"][name] = probe_target(ds, lat_of, name, rows_tr, rows_va, partial)
            if partial:
                res["targets"][name]["plain_r2"] = probe_target(ds, lat_of, name, rows_tr, rows_va, False)["r2"]
                res["targets"][name]["r2_partial_time_only"] = probe_target(
                    ds, lat_of, name, rows_tr, rows_va, True, index="time")["r2_partial"]
        res["nu"] = nu_probes(ds, lat_of, rows_tr, rows_va)
        res["slow"] = slow_direction_baseline(ds, lat_of, cfg["n_pca"])
        per_seed.append(res)
        print(f"seed {seed} done ({time.perf_counter() - t0:.0f} s)")

    def agg(get):
        v = np.array([get(r) for r in per_seed], float)
        return {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0}

    summary = {"targets": {}, "nu": {}, "slow": {}}
    for name in names:
        key = "r2" if name == "time" else "r2_partial"
        summary["targets"][name] = {"val_r2_partial" if key == "r2_partial" else "val_r2": agg(lambda r: r["targets"][name][key]),
                                    "val_r2_plain": agg(lambda r: r["targets"][name].get("plain_r2", r["targets"][name]["r2"])),
                                    "boot_sd": agg(lambda r: r["targets"][name]["boot_sd"]),
                                    "val_r2_partial_time_only": agg(lambda r: r["targets"][name].get("r2_partial_time_only", np.nan)),
                                    "alpha_at_edge_any": any(r["targets"][name]["alpha_at_edge"] for r in per_seed)}
        if name in S1_TARGETS:
            summary["targets"][name]["needed_r2_trained"] = needed_r2(summary["targets"][name]["val_r2_partial"]["mean"])
    summary["nu"] = {"pair_r2": agg(lambda r: r["nu"]["pair_r2"]), "single_r2": agg(lambda r: r["nu"]["single_r2"])}
    summary["nu"]["needed_r2_trained"] = max(0.6, summary["nu"]["pair_r2"]["mean"] + 0.3)
    for k in ("log_E", "log_Z", "log_C4 (approx.)"):
        summary["slow"][k] = {"max_abs_r": agg(lambda r: r["slow"]["max_abs_r"][k]),
                              "ci95_seed0": per_seed[0]["slow"]["max_abs_r_ci95"][k]}
        summary["slow"][k]["needed_r_trained"] = max(0.8, summary["slow"][k]["max_abs_r"]["mean"] + 0.1)
    summary["slow"]["lambda_heldout_top3_seed_mean"] = np.mean([r["slow"]["lambda_heldout"] for r in per_seed], 0).tolist()
    summary["effective_rank"] = agg(lambda r: r["effective_rank"])
    summary["latent_std_over_mean"] = agg(lambda r: r["latent_std_over_mean"])
    wall = time.perf_counter() - t0
    save_json({"environment": environment(), "config": cfg, "per_seed": per_seed, "summary": summary,
               "n_rows": {"train": int(len(rows_tr)), "val": int(len(rows_va))}}, cfg["output_json"])

    lines = ["| target | untrained val R², time × family partialled (pre-registered), mean ± std over 3 seeds | time-only partialled | plain R² | bootstrap SD | R² a trained model needs |",
             "|---|---|---|---|---|---|"]
    for name in names:
        s = summary["targets"][name]
        main_key = "val_r2_partial" if "val_r2_partial" in s else "val_r2"
        need = f"{s['needed_r2_trained']:.3f}" if "needed_r2_trained" in s else "—"
        t_only = s["val_r2_partial_time_only"]["mean"]
        t_only = "—" if np.isnan(t_only) else f"{t_only:.3f}"
        lines.append(f"| {name}{' (S1)' if name in S1_TARGETS else ''} | {s[main_key]['mean']:.3f} ± {s[main_key]['std']:.3f} | "
                     f"{t_only} | {s['val_r2_plain']['mean']:.3f} | {s['boot_sd']['mean']:.3f} | {need} |")
    n = summary["nu"]
    lines.append(f"| log ν, frame pair (S2) | {n['pair_r2']['mean']:.3f} ± {n['pair_r2']['std']:.3f} | — | — | — | {n['needed_r2_trained']:.3f} |")
    lines.append(f"| log ν, single frame (unscored) | {n['single_r2']['mean']:.3f} ± {n['single_r2']['std']:.3f} | — | — | — | — |")
    lines += ["", "| invariant (inviscid, fit train → score val) | untrained max \\|r\\| over top-3, mean ± std | 95% CI (seed 0) | \\|r\\| a trained model needs |",
              "|---|---|---|---|"]
    for k in ("log_E", "log_Z", "log_C4 (approx.)"):
        s = summary["slow"][k]
        lines.append(f"| {k} | {s['max_abs_r']['mean']:.3f} ± {s['max_abs_r']['std']:.3f} | {s['ci95_seed0'][0]:.2f}–{s['ci95_seed0'][1]:.2f} | "
                     f"{s['needed_r_trained']:.3f}{' (unreachable)' if s['needed_r_trained'] > 1 else ''} |")
    md = "\n".join(lines)
    with open(cfg["output_md"], "w", encoding="utf-8") as f:
        f.write("# Phase 1 untrained-encoder baselines (val split; fit on train)\n\n" + md + "\n")
    print(md)
    append_log("Phase 1 untrained baselines (train/val only)", cfg["_config_path"], command_line(), "ok",
               {"S1 partial R² (Z, P, count, OW, eps)": ", ".join(f"{summary['targets'][t]['val_r2_partial']['mean']:.3f}" for t in S1_TARGETS),
                "S2 log-nu pair R²": f"{n['pair_r2']['mean']:.3f}",
                "S3 untrained max|r| E / Z": f"{summary['slow']['log_E']['max_abs_r']['mean']:.3f} / {summary['slow']['log_Z']['max_abs_r']['mean']:.3f}",
                "results": cfg["output_json"]}, wall, device)


if __name__ == "__main__":
    main()
