"""[BUILD] Exploratory autopsy of the v3 Stage 5 run (results/build/PREDICTIONS.md entry 11). Post hoc; nothing
here changes REPORT_v3's verdict. No training. The sealed sets are not opened: directions, g and every
sealed-set score are read from Stage 5's stored output; everything new is computed on exploration data.

    python -m dials.autopsy --config configs/build/autopsy.yaml

A1  seed 102: g (frame-mean z-space gradient of a), PC2 in z-space, h (the same gradient of s); cos^2(g, PC2),
    cos^2(g, h), the fraction of PC2's shape effect removed by projecting it onto g-perp (linear and steered),
    and the projected z-data's variance along the chosen direction vs along the projected PC2.
A2  the union "loudness-null OR PC2" on the controls, from the stored sealed-set scores.
A3  transformation-matched shape rule: the dial direction whose steered change is best explained by the
    content distortion's family (shells 1..21 times c*exp(t1 phi1 + t2 phi2), c keeping their energy; mean and
    shells > 21 unchanged), found as the symmetry rule finds loudness; scored on the exploration sets.
A4  seed 103's loudness and loudness-null directions with the zero-mean step off and on (exploration sets).
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from common.run import REPO, append_log, command_line, environment, load_config, seed_everything  # noqa: E402
from data.combined import CombinedDataset  # noqa: E402
from dials import fields  # noqa: E402
from dials.adapters import adapter_for  # noqa: E402
from dials.harness import classify, envelope, pca_directions, steer  # noqa: E402
from dials.shape_noise import basis  # noqa: E402
from dials.stage0 import frames  # noqa: E402
from dials.stage1 import load  # noqa: E402
from dials.stepA import OPT_ALPHAS, artefact_fraction, dial_latents, label  # noqa: E402
from dials.stepB import loudness_null  # noqa: E402
from dials.v3_confirm import ZeroMean  # noqa: E402
from dials.whiten import K_WHITE  # noqa: E402
from solver.ns2d import SpectralGrid  # noqa: E402
from solver.validate import INK, SERIES, _style  # noqa: E402

FORBIDDEN = ("confirm3", "matched_energy_confirm", "confirm_v2")  # sealed or retired: never loaded here
STORED = ("loudness", "shape (loudness-null)", "PC2")
TM = "shape (transformation-matched)"


class ShapeFamily:
    """Entry 11's family on decoded vorticity, through per-shell sums (Parseval-weighted rfft2 modes)."""

    def __init__(self, g, grid_levels):
        shell = torch.round(g.kmag).long().clamp(max=K_WHITE + 1)  # buckets: 0 = mean, 1..21, 22 = beyond
        self.oh = F.one_hot(shell.flatten(), K_WHITE + 2).double()
        self.w = g.parseval_w.expand_as(g.kmag).flatten().double()
        self.ik2 = g.inv_k2.flatten().double()
        self.phi = torch.as_tensor(np.stack(basis()), dtype=torch.float64, device=self.oh.device)  # (2, 21)
        self.levels = grid_levels
        self.edge_hits = 0
        self.fits = 0

    def sums(self, y0, ya):
        x0 = torch.fft.rfft2(y0.double()[:, 0]).flatten(1)
        xa = torch.fft.rfft2(ya.double()[:, 0]).flatten(1)
        p0 = x0.real**2 + x0.imag**2
        band = slice(1, K_WHITE + 1)
        C = ((self.w * p0) @ self.oh)[:, band]
        E = ((self.w * p0 * self.ik2) @ self.oh)[:, band]
        A = ((self.w * (xa.real**2 + xa.imag**2)) @ self.oh)[:, band]
        B = ((self.w * (xa.real * x0.real + xa.imag * x0.imag)) @ self.oh)[:, band]
        dd = (self.w * ((xa - x0).real ** 2 + (xa - x0).imag ** 2)) @ self.oh
        return A, B, C, E, dd[:, 0] + dd[:, -1], dd.sum(1)

    def residual(self, t, A, B, C, E):
        """In-band residual for t of shape (S, m, 2) -> (S, m)."""
        G = torch.exp(t @ self.phi)
        c2 = E.sum(-1, keepdim=True) / (G**2 * E[:, None]).sum(-1)
        return (A[:, None] - 2 * torch.sqrt(c2)[..., None] * G * B[:, None] + c2[..., None] * G**2 * C[:, None]).sum(-1)

    @torch.no_grad()
    def fit(self, A, B, C, E, chunk=256):
        out = []
        for i in range(0, len(A), chunk):
            a, b, c, e = A[i:i + chunk], B[i:i + chunk], C[i:i + chunk], E[i:i + chunk]
            best = torch.zeros(len(a), 2, dtype=torch.float64, device=a.device)
            for li, (step, half) in enumerate(self.levels):
                o = torch.arange(-half, half + 1, dtype=torch.float64, device=a.device) * step
                grid = torch.stack(torch.meshgrid(o, o, indexing="ij"), -1).reshape(-1, 2)
                t = best[:, None] + grid[None]
                best = t[torch.arange(len(a)), self.residual(t, a, b, c, e).argmin(1)]
                if li == 0:
                    self.edge_hits += int((best.abs() >= half * step - 1e-9).any(1).sum())
            self.fits += len(a)
            out.append(best)
        return torch.cat(out)

    def terms(self, y0, ya):
        """(unexplained, change) for these frames; differentiable in ya (t fitted, then held fixed)."""
        A, B, C, E, out_band, change = self.sums(y0, ya)
        t = self.fit(A.detach(), B.detach(), C, E)
        return self.residual(t[:, None], A, B, C, E)[:, 0].sum() + out_band.sum(), change.sum()


def transform_objective(model, content, dial0, y0, d, cov, fam):
    d = d / d.norm()
    sigma = torch.sqrt(d @ cov @ d)
    num = den = 0.0
    for a in OPT_ALPHAS.to(d.device):
        n_, d_ = fam.terms(y0, model.head.dec(content, dial0 + a * sigma * d).float())
        num, den = num + n_, den + d_
    return num / den


def find_shape(model, enc_opt, cov, inits, cfg, fam):
    content, dial0, y0 = enc_opt
    gen = torch.Generator(device="cpu").manual_seed(cfg["seed"])
    results = []
    for name, init in inits.items():
        v = torch.as_tensor(init, dtype=torch.float32, device="cuda").clone().requires_grad_(True)
        opt = torch.optim.Adam([v], lr=cfg["opt_lr"])
        for _ in range(cfg["opt_steps"]):
            idx = torch.randperm(len(content), generator=gen)[:cfg["opt_batch"]].cuda()
            loss = transform_objective(model, content[idx], dial0[idx], y0[idx], v, cov, fam)
            opt.zero_grad()
            loss.backward()
            opt.step()
        d = (v / v.norm()).detach()
        results.append({"init": name, "direction": d.cpu().numpy(), "J_T": full_jt(model, enc_opt, d, cov, fam)})
    return results


@torch.no_grad()
def full_jt(model, enc_opt, d, cov, fam):
    content, dial0, y0 = enc_opt
    d = torch.as_tensor(d, dtype=torch.float32, device="cuda")
    d = d / d.norm()
    sigma = torch.sqrt(d @ cov @ d)
    num = den = 0.0
    for i in range(0, len(content), 64):
        for a in OPT_ALPHAS.cuda():
            n_, d_ = fam.terms(y0[i:i + 64], model.head.dec(content[i:i + 64], dial0[i:i + 64] + a * sigma * d).float())
            num, den = num + float(n_), den + float(d_)
    return num / den


@torch.no_grad()
def a1(model, G, x_opt, g, stored):
    """Seed-102 mechanism readouts (z-space of the loudness-null rule)."""
    mu, s = G.mean(0), G.std(0)
    content, dial0 = model.enc(x_opt), model.head.dial(x_opt)

    def st(dial):
        return fields.stats(fields.vorticity_hat(model.head.dec(content, dial).double() * model.ae_scales["vorticity"], "vorticity", g), g)

    ga, gs = [], []
    for j in range(G.shape[1]):
        e = torch.zeros(G.shape[1], device="cuda")
        e[j] = 0.5 * s[j]
        p, m = st(dial0 + e), st(dial0 - e)
        ga.append(((p["a"] - m["a"]) / 1.0).cpu().numpy())
        gs.append(((p["s"] - m["s"]) / 1.0).cpu().numpy())
    gm, hm = np.stack(ga, 1).mean(0), np.stack(gs, 1).mean(0)
    gh, hh = gm / np.linalg.norm(gm), hm / np.linalg.norm(hm)
    p2 = np.asarray(stored["PC2"]["direction"])
    v = (p2 / s) / np.linalg.norm(p2 / s)
    pv = v - (v @ gh) * gh
    sig2 = stored["PC2"]["sigma"]

    def ds(delta_raw):
        dl = torch.as_tensor(delta_raw, dtype=torch.float32, device="cuda")
        return float(np.median(((st(dial0 + dl)["s"] - st(dial0 - dl)["s"]) / 2).cpu().numpy()))

    delta = sig2 * p2  # raw displacement of alpha = 1 along PC2
    delta_p = s * ((delta / s) - ((delta / s) @ gh) * gh)  # the same z-displacement minus its g component
    ds_v, ds_pv = ds(delta), ds(delta_p)
    Z = (G - mu) / s
    Zp = Z - np.outer(Z @ gh, gh)
    cov_p = np.cov(Zp.T)
    lam, vec = np.linalg.eigh(cov_p)
    lam, vec = lam[::-1], vec[:, ::-1]
    u_st = np.asarray(stored["shape (loudness-null)"]["direction"]) / s
    u_st /= np.linalg.norm(u_st)
    pvh = pv / np.linalg.norm(pv)
    loud_z = np.asarray(stored["loudness"]["direction"]) / s
    loud_z /= np.linalg.norm(loud_z)
    return {"g_recomputed_vs_stored_cos": float(gh @ (np.asarray(stored["g_z"]) / np.linalg.norm(stored["g_z"]))),
            "cos2_g_PC2": float((gh @ v) ** 2), "cos2_g_h": float((gh @ hh) ** 2), "cos2_g_loudness": float((gh @ loud_z) ** 2),
            "cos2_h_PC2": float((hh @ v) ** 2),
            "removed_linear": float((hm @ gh) * (gh @ v) / (hm @ v)),
            "ds_PC2": ds_v, "ds_projected_PC2": ds_pv, "removed_steered": float(1 - ds_pv / ds_v),
            "var_top": float(lam[0]), "var_second": float(lam[1]), "var_along_projected_PC2": float(pvh @ cov_p @ pvh),
            "projected_PC2_rank_by_variance_overlap": [float((pvh @ vec[:, i]) ** 2) for i in range(len(lam))],
            "cos2_top_vs_stored_choice": float((vec[:, 0] @ u_st) ** 2),
            "raw_std_per_coordinate": s.tolist()}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--check", action="store_true", help="code check: one model, 2 restarts x 3 steps, 32 frames; no outputs")
    args = ap.parse_args()
    cfg = load_config(args.config)
    for path, splits in cfg["data_parts"]:
        if any(f in path for f in FORBIDDEN) or ("data/generated/v1" in path and set(splits) - {"train", "val"}):
            raise SystemExit(f"{path} {splits}: sealed or retired data is not loaded by the autopsy")
    if args.check:
        cfg = {**cfg, "opt_steps": 3, "n_frames": 32, "groups": {"main": [cfg["groups"]["main"][2]]}}
    seed_everything(cfg["seed"])
    t0 = time.perf_counter()
    stage5 = json.load(gzip.open(REPO / cfg["stage5_json"], "rt", encoding="utf-8"))["results"]
    stored = {e["run"]: {**e["directions"], "g_z": e["loudness_gradient"]["g_z"], "_entry": e} for rs in stage5.values() for e in rs}
    ds = CombinedDataset(cfg["data_parts"], cfg["cache_tag"])
    g = SpectralGrid(64, "cuda")
    to_gpu = lambda rows: torch.from_numpy(ds.velocity[rows]).cuda() / ds.input_scale  # noqa: E731
    tr = ds.rows(["train"])
    env = envelope(to_gpu(np.sort(np.random.default_rng(cfg["seed"]).choice(tr, cfg["n_envelope"], replace=False))), g)
    excl = frames(ds, "explore", 256, 4, cfg["seed"])  # the build's steering frames
    x_opt = to_gpu(np.sort(np.random.default_rng(cfg["opt_seed"]).choice(np.setdiff1d(ds.rows(["explore"]), excl),
                                                                          cfg["n_opt_frames"], replace=False)))
    sets = {name: frames(ds, split, cfg["n_frames"], 4, cfg["seed"]) for name, split in cfg["score_sets"].items()}
    xs = {name: (to_gpu(r - 1), to_gpu(r)) for name, r in sets.items()}
    crit = {**cfg["criteria"], **cfg["artefact"]}
    fam = ShapeFamily(g, cfg["fit_grid"])

    def score(model, d, sigma, zero_mean=True):
        e = {"direction": np.asarray(d).tolist(), "sigma": float(sigma)}
        for s_, (xp, xt) in xs.items():
            ad = ZeroMean(adapter_for(model, "dial")) if zero_mean else adapter_for(model, "dial")
            m = steer(ad, xp, xt, d, sigma, g, env)
            m.pop("heads", None)
            if zero_mean:
                m["pre_removal_mean_gt10_by_alpha"] = ad.pre
            e[s_] = {**m, **classify(m, crit), "artefact_fraction": artefact_fraction(m, crit)}
        for k_ in ("dial", "loudness", "shape"):
            e[k_] = bool(all(e[s_][k_] for s_ in xs))
        return e

    res = {}
    for group, runs in cfg["groups"].items():
        res[group] = []
        for run in runs:
            model = load(run)
            st = stored[run]
            G = dial_latents(model, ds, tr)
            pcs, sig, _ = pca_directions(G, n=8)
            cov = torch.as_tensor(np.cov(G.T), dtype=torch.float32, device="cuda")
            with torch.no_grad():
                c_o, d_o = model.enc(x_opt), model.head.dial(x_opt)
                enc_opt = (c_o, d_o, model.head.dec(c_o, d_o).float())
            d_n_now = loudness_null(model, G, x_opt, g)[0]
            entry = {"run": run, "group": group,
                     "check_loudness_null_recomputed_vs_stored_cos": float(abs(d_n_now @ np.asarray(st["shape (loudness-null)"]["direction"]))),
                     "check_PC2_recomputed_vs_stored_cos": float(abs(pcs[1] @ np.asarray(st["PC2"]["direction"])))}
            if group == "main":
                entry["A1"] = a1(model, G, x_opt, g, st)
            rng = np.random.default_rng(cfg["seed"])
            inits = {**{f"PC{i + 1}": pcs[i] for i in range(8)}, **{f"random{j}": rng.standard_normal(8) for j in range(cfg["n_random_inits"])}}
            if args.check:
                inits = dict(list(inits.items())[:2])
            restarts = find_shape(model, enc_opt, cov, inits, cfg, fam)
            best = min(restarts, key=lambda r_: r_["J_T"])
            d_t = best["direction"].astype(np.float64)
            d_t /= np.linalg.norm(d_t)
            entry["J_T"] = {"best": best["J_T"], "init_of_best": best["init"], "restarts": {r_["init"]: r_["J_T"] for r_ in restarts},
                            **{f"PC{i + 1}": full_jt(model, enc_opt, pcs[i], cov, fam) for i in range(8)},
                            **{n: full_jt(model, enc_opt, np.asarray(st[n]["direction"]), cov, fam) for n in STORED}}
            entry["cos_TM_vs"] = {**{n: float(abs(d_t @ np.asarray(st[n]["direction"]))) for n in STORED},
                                  **{f"PC{i + 1}": float(abs(d_t @ pcs[i])) for i in range(8)}}
            entry["directions"] = {TM: score(model, d_t, float((G @ d_t).std()))}
            if group == "main":
                for n in STORED:
                    entry["directions"][n] = score(model, np.asarray(st[n]["direction"]), st[n]["sigma"])
            if run == cfg["a4_run"]:
                entry["A4_zero_mean_off"] = {n: score(model, np.asarray(st[n]["direction"]), st[n]["sigma"], zero_mean=False)
                                             for n in ("loudness", "shape (loudness-null)")}
            entry["_strip"] = strip(model, d_t, entry["directions"][TM]["sigma"], xs["explore"][1][:1], g)
            res[group].append(entry)
            msg = ", ".join(f"{n} {label(e)}" for n, e in entry["directions"].items())
            print(f"{group} {run}: J_T {best['J_T']:.3f} (from {best['init']}); {msg} ({time.perf_counter() - t0:.0f} s)", flush=True)
            del model
            torch.cuda.empty_cache()
    if args.check:
        print(f"check ok; grid-edge fits {fam.edge_hits} of {fam.fits}")
        return
    wall = time.perf_counter() - t0
    write(cfg, res, stage5, fam, wall)


@torch.no_grad()
def strip(model, d, sigma, x1, g):
    ad = ZeroMean(adapter_for(model, "dial"))
    dt = torch.as_tensor(d, dtype=torch.float32, device="cuda")
    fs = [ad.decode(None, x1, float(a * sigma) * dt)[0][2] for a in (-2.0, -1.0, 0.0, 1.0, 2.0)]
    return {"fields": [f[0, 0].cpu().numpy() for f in fs],
            "spectra": [fields.shell_spectrum(fields.vorticity_hat(f, "vorticity", g), g)[0].cpu().numpy() for f in fs]}


def a2(stage5) -> dict:
    out = {}
    for group in ("tau0", "untrained"):
        rows = stage5[group]
        out[group] = {"PC2": sum(e["directions"]["PC2"]["shape"] for e in rows),
                      "loudness-null": sum(e["directions"]["shape (loudness-null)"]["shape"] for e in rows),
                      "union": sum(e["directions"]["PC2"]["shape"] or e["directions"]["shape (loudness-null)"]["shape"] for e in rows),
                      "of": len(rows)}
    out["main"] = {"union_both_dials": sum(e["loudness_dial"] and (e["directions"]["PC2"]["shape"] or e["shape_dial"]) for e in stage5["main"]),
                   "of": len(stage5["main"])}
    return out


def write(cfg, res, stage5, fam, wall):
    sets = list(cfg["score_sets"])
    f2 = lambda v: f"{v:.2f}"  # noqa: E731
    yn = lambda b: "yes" if b else "no"  # noqa: E731
    A2 = a2(stage5)
    L = ["# [BUILD] Autopsy of the v3 Stage 5 run (exploratory, post hoc; PREDICTIONS entry 11)", "",
         "Nothing here changes REPORT_v3's verdict. No training; `confirm3` and `matched-energy-confirm` were not opened. Directions and "
         "sealed-set scores come from `results/build/v3/confirm.json.gz`; every new measurement uses exploration data (v1 train, "
         "the build's 256 explore optimization frames, and 256 frames each of `explore` and the exploration matched-energy set), "
         "mean removed before every scoring measurement, criteria as in v3.", ""]
    L += ["## A1: why loudness-null picked a weak direction (z-scored dial space of the rule)", "",
          "| seed | cos²(g, PC2) | cos²(g, h) | cos²(h, PC2) | PC2 shape effect removed by projecting out g: linear / steered | "
          "variance of projected z-data: top PC / 2nd / along projected PC2 | loudness-null σ_d / PC2 σ_d | check: g recomputed vs stored (cos) |",
          "|---|---|---|---|---|---|---|---|"]
    for e in res["main"]:
        a = e["A1"]
        sd = {n: e["directions"][n]["sigma"] for n in STORED}
        L.append(f"| {e['run'][-3:]} | {a['cos2_g_PC2']:.3f} | {a['cos2_g_h']:.3f} | {a['cos2_h_PC2']:.3f} | "
                 f"{a['removed_linear']:+.2f} / {a['removed_steered']:+.2f} | {a['var_top']:.2f} / {a['var_second']:.2f} / "
                 f"{a['var_along_projected_PC2']:.2f} | {sd['shape (loudness-null)']:.2f} / {sd['PC2']:.2f} | {a['g_recomputed_vs_stored_cos']:.4f} |")
    L += ["", "g: frame-mean gradient of a in z-units; h: the same for s; PC2 mapped to z-space. Steered: median Δs at α = ±1 σ_PC2 "
          "along PC2 and along its projection onto g⊥ (same z-displacement minus its g component), optimization frames.", "",
          "## A2: is the union lenient? (stored sealed-set scores)", "",
          "| control | PC2 shape dial | loudness-null shape dial | union (either) |", "|---|---|---|---|"]
    for grp in ("tau0", "untrained"):
        a = A2[grp]
        L.append(f"| {grp} | {a['PC2']} of {a['of']} | {a['loudness-null']} of {a['of']} | {a['union']} of {a['of']} |")
    L += ["", f"Main model, both dials with the union as the shape rule (sealed sets; post hoc): {A2['main']['union_both_dials']} of "
          f"{A2['main']['of']}.", "",
          "## A3: transformation-matched shape rule (exploration sets)", "",
          "| group | run | J_T: TM / loudness-null / PC2 / loudness / PC1 | TM found from | |cos| TM vs loudness-null / PC2 / loudness | "
          "TM shape dial | TM σ_d | stored loudness dial (expl.) | both dials (TM) | loudness-null shape dial (expl.) | PC2 shape dial (expl.) |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    counts = {"main": 0, "tau0": 0, "untrained": 0}
    both_tm = 0
    for grp, rs in res.items():
        for e in rs:
            j, c, dirs = e["J_T"], e["cos_TM_vs"], e["directions"]
            tm = dirs[TM]
            counts[grp] += tm["shape"]
            stored_cells = ["—"] * 3
            if grp == "main":
                both = dirs["loudness"]["loudness"] and tm["shape"]
                both_tm += both
                stored_cells = [yn(dirs["loudness"]["loudness"]), f"**{yn(both)}**", yn(dirs["shape (loudness-null)"]["shape"])]
                pc2_cell = yn(dirs["PC2"]["shape"])
            else:
                pc2_cell = "—"
            L.append(f"| {grp} | {e['run']} | {j['best']:.3f} / {j['shape (loudness-null)']:.3f} / {j['PC2']:.3f} / {j['loudness']:.3f} "
                     f"/ {j['PC1']:.3f} | {j['init_of_best']} | {c['shape (loudness-null)']:.2f} / {c['PC2']:.2f} / {c['loudness']:.2f} | "
                     f"**{yn(tm['shape'])}** | {tm['sigma']:.3g} | " + " | ".join(stored_cells) + f" | {pc2_cell} |")
    L += ["", f"TM shape dial: main {counts['main']} of {len(res['main'])}, τ = 0 {counts['tau0']} of {len(res['tau0'])}, "
          f"untrained {counts['untrained']} of {len(res['untrained'])}. Both dials with TM (stored loudness direction, exploration sets): "
          f"{both_tm} of {len(res['main'])}.", "",
          f"Family fits at the edge of the first grid level (|t| = 8): {fam.edge_hits} of {fam.fits}.", ""]
    for s_ in sets:
        L += [f"### Every scored direction, every metric — {s_} (mean removed)", "",
              "| group | run | direction | σ_d | Δa | Δs | effect | σ_shape | monot. | cycle corr / gain | env − ideal | shared | artefact frames | "
              "pre-removal mean > 10% (max over α) | D L S | fails |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for grp, rs in res.items():
            for e in rs:
                for n, d in e["directions"].items():
                    m = d[s_]
                    L.append(f"| {grp} | {e['run']} | {n} | {d['sigma']:.3g} | {m['d_a']:+.3f} | {m['d_s']:+.3f} | {m['effect']:.3f} | "
                             f"{f2(m['sigma_shape'])} | {f2(m['monotonicity'])} | {f2(m['cycle_corr'])} / {f2(m['cycle_gain'])} | "
                             f"{m['envelope_rel_min']:+.2f} | {f2(m['shared_pattern_fraction'])} | {m['artefact_fraction']:.1%} | "
                             f"{max(m['pre_removal_mean_gt10_by_alpha']):.1%} | {label(m)} | {', '.join(m['fails']) or '—'} |")
        L.append("")
    a4 = next(e for e in res["main"] if e["run"] == cfg["a4_run"])
    L += [f"## A4: {cfg['a4_run']} with the zero-mean step off (exploration sets; confirm3 not reopened)", "",
          "| direction | set | mean removed: D L S (fails) | mean kept: D L S (fails) | frames with > 10% of ω² in the mean, max over α (raw) | "
          "Δs removed / kept | env − ideal removed / kept |", "|---|---|---|---|---|---|---|"]
    for n in ("loudness", "shape (loudness-null)"):
        for s_ in sets:
            on, off = a4["directions"][n][s_], a4["A4_zero_mean_off"][n][s_]
            L.append(f"| {n} | {s_} | {label(on)} ({', '.join(on['fails']) or '—'}) | {label(off)} ({', '.join(off['fails']) or '—'}) | "
                     f"{max(on['pre_removal_mean_gt10_by_alpha']):.1%} | {on['d_s']:+.3f} / {off['d_s']:+.3f} | "
                     f"{on['envelope_rel_min']:+.2f} / {off['envelope_rel_min']:+.2f} |")
    L += ["", "Checks: |cos| between recomputed and stored loudness-null / PC2 directions: "
          + ", ".join(f"{e['run'][-3:]} {e['check_loudness_null_recomputed_vs_stored_cos']:.4f} / {e['check_PC2_recomputed_vs_stored_cos']:.4f}"
                      for rs in res.values() for e in rs) + ".", "", f"Wall-clock {wall / 60:.0f} min."]
    out = REPO / cfg["output_md"]
    out.parent.mkdir(parents=True, exist_ok=True)
    strips = {e["run"]: e.pop("_strip") for rs in res.values() for e in rs}
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    with gzip.open(REPO / cfg["output_json_gz"], "wt", encoding="utf-8") as f:
        json.dump({"environment": environment(), "config": cfg, "results": res, "A2": A2, "wall_seconds": wall,
                   "family_edge_hits": [fam.edge_hits, fam.fits]}, f, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))
    _style()
    runs = [e for e in res["main"]]
    fig, axes = plt.subplots(len(runs), 6, figsize=(16, 2.6 * len(runs)), constrained_layout=True, gridspec_kw={"width_ratios": [1] * 5 + [1.5]})
    for row, e in zip(axes, runs):
        st = strips[e["run"]]
        lim = float(np.abs(st["fields"][2]).max())
        for ax, fld, a in zip(row[:5], st["fields"], (-2, -1, 0, 1, 2)):
            ax.imshow(fld, cmap="RdBu_r", vmin=-lim, vmax=lim)
            ax.set_title((f"{e['run']}, TM shape: " if a == -2 else "") + f"α = {a:+d}", fontsize=7)
            ax.set_xticks([])
            ax.set_yticks([])
        k = np.arange(1, fields.K_MAX + 1)
        for spec, a, col in zip(st["spectra"], (-2, -1, 0, 1, 2), (SERIES[0], SERIES[0], INK, SERIES[1], SERIES[1])):
            row[5].loglog(k, spec, color=col, alpha=1.0 if a in (-2, 0, 2) else 0.45)
    fig.suptitle("[BUILD] Autopsy (exploratory): transformation-matched shape direction, one explore frame (mean removed)",
                 fontweight="bold", color=INK)
    fig.savefig(REPO / cfg["figure"], dpi=90)
    plt.close(fig)
    append_log("[BUILD] Autopsy of v3 Stage 5 (exploratory; no training; sealed sets not opened)", cfg["_config_path"], command_line(), "ok",
               {"TM shape dials (main / tau0 / untrained)": f"{counts['main']} / {counts['tau0']} / {counts['untrained']}",
                "both dials with TM (exploration sets)": f"{both_tm} of {len(res['main'])}",
                "results": f"{cfg['output_md']}, {cfg['output_json_gz']}, {cfg['figure']}"}, wall, environment().get("gpu"))
    print("\n".join(L[:40]))


if __name__ == "__main__":
    main()
