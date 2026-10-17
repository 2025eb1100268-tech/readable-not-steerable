# Report v2: confirmation of the exploratory line

2026-10-17. Pre-registration: `PREREGISTRATION_v2.md` (committed `8797966` before the sealed set
was opened). Single confirmation run: `results/explore/stage3/confirm.md` / `.json` (17:33–17:48, run
once, no rerun). Exploration that led here: `results/explore/STAGE2_SUMMARY.md`.

Wording rule: these statistics say which conserved quantities are **prominent in a latent** (its
slowest unsupervised directions). They are not claims that a model discovered conservation, nor
measures of what a model knows.

## Bottom line

- **Verdict: partially supported.** H1 holds; H2 fails, on one of its two pairs.
- **H1, rendering, is confirmed strongly.** On data no model had seen and with freshly trained seeds,
  plain autoencoders that render the current frame make enstrophy prominent in their slow directions:
  E1 **0.86** (velocity) and **0.80** (vorticity), against **0.38** for the untrained control (gaps
  +0.48 and +0.42, CIs far from 0).
- **H2, regularization, holds in direction but not in size.** Adding VICReg lowered E1 in every
  comparison on the confirm set, but for the fresh vorticity autoencoders the drop was **+0.149**
  [+0.105, +0.193], below the pre-registered 0.20. The velocity pair passed (+0.338).
- **H4 (secondary) passed:** the pixel predictor with VICReg drops from 0.815 to 0.500 (+0.315
  [+0.272, +0.358]). That closes about two-thirds of the imaginer-vs-summarizer gap (0.483), so
  regularization explains much, but not all, of the contrast this exploration started from.
- **The fresh seeds mattered.** The Stage 2 vorticity seeds would have passed H2 on the same data
  (+0.262). The shortfall comes from training-run variability, which is exactly what fresh seeds were
  added to expose.

## Primary hypotheses

| | rule | result | outcome |
|---|---|---|---|
| H1, velocity AE (fresh seeds 3–5) | E1 ≥ 0.70; ≥ control + 0.25; CI > 0 | E1 0.860; +0.480 [+0.390, +0.565] | pass |
| H1, vorticity AE (fresh) | same | E1 0.799; +0.418 [+0.360, +0.486] | pass |
| **H1** | both pass | | **holds** |
| H2, velocity: plain − VICReg (fresh) | ≥ 0.20; CI > 0 | +0.338 [+0.289, +0.390] | pass |
| H2, vorticity: plain − VICReg (fresh) | same | **+0.149** [+0.105, +0.193] | **fail** |
| **H2** | both pass | | **fails** |

Per seed (E1): vorticity AE 0.709, 0.837, 0.850; vorticity AE + VICReg 0.621, 0.649, 0.679; velocity
AE 0.894, 0.856, 0.830; velocity AE + VICReg 0.568, 0.501, 0.497. The vorticity shortfall comes from
a weaker plain seed 3 (0.709) and VICReg seeds that kept more enstrophy prominence (0.62–0.68) than
the Stage 2 VICReg seeds did on the same data (0.551 mean).

## Secondary hypotheses

| | rule | result | outcome |
|---|---|---|---|
| H3, render target (Stage 2 seeds) | vorticity AE's E4 a/b closer to 1 than streamfunction AE's; streamfunction > 2 | 1.00 vs 3.13 | pass |
| H4, imaginer k = 4 − imaginer k = 4 + VICReg | ≥ 0.20; CI > 0 | +0.315 [+0.272, +0.358] | pass |

## Everything else (reported, no rule)

| group (confirm set) | seeds | E1 [95% CI] | E4 a/b | log Z R² | log P R² |
|---|---|---|---|---|---|
| untrained, seeds 3–5 | 3 | 0.380 [0.30, 0.46] | ill-defined (see note) | 0.738 | 0.410 |
| velocity AE (fresh) | 3 | 0.860 [0.84, 0.88] | 0.83 | 0.862 | 0.700 |
| velocity AE + VICReg (fresh) | 3 | 0.522 [0.47, 0.57] | 0.90 | 0.933 | 0.799 |
| vorticity AE (fresh) | 3 | 0.799 [0.78, 0.82] | 1.38 | 0.806 | 0.694 |
| vorticity AE + VICReg (fresh) | 3 | 0.650 [0.61, 0.69] | 1.07 | 0.955 | 0.912 |
| untrained, seeds 0–2 | 3 | 0.376 [0.29, 0.45] | 13.98 | 0.737 | 0.407 |
| velocity AE (Stage 2 seeds) | 3 | 0.813 [0.79, 0.83] | 0.81 | 0.855 | 0.698 |
| velocity AE + VICReg (Stage 2) | 3 | 0.468 [0.41, 0.53] | 1.19 | 0.934 | 0.804 |
| vorticity AE (Stage 2) | 3 | 0.813 [0.80, 0.83] | 1.00 | 0.799 | 0.698 |
| vorticity AE + VICReg (Stage 2) | 3 | 0.551 [0.50, 0.61] | 1.19 | 0.959 | 0.918 |
| streamfunction AE (Stage 2) | 3 | 0.558 [0.52, 0.60] | 3.13 | 0.861 | 0.657 |
| streamfunction AE + VICReg (Stage 2) | 3 | 0.390 [0.34, 0.45] | 1.13 | 0.917 | 0.719 |
| imaginer k = 4 | 3 | 0.815 [0.79, 0.84] | 0.87 | 0.941 | 0.830 |
| imaginer k = 4 + VICReg (new) | 3 | 0.500 [0.45, 0.55] | 0.71 | 0.937 | 0.782 |
| summarizer k = 4 | 3 | 0.332 [0.29, 0.40] | 0.88 | 0.907 | 0.725 |

- **The H1/H2 rules on the Stage 2 seeds** (same confirm data): all four pass (velocity +0.437 and
  vorticity +0.437 over control; VICReg drops +0.345 and +0.262).
- **Imaginer − summarizer** at k = 4: +0.483 [+0.42, +0.52], the Stage 2 contrast, replicated.
- **Prominence and decodability dissociate again.** Every VICReg model has *higher* linear-probe R²
  for log P (and log Z) than its plain counterpart while its E1 is lower: VICReg keeps the
  information but spreads it so it no longer dominates the slow directions.
- **Note on E4.** The ratio a/b is ill-defined when b ≈ 0. For untrained seed 5, b is 0 to four
  decimals and a/b is −20,653, so the seeds 3–5 mean (−6,878) is meaningless; the untrained control's
  direction is an energy direction (seeds 3 and 4: 11.3 and 6.8). E4 is also seed-sensitive for the
  fresh vorticity AE (1.38 vs 1.00 for the Stage 2 seeds), which is why H3 was secondary.

## What this means

1. **Confirmed:** models trained to *render* a field, without a whitening regularizer, make
   enstrophy (together with energy) prominent in their slowest latent directions, whether they render
   the present or the future. This replicates on unseen trajectories and new training runs, with a
   large margin.
2. **Partially confirmed:** the VICReg regularizer reduces that prominence in every comparison (5 of
   5 drops positive with CIs above 0), but the size depends on the render target and on the training
   seeds; the pre-registered claim that it drops by at least 0.20 for both targets fails for vorticity.
   Rendering quality also falls under VICReg (about 20–25× worse losses), so part of each drop may come
   from worse rendering rather than latent geometry.
3. **For the original question** (does a JEPA organize its latent around fluid quantities?): the
   JEPA's poor showing in study 1 is consistent with its regularizer. Adding the same regularizer to
   the pixel predictor removes about two-thirds of its advantage over the JEPA (H4, secondary).
   Prediction of dynamics was never required for the effect.

## Threats and limits

- Same simulator and initial-condition distribution as the exploration data: this confirms
  reproducibility on fresh trajectories and fresh training runs, not generality to other flows.
- Three seeds per model; the bootstrap CIs resample trajectories, not seeds. H2's failure shows that
  seed variation matters at this effect size.
- One small CNN encoder, one unsupervised method (z-score → PCA 32 → Ledoit–Wolf within/between
  eigenproblem); the result is about prominence under that method.
- VICReg's rendering cost is a confound for H2 and H4 that this design cannot remove.
- The secondary results (H3, H4) are reported with their rules but do not change the verdict.
- 2D flow only (no vortex stretching).

## Process record

- Confirm set generated 2026-10-15, sealed in code, unsealed once by the Stage 3 script after
  `PREREGISTRATION_v2.md` was committed (logged 17:33). The log entry shows `8797966+dirty` because
  the unsealing itself appended to `results/LOG.md` during the run; the tree was clean when it started.
- All 39 checkpoint SHA-256 values and the confirm checksums were verified before evaluation.
- The 15 fresh models were never evaluated before the confirmation run; the script was checked with a
  stand-in config of old models on `explore`.
- None of the 15 fresh runs diverged, so no exclusions.

## Compute

Exploratory line and Stage 3 together: **29.6 GPU-hours** logged (24.5 exploration, 5.1 Stage 3),
within the agreed 30. RTX 3060.
