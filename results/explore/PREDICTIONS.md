# [EXPLORE] Predictions (append-only)

Each entry is written and committed **before** the stage it covers runs, and never edited
afterwards (git history is the check). Everything here is exploratory (`results/explore/BRIEF.md`).
Wording rule: claims are about which conserved quantities are **prominent in a latent**, never
that a model "discovered conservation".

---

## Entry 1 — 2026-10-15 — Stages 0 and 1

Written before the new datasets exist and before any Stage 0/1 computation.

### Fixed in advance

- **Models** (existing checkpoints, 3 seeds each): `untrained_s{0,1,2}`, `jepa_spatial_lr0.001_s*`,
  `pixel_lr0.001_s*`, `jepa_pooled_lr0.001_s*`, `static_vicreg_lr0.001_s*`,
  `nonphys_jepa_spatial_lr0.001_s*`.
- **Fitting vs scoring.** Probes are fitted on the original v1 train split (ridge α chosen on v1 val,
  time × family partialling exactly as pre-registered). Slow directions are fitted on v1 train+val
  inviscid runs (z-score → PCA 32 → Ledoit–Wolf generalized eigenproblem, as pre-registered).
  Everything is scored on `explore` (all splits there are evaluation) and, for slow directions, on
  `matched-energy`.
- **Readouts (Stage 0):** pooled latent; ridge on PCA-512 of the flattened 8×8×128 map (PCA fitted
  on v1 train frames); per-channel spatial mean ⊕ spatial std of the map (384 dims).
- **"Enstrophy |r|":** per seed, the maximum over the top-3 slow directions of |Pearson r| between
  the direction's trajectory-mean projection and log Z₀; then the mean over seeds. Every direction
  is also reported separately. CIs: 1,000 paired trajectory-bootstrap resamples (as in
  `probes/criteria.py`).
- **Partial r (unscored):** same construction, with log Z₀ and the projection both residualized on
  [log E₀, family indicator] before correlating.
- **Specificity (reported next to every slow-direction result):** |r| of the same directions with
  trajectory-mean log palinstrophy (not conserved), trajectory-mean log kurtosis and trajectory-mean
  log max|ω|.
- **1c descriptors** (t = 0, per trajectory): log E₀, log(Z₀/E₀), high-k spectral slope (least-squares
  slope of log E(k) on log k for k ∈ [8, 20]), log(P₀/Z₀), log kurtosis₀. Regression of the pixel
  model's enstrophy-tracking direction (per seed, the top-3 direction with the largest |r| with
  log Z₀) on the standardized descriptors.

### Kill gate (final, as agreed; decides whether Stage 2 starts)

The lead survives only if **both** hold:

- **(a)** On explore inviscid runs, the pixel model beats the untrained control on enstrophy |r|
  (best of top-3 slow directions, mean over seeds) by **≥ 0.1**, with the trajectory-bootstrap 95%
  CI on the gap excluding zero.
- **(b)** On matched-energy, the pixel model's best-of-top-3 |r| with log Z₀ is **≥ 0.5** and
  exceeds the untrained control's, with the CI on the gap excluding zero.

The partial r on explore (controlling for log E₀ and family) is reported alongside, unscored. If
either (a) or (b) fails: stop, report, do not start Stage 2.

### Predictions, Stage 0

- **P0.1** Pooled readout on explore reproduces the test-ID ordering on the S1 structure targets:
  static VICReg > non-physical JEPA > JEPA > {pixel, pooled JEPA} > untrained (log Z, log P), with
  values within ±0.07 of test-ID. Reason: explore uses the same recipe; only the inviscid share
  differs.
- **P0.2** The map PCA-512 readout is within ±0.03 of the pooled readout for every model on log Z
  and log P. Reason: the maps are close to shift-equivariant, so the best linear readout of a
  shift-invariant target is close to spatial averaging.
- **P0.3** The map mean ⊕ std readout raises every model's S1 R², the untrained control's included.
  The JEPA − control gap shrinks on at least 3 of the 5 S1 targets, and the JEPA stays below static
  VICReg and the non-physical JEPA on log Z, log P and vortex count. Reason: spatial std of random
  conv features already measures local gradient energy, so the control gains as much as anyone.
- **P0.4** No readout lifts the JEPA's ν frame-pair R² above the control's by more than 0.1.
- **P0.5** Under every readout, the JEPA's enstrophy |r| stays below the pixel model's.

Consequence I expect: the pooled readout did not hide a JEPA advantage, so the imaginer/summarizer
contrast is not a readout artefact.

### Predictions, Stage 1

- **P1.1 (gate a)** Pixel enstrophy |r| on explore in 0.70–0.88; untrained control in 0.35–0.60;
  gap ≥ 0.1 with CI > 0. Probability gate (a) passes: about 70%.
- **P1.2** Partial r (log E₀, family) on explore: pixel ≥ 0.5, control < 0.4. Reason: across the
  original inviscid runs corr(log E₀, log Z₀) is only 0.33 (Z₀ is set directly by the generator),
  and on val the pixel model's enstrophy direction had |r| 0.22 with energy.
- **P1.3 (gate b)** Matched-energy: pixel best-of-top-3 |r| with log Z₀ in 0.45–0.75 and above the
  control's. Probability gate (b) passes: about 50%. Reason for the doubt: at fixed energy,
  enstrophy varies only through spectral shape, and the matched-energy fields use slopes
  (p ∈ [3, 5]) the models never saw (training used p = 4).
- **P1.4 (1c)** In the regression, log E₀ and log(Z₀/E₀) together carry most of the explained
  variance of the pixel model's enstrophy-tracking direction; log(P₀/Z₀) and kurtosis carry little.
  Plain language I expect to write: "the direction measures how much small-scale structure the
  field has for its energy, i.e. its gradient amplitude".
- **P1.5 (specificity)** For that direction, |r| with trajectory-mean log palinstrophy is at least
  0.1 lower than |r| with log Z₀.
- **Overall:** probability that both gate conditions pass and Stage 2 starts: about 45%.

---

## Entry 2 — 2026-10-15 — before Stage 0

Written after the datasets were generated (generation statistics only) and before any model was
run on them.

- **Correction to entry 1.** The map mean ⊕ std readout is **256** dims (per-channel spatial mean
  and spatial std of 128 channels), not 384; that was an arithmetic slip. The definition is
  unchanged.
- **Data fact (from generation, no model outputs).** The matched-energy set thermalizes more than
  the others: top-shell enstrophy share at the last frame has median 1.5% and exceeds 5% in 16.7%
  of runs (v1: 0.9% / 6.2%; explore: 0.8% / 6.1%). E and Z remain exact invariants of the truncated
  system, so log Z₀ (the gate's target) is unaffected, but trajectory means of the slow-direction
  projections include some unrealistic late frames.
- **Added unscored robustness check for gate (b).** The gate (b) statistics recomputed with
  trajectory means over the first 16 frames only (t ≤ 0.60). Gate (b) as written (all 24 frames)
  still decides.
- **Prediction.** For both the pixel model and the control, the 16-frame value differs from the
  24-frame value by less than 0.05, and the robustness check agrees with gate (b)'s outcome.

---

## Entry 3 — 2026-10-15 — Stage 2 (before any Stage 2 code or training)

Written after Stage 1 (gate passed) and before the Stage 2 training code exists.

### Fixed in advance: training

- **Data and anchors.** All Stage 2 models except the bridge use one common anchor set: v1
  trajectories with 64-frame windows (the viscous runs), input frames t ∈ [1, 31], target frame
  t + k. Inputs are identical across horizons; only the target moves. The fake-physics imaginer uses
  the same rule on `nonphys_v1` (its 64-frame runs). The **bridge** uses the original rule for a
  single k = 4: every trajectory, t ∈ [1, n_frames − 5].
- **Models.**
  - Imaginer: pixel predictor (same encoder, decoder, loss), k ∈ {1, 4, 16, 32}, seeds 0–2.
  - Summarizer: spatial JEPA (same setup), single horizon, k ∈ {1, 4, 16, 32}, seeds 0–2.
  - Bridge: imaginer at k = 4 on the original anchor rule, seeds 0–2.
  - Fake-physics imaginer: imaginer trained on `nonphys_v1` at k*, seeds 0–2.
  - Pixel + VICReg: imaginer loss 25·MSE + 25·variance hinge + 1·covariance on the online pooled
    latent (the JEPA's regularizer and weights), at k*, seeds 0–2.
- **k\*** (logged before those runs): the horizon maximizing the imaginer's seed-mean explore
  enstrophy |r| (E1 below); if the top two are within 0.02, the one with the larger E2.
- **Learning rate.** Peak lr ∈ {1e-3, 3e-3}, chosen once per family (imaginer, summarizer) at k = 4,
  seed 0, by the final label-free val metric (val loss / copy baseline at k), collapsed runs excluded
  (amendment A1 rule). The imaginer's lr is used for the bridge, fake-physics and pixel + VICReg runs.
- **Schedule (identical for every Stage 2 model).** Batch 256, AdamW (weight decay 0.05), bf16
  autocast. Linear warmup over 575 steps, then constant lr. Val metric every 230 steps on v1 val
  (same anchor rule and k). Improvement = metric below best × (1 − 0.005). After 4 consecutive evals
  without improvement: lr × 0.3 if fewer than 2 reductions so far, otherwise stop. Hard cap 12,000
  steps. The checkpoint is the model at stop. JEPA EMA momentum m = 1 − 0.004 · (current plateau lr /
  peak lr), with m = 0.996 during warmup.
- **Diagnostics logged per run.** Val loss / copy baseline at k; steps and stop reason; for pixel
  models, the energy spectrum of predicted, true and copied frames at t + k on 1,024 fixed val
  anchors; for JEPA models, the variance of the predicted map relative to the target map.

### Fixed in advance: evaluation (explore and matched-energy; pooled latent primary, map mean ⊕ std secondary)

- **E1 (primary).** Explore enstrophy |r|: best of top-3 slow directions with log Z₀, seed mean (slow
  directions fitted on v1 train+val inviscid runs as pre-registered). CIs: paired trajectory
  bootstrap vs the untrained control, and imaginer − summarizer at each k (`probes/criteria.py`).
- **E2 (primary companion).** Explore partial r with log Z₀ given log E₀ and family, best of top 3.
- **E3 (specificity, added after Stage 1).** For each model's enstrophy-tracking direction (top-3
  direction with the largest |r| with log Z₀): partial r with log Z₀ given trajectory-mean log P, and
  partial r with trajectory-mean log P given log Z₀ (explore; on matched-energy the two correlate at
  0.996 in the data, so it cannot separate them).
- **E4 (the enstrophy combination, added after Stage 1).** Regression of that direction on the
  unstandardized log E₀ and log(Z₀/E₀) (explore): coefficient ratio a/b and R². a/b = 1 means the
  direction is a function of log Z₀ = log E₀ + log(Z₀/E₀) alone within this family.
- **E5.** Matched-energy best-of-top-3 |r| with log Z₀.
- **Secondary.** S1 partialled R² (5 targets), log ν pair R², Okubo–Weiss fraction, all vs k
  (explore). Traversals along the imaginer's enstrophy direction at k*.

### Predictions, Stage 2

- **P2.1 (bridge)** The common-anchor imaginer at k = 4 has E1 ≥ 0.6, and the bridge is within ±0.1
  of it. Probability: 65%.
- **P2.2 (horizon)** The imaginer's E1 is ≥ 0.6 at every k, and its range across k is < 0.15 (no
  strong horizon effect). Probability: 55%. Reason: rendering any future frame needs the field's
  amplitude and spectral scale, at every horizon.
- **P2.3 (summarizer)** The summarizer's E1 is below the imaginer's at every k by ≥ 0.2 (70%), and
  ≤ 0.5 at every k (65%).
- **P2.4 (matched-energy)** Every imaginer and summarizer has E5 ≥ 0.8 at every k; the control stays
  near 0.18. Probability: 70%. So E5 will not separate imaginer from summarizer.
- **P2.5 (blur)** The imaginer's predicted / true energy in shells k ≥ 10 falls with horizon: < 0.5 at
  k = 16 and < 0.25 at k = 32. Probability: 70%.
- **P2.6 (regularization)** Pixel + VICReg at k* has E1 at least 0.2 below the plain imaginer at k*.
  Probability: 50%. Reason: every VICReg-regularized model so far (JEPA, static VICReg) has low E1.
- **P2.7 (fake physics)** The fake-physics imaginer at k* has E1 within 0.15 of the real imaginer.
  Probability: 55%.
- **P2.8 (specificity)** At k = 4, the imaginer's partial r with log Z₀ given trajectory-mean log P is
  ≥ 0.3. Probability: 50%.
- **P2.9 (lr)** The summarizer selects 1e-3 and the imaginer 3e-3. Probability: 40%.
- **P2.10 (compute)** Stage 2 training totals about 13 GPU-hours (35 runs); at least one run hits the
  12,000-step cap.
