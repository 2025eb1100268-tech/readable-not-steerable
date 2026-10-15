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
