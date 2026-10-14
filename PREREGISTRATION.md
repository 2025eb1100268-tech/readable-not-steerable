# Pre-registration — geometric-scout

Status: **confirmed by the investigator on 2026-10-14 and committed before any model was
trained.** Not edited afterwards; later changes go in the dated AMENDMENTS section at the end.

Question: does a network trained only to predict how a 2D fluid evolves (no labels, no physics
hints) organize its latent space around fluid quantities (energy, enstrophy, vortex structure,
invariants, viscosity) measurably better than an untrained network of the same architecture?

## 1. Success criteria (copied verbatim from the brief)

The hypothesis is **supported** if all of the following hold on test-ID, averaged over 3 seeds:
- **S1.** JEPA linear-probe R² exceeds the random-init encoder by ≥ 0.20 on at least 3 of: enstrophy, palinstrophy, vortex count, Okubo–Weiss fraction, dissipation rate, after partialling out time.
- **S2.** ν is linearly decodable from JEPA latents with R² ≥ 0.6 (log ν), and ≥ 0.3 above random-init.
- **S3.** At least one of the top-3 unsupervised slow directions from Phase 4 correlates with a true inviscid invariant at |r| ≥ 0.8, and this does not happen for random-init.

**Partially supported** if S1 holds but S2 or S3 fails. **Not supported** if S1 fails.

Separately report (no pass/fail): JEPA vs pixel-predictor comparison, test-OOD degradation, and the non-physical control if run.

## 2. Clarifications and amendments approved before data generation and training (2026-10-14)

Agreed with the investigator after a design review (planning pilots summarized at the top of
`results/LOG.md`). Where these differ from §1, these govern.

**2.1 S1 (approved rule).** For each of the five targets, compare the JEPA and the untrained
control on test-ID, each averaged over its 3 seeds, using the time-partialled linear-probe R²
(§5.3). A target *counts* if
- R²_JEPA − R²_control ≥ 0.20, **or** JEPA halves the unexplained variance:
  (1 − R²_JEPA) ≤ 0.5 · (1 − R²_control); **and**
- the 95% trajectory-bootstrap CI of R²_JEPA − R²_control (§5.6) excludes zero.

S1 holds if at least 3 of the 5 targets count **and the Okubo–Weiss fraction is one of them**.
Dissipation still counts toward the 3 (it is 2νZ and partly re-tests S2).
Reason: with a fair control, enstrophy, palinstrophy and vortex count sit near the R² ceiling,
so the +0.20 margin alone is unreachable for them. The halving branch is noise-guarded by the CI.
The mandatory Okubo–Weiss target is there because a dissipation gain can come from reading ν off
a single frame's spectrum rather than from learned geometric structure.

**2.1a Partialling for S1: time × family (adopted).** "After partialling out time" in S1 is
implemented as partialling out a one-hot of (frame index × initial-condition family). The
family is vortex blobs vs random spectral fields.

Reason: the Phase 1 baselines (train/val only) showed family is a large confound for two S1
targets. Family alone explains 0.79 of vortex-count and 0.76 of Okubo–Weiss-fraction variance
after time. The untrained control reads the Okubo–Weiss fraction at R² 0.72 within blob runs
but 0.08 within spectral runs. A model gets no credit for recognizing which generator made the
field, just as it gets none for "how late is it". Time-only partialled R² is reported as a
secondary number.

**2.2 S2.** As written, on viscous trajectories only (log ν is undefined for ν = 0), with the
frame pair defined in §5.4. A single-frame log-ν probe and the pair-minus-single gain are
reported, unscored. An S2 pass alone does not show ν was inferred from dynamics.

**2.3 S3 (approved rule).** Per-invariant reading with a guard. Scored invariants: energy E and
enstrophy Z, the exact invariants of the discretized inviscid equations; ∫ω⁴ is reported but
unscored. S3 holds if, for at least one scored invariant I:
- JEPA's best held-out |r| over its top-3 slow directions, averaged over seeds, is ≥ 0.8;
- the control's corresponding value is < 0.8;
- JEPA − control ≥ 0.1; and
- the 95% trajectory-bootstrap CI of that gap excludes zero.

The protocol is in §6. Energy cannot pass, because the untrained control already reaches
|r| ≈ 0.94 (energy is a quadratic function of the input itself), so S3 effectively rests on
enstrophy.

**2.4 Model family (approved).** The main JEPA predicts the target encoder's *spatial feature
map*. Pooled-vector prediction becomes a 3-seed ablation.

**2.5 Controls (approved).**
- Untrained control: the shared encoder at its Kaiming initialization, with seeds equal to the
  trained models' seeds. Control seed s is exactly model s's starting weights.
- Non-physical dynamics control: §4.5.
- Static control: same encoder, VICReg on two symmetry-transformed views of one frame, no time.

**2.6 Readout (approved).** The raw pooled latent is the primary probed representation. The
latent averaged over the 64 sub-stride circular shifts is a pre-registered secondary readout.
**Interpretation rule:** if a JEPA gain over the control disappears in the shift-averaged readout,
the report attributes it to shift-tolerance, not physical structure.

**2.7 Order of work.** Data and untrained baselines (train/val only) before this document is
committed; no trained model before the commit. Hyperparameters are frozen on val. The test sets
(test-ID, test-OOD) are evaluated exactly once, by one scripted run after the Phase 4 protocol is
frozen (§8).

## 3. Data (generated; `configs/phase1_data.yaml`, `data/generated/v1/manifest.json`)

| Item | Value |
|---|---|
| Solver | 2D incompressible NS, vorticity form, [0, 2π]², pseudo-spectral 64², 2/3 dealiasing, integrating-factor RK4, float64, CFL 0.3; validated in Phase 0 (`results/solver_validation.png`) |
| Trajectories | 2,000; 25% inviscid (ν = 0), 75% viscous with ν log-uniform in [3e-3, 5e-2] (3e-3 = smallest ν resolved at 64², Phase 0 pilot) |
| Initial conditions | 50% periodic Gaussian vortex blobs (2–12, random signs, radius log-uniform 0.25–0.6, random positions), 50% random-phase fields with E(k) ∝ k⁴ exp(−2(k/k₀)²), k₀ ∈ [2, 5]; families balanced within inviscid and viscous; each normalized to rms vorticity log-uniform in [2, 4]; unique initial condition per trajectory |
| Snapshots | fixed Δt = 0.04; viscous 64 frames (t ≤ 2.52); inviscid 24 frames (t ≤ 0.92, before thermalization of the truncated system: median top-shell enstrophy 0.9% at the last frame, 6.2% of runs above 5%) |
| Model input | (u, v) float32 only, divided by one scale (train velocity std = 0.9548) |
| Re | U₀L₀/ν with U₀ = √(2E₀), L₀ = 2π Σ E(k)/k / Σ E(k) at t = 0; viscous range 13–4,229 |
| Checksums (SHA-256) | velocity.npy `ef3fbb8d3c5ac41b…`, frames.npz `3991dce2113f200f…`, trajectories.json `8c5b32750f92d888…` (full values in the manifest) |
| Seeds | data 20261014 (SeedSequence child per trajectory); split 7 |

**Split policy (by trajectory, never by frame).**
- test-OOD: viscous trajectories whose log Re lies in the interior quantile band [0.45, 0.55] of
  viscous log Re (Re 286–389). These are never used in train or val.
- The remaining trajectories are split 70/15/15 into train/val/test-ID, stratified by family ×
  (inviscid, or viscous Re tercile).
- Result: train 1,295, val 279, test-ID 276, test-OOD 150 trajectories. Inviscid: 350 / 76 / 74 / 0.
- Leakage is tested in `tests/test_data.py`.

**Probe targets** (`data/targets.py`, from the true float64 fields):
- E, Z, P (Z = ½⟨ω²⟩, so dE/dt = −2νZ); max|ω|; vorticity kurtosis; ∫ω⁴.
- Okubo–Weiss Q, with vorticity-dominated Q < −0.2σ_Q and strain-dominated Q > +0.2σ_Q; the
  Okubo–Weiss fraction is the vorticity-dominated area.
- Vortex count: 4-connected components of the vorticity-dominated mask, merged across the
  periodic boundaries, area ≥ 4 cells.
- Dissipation ε = 2νZ; ν; time since t = 0.
- Transforms: log for E, Z, P, max|ω|, kurtosis, ε, ν; log1p for vortex count; none for the
  fractions or time.

## 4. Models and training

All encoders are identical (`models/encoder.py`):
- circular padding, GELU, no normalization layers;
- widths 32-64-128-128 with three stride-2 stages to an 8×8×128 spatial map; pooled latent
  d = 128 is its spatial mean;
- Kaiming-normal init, zero biases; trained-model seeds {0, 1, 2}.

Common training settings:
- train split only, one sample = (trajectory, t, k) with k ∈ {1, 4} uniform;
- AdamW, batch 256, weight decay 0.05, 5% warmup then cosine decay, bf16 autocast;
- a fixed budget of 40 epochs over train samples;
- no data augmentation except in the static control.

**4.1 JEPA (main).**
- Online encoder f_θ; target encoder f_ξ = EMA(θ), momentum 0.996 → 1 (cosine).
- Predictor: three 3×3 circular conv layers (width 256, GELU). Input: the concatenated online
  spatial maps of frames t−1 and t, plus a learned embedding of k. Output: the target map f_ξ(x_{t+k}).
- Loss: 25·MSE(prediction, stop-grad target) + 25·variance hinge + 1·covariance (VICReg terms,
  on the pooled online latents).
- The probed encoder is the EMA target encoder.

**4.2 Pixel predictor.** Same encoder on frames t−1 and t; the concatenated maps plus the k
embedding feed a conv decoder (three ×2 upsampling stages) that predicts (u, v) at t+k; MSE loss.

**4.3 Pooled-JEPA ablation.** As 4.1, but the predictor is an MLP from [z_{t−1}, z_t, emb(k)]
to the pooled target latent.

**4.4 Static control.** VICReg (25 / 25 / 1) on the pooled latents of two views of the same
frame. Each view gets a random circular shift and a random D4 element, with correct vector
transformation of (u, v).

**4.5 Non-physical dynamics control.**
- Same JEPA objective and budget, trained on a second dataset with the same initial-condition
  distribution, Δt, frame counts and split structure.
- Its dynamics are linear and per Fourier mode: ω̂_k(t) = ω̂_k(0)·exp((−r·g_k + i·c_k)t). The
  multipliers g_k > 0 and phase speeds c_k are random but fixed for the dataset; the hidden rate
  r per trajectory is log-uniform. There are no r = 0 runs, so no quadratic integral or Casimir
  is conserved.
- The r range is set so the median field keeps ≥ 20% of its initial enstrophy by the last frame.
  The exact constants are recorded in that dataset's config before training it.
- The trained encoder is evaluated on the Navier–Stokes test data.

**4.6 Label-free model selection** (the only tuning, on val, seed 0). Grid: peak learning rate
∈ {3e-4, 1e-3}. Selection rule:
- Discard configurations whose val pooled latents collapse: effective rank < 8 or median
  per-dimension std < 0.1.
- Among the rest, choose the lowest val prediction loss divided by the copy baseline. For JEPA,
  the copy baseline is the target map of frame t used as the prediction for t+k; for the pixel
  model it is x_t; for the static control, the invariance loss itself.
- The chosen configuration is then trained with seeds 0, 1, 2.
- If no configuration survives, the run is logged as failed and I stop and report. No probe or
  label is used for selection.

Collapse diagnostics (per-dimension std, effective rank) and training curves are reported for
every run.

## 5. Probes (Phase 3)

5.1 **Representation.** Raw pooled latent (primary), shift-averaged pooled latent (secondary),
for every encoder and seed. Latents are extracted in float32.

5.2 **Ridge.**
- Features z-scored with train statistics.
- α chosen on val by MSE from logspace(−3, 7, 41); the only label-based selection.
- Fit on all train frames of the relevant trajectories (viscous only for ε and ν).
- Secondary: a one-hidden-layer MLP probe (256 GELU units, early stopping on val).

5.3 **Time partialling (S1).**
- Targets and features are residualized on a one-hot of the frame index crossed with
  initial-condition family (§2.1a; `Dataset.partial_index()`), using per-index means from train.
- The ridge is fitted on the train residuals.
- On an evaluation set: R²_partial = 1 − SSE(time-only + latent) / SSE(time-only), where the
  time-only model predicts the train per-index mean.
- Plain R² (against the evaluation set's own mean) is also reported.

5.4 **ν (S2).** Ridge on [z_{t−1}, z_t], with each half z-scored using train statistics, over
consecutive stored frames (Δt = 0.04) of viscous trajectories; no time partialling. The same
construction is used for every encoder.

5.5 **Out-of-distribution test (no pass/fail).**
- Probes are fitted on train and scored on test-OOD.
- Reported: MSE, compared with (a) MSE on test-ID viscous trajectories in the neighbouring
  log-Re quantile bands [0.35, 0.45) ∪ (0.55, 0.65], and (b) a constant predictor at the train
  mean.
- R² against the narrow band's own variance is not used.

5.6 **Uncertainty.**
- Mean ± std over 3 seeds.
- Trajectory bootstrap (1,000 resamples of test trajectories), recomputing the seed-averaged
  metric of each model in each resample. The CI of a difference uses paired resamples.

## 6. Unsupervised invariant discovery (Phase 4)

- On inviscid trajectories only: z-score latents and apply PCA to 32 dimensions, fitted on the
  train+val inviscid frames (426 trajectories).
- Estimate Σ_within with Ledoit–Wolf shrinkage and Σ_between from trajectory means. Solve
  Σ_within w = λ Σ_between w and keep the 3 lowest-λ directions.
- Score on the 74 test-ID inviscid trajectories: |Pearson r| between each direction's
  trajectory-mean projection and log E(0), log Z(0) (scored) and log ∫ω⁴(0) (reported). Take
  the maximum over the top 3 per seed, then average over seeds.
- Run identically for every encoder (JEPA, pixel, controls).
- Also reported, unscored:
  - held-out λ;
  - specificity, i.e. |r| with trajectory means of near-invariant non-scored quantities
    (kurtosis, max|ω|, OW fraction). Inviscid flow conserves the vorticity distribution, so
    these are slow by physics.
  - PCA and t-SNE views coloured by enstrophy, vortex count and ν;
  - latent traversals, decoded into the energy spectrum and vorticity PDF, with nearest-neighbour
    frames.

## 7. Untrained-control baselines measured before this commit (val, fit on train, seeds 0–2)

These are for calibration only (`results/phase1/untrained_baselines.md`). The test values decide.

| Target | Control R², time × family partialled (pre-registered) | Needed by JEPA under 2.1 | Control R², time only (secondary) |
|---|---|---|---|
| log enstrophy | 0.762 | 0.881 | 0.782 |
| log palinstrophy | 0.443 | 0.643 | 0.511 |
| log1p vortex count | 0.580 | 0.780 | 0.899 |
| Okubo–Weiss fraction (mandatory) | 0.606 | 0.803 | 0.844 |
| log dissipation | 0.236 | 0.436 | 0.259 |
| log ν, frame pair (S2) | 0.486 | ≥ 0.786 | — |
| S3, log E / log Z (max \|r\|) | 0.938 / 0.233 | unreachable / 0.800 | — |

## 8. Final evaluation and reporting

- One scripted run computes every test-ID and test-OOD metric for every model, seed and readout
  after §4–§6 are frozen on val. Its output is final; any rerun (e.g. after a crash) is logged
  with the reason.
- `REPORT.md` reports the verdict against §1–§2, every table (models × targets, ID and OOD,
  mean ± std), the JEPA vs pixel comparison, the controls, threats to validity, the 2D
  limitation (no vortex stretching) and the actual compute used.
- Every run is in `results/LOG.md`, including failed runs.

## AMENDMENTS

(none yet)
