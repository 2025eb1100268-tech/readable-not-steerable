# [BUILD] Predictions (append-only)

Each entry is written and committed **before** the stage it covers runs and is never edited
afterwards (git history is the check). Everything here is exploratory (`results/build/BRIEF.md`).
Wording: a "dial" is a latent direction whose movement produces a measured change in the decoded
field; no claims about models "knowing" anything.

---

## Entry 1 — 2026-10-17 — Stage 0 (baselines on existing models; no training)

Written before the steering harness exists.

### Fixed definitions

- **Models.** Stage 2 checkpoints, seeds 0–2: plain and VICReg autoencoders rendering vorticity,
  velocity and streamfunction (6 kinds), and the k = 4 imaginer at lr 1e-3: 21 models. The untrained
  control (seeds 0–2) has no decoder, so it cannot be steered; for it only the label-based
  readability below is reported.
- **Dial subspace** for these models: the whole pooled latent (128 dims).
- **Candidate dials:** the top-4 principal directions of the raw (centred, not z-scored) pooled
  latents of v1 train frames; σ_d = std of the projection on those frames.
- **Supervised upper bound:** the ridge probe (`probes/ridge.py`, α chosen on v1 val) for
  s = log(Z/E) from the pooled latent, fitted on v1 train frames; its direction mapped back to raw
  pooled space and unit-normalized; σ_d as above. (Dial subspace = full latent here, so reported once.)
- **Steering.** z′ = z + α·σ_d·d with α ∈ {−2, −1.5, …, +2}. Applied to the 8×8 map by adding the
  shift at every position (so the pooled latent moves by exactly α·σ_d·d); for the imaginer, to both
  context maps.
- **Frames:** 256 `explore` frames and 256 `matched-energy` frames, drawn with seed 0 (t ≥ 1; for the
  imaginer, t + 4 inside the trajectory).
- **Decoded field → vorticity ω:** vorticity output as is; velocity output → curl; streamfunction
  output → −∇²ψ; imaginer: its predicted velocity → curl. From ω: E (through the inverse Laplacian),
  Z, P; a = log E, s = log(Z/E).
- **Per direction** (slopes per frame by least squares over α, then the median over frames): Δa, Δs,
  Δlog P per unit α; effect = |Δa| + |Δs|; σ_shape = |Δs| / (|Δa| + |Δs|); monotonicity = median
  over frames of the Spearman correlation between α and whichever of a, s has the larger median slope.
- **Realism:** (i) cycle: re-encode the decoded field (ω → velocity through the inverse Laplacian,
  scaled like the inputs; velocity outputs used directly) and compare the change of its projection
  on d (relative to α = 0) with the commanded α·σ_d: Pearson correlation over (frame, α), gain, and
  drift = RMS change orthogonal to d / |commanded shift|; (ii) spectrum envelope: per-shell energy
  (k = 1…21) of the decoded field against the per-shell 1st–99th percentiles of 4,096 v1 train frames,
  fraction of shells inside, mean over frames, minimum over α; (iii) velocity outputs: relative
  divergence.
- **Dial criteria for this stage (brief's starting proposal + the agreed floor):** a direction is a
  dial if effect ≥ 0.1 log-units per unit α, monotonicity ≥ 0.9, cycle correlation ≥ 0.9 and envelope
  fraction ≥ 0.9 at every α. Loudness dial: σ_shape ≤ 0.2; shape dial: σ_shape ≥ 0.7. Measured on
  `explore` and on `matched-energy` separately; a dial must qualify on both.
- **Readability (reported, label-based):** |Pearson r| of each top-4 direction's projection with a and
  with s over the `explore` frames, for every model including the untrained control.

### Predictions

- **P0.1** Fewer than 25% of the 84 candidate dials (21 models × 4) clear the effect floor with
  monotonicity ≥ 0.9. Probability 75%. Reason: a pooled shift is a uniform offset of the map, and the
  imagination and contrast probes found such offsets barely change the rendered field (about
  0.004–0.08 log-units per std).
- **P0.2** Of the directions that do clear the floor, at least 75% are loudness-like (σ_shape ≤ 0.2).
  Probability 70%.
- **P0.3** No candidate dial in any model is a shape dial (all criteria above). Probability 85%.
- **P0.4** The supervised s-direction fails the effect floor in at least 4 of the 7 model kinds, so
  shape is not steerable through pooled shifts in these latents and the build's job is to make a
  shape dial **exist**, not just make it findable. Probability 60%.
- **P0.5** The top-1 direction is readable as loudness (|r| with a ≥ 0.7) in the plain autoencoders
  and the imaginer, while its steering effect is below the floor in most of them (readable but not
  steerable). Probability 55%.
- **P0.6** At least 80% of the directions that clear the floor also pass realism. Probability 60%.
