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

---

## Entry 2 — 2026-10-18 — Final dial thresholds (approved by the investigator, before any Stage 1 training)

Set from Stage 0 (`results/build/stage0.md`, `results/build/stage0_ideal_dials.txt`) and a check of
head consistency on existing ω/ψ autoencoder pairs. They apply from Stage 1 on; Stage 0's numbers
stay under entry 1's definitions. Every criterion is measured on `explore` and on `matched-energy`
separately and must pass on both; it is scored on the vorticity (primary) head, and every other
head is reported.

- **Effect:** |Δa| + |Δs| ≥ **0.2** log-units per unit α (median over frames). Raised from 0.1: the
  best of Stage 0's 84 candidates was 0.086.
- **Monotonicity:** Spearman ≥ 0.9 on the dominant coordinate. (At Stage 0 it was 1.000 even at the
  10th percentile, so on its own it separates nothing.)
- **Loudness dial:** σ_shape ≤ 0.2. **Shape dial:** σ_shape ≥ 0.7.
- **Realism:**
  - *Cycle:* correlation ≥ 0.9 **and gain in [0.5, 2]** (decoded vorticity → velocity through the
    inverse Laplacian → re-encoded; the change of its projection on d against the commanded α·σ_d).
  - *Watermark:* shared-pattern fraction ≤ 0.5 (the fraction of the α = ±2 change, relative to α = 0,
    that is one pattern common to all frames; the larger of the two ends).
  - *Envelope relative to an ideal dial:* at every α, the envelope fraction (shells k = 1…21 inside the
    1st–99th percentiles of 4,096 v1 train spectra, mean over frames) ≥ the ideal reference's − 0.05.
    Ideal reference: per frame and α, the unsteered decoded field (α = 0) tilted by k^τ (τ by
    bisection) to the steered field's s, then rescaled to its a; same a and s, so a shortfall comes
    from how the spectrum is shaped beyond (a, s). The absolute envelope is reported, not scored.
  - *Two-head models:* head inconsistency (‖ω_head − (−∇²ψ_head)‖ / ‖ω_head‖, median over frames) ≤
    **1.5× its α = 0 value** at every α. The absolute value is reported, not scored. (The proposed
    absolute bar of 0.3 was dropped by the investigator: existing ω/ψ autoencoder pairs give 0.17–0.21
    plain and 0.83–0.93 with VICReg at α = 0, so it would reject every model before any steering.)
  - *Velocity outputs:* relative divergence, reported.
- **Model level:** a variant *produces* a shape (loudness) dial if at least 2 of its 3 seeds have one
  among PC1–8 of its dial subspace. Every seed and every direction is reported, and the PC1–4 subset
  too.

---

## Entry 3 — 2026-10-18 — Stage 1 (M1-spec and M1-G, 3 seeds each)

Written before any Stage 1 training.

### Fixed definitions

- **Models** (`dials/split.py`; configs from `dials/stage1_configs.py`). The shared encoder, initialized
  as the Stage 2 autoencoders with the same seed; map channels [0:4] dial-ω, [4:8] dial-ψ, [8:128]
  content.
  - **M1-spec:** vorticity head = the autoencoders' decoder on the [dial-ω, content] channels;
    streamfunction head = the same design on [dial-ψ, content].
  - **M1-G:** the dial channels are pooled to an 8-dim vector; each head decodes the content map alone,
    with FiLM after its input stage and after each upsampling stage (h ← h·(1 + γ) + β; γ, β linear in
    that head's 4 dial dims, zero-initialized). The content map is divided by its per-sample RMS and
    multiplied by one learned scalar gain, the same for every sample. (Implementation fix: with the RMS
    fixed at 1, VICReg's unit-std target on the pooled content is reachable only by a spatially constant
    map.)
- **Training:** the autoencoder recipe (v1 train, batch 256, AdamW lr 1e-3, weight decay 0.05, cosine
  decay with 5% warmup, gradient clipping 1.0, 30 epochs, bf16). Targets ω and ψ, each scaled to unit
  std over v1 train (2.6306 and 1.1693, the Stage 2 values). Loss = 14.699·MSE_ω + 42.520·MSE_ψ
  + 25·VICReg variance + 1·VICReg covariance, the last two on the pooled content (120 dims) only.
  **Head-loss rule:** weights inversely proportional to each head's VICReg-autoencoder counterpart's
  final val normalized MSE (0.01047 ω, 0.00362 ψ; mean of 3 seeds), with geometric mean 25 (the Stage 2
  rendering weight), so both heads contribute equally at their counterparts' quality.
- **Latents:** dial subspace = the 8 pooled dial dims (M1-spec: spatial mean of the dial channels;
  M1-G: the FiLM input vector). Content latent = the pooled content as the heads see it (after
  normalization in M1-G). Full latent = both (128 dims).
- **Steering:** the shift is added to the dial vector (M1-G) or at every position of the dial channels
  (M1-spec); for full-latent directions, the content part is added at every position of the content map.
- **Candidates per run:** PC1–8 (raw centred PCA of the dial latent on v1 train frames; σ_d = train
  std); the **ω−ψ contrast direction** (label-free: the top raw PC of each group, u_ψ's sign chosen so
  the two projections correlate positively, c = [u_ω/sd_ω, −u_ψ/sd_ψ], unit norm); the supervised
  s-direction (ridge) in the dial subspace and in the full latent (upper bound; reported, not part of
  "produces").
- **Frames, harness and criteria:** the Stage 0 frames (256 explore, 256 matched-energy) and envelope;
  entry 2's criteria.
- **Rendering trigger** (stop and report): a head's val normalized MSE > 2× its VICReg-autoencoder
  counterpart (ω 0.01047, ψ 0.00362), or > 0.02, or < 50% of the true k ≥ 10 power of its target kept.
  **Dial collapse:** the largest dial-dim std < 1% of the median content-dim std (train frames).
- **Readouts:** prominence (E1-style: best of the top-3 slow directions, |r| with log Z₀ and with
  log E₀ on explore inviscid trajectories, fitted on v1 train + val inviscid; PCA step 8 in the dial
  subspace, 32 otherwise) in the dial, content and full latents. Ridge probes scored on explore: log E,
  log Z, log P, s, log(P/Z) (plain and time-partialled R²) and log ν (single frame, frame pair) from
  each latent. The Stage 2 autoencoders get the same prominence and probes, for reference.
- **Wording:** M1-G's loudness dial exists by construction and is reported as a design property, not a
  finding. The finding is whether a label-free shape dial emerges.

### Predictions

- **P1.1** Neither model fires the rendering trigger in any seed, and no dial subspace collapses. 70%.
- **P1.2** M1-spec produces no shape dial (fewer than 2 of 3 seeds). 85%.
- **P1.3** M1-spec produces no dial of any kind among PC1–8 (fewer than 2 seeds with a loudness or a
  shape dial): pooled shifts of spatial channels stay weak, as in Stage 0. 65%.
- **P1.4** (design check, not a finding) M1-G has a loudness dial among PC1–8 in at least 2 of 3 seeds. 80%.
- **P1.5** M1-G produces a shape dial. 35%.
- **P1.6** (investigator's prediction) If a shape dial emerges in M1-G, it is, or loads mainly on, the
  ω−ψ contrast direction: cos² ≥ 0.5 with c for the largest-effect shape dial in most seeds that have
  one. Recorded either way; my probability 60%. Also reported: whether c itself qualifies as a shape dial.
- **P1.7** The two dial groups' top directions correlate at |r| ≥ 0.8 in every seed: in M1-G (both
  carry loudness for their head) 70%; in M1-spec (the two groups learn the same loudness direction,
  the M3 trigger) 55%.
- **P1.8** M1-G's content decodes s (plain R² ≥ 0.8 on explore) but not log E (R² < 0.5), and the full
  latent decodes log E and log Z with R² ≥ 0.95. 50%.
- **P1.9** The supervised s-direction in M1-G's dial subspace is a shape dial in at least 2 seeds
  (shape is steerable through the dials at all). 50%.
- **P1.10** In M1-G, the dial subspace's E1 with log E₀ is ≥ 0.8 and the content subspace's E1 with
  log Z₀ is < 0.5 (loudness moved out of content). 55%.

Compute: 6 runs ≈ 2.5 GPU-hours, readouts ≈ 0.2.

---

## Entry 4 — 2026-10-21 — Stage 2: M2-SF, 3 seeds (variant approved by the investigator)

Written before any Stage 2 training. Why this variant (Stage 1, `results/build/stage1.md` and the
LOG notes): M1-G's RMS-normalized content still decoded log E at R² 0.99, so the decoder took most
of the loudness from content and its loudness knob failed the cycle check (gain 0.18–0.34); and each
dial group reached only its own head, so the ω−ψ contrast could change only the heads' relative
loudness, never the spectral shape of one field. M2-SF removes both causes.

### Fixed definitions

- **Model M2-SF** (`dials/split.py`: `ScaleFree`, `BranchHeads`; training kind `branch_ae`).
  - *Content encoder:* the shared encoder design (initialized as the Stage 2 autoencoders with the same
    seed; last layer 120 channels) applied to x / RMS(x), the RMS taken per sample over channels and
    pixels. Its output is exactly invariant to the input's overall scale.
  - *Dial encoder:* the same design (init seed + 500, 8 output channels) on the raw input, spatially
    pooled to the 8-dim dial vector.
  - *One rendered field:* one FiLM decoder (as in M1-G, FiLM zero-initialized) renders ω from the
    content map, every stage modulated by all 8 dials; ψ is the inverse Laplacian of that ω (dealiasing
    mask applied), so the two agree by construction.
  - Parameters 3.00 M (M1-G 4.25 M: one decoder instead of two).
- **Training:** the Stage 1 recipe and loss: 14.699·MSE_ω + 42.520·MSE_ψ (on the derived ψ)
  + 25·VICReg variance + 1·VICReg covariance on the pooled content (120 dims); configs from
  `dials/stage2_configs.py`.
- **Latents:** dial = the 8-dim dial vector; content = the pooled content map; full = both (128 dims).
- **Steering:** a dial shift is added to the dial vector; full-latent directions also add their content
  part at every position of the content map.
- **Candidates per run:** PC1–8 of the dial vector (raw centred PCA on v1 train frames); the supervised
  s-direction in the dial subspace and in the full latent (upper bound). No ω−ψ contrast candidate: the
  dial dims no longer belong to a head, so the investigator's P1.6 has no counterpart here.
- **Criteria:** entry 2, scored on the single rendered field (ω). The head-consistency criterion does
  not apply (ψ is derived from ω). Newly reported, not scored: the decoded ω's spatial-mean share of
  ⟨ω²⟩ (enstrophy counts it, energy does not; at most 1.5×10⁻⁴ in every Stage 1 decode, PC1–8 at
  α ∈ {−2, 0, 2}, so it cannot move s measurably).
- **Rendering trigger, dial collapse and readouts:** as entry 3 (the ψ trigger on the derived ψ);
  readout script `dials/stage1.py` with `configs/build/stage2_eval.yaml`.
- **Wording:** M2-SF's loudness dial exists by construction and is reported as a design property, not
  a finding. The finding is whether a label-free shape dial emerges.

### Predictions

- **P2.1** No rendering trigger and no dial collapse in any seed. 75%.
- **P2.2** (design check, not a finding) A loudness dial among PC1–8 that passes every criterion,
  including the cycle check, in at least 2 of 3 seeds. 70%.
- **P2.3** M2-SF produces a shape dial (at least 2 of 3 seeds). **20%.** Reason: the scale-free content
  still sees the field's spectral shape, so the decoder needs only one number from the dials (the
  loudness); nothing pushes shape into them.
- **P2.4** The supervised s-direction in the dial subspace is a shape dial in at least 2 seeds. 25%.
- **P2.5** The dial subspace decodes log E with R² ≥ 0.95 (explore) and its PC1 is readable as
  loudness (|r| with a ≥ 0.9), in every seed. 70%.
- **P2.6** The content latent decodes log E with R² < 0.9 (below M1-G's 0.99: loudness now reaches it
  only through pattern correlations in the data) while it decodes s with R² ≥ 0.9. 55%.
- **P2.7** ω rendering within 1.5× of M1-G's (val normalized MSE ≤ 0.014) in every seed. 70%.

Compute: 3 runs ≈ 1.3 GPU-hours, readouts ≈ 0.05 (≈ 2.7 of 15 used before this stage).
