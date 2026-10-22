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

---

## Entry 5 — 2026-10-22 — Stage 2b: M2-SW (3 seeds) and M2-SF with the ω-only loss (1 seed, attribution)

Written before any Stage 2b training. The investigator's specification: M2-SW, 3 seeds, ω-only loss
(ψ reported, not trained), plus 1 seed of M2-SF with the ω-only loss for attribution; a whitening
gain cap justified from the explore spectra, shells above the dealiasing cutoff zeroed, and a check
with a figure that the whitened inputs carry no high-k noise artefacts; a second, secondary
dial-identification method in effect space. Why (Stage 2): M2-SF's dials carried s (R² 0.95–0.96)
but the decoder took shape from the scale-free content, which still saw the spectral shape.

### Fixed definitions

- **M2-SW** = M2-SF (entry 4) with one change: the content encoder sees W(x) (`dials/whiten.py`)
  instead of x / RMS(x).
  - *W:* RMS-normalize; rescale each integer shell k = 1..21 (round |k|; both velocity components
    alike, so incompressibility is kept) to ref_k, the median RMS-normalized shell power of the v1
    train frames train.py samples for its target scales (every (n // 4096)-th train frame); the gain
    is capped at **G_max = 1000**; the mean and every shell k > 21 are set to zero. Shells 22–29 hold
    only the corner modes of the solver's square dealiasing mask (partial rings that carry real energy
    in thermalized inviscid frames); shells ≥ 30 are empty. Float64 with autocast off. W(c·x) = W(x)
    for c > 0.
  - *Cap justification* (`results/build/stage2b_whitening.md`, 4,096 explore frames). The noise floor
    is taken empirically: the power the stored fields carry where the solver puts none (shells k ≥ 31),
    per mode, scaled to each shell's mode count, 99th percentile over frames; it is about 2× the
    simulated float32 rounding, so it is the one used. The largest gain that keeps the amplified floor
    ≤ 1% of ref_k is 3,212 (binding at k = 21); G_max = 1000 keeps it ≤ 0.10% of ref_k in every shell.
  - *What the cap costs:* it binds in 1.22% of explore (frame, shell) pairs and 5.6% of frames (the
    smoothest viscous frames, shells k ≥ 8). This is the only route by which spectral shape can still
    reach the content encoder.
  - *Artefact check* (`results/build/stage2b_whitening.png`): power beyond k = 21 is 7×10⁻³² of the
    power in shells 1–21; uncapped shells equal ref_k to 1×10⁻¹³; the amplified floor is ≤ 5.8×10⁻¹⁰
    of the whitened power in every frame; the whitened fields show no grid-scale noise.
- **Loss (both models):** 14.699·MSE_ω + 0·MSE_ψ + 25·VICReg variance + 1·VICReg covariance (pooled
  content); ψ is derived from ω and reported (val normalized MSE of the derived ψ). Otherwise the
  Stage 1–2 recipe. Configs from `dials/stage2b_configs.py`.
- **M2-SF, ω-only loss** (`b2b_m2sfw_s0`): entry 4's M2-SF, seed 0, with this loss. Attribution:
  compared with Stage 2's M2-SF seed 0 (ω val normalized MSE 0.0182; k ≥ 10 power kept 0.48).
- **Dial identification:**
  - *Primary (unchanged):* PC1–8 of the 8-dim dial vector, and the 2-of-3-seeds rule (entry 2).
  - *Secondary, reported:* effect space (`dials/effect_space.py`). J = the mean over the 256 explore
    steering frames of ∂ log E_k(decoded ω) / ∂g, for shells k = 1..21 and the raw dial vector g
    (autograd; checked against finite differences in `tests/test_dials.py`). SVD of J; the top-4
    right-singular vectors ES1–4 (unit norm; σ_d = train std along each) are steered with the same
    harness and criteria. "The effect-space method produces a shape dial" = at least 2 of 3 seeds
    have one among ES1–4.
  - *Agreement, per seed:* both methods find a shape dial or neither does, and when both do, their
    largest-effect shape dials have cos² ≥ 0.5. "The methods agree" = every M2-SW seed agrees.
  - Also reported: cos² of each ES with each PC; the supervised s-direction (dial and full latent).
- **Criteria:** entry 2 on the single rendered field (no head criterion). Rendering trigger, dial
  collapse and readouts as entries 3–4; readout script `dials/stage1.py` with
  `configs/build/stage2b_eval.yaml`.
- **Wording:** in M2-SW, shape information in the dials is a design property (content can't carry it,
  except through capped shells), and so is the loudness dial. The finding is whether label-free
  analysis of the dials yields separate loudness and shape directions that each steer cleanly and pass
  realism.

### Predictions

- **P2b.1** (attribution) M2-SF with the ω-only loss does not fire the rendering trigger (k ≥ 10 power
  kept ≥ 50%) and renders ω better than Stage 2's M2-SF seed 0 (val normalized MSE < 0.0182): the ψ
  term caused the Stage 2 trigger. 65%.
- **P2b.2** M2-SW fires no rendering trigger in any seed. 45%. (The content no longer tells the decoder
  the spectrum; 8 dials acting through 4 FiLM stages must set it.)
- **P2b.3** Shape leaves the content: s decoded from M2-SW's content latent with R² < 0.5 on explore
  (0.88–0.92 in M2-SF). 55%. (Against: the capped shells, and pattern geometry such as vortex size
  that whitening does not remove.)
- **P2b.4** (design check) M2-SW's dials decode s with R² ≥ 0.95, and a loudness dial appears among
  PC1–8 in at least 2 seeds. 65%.
- **P2b.5** PCA: M2-SW produces a shape dial (at least 2 of 3 seeds). 35%. Reason: a and s are
  correlated in the data (r = −0.82 on explore), so the variance-ordered PCs are likely mixtures.
- **P2b.6** Effect space: a shape dial among ES1–4 in at least 2 seeds. 45%.
- **P2b.7** The two methods agree in every seed: 45%. If they disagree, the effect-space method is the
  one that finds a shape dial: 75% (conditional).
- **P2b.8** The supervised s-direction in M2-SW's dial subspace is a shape dial in at least 2 seeds
  (shape is steerable through these dials at all). 55%.

Compute: 4 runs ≈ 1.75 GPU-hours, readouts ≈ 0.1 (≈ 4.1 of 15 used before this stage).

---

## Entry 6 — 2026-10-23 — Grid-artefact criterion; Step A (loudness by symmetry on M2-SW, no training)

Written before Step A is run. The investigator's specification: a grid-artefact metric with a threshold
that catches M2-SW seed 1's blocky output at α = −2, committed here, applied to every steered field
from now on and retroactively to the Stage 2b dials (reported, labelled as calibrated on seed 1); then
a symmetry-based loudness direction for each M2-SW seed, run through the full harness; the effect-space
SVD is dropped as a discovery method (its Stage 2b numbers stay in the record).

### Grid-artefact criterion (calibrated on M2-SW seed 1's PC2 at α = −2)

- **Metric** (`dials/fields.py: out_of_band_share`): for each decoded field, the share of its power
  outside the solver's square dealiasing band (|kx| or |ky| > 21), from the unmasked FFT. The solver puts
  no power there (true explore and matched frames: ~4×10⁻³²), so a decoder's power there is artefact; the
  decoder's 2×-upsampling stages put their block and checkerboard harmonics there.
- **Criterion:** a direction fails on a dataset if more than **1% of frames** have an out-of-band share
  above **τ = 1×10⁻²** at some α. It is applied to the primary decoded field when that field is a scalar
  (vorticity or streamfunction output), on explore and matched-energy, alongside entry 2's criteria.
- **Calibration** (`results/build/artefact_calibration.png`, `dials/artefact_calibration.py`): seed 1's
  blocky frame (frame 0, α = −2) has a share of 3.2×10⁻²; its striped ES3 frames 1.1–1.3×10⁻¹. No
  unsteered decoded frame of any Stage 2b model exceeds 2.5×10⁻³, so τ is 4× above that and 3× below
  the calibration case. In the figure, frames above about 1.3×10⁻² look visibly broken (blocky,
  striped, checkerboard or mottled); frames near 10⁻² are mixed. A ring-relative lattice statistic was
  tried first and rejected: on true data it reaches 1.2×10⁻², and it flags weak, invisible structure
  in rings the steering has emptied.
- **What was seen before the tolerance was chosen** (frames with share > 10⁻², explore / matched):
  M2-SW PC2, seed 0: 0.8% / 0%; seed 1: 1.2% / 3.5%; seed 2: 2.0% / 0%. M2-SF with the ω-only loss,
  PC1: 4.7% / 0% (checkerboards at α = +2, which the median-based criteria missed). Sensitivity of the
  tolerance: at 0% no M2-SW PC2 survives; at 2% seed 2's would too; seed 1 is caught at any tolerance
  below 3.5%. **Known consequence, not a prediction:** after this criterion, M2-SW keeps a PCA shape
  dial in seed 0 only (1 of 3), so M2-SW no longer meets the 2-of-3 rule for a shape dial.
- **Retroactive application:** every dial-space direction of the Stage 2b models, and (added, because
  the calibration figure showed the failure mode) Stage 2's M2-SF loudness dials (PC1, recomputed the
  same way: that stage's results file keeps no direction vectors). Reported, labelled as calibrated on
  seed 1.

### Step A definitions (`dials/stepA.py`, `configs/build/stepA.yaml`)

- **Loudness by symmetry:** in the 8-dim dial space of each M2-SW seed, the unit direction d minimizing
  J(d) = Σ ‖y_α − c·y_0‖² / Σ ‖y_α − y_0‖², where y_α is the decoded vorticity at z + α·σ_d·d, the sums
  run over α ∈ {−2, −1.5, …, 2} \ {0} and over 256 explore frames disjoint from the steering frames, and
  σ_d = √(dᵀCd) with C the v1-train covariance of the dial vector.
  - *Choices made explicit:* c is fitted per frame and α (least squares), i.e. each field may be
    rescaled by its own factor. J is the fraction of the steered change that rescaling does not explain:
    the investigator's "residual after the best scalar fit", normalized by the size of the change,
    because the unnormalized residual is minimized trivially by a direction that changes nothing.
  - *Optimization:* Adam (lr 0.05, 150 steps, batches of 32 frames) on an unnormalized vector, restarts
    from PC1–8 and 4 random directions; the restart with the lowest J on all 256 optimization frames is
    kept.
  - *Validation on a model with a known loudness dial (M2-SF, ω-only, seed 0; not part of Step A):*
    PC1 has J = 0.0061; started from PC2 the search reaches J = 0.0028 at σ_d 6.6 (PC1 7.2), cos² 0.83
    with PC1. Random starts stall at J ≈ 0.20. J falls mildly with step size (PC1: 0.0061 at full
    steps, 0.0042 at quarter steps), so the search can favour lower-variance directions; the chosen
    direction's σ_d is reported next to PC1's.
- **Scoring:** the full harness on the standard steering frames with entry 2's criteria plus the
  artefact criterion; loudness dial = all criteria and σ_shape ≤ 0.2, on both datasets.
- **Both-dials condition:** in the same seed, the loudness direction is a loudness dial and PC2 is a
  shape dial, both under all criteria; M2-SW meets it if this holds in at least 2 of 3 seeds, subject to
  the rendering check (which fired for M2-SW in Stage 2b). Also reported: the same under entry 2's
  criteria alone.

### Predictions

- **PA.1** Under entry 2's criteria alone, the loudness direction is a loudness dial in at least 2 of 3
  seeds. 65%.
- **PA.2** Under all criteria (with the artefact criterion), it is a loudness dial in at least 2 of 3
  seeds. 35%. (Strong loudness steering made checkerboards in M2-SF's loudness dial.)
- **PA.3** Under entry 2's criteria alone, both dials are valid in the same seed in at least 2 of 3
  seeds. 45%. (Under all criteria at most 1 seed can qualify; see the known consequence above.)
- **PA.4** The loudness direction lies mostly in the PC1–PC2 plane (cos² ≥ 0.8) in every seed. 70%.
- **PA.5** Its σ_shape is ≤ 0.2 on both datasets in every seed, whatever the other criteria. 60%.
- **PA.6** (retroactive) Stage 2's M2-SF loudness dials (with the ψ term) fail the artefact criterion in
  at least 2 of 3 seeds. 55%.

---

## Entry 7 — 2026-10-23 — Step A′: shape dial = top PC after projecting out the symmetry-loudness direction (no training)

Written before Step A′ is run. The investigator's specification: a new label-free shape rule, applied to
the three M2-SW checkpoints with the full harness and the artefact criterion, reported per seed next to
the PC2 rule, with the both-dials check under each rule. It was chosen after seeing results, so it
stays exploratory; a v3 would fix the rule in advance.

### Procedure (`dials/stepA_prime.py`)

1. Dial latents G (n × 8) of the v1 train frames; per-dimension mean μ and std s; Z = (G − μ) / s.
2. The Step A loudness direction d_L (raw dial space, unit; `results/build/stepA.json.gz`) in z-coordinates:
   l = (d_L / s) / ‖d_L / s‖ (a raw shift along d_L is a z-shift along d_L / s).
3. Projection: Z′ = Z − (Z l) lᵀ (Euclidean, in z-space).
4. u = the top principal direction of Z′ (centred SVD); u ⟂ l.
5. Raw steering direction: d_S = (s ⊙ u) / ‖s ⊙ u‖ (a z-shift along u is a raw shift along s ⊙ u);
   σ_d = the std of G·d_S over the train frames, as for every candidate.
6. Steering, criteria and frames as Step A: entry 2's criteria plus the artefact criterion, on the
   standard explore and matched-energy steering frames; a shape dial needs σ_shape ≥ 0.7 on both.
- **Both dials, per rule:** the Step A loudness direction is a loudness dial and the rule's direction
  (PC2, or d_S) is a shape dial, in the same seed, under all criteria (entry 2's alone also reported).
- Also reported: cos² of d_S with PC2 and with d_L (raw coordinates), σ_d, and the share of Z's
  variance that u explains.

### Predictions

- **PA′.1** d_S is a shape dial (all criteria) in at least 2 of 3 seeds. 35%.
- **PA′.2** d_S is shape-dominant (σ_shape ≥ 0.5 on both datasets) in every seed, whatever the other
  criteria. 55%.
- **PA′.3** d_S and PC2 point the same way (cos² ≥ 0.5) in every seed. 50%.
- **PA′.4** Both dials are valid in the same seed under the new rule (all criteria) in at least 2 of 3
  seeds. 20%. (Seed 2's loudness direction failed the cycle gain in Step A, so at most seeds 0 and 1.)
