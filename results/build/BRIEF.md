# [BUILD] Brief: the dial model (split latent: readable dials + rich content)

The investigator's brief, committed verbatim (2026-10-17), followed by the changes agreed before
any work started. Where the two differ, the agreed changes govern. Everything under this brief is
exploratory until a v3 pre-registration is committed; log lines carry the `[BUILD]` tag.

---

## Original brief (verbatim)

## Context

Two studies are done (`REPORT.md`, `REPORT_v2.md`). What we now know:

- **Confirmed (H1):** models trained to render a field, without whitening, make enstrophy (with energy) prominent in their slowest latent directions. E1 0.80–0.86 vs 0.38 for the untrained control, on sealed data and fresh seeds.
- **Direction only (H2):** VICReg lowers that prominence in 5 of 5 comparisons, but the pre-registered size failed for vorticity (+0.149 < 0.20). It's confounded with VICReg's 20–25× worse rendering.
- **Secondary (H3):** the render target shifts which quantity is prominent. Vorticity gives a pure enstrophy direction (a/b ≈ 1.0); streamfunction leans to energy (a/b ≈ 3.1). Seed-sensitive.
- **Secondary (H4):** VICReg on the pixel predictor removes about two-thirds of its lead over the JEPA.
- **Dissociation:** VICReg models decode log Z and log P *better* (higher probe R²) while showing them less. Information and prominence are different properties.
- **Prominent ≠ steerable:** exploration found the prominent direction acts as an overall amplitude (loudness) knob, not an enstrophy-specific control.

**The build goal.** A model whose latent has two parts:
1. a small **dial subspace** whose main directions can be found *without labels* and *steered*, with at least one dial that changes the field's fine-scale content independently of its overall loudness;
2. a larger **content subspace** that keeps rich, decodable information.

This is the foundation for a non-text scratchpad an LLM could look into and adjust. Read the whole brief, reply with a plan and compute estimate, push back on anything wrong, and wait for my go-ahead.

## The key idea, stated precisely

In 2D, log Z = log E + log(Z/E), and Z/E is the energy-weighted mean of k² (a spectral centroid). So every field has two coordinates:
- **amplitude** a = log E (loudness), and
- **shape** s = log(Z/E) (how much of the field lives at fine scales).

Scaling a field by c moves a and leaves s unchanged. Existing models already have a loudness knob. **The build succeeds if it also gets a shape knob: a label-free dial that moves s while a stays roughly fixed.** That is "turn enstrophy up at fixed energy."

Why two render heads should help: vorticity is the Laplacian of the streamfunction, so rendering vorticity weights fine scales (enstrophy) and rendering the streamfunction weights large scales (energy). Two heads that each own part of the dial subspace may produce two dials sensitive to different parts of the spectrum.

## Hard rules

1. **Everything here is exploratory** until a v3 pre-registration is committed. Label it in every file, figure and log line (`[BUILD]` tag in `results/LOG.md`).
2. **The v1 test sets and the v2 confirm set are retired.** Don't read them.
3. **Generate `confirm3/` now**, same recipe as `confirm/`, sealed with the same code guard (loads only under a flag set by a future Stage 5 script), with a disjointness test against every existing dataset. Exploration uses `explore/` and `matched-energy/` only.
4. **Predictions before each stage** in `results/build/PREDICTIONS.md`, never edited afterwards.
5. **Report everything:** every candidate dial, every seed, every variant, including failures and diverged runs.
6. **Reuse the infrastructure:** encoder, VICReg terms, plateau schedule, LR rule, probe code, bootstrap, slow-direction method. Only change what this brief says.
7. **Stop after each stage** with a 5-line summary and the key numbers. Ask before passing a gate, adding a variant, or exceeding budget.
8. **Wording:** a "dial" is a latent direction whose movement produces a measured change in the decoded field. No claims about a model "knowing" or "understanding" anything.

**Budget:** about 15 GPU-hours on the RTX 3060. Tell me up front if the plan doesn't fit.

## Measurements (build these first, in Stage 0)

**Steering harness.** For a latent direction d and a frame with latent z, decode z + α·σ_d·d for α ∈ {−2, −1.5, …, +2}, where σ_d is the train-set std along d. From each decoded field compute log E, log Z, log P, a and s. Report per direction:
- **Δa and Δs per unit α** (slopes, median over frames), and monotonicity (Spearman over α).
- **Shape selectivity** σ_shape = |Δs| / (|Δa| + |Δs|). A pure loudness dial gives ≈ 0; a pure shape dial gives ≈ 1.
- Measure on `explore/` frames and on `matched-energy/` frames (where every frame starts at the same E).

**Which decoded field to measure.** For two-head models, compute a, s from the vorticity head's output (get E through the inverse Laplacian), and separately from the streamfunction head's. Report the **head-consistency error** (vorticity head vs −∇² of the streamfunction head) at every α. Steering that makes the heads disagree isn't a real dial.

**Realism checks** for every steered field. A dial that produces garbage with a high Z doesn't count:
- **Cycle consistency:** encode(decode(z′)) vs z′, correlation along d, and drift in other directions;
- **Spectrum envelope:** fraction of shells whose energy stays within the 1st–99th percentile envelope of training spectra.
- For velocity outputs, divergence ≈ 0.

**Finding dials without labels.** Candidate dials are the top-4 PCA directions of the dial subspace (pooled, fitted on train). Which one is "loudness" and which is "shape" is decided afterwards by measurement, not by labels. That's how a scratchpad reader would find them too.

**Supervised upper bound** (reported, not a success criterion): steer along the ridge-probe direction for s, in the dial subspace and in the full latent. This tells us whether shape is steerable at all in a given latent, separate from whether it's findable.

## Stage 0 — Baselines on existing models (no training)

Run the harness on the existing plain vorticity AE, velocity AE and streamfunction AE, their VICReg variants, and the imaginer at k = 4 (3 seeds each), plus the untrained control.

Expected (write it in PREDICTIONS.md first): the top unsupervised direction is a loudness dial (σ_shape near 0), and no top-4 direction is a shape dial. The supervised upper bound decides what the build is for:
- If shape *is* steerable along the supervised direction, the build's job is to make it **findable** (prominent).
- If it isn't, the build's job is to make it **exist** as a steerable direction at all.

After Stage 0, propose numeric thresholds for "shape dial", "loudness dial" and "realistic", based on the baseline distributions. My starting proposal: shape dial σ_shape ≥ 0.7; loudness dial σ_shape ≤ 0.2; realism means cycle correlation ≥ 0.9 and ≥ 90% of shells inside the envelope across α ∈ [−2, 2]. Commit the final thresholds to PREDICTIONS.md **before any Stage 1 training.**

## Stage 1 — The split model (M1)

- **Encoder:** same CNN, same input (velocity u, v), same 8×8×128 spatial map.
- **Channel split:**
  - 4 channels **dial-ω**, 4 channels **dial-ψ**, 120 channels **content**;
  - pooled latent = spatial mean, so 4 + 4 + 120 = 128 dims, the same size as before.
- **Two render heads**, the same decoder design as the existing AEs:
  - **vorticity head** decodes from [dial-ω, content] channels, target ω;
  - **streamfunction head** decodes from [dial-ψ, content] channels, target ψ.
  - Each dial group gets rendering gradients from one head only.
- **Regularization:** VICReg variance and covariance terms on the pooled **content** dims only, at the weights used for the Stage 2 AE+VICReg models. **No whitening on the dial dims.**
- **Training:** loss balance between the heads set so each head's loss relative to its plain-AE counterpart is comparable (state the rule in the plan). Same plateau schedule and LR rule as before. 3 seeds.

**Readouts:**
- the steering harness on the top-4 dial directions;
- E1-style prominence of enstrophy and energy in the dial subspace vs the content subspace;
- linear-probe R² for log Z, log P, s and log ν from the content subspace (does it keep the regularized model's information advantage?);
- each head's rendering loss relative to the plain AE with the same target.

**Rendering check:** if either head's loss is more than 2× its plain-AE counterpart, or the dial dims collapse (near-zero variance), stop and tell me. That's the trigger for M2.

## Stage 2 — Variants (only what Stage 1 calls for, ask first)

- **M2, separate branches:** a dial encoder branch and a content encoder branch with no shared trunk, same total parameters. Use it if M1's rendering or dials suffer from the shared trunk (VICReg degrading rendering through shared weights).
- **M3, decorrelated dial groups:** M1 plus a cross-covariance penalty between the dial-ω and dial-ψ pooled vectors, only *between* the groups, with no whitening *within* them. Use it if M1's two groups learn the same loudness direction.

3 seeds each. Same readouts.

## Stage 3 — The H2 dose curve

On the best of M1–M3: apply VICReg variance and covariance terms to the dial dims at weights {0, 0.1, 0.3, 1.0} × the content weight, 3 seeds per dose (0 is the existing model). Plot dial prominence, σ_shape of the best dial, realism and rendering loss against dose. This turns H2's failed threshold into a design curve: how much whitening the dials tolerate before they stop being readable.

## Stage 4 — Build report

Write `results/build/BUILD_REPORT.md` in plain language:
- Does a label-free shape dial exist, how selective is it, is it realistic?
- Does the content subspace keep its information advantage?
- What does the dose curve say?
- Steering figures: decoded field strips along each dial, with spectra.

**If** a model meets the Stage 0 thresholds on both a loudness dial and a shape dial, draft `PREREGISTRATION_v3.md`: which model, which dial (identified by a fixed label-free rule), which thresholds, which controls (plain AE, untrained). Don't open `confirm3/`; I'll approve Stage 5 separately.

**If not:** say so plainly. Include whether the supervised upper bound showed shape was steerable at all, since that decides whether the next attempt should make the dial findable or make it exist.

## Threats to keep in view (put them in your plan)

- Steering off the data manifold: realism checks and cycle consistency exist for this.
- A "shape dial" that's really a head-disagreement artefact: head-consistency error exists for this.
- Picking the best of 4 candidate dials is a forking path. Fine for exploration; v3 must fix the identification rule in advance.
- H3 was seed-sensitive: the two-head bet may simply not separate the dials.
- 2D flow, one small CNN, 3 seeds.

---

## Agreed changes (2026-10-17, before any work)

1. **Two Stage 1 models, 3 seeds each.** **M1-spec** exactly as written (spatial dial channels; the
   control). **M1-G**: the 4 + 4 dial channels are spatially pooled into an 8-dim global vector that
   reaches each head through per-decoder-stage FiLM (gain and bias; dial-ω → vorticity head, dial-ψ →
   streamfunction head); the 120 content channels are RMS-normalized per sample, so loudness must
   flow through the dials. **Wording rule:** M1-G's loudness dial exists by construction and is
   reported as a design property, not a finding; the finding is whether a label-free shape dial
   emerges.
2. **Stage 1 prediction to record (investigator):** if a shape dial emerges in M1-G, it is (or loads
   mainly on) the direction contrasting the dial-ω group with the dial-ψ group, since with loudness
   forced into the dials their difference is the spectral shape (ω = −∇²ψ).
3. **M1-G content readout:** decodability of s, log(P/Z) and log ν from content; of log Z and log E
   from the full latent (dials + content).
4. **If M1-G produces a shape dial,** Stage 2 starts with a FiLM-without-normalization ablation
   (3 seeds) before M2/M3, to attribute the result.
5. **Recipe:** every build model uses the autoencoder recipe (cosine decay, gradient-norm clipping
   1.0, 30 epochs, peak lr 1e-3), not the plateau schedule.
6. **Rendering trigger:** stop and report if a head is worse than 2× its VICReg-autoencoder
   counterpart, OR its normalized error exceeds 0.02, OR it keeps < 50% of the true small-scale
   (k ≥ 10) energy.
7. **Effect-size floor:** a direction counts as a dial only if it moves a + s by ≥ 0.1 log-units per
   unit α (median over frames), with Spearman ≥ 0.9, and passes realism. Stage 0 may raise this floor
   but not lower it. Final thresholds are committed before any Stage 1 training.
8. **Harness details:** for cycle consistency the vorticity field is converted to velocity (inverse
   Laplacian) before re-encoding (vorticity head; the streamfunction head is reported too); the
   slow-direction method's PCA step uses min(32, dims) components (8 in the dial subspace).
9. **Seals:** each sealed set names its own pre-registration file; `confirm3` opens only under a
   committed `PREREGISTRATION_v3.md`; `confirm_v2` stays sealed and retired.
10. **Budget:** 15 GPU-hours.
