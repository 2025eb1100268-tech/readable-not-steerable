# Pre-registration v3: confirmation of the dial build

Status: **DRAFT for the investigator's review.** It becomes binding when committed. The code
(`data/seal.py`) refuses to open the two sealed sets named below until this file is committed
unmodified; every later change goes into a dated AMENDMENTS section.

Wording rule (from the build brief): a **dial** is a direction in a model's latent space whose movement
produces a measured change in the decoded field. A **loudness dial** changes a = log E and leaves
s = log(Z/E) nearly unchanged; a **shape dial** changes s at nearly fixed a. Nothing here says a model
"knows" or "understands" anything. That loudness reaches the decoder only through the dials is a design
property of these models (their content branch cannot see overall scale); the tested claim is that
pre-registered, label-free rules **find** both dials and that both **pass every realism criterion**,
reliably across fresh training runs.

## 1. Background

The build line (`results/build/BRIEF.md`, `PREDICTIONS.md` entries 1–10, `BUILD_REPORT.md`) was
exploratory: it used only `explore` and the exploration matched-energy set, with the v1 test sets and
`confirm_v2` retired. Its final design is a two-branch autoencoder (scale-free content branch, 8-number
dial branch, one FiLM decoder) whose content input carries random spectral tilt/curvature noise
(dose τ = 0.2) and whose decoder upsamples by resize-convolution. In exploration (Step D, 3 seeds):

- both dials valid under every committed criterion in **3 of 3** seeds with the loudness rule and shape
  rule fixed here (symmetry loudness, loudness-null shape), and in **2 of 3** with PC2 as the shape rule
  (Step C, τ = 0.5: 1 of 3);
- the rules, the dose, the decoder, the artefact criterion's threshold and the zero-mean handling were
  all chosen during exploration, several of them after seeing results.

That is why it is tested here, once, with **newly trained models** (fresh seeds, never evaluated) on
**two sealed sets** no model or analysis has seen.

## 2. Primary hypothesis and verdict

**H1.** For each fresh main-model seed, the loudness direction and the shape direction identified by the
rules in §5 are both **valid**: the loudness direction is a loudness dial and the shape direction is a
shape dial under every criterion of §5, on **both** `confirm3` **and** `matched-energy-confirm`.

| Outcome | Rule (5 fresh main-model seeds) | What it means (fixed now) |
|---|---|---|
| **Pass** | both dials valid in **≥ 4 of 5** seeds | *Supported:* the design reliably yields a label-free loudness dial and a label-free shape dial that pass every realism criterion on unseen data; the exploration result replicates on fresh training runs. |
| **Partial** | **3 of 5** | *Partially supported:* the design produces both dials in most, but not reliably in all, training runs; it would need a more robust identification rule or training before use. |
| **Fail** | **≤ 2 of 5** | *Not supported:* the exploration result does not replicate on fresh seeds and sealed data; the build's 3-of-3 was not robust. |

**Controls (their rules are part of the verdict statement):**
- **τ = 0 control** (same model without the content noise): a shape dial in **≤ 1 of 3** seeds.
  Pre-registered expectation, reported but not a rule: a loudness dial present, no shape dial.
- **Untrained control** (the main architecture at initialization, randomly initialized FiLM): a shape
  dial in **0 of 3** seeds.

If a control breaks its rule, the verdict says so explicitly, with the text fixed now:
- **τ = 0:** "Control failure: the no-noise control also shows a shape dial in n of 3 seeds (rule: at
  most 1), so a shape dial in the main model cannot be attributed to the shape-noise content input."
- **Untrained:** "Control failure: the identification rules find a shape dial in n of 3 untrained
  networks (rule: none), so a shape dial found by these rules is not evidence of anything learned."

The Stage 5 script applies these rules and writes the verdict mechanically; the texts are in
`configs/build/v3_confirm.yaml`.

## 3. Secondary (reported with their record; not part of the verdict)

- **PC2 as the shape rule:** shape dial and both-dial counts per group with PC2 in place of the
  loudness-null direction. Exploration record: Step C 1 of 3, Step D 2 of 3.
- Per group: loudness dials alone, shape dials alone, and every metric of every direction on both sets.
- **Rendering cost (not a gate, stated in advance):** vorticity val normalized MSE on v1 val (from
  training) and on the 256 `confirm3` steering frames (mean removed), for the main model, the τ = 0
  control and, on the same frames, the Stage 2 VICReg vorticity autoencoders (seeds 0–2; 0.0105 on v1
  val). In exploration the main design was 0.022 on v1 val, 2.1× the VICReg autoencoder; the build
  brief's rendering trigger would fire, and it is not applied here.
- **Spatial-mean diagnostic:** before the mean is removed (§5), the fraction of frames with more than
  10% of ω² in the spatial mean, per direction, set and α.

## 4. Data

- **`confirm3`** (`data/generated/confirm3`): 1,007 trajectories (378 inviscid), generated 2026-10-17
  with the `explore` recipe and a new seed, sealed for this file, never loaded.
- **`matched-energy-confirm`** (`data/generated/matched_energy_confirm`): 300 inviscid trajectories, all
  at the same initial energy, generated 2026-10-27 with the exploration matched-energy recipe and a new
  seed, sealed for this file, never loaded; no seeds or initial fields shared with any other set
  (`tests/test_explore_data.py`).
- SHA-256 of every file of both sets is in `configs/build/v3_confirm.yaml` and is verified before
  unsealing. Each set is opened exactly once, by the Stage 5 script.
- **Steering frames:** 256 frames from each set (t ≥ 1 and t + 4 inside the trajectory; drawn with seed 0,
  as in the build). Everything used to *identify* dials comes from exploration data: PCA and σ_d from v1
  train frames; the symmetry search and the loudness gradient on the build's 256 `explore` optimization
  frames (seed 1, disjoint from the build's steering frames); the spectral envelope from 4,096 v1 train
  frames.

## 5. Models, rules and criteria (frozen)

**Models** (`configs/build/v3/`, written by `dials/v3_configs.py` from Step D's config):
- **Main:** two-branch model, shape-noise content input τ = 0.2 (log-amplitude tilt/curvature gains,
  σ from the data's own coefficients; k > 21 zeroed; hash-fixed draws in evaluation), resize-convolution
  decoder, ω-only loss, the autoencoder recipe (v1 train, batch 256, AdamW lr 1e-3, weight decay 0.05,
  5% warmup then cosine over 30 epochs, gradient clipping 1.0, bf16). **Seeds 100–104.**
- **τ = 0 control:** identical with τ = 0. **Seeds 200–202.**
- **Untrained control:** the main architecture at its initialization for **seeds 300–302**, exactly as
  training would build it, with the FiLM layers re-initialized by PyTorch's default `nn.Linear`
  initialization (weights and biases uniform on ±1/√8) under seed + 2000; no training step
  (`dials/v3_untrained.py`).
- None of these seeds has been trained or evaluated before. Training logs contain only label-free
  validation losses. A main or τ = 0 run whose final vorticity val normalized MSE is ≥ 1 is diverged and
  counts as a seed without dials; nothing is retrained.

**Identification rules** (exactly as in the build; no labels):
- **Loudness (symmetry):** the unit dial direction d minimizing J(d), the fraction of the steered
  change not explained by rescaling each decoded field (`dials/stepA.py`, PREDICTIONS entry 6), over
  α ∈ {−2, −1.5, …, 2} \ {0}; Adam (lr 0.05, 150 steps, batches of 32), restarts from PC1–8 and 4 random
  directions (seed 0); the lowest J on all optimization frames is kept.
- **Shape (loudness-null PCA):** z-score the dial vector; g = the frame-mean finite-difference gradient
  of the decoded field's a = log E with respect to it (±0.5 z-units per coordinate); restrict to the
  subspace orthogonal to g; take the top principal direction of the projected data; map to a raw
  direction (`dials/stepB.py`, entry 8).
- **PC2 (secondary):** the second principal direction of the dial vector (v1 train frames).
- σ_d for every direction = the std of the train frames' dial vectors along it.
- These rules are computed on the decoder's raw output, as in the build.

**Scoring:**
- **Zero-mean:** the spatial mean of every decoded vorticity field is removed before **every scoring
  measurement** (fixed post-processing, no retraining); the pre-removal share is reported (§3). In Step C,
  removing the mean changed no classification.
- Each direction is steered by α·σ_d (α = −2 … 2 in steps of 0.5) with the harness of the build
  (`dials/harness.py`).
- **A dial** (per set) needs every criterion of PREDICTIONS entry 2 plus the artefact criterion of
  entry 6:
  - effect |Δa| + |Δs| ≥ 0.2 per unit α (median over frames);
  - monotonicity ≥ 0.9;
  - cycle correlation ≥ 0.9 with gain in [0.5, 2];
  - shared-pattern fraction ≤ 0.5;
  - envelope within 0.05 of an ideal dial's at every α;
  - at most 1% of frames with more than 1% of power outside the dealiasing band at some α.
- **Loudness dial:** σ_shape ≤ 0.2. **Shape dial:** σ_shape ≥ 0.7.
- **Valid** means all of the above on **both** sets.

**Code:** `dials/` and `train/train.py` as committed with this file; the single Stage 5 script is
`dials/v3_confirm.py` with `configs/build/v3_confirm.yaml`.

## 6. Procedure

1. Commit this file with the configs and scripts above.
2. Train the 5 main and 3 τ = 0 models; build the 3 untrained controls. No evaluation of any of them.
3. `python -m dials.v3_confirm --config configs/build/v3_confirm.yaml --freeze` writes every
   checkpoint's SHA-256 (with its final val loss) to `configs/build/v3_frozen.json`, which is committed.
4. `python -m dials.v3_confirm --config configs/build/v3_confirm.yaml`, once. It:
   - verifies that the frozen file is committed and unmodified, every checkpoint and every data file;
   - unseals both sets (logged);
   - identifies, scores and applies §2;
   - writes `results/build/v3/`.

   A rerun is allowed only after a crash, with `--rerun-reason`, which is logged.
5. Write `REPORT_v3.md` with the verdict, §3, and the threats below. Nothing in §2–§5 changes after
   unsealing.

**Dry run (code check, before this file was committed, 2026-10-27):** the Stage 5 script was run on
stand-ins (one Step D model as "main", one Step B τ = 0 model, one throwaway random-FiLM untrained model;
the exploration sets; 64 frames; a shortened search). It exercised verification-free loading, both
rules, scoring with the mean removed, the rendering readout, the verdict and the figure. The Step D
stand-in had both dials, the τ = 0 and untrained stand-ins had none. No sealed data was opened and no v3
model existed.

## 7. Threats and limits

- **Same distributions.** `confirm3` and `matched-energy-confirm` come from the same simulator and
  initial-condition distributions as the exploration sets. A pass shows the result reproduces on fresh
  trajectories and fresh training runs, not that it generalizes to other flows or architectures.
- **Rules from exploration.** The identification rules and the artefact threshold were chosen during
  exploration, some after seeing results; that is what this test is for. The pass rule (4 of 5, both
  sets, every criterion) is strict, and no multiplicity correction is applied.
- **Partly by design.** The dials are partly design properties: loudness cannot pass through the content
  branch, and the content noise removes most of the content's shape signal (calibrated content shape
  R² 0.61). The test is whether label-free rules find both dials reliably and whether they pass realism.
  It is not whether a network invents them unaided.
- **Rendering is worse** than the regularized and plain autoencoders; it is reported, not gated.
- **Tail effects** that the criteria test only in aggregate (spatial-mean offsets) are reported as a
  diagnostic.
- **Shape is energy-weighted.** s weights scales by energy, so changes confined to fine scales register
  weakly in it.
- **Seed counts:** 5 main seeds and 3 per control.

## 8. Budget

10 GPU-hours:

| Item | GPU-hours |
|---|---|
| Training (8 runs, about 48 min each) | ≈ 6.4 |
| Stage 5 run (11 models) | ≈ 2.0 |
| Untrained controls and the freeze | < 0.1 |

## 9. Expectations (mine, for calibration; not part of the verdict)

- **Main outcome:** pass 60%, partial 20%, fail 20%.
- **τ = 0 control:** within its rule 85%; it shows a loudness dial in ≥ 2 of 3 seeds 70%.
- **Untrained control:** within its rule 90%.

## AMENDMENTS
