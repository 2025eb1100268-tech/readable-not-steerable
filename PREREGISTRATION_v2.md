# Pre-registration v2: confirmation of the exploratory line

Status: **DRAFT for the investigator's review.** It becomes binding when committed. The code
(`data/seal.py`) refuses to open the sealed `confirm` set until this file is committed unmodified,
and every later change goes into a dated AMENDMENTS section.

Wording rule (from the exploratory brief): the statistics below say which conserved quantities are
**prominent in a latent** (its slowest unsupervised directions). They do not say that a model
"discovered conservation", and they are not measures of what a model knows.

## 1. Background

The first study (`PREREGISTRATION.md`, `REPORT.md`) ended *not supported*. An exploratory line
followed (`results/explore/BRIEF.md`, `PREDICTIONS.md` entries 1–7, `STAGE2_SUMMARY.md`), using only
fresh data (`explore`, `matched-energy`); the original test sets stayed retired. On `explore` it
suggested:

- models trained to **render** a field with an **unregularized** latent make enstrophy prominent in
  their slow directions, whether they render the present (autoencoders: E1 0.82 for velocity, 0.81
  for vorticity) or the future (pixel predictor, 0.80 at k = 4), against 0.42 for an untrained
  encoder;
- adding the JEPA's **VICReg** variance/covariance terms to the autoencoders removes this (velocity
  0.82 → 0.42, vorticity 0.81 → 0.52) while raising linear decodability.

These patterns were found by exploring. This pre-registration tests them once on data no model or
analysis has seen, **with newly trained models** (fresh seeds) for the primary hypotheses, so that a
pass replicates both the data and the training, not just the data.

## 2. Primary hypotheses (they decide the verdict)

**E1** (as in Stage 2): for each model, the slow directions are fitted on the pooled latents of the
v1 train+val inviscid runs and scored on the confirm inviscid trajectories; E1 is the mean over the
model's seeds of the largest |Pearson r|, among the top-3 slow directions, between a direction's
trajectory-mean projection and the trajectory's log Z₀ (enstrophy of its first frame). CIs: 1,000
paired trajectory-bootstrap resamples (`probes/criteria.py`), seed 0.

**Fresh models** (trained 2026-10-17 after this design was fixed, evaluated only on `confirm`): the
plain and VICReg autoencoders rendering velocity and rendering vorticity, seeds **3, 4, 5**, with the
frozen Stage 2 autoencoder recipe (v1 train frames t ≥ 1, batch 256, AdamW, weight decay 0.05, bf16,
peak lr 1e-3, 5% warmup then cosine over 30 epochs, gradient-norm clipping 1.0; VICReg variants
25·MSE + 25·var + 1·cov on the pooled latent). The untrained control for them is the shared encoder
at its Kaiming initialization with the same seeds 3, 4, 5.

- **H1 (rendering).** For **each** of the fresh plain velocity autoencoder and the fresh plain
  vorticity autoencoder: E1 ≥ **0.70**, and E1 − E1(untrained, seeds 3–5) ≥ **0.25** with the 95% CI
  of that difference excluding 0. H1 holds if both pass.
- **H2 (regularization).** For **each** of the fresh pairs (plain velocity AE, velocity AE + VICReg)
  and (plain vorticity AE, vorticity AE + VICReg): E1(plain) − E1(+VICReg) ≥ **0.20** with the 95% CI
  of that difference excluding 0. H2 holds if both pass.

**Verdict:** *supported* if H1 and H2 both hold; *partially supported* if exactly one holds; *not
supported* if neither holds.

## 3. Secondary hypotheses (rules stated, not part of the verdict)

- **H3 (render target; Stage 2 seeds 0–2).** E4 ratio a/b (the enstrophy-tracking direction
  regressed on log E₀ and log(Z₀/E₀); a/b = 1 means the direction is a function of log Z₀ alone):
  the plain vorticity autoencoder's seed-mean ratio is closer to 1 on a log scale than the plain
  streamfunction autoencoder's, and the streamfunction autoencoder's ratio is > 2. Secondary because
  the streamfunction autoencoder was seed-sensitive on `explore`.
- **H4 (back to the original question).** The pixel predictor at k = 4 **with** the VICReg terms
  (new model, seeds 0–2; exactly the Stage 2 k = 4 imaginer's recipe — common viscous anchors,
  plateau schedule, lr 1e-3, no clipping — plus 25·var + 1·cov on the pooled latent, with the pixel
  MSE weighted 25) has E1 at least **0.20** below the plain k = 4 imaginer (Stage 2 seeds 0–2), 95% CI
  excluding 0. If it holds, the regularization account also covers the original imaginer-vs-summarizer
  gap. Never evaluated on any data before this run.

## 4. Reported without a rule

E1 with its CI for every group; E4 for every group; the H1 and H2 rules applied to the Stage 2 seeds
0–2 (same data, existing models); the imaginer (k = 4) − summarizer (k = 4) E1 difference with its
CI; the linear-probe R² of log Z and log P (S1 partialling) for every group, to show whether VICReg
again raises decodability while lowering prominence.

## 5. What is frozen

- **Data:** `data/generated/confirm_v2` (1,005 trajectories, 378 inviscid; generated 2026-10-15
  with the explore recipe and a new seed; sealed and never loaded since). Its SHA-256 checksums are
  in `configs/explore/stage3_confirm.yaml` and are verified before unsealing.
- **Models:** every checkpoint used (Stage 2 and the fresh Stage 3 runs) is identified by SHA-256 in
  the same config and verified before use; nothing is retrained after this file is committed. The
  fresh models were never evaluated before the confirmation run (their training logs contain only
  label-free validation losses).
- **Exclusions:** a pixel-family or autoencoder run whose final normalized validation loss is ≥ 1
  (worse than predicting zero / copying) is diverged and excluded. A group left with fewer than 2
  seeds makes every hypothesis that uses it **not evaluable**, which counts as not holding.
- **Method and code:** pooled latent in float32; `probes/slow_features.py` (z-score → PCA 32 →
  Ledoit–Wolf within/between eigenproblem); `probes/evaluate.py`, `probes/criteria.py`,
  `explore/stage2_eval.py` (E4); the single script `explore/stage3_confirm.py` with
  `configs/explore/stage3_confirm.yaml` (written by `explore/stage3_config.py`).

## 6. Procedure

1. Commit this file together with the final config (with all checksums) and the script.
2. Run `python -m explore.stage3_confirm --config configs/explore/stage3_confirm.yaml` once. It
   verifies all checksums, unseals the confirm set (logged), evaluates, and writes
   `results/explore/stage3/`. A rerun is allowed only after a crash, logged with the reason.
3. Write `REPORT_v2.md` with the verdict against §2, the secondary results, everything in §4, and the
   threats below. Nothing in §2–§5 changes after unsealing.

## 7. Threats and limits

- The confirm set comes from the same simulator and initial-condition distribution as `explore`.
  A pass shows the pattern reproduces on fresh trajectories and fresh training runs, not that it
  generalizes to other flows or architectures.
- Three seeds per model; the bootstrap CIs resample trajectories, not seeds.
- The hypotheses were chosen after exploration (many comparisons, some predictions failed); that is
  why they are tested here, once, on sealed data and new training runs. The conjunction rules (every
  listed model or pair must pass) make each hypothesis stricter, so no further multiplicity
  correction is applied.
- The autoencoders used gradient clipping and a cosine schedule; the imaginer, summarizer and the H4
  model did not (pixel models without clipping diverged often in Stage 2, so H4 may end up not
  evaluable).
- E1 measures prominence under this specific unsupervised method; VICReg changes the latent's
  variance allocation, which this method is sensitive to by design.
- **VICReg also costs rendering quality** (training losses only, known before unsealing): the
  VICReg autoencoders reconstruct about 25× worse than the plain ones (normalized MSE 0.003–0.011 vs
  0.0001–0.0013), and the pixel predictor + VICReg predicts about 20× worse than the plain imaginer
  (val loss / copy 0.17–0.23 vs about 0.011). A drop in E1 under H2 or H4 may therefore reflect worse
  rendering as well as latent geometry; this run cannot separate the two.
- Training outcome of the fresh models (known before unsealing): none of the 12 autoencoders or 3
  pixel + VICReg runs diverged or collapsed.
- Dry run (code check, confirm still sealed, 2026-10-17): the script was run on `explore` with a
  stand-in config that uses only Stage 2 models (the fresh groups replaced by seeds 0–2, the H4 model
  by the summarizer), so no fresh model was evaluated. It exercised every hypothesis and report path
  and reproduced the Stage 2 values exactly. The divergence-exclusion rule was checked separately on
  known diverged and healthy runs.

## AMENDMENTS
