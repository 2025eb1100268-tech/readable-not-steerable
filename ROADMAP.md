# Roadmap: a model that lives up to the original thesis (written 2026-11-10)

This file is the plan for the next line of work, written so that anyone (the investigator, or a model
starting cold) can pick it up at any stage. Nothing in it has been run. Everything marked **hypothesis**
is untested. Update the status table at the bottom as stages finish.

## 1. Where things stand

The original thesis: *a world model trained only to predict 2D Navier–Stokes flow organizes its latent
around the quantities used to describe fluids.* Three pre-registered studies tested it (paper:
`paper/out/preprint.pdf`; reports: `REPORT.md`, `REPORT_v2.md`, `REPORT_v3.md`, `AUTOPSY.md`):

| study | what it showed | consequence for a new model |
|---|---|---|
| 1 | The JEPA failed its criteria. A JEPA trained on non-physical linear dynamics and a static model that never sees time read fluid quantities as well or better. | **Prediction of real dynamics has so far added nothing measurable.** Any new claim must beat these two controls, not only an untrained network. |
| 2 | Rendering velocity or vorticity without a whitening regularizer makes enstrophy prominent; VICReg removes prominence while raising decodability. | Keep the code that should be prominent **unregularized**; regularize only the rest. |
| 3 | Routing gives steerable dials: loudness in 8 of 8 models, shape in 3 of 5 seeds (bar was 4). | Routing works; the identification rule is the weak point (`AUTOPSY.md`). |

So the three properties (decodable, prominent, steerable) can each be produced, but **none of them has
been shown to come from the dynamics**. That is the gap this roadmap targets.

## 2. The thesis, restated so that it can be tested

> A model trained to predict Navier–Stokes flow has dials for physical quantities **whose effect on the
> predicted future matches what the equations do**, and a model trained on non-physical dynamics does not.

The new ingredient is a **counterfactual test with the solver**. Steering a dial and decoding a frame *k*
steps ahead gives a predicted future. The solver can produce the true future for the same change. For
example:

- **Loudness at k > 0 is not a rescaling.** Navier–Stokes is nonlinear: a field with twice the amplitude
  evolves twice as fast (the symmetry is u → cu, t → t/c, ν → cν). A model that learned the dynamics must
  render a louder *and further evolved* future. A model trained on the linear non-physical dynamics should
  only rescale. This separates the two by construction, which Study 1's probes could not do.
- **Viscosity.** Steering a ν dial should change how much the predicted field decays, as the solver does
  when ν changes (dE/dt = −2νZ).

Read-outs and slow directions stay as supporting measures. The counterfactual test is the one that makes
"learned the physics" a checkable statement.

## 3. Candidate design (hypothesis): a predictive dial model

Start from the Study 3 model (`dials/split.py`, `BranchHeads`; config `configs/build/v3/b6_main_s100.yaml`)
and change one thing: **it renders ω at t + k instead of at t.**

- **Dial encoder:** reads the raw frames (t − 1, t), pooled, d = 8, no regularizer. Two frames, so that the
  decay rate (ν) is visible.
- **Content encoder:** as in Study 3 (frame t, RMS-normalized, shape noise τ = 0.2, d = 120, VICReg terms).
- **Decoder:** the resize-convolution decoder with FiLM from the dials; add the horizon k as an extra FiLM
  input. Sample k from {0, 1, 4, 8} per batch.
- **Loss:** as in Study 3, on ω(t + k).

Why this might work: the future's loudness and coarse shape can reach the decoder only through the dials,
and at k > 0 getting them right requires the decay rate and the nonlinear evolution speed. At k = 0 it is
exactly the Study 3 model, which gives a sanity anchor.

Known risks:
- **Rendering quality.** A small decoder may predict poorly at k = 8; the pixel predictor managed k = 4
  (validation loss 0.035 × the copy baseline).
- **Dial capacity.** The dials may carry loudness and shape and leave evolution to the content path. Then
  the counterfactual test fails, which is a clean negative result.
- **Two-frame dial input** changes the dial code's statistics; the Study 3 rules may need re-checking.

## 4. Stages and gates

Each stage ends with a committed report in `results/next/` and a go/no-go from the investigator. Estimated
compute on the RTX 3060; one main model costs about 0.77 GPU-hours (30 epochs).

| stage | work | gate to pass | GPU-h |
|---|---|---|---|
| **0. Brief** | Investigator writes `results/next/BRIEF.md` (question, budget, controls). Commit predictions with probabilities (`results/next/PREDICTIONS.md`, entry 1). | Brief committed. | 0 |
| **1. Free checks** (no training, exploration data only) | (a) Test **loudness-null PCA without z-scoring** on the 11 v3 checkpoints with `dials/autopsy.py`'s harness: does it fix seed 102 without creating dials in the 6 controls? (b) Probe ν from the v3 dial code and from two-frame inputs. | One shape rule chosen and frozen for everything below. | ≈ 2 |
| **2. Counterfactual harness** | New module `dials/counterfactual.py`: from a stored frame, run the solver (`solver/ns2d.py`) with rescaled amplitude or changed ν to get the true ω(t + k); score a steered prediction against it and against two baselines, **rescale-only** and **no change**. Unit tests against the scaling symmetry. | Tests pass; the baselines score as expected on true data. | ≈ 0.5 |
| **3. Build** (exploration) | Train the predictive dial model, 3 seeds. Score: Study 3 dial criteria at k = 0; rendering error per k; counterfactual fidelity per dial and k. Iterate at most 3 design changes, each with a prediction entry first. | **Kill gate:** steered predictions beat the rescale-only baseline at k ≥ 4 in ≥ 2 of 3 seeds. If not, stop and report the negative result. | ≈ 8 |
| **4. Controls** (exploration) | Same model trained on the non-physical data (`configs/phase2b_nonphys_data.yaml`), a k = 0 only model, τ = 0, untrained. | Controls fail the counterfactual test where the hypothesis says they should. If the non-physical model passes, the thesis is not supported; report it. | ≈ 5 |
| **5. Pre-registration** | Freeze the design and one identification rule. Generate and seal new sets (`confirm4`, `matched-energy-confirm2`; `data/seal.py`). Write `PREREGISTRATION_v4.md` with thresholds set from Stages 3–4, ≥ 10 fresh main seeds, 3 seeds per control, and diagnostics reported but not gated (`AUTOPSY.md`, next-step list). Investigator approves before commit. | Committed before any sealed set is opened. | ≈ 1 |
| **6. Confirmation** | Train fresh seeds, freeze checkpoint hashes, open each sealed set once with a frozen script, write `REPORT_v4.md`. | — | ≈ 18 |

Total ≈ 35 GPU-hours. **By 16 November**, Stages 0–3 are realistic (about 10 GPU-hours and the new
harness); Stage 4 if the kill gate passes early. Stages 5–6 can be done later by following this file.

## 5. Rules that carry over (do not skip)

- **Spent data:** `confirm_v2`, `confirm3`, `matched-energy-confirm` and the v1 test splits are never
  opened again. Every new script gets the path guard (see `results/paper/figures/_common.py`).
- **Exploration data** for all build work: `data/generated/explore_v2`, `matched_energy_v2`, v1 train/val.
- **Before each stage:** commit predictions with probabilities; never edit them afterwards.
- **Every run** is logged in `results/LOG.md`; every outcome is committed, including failures.
- **Criteria** are the Study 3 ones (`results/build/PREDICTIONS.md` entries 2 and 6). Nothing is relaxed
  after seeing results. New criteria (counterfactual fidelity) are fixed on exploration data and committed
  before confirmation.
- **Fresh seeds** for every confirmation; one identification rule, fixed in advance.
- **The investigator approves** each gate, each pre-registration and each sealed-set opening.

## 6. Practical notes for whoever continues

- **Environment:** Windows, RTX 3060 (12 GB), `.venv\Scripts\python.exe`, run from the repo root.
- **Read first:** `README.md`, `AUTOPSY.md`, `results/build/BUILD_REPORT.md`, `PREREGISTRATION_v3.md`,
  `results/paper/CLAIMS.md` (what may and may not be claimed).
- **Code to reuse:** `dials/split.py` (model), `dials/harness.py` (dial criteria), `dials/stepA.py`
  (symmetry rule), `dials/v3_confirm.py` (frozen evaluation pattern), `train/train.py` (training; already
  supports predicting k frames ahead), `data/nonphysical.py`.
- **Known traps:** PixelShuffle decoders produce grid artefacts under steering (use the resize decoder);
  steered fields can gain a spatial mean (remove it before scoring, as in v3); a non-deterministic GPU
  median once crashed start-up (compute medians on CPU); Git Bash heredocs strip backslashes (write scripts
  as files).
- **Paper:** a positive or negative result here is a follow-up paper, not an edit to the posted one. The
  build tooling in `paper/` can be reused (`paper/BUILD_REPORT.md`).

## 7. Status

| stage | status | commit | notes |
|---|---|---|---|
| 0. Brief | not started | | |
| 1. Free checks | not started | | |
| 2. Counterfactual harness | not started | | |
| 3. Build | not started | | |
| 4. Controls | not started | | |
| 5. Pre-registration | not started | | |
| 6. Confirmation | not started | | |
