# [BUILD] Report v3: confirmation of the dial build

2026-10-29. Pre-registration: `PREREGISTRATION_v3.md` (committed `781b2ad` on 2026-10-27, before any v3
model existed; no amendments). Checkpoints frozen: `configs/build/v3_frozen.json` (`fc3b703`). Single
confirmation run: `results/build/v3/confirm.md`, `confirm.json.gz`, `confirm_dials.png`. The run took
16:52–18:48, ran once with no rerun, and its outputs were committed unchanged in `258d6af`. The build
that led here is in `results/build/BUILD_REPORT.md`.

Wording rule: a **dial** is a latent direction whose movement produces a measured change in the
decoded field. A **loudness dial** changes a = log E at nearly fixed s = log(Z/E). A **shape dial**
changes s at nearly fixed a. Nothing here says a model "knows" or "understands" anything.

## Bottom line

- **Verdict: partially supported.** Both dials are valid on both sealed sets in **3 of 5** fresh
  main-model seeds (seeds 100, 101 and 103); the pass rule needed 4. By the text fixed in advance, this
  means *the design produces both dials in most, but not reliably in all, training runs; it would need a
  more robust identification rule or training before use.*
- **The loudness half replicates completely.** The symmetry rule finds a loudness dial in **5 of 5**
  main seeds and **3 of 3** τ = 0 seeds, on both sets. Each dial has σ_shape ≤ 0.05 and an effect of
  0.57–0.85 per unit α.
- **The shape half replicates in 3 of 5 seeds.** The loudness-null rule failed in two seeds, for
  different reasons:
  - **Seed 102 (a clear failure):** the rule picked a weak direction, with less than half the spread of
    the others (σ_d 2.11 vs 3.7–5.3). Its effect was 0.10, its monotonicity 0.36, and it showed grid
    artefacts at α = +2.
  - **Seed 104 (a narrow failure):** it passed every criterion on `confirm3`. On
    `matched-energy-confirm` it failed one criterion, the spectral envelope, at −0.09 against a
    tolerance of −0.05.
- **Both controls behave as pre-registered:**
  - **τ = 0:** a shape dial in 0 of 3 seeds. Its loudness-null directions barely change the field
    (effect 0.02–0.10, below the 0.2 floor).
  - **Untrained:** a shape dial in 0 of 3 seeds, and no loudness dial either. Every steered field shows
    grid artefacts.
- **PC2, the secondary rule,** gives both dials in **3 of 5** seeds (100, 102 and 104). Its exploration
  record was 1 of 3 (Step C) and 2 of 3 (Step D).
- **A post-hoc observation, not a result:** the two shape rules fail on different seeds, and between
  them they cover all 5. No combined rule was pre-registered, and picking the better rule for each seed
  would mean looking at the scores. This observation therefore counts for nothing in the verdict.

## Primary hypothesis

H1 says that in each fresh main-model seed, the symmetry-loudness direction is a loudness dial and the
loudness-null direction is a shape dial, under every criterion, on both `confirm3` and
`matched-energy-confirm`.

| seed | loudness dial | shape dial (loudness-null) | both | what failed |
|---|---|---|---|---|
| 100 | yes | yes | **yes** | — |
| 101 | yes | yes | **yes** | — |
| 102 | yes | no | **no** | shape, on both sets: effect, monotonicity, cycle correlation and gain, grid artefacts (3.5% / 4.7% of frames); also the envelope on matched-energy-confirm |
| 103 | yes | yes | **yes** | — |
| 104 | yes | no | **no** | shape, matched-energy-confirm only: envelope vs ideal −0.09 (tolerance −0.05) |
| **count** | 5 / 5 | 3 / 5 | **3 / 5** | rule: pass ≥ 4, partial 3, fail ≤ 2 → **partial** |

Valid dials in the main model:

| | confirm3 | matched-energy-confirm |
|---|---|---|
| Loudness dial, all 5 seeds: effect per unit α | 0.57–0.65 | 0.58–0.65 |
| Loudness dial, all 5 seeds: σ_shape | 0.00 | 0.03–0.05 |
| Shape dial, seeds 100, 101, 103: effect | 0.33–0.39 | 0.37–0.46 |
| Shape dial, seeds 100, 101, 103: σ_shape | 0.94–1.00 | 0.78–0.94 |

Monotonicity was 1.00 for every valid dial. Cycle correlation was 0.95–1.00, and cycle gain 0.60–0.97.

## Controls

| control | rule | result | outcome |
|---|---|---|---|
| τ = 0 (no content noise), seeds 200–202 | shape dial in ≤ 1 of 3 | **0 of 3** | within rule |
| untrained (random FiLM), seeds 300–302 | shape dial in 0 of 3 | **0 of 3** | within rule |

- **τ = 0:** its loudness dial is valid in all 3 seeds (effect 0.78–0.85, σ_shape ≤ 0.01). Neither shape
  rule finds a direction that moves the field much: loudness-null effects are 0.02–0.10 and PC2 effects
  0.12–0.23. Every one fails the cycle check. With τ = 0 the content branch can carry shape, and no
  shape dial appears. This matches the pre-registered expectation.
- **Untrained:** every direction fails, in 100% of frames on the artefact criterion and in most cases on
  monotonicity and the cycle check. Its rendering error is 1.9–16, against about 0.02 for the trained
  models. The rules therefore did not find something that looks like a dial in a network that learned
  nothing.

## Secondary (reported, not part of the verdict)

**PC2 as the shape rule:**

| seed | PC2 shape dial (both sets) | both dials with PC2 | if not, why |
|---|---|---|---|
| 100 | yes | yes | |
| 101 | no | no | σ_shape 0.63 on confirm3 (needs ≥ 0.7): too much loudness mixed in |
| 102 | yes | yes | |
| 103 | no | no | σ_shape 0.36 / 0.30: mostly loudness |
| 104 | yes | yes | |
| **count** | 3 / 5 | **3 / 5** | exploration record: Step C 1 / 3, Step D 2 / 3 |

**How the two shape directions relate** (|cosine| between the raw directions, computed from the stored
directions after the run; descriptive only):

| seed | loudness-null vs PC2 | σ_d loudness-null / PC2 |
|---|---|---|
| 100 | 0.99 | 5.12 / 5.12 |
| 101 | 0.95 | 5.22 / 5.16 |
| 102 | **0.35** | **2.11** / 4.47 |
| 103 | 0.87 | 5.34 / 5.05 |
| 104 | 0.96 | 3.70 / 3.84 |

- In four seeds the two rules pick nearly the same direction. Whether that direction passes then
  depends on how much loudness it mixes in (PC2 fails in seeds 101 and 103) or on a narrow envelope miss
  (loudness-null fails in seed 104).
- In seed 102 the loudness-null rule picked a different, low-spread direction. That direction produced
  a small, non-monotone change, and at α = +2 a striped, artefact-laden field (`confirm_dials.png`, row
  6).
- These cosines are a description of what happened, not a diagnosis. Why the rule chose that direction
  in seed 102 has not been investigated.

**Rendering cost (not a gate):** vorticity normalized MSE.

| model | v1 val | 256 confirm3 frames (mean removed) |
|---|---|---|
| main, seeds 100–104 | 0.021–0.023 | 0.021–0.023 |
| τ = 0, seeds 200–202 | 0.014–0.016 | 0.014–0.016 |
| Stage 2 VICReg vorticity AE, seeds 0–2 | 0.0105 | 0.0097–0.0100 |
| untrained, seeds 300–302 | — | 1.9–16.2 |

The dial model renders about 2.2× worse than the VICReg autoencoder and about 1.5× worse than its own
τ = 0 version. That is the cost of the content noise. It matches exploration (0.022).

**Spatial-mean diagnostic** (the fraction of frames with more than 10% of ω² in the spatial mean before
it is removed, at the worst α):

- For every main-model loudness dial on both sets it is ≤ 2.3%, and the same holds for every shape
  direction except one.
- The exception is **seed 103's shape dial on confirm3: 23.0% of frames at α = −2** (3.5% at α = −1.5,
  0% elsewhere; 0% on matched-energy-confirm). That is one of the three passing seeds. Its steered tail
  on confirm3 carries a sizeable mean offset, and the pre-registered zero-mean post-processing removes
  it before scoring.
- Whether seed 103 would pass without the removal was not computed.
- Untrained models: 41–100% of frames.

## What this means

- **The loudness dial is reliable.** A label-free rule (minimize what rescaling cannot explain) finds
  it in every trained model, fresh seeds, on both sealed sets, with almost no shape leakage. This is
  partly a design property: the content branch cannot see overall scale, so loudness can reach the
  decoder only through the dials. The confirmation shows that the rule finds that route every time and
  that steering along it passes every realism criterion.
- **The shape dial depends on the content noise.** No τ = 0 model has a direction that moves shape by
  more than a fraction of the threshold. With τ = 0.2, a pre-registered rule finds a valid shape dial in
  3 of 5 seeds. The controls support the attribution: the shape dial appears only when the content
  branch cannot carry shape, and never in an untrained network.
- **The weak point is identification, as far as the data show.**
  - In seed 104 the loudness-null direction is nearly the PC2 direction (|cos| 0.96), and PC2 passes,
    so the shortfall is one criterion missed by 0.04 on one set.
  - In seed 102 the rule picked a different, weaker direction.
  - No single fixed rule found a valid shape direction in all 5 seeds. Whether a better rule would is
    exactly the kind of question this confirmation cannot answer, because it was not pre-registered.
- **Calibration:** I expected pass 60%, partial 20%, fail 20%; the outcome is partial. The control
  expectations held: τ = 0 within its rule (85%), with a loudness dial in at least 2 of 3 seeds (70%;
  observed 3 of 3), and untrained within its rule (90%).

## Threats and limits

- **Five seeds is few.**
  - 3 of 5 against 4 of 5 is one seed. The exact 95% interval for a 3-of-5 rate is 15%–95%.
  - The verdict is the pre-registered reading of these five runs. It is not a precise estimate of how
    often the design works.
- **The sealed sets are now spent.**
  - `confirm3` and `matched-energy-confirm` have each been opened once.
  - Any revised rule, such as one that combines loudness-null and PC2, was suggested by these results.
    Testing it would need fresh seeds and new sealed data.
- **The two shape rules are not independent evidence.** PC2's 3 of 5 is secondary. The fact that the
  rules fail on different seeds is post hoc. Reading "5 of 5 by one rule or the other" as a result would
  be choosing the rule after seeing the scores.
- **Zero-mean post-processing mattered once.** It was fixed in advance and applied uniformly. But in
  seed 103 on confirm3, one of the three passes, it removed a sizeable mean offset at α = −2 in 23% of
  frames.
- **Same distributions.** Both sealed sets come from the same simulator and initial-condition
  distributions as exploration. This is reproduction on fresh trajectories and fresh training runs, not
  generalization to other flows or architectures.
- **Partly by design.** The content branch is scale-free, and the content noise removes most of its
  shape signal (calibrated content shape R² 0.61). The test was whether label-free rules find both dials
  and whether they pass realism, not whether a network invents them unaided.
- **Shape is energy-weighted.** s weights scales by energy, so changes confined to fine scales register
  weakly in it.
- **Rendering is worse:** 2.2× the VICReg autoencoder. It is reported, not gated.
- **Git state during the run.** The environment record shows `fc3b703+dirty`. The only change was
  `results/LOG.md`, which the unsealing step appends to before scoring. The script verified that the
  frozen file was committed and unmodified, and that all 11 checkpoints and every data file matched
  their recorded SHA-256s.

## Process record

| step | when | commit |
|---|---|---|
| Pre-registration committed (approved by the investigator) | 2026-10-27 | `781b2ad` |
| `matched-energy-confirm` generated, disjointness-tested and sealed | 2026-10-27 | `525b5e0` |
| Untrained controls built | 2026-10-27 | `2733e33` |
| Power outages during the first run, then a pause requested by the investigator; the interrupted run was restarted from scratch, and no partial checkpoint was used | 2026-10-27 | `2733e33`, `d8312f7`, `0433882` |
| Training: 8 runs, all 30 epochs, none diverged | 2026-10-29, 00:56–07:08 | — |
| Checkpoints frozen | 2026-10-29 | `fc3b703` |
| Stage 5, run once with the investigator's approval | 2026-10-29, 16:52–18:48 | `258d6af` |

No amendments were made, and nothing in §2–§5 of the pre-registration changed.

## Compute

| item | GPU-hours |
|---|---|
| Training, 8 runs (completed pass) | 6.19 |
| Lost to the two outages and the pause | ≈ 0.7 |
| Untrained controls, freeze | < 0.05 |
| Stage 5 | 1.94 |
| **Total** | **≈ 8.8 of 10** |
