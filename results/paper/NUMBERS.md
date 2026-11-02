# Paper materials: numbers

Every number the paper is expected to cite, with its source file and the commit that holds that version
of the file. Nothing here is new: each value is copied from the committed source named. Commits are
local (branch `master`).

**Status tags:** **C** = confirmatory (pre-registered, sealed data, run once), **E** = exploratory,
**PH** = post hoc. CLAIMS.md defines them.

## Source keys

| key | file | commit |
|---|---|---|
| P1 | `PREREGISTRATION.md` (amendment A1 added in `1671301`) | `f7237f2` |
| R1 | `REPORT.md` | `31eba06` |
| F1 | `results/final/final_evaluation.json`, `.md` | `e8d619e` |
| EB | `results/explore/BRIEF.md` | `63e0ad2` |
| E0 | `results/explore/stage0.md`, `stage0_pca_diagnostic.md` | `153f20c` |
| E1 | `results/explore/stage1.md` | `4b9be90` |
| ES | `results/explore/STAGE2_SUMMARY.md` | `244f625` |
| P2 | `PREREGISTRATION_v2.md` | `8797966` |
| C2 | `results/explore/stage3/confirm.md`, `confirm.json` | `a88ddee` |
| R2 | `REPORT_v2.md` | `a88ddee` |
| BB | `results/build/BRIEF.md` | `08c1511` |
| BP*n* | `results/build/PREDICTIONS.md`, entry *n* | see TIMELINE.md |
| BR | `results/build/BUILD_REPORT.md` | `70ec339` |
| S0, S1, S2, S2b, SA, SA′, SB, SC, SD | `results/build/stage0.md` `ae98ffa`; `stage1.md` `4ae7259`; `stage2.md` `5ecd024`; `stage2b.md` `c9c3e6c`; `stepA.md` `f40a279`; `stepA_prime.md` `6c9e29b`; `stepB.md` `5234145`; `stepC.md` `f0f6c1f`; `stepD.md` `6a3420d` | as listed |
| P3 | `PREREGISTRATION_v3.md` | `781b2ad` |
| FZ | `configs/build/v3_frozen.json` | `fc3b703` |
| C3 | `results/build/v3/confirm.md`, `confirm.json.gz` | `258d6af` |
| R3 | `REPORT_v3.md` | `aff659d` |
| AO | `results/build/autopsy/autopsy.md`, `autopsy.json.gz` | `3c55d66` |
| AU | `AUTOPSY.md` | `41b1213` |
| LOG | `results/LOG.md`, entry at the time given | the entry's commit |

---

## 1. Study 1: does a JEPA organize its latent around fluid quantities? (C)

Test-ID, raw pooled latent, mean of 3 seeds, 95% paired trajectory-bootstrap CIs. Verdict rules: P1 §1–§2.

### 1.1 Verdict table

| criterion | rule | JEPA | control | needed | outcome | src |
|---|---|---|---|---|---|---|
| S1 log enstrophy | gain ≥ 0.20 or unexplained variance halved, CI > 0 | 0.930 | 0.727 | 0.864 | counts: +0.204 [+0.175, +0.235] | R1 §1, A5 |
| S1 log palinstrophy | same | 0.746 | 0.437 | 0.637 | counts: +0.310 [+0.254, +0.366] | R1 A5 |
| S1 log(1 + vortex count) | same | 0.752 | 0.531 | 0.731 | counts: +0.221 [+0.178, +0.275] | R1 A5 |
| S1 Okubo–Weiss fraction (mandatory) | same, and must be among the ≥ 3 | 0.507 | 0.456 | 0.656 | does not count: +0.051 [+0.002, +0.110] | R1 A5 |
| S1 log dissipation | same | 0.176 | 0.134 | 0.334 | does not count: +0.042 [−0.005, +0.085] | R1 A5 |
| **S1** | ≥ 3 count, including Okubo–Weiss | | | | **fails** (3/5, OW not among them) | R1 §1 |
| **S2** log ν, frame pair | R² ≥ 0.6 and ≥ control + 0.3 | 0.457 | 0.481 | 0.781 | **fails** | R1 §1 |
| **S3** enstrophy, best of top-3 slow directions | \|r\| ≥ 0.8, control < 0.8, gap ≥ 0.1, CI > 0 | 0.356 | 0.514 | 0.800 | **fails**, gap CI [−0.31, +0.13] | R1 §1, A4 |
| S3 energy | same | 0.794 | 0.948 | unreachable | cannot pass | R1 §1 |
| **Verdict** | | | | | **not supported** | R1, F1 |

- **Shift-averaged readout:** the vortex-count gain falls to +0.176. Under the pre-registered rule it is
  attributed to shift tolerance, leaving 2 attributable S1 gains. Source: R1 §1.1.
- **S1 rule variants:** literal brief 1/5; pre-registered 3/5 (fails); shift-averaged 2/5. Dropping the
  OW clause gives 3/5 (would be "partially supported"), but the same rule then also passes the
  non-physical JEPA (3/5) and static VICReg (4/5). Source: R1 §1.2.

### 1.2 Controls (paired differences, JEPA − control, test-ID, raw readout)

| quantity | JEPA − non-physical JEPA | JEPA − static VICReg | JEPA − pooled JEPA |
|---|---|---|---|
| log enstrophy | −0.026 [−0.034, −0.020] | −0.043 [−0.056, −0.034] | +0.068 [+0.056, +0.082] |
| log palinstrophy | −0.055 [−0.075, −0.038] | −0.154 [−0.192, −0.124] | +0.081 [+0.062, +0.104] |
| log(1 + vortex count) | −0.037 [−0.052, −0.023] | −0.090 [−0.113, −0.068] | +0.115 [+0.092, +0.141] |
| Okubo–Weiss fraction | +0.003 [−0.042, +0.051] | −0.166 [−0.233, −0.109] | +0.084 [+0.042, +0.135] |
| log dissipation | −0.002 [−0.031, +0.029] | −0.015 [−0.072, +0.047] | +0.028 [−0.006, +0.061] |
| log ν pair R² (no CI) | −0.066 | −0.242 | +0.139 |
| S3 enstrophy \|r\| | −0.114 [−0.21, +0.06] | −0.131 [−0.22, −0.02] | −0.076 [−0.11, +0.09] |

Source: R1 §2.

- **Static VICReg under the pre-registered rules:** S1 4/5 including OW, which rates "partially
  supported" against the control. Source: R1 §2, A4.
- **R² for every model** (untrained, JEPA, pixel, pooled JEPA, static VICReg, non-physical JEPA), as
  mean ± std: R1 appendix A1 (raw) and A2 (shift-averaged); F1.

### 1.3 JEPA vs pixel predictor (pre-registered comparison, no pass/fail)

| quantity | JEPA | pixel | JEPA − pixel [CI] |
|---|---|---|---|
| log enstrophy | 0.930 | 0.890 | +0.041 [+0.024, +0.057] |
| log palinstrophy | 0.746 | 0.733 | +0.013 [−0.037, +0.052] |
| log(1 + vortex count) | 0.752 | 0.521 | +0.231 [+0.179, +0.284] |
| Okubo–Weiss fraction | 0.507 | 0.270 | +0.237 [+0.160, +0.315] |
| log ν pair R² | 0.457 | 0.710 | −0.253 |
| S3 enstrophy \|r\| | 0.356 | 0.821 | −0.465 [−0.55, −0.32] |

Source: R1 §3.

- **Pixel predictor under the S3 rule (PH: not hypothesized, not predicted):** 0.821 against 0.514 for
  the control, gap CI [+0.15, +0.50] (raw). Shift-averaged: 0.855 against 0.514, [+0.18, +0.52]. Source:
  R1 §3, A4.
- **Pixel slow direction 2 against enstrophy:** trajectory |r| 0.80 with log Z₀, 0.22 with log E₀
  (Phase 4 view: seed 0, val frames, unscored). Source: `results/phase4_traversals_pixel.png` @ `c2c3a79`.
- **Training state:**
  - pixel val loss 0.035× its copy baseline, flat;
  - JEPA 0.60× its copy baseline, still improving about 2% per 4 epochs at epoch 40.

  Source: R1 §3.

### 1.4 Other study-1 numbers

| item | value | src |
|---|---|---|
| OOD / neighbouring-band MSE ratio, all models and targets | 0.71–1.46 | R1 §4 |
| JEPA OOD ratios: enstrophy, palinstrophy, vortex count, OW, ν | 0.98, 1.17, 1.10, 0.76, 1.31 | R1 §4 |
| Untrained latent: PCA variance in one direction | 86% | R1 §5 |
| Data: trajectories (train / val / test-ID / test-OOD) | 2,000 (1,295 / 279 / 276 / 150) | P1 §3 |
| Inviscid trajectories in test-ID (S3 rests on these) | 74 | P1 §3, R1 §6 |
| S3 instability: control \|r\| on val vs test | 0.23 vs 0.51 (read S3 as ± 0.2) | R1 §6 |
| Compute logged | 11.5 h (RTX 3060) | R1 §9 |

---

## 2. Exploratory line between study 1 and v2 (E)

The test sets were retired. These numbers come from `explore` (1,006 trajectories, 378 inviscid) and
`matched-energy` (300 inviscid, E₀ = 0.2893).

| item | value | src |
|---|---|---|
| Stage 0: JEPA enstrophy \|r\| by readout (pooled / map mean⊕std / map PCA-512) | 0.353 / 0.303 / 0.295 | LOG 2026-10-15 20:42; E0 |
| Stage 0: share of the spatial-mean subspace inside the top-512 map PCs (untrained / JEPA / pixel / static, seed 0) | 0.101 / 0.488 / 0.083 / 0.643 | E0 (diagnostic) |
| Stage 1 kill gate (a), explore: pixel vs untrained | 0.829 vs 0.418, gap +0.411 [+0.326, +0.514]: pass | E1 |
| Stage 1 kill gate (b), matched-energy | 0.979 vs 0.177, gap +0.802 [+0.718, +0.827]: pass | E1 |
| Stage 1, matched-energy: JEPA vs untrained (context) | 0.895 vs 0.177 | E1 |
| Stage 2 E1 by horizon, imaginer / summarizer (k = 1, 4, 16, 32) | 0.68/0.31, 0.80/0.37, 0.69 (1 seed)/0.38, 0.76/0.46 | ES |
| Stage 2 imaginer − summarizer gap by k | +0.37 [+0.31, +0.43], +0.42 [+0.36, +0.48], +0.31 [+0.24, +0.39], +0.30 [+0.25, +0.35] | ES |
| Bridge (k = 4, original distribution) | 0.84 | ES |
| Autoencoder E1, plain: renders ψ / u / ω | 0.53 / 0.82 / 0.81 | ES |
| Autoencoder E1, + VICReg: renders ψ / u / ω | 0.33 / 0.42 / 0.52 | ES |
| Untrained control E1 (explore) | 0.42 | ES |
| E4 a/b, plain: renders ω / u / ψ; untrained | 1.01 / 0.82 / 3.10; 6.1 | ES |
| log P R², ω autoencoder plain → + VICReg | 0.68 → 0.93 | ES |
| Imagination probe: rendered log Z change at ±2 std along the enstrophy direction | about ±1–5% (P6.1 failed, 3 of 3 seeds) | ES |
| Divergences: imaginer-family runs at constant lr | 11 of 30: 8 of 15 at 3 × 10⁻³, 3 of 15 at 10⁻³ (corrected 2026-11-03; see Corrections) | LOG 2026-10-15/16 (the 30 Stage 2 imaginer-family training entries); explore PREDICTIONS entry 5 @ `33ba39b` |
| Compute, exploratory line | 24.5 h | ES; R2 |

---

## 3. v2 confirmation: rendering and regularization (C)

Confirm set: 1,005 trajectories, 378 inviscid; opened once. Fresh seeds 3–5 for the primary hypotheses.
Rules: P2 §2–§3.

| hypothesis | rule | result | outcome | src |
|---|---|---|---|---|
| H1, velocity AE (fresh) | E1 ≥ 0.70; ≥ control + 0.25; CI > 0 | E1 0.860; +0.480 [+0.390, +0.565] | pass | C2; R2 |
| H1, vorticity AE (fresh) | same | E1 0.799; +0.418 [+0.360, +0.486] | pass | C2; R2 |
| **H1** | both pass | | **holds** | R2 |
| H2, velocity: plain − VICReg (fresh) | ≥ 0.20; CI > 0 | +0.338 [+0.289, +0.390] | pass | C2 |
| H2, vorticity: plain − VICReg (fresh) | same | +0.149 [+0.105, +0.193] | **fail** | C2 |
| **H2** | both pass | | **fails** | R2 |
| H3 (secondary), E4 a/b: vorticity AE vs streamfunction AE (Stage 2 seeds) | ω closer to 1; ψ > 2 | 1.00 vs 3.13 | pass | C2 |
| H4 (secondary), imaginer k = 4 − imaginer k = 4 + VICReg | ≥ 0.20; CI > 0 | +0.315 [+0.272, +0.358] | pass | C2 |
| **Verdict** | | | **partially supported** | C2; R2 |

- **Untrained control, seeds 3–5:** E1 0.380 [0.30, 0.46]. Source: C2.
- **Per-seed E1** (source: R2):
  - vorticity AE 0.709, 0.837, 0.850;
  - vorticity AE + VICReg 0.621, 0.649, 0.679;
  - velocity AE 0.894, 0.856, 0.830;
  - velocity AE + VICReg 0.568, 0.501, 0.497.
- **Reported without a rule** (source: C2):
  - Stage 2 seeds under the H1/H2 rules: velocity +0.437 [+0.345, +0.530]; vorticity +0.437
    [+0.354, +0.517]; VICReg drops +0.345 [+0.285, +0.412] (velocity), +0.262 [+0.207, +0.313]
    (vorticity). All pass.
  - Imaginer k = 4 − summarizer k = 4: +0.483 [+0.42, +0.52].
  - Imaginer k = 4: 0.815. Imaginer k = 4 + VICReg: 0.500. Summarizer k = 4: 0.332.
- **H4 closes about two-thirds of the imaginer-vs-summarizer gap:** a drop of 0.315 against a gap of
  0.483. Source: R2.
- **Rendering cost of VICReg** (known before unsealing): about 25× worse reconstruction for the
  autoencoders, about 20× for the pixel predictor. Source: P2 §7; R2.
- **Process numbers:**
  - 39 checkpoint SHA-256 values verified;
  - 15 fresh runs, none diverged;
  - compute 29.6 GPU-h (24.5 exploration + 5.1 Stage 3) of 30.

  Source: R2.

---

## 4. The dial build, Stage 0 to Step D (E)

Exploration data only. The 2-of-3-seeds model rule and the dial criteria are in BP2 (and BP6 for
artefacts). The trajectory, one line per stage (source: BR §3, with the per-stage file in the last
column):

| stage | what changed | what it showed | per-stage file |
|---|---|---|---|
| 0 | Existing autoencoders and imaginer (21 models); measure only | 0 of 84 candidate directions are dials. Best effect 0.086 per unit α. The supervised s-direction is readable (\|r\| 0.97–0.99) but moves nothing: "readable, not steerable". The absolute envelope test rejects ideal dials (ideal shape dial min 0.791 / 0.685). | S0; LOG 2026-10-17 18:55 |
| 1 | M1-spec, M1-G: channel-split latent, two heads (+ FiLM variant) | No dials. M1-G's loudness knob fails the round trip: normalized content still carries loudness (R² 0.99). Content keeps most of VICReg's information advantage (log P R² 0.90–0.92, against 0.94 VICReg and 0.76 plain). | S1 |
| 2 | M2-SF: separate branches, scale-free content | Loudness dial 3/3 (withdrawn at Step A). No shape dial: the dials hold s (R² 0.95), but steering them does not move it. | S2 |
| 2b | M2-SW: whitened content | PCA shape dial 2/3 (1/3 after the artefact criterion). No clean loudness dial. Rendering 3.8× worse than the VICReg AE. | S2b |
| A | No training: loudness by symmetry; grid-artefact criterion | The rescaling-closest direction is a pure loudness dial in 2/3 seeds. Retroactively, M2-SF's Stage 2 loudness dials fail the artefact criterion (checkerboards in 5–14% of frames at α = +2). | SA |
| A′ | No training: shape = top PC after projecting out loudness in latent space | Mixtures (σ_shape 0.50–0.54). Both dials 0/3 under this rule, 1/3 under PC2. | SA′ |
| B | Shape noise on the content spectrum, τ = 0 / 0.5 / 1 / 2 (12 runs) | Loudness-null direction σ_shape rises from 0.07 to 0.93 as content shape R² falls. Both dials 3/3 at τ = 0.5 under BP2 alone, 0/3 after the artefact criterion. Rendering 0.015 → 0.374. | SB |
| C | τ = 0.5, resize-convolution decoder | 0.0% artefact frames. Both dials 3/3 (loudness-null); PC2 1/3. Rendering 0.039. | SC |
| D | τ = 0.2, resize-convolution decoder | Both dials 3/3 (loudness-null); PC2 2/3. Rendering 0.022. The v3 dose rule (BP10) picks this model. | SD |

### 4.1 Final exploration model (Step D, τ = 0.2), per seed

| seed | shape dial: Δs per unit α, explore / matched | σ_shape | loudness dial: Δa per unit α | σ_shape | artefact frames |
|---|---|---|---|---|---|
| 0 | +0.43 / +0.46 | 0.99 / 0.95 | +0.66 / +0.69 | 0.00 / 0.03 | 0% |
| 1 | +0.33 / +0.29 | 0.87 / 0.82 | +0.53 / +0.50 | 0.00 / 0.02 | 0% |
| 2 | +0.35 / +0.33 | 0.88 / 0.86 | −0.62 / −0.66 | 0.00 / 0.05 | 0% |

Source: BR §4; BD.

### 4.2 Dose table

| τ | content shape R² | decoder | shape knob σ_shape (median) | both dials (all criteria) | ω error |
|---|---|---|---|---|---|
| 0 | 0.97 | pixel-shuffle | 0.07 | 0 / 3 | 0.015 |
| 0.2 | 0.61 | resize | 0.88 | 3 / 3 | 0.022 |
| 0.5 | 0.29 | pixel-shuffle | 0.93 | 0 / 3 (artefacts) | 0.041 |
| 0.5 | 0.29 | resize | 0.96 | 3 / 3 | 0.039 |
| 1 | 0.18 | pixel-shuffle | 0.88 | 0 / 3 | 0.125 |
| 2 | 0.07 | pixel-shuffle | 0.93 | 0 / 3 | 0.374 |

Source: BR §4.

### 4.3 Other build numbers

| item | value | src |
|---|---|---|
| Correlation of loudness and shape in the data | r = −0.82 | BR §5 |
| Rendering, final exploration model vs references (ω normalized MSE) | 0.022, against τ = 0: 0.015 (1.5×), VICReg AE: 0.0105 (2.1×), plain AE: 0.0012 (19×) | BR §1 |
| Spatial-mean offset, final model: explore frames at extreme α | up to 5.5% | BR §4 |
| Effect floor raised after Stage 0 | 0.1 → 0.2 | BP2 |
| Artefact criterion | out-of-band share > 1×10⁻² in > 1% of frames; calibrated on one known-bad case | BP6 |
| Compute | about 17.8 of 18.5 GPU-h (17.2 logged); 31 models | BR §1 |

---

## 5. v3 confirmation of the dial build (C)

The models are 5 fresh main seeds (100–104), τ = 0 seeds 200–202 and untrained seeds 300–302. They were
scored on `confirm3` (1,007 trajectories, 378 inviscid) and `matched-energy-confirm` (300 inviscid),
each opened once. Rules: P3 §2, §5.

### 5.1 Verdict and per seed

| seed | loudness dial | shape dial (loudness-null) | both | σ_shape loudness / shape, confirm3 | σ_shape loudness / shape, matched-energy-confirm | what failed |
|---|---|---|---|---|---|---|
| 100 | yes | yes | **yes** | 0.00 / 0.94 | 0.03 / 0.94 | — |
| 101 | yes | yes | **yes** | 0.00 / 0.98 | 0.04 / 0.78 | — |
| 102 | yes | no | **no** | 0.00 / 0.41 | 0.05 / 0.46 | shape, both sets: effect, monotonicity, cycle correlation and gain, grid artefacts (3.5% / 4.7% of frames); also the envelope on matched-energy-confirm |
| 103 | yes | yes | **yes** | 0.00 / 1.00 | 0.05 / 0.81 | — |
| 104 | yes | no | **no** | 0.00 / 0.96 | 0.03 / 0.96 | shape, matched-energy-confirm only: envelope vs ideal −0.09 (tolerance −0.05) |
| **count** | 5/5 | 3/5 | **3/5** | | | pass ≥ 4, partial 3, fail ≤ 2 → **partially supported** |

Source: C3 (per-model table); R3 (primary hypothesis).

- **Seed 102's loudness-null direction on confirm3:**
  - σ_d 2.11 (others 3.7–5.3);
  - effect 0.099;
  - monotonicity 0.36;
  - cycle correlation −0.21.

  Source: C3; R3.
- **Valid dials, ranges** (source: R3):

  | | confirm3 | matched-energy-confirm |
  |---|---|---|
  | loudness effect per unit α (5 seeds) | 0.57–0.65 | 0.58–0.65 |
  | loudness σ_shape | 0.00 | 0.03–0.05 |
  | shape effect (seeds 100, 101, 103) | 0.33–0.39 | 0.37–0.46 |
  | shape σ_shape | 0.94–1.00 | 0.78–0.94 |

  For every valid dial: monotonicity 1.00, cycle correlation 0.95–1.00, cycle gain 0.60–0.97.
- **Exact 95% interval for 3 of 5:** 15%–95%. Source: R3 (threats).

### 5.2 Controls

| control | rule | result | outcome | details | src |
|---|---|---|---|---|---|
| τ = 0, seeds 200–202 | shape dial in ≤ 1 of 3 | 0 of 3 | within rule | loudness dial 3/3 (effect 0.78–0.85, σ_shape ≤ 0.01); loudness-null effects 0.02–0.10, PC2 effects 0.12–0.23; every one fails the cycle check | R3; C3 |
| untrained, seeds 300–302 | shape dial in 0 of 3 | 0 of 3 | within rule | no loudness dial either; artefacts in 100% of frames; rendering error 1.9–16.2 | R3; C3 |

### 5.3 Secondary and descriptive (pre-registered, not in the verdict)

| item | value | src |
|---|---|---|
| PC2 shape rule: shape dial / both dials | 3/5 (seeds 100, 102, 104) / 3/5 | C3; R3 |
| PC2 failures | 101: σ_shape 0.63 on confirm3; 103: σ_shape 0.36 / 0.30 (mostly loudness) | R3 |
| PC2 exploration record | Step C 1/3, Step D 2/3 | P3 §3 |
| \|cos\| loudness-null vs PC2, seeds 100–104 | 0.99, 0.95, 0.35, 0.87, 0.96 (descriptive, computed after the run) | R3 |
| σ_d loudness-null / PC2, seeds 100–104 | 5.12/5.12, 5.22/5.16, 2.11/4.47, 5.34/5.05, 3.70/3.84 | R3 |
| Rendering ω normalized MSE, main (v1 val and confirm3) | 0.021–0.023 | C3; R3 |
| Rendering, τ = 0 | 0.014–0.016 | C3; R3 |
| Rendering, Stage 2 VICReg vorticity AE (v1 val; confirm3 frames) | 0.0105; 0.0097–0.0100 | C3; R3 |
| Rendering ratios | main ≈ 2.2× the VICReg AE, ≈ 1.5× its τ = 0 version | R3 |
| Spatial-mean diagnostic: main loudness dials, both sets | ≤ 2.3% of frames with > 10% of ω² in the mean | R3 |
| Spatial-mean diagnostic: seed 103 shape dial, confirm3 | 23.0% of frames at α = −2 (3.5% at −1.5; 0% on matched-energy-confirm) | C3; R3 |
| Spatial-mean diagnostic: untrained | 41–100% of frames | R3 |
| Pre-registered calibration expectations (P3 §9, written by the assistant, not the investigator) vs outcome | pass 60 / partial 20 / fail 20 → partial; τ = 0 within rule 85% → yes; τ = 0 loudness dial ≥ 2/3 70% → 3/3; untrained within rule 90% → yes | P3; R3 |

- **Post hoc, not countable:** the union "loudness-null OR PC2" covers 5 of 5 main seeds on the sealed
  sets. Source: R3; AO (A2).
- **Compute** (source: R3):
  - training 6.19 GPU-h;
  - lost to outages and the pause ≈ 0.7;
  - Stage 5 1.94;
  - total ≈ 8.8 of 10.
- **Checkpoints:** 11 SHA-256 values frozen before unsealing. Source: FZ.

---

## 6. Autopsy of v3 Stage 5 (PH)

Exploration data only; the sealed sets were not reopened. Nothing changes the v3 verdict. Procedure and
predictions were committed before computation (BP11, `1cbc2e7`).

| item | value | src |
|---|---|---|
| Seed 102: cos²(g, PC2) | 0.000 | AO; AU |
| Seed 102: fraction of PC2's shape effect removed by projecting out g (linear / steered) | −0.00 / −0.00 | AO; AU |
| Seed 102: projected z-variance, top direction vs PC2's projection | 2.39 vs 1.48 (other seeds within 0.15) | AO; AU |
| Seed 104: cos²(g, PC2); removed fraction | 0.001; +0.02 / +0.01 | AO |
| Seed 104: envelope vs ideal on the exploration matched-energy set | −0.09 (same miss as on matched-energy-confirm); PC2 +0.00 | AO; AU |
| Seed 103: removed fraction (projecting out g increases PC2's shape effect) | −0.66 / −0.70 | AO |
| Union rule on controls (sealed-set outputs): PC2, loudness-null, union | 0/3, 0/3, 0/3 (τ = 0); 0/3, 0/3, 0/3 (untrained) | AO (A2) |
| Transformation-matched (TM) shape rule: shape dial | main 0/5, τ = 0 0/3, untrained 0/3 | AO |
| TM directions, main seeds (explore): effect / cycle gain / σ_shape | 0.12–0.17 / 0.35–0.49 / 0.89–0.97 | AO; AU |
| TM directions, main seeds (matched, exploration): effect / cycle gain / σ_shape | 0.18–0.24 / 0.30–0.45 / 0.62–0.74 | AO; AU |
| J_T of the TM direction: main / τ = 0 / untrained | 0.102–0.111 / 0.344–0.410 / 0.819–0.866 | AO |
| J_T of the stored loudness-null direction: seed 102 vs others | 0.69 vs 0.19–0.35 | AO |
| Seed 103, mean kept vs removed (exploration sets) | same classification; Δs +0.343 / +0.340 (explore); pre-removal mean > 10% in 16.4% of explore frames | AO (A4) |
| Grid-edge fits of the TM family | 30,781 of 5,586,944 (0.55%) | AO |
| Predictions (BP11) | 4 right of 9 graded; PA.4 and PA.5 (TM optimism) wrong | AU |
| Compute (outside the v3 budget) | 2.27 GPU-h + 37 s code check | AU; LOG 2026-10-29 21:38 |

---

## 7. Compute across the project

| line | GPU-hours | src |
|---|---|---|
| Study 1 | 11.5 (logged wall-clock) | R1 §9 |
| Exploratory line + v2 | 29.6 (24.5 + 5.1) | R2 |
| Build | ≈ 17.8 (17.2 logged) | BR §1 |
| v3 | ≈ 8.8 | R3 |
| Autopsy | 2.27 | AU |
| **Sum** (computed here, not in any report) | **≈ 70** | — |

All runs used one RTX 3060 (12 GB).

---

## Corrections

**2026-11-03, §2, divergences.**
- **What changed:** "11 of 24 (across two lrs)" now reads "11 of 30: 8 of 15 at 3 × 10⁻³, 3 of 15 at
  10⁻³".
- **Why:** the old value was copied from `results/explore/STAGE2_SUMMARY.md` @ `244f625`, whose
  denominator is wrong.
  - `results/LOG.md` lists 30 imaginer-family Stage 2 training runs: 15 at each learning rate, each set
    including the k = 4 seed-0 selection run.
  - 8 diverged at 3 × 10⁻³ (also stated in explore PREDICTIONS entry 5 @ `33ba39b`). 3 diverged at
    10⁻³: k = 1 seed 1, and k = 16 seeds 0 and 1.
- **Found by:** `results/paper/AUDIT_v1.md` (CHECK-3, rows P-10 and AP-F7).
- **Not changed:** `STAGE2_SUMMARY.md` is a committed report and keeps its original wording.
