# Audit of DRAFT_v1 (Phase 1)

What was audited: `results/paper/DRAFT_v1.md`, the draft of "Readable Is Not Steerable", with every
section, table, caption, appendix, the process section and the AI-assistance statement.

What it was checked against:
- the committed reports, outputs, pre-registrations and logs;
- the code and configs;
- the paper-materials pack (`results/paper/*.md` @ `8fc7f02`).

**Scope and conduct:**
- No sealed or retired set was opened. No training, and no new experiment.
- One helper was written: `param_count.py`, in the session scratchpad. It builds the decoder classes to
  count parameters, loads no data, and has a guard that refuses any path naming `confirm_v2`,
  `confirm3`, `matched-energy-confirm` or the v1 test splits.
- The rest were reads of committed files, plus three read-only queries of committed JSON outputs
  (Stage 0 readability, Stage 2b cycle gains, Study 1 stored CIs).
- **`DRAFT_v1.md` was not in the repository.** It was saved verbatim from the text the investigator
  supplied, and it was not otherwise edited. Locations below use section and paragraph numbers, not
  line numbers. If the source export differs, replace the file; the locations still hold.

**How to read the table:**
- "Source" gives the file and the commit that holds the cited version.
- Paragraphs are counted within a section (¶1 = first paragraph).
- "ext." marks a claim about cited literature. These cannot be checked against this repository and
  are deferred to the Phase 3 reference check.
- Status follows the brief:
  - **OK**;
  - **MISMATCH**: the correct value and its source are given;
  - **UNSUPPORTED**: no committed source;
  - **TAG**: wrong C/E/PH tag;
  - **WORDING**: breaks a wording rule or the "cannot make" list, or overstates.

---

## Summary

### Counts

| status | rows | of which external literature (deferred to Phase 3) |
|---|---|---|
| OK | 193 | 1 |
| MISMATCH | 44 | 0 |
| UNSUPPORTED | 16 | 11 |
| TAG | 4 | 0 |
| WORDING | 37 | 0 |
| **total** | **294** | |

There are 5 internal UNSUPPORTED rows. Three are claims with no source in the repository: §4.1's pilot
results on non-physical dynamics, §6.2's rationale that per-shell noise would average out, and §8.5's
account of chat-assistant use. The other two are not yet true: the release statement and the claim that
every number and reference was checked.

### The ten fixes that matter most

1. **"Usable dials had to be identified by their effect on the output rather than by data variance"**
   (Abstract; §1 contribution 3; §7.4).
   - The sealed data contradict this for shape. PC2, a pure variance rule, gave a valid shape dial in 3
     of 5 seeds, the same as loudness-null PCA. The claim holds for loudness only.
   - Rows A-13, I-15, D-17.
2. **Study 1 overstated in the negative.** "Does not organize its latent around fluid quantities
   measurably better than an untrained network" (Abstract; §1).
   - The JEPA beat the control significantly on three S1 targets.
   - What failed is the pre-registered criteria. Say that.
   - Rows A-2, I-8.
3. **Study 1's test sets were not sealed in code, and its checkpoints were not hashed.**
   - §3.4 says every test set was sealed in code with checkpoint SHA-256s committed. That is true for
     Studies 2–3 only.
   - Study 1's sets were single-use by protocol (PREREGISTRATION.md §2.7). Its final config names runs,
     not hashes.
   - The phrase "each confirmed once on sealed data" (Abstract, §1) needs the same correction.
   - Rows A-1, I-2, I-7, M-38, M-39, P-6.
4. **"Any latent that encodes enstrophy strongly will show a slow direction tracking it"** (§3.3).
   - Study 2 refutes this. The VICReg models decode log Z better (R² 0.93–0.96) yet have lower E1.
   - Row M-28.
5. **The dial model's loss is misdescribed.**
   - §6.2 omits the VICReg variance and covariance terms (weights 25 and 1) on the pooled content
     latent.
   - It also describes the dial code as "two groups of four", which was the Stage 1 model. The final
     design feeds one 8-dimensional vector, from a separate dial encoder, to every decoder stage.
   - Rows S3-8, S3-16.
6. **The dial criteria are wrong or incomplete** (§3.3).
   - The cycle correlation threshold is 0.9, not 0.5, and the gain must lie in [0.5, 2].
   - The shared-pattern (watermark) criterion is missing.
   - Every criterion must hold on both scoring sets.
   - Rows M-33, M-36, CHECK-1.
7. **The Stage 2 divergence record is misstated** (§8.2; Appendix F).
   - 11 of **30** imaginer-family runs diverged, not 11 of 24.
   - All 15 runs were rerun at 10⁻³, not only the eight that diverged.
   - The three that then diverged at 10⁻³ were excluded under a committed rule, not rerun.
   - Rows P-10, P-11, AP-F7, CHECK-3.
8. **The artefact rates are misattributed** (§6.4).
   - "5–14% of frames" belongs to the Stage 2 (M2-SF) loudness dials.
   - At τ = 0.5 the loudness directions had 0–4.3%.
   - Row S3-34.
9. **The AI-assistance statement doesn't match the record** (§8.5).
   - The pre-registrations were drafted by the coding agent, in the repository, and approved by the
     investigator.
   - Every prediction entry's probabilities were written by the agent, not only the v3 calibration.
   - The investigator approved every decision; several were proposed by the agent.
   - "Every number is traced and every reference checked" is not yet true.
   - Rows P-20 to P-25.
10. **"Whenever"** in §7.2 ("enstrophy became prominent whenever a model had to render the field and
    its latent was not whitened") is contradicted by the plain streamfunction autoencoder (E1 0.558 on
    the sealed set). The same overreach appears in §5's opening ("is what makes").
    - Rows D-7, S2-1.

**Next in importance:**
- ν range: 3 × 10⁻³, not 2 × 10⁻³ (M-6).
- The symmetry objective is normalized by the size of the change (S3-21).
- The summarizer's E1 rises with horizon (S2-7).
- The "after the build" cycle gains are from Stage 2b (D-5).
- Ledoit–Wolf shrinkage is on Σ_within only (M-24).
- §4.1's non-physical pilot claim has no source (S1-8).
- The resize decoder did not remove artefacts on sealed data for seed 102 (S3-37, I-16).

---

## Resolved [CHECK] markers (exact replacement text)

### CHECK-1: §3.3, the effect statistic and the cycle thresholds

**Sources:**
- `results/build/PREDICTIONS.md` entry 1, definitions (@ `08c1511`);
- entry 2, final thresholds (@ `5ef729a`);
- `dials/harness.py` (@ `a7e4b32`, lines 140–151 and 185–193).

**Replace the first two bullets with:**

> - its effect is at least 0.2 log-units per unit α. For each frame, the slopes of a and s against α
>   are fitted by least squares; the median of each slope is taken over frames, and the effect is
>   |median Δa| + |median Δs|. It must also be monotone: the median over frames of the absolute Spearman
>   correlation between α and the dominant coordinate (a or s, whichever has the larger median slope) is
>   at least 0.9;
> - re-encoding the steered field recovers the commanded move. The decoded vorticity is converted to
>   velocity through the inverse Laplacian and re-encoded. The change of its projection on d, against
>   the commanded α σ\_d, has Pearson correlation ≥ 0.9 over all (frame, α) pairs, and least-squares
>   gain between 0.5 and 2;
> - the change is frame-specific: at most half of the α = ±2 change is one pattern shared by all frames
>   (shared-pattern fraction ≤ 0.5);

**Add after the list:** "Every criterion must hold separately on each of the two scoring sets."

### CHECK-2: §5.2, the definition of E4

**Sources:**
- `PREREGISTRATION_v2.md` §3 (@ `8797966`);
- `results/explore/PREDICTIONS.md` entry 3 (@ `d8c75fa`);
- `explore/stage2_eval.py`, `direction_stats` (@ `4e52bad`);
- `explore/stage3_confirm.py` (@ `8797966`, seed mean, |log| comparison).

**Replacement for the H3 bullet and the marker:**

> - **H3 (secondary): the render target sets the mix.** For each seed, take the enstrophy-tracking
>   direction: the one of the three slowest directions with the largest |r| with log Z₀. Regress its
>   trajectory-mean projection by least squares on an intercept, log E₀ and log(Z₀/E₀), all
>   unstandardized. E4 = a/b, the coefficient on log E₀ divided by the coefficient on log(Z₀/E₀),
>   averaged over seeds. Because log Z₀ = log E₀ + log(Z₀/E₀), E4 = 1 means the direction tracks log Z₀
>   alone, and E4 > 1 means it leans toward energy. H3 holds if the vorticity autoencoder's E4 is
>   closer to 1 on a log scale than the streamfunction autoencoder's (|log E4| smaller), and the
>   streamfunction autoencoder's E4 exceeds 2 (Study 2 exploration seeds 0–2).

### CHECK-3: §8.2, the imaginer-family divergences

**Sources:**
- `results/explore/PREDICTIONS.md` entry 5 (@ `33ba39b`);
- `results/LOG.md`: the 30 `[EXPLORE] Stage 2 training: s2_imaginer_*` / `s2_bridge_*` entries and the
  2026-10-16 11:34 override entry (@ `7c41a5e`);
- `results/explore/STAGE2_SUMMARY.md` (@ `244f625`).

**Replacement for the Exploration bullet:**

> - **Exploration.** At the learning rate chosen by the label-free selection rule (3 × 10⁻³), 8 of 15
>   imaginer-family runs diverged.
>   - Before any evaluation, every imaginer-family run was rerun at 10⁻³, and all 3 × 10⁻³ runs were
>     excluded.
>   - At 10⁻³, 3 of 15 runs diverged, one at k = 1 and two at k = 16. A divergence rule committed
>     before these runs applied: final validation loss at or above the copy-the-frame baseline. Under
>     it, the three were reported and excluded, not rerun.
>   - That left two seeds at k = 1 and one at k = 16. In total, 11 of 30 imaginer-family runs
>     diverged.
>
>   The plan was revised once before evaluation, and one probe designed after seeing another probe's
>   result is labelled post hoc.

**Note:** the "11 of 24" in `STAGE2_SUMMARY.md` and in the pack (NUMBERS.md §2) has the wrong
denominator. The log lists 30 imaginer-family runs: 15 at each learning rate, each set including the
k = 4 seed-0 selection run.

---

## Setup details checked against code and config

| item | draft says | code / config (file @ commit) | status |
|---|---|---|---|
| Snapshot interval | 0.04 | `configs/phase1_data.yaml` @ `f7237f2`: `dt_out: 0.04` | OK |
| Inviscid length | 24 snapshots | same: `frames_inviscid: 24` (t ≤ 0.92, before the truncated system thermalizes) | OK |
| Viscous length | not stated | same: `frames_viscous: 64` (t ≤ 2.52) | add to §3.1 |
| Initial-condition families | blobs, 2–12 vortices of random sign, size and position; spectral fields | same: `blob_fraction 0.5`, `blob_count [2, 12]`, `blob_radius [0.25, 0.6]`, `spectral_k0 [2, 5]`, `omega_rms [2, 4]` (log-uniform). `PREREGISTRATION.md` §3: E(k) ∝ k⁴ exp(−2(k/k₀)²) | OK |
| ν range | about 2 × 10⁻³ to 5 × 10⁻² | same: `nu_range: [0.003, 0.05]` | **MISMATCH** (M-6) |
| Inviscid fraction | about a quarter | same: `inviscid_fraction: 0.25` (Study 1). Later sets differ: `explore` and `confirm_v2` 378 of about 1,000 (38%); matched-energy sets all inviscid | OK for Study 1; qualify (M-7) |
| Time stepping | not stated | `solver/ns2d.py` @ `f7237f2`: integrating-factor (Lawson) RK4, CFL 0.3, float64; measured convergence orders 4.13, 4.04, 4.03 (LOG 2026-10-14 17:20) | add to §3.1 |
| Encoder | small CNN, Kaiming, no normalization, 8 × 8 × 128 map, pooled 128-d | `models/encoder.py` @ `f7237f2`: circular padding, GELU. 3 × 3 conv 2→32, then three stages (stride-2 3 × 3 conv + 3 × 3 conv) at widths 64, 128, 128, then a 1 × 1 conv to d. Kaiming-normal (fan-in), zero biases. Pooled = spatial mean | OK for Studies 1–2; Study 3 differs (M-17) |
| Decoder parameters | 1.45 M, 79% of the sub-pixel decoder | `dials/split.py` @ `fa7e95d`, `FiLMDecoder(120, 8)`, counted by the audit helper: resize 1,447,553, sub-pixel 1,834,849 (each including 8,640 FiLM parameters); ratio 0.789 | OK |
| Build training recipe | cosine decay, gradient clipping, renders vorticity only | `configs/build/v3/b6_main_s100.yaml` @ `26921ea`; `train/train.py` @ `afd31a1`. v1 train, batch 256, AdamW lr 10⁻³, weight decay 0.05, 5% warmup then cosine over 30 epochs, clip 1.0, bf16. **Loss = 14.699 · MSE(ω) + 25 · VICReg variance + 1 · VICReg covariance on the pooled 120-d content latent** | **MISMATCH** (S3-16) |
| FiLM layout | 8-d vector, two groups of four | `dials/split.py`, `BranchHeads` and `FiLMDecoder`. One 8-d vector from a second encoder of the same design (raw input, pooled, d = 8). FiLM h ← h·(1 + γ) + β after the input stage and after each of 3 upsampling stages; γ and β are linear in **all 8** dials; zero-initialized. Two groups of four was M1-G (Stage 1) only | **MISMATCH** (S3-8) |
| Content-noise pipeline order | distort, renormalize, zero k > 21 | `dials/shape_noise.py` @ `afd31a1`: <br>1. RMS-normalize. <br>2. Multiply shells k = 1…21 (round \|k\|) by exp(L_k), with L_k = τ(ε₁σ₁φ₁ + ε₂σ₂φ₂)/2, φ₁ the centred log k and φ₂ the centred φ₁². <br>3. Zero the mean and every shell k > 21. <br>4. RMS-normalize again. <br>Training draws are fresh; evaluation draws are hash-fixed per frame; computed in float64 | **MISMATCH**, minor (S3-12) |
| Zero-mean step | mean removed before measurement | `dials/v3_confirm.py` @ `26921ea`, class `ZeroMean`. Subtracts each decoded field's spatial mean before every scoring measurement, and records per α the pre-removal fraction of frames with > 10% of ω² in the mean. The identification rules (symmetry search, g, PCA) use the raw decoder output (`PREREGISTRATION_v3.md` §5) | OK; add the raw-output point to §6.5 |

---

## Confirmatory vs exploratory: places to fix

**Exploratory or post hoc results presented as if confirmed, or untagged:**
- **Abstract ¶3, "A supervised direction … moves nothing"** (A-12). This is Stage 0 of the build (E),
  presented as a cross-study conclusion.
- **Abstract ¶3, "usable dials had to be identified by their effect …"** (A-13). Exploratory, and
  contradicted for shape on sealed data.
- **§1 ¶1** (I-1). A general causal statement resting on a mix of C and E results.
- **§1 contribution 3** (I-15, I-16). "Usable dials had to be found by their effect" and "decoder
  upsampling … decide[s]" are untagged E results in a list of confirmatory contributions.
- **§1 contribution 4** (I-18). "Every claim is tagged" is not true of the abstract, introduction, §6.2
  or §7.
- **§6.4** (S3-32). "The decoder takes shape from the dial code once that route is the more reliable
  one" is an interpretation; tag it E.
- **§7.1** (D-5). The "after the build" cycle gains are a Stage 2b (E) number, mis-placed in time.
- **§7.2** (D-7). "Whenever …" generalizes E and C results beyond what either shows.

**Tags that are wrong:**
- **§4 opening** (S1-1). "(C, primary)" covers the controls claim, which is C descriptive.
- **§5.4** (S2-27). "(C, secondary)" covers the gap replication, which is C descriptive. H4 is
  secondary.
- **§7.4 header** (D-16). It says all bullets are exploration or post hoc, but the fresh-seeds bullet is
  C descriptive.

**PH tags present where needed:** §4.5 rule robustness, §6.6, and Appendix E. No PH number was found
without its tag.

---

## Claim table

### Abstract

| id | location | claim | source @ commit | status | proposed fix |
|---|---|---|---|---|---|
| A-1 | ¶1 | three studies "each confirmed once on sealed data" | TIMELINE.md §2 @ `8fc7f02`; `configs/final_evaluation.yaml` @ `c2c3a79`; `data/dataset.py` @ `f7237f2` | WORDING | Study 1 was not confirmed (not supported), and its test sets were single-use by protocol, not code-sealed. Write "each tested once on held-out data opened a single time". |
| A-2 | ¶1 | Study 1 JEPA "does not organize its latent around fluid quantities measurably better than an untrained network" | `REPORT.md` @ `31eba06` §1, A5 | WORDING | The JEPA beat the control significantly on three S1 targets (+0.204, +0.310, +0.221). Write "does not meet the pre-registered criteria for organizing its latent around fluid quantities better than an untrained network …". |
| A-3 | ¶1 | the non-physical JEPA and the static model "do as well or better" | `REPORT.md` §2 | OK | — |
| A-4 | ¶2 | autoencoder E1 0.80–0.86 against 0.38; fresh seeds, sealed data | `REPORT_v2.md` @ `a88ddee`; `results/explore/stage3/confirm.md` @ `a88ddee` | OK | (0.799 → 0.80.) |
| A-5 | ¶2 | VICReg lowers prominence "in every comparison" | confirm.md @ `a88ddee` | OK | All six plain/VICReg pairs drop; five have CIs, all above 0. |
| A-6 | ¶2 | the drop is not at the pre-registered margin for vorticity | confirm.md | OK | — |
| A-7 | ¶2 | "while raising linear decodability" | confirm.md | WORDING | Not for the imaginer (log Z R² 0.941 → 0.937, log P R² 0.830 → 0.782). Write "while raising the autoencoders' linear decodability". |
| A-8 | ¶2 | Study 2 verdict partially supported | `REPORT_v2.md` | OK | — |
| A-9 | ¶2 | scale reaches the decoder only through the code; the other pathway has a degraded copy of shape | `dials/split.py` @ `fa7e95d`; `dials/shape_noise.py` @ `afd31a1` | OK | — |
| A-10 | ¶2 | loudness dial in every trained model; shape dial 3 of 5 against a bar of 4; no control shape dial | `REPORT_v3.md` @ `aff659d` | OK | "Every trained model" = the 8 trained confirmation models; consider saying "all 8". |
| A-11 | ¶2 | Study 3 verdict partially supported | `REPORT_v3.md` | OK | — |
| A-12 | ¶3 | supervised direction reads shape almost perfectly and moves nothing | `results/build/stage0.md` @ `ae98ffa`; `stage0.json.gz` @ `ae98ffa` | TAG | Exploratory (Stage 0 of the build); say "in exploration". The supervised direction's readability is \|r\| 0.968–0.991 in trained models, and 0.956–0.958 in an untrained encoder. |
| A-13 | ¶3 | "usable dials had to be identified by their effect on the output rather than by data variance" | `REPORT_v3.md` (PC2 3 of 5 = loudness-null 3 of 5); `results/build/stepA.md` @ `f40a279` | WORDING | Holds for loudness only. Replace: "the loudness dial had to be identified by its effect on the output (closest to a rescaling) rather than by data variance; for shape, a variance rule and an effect-constrained rule each worked in 3 of 5 seeds." |
| A-14 | ¶3 | about 70 GPU-hours on one consumer GPU | NUMBERS.md §7 @ `8fc7f02` | OK | The sum is the pack's. Study 1's 11.5 h is logged wall-clock, including CPU phases. |
| A-15 | ¶3 | "all pre-registrations, deviations, code and outputs are released" | `.gitignore` (checkpoints and data arrays not tracked) | UNSUPPORTED | A future action. True at release for files in the repository. Checkpoints are not in the repository, so the hashes cannot be checked by readers unless the checkpoints are released. |

### 1 Introduction

| id | location | claim | source @ commit | status | proposed fix |
|---|---|---|---|---|---|
| I-1 | ¶1 | latent structure "is set by the training objective and the decoder rather than by learned dynamics" | `REPORT.md` §2; `REPORT_v2.md` | WORDING | A causal and general claim from one architecture. Write "In our experiments, the latent structure … depended more on the training objective and its regularizer than on the dynamics the model was trained to predict." |
| I-2 | ¶1 | "each confirmed once on sealed data" | as A-1 | WORDING | As A-1. |
| I-3 | ¶2 | JEPAs predict in latent space (LeCun; Assran; Bardes) | ext. | UNSUPPORTED | Phase 3. |
| I-4 | ¶2 | Piloto 2022 and Liu & Tegmark 2021 "encouraged reading the latent structure of such models as evidence …" | ext. | UNSUPPORTED | Phase 3. A characterization of others' work; check that it is fair, or soften ("invites"). |
| I-5 | ¶3 | an untrained CNN already exposes many field statistics linearly | `REPORT.md` §6.6, A1 (enstrophy R² 0.727) | OK | — |
| I-6 | ¶5 | quantities can be computed exactly | `data/targets.py` @ `f7237f2` | OK | — |
| I-7 | ¶5 | "Every study was … evaluated once on a sealed test set" | TIMELINE.md §2 | MISMATCH | Study 1's test sets were not code-sealed. Write "on a held-out test set opened once". |
| I-8 | contribution 1 | the JEPA "does not organize its latent … better than an untrained network" | `REPORT.md` §1 | WORDING | As A-2. |
| I-9 | contribution 1 | non-physical JEPA significantly better on Z, P and vortex count; static VICReg beats the JEPA on all four structure targets | `REPORT.md` §2 | OK | — |
| I-10 | contribution 2 | autoencoders make enstrophy prominent, rendering present or future | `REPORT_v2.md` | OK | The future is the Stage 2 imaginer (0.815, C descriptive). |
| I-11 | contribution 2 | VICReg lowers prominence in every comparison while raising decodability | confirm.md | WORDING | Imaginer exception, as A-7. |
| I-12 | contribution 2 | the drop's size held for one target, so partial | `REPORT_v2.md` | OK | — |
| I-13 | contribution 3 | loudness dial in every trained model; shape dial 3 of 5 against 4 | `REPORT_v3.md` | OK | — |
| I-14 | contribution 3 | the shape dial forms only when the content's copy is degraded, never in either control | `REPORT_v3.md`; BUILD_REPORT.md @ `70ec339` §4 | OK | C primary for the controls. |
| I-15 | contribution 3 | "Usable dials had to be found by their effect on the output" | as A-13 | WORDING | As A-13. |
| I-16 | contribution 3 | "decoder upsampling turned out to decide whether strongly steered fields stay clean" | `results/build/stepC.md` @ `f0f6c1f`; `results/build/v3/confirm.md` @ `258d6af` | WORDING | Exploratory, and too strong. On sealed data, a resize-decoder direction still failed the artefact criterion (seed 102's loudness-null direction, 3.5% / 4.7% of frames). Write "in exploration, replacing sub-pixel upsampling with resize-convolution removed the grid artefacts that had disqualified strong dials (E)". |
| I-17 | contribution 4 | every pre-registration, amendment, deviation and sealed-set opening is dated and tied to a commit | TIMELINE.md @ `8fc7f02` | OK | — |
| I-18 | contribution 4 | "Every claim in this paper is tagged" | the draft | WORDING | Many claims are untagged. Write "Every result is tagged …", and tag the results in §7. |
| I-19 | contribution 4 | about 70 GPU-hours | as A-14 | OK | — |
| I-20 | scope ¶ | 64 × 64 periodic decaying flow, one simulator and initial-condition distribution, small CNN | `configs/phase1_data.yaml`; `models/encoder.py` | OK | — |

### 2 Related work

| id | location | claim | source @ commit | status | proposed fix |
|---|---|---|---|---|---|
| R-1 | ¶1 | JEPAs predict latent representations and avoid pixel reconstruction (refs) | ext. | UNSUPPORTED | Phase 3. |
| R-2 | ¶1 | collapse prevented by an EMA target and, here, VICReg-style variance–covariance terms | `PREREGISTRATION.md` §4.1 | OK | — |
| R-3 | ¶1 | Mialon 2023 uses Lie point symmetries as augmentations | ext. | UNSUPPORTED | Phase 3. |
| R-4 | ¶1 | the static control learns invariance to symmetry transformations of one frame and never sees time | `PREREGISTRATION.md` §4.4 | OK | — |
| R-5 | ¶2 | neural simulators and operators; Piloto (violation of expectation); Liu & Tegmark; Greydanus; Alet | ext. | UNSUPPORTED | Phase 3. |
| R-6 | ¶3 | Locatello impossibility; β-VAE; FiLM | ext. | UNSUPPORTED | Phase 3. |
| R-7 | ¶3 | the dial code conditions the decoder through FiLM | `dials/split.py` @ `fa7e95d` | OK | — |
| R-8 | ¶4 | Odena; sub-pixel conv equivalent to strided transposed conv; Aitken's initialization removes artefacts at the start of training | ext. | UNSUPPORTED | Phase 3. Check Aitken's exact claim (checkerboard-free at initialization). |
| R-9 | ¶4 | the same artefacts appear in the tails of strongly steered latents | `results/build/stepA.md`, `stepB.md`; `artefact_calibration.png` @ `2382399` | OK | E. |
| R-10 | ¶5 | Belinkov; Hewitt & Liang control tasks; Elazar amnesic probing | ext. | UNSUPPORTED | Phase 3. |
| R-11 | ¶5 | "Our untrained-network baselines play the role of control tasks" | — | WORDING | Control tasks randomize the labels, not the model. Write "play a role similar to control tasks". |
| R-12 | ¶6 | pre-registered submissions piloted at NeurIPS 2020 and 2021 | ext. | UNSUPPORTED | Phase 3. |
| R-13 | ¶6 | our protocol: everything committed before a sealed test set is opened once by a frozen script | TIMELINE.md §2 | MISMATCH | Study 1 was not code-sealed. Write "before a held-out test set is opened once by a frozen script". |

### 3 Setting and protocol

| id | location | claim | source @ commit | status | proposed fix |
|---|---|---|---|---|---|
| M-1 | §3.1 ¶1 | NS in vorticity form, [0, 2π]², pseudo-spectral 64², 2/3 dealiasing, float64 | `solver/ns2d.py` @ `f7237f2`; `PREREGISTRATION.md` §3 | OK | Add the time stepping (setup table). |
| M-2 | §3.1 ¶1 | unforced and decaying | `configs/phase1_data.yaml` @ `f7237f2` | OK | — |
| M-3 | §3.1 ¶1 | blobs: 2–12 vortices of random sign, size and position | same (`blob_count [2, 12]`, `blob_radius [0.25, 0.6]`) | OK | Optionally add "periodic Gaussian, radius log-uniform 0.25–0.6". |
| M-4 | §3.1 ¶1 | random fields with a prescribed energy spectrum | `PREREGISTRATION.md` §3 | OK | Optionally state E(k) ∝ k⁴ exp(−2(k/k₀)²), k₀ ∈ [2, 5]. |
| M-5 | §3.1 ¶1 | scaled to a target rms vorticity, which sets Z₀ | `configs/phase1_data.yaml` (`omega_rms [2, 4]`, log-uniform) | OK | The matched-energy sets fix E₀ instead; say so where they appear. |
| M-6 | §3.1 ¶1 | ν log-uniform "from about 2 × 10⁻³ to 5 × 10⁻²" | `configs/phase1_data.yaml` (`nu_range [0.003, 0.05]`) | MISMATCH | 3 × 10⁻³ to 5 × 10⁻². |
| M-7 | §3.1 ¶1 | "about a quarter of trajectories are inviscid" | same (`inviscid_fraction 0.25`); LOG 2026-10-15 19:58 | OK | True for Study 1. Add: the later sets raised the inviscid share (378 of about 1,000), and the matched-energy sets are all inviscid. |
| M-8 | §3.1 ¶1 | the lower ν bound is the smallest value resolved "without palinstrophy errors" | LOG 2026-10-14 17:34 (pilot) @ `f7237f2`; `solver/pilot.py` | MISMATCH | The criterion was agreement with 128² runs: 95th-percentile palinstrophy error 6.0% at 3 × 10⁻³ (11.7% at 2 × 10⁻³). Write "the smallest value at which 64² runs track 128² runs (95th-percentile palinstrophy error 6%)". |
| M-9 | §3.1 ¶2 | snapshots every 0.04 | `configs/phase1_data.yaml` | OK | Add the viscous length, 64 snapshots. |
| M-10 | §3.1 ¶2 | inviscid runs truncated at 24 snapshots because enstrophy accumulates in the finest shells | same; `PREREGISTRATION.md` §3 | OK | — |
| M-11 | §3.1 ¶2 | tests for inviscid conservation, dE/dt = −2νZ, an analytic vortex and divergence-free velocity, "which the first four tests miss" a sign error | `tests/test_solver.py` @ `f7237f2`, `test_flipped_advection_sign_is_caught` | MISMATCH | The tests a flipped sign passes are inviscid conservation, the energy budget and two analytic solutions (Taylor–Green, Lamb–Oseen). The divergence-free test is separate. There is also a fourth-order convergence test. Rewrite the list. |
| M-12 | §3.1 ¶2 | three tests catch the sign error: analytic time derivative, vortex-pair rotation, ν recovery from stored frames | same; LOG header (pilots) | OK | "Analytic right-hand side" is the test's name. |
| M-13 | §3.1 ¶3 | splits by trajectory; 2,000 = 1,295 / 279 / 276 / 150; OOD band Re 286–389 | LOG 2026-10-14 18:21; `PREREGISTRATION.md` §3 | OK | — |
| M-14 | §3.1 ¶3 | later sets "with the same recipe", disjoint seeds and hashes "(Section 3.4)" | `tests/test_explore_data.py` @ `26921ea`; `d8c94eb`; `8e8bd1b` | MISMATCH | Not the same recipe: later sets exclude the retired OOD band and raise the inviscid share. §3.4 does not discuss disjointness; fix the cross-reference. |
| M-15 | §3.2 | every model reads (u, v) with one global input scale | `models/encoder.py`; `train/train.py` | OK | The scale is 0.9548 (train velocity std). |
| M-16 | §3.2 | small CNN, Kaiming, no normalization, 8 × 8 × 128, pooled 128-d | `models/encoder.py` @ `f7237f2` | OK | — |
| M-17 | §3.2 | "Every model … The encoder …" | `dials/split.py` @ `fa7e95d` | MISMATCH | Study 3's models use two encoders of this design: content (d = 120), whose input is divided by its own RMS (a per-sample normalization, by design), and dial (d = 8). Scope the sentence to Studies 1–2 and refer to §6.2. |
| M-18 | §3.2 | pilots: instance or group normalization "erased energy from the latent entirely" | LOG header, planning pilots (not reproducible from the repository) | WORDING | Cite it as a pre-project pilot: instance norm, or group/layer norm with bias-free convolutions, gave log-E R² 0.01–0.02. Note that the pilots are not reproducible from the repository. |
| M-19 | §3.2 | the untrained control uses exactly the initialization trained models start from | `models/encoder.py`; `PREREGISTRATION.md` §2.5; `dials/v3_untrained.py` @ `26921ea` | MISMATCH | False for Study 3. Its untrained control re-initializes FiLM randomly; trained dial models start with zero FiLM. Scope to Studies 1–2. |
| M-20 | §3.3 table | decodability "Studies 1–3" | `results/build/stage1.md` (content probes) | OK | Study 3 used probes in exploration only. |
| M-21 | §3.3 Targets | the target list | `PREREGISTRATION.md` §3; `data/targets.py` | OK | — |
| M-22 | §3.3 Targets | vortex cores: Q < −0.2 σ_Q, connected across the periodic boundary, ≥ 4 pixels; OW fraction = vorticity-dominated area | `PREREGISTRATION.md` §3 | OK | "4-connected". |
| M-23 | §3.3 Targets | "probe scores are reported after partialling out time × family" | `PREREGISTRATION.md` §2.1a, §5.3–5.4 | MISMATCH | S1 targets only; the ν probe (S2) is not partialled. Say "S1 probe scores". |
| M-24 | §3.3 Prominence | Ledoit–Wolf shrinkage "on both covariances" | `probes/slow_features.py` @ `f7237f2` | MISMATCH | Shrinkage on Σ_within only; Σ_between comes from trajectory means. |
| M-25 | §3.3 Prominence | "a linear analogue of slow feature analysis" | `probes/slow_features.py` | WORDING | The criterion is within- versus between-trajectory variance (Fisher-type, trajectories as classes), not temporal slowness. Write "related to slow feature analysis and to linear discriminant analysis with trajectories as classes". |
| M-26 | §3.3 Prominence | E1 definition | `PREREGISTRATION_v2.md` §2 @ `8797966` | OK | Add "averaged over seeds". |
| M-27 | §3.3 Prominence | directions fitted on train and val, scored on held-out | `PREREGISTRATION.md` §6; `PREREGISTRATION_v2.md` §2 | OK | — |
| M-28 | §3.3 Prominence | "any latent that encodes enstrophy strongly will show a slow direction tracking it" | confirm.md @ `a88ddee` | MISMATCH | Contradicted by Study 2: the VICReg autoencoders decode log Z at R² 0.93–0.96, yet their E1 is 0.47–0.65. Replace the clause: "… does not mean the model discovered or knows that enstrophy is conserved; and a latent can encode enstrophy well without making it prominent (Section 5.3)." |
| M-29 | §3.3 Steerability | decode z + ασ_d d for α from −2 to +2 | `dials/harness.py` @ `a7e4b32`; `PREREGISTRATION_v3.md` §5 | OK | Add "in steps of 0.5". |
| M-30 | §3.3 Steerability | a = log E; s = log(Z/E) = log of the energy-weighted mean squared wavenumber | `solver/ns2d.py` (conventions) | OK | — |
| M-31 | §3.3 Steerability | rescaling changes a and leaves s fixed; σ_shape definition | `dials/harness.py` | OK | — |
| M-32 | §3.3 bullet 1 | effect ≥ 0.2 per unit α (median over frames); Spearman ≥ 0.9 | entry 2 @ `5ef729a`; `dials/harness.py` | OK | Make it precise (CHECK-1). |
| M-33 | §3.3 bullet 2 | "cycle correlation and cycle gain ≥ 0.5" | entry 2 @ `5ef729a` | MISMATCH | Correlation ≥ 0.9; gain in [0.5, 2] (CHECK-1). |
| M-34 | §3.3 bullet 3 | envelope no worse than an ideal dial (−0.05); ≤ 1% of frames with > 10⁻² power outside the band | entry 2; entry 6 @ `2382399` | OK | Add "at every α" and "at some α" respectively. |
| M-35 | §3.3 bullet 4 | σ_shape ≥ 0.7 shape, ≤ 0.2 loudness | entry 2 | OK | — |
| M-36 | §3.3 bullets | "all of the following" | entry 2 | MISMATCH | Omissions: the shared-pattern (watermark) criterion ≤ 0.5, and the requirement that every criterion hold on both scoring sets (CHECK-1). |
| M-37 | §3.3 last ¶ | floor raised 0.1 → 0.2 after the first stage; artefact criterion calibrated on one case; both committed before use | entry 2 @ `5ef729a`; entry 6 @ `2382399` | OK | The artefact criterion was also applied retroactively to directions scored earlier (said in §8.2). |
| M-38 | §3.4 ¶1 | "the identity of every scored checkpoint (SHA-256) [was] committed before the test data were opened" | `configs/final_evaluation.yaml` @ `c2c3a79` (run names only); `configs/explore/stage3_confirm.yaml` @ `8797966` (39 hashes); `configs/build/v3_frozen.json` @ `fc3b703` (11) | MISMATCH | True for Studies 2–3. Study 1 identified checkpoints by run name. |
| M-39 | §3.4 ¶1 | each test set "sealed in code: the loader refused …" | `data/seal.py` @ `d8c94eb` / `8e8bd1b`; `data/dataset.py` @ `f7237f2` (no guard) | MISMATCH | True for Studies 2–3. Study 1's test splits were single-use by protocol (`PREREGISTRATION.md` §2.7), with no code guard. Its final config was committed before the run, and the code path was dry-run on train/val. |
| M-40 | §3.4 ¶1 | the script opened the set once; outputs committed as written | `e8d619e`, `a88ddee`, `258d6af` | OK | — |
| M-41 | §3.4 ¶1 | predictions with probabilities committed before each exploratory stage, never edited | `results/explore/PREDICTIONS.md`; `results/build/PREDICTIONS.md` | OK | Their author is covered in P-23. |
| M-42 | §3.4 ¶2 | tag definitions | CLAIMS.md @ `8fc7f02` | OK | — |
| M-43 | §3.4 ¶2 | uncertainty as mean ± SD over three or more seeds with paired bootstrap CIs | `REPORT_v3.md`; STAGE2_SUMMARY.md @ `244f625` | MISMATCH | Study 3 reports per-seed counts and an exact binomial interval, not bootstraps. Some exploratory cells rest on one or two seeds (k = 16: 1; k = 1: 2). |

### 4 Study 1

| id | location | claim | source @ commit | status | proposed fix |
|---|---|---|---|---|---|
| S1-1 | §4 ¶1 | not supported; fails all three; two controls do as well or better "(C, primary)" | `REPORT.md` §1–2 | TAG | The verdict is C primary; the controls clause is C descriptive. |
| S1-2 | §4.1 ¶1 | six models share the encoder; three seeds each | `PREREGISTRATION.md` §4 | OK | "Trained" does not apply to the untrained control. |
| S1-3 | §4.1 table: JEPA | predicts the target map from (t − 1, t), k ∈ {1, 4}; EMA; VICReg terms; no pixel loss | `PREREGISTRATION.md` §4.1 | OK | — |
| S1-4 | §4.1 table: pixel, untrained, pooled | as described | `PREREGISTRATION.md` §2.5, §4.2–4.3 | OK | — |
| S1-5 | §4.1 table: static | random shifts and square symmetries | `PREREGISTRATION.md` §4.4 | OK | Circular shift and D4. |
| S1-6 | §4.1 table: non-physical | same initial fields; random damped linear per-mode dynamics; hidden rate per trajectory | `data/nonphysical.py` @ `b18f7ea`; `configs/phase2b_nonphys_data.yaml` @ `b18f7ea` | OK | The modes also rotate in phase (c_k); consider "damped and phase-rotating". |
| S1-7 | §4.1 ¶ | conserve no quadratic or Casimir quantity | `PREREGISTRATION.md` §4.5; `data/nonphysical.py` | OK | — |
| S1-8 | §4.1 ¶ | simpler alternatives failed in pilots (uniform, divergence-free and compressible advection) | none: not in the log's pilot summary, the pre-registration or the code | UNSUPPORTED | Remove, or cite a record. |
| S1-9 | §4.2 S1 | the counting rule, CI, ≥ 3 including OW | `PREREGISTRATION.md` §2.1 | OK | — |
| S1-10 | §4.2 S1 | the OW clause was fixed so that "a gain in amplitude-like quantities could not pass for learned geometry" | `PREREGISTRATION.md` §2.1; `REPORT.md` §1.2 | OK | The pre-registration's reason is narrower: a dissipation gain can come from reading ν off one frame's spectrum. Consider quoting it. |
| S1-11 | §4.2 S2 | frame-pair ν; R² ≥ 0.6 and ≥ control + 0.3 | `PREREGISTRATION.md` §1, §2.2 | OK | Viscous trajectories only. |
| S1-12 | §4.2 S3 | enstrophy \|r\| ≥ 0.8; control < 0.8; gap ≥ 0.1; CI | `PREREGISTRATION.md` §2.3 | OK | Scored for E and Z; energy cannot pass (said in §4.3). |
| S1-13 | §4.2 ¶ | verdict rule; the shift-tolerance rule over 64 sub-stride shifts | `PREREGISTRATION.md` §1, §2.6 | OK | — |
| S1-14 | Table 1 | every value and CI | `REPORT.md` §1, A4, A5; `results/final/final_evaluation.json` @ `e8d619e` | OK | — |
| S1-15 | Table 1 caption | test set, pooled latent, 3 seeds, paired bootstrap; not supported | same | OK | — |
| S1-16 | §4.3 ¶ | vortex-count gain +0.176 shift-averaged; two attributable gains | `REPORT.md` §1.1 | OK | — |
| S1-17 | §4.3 ¶ | energy: the control's "slowest direction" \|r\| 0.948; "energy is the mean square of the input" | `REPORT.md` A1; `PREREGISTRATION.md` §2.3 | WORDING | Best of the three slowest directions; E is half the mean square of the input velocity. |
| S1-18 | Figure 1 caption | content | FIGURES.md @ `8fc7f02` | OK | Stored CIs: `rules.raw.<model>.S1.targets.*.ci95`. |
| S1-19 | Table 2 | every value and CI | `REPORT.md` §2 | OK | — |
| S1-20 | Table 2 caption | C, descriptive | `PREREGISTRATION.md` §1 (controls reported) | OK | — |
| S1-21 | §4.4 ¶ | real NS dynamics added nothing measurable over arbitrary smooth dynamics | `REPORT.md` §2 | OK | — |
| S1-22 | §4.4 ¶ | non-physical JEPA significantly better on Z, P, vortex count; no worse elsewhere | Table 2 | OK | — |
| S1-23 | §4.4 ¶ | static VICReg strongest on every S1 target; the only model meeting S1 (4 of 5 incl. OW) | `REPORT.md` A1, A4 | OK | — |
| S1-24 | §4.4 ¶ | "is generic self-supervised feature quality … the most useful ingredient appears to be invariance to symmetry transformations rather than prediction" | `REPORT.md` §2 | WORDING | Keep the report's alternative: "the symmetry-invariance objective (or its augmentations)". Write "is consistent with generic …". |
| S1-25 | §4.4 ¶ | caveats: slower frame change; augmentations | `REPORT.md` §6.8 (median 0.027 vs 0.038) | OK | — |
| S1-26 | §4.4 ¶ | the spatial map was better "on every measure" | Table 2 (S3 −0.076 [−0.11, +0.09]) | MISMATCH | Better on every S1 target and ν (significant on 4 of 5 S1 targets). The S3 point estimate favours the pooled JEPA. The same overstatement is in `REPORT.md` §2 and the pack's CLAIMS.md 1.4. |
| S1-27 | §4.5 ¶1 | JEPA vortex count 0.752 vs 0.521, OW 0.507 vs 0.270; pixel ν 0.710 vs 0.457 | `REPORT.md` §3 | OK | — |
| S1-28 | §4.5 ¶1 | pixel S3 0.821 vs 0.514, [+0.15, +0.50] (PH) | `REPORT.md` §3, A4 | OK | — |
| S1-29 | §4.5 ¶1 | pixel val loss 0.035× copy, flat; JEPA 0.60×, still falling at epoch 40 | `REPORT.md` §3 | OK | — |
| S1-30 | §4.5 ¶2 | rule variants 1 / 3 / 2; without OW 3 of 5, also non-physical 3 of 5 and static 4 of 5 (PH) | `REPORT.md` §1.2 | OK | — |
| S1-31 | §4.5 ¶2 | OOD error 0.71–1.46× neighbouring bands, every model and target | `REPORT.md` §4 | OK | — |

### 5 Study 2

| id | location | claim | source @ commit | status | proposed fix |
|---|---|---|---|---|---|
| S2-1 | §5 ¶1 | "Rendering the field, not predicting its dynamics, is what makes enstrophy prominent" | confirm.md @ `a88ddee` (streamfunction AE 0.558) | WORDING | Causal, and too general: the plain streamfunction autoencoder renders without regularization yet reaches only 0.558. Write "Models trained to render velocity or vorticity, without a whitening regularizer, make enstrophy prominent; predicting dynamics is not required." |
| S2-2 | §5 ¶1 | the regularizer reduces prominence while raising decodability | confirm.md | WORDING | Imaginer exception (A-7), and the size failed for vorticity. Write "reduces it in every comparison, though not always by the pre-registered margin, while raising the autoencoders' decodability". |
| S2-3 | §5 ¶1 | H1 holds by a wide margin; H2 in direction but not size for one target; partial (C) | `REPORT_v2.md` | OK | — |
| S2-4 | §5.1 ¶1 | the only physics-like positive was unplanned; followed on fresh data; test sets retired | `REPORT.md` §3; `results/explore/BRIEF.md` @ `c11e97a` | OK | E. |
| S2-5 | §5.1 ¶1 | exploration set 1,006 (378 inviscid); matched 300 at E₀ = 0.2893 | LOG 2026-10-15 19:58–19:59; `results/explore/stage1.md` @ `4b9be90` | OK | — |
| S2-6 | §5.1 ¶1 | kill gate committed before any computation; 0.829 vs 0.418; 0.979 vs 0.177 | `results/explore/BRIEF.md` change 5 @ `c11e97a`; stage1.md | OK | — |
| S2-7 | §5.1 ¶2 | imaginer 0.68–0.80, summarizer 0.31–0.46, "with no trend in k" | STAGE2_SUMMARY.md @ `244f625` | MISMATCH | The summarizer rises with k (0.31, 0.37, 0.38, 0.46). Only the imaginer shows no trend, and its k = 16 point is one seed. |
| S2-8 | §5.1 ¶2 | autoencoders ω 0.81, u 0.82, ψ 0.53 | same | OK | — |
| S2-9 | §5.1 ¶2 | + VICReg 0.52 / 0.42 / 0.33 against control 0.42; log P R² 0.68 → 0.93 | same | OK | — |
| S2-10 | §5.1 ¶2 | ±2 SD along the enstrophy direction changed rendered "log enstrophy by only about 1–5%" | `results/explore/probe.md` @ `2187603` | WORDING | The shift moved log Z by about 0.01–0.05, so it is enstrophy, not log enstrophy, that changed by about 1–5%. Name the model (the k = 4 imaginer, context maps shifted). Add that the change was within the random-direction range, with E and P moving together. |
| S2-11 | Figure 2 caption | content | FIGURES.md | OK | — |
| S2-12 | §5.2 ¶1 | four hypotheses; sealed set 1,005 (378 inviscid); fresh, never-evaluated seeds | `PREREGISTRATION_v2.md` §2, §5 | OK | — |
| S2-13 | §5.2 H1, H2 | the rules | `PREREGISTRATION_v2.md` §2 | OK | — |
| S2-14 | §5.2 H3 | "closer to 1" | `PREREGISTRATION_v2.md` §3 | MISMATCH | "Closer to 1 on a log scale", Stage 2 seeds 0–2; E4 definition missing (CHECK-2). |
| S2-15 | §5.2 H4 | the rule | `PREREGISTRATION_v2.md` §3 | OK | Plain imaginer: Stage 2 seeds 0–2. The + VICReg model: new, seeds 0–2. |
| S2-16 | Table 3 | every value and CI | confirm.md @ `a88ddee` | OK | — |
| S2-17 | Table 3 caption | control 0.380 [0.30, 0.46]; partial | confirm.md | OK | — |
| S2-18 | §5.3 ¶1 | per-seed E1 values | `REPORT_v2.md` | OK | — |
| S2-19 | §5.3 ¶1 | VICReg seeds kept more prominence than the exploration seeds | `REPORT_v2.md` (0.62–0.68 vs 0.551) | OK | — |
| S2-20 | §5.3 ¶1 | exploration seeds would have passed H2 (+0.262) | confirm.md | OK | — |
| S2-21 | Figure 3 note | draw from confirm.json | — | OK | Phase 2. |
| S2-22 | §5.3 ¶2 | "VICReg lowered E1 in all five comparisons …, every drop with an interval above zero" | confirm.md | MISMATCH | There are six plain/VICReg pairs on the sealed set. The sixth (streamfunction, Stage 2 seeds: 0.558 → 0.390) has no computed interval. Write "in all six pairs; the five with computed intervals all exclude zero". |
| S2-23 | §5.3 ¶2 | every VICReg autoencoder decoded log Z and log P better; imaginer exception 0.937 / 0.782 vs 0.941 / 0.830 | confirm.md | OK | Correct. `REPORT_v2.md` overstated "every VICReg model". |
| S2-24 | §5.3 ¶2 | velocity 0.862 → 0.933, 0.700 → 0.799; vorticity 0.806 → 0.955, 0.694 → 0.912 | confirm.md | OK | — |
| S2-25 | §5.3 ¶2 | "The regularizer keeps the information but spreads it across the latent, so it no longer dominates the slowest directions" | `REPORT_v2.md`; CLAIMS.md "cannot make" | WORDING | Stated as the mechanism before the rendering confound; CLAIMS forbids "the VICReg drop is purely latent geometry". Write "One reading consistent with this is that …", and keep the confound paragraph right after it. |
| S2-26 | §5.3 ¶3 | rendering about 25× worse (autoencoders), 20× (pixel) | `PREREGISTRATION_v2.md` §7 | OK | — |
| S2-27 | §5.4 | gap +0.483 [+0.42, +0.52]; VICReg removed +0.315, about two-thirds "(C, secondary)" | confirm.md | TAG | The gap replication is C descriptive; H4 is C secondary. |
| S2-28 | §5.4 | "consistent with its regularizer rather than with a failure to learn dynamics" | `REPORT_v2.md` | WORDING | Drop "rather than with a failure to learn dynamics". It was not tested, and Study 1 found that real dynamics added nothing measurable. |
| S2-29 | §5.4 | not shown to be caused; a JEPA without VICReg collapses | STAGE2_SUMMARY.md (threats) | OK | — |
| S2-30 | §5.4 | predicting dynamics was never required | `REPORT_v2.md` | OK | — |

### 6 Study 3

| id | location | claim | source @ commit | status | proposed fix |
|---|---|---|---|---|---|
| S3-1 | §6 ¶1 | dials possible "but only by controlling which pathway carries each property" | BUILD_REPORT.md @ `70ec339` | WORDING | "Only" was not shown. Write "in our build, dials appeared only once we controlled …". |
| S3-2 | §6 ¶1 | loudness dial in all 8 trained models; shape dial 3 of 5 against 4; no control shape dial; partial | `REPORT_v3.md` @ `aff659d` | OK | — |
| S3-3 | §6 ¶1 | section tags E / C / PH | — | OK | — |
| S3-4 | §6.1 | "existing models from Study 2: 21 …, with their four leading unsupervised directions" | `results/build/stage0.md` @ `ae98ffa` | MISMATCH | These are the exploration-stage models (Stage 2 seeds 0–2), not Study 2's fresh confirmation models. The directions are PC1–4 of the pooled latent. |
| S3-5 | §6.1 | none of 84 met the threshold; largest effect 0.086 | stage0.md; BUILD_REPORT.md §3 | OK | The Stage 0 floor was 0.1. |
| S3-6 | §6.1 | supervised s readable at \|r\| 0.97–0.99; barely changed the field | `stage0.json.gz` @ `ae98ffa` (0.968–0.991); stage0.md (effect ≤ 0.004) | OK | Consider adding that an untrained encoder reads s nearly as well (0.956–0.958). |
| S3-7 | Figure 4 caption | content | FIGURES.md | OK | — |
| S3-8 | §6.2 Dial code | "8-dimensional global vector (two groups of four) that conditions every decoder stage" | `dials/split.py` @ `fa7e95d` (`BranchHeads`, `FiLMDecoder`) | MISMATCH | One 8-d vector from a separate dial encoder (same design, raw input, pooled). All 8 dials modulate every stage (the input stage and the 3 upsampling stages). FiLM is zero-initialized. Two groups of four was Stage 1's M1-G. |
| S3-9 | §6.2 Dial code | "per-stage gains act like a spectral equalizer" | — | WORDING | A design motivation, not a measured property. Write "were intended to act like …". |
| S3-10 | §6.2 Scale-free | content sees the input at unit RMS; loudness reaches the decoder only through the code | `dials/split.py` (`ScaleFree`); `dials/shape_noise.py` | OK | — |
| S3-11 | §6.2 Scale-free | "The loudness dial is therefore a design property of these models, not a finding" | `PREREGISTRATION_v3.md` @ `781b2ad` (wording rule) | WORDING | The pre-registration makes the route a design property, and the tested claim that the rule finds a valid dial (untrained models have none). Quote it: "That loudness reaches the decoder only through the dials is a design property; the tested claim is that label-free rules find both dials and that both pass every realism criterion." |
| S3-12 | §6.2 Shape noise | tilt and curvature "(its square)"; renormalized after distortion; shells k > 21 zeroed | `dials/shape_noise.py` @ `afd31a1` | MISMATCH | Order and details: RMS-normalize → gains on shells 1–21 (log-amplitude L_k = τ(ε₁σ₁φ₁ + ε₂σ₂φ₂)/2, φ₂ = centred square of the centred log k) → zero the mean and shells round\|k\| > 21 → RMS-normalize. Shell 21 is the square band's edge, so zeroing round\|k\| > 21 also removes the band's corners. |
| S3-13 | §6.2 Shape noise | draws fresh each step, fixed per frame at evaluation | `dials/shape_noise.py` | OK | — |
| S3-14 | §6.2 Shape noise | "independent noise on each of the 21 shells would largely average out" | none found (entry 8, `shape_noise.py`) | UNSUPPORTED | Remove, or present as untested motivation. |
| S3-15 | §6.2 Decoder | nearest resize + two 3 × 3 convs per stage; 1.45 M; 79% | `dials/split.py`; audit helper (1,447,553 vs 1,834,849) | OK | — |
| S3-16 | §6.2 Decoder and loss | trained to render vorticity only, cosine decay, clipping | `configs/build/v3/b6_main_s100.yaml` @ `26921ea`; `train/train.py` @ `afd31a1` | MISMATCH | Omits the VICReg variance (25) and covariance (1) terms on the pooled 120-d content latent. That matters given Study 2's theme. Add the recipe: AdamW lr 10⁻³, weight decay 0.05, batch 256, 30 epochs, 5% warmup, clip 1.0, bf16. |
| S3-17 | §6.2 ¶2 | shared trunk: no dials "because normalized content still carried loudness (R² 0.99)" | `results/build/stage1.md` @ `4ae7259`; BUILD_REPORT.md §3 | WORDING | R² 0.99 explains why M1-G's loudness knob failed the round trip. Stage 1 had no dial in either model. Say that. |
| S3-18 | §6.2 ¶2 | separate branches: a loudness dial, no shape dial; s at R² 0.95; content supplied shape | `results/build/stage2.md` @ `5ecd024`; BUILD_REPORT.md | OK | E. Add "(later withdrawn by the artefact criterion)". "Because" is an interpretation. |
| S3-19 | §6.2 ¶2 | whitening: shape dial in some seeds; rendering 3.8× the VICReg AE | `results/build/stage2b.md` @ `c9c3e6c` | OK | — |
| S3-20 | §6.3 ¶1 | r = −0.82 between loudness and shape | BUILD_REPORT.md §5 | OK | — |
| S3-21 | §6.3 symmetry | "the residual after the best scalar fit c(α)·decode(z) at each α, minimized over unit directions" | entry 6 @ `2382399`; `dials/stepA.py` @ `2382399` | MISMATCH | J = Σ‖y_α − c·y₀‖² / Σ‖y_α − y₀‖², with c fitted per frame and α. Normalized by the size of the change, because the unnormalized residual is minimized by a direction that changes nothing. |
| S3-22 | §6.3 symmetry | 3.6% unexplained against 31–59% for PC1 | `results/build/stepA.md` @ `f40a279` (J 0.035–0.037; PC1 0.308–0.587) | OK | "3.5–3.7%". |
| S3-23 | §6.3 symmetry | "gave a pure loudness dial (σ_shape 0.02–0.08) with no artefact frames" | stepA.md | MISMATCH | A valid loudness dial in 2 of 3 seeds; seed 2 failed the cycle gain (0.48 on matched). σ_shape and 0% artefacts hold in all three. |
| S3-24 | §6.3 symmetry | 65–79% in the PC1–PC2 plane | stepA.md (cos² 0.79, 0.77, 0.65) | OK | — |
| S3-25 | §6.3 ¶3 | projected-out rule: 50/50 mixtures in every seed (σ_shape 0.50–0.54) | `results/build/stepA_prime.md` @ `6c9e29b` | OK | Explore set; on matched 0.58–0.76. |
| S3-26 | §6.3 ¶3 | exactly orthogonal to loudness in the latent, yet changed loudness about as much as shape | stepA_prime.md (cos² with loudness 0.00) | OK | — |
| S3-27 | §6.3 ¶3 | "The decoder is not linear in the code, so orthogonality has to be imposed on the output" | — | WORDING | Technically off: a linear decoder would not make latent-orthogonal directions output-independent either. Write "Orthogonality in the latent's (z-scored) coordinates says nothing about the decoder's response, so independence has to be imposed on the output." |
| S3-28 | §6.3 loudness-null | g by finite differences "at natural scale" | `PREREGISTRATION_v3.md` §5 | MISMATCH | ±0.5 z-units per coordinate, averaged over 256 exploration frames. |
| S3-29 | §6.3 loudness-null | PC2 frozen as primary; loudness-null reported alongside; chosen after results | entry 8 @ `afd31a1`; `results/build/stepB.md` @ `5234145` | OK | — |
| S3-30 | §6.3 ¶5 | effect-space SVD found directions the data barely use | BUILD_REPORT.md §5; entry 6 | OK | — |
| S3-31 | Table 4 | every value | BUILD_REPORT.md §4; stepB.md, stepC.md, stepD.md | OK | — |
| S3-32 | §6.4 ¶1 | R² 0.97 / σ 0.07; τ = 0.2 R² 0.61 / σ 0.88; "the decoder takes shape from the dial code once that route is the more reliable one" | BUILD_REPORT.md §4 | WORDING | The numbers are OK. The last sentence is an interpretation; mark it as one (E). The τ = 0 point uses the sub-pixel decoder. |
| S3-33 | §6.4 ¶2 | at τ = 0.5 the artefact criterion removed every model that had both dials | BUILD_REPORT.md §3; stepB.md | OK | — |
| S3-34 | §6.4 ¶2 | at τ = 0.5 "loudness dials turned 5–14% of frames into checkerboards at α = +2" | stepA.md @ `f40a279` (M2-SF Stage 2, retroactive); stepB.md @ `5234145` (τ = 0.5 loudness 0–4.3%) | MISMATCH | 5–14% belongs to M2-SF's Stage 2 loudness dials. At τ = 0.5 the loudness directions had 0–4.3% artefact frames. |
| S3-35 | §6.4 ¶2 | shape directions broke up to 9.4% of frames | stepB.md (τ = 0.5, seed 2, loudness-null, explore) | OK | — |
| S3-36 | §6.4 ¶2 | clean in median criteria, failed only in the tails | entry 6; stepA.md | OK | — |
| S3-37 | §6.4 ¶2 | resize-convolution "removed the artefacts completely (0.0% of frames, every direction, every seed)" | stepC.md @ `f0f6c1f`; `results/build/v3/confirm.md` @ `258d6af` | WORDING | True for Step C. On sealed data, one resize-decoder direction exceeded the criterion (seed 102's loudness-null direction, 3.5% / 4.7%), and untrained resize models had artefacts in 100% of frames. Scope the claim to Step C. |
| S3-38 | §6.4 ¶2 | the shape knob survived (σ_shape 0.86–1.00 against 0.92–1.00) | stepC.md; stepB.md | OK | — |
| S3-39 | §6.4 ¶2 | the new decoder differs in depth and parameters | `dials/split.py` | OK | — |
| S3-40 | Figures 5 and 6 captions | content | FIGURES.md | OK | — |
| S3-41 | §6.4 ¶3 | ω error 0.022: 1.5×, 2.1×, 19× | BUILD_REPORT.md §1 | OK | — |
| S3-42 | §6.4 ¶3 | loudness-null 3 of 3, PC2 2 of 3; a committed dose rule picked τ = 0.2 | entry 10 @ `a7e4b32`; `results/build/stepD.md` @ `6a3420d` | OK | — |
| S3-43 | §6.5 ¶1 | the pre-registration fixed the model, five seeds, controls and rules | `PREREGISTRATION_v3.md` §2, §5 | OK | — |
| S3-44 | §6.5 ¶1 | τ = 0 control "expected to show a loudness dial and no shape dial" | `PREREGISTRATION_v3.md` §2 | MISMATCH | State the rule: a shape dial in ≤ 1 of 3 seeds. The loudness-dial expectation was not a rule. |
| S3-45 | §6.5 ¶1 | untrained control with random FiLM, no shape dial | `PREREGISTRATION_v3.md` §2, §5 | OK | The rule is 0 of 3. |
| S3-46 | §6.5 ¶1 | symmetry rule; loudness-null primary; PC2 secondary with its record | `PREREGISTRATION_v3.md` §3, §5 | OK | — |
| S3-47 | §6.5 ¶1 | spatial mean removed before measurement | `dials/v3_confirm.py` @ `26921ea` (`ZeroMean`) | OK | Add that identification used the raw decoder output. |
| S3-48 | §6.5 ¶1 | sealed sets 1,007 (378 inviscid) and 300 | `PREREGISTRATION_v3.md` §4 | OK | — |
| S3-49 | §6.5 ¶1 | pass / partial / fail rule; rendering not gated | `PREREGISTRATION_v3.md` §2, §3 | OK | — |
| S3-50 | Table 5 | per-seed values | `results/build/v3/confirm.md` @ `258d6af`; `REPORT_v3.md` | OK | Seed 102 also failed the envelope on matched energy; "cycle" means correlation and gain. |
| S3-51 | Table 5 caption | exact interval 15–95% | `REPORT_v3.md` | OK | — |
| S3-52 | §6.5 ¶2 | verdict wording "… in most, but not reliably all, training runs …" | `PREREGISTRATION_v3.md` §2 | MISMATCH | The fixed text reads "in most, but not reliably in all, training runs; it would need a more robust identification rule or training before use." Quote it exactly. |
| S3-53 | §6.5 ¶2 | loudness dials "moved a by 0.57–0.65 per unit α" | `REPORT_v3.md` | MISMATCH | 0.57–0.65 is the effect (\|Δa\| + \|Δs\|). Write "effect 0.57–0.65". |
| S3-54 | §6.5 ¶2 | shape dials "moved s by 0.33–0.46" with σ_shape 0.78–1.00 | `REPORT_v3.md` (effect 0.33–0.39 / 0.37–0.46); confirm.md (Δs 0.325–0.386 on confirm3) | MISMATCH | Write "effect 0.33–0.46". |
| S3-55 | §6.5 ¶2 | monotonicity 1.00; cycle correlation 0.95–1.00; gain 0.60–0.97 | `REPORT_v3.md` | OK | — |
| S3-56 | §6.5 ¶3 | τ = 0: loudness 3 of 3, no shape, effects 0.02–0.10; untrained: no dials, artefacts in every frame | `REPORT_v3.md` | OK | — |
| S3-57 | §6.5 ¶3 | shape dial only when the content's copy was degraded; never untrained (C, primary) | `REPORT_v3.md` | OK | — |
| S3-58 | §6.5 ¶4 | PC2: both dials in 3 of 5 (100, 102, 104) | confirm.md | OK | — |
| S3-59 | §6.5 ¶4 | rendering 0.021–0.023; 2.2× VICReg AE; 1.5× τ = 0 | `REPORT_v3.md` | OK | — |
| S3-60 | §6.5 ¶4 | seed 103: zero-mean removed an offset in 23% of frames at α = −2; not computed on sealed data | confirm.md (23.0%) | OK | — |
| S3-61 | Figure 7 caption | exploration frame; seed 102 fails | — | OK | Phase 2. |
| S3-62 | §6.6 ¶1 | post hoc, exploration only, verdict unchanged | `AUTOPSY.md` @ `41b1213` | OK | — |
| S3-63 | §6.6 ¶2 | cos² 0.000; projection removes none of PC2's effect | `results/build/autopsy/autopsy.md` @ `3c55d66` | OK | Add "in the z-scored dial space". |
| S3-64 | §6.6 ¶2 | 2.39 vs 1.48; others within 0.15; σ_d 2.11 vs 4.47; weak and non-monotone | autopsy.md; confirm.md | OK | — |
| S3-65 | §6.6 ¶2 | z-scoring inferred, not tested; seed 104 \|cos\| 0.96, envelope −0.09 on both sets | autopsy.md; `REPORT_v3.md` | OK | — |
| S3-66 | §6.6 ¶3 | union 5 of 5; none in 6 controls; not pre-registered | autopsy.md A2; `REPORT_v3.md` | OK | — |
| S3-67 | §6.6 ¶3 | TM rule: no shape dial; σ_shape 0.89–0.97 "on exploration frames"; effect 0.12–0.17; cycle gain 0.35–0.49 | autopsy.md | MISMATCH | Those ranges are for the explore set. On the exploration matched-energy set: effect 0.18–0.24, gain 0.30–0.45, σ_shape 0.62–0.74. Write "on the explore set". |
| S3-68 | §6.6 ¶3 | "because its objective rewards a change that is well explained rather than one that is large" | `AUTOPSY.md` | WORDING | An interpretation; write "consistent with its objective rewarding …". The low cycle gain was not explained. |
| S3-69 | §6.6 ¶3 | J_T "a clean model-level diagnostic": ≈ 0.10 / 0.34–0.41 / 0.82–0.87 | autopsy.md | WORDING | The values are OK. "Clean" overstates a post hoc reading of 11 models. Write "separated the three groups here". |

### 7 Discussion

| id | location | claim | source @ commit | status | proposed fix |
|---|---|---|---|---|---|
| D-1 | §7 ¶1 | each property "was set more by design choices than by what the model was asked to predict" | — | WORDING | A general interpretation. Write "depended in our experiments more on …". |
| D-2 | §7.1 | the JEPA decoded well but no better than fake-physics or no-time models | `REPORT.md` §2 | OK | It was significantly worse on several targets. |
| D-3 | §7.1 | VICReg raised decodability while lowering prominence | confirm.md | WORDING | Autoencoders only (A-7). |
| D-4 | §7.1 | the supervised direction read shape almost perfectly and moved nothing | stage0.md | OK | E. |
| D-5 | §7.1 | "Even after the build, the supervised shape direction steered worse than label-free directions in exploration (cycle gain 0.20–0.31)" | `results/build/stage2b.json.gz` @ `c9c3e6c` | MISMATCH | The values are the whitened-content model (Stage 2b), explore set (matched 0.14–0.27), mid-build. No supervised direction was measured in the final design. Write "In the whitened-content model (Stage 2b), the supervised shape direction failed the round trip (cycle gain 0.20–0.31, below 0.5)". |
| D-6 | §7.1 | recommendations | — | OK | — |
| D-7 | §7.2 | "enstrophy became prominent whenever a model had to render the field and its latent was not whitened" | confirm.md (streamfunction AE 0.558) | WORDING | "Whenever" is contradicted. Restrict it to rendering velocity or vorticity. |
| D-8 | §7.2 | present-frame autoencoders as strong as future-rendering models | confirm.md (0.860 / 0.799 vs 0.815) | OK | — |
| D-9 | §7.2 | the regularizer "removed about two-thirds of the pixel predictor's advantage" | confirm.md (H4) | WORDING | Write "its advantage over the summarizer at k = 4". |
| D-10 | §7.2 | evaluation recommendations | — | OK | — |
| D-11 | §7.3 ¶1 | disentanglement needs inductive biases (Locatello) | ext. | UNSUPPORTED | Phase 3. |
| D-12 | §7.3 ¶1 | a dial formed when the code was the more reliable route | BUILD_REPORT.md §1 (one-route principle) | OK | E interpretation; tag it. |
| D-13 | §7.3 ¶1 | loudness dial in all 8; shape dial in most seeds; content still readable (R² 0.61) | `REPORT_v3.md`; `results/build/stepD_calibration.md` @ `6a3420d` | OK | R² 0.61 is the exploration calibration. |
| D-14 | §7.3 ¶2 | the designer supplies a transformation, not a formula; loudness identified by its transformation | — | OK | Interpretation. |
| D-15 | §7.3 ¶2 | one attempt (TM) found clean but weak directions | `AUTOPSY.md` | OK | PH. |
| D-16 | §7.4 header | "observations from exploration and post hoc analysis" | — | TAG | The fresh-seeds bullet is C descriptive. |
| D-17 | §7.4 bullet 1 | output-based rules "beat rules based on data variance" | stepC.md (3 of 3 vs 1 of 3); stepD.md (3 of 3 vs 2 of 3); `REPORT_v3.md` (3 of 5 vs 3 of 5) | WORDING | True in exploration only. On sealed data the two shape rules tied. Holds for loudness. |
| D-18 | §7.4 bullet 2 | 5–14% of frames at the extremes; overturned earlier positives | stepA.md | OK | — |
| D-19 | §7.4 bullet 3 | resize-convolution removed the artefacts | stepC.md | OK | E. See S3-37 for scope. |
| D-20 | §7.4 bullet 4 | exploration seeds would have passed H2; fresh seeds did not | confirm.md | OK | C descriptive (D-16). |
| D-21 | §7.5 bullet 1 | 2D: no vortex stretching; inverse cascade (Kraichnan); enstrophy invariant only in 2D | `REPORT.md` §7; ext. | OK | Kraichnan citation: Phase 3. |
| D-22 | §7.5 bullet 2 | one family without normalization; 3–5 seeds per configuration; CIs resample trajectories | STAGE2_SUMMARY.md; `dials/split.py` | MISMATCH | Some exploratory cells rest on 1–2 seeds. The Study 3 content branch normalizes its input per sample. |
| D-23 | §7.5 bullet 3 | method-relative prominence | `probes/slow_features.py` | OK | — |
| D-24 | §7.5 bullet 4 | VICReg degraded rendering 20–25× | `PREREGISTRATION_v2.md` §7 | OK | — |
| D-25 | §7.5 bullet 5 | "The loudness dial exists by construction" | `PREREGISTRATION_v3.md` wording rule | WORDING | As S3-11: the route exists by construction; finding the valid dial was the test. |
| D-26 | §7.5 bullet 5 | shape: known transformation; 2.2×; rule chosen in exploration; 3 of 5; dose-curve limits | `REPORT_v3.md`; BUILD_REPORT.md | OK | — |
| D-27 | §7.5 bullet 6 | adaptive criteria, each committed before confirmatory use | TIMELINE.md | OK | — |
| D-28 | §7.6 | 3 of 5 → 15–95%; at least ten seeds | `REPORT_v3.md` | OK | Even 8 of 10 has an interval of 44–97% (`AUTOPSY.md`); consider saying so. |

### 8 Process, reproducibility and AI assistance

| id | location | claim | source @ commit | status | proposed fix |
|---|---|---|---|---|---|
| P-1 | §8 ¶1 | "about two and a half weeks" | TIMELINE.md (2026-10-14 to 2026-10-29) | MISMATCH | 16 days (17 including the 10-13 pilots). Write "about two weeks (14–29 October 2026)", or leave dates out for anonymity. |
| P-2 | §8 ¶1 | "Appendix F lists every deviation with its date and commit" | Appendix F | MISMATCH | Appendix F has neither dates nor commits. Add them from TIMELINE.md, or point to TIMELINE.md. |
| P-3 | §8.1 ¶1 | three pre-registrations, each committed before its test data were opened | TIMELINE.md §1 | OK | — |
| P-4 | §8.1 ¶1 | Study 1 pre-registered before training; amendment 7 minutes later after two smoke runs, before evaluation | `f7237f2` 18:46, `1671301` 18:53; LOG 18:49–18:50 | OK | — |
| P-5 | §8.1 ¶1 | no amendments for Studies 2 and 3 | `PREREGISTRATION_v2.md`, `PREREGISTRATION_v3.md` (empty AMENDMENTS) | OK | — |
| P-6 | §8.1 ¶1 | "Every sealed set was opened exactly once by a frozen script", including Study 1's | TIMELINE.md §2 | MISMATCH | Study 1's sets were single-use by protocol, not code-sealed. Write "every held-out set". |
| P-7 | §8.1 ¶1 | Study 1 and 2 sets retired; Study 3 sets spent | TIMELINE.md §2 | OK | — |
| P-8 | §8.1 ¶2 | DRAFT status line; binds on commit; approvals in commit messages; seal code checks | commit messages `8797966`, `781b2ad`; `data/seal.py` | OK | — |
| P-9 | §8.2 Study 1 | val seen before test; only the encoding fix and an unscored readout changed after; config committed before | `REPORT.md` §6.9, §10 | OK | — |
| P-10 | §8.2 Exploration | "11 of 24 imaginer-family runs diverged" | LOG: 30 Stage 2 imaginer-family entries | MISMATCH | 11 of 30 (CHECK-3). |
| P-11 | §8.2 Exploration | "the eight that diverged at 3 × 10⁻³ were rerun at 10⁻³ …" [CHECK] | entry 5 @ `33ba39b`; LOG 2026-10-16 11:34 | MISMATCH | All 15 were rerun at 10⁻³, and the three later divergences were excluded (CHECK-3). |
| P-12 | §8.2 Exploration | plan revised once before evaluation; the post hoc probe is labelled | entries 6 and 7 (`63d27e9`, `9114f97`) | OK | — |
| P-13 | §8.2 Build | floor, envelope and cycle-gain changes after the first stage, before the training they governed | entry 2 @ `5ef729a` | OK | — |
| P-14 | §8.2 Build | artefact criterion calibrated on one case, applied later and retroactively, withdrew loudness dials | entry 6; stepA.md | OK | — |
| P-15 | §8.2 Build | crash from the GPU median; "several power outages" | LOG 2026-10-22 18:20; 2026-10-23 | MISMATCH | One outage during the build (Step B). Two more came during Study 3's training. |
| P-16 | §8.2 Study 3 | ≈ 0.7 GPU-h lost; 11 checkpoints frozen and verified | `REPORT_v3.md`; `fc3b703` | OK | — |
| P-17 | §8.2 last ¶ | "+dirty" explained; the tree was clean when each run started | LOG; `REPORT.md` §10; `REPORT_v2.md`; `REPORT_v3.md` | OK | — |
| P-18 | Table 6 | compute per line; total ≈ 70 | NUMBERS.md §7 | OK | Study 1's 11.5 is logged wall-clock, including CPU phases; consider "hours (logged)". |
| P-19 | Table 6 caption | ~70 CPU pilots not included | LOG header | OK | — |
| P-20 | §8.5 | "The author conceived the research question and made every decision on hypotheses, decision rules, controls, budgets, sealed-set openings and the interpretation of verdicts" | briefs committed "verbatim" (`c11e97a`, `08c1511`); BUILD_REPORT.md §7 ("draft v3 design choices for your decision"); entry 6 (the agent normalized the investigator's objective); commit messages "approved by the investigator" | WORDING | The record supports "approved". Suggested: "The author conceived the research question, wrote the briefs, and approved every hypothesis, decision rule, control, budget and sealed-set opening; several designs and rule definitions were proposed by the coding agent and accepted, modified or rejected by the author. Verdict texts were fixed in advance and applied by the frozen scripts." |
| P-21 | §8.5 | a chat assistant was used "to propose and critique experimental designs and to draft pre-registration and brief text" | `PREREGISTRATION_v2.md` / `_v3.md` status lines ("DRAFT for the investigator's review"); all 73 commits carry the coding agent's co-author trailer; briefs committed as "the investigator's brief verbatim" | UNSUPPORTED | Chat use is outside the repository record; only the author can attest to it. The record shows the pre-registrations were drafted in the repository by the coding agent and approved by the author before commit. State both. |
| P-22 | §8.5 | the coding agent wrote and ran the code, kept the log and drafted the stage reports | git history (73 of 73 commits co-authored) | OK | — |
| P-23 | §8.5 | "the calibration probabilities in the Study 3 pre-registration were written by the agent and are marked as such" | `PREREGISTRATION_v3.md` §9 ("Expectations (mine …)"); both PREDICTIONS.md files | MISMATCH | The agent wrote the probabilities in every prediction entry (exploration entries 1–7, build entries 1–11), except predictions marked as the investigator's (e.g., build P1.6). §9 is marked "mine", in the agent's voice; it does not name the agent. |
| P-24 | §8.5 | "every number in it is traced to a committed output, and every reference was checked against its source" | this audit | UNSUPPORTED | Not true yet. Keep it only after these fixes and the Phase 3 reference check. |
| P-25 | §8.5 | TMLR's policy allows language models as general-purpose assistive tools | ext. | UNSUPPORTED | Phase 3: quote TMLR's current policy text. |

### Appendices

| id | location | claim | source @ commit | status | proposed fix |
|---|---|---|---|---|---|
| AP-A1 | Table A1 | all 48 values | `REPORT.md` A1 @ `31eba06` | OK | — |
| AP-A2 | Table A1 caption | R² after partialling; only static meets S1; only pixel meets S3; none meets S2 | `REPORT.md` A4 | OK | The ν-pair row is not partialled, and the S3 rows are \|r\|, not R². |
| AP-B1 | Table B1 | every row | NUMBERS.md §2; STAGE2_SUMMARY.md; stage1.md; LOG 2026-10-15 20:42 | OK | "24.5 GPU-hours" is logged wall-clock. |
| AP-B2 | Table B1 caption | horizon confounded with training length; k = 16 one seed | STAGE2_SUMMARY.md | OK | — |
| AP-C1 | Table C1 | every row | BUILD_REPORT.md §3 @ `70ec339` | OK | — |
| AP-C2 | Table C2 | every value | BUILD_REPORT.md §4 | OK | — |
| AP-D1 | Table D1 | every row | `REPORT_v3.md`; confirm.md @ `258d6af` | OK | — |
| AP-D2 | Table D1 caption | C, except where marked | — | OK | — |
| AP-E1 | Table E1 | every value | autopsy.md @ `3c55d66` | OK | — |
| AP-E2 | Appendix E text | TM definition; no shape dial in 11 models; J_T ranges; 102 flagged (0.69 vs 0.19–0.35); 104 not (0.27); seed 103 unchanged; 16.4%; 2.27 GPU-h | autopsy.md; `AUTOPSY.md` | OK | 16.4% is the explore set at the worst α (0% on matched). |
| AP-F1 | Table F1: pilots, A1, crash, val-before-test | as stated | TIMELINE.md | OK | — |
| AP-F2 | Table F1: launch aborted | as stated | LOG 2026-10-15 21:13 | OK | — |
| AP-F3 | Table F1: budget raised four times, 14 → 30 | as stated | `results/explore/BRIEF.md` changes 11–14 | OK | — |
| AP-F4 | Table F1: plan revised; contrast probe post hoc | as stated | entries 6, 7 | OK | — |
| AP-F5 | Table F1: Study 2 change before unsealing | as stated | BRIEF change 14 @ `63e0ad2` | OK | — |
| AP-F6 | Table F1: build rows (Stage 0 reruns; floor, envelope and cycle gain; median crash; artefact criterion; shape-rule changes; outage) | as stated | TIMELINE.md | OK | — |
| AP-F7 | Table F1: divergences "(11 of 24 …)" | — | LOG | MISMATCH | 11 of 30 (CHECK-3). |
| AP-F8 | Table F1: Study 3 outages and pause; seed 103 rerun declined, done on exploration data | as stated | TIMELINE.md | OK | — |
| AP-F9 | Table F1 | completeness ("deviations and incidents") | TIMELINE.md @ `8fc7f02` | MISMATCH | Missing items: <br>• build budget raises (15 → 18 → 18.5); <br>• the build brief's VICReg-on-dials dose curve, replaced by the content-side dose (entry 8); <br>• the build brief's rendering trigger, not applied in Study 3 (stated in advance); <br>• exploration entry 6 dropping the fake-physics and pixel + VICReg models (the latter reinstated as H4); <br>• Study 3's untrained control moved to random FiLM (decided before pre-registration). |
| AP-F10 | Table F1 caption | every entry dated and tied to a commit in TIMELINE.md | TIMELINE.md | OK | Contradicts §8 ¶1 (P-2). |

---

## Corrections found in the paper-materials pack (not applied; your call)

The audit found errors that the pack carried over from the committed reports:
- **`NUMBERS.md` §2:** "Divergences … 11 of 24" should be 11 of 30. It comes from `STAGE2_SUMMARY.md`.
- **`CLAIMS.md` 1.4:** "better on every measure" is not supported. The S3 point estimate favours the
  pooled JEPA. It comes from `REPORT.md` §2.
- **`CLAIMS.md` 2.4 and E.4:** "while raising linear decodability" holds for the autoencoders, not the
  imaginer. `REPORT_v2.md` says "every VICReg model", which is wrong for the imaginer.

The committed reports themselves are records and stay as they are.

## Notes for Phases 2–3

- **Anonymity.** The byline reads "Working draft … · @manvendra". Remove it in the anonymized build.
  Commit hashes in the text are not identifying in themselves, but they link to a named repository.
- **Figures.** The captions check out against FIGURES.md. Figure 7 is to be drawn on an exploration
  frame (the sealed set is not reopened).
- **References.** Every "ext." row above goes to the Phase 3 reference check, as do the [CHECK] note on
  references and the TMLR policy statement (P-25).
