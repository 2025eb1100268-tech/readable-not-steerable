# Paper materials: claims

These are the claims the paper can make, each tagged, with the evidence pointer (NUMBERS.md section)
and the wording constraint that applies. The wording rules are quoted exactly from the source that set
them. These are claim statements for the paper to draw on, not paper prose.

## Tags

- **C: confirmatory.** The rule or comparison was fixed in a committed pre-registration before the data
  were opened, and evaluated once on sealed or single-use data. It comes in three roles:
  - **primary**: decides the verdict;
  - **secondary**: rule stated in advance, not part of the verdict;
  - **descriptive**: pre-registered to be reported, with no rule.
- **E: exploratory.** From exploration stages or data, labelled exploratory at the time. Predictions were
  usually written beforehand, but stages and rules were chosen adaptively.
- **PH: post hoc.** The analysis, rule or reading was chosen after seeing the results it concerns, even
  when it was computed on confirmatory data.

## Wording rules, verbatim

| line | rule | source @ commit |
|---|---|---|
| Study 1 | "The hypothesis is **supported** if all of the following hold on test-ID, averaged over 3 seeds: … **Partially supported** if S1 holds but S2 or S3 fails. **Not supported** if S1 fails." | `PREREGISTRATION.md` §1 @ `f7237f2` |
| Study 1 | "**Interpretation rule:** if a JEPA gain over the control disappears in the shift-averaged readout, the report attributes it to shift-tolerance, not physical structure." | `PREREGISTRATION.md` §2.6 @ `f7237f2` |
| Study 1 | "An S2 pass alone does not show ν was inferred from dynamics." | `PREREGISTRATION.md` §2.2 @ `f7237f2` |
| Study 1 | (pixel S3 result) "This is reported, unscored, and was not predicted. Treat it as a lead, not a finding." | `REPORT.md` §3 @ `31eba06` |
| Study 1 | "Even a positive S3 here would not have shown that such models find invariants of 3D flow. The negative result doesn't transfer either. It says nothing about larger models, other readouts, or 3D." | `REPORT.md` §7 @ `31eba06` |
| Exploratory line | "**Framing rule (agreed):** nothing in this line of work may claim a model 'discovered conservation'. The supportable claim is about which conserved quantities are *prominent in the latent*." | `results/explore/BRIEF.md` @ `c11e97a` |
| Exploratory line | "Everything in Stages 0–2 is exploratory. … Nothing from Stages 0–2 is a 'finding'. Only Stage 3 can produce a claim." | `results/explore/BRIEF.md`, hard rule 1 @ `c11e97a` |
| v2 | "Wording rule: these statistics say which conserved quantities are **prominent in a latent** (its slowest unsupervised directions). They are not claims that a model discovered conservation, nor measures of what a model knows." | `REPORT_v2.md` @ `a88ddee` (same in `PREREGISTRATION_v2.md` @ `8797966`) |
| Build | "**Wording:** a 'dial' is a latent direction whose movement produces a measured change in the decoded field. No claims about a model 'knowing' or 'understanding' anything." | `results/build/BRIEF.md`, hard rule 8 @ `08c1511` |
| Build | "M1-G's loudness dial exists by construction and is reported as a design property, not a finding; the finding is whether a label-free shape dial emerges." | `results/build/BRIEF.md`, agreed change 1 @ `08c1511` |
| Build | "**Status: exploratory.** Nothing in this report is confirmed." | `results/build/BUILD_REPORT.md` @ `70ec339` |
| v3 | "Nothing here says a model 'knows' or 'understands' anything. That loudness reaches the decoder only through the dials is a design property of these models (their content branch cannot see overall scale); the tested claim is that pre-registered, label-free rules **find** both dials and that both **pass every realism criterion**, reliably across fresh training runs." | `PREREGISTRATION_v3.md` @ `781b2ad` |
| v3 | Partial verdict text, fixed in advance: "*Partially supported:* the design produces both dials in most, but not reliably in all, training runs; it would need a more robust identification rule or training before use." | `PREREGISTRATION_v3.md` §2 @ `781b2ad` |
| v3 | (union of shape rules) "No combined rule was pre-registered, and picking the better rule for each seed would mean looking at the scores. This observation therefore counts for nothing in the verdict." | `REPORT_v3.md` @ `aff659d` |
| Autopsy | "Status: exploratory and post hoc. Nothing here changes REPORT_v3's verdict." | `AUTOPSY.md` @ `41b1213` |

## Study 1: JEPA latent geometry (NUMBERS §1)

| id | claim | tag | constraint |
|---|---|---|---|
| 1.1 | The pre-registered hypothesis is **not supported**. S1 fails (the mandatory Okubo–Weiss target does not count), S2 fails (no gain in decoding ν), and S3 fails (enstrophy is less prominent in the JEPA's slow directions than in the untrained control's). | C primary | Test-ID, raw pooled latent, 3 seeds; CIs resample trajectories, not seeds. |
| 1.2 | Of the JEPA's three counted S1 gains, the vortex-count gain is attributed to shift tolerance by the pre-registered rule, leaving two (enstrophy, palinstrophy). | C primary (§2.6 rule) | — |
| 1.3 | A JEPA trained on non-physical linear dynamics reads enstrophy, palinstrophy and vortex count significantly better than the JEPA trained on Navier–Stokes. A static VICReg model with no time beats it on all four structure targets. | C descriptive | Pre-registered controls, no pass/fail. The non-physical task is easier per frame, and static VICReg sees augmentations the JEPA does not (`REPORT.md` §6.8). |
| 1.4 | Predicting the spatial map is better than predicting the pooled vector on every S1 target and on ν (significantly on 4 of 5 S1 targets); the S3 point estimate favours the pooled JEPA (−0.076 [−0.11, +0.09]). | C descriptive | Corrected 2026-11-03 (see Corrections). Do not write "on every measure". |
| 1.5 | The JEPA and the pixel predictor keep different information: the JEPA keeps vortex count and Okubo–Weiss; the pixel model keeps ν and an enstrophy-tracking slow direction. | C descriptive | Pre-registered comparison, no pass/fail. |
| 1.6 | No model breaks down out of distribution (an interior Re band): OOD / neighbour MSE ratios are 0.71–1.46. | C descriptive | — |
| 1.7 | The verdict does not hinge on the rule version: S1 fails under every version except dropping the Okubo–Weiss clause, which would also pass the non-physical and static controls. | PH | The rule variants are a robustness reading done after the test run. |
| 1.8 | The pixel predictor meets the S3 rule for enstrophy (0.821 vs 0.514, gap CI [+0.15, +0.50]). | PH | "a lead, not a finding". It motivated the exploratory line; v2 H1 later confirmed a related claim for autoencoders (2.1), not this one. |

## Exploratory line (NUMBERS §2): exploratory claims only

| id | claim | tag | constraint |
|---|---|---|---|
| E.1 | Probing the spatial maps did not reveal enstrophy prominence that pooling hid (JEPA: pooled 0.353, map mean ⊕ std 0.303). | E | The PCA-512 map readout is not a superset of the pooled latent (diagnostic). |
| E.2 | The pixel predictor's enstrophy prominence survives on fresh data and at matched energy (kill gate passed). | E | — |
| E.3 | Prominence does not grow with prediction horizon. A plain autoencoder that renders the present reaches the imaginer's level. | E | Horizon is confounded with training length; k = 16 rests on one seed. |
| E.4 | Adding VICReg to autoencoders drops enstrophy prominence to control level while raising the autoencoders' linear decodability. | E | Confirmed in part by v2 (2.2, 2.4). Applies to autoencoders only; scope made explicit 2026-11-03 (see Corrections). |
| E.5 | Shifting the latent along the enstrophy direction barely changes the rendered field ("prominent is not the same as used"). | E (imagination probe); PH (contrast probe) | P6.1 failed in all 3 seeds. |

## v2: rendering and regularization (NUMBERS §3)

| id | claim | tag | constraint |
|---|---|---|---|
| 2.1 | **H1 holds.** On sealed data with fresh training seeds, plain autoencoders rendering velocity or vorticity make enstrophy prominent in their slowest unsupervised directions: E1 0.86 and 0.80 against 0.38 for the untrained control. | C primary | "Prominent in a latent", not "discovered conservation". The result is about prominence under this method (z-score → PCA 32 → Ledoit–Wolf within/between). |
| 2.2 | **H2 fails.** VICReg lowered E1 for both render targets, but the pre-registered drop of ≥ 0.20 held for velocity (+0.338) and not for vorticity (+0.149). | C primary | Do not state "VICReg removes prominence" for both targets. VICReg also costs about 25× in rendering, so the drop cannot be separated from worse rendering. |
| 2.3 | **Verdict: partially supported.** | C primary | — |
| 2.4 | VICReg lowered E1 in all six plain/VICReg pairs on the confirm set (the five with computed CIs all above 0), while raising the autoencoders' linear decodability (log Z and log P R²): prominence and decodability dissociate. The imaginer with VICReg did not decode better (log Z R² 0.937 vs 0.941; log P R² 0.782 vs 0.830). | C descriptive | The size depends on the render target and the seeds. Corrected 2026-11-03 (see Corrections). |
| 2.5 | **H4 holds** (secondary): the pixel predictor with VICReg drops from 0.815 to 0.500 (+0.315), about two-thirds of the imaginer-vs-summarizer gap (0.483). **H3 holds** (secondary): the vorticity AE's E4 is 1.00 vs 3.13 for streamfunction. | C secondary | Not part of the verdict. H4 carries the same rendering confound (about 20×). |
| 2.6 | The imaginer-vs-summarizer gap at k = 4 replicates on sealed data (+0.483 [+0.42, +0.52]). | C descriptive | Uses the frozen Stage 2 seeds, not fresh ones. |
| 2.7 | Study 1's JEPA result is *consistent with* its VICReg regularizer; predicting dynamics was never required for the effect. | C secondary (rests on H4) + interpretation | "Consistent with", not "caused by". A JEPA without VICReg collapses, so the summarizer could not be tested unregularized. |
| 2.8 | Fresh seeds mattered: the Stage 2 vorticity seeds would have passed H2 on the same data (+0.262). | C descriptive | — |

## The dial build (NUMBERS §4): exploratory claims only

| id | claim | tag | constraint |
|---|---|---|---|
| B.1 | In 21 existing models, no latent direction is a steering dial. A supervised shape direction that predicts s almost perfectly moves nothing: readable is not steerable. | E | Stage 0 criteria (entry 1). |
| B.2 | A shape dial appeared only once the content branch could no longer supply the field's spectral shape (the one-route principle). | E | Adaptive sequence of designs; several choices were made after seeing results. |
| B.3 | Sub-pixel-convolution (pixel-shuffle) upsampling produced grid artefacts in the steered tails; resize-convolution removed them (0.0% artefact frames). | E | The artefact threshold was calibrated after seeing data on one known-bad case (entry 6). |
| B.4 | Reducing the content's shape signal turns the loudness-null direction into a shape knob, and rendering error rises with dose. | E | Do not claim a jump: one intermediate point, and the curve spans two decoders. |
| B.5 | In exploration, the final design gave both dials in 3 of 3 seeds under the loudness-null rule, and 2 of 3 under PC2. | E | The loudness-null rule was chosen after seeing results; PC2 was the frozen exploratory primary. |
| B.6 | Loudness found by its effect (the rescaling symmetry) separates it from shape, where the top PCA direction mixes the two (r = −0.82 in the data). | E | — |

## v3: confirmation of the dial build (NUMBERS §5)

| id | claim | tag | constraint |
|---|---|---|---|
| 3.1 | **Verdict: partially supported.** Both dials are valid on both sealed sets in 3 of 5 fresh main-model seeds; the pass rule needed 4. | C primary | Use the verdict text fixed in advance (above). The exact 95% interval for 3 of 5 is 15–95%; it is not a precise rate. |
| 3.2 | The label-free symmetry rule finds a loudness dial in every trained model (5 of 5 main, 3 of 3 τ = 0) on both sealed sets, with σ_shape ≤ 0.05. | C primary (component of H1) | Partly a design property: the content branch cannot see overall scale. The claim is that the rule *finds* it and that it passes realism. |
| 3.3 | The pre-registered loudness-null rule finds a valid shape dial in 3 of 5 fresh seeds. One failure is clear (102: weak, non-monotone, artefacts); one is narrow (104: envelope −0.09 on one set). | C primary (component of H1) | — |
| 3.4 | Both controls stay within their pre-registered rules: no shape dial in 0 of 3 τ = 0 models and 0 of 3 untrained models. The shape dial appears only when the content branch cannot carry shape, and never in an untrained network. | C primary (control rules) | Attribution is to the content-noise design under these rules. It is not "the network invents a shape dial unaided". |
| 3.5 | With PC2 as the shape rule, both dials are valid in 3 of 5 seeds (100, 102, 104). | C secondary | Not in the verdict. |
| 3.6 | The dial model renders about 2.2× worse than the VICReg autoencoder and about 1.5× worse than its τ = 0 version. | C descriptive | Reported, not gated (stated in advance). The build brief's rendering trigger would fire. |
| 3.7 | In one passing seed (103), the pre-registered zero-mean step removed a sizeable spatial-mean offset (23% of `confirm3` frames at α = −2). | C descriptive | Whether 103 passes on `confirm3` without the removal was never computed. The exploration analogue is 7.4. |
| 3.8 | The two shape rules fail on different seeds and together cover all 5. | PH | "counts for nothing in the verdict". |

## Autopsy (NUMBERS §6): post hoc only

| id | claim | tag | constraint |
|---|---|---|---|
| 7.1 | Seed 102's failure is not explained by shape lying along the loudness gradient: cos²(g, PC2) = 0.000, and projecting out g removes none of PC2's shape effect. | PH | Exploration data. |
| 7.2 | In 102, after z-scoring and projection, the top direction is not PC2's projection (variance 2.39 vs 1.48); the chosen direction has half PC2's raw spread. | PH | That z-scoring *causes* the failure is inferred, not tested. |
| 7.3 | Seed 104's envelope miss reproduces on exploration matched-energy data (−0.09), so it belongs to the direction and is not chance in the sealed set. | PH | — |
| 7.4 | Seed 103's classification is unchanged with the mean kept, on exploration sets. | PH | This is not the `confirm3` condition. |
| 7.5 | A transformation-matched shape rule (post hoc) finds no shape dial in any model: its directions are clean but weak (effect 0.12–0.24, cycle gain 0.30–0.49). Its objective, J_T, separates main (≈ 0.10), τ = 0 (0.34–0.41) and untrained (0.82–0.87) models. | PH | J_T is a post hoc diagnostic, not a rule. |
| 7.6 | The union of the two shape rules finds no shape dial in any control (0 of 6). | PH | Two attempts per seed, and only 6 controls. |

## Claims the paper cannot make

| claim | why not | source |
|---|---|---|
| Any model "discovered conservation", "knows" or "understands" anything | Wording rules of every line | above |
| The JEPA learned fluid dynamics, or predicting real dynamics added anything measurable | Study 1 is not supported, and the controls match or beat the JEPA | 1.1, 1.3 |
| VICReg lowers prominence by ≥ 0.20 for every render target | H2 failed for vorticity | 2.2 |
| The VICReg drop is purely latent geometry | It is confounded with about 20–25× worse rendering | 2.2, 2.5 |
| Imagining the future, or horizon, is what makes enstrophy prominent | Exploration showed rendering the present suffices; not confirmed as such | E.3, 2.1 |
| The dial design reliably yields both dials | v3: 3 of 5, partial | 3.1 |
| A combined shape rule finds both dials in every seed | Post hoc; no fresh-seed test | 3.8, 7.6 |
| The shape dial emerges without design help | Partly by design (scale-free content, content noise); the controls attribute it to the noise | 3.2, 3.4 |
| The dose response is a switch or jump | One intermediate point, and two decoders | B.4 |
| Seed 102 failed because of z-scoring | Inferred, not tested | 7.2 |
| Seed 103 passes `confirm3` without mean removal | Never computed on `confirm3` | 3.7, 7.4 |
| Any result generalizes beyond 2D decaying flow, this simulator and initial-condition distribution, or this small CNN family | Every report's scope statement | `REPORT.md` §6.10, §7; `REPORT_v2.md`; `REPORT_v3.md` |

---

## Corrections

All three were found by `results/paper/AUDIT_v1.md`. The committed reports they came from are left
unchanged.

**2026-11-03, claim 1.4.**
- **What changed:** "on every measure" now reads "on every S1 target and on ν (significantly on 4 of
  5 S1 targets); the S3 point estimate favours the pooled JEPA (−0.076 [−0.11, +0.09])".
- **Why:** `REPORT.md` @ `31eba06` §2 table, JEPA − pooled JEPA:
  - S3 enstrophy \|r\| −0.076, CI spans zero;
  - log dissipation +0.028 [−0.006, +0.061], also spans zero.
  The old wording repeated `REPORT.md` §2's "worse everywhere".
- **Found by:** audit row S1-26.

**2026-11-03, claim 2.4.**
- **What changed:**
  - "while raising linear decodability" now reads "while raising the autoencoders' linear
    decodability", with the imaginer exception stated.
  - "every comparison … (5 of 5 …)" now reads "all six plain/VICReg pairs (the five with computed CIs
    all above 0)".
- **Why:** `results/explore/stage3/confirm.md` @ `a88ddee`.
  - The imaginer + VICReg has log Z / log P R² 0.937 / 0.782, against 0.941 / 0.830 for the plain
    imaginer. `REPORT_v2.md` says "every VICReg model", which is wrong for the imaginer.
  - The sixth pair (streamfunction autoencoder, Stage 2 seeds: E1 0.558 → 0.390) also drops but has
    no computed interval.
- **Found by:** audit rows A-7, S2-22 and S2-23.

**2026-11-03, claim E.4.**
- **What changed:** "raising linear decodability" now reads "raising the autoencoders' linear
  decodability".
- **Why:** the exploratory claim was already about autoencoders, since no imaginer + VICReg model was
  trained in exploration. The edit only makes the scope explicit, so it cannot be read as covering the
  imaginer, whose v2 result is the exception in 2.4.
