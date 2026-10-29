# Paper materials: figures

These are existing figures, in story order. Nothing has been regenerated.

- **Caption** is one line.
- **Status** gives the claim tag from CLAIMS.md.
- **Flag** says whether the figure needs regenerating for publication quality, and why.
- All paths are relative to the repo root; the commit is the one holding that version of the file.

## Main sequence

| # | figure | path @ commit | status | caption | flag |
|---|---|---|---|---|---|
| 1 | Study 1 scorecard | `results/final/summary.png` @ `31eba06` | C | Every pre-registered S1–S3 quantity on test-ID, per model and seed, against the level the rule needed: the JEPA fails S1 (Okubo–Weiss), S2 and S3, while static VICReg and the non-physical JEPA match or beat it. | **Minor.** Usable as is. Shows seed dots only, with no bootstrap CIs. The paired CIs against the control are stored in `final_evaluation.json` (`rules.raw.<model>.S1.targets.*.ci95`), so they can be added without recomputing. |
| 2 | Pixel predictor's enstrophy direction | `results/phase4_traversals_pixel.png` @ `c2c3a79` | PH (unscored view) | Traversal along the pixel predictor's slow directions: direction 2 tracks initial enstrophy (trajectory \|r\| 0.80; energy 0.22), with spectra filling in at high k. | **Regenerate.** <br>• Three rows where only direction 2 carries the point. <br>• Seed 0 on val frames, and the panels are the *nearest real frames*, not decoded fields; the caption must say so. <br>• Small spectrum legends. |
| 3 | Kill gate of the exploratory line | `results/explore/stage1.png` @ `4b9be90` | E | On fresh exploration data, the pixel predictor's slow directions track enstrophy far above the untrained control, including at matched energy (0.98 vs 0.18). | **Minor.** <br>• Remove the `[EXPLORE]` tag from the title. <br>• Label the right panel's partial-r controls. |
| 4 | Horizon sweep | `results/explore/stage2/primary.png` @ `244f625` | E | Enstrophy prominence (E1) vs prediction horizon: the imaginer stays above the summarizer at every k, and prominence does not grow with horizon. | **Regenerate.** <br>• The k = 16 imaginer point is a single seed with no error bar; it must be marked. <br>• `[EXPLORE]` title. <br>• Three panels where E1 alone carries the point. |
| 5 | Rendering vs regularization | `results/explore/stage2/autoencoders.png` @ `244f625` | E | Plain autoencoders that only render the present match the imaginer's enstrophy prominence. Adding VICReg drops it to control level while raising linear decodability (log P R²). | **Minor.** <br>• Remove the `[EXPLORE]` title tag. <br>• The middle panel (E4, log axis) is secondary and could move to the supplement. |
| — | **v2 confirmation** | none exists | C | (E1 per group on the sealed confirm set, with the H1 and H2 margins) | **Missing.** <br>• `results/explore/stage3/` has no figure. <br>• The paper needs one, made from `results/explore/stage3/confirm.json` @ `a88ddee` alone (no data access). |
| 6 | Readable, not steerable | `results/build/stage0.png` @ `ae98ffa` | E | Of 84 candidate latent directions in 21 existing models, none moves the decoded field by the dial threshold; even the supervised shape direction (stars), which predicts s almost perfectly, barely moves it. | **Regenerate.** <br>• Effects below 10⁻⁴ are clipped onto the left axis without marking. <br>• The dashed vertical line is the brief's original floor (0.1), not the committed 0.2 (BP2). <br>• Seven model colours are hard to tell apart. |
| 7 | Grid artefacts from sub-pixel upsampling | `results/build/artefact_calibration.png` @ `2382399` | E | Worst-frame steered fields from pixel-shuffle decoders show block and checkerboard patterns; the out-of-band power criterion was calibrated on these. | **Regenerate (or supplement).** <br>• Internal labels (M2-SW, ES3), tiny titles, 25 panels. <br>• For the main text, 2–3 panels next to a resize-convolution counterpart would show the fix. |
| 8 | The shape-noise dose | `results/build/build_summary.png` @ `70ec339` | E | As the content's shape signal is removed, the loudness-null direction becomes a shape knob. Both dials appear only with the resize-convolution decoder, and rendering error rises with dose. | **Regenerate.** In the middle panel: <br>• the y-axis has fractional ticks for a seed count; <br>• the four pixel-shuffle lines overlap at 0; <br>• the legend sits over the data. <br>The figure also mixes decoders across doses; the caption must state this. |
| 9 | v3 confirmation steering | `results/build/v3/confirm_dials.png` @ `258d6af` | C | Both dials of the five fresh main models on one sealed `confirm3` frame. The loudness dial scales the field; the shape dial moves energy between scales. Seed 102's shape direction fails, with artefacts at α = +2. | **Regenerate, but see the note below.** <br>• Row titles are clipped at the left edge and use run IDs (`b6_main_s100`). <br>• The spectra have no axis labels or legend. <br>• No pass/fail marks, and no controls. |

## Supplementary

| # | figure | path @ commit | status | caption | flag |
|---|---|---|---|---|---|
| S1 | Final exploration model | `results/build/stepD_dials.png` @ `6a3420d` | E | Both dials of the τ = 0.2 resize-convolution model, every seed, explore and matched-energy frames. | **Regenerate if used.** <br>• 12 rows with tiny text. <br>• `[BUILD]` title. |
| S2 | Transformation-matched rule | `results/build/autopsy/autopsy_tm.png` @ `3c55d66` | PH | The post hoc transformation-matched shape direction on one explore frame, per main seed: a clean but weak reshaping. | **Minor.** Post hoc only; include only if the autopsy is reported. |

## Note on regenerating figure 9

- The figure shows a `confirm3` frame. Re-plotting it means loading `confirm3` again, which is a spent
  sealed set.
- That would be plotting already-scored data, not a new test. Even so, it is an access to the set after
  its single use, and the investigator should decide it explicitly. It would also be logged.
- The alternatives:
  - re-style the existing PNG (crop, relabel) without re-rendering;
  - draw the same panels on an exploration frame, labelled as such.
- Figures 1 and 2 are re-plotted from stored outputs or val data and touch no sealed or retired set.
  Figure 1 uses `results/final/final_evaluation.json`.
