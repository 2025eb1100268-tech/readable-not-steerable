# Paper figures (Phase 2, 2026-11-03)

Vector PDFs for `DRAFT_v2.md`, one script per figure. Every script imports `_common.py`, which sets the
style, writes the PDFs with neutral metadata (title only) and guards the data:
- it refuses any path naming `confirm_v2`, `confirm3` or `matched-energy-confirm`;
- it refuses any dataset spec that would load the v1 test splits.

Figures 1–4 and 6 re-plot committed outputs. Figures 5 and 7 decode checkpoints on one exploration frame;
no training, no new measurement.

Run from the repo root, e.g. `python results/paper/figures/fig1_study1.py`.

**Style:**
- one font family (DejaVu Sans, embedded), 6–8.5 pt at TMLR text width (6.5 in);
- colour-blind-safe categorical slots: blue `#2a78d6`, orange `#eb6834`, aqua `#1baf7a`. They were
  validated all-pairs on white: CVD ΔE ≥ 9.2, normal-vision ΔE ≥ 24. Aqua is below 3:1 contrast, so it
  always carries a legend entry;
- diverging `RdBu_r` for fields;
- no run IDs and no stage tags in any figure.

## Draft figure → file

| draft | file | script | source @ commit | status |
|---|---|---|---|---|
| Figure 1 | `fig1_study1.pdf` | `fig1_study1.py` | `results/final/final_evaluation.json` @ `e8d619e` | C |
| Figure 2 | `fig2_autoencoders.pdf` (main); `fig2b_e4.pdf` (appendix: the E4 panel) | `fig2_autoencoders.py` | `results/explore/stage2/eval.json.gz` @ `244f625` | E |
| Figure 3 | `fig3_study2.pdf` | `fig3_study2.py` | `results/explore/stage3/confirm.json` @ `a88ddee` only | C |
| Figure 4 | `fig4_stage0.pdf` | `fig4_stage0.py` | `results/build/stage0.json.gz` @ `ae98ffa` | E |
| Figure 5 | `fig5_artefacts.pdf` | `fig5_artefacts.py` | checkpoints `b3_sn_t0p5_s{0,2}` (sub-pixel), `b4_rc_t0p5_s{0,2}` (resize); directions from `stepB.json.gz` @ `5234145`, `stepC.json.gz` @ `f0f6c1f`; explore steering frames | E |
| Figure 6 | `fig6_dose.pdf` | `fig6_dose.py` | `stepB.json.gz` @ `5234145`, `stepC.json.gz` @ `f0f6c1f`, `stepD.json.gz` @ `6a3420d` | E |
| Figure 7 | `fig7_v3_dials.pdf` (main); `fig7b_v3_dials_all.pdf` (appendix: all 11 models) | `fig7_v3_dials.py` | frozen v3 checkpoints, SHA-256-verified against `configs/build/v3_frozen.json` @ `fc3b703`; directions and verdicts from `results/build/v3/confirm.json.gz` @ `258d6af`; explore steering frame 0 | C (verdicts); the frame is exploration data |

## What changed from the flagged originals (FIGURES.md)

- **Figure 1:**
  - Each panel now shows the gain over the untrained control. Seeds are paired: the control seed
    shares each model's initialization.
  - The stored 95% paired bootstrap CI and the gain each rule needed are drawn. S2 has no stored CI;
    the legend says so.
  - Checks: JEPA log-enstrophy gain +0.204 [+0.175, +0.235]; S3 −0.158 [−0.31, +0.13]; matches Table 1.
- **Figure 2:**
  - The E4 panel moved to `fig2b_e4.pdf`.
  - The reference lines (untrained, imaginer k = 4, summarizer k = 4) are labelled directly.
- **Figure 3:** new. E1 per group, with seeds, mean and CI; the H1 threshold; brackets for the H2 and H4
  drops. Fresh seeds and exploration seeds are separate blocks.
- **Figure 4:**
  - Both effect floors are drawn: 0.1, the threshold used at Stage 0, and 0.2, committed afterwards.
    This resolves delta-check item N-8.
  - Effects below 10⁻⁴ sit in a marked strip (16 explore and 13 matched points).
  - Colour = family (autoencoder / + VICReg / imaginer); marker = render target; hollow = supervised
    shape direction.
- **Figure 5:**
  - Matched comparison: the same dose (τ = 0.5), seed and frame, sub-pixel against resize-convolution.
    Each decoder is shown at its own worst α, because a direction's sign is arbitrary per model.
  - The frame is the sub-pixel direction's worst explore frame by the stored per-frame shares. The
    recomputed shares match the stored ones: 0.0907 and 0.1307.
  - Shares are computed on the raw output. Fields are shown with the mean removed and a per-field colour
    scale, with a zoom on the strongest grid-scale texture.
- **Figure 6:**
  - The seed-count axis now has integer ticks.
  - The two rules are dodged horizontally; a note says the sub-pixel decoder has 0 of 3 under both rules
    at every dose.
  - The legend is below the panels; a top axis gives τ.
- **Figure 7:**
  - Decoded on an exploration frame.
  - Each spectrum panel has a k axis, and a legend for α is added.
  - ✓/✗ give each direction's verdict on both sealed sets.
  - The controls are included: one seed each in the main figure, all 11 models in the appendix figure.

## Suggested caption text (for the author; no number in the draft captions changes)

- **Figure 1:** "Each model's gain over the untrained control on every pre-registered S1–S3 quantity
  (test set, pooled latent): seeds, seed mean and stored 95% paired trajectory-bootstrap interval, against
  the gain each rule needed. The JEPA fails S1 (Okubo–Weiss), S2 and S3; static VICReg and the
  non-physical JEPA match or beat it (C)."
- **Figure 3:** "Enstrophy prominence E1 on the sealed Study 2 set (seeds, mean, 95% trajectory-bootstrap
  interval). Fresh seeds 3–5 decide H1 and H2; exploration seeds 0–2 are reported and carry H4 (C)."
- **Figure 4:** add "Dashed line: the 0.1 floor used at the time; dotted: the 0.2 floor committed
  afterwards. Effects below 10⁻⁴ are drawn in the shaded strip."
- **Figure 5:** "Steered fields at τ = 0.5 from a sub-pixel decoder and from the resize-convolution
  decoder: same seed and frame, each at its worst steering amplitude, with the out-of-band power share
  above each panel (E)."
- **Figure 6:** keep the draft caption; it already states that the curve mixes decoders.
- **Figure 7:** "Both dials of the five fresh main models and one seed of each control, decoded on one
  exploration frame (the sealed sets are spent and are not reopened for plotting); ✓/✗ give each
  direction's verdict on both sealed sets. All 11 models: Appendix."

## Notes for Phase 3

- **New numbers inside figures.** Figure 5's panel titles show out-of-band shares: 9.1×10⁻² and 1.3×10⁻¹
  for sub-pixel, 1.2×10⁻⁴ and 3.5×10⁻³ for resize. They are recomputed by the script and printed. If the
  prose cites them, they need a NUMBERS.md entry; the number lint covers prose only.
- **Not regenerated.** The pixel-predictor traversal (FIGURES.md #2) is not used in DRAFT_v2.
- **Previews.** PNG previews were rendered to the session scratchpad for visual checks and are not
  committed. Setting `FIG_PREVIEW_DIR` in the environment reproduces them.
