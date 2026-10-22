# [BUILD] Step A: loudness by symmetry on M2-SW, and the grid-artefact criterion (exploratory)

Definitions and predictions: results/build/PREDICTIONS.md entry 6. Criteria: entry 2 plus the grid-artefact criterion (a direction fails on a dataset if more than 1% of frames put more than 0.01 of the decoded vorticity's power outside the dealiasing band at some α).

## Step A summary

| seed | J (fraction of change not explained by rescaling): best / PC1 / PC2 | best found from | cos² with PC1 / PC2 / PC1–PC2 plane | loudness direction: D L S (entry 2 only) | its σ_shape explore / matched | effect explore / matched | artefact frames explore / matched | PC2: D L S (entry 2 only) | PC2 artefact frames | both dials valid? (entry 2 only) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.036 / 0.418 / 0.713 | PC3 | 0.65 / 0.14 / 0.79 | DL· (DL·) | 0.05 / 0.08 | 0.555 / 0.525 | 0.0% / 0.0% | D·S (D·S) | 0.8% / 0.0% | **yes** (yes) |
| 1 | 0.035 / 0.308 / 0.738 | PC5 | 0.69 / 0.08 / 0.77 | DL· (DL·) | 0.02 / 0.06 | 0.488 / 0.451 | 0.0% / 0.0% | ··· (D··) | 1.2% / 3.5% | no (no) |
| 2 | 0.037 / 0.587 / 0.635 | random2 | 0.36 / 0.29 / 0.65 | ··· (···) | 0.02 / 0.07 | 0.370 / 0.415 | 0.0% / 0.0% | ··· (D·S) | 2.0% / 0.0% | no (no) |

Both dials valid in the same seed (all criteria): 1 of 3 seeds → M2-SW does not meet the both-dials condition (≥ 2 of 3), subject to the rendering check (which fired for M2-SW in Stage 2b).

### Loudness direction, every metric — explore

| seed | σ_d | Δa | Δs | Δlog P | effect | σ_shape | monot. | cycle corr / gain | env − ideal | shared | artefact frames | fails |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 5.38 | -0.528 | +0.026 | -0.535 | 0.555 | 0.05 | 1.00 | 0.99 / 0.76 | -0.01 | 0.01 | 0.0% | — |
| 1 | 5.82 | +0.476 | -0.012 | +0.474 | 0.488 | 0.02 | 1.00 | 0.99 / 0.69 | -0.01 | 0.01 | 0.0% | — |
| 2 | 4.57 | +0.364 | -0.007 | +0.387 | 0.370 | 0.02 | 1.00 | 0.99 / 0.53 | -0.00 | 0.01 | 0.0% | — |

### Loudness direction, every metric — matched

| seed | σ_d | Δa | Δs | Δlog P | effect | σ_shape | monot. | cycle corr / gain | env − ideal | shared | artefact frames | fails |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 5.38 | -0.485 | +0.041 | -0.431 | 0.525 | 0.08 | 1.00 | 1.00 / 0.71 | -0.00 | 0.01 | 0.0% | — |
| 1 | 5.82 | +0.424 | -0.027 | +0.396 | 0.451 | 0.06 | 1.00 | 1.00 / 0.62 | -0.00 | 0.01 | 0.0% | — |
| 2 | 4.57 | +0.388 | -0.028 | +0.337 | 0.415 | 0.07 | 1.00 | 0.98 / 0.48 | -0.00 | 0.01 | 0.0% | cycle gain |

## Grid-artefact criterion applied retroactively (calibrated on M2-SW seed 1's PC2 at α = −2)

Only directions that were dials before, or are dials now, are listed; every direction is in the JSON.

| model | run | dir | before (Stage 2b) | after (+ artefact) | artefact frames explore / matched |
|---|---|---|---|---|---|
| M2-SW | b2b_m2sw_s0 | PC1 | D·· | ··· | 1.2% / 0.8% |
| M2-SW | b2b_m2sw_s0 | PC2 | D·S | D·S | 0.8% / 0.0% |
| M2-SW | b2b_m2sw_s0 | ES2 | D·· | ··· | 0.8% / 2.0% |
| M2-SW | b2b_m2sw_s0 | ES3 | D·· | D·· | 0.0% / 0.4% |
| M2-SW | b2b_m2sw_s1 | PC1 | D·· | ··· | 5.1% / 1.6% |
| M2-SW | b2b_m2sw_s1 | PC2 | D·· | ··· | 1.2% / 3.5% |
| M2-SW | b2b_m2sw_s1 | ES4 | D·· | D·· | 0.0% / 0.0% |
| M2-SW | b2b_m2sw_s2 | PC1 | D·· | ··· | 2.3% / 0.8% |
| M2-SW | b2b_m2sw_s2 | PC2 | D·S | ··· | 2.0% / 0.0% |
| M2-SW | b2b_m2sw_s2 | ES2 | D·· | ··· | 4.3% / 4.3% |
| M2-SW | b2b_m2sw_s2 | ES3 | DL· | DL· | 0.0% / 0.0% |
| M2-SF, omega-only loss | b2b_m2sfw_s0 | PC1 | DL· | ··· | 4.7% / 0.0% |
| M2-SF (Stage 2) | b2_m2sf_s0 | PC1 | DL· | ··· | 14.1% / 0.0% |
| M2-SF (Stage 2) | b2_m2sf_s1 | PC1 | DL· | ··· | 7.0% / 0.0% |
| M2-SF (Stage 2) | b2_m2sf_s2 | PC1 | DL· | ··· | 5.1% / 0.0% |

M2-SW seeds with a PCA shape dial after the artefact criterion: 1 of 3 (before: 2 of 3).
