# [BUILD] Autopsy of the v3 Stage 5 run (exploratory, post hoc; PREDICTIONS entry 11)

Nothing here changes REPORT_v3's verdict. No training; `confirm3` and `matched-energy-confirm` were not opened. Directions and sealed-set scores come from `results/build/v3/confirm.json.gz`; every new measurement uses exploration data (v1 train, the build's 256 explore optimization frames, and 256 frames each of `explore` and the exploration matched-energy set), mean removed before every scoring measurement, criteria as in v3.

## A1: why loudness-null picked a weak direction (z-scored dial space of the rule)

| seed | cos²(g, PC2) | cos²(g, h) | cos²(h, PC2) | PC2 shape effect removed by projecting out g: linear / steered | variance of projected z-data: top PC / 2nd / along projected PC2 | loudness-null σ_d / PC2 σ_d | check: g recomputed vs stored (cos) |
|---|---|---|---|---|---|---|---|
| 100 | 0.009 | 0.279 | 0.473 | +0.07 / +0.08 | 2.91 / 0.85 / 2.90 | 5.12 / 5.12 | 1.0000 |
| 101 | 0.057 | 0.285 | 0.588 | +0.17 / +0.18 | 2.44 / 1.44 / 2.43 | 5.22 / 5.16 | 1.0000 |
| 102 | 0.000 | 0.230 | 0.498 | -0.00 / -0.00 | 2.39 / 1.61 / 1.48 | 2.11 / 4.47 | 1.0000 |
| 103 | 0.117 | 0.495 | 0.133 | -0.66 / -0.70 | 2.32 / 1.90 / 2.17 | 5.34 / 5.05 | 1.0000 |
| 104 | 0.001 | 0.190 | 0.432 | +0.02 / +0.01 | 3.08 / 1.06 / 2.94 | 3.70 / 3.84 | 1.0000 |

g: frame-mean gradient of a in z-units; h: the same for s; PC2 mapped to z-space. Steered: median Δs at α = ±1 σ_PC2 along PC2 and along its projection onto g⊥ (same z-displacement minus its g component), optimization frames.

## A2: is the union lenient? (stored sealed-set scores)

| control | PC2 shape dial | loudness-null shape dial | union (either) |
|---|---|---|---|
| tau0 | 0 of 3 | 0 of 3 | 0 of 3 |
| untrained | 0 of 3 | 0 of 3 | 0 of 3 |

Main model, both dials with the union as the shape rule (sealed sets; post hoc): 5 of 5.

## A3: transformation-matched shape rule (exploration sets)

| group | run | J_T: TM / loudness-null / PC2 / loudness / PC1 | TM found from | |cos| TM vs loudness-null / PC2 / loudness | TM shape dial | TM σ_d | stored loudness dial (expl.) | both dials (TM) | loudness-null shape dial (expl.) | PC2 shape dial (expl.) |
|---|---|---|---|---|---|---|---|---|---|---|
| main | b6_main_s100 | 0.111 / 0.217 / 0.266 / 0.621 / 0.874 | PC6 | 0.47 / 0.47 / 0.15 | **no** | 2.69 | yes | **no** | yes | yes |
| main | b6_main_s101 | 0.108 / 0.191 / 0.318 / 0.632 / 0.874 | random2 | 0.56 / 0.51 / 0.12 | **no** | 3.06 | yes | **no** | yes | no |
| main | b6_main_s102 | 0.108 / 0.690 / 0.162 / 0.624 / 0.878 | PC6 | 0.29 / 0.69 / 0.20 | **no** | 3.25 | yes | **no** | no | yes |
| main | b6_main_s103 | 0.103 / 0.345 / 0.237 / 0.629 / 0.869 | PC6 | 0.54 / 0.63 / 0.44 | **no** | 3.41 | yes | **no** | yes | no |
| main | b6_main_s104 | 0.102 / 0.268 / 0.221 / 0.625 / 0.861 | random0 | 0.36 / 0.44 / 0.21 | **no** | 2.32 | yes | **no** | no | yes |
| tau0 | b6_tau0_s200 | 0.410 / 0.667 / 0.760 / 0.658 / 0.700 | PC8 | 0.30 / 0.30 / 0.30 | **no** | 1.8 | — | — | — | — |
| tau0 | b6_tau0_s201 | 0.344 / 0.891 / 0.789 / 0.668 / 0.709 | PC8 | 0.19 / 0.27 / 0.32 | **no** | 1.36 | — | — | — | — |
| tau0 | b6_tau0_s202 | 0.357 / 0.691 / 0.715 / 0.664 / 0.696 | PC1 | 0.04 / 0.14 / 0.28 | **no** | 1.36 | — | — | — | — |
| untrained | b6_untrained_s300 | 0.819 / 0.857 / 0.978 / 0.902 / 0.866 | PC1 | 0.88 / 0.12 / 0.89 | **no** | 2.84 | — | — | — | — |
| untrained | b6_untrained_s301 | 0.866 / 0.943 / 0.991 / 0.974 / 0.961 | random2 | 0.69 / 0.18 / 0.44 | **no** | 1.46 | — | — | — | — |
| untrained | b6_untrained_s302 | 0.864 / 0.941 / 0.989 / 0.893 / 0.929 | PC2 | 0.19 / 0.28 / 0.85 | **no** | 0.903 | — | — | — | — |

TM shape dial: main 0 of 5, τ = 0 0 of 3, untrained 0 of 3. Both dials with TM (stored loudness direction, exploration sets): 0 of 5.

Family fits at the edge of the first grid level (|t| = 8): 30781 of 5586944.

### Every scored direction, every metric — explore (mean removed)

| group | run | direction | σ_d | Δa | Δs | effect | σ_shape | monot. | cycle corr / gain | env − ideal | shared | artefact frames | pre-removal mean > 10% (max over α) | D L S | fails |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| main | b6_main_s100 | shape (transformation-matched) | 2.69 | +0.015 | +0.113 | 0.128 | 0.89 | 1.00 | 0.95 / 0.35 | +0.00 | 0.00 | 0.0% | 0.0% | ··· | effect, cycle gain |
| main | b6_main_s100 | loudness | 5.5 | -0.575 | +0.001 | 0.576 | 0.00 | 1.00 | 1.00 / 0.86 | -0.00 | 0.01 | 0.0% | 2.3% | DL· | — |
| main | b6_main_s100 | shape (loudness-null) | 5.12 | -0.030 | +0.357 | 0.387 | 0.92 | 1.00 | 0.97 / 0.70 | +0.00 | 0.01 | 0.0% | 0.4% | D·S | — |
| main | b6_main_s100 | PC2 | 5.12 | -0.127 | +0.390 | 0.517 | 0.75 | 1.00 | 0.96 / 0.67 | +0.00 | 0.01 | 0.0% | 0.4% | D·S | — |
| main | b6_main_s101 | shape (transformation-matched) | 3.06 | +0.006 | +0.116 | 0.123 | 0.95 | 1.00 | 0.98 / 0.35 | -0.00 | 0.00 | 0.0% | 0.0% | ··· | effect, cycle gain |
| main | b6_main_s101 | loudness | 5.49 | +0.564 | -0.002 | 0.566 | 0.00 | 1.00 | 1.00 / 0.85 | -0.00 | 0.01 | 0.0% | 0.0% | DL· | — |
| main | b6_main_s101 | shape (loudness-null) | 5.22 | +0.017 | -0.388 | 0.405 | 0.96 | 1.00 | 0.99 / 0.90 | +0.00 | 0.01 | 0.0% | 1.2% | D·S | — |
| main | b6_main_s101 | PC2 | 5.16 | +0.274 | -0.450 | 0.724 | 0.62 | 1.00 | 0.98 / 0.79 | +0.00 | 0.01 | 0.0% | 1.2% | D·· | — |
| main | b6_main_s102 | shape (transformation-matched) | 3.25 | +0.005 | +0.165 | 0.170 | 0.97 | 1.00 | 0.98 / 0.49 | -0.00 | 0.00 | 0.0% | 0.0% | ··· | effect, cycle gain |
| main | b6_main_s102 | loudness | 5.5 | +0.568 | -0.001 | 0.570 | 0.00 | 1.00 | 0.99 / 0.79 | -0.00 | 0.01 | 0.0% | 0.0% | DL· | — |
| main | b6_main_s102 | shape (loudness-null) | 2.11 | +0.055 | +0.036 | 0.092 | 0.40 | 0.37 | -0.21 / -0.05 | -0.04 | 0.00 | 3.9% | 2.3% | ··· | effect, monotonicity, cycle corr, cycle gain, grid artefacts |
| main | b6_main_s102 | PC2 | 4.47 | -0.031 | +0.356 | 0.386 | 0.92 | 1.00 | 0.98 / 0.80 | +0.00 | 0.00 | 0.0% | 0.0% | D·S | — |
| main | b6_main_s103 | shape (transformation-matched) | 3.41 | -0.008 | -0.156 | 0.165 | 0.95 | 1.00 | 0.98 / 0.45 | -0.00 | 0.00 | 0.0% | 0.0% | ··· | effect, cycle gain |
| main | b6_main_s103 | loudness | 5.22 | +0.625 | -0.001 | 0.626 | 0.00 | 1.00 | 1.00 / 0.93 | -0.00 | 0.01 | 0.0% | 2.7% | DL· | — |
| main | b6_main_s103 | shape (loudness-null) | 5.34 | -0.005 | +0.343 | 0.348 | 0.98 | 1.00 | 0.96 / 0.61 | +0.00 | 0.01 | 0.0% | 16.4% | D·S | — |
| main | b6_main_s103 | PC2 | 5.05 | +0.304 | +0.185 | 0.489 | 0.38 | 1.00 | 0.99 / 0.85 | +0.00 | 0.00 | 0.0% | 0.0% | D·· | — |
| main | b6_main_s104 | shape (transformation-matched) | 2.32 | -0.014 | -0.136 | 0.150 | 0.91 | 1.00 | 0.97 / 0.41 | -0.00 | 0.00 | 0.0% | 0.0% | ··· | effect, cycle gain |
| main | b6_main_s104 | loudness | 5.51 | -0.647 | +0.000 | 0.648 | 0.00 | 1.00 | 1.00 / 0.89 | -0.00 | 0.01 | 0.0% | 0.0% | DL· | — |
| main | b6_main_s104 | shape (loudness-null) | 3.7 | -0.021 | +0.339 | 0.359 | 0.94 | 1.00 | 0.95 / 0.59 | -0.00 | 0.01 | 0.8% | 1.2% | D·S | — |
| main | b6_main_s104 | PC2 | 3.84 | +0.089 | -0.374 | 0.463 | 0.81 | 1.00 | 0.97 / 0.65 | +0.00 | 0.01 | 0.0% | 0.0% | D·S | — |
| tau0 | b6_tau0_s200 | shape (transformation-matched) | 1.8 | +0.009 | +0.006 | 0.014 | 0.39 | 1.00 | 0.70 / 0.04 | -0.00 | 0.01 | 0.0% | 0.0% | ··· | effect, cycle corr, cycle gain |
| tau0 | b6_tau0_s201 | shape (transformation-matched) | 1.36 | -0.003 | -0.005 | 0.008 | 0.68 | 1.00 | 0.66 / 0.05 | -0.00 | 0.01 | 0.0% | 0.0% | ··· | effect, cycle corr, cycle gain |
| tau0 | b6_tau0_s202 | shape (transformation-matched) | 1.36 | -0.006 | -0.005 | 0.011 | 0.48 | 1.00 | 0.74 / 0.03 | -0.00 | 0.01 | 0.0% | 0.0% | ··· | effect, cycle corr, cycle gain |
| untrained | b6_untrained_s300 | shape (transformation-matched) | 2.84 | -0.847 | -0.203 | 1.050 | 0.19 | 0.78 | 0.49 / 2.87 | -0.05 | 0.01 | 100.0% | 100.0% | ··· | monotonicity, cycle corr, envelope vs ideal, cycle gain, grid artefacts |
| untrained | b6_untrained_s301 | shape (transformation-matched) | 1.46 | -0.286 | +0.010 | 0.296 | 0.03 | 0.85 | 0.56 / 0.28 | -0.04 | 0.01 | 100.0% | 94.9% | ··· | monotonicity, cycle corr, cycle gain, grid artefacts |
| untrained | b6_untrained_s302 | shape (transformation-matched) | 0.903 | -0.647 | +0.071 | 0.718 | 0.10 | 1.00 | 0.62 / 0.60 | -0.02 | 0.01 | 100.0% | 23.4% | ··· | cycle corr, grid artefacts |

### Every scored direction, every metric — matched-energy (exploration) (mean removed)

| group | run | direction | σ_d | Δa | Δs | effect | σ_shape | monot. | cycle corr / gain | env − ideal | shared | artefact frames | pre-removal mean > 10% (max over α) | D L S | fails |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| main | b6_main_s100 | shape (transformation-matched) | 2.69 | +0.055 | +0.121 | 0.176 | 0.69 | 1.00 | 0.93 / 0.30 | -0.00 | 0.01 | 0.0% | 0.0% | ··· | effect, cycle gain |
| main | b6_main_s100 | loudness | 5.5 | -0.559 | +0.019 | 0.577 | 0.03 | 1.00 | 1.00 / 0.87 | -0.01 | 0.01 | 0.0% | 0.0% | DL· | — |
| main | b6_main_s100 | shape (loudness-null) | 5.12 | +0.021 | +0.343 | 0.364 | 0.94 | 1.00 | 0.99 / 0.86 | +0.00 | 0.01 | 0.0% | 0.0% | D·S | — |
| main | b6_main_s100 | PC2 | 5.12 | -0.057 | +0.370 | 0.426 | 0.87 | 1.00 | 0.98 / 0.81 | +0.00 | 0.01 | 0.0% | 0.0% | D·S | — |
| main | b6_main_s101 | shape (transformation-matched) | 3.06 | +0.070 | +0.114 | 0.184 | 0.62 | 1.00 | 0.98 / 0.31 | -0.00 | 0.01 | 0.0% | 0.0% | ··· | effect, cycle gain |
| main | b6_main_s101 | loudness | 5.49 | +0.578 | -0.024 | 0.602 | 0.04 | 1.00 | 1.00 / 0.89 | -0.01 | 0.01 | 0.0% | 0.0% | DL· | — |
| main | b6_main_s101 | shape (loudness-null) | 5.22 | -0.101 | -0.366 | 0.467 | 0.78 | 1.00 | 0.99 / 0.96 | +0.00 | 0.01 | 0.0% | 0.4% | D·S | — |
| main | b6_main_s101 | PC2 | 5.16 | +0.122 | -0.421 | 0.543 | 0.77 | 1.00 | 0.99 / 0.82 | +0.00 | 0.01 | 0.0% | 1.2% | D·S | — |
| main | b6_main_s102 | shape (transformation-matched) | 3.25 | +0.085 | +0.150 | 0.235 | 0.64 | 1.00 | 0.99 / 0.45 | +0.00 | 0.01 | 0.0% | 0.0% | ··· | cycle gain |
| main | b6_main_s102 | loudness | 5.5 | +0.594 | -0.031 | 0.625 | 0.05 | 1.00 | 1.00 / 0.78 | -0.01 | 0.01 | 0.0% | 0.0% | DL· | — |
| main | b6_main_s102 | shape (loudness-null) | 2.11 | +0.051 | +0.045 | 0.095 | 0.47 | 0.38 | -0.20 / -0.04 | -0.16 | 0.01 | 3.5% | 3.9% | ··· | effect, monotonicity, cycle corr, envelope vs ideal, cycle gain, grid artefacts |
| main | b6_main_s102 | PC2 | 4.47 | +0.104 | +0.323 | 0.427 | 0.76 | 1.00 | 0.99 / 0.89 | +0.00 | 0.01 | 0.0% | 0.0% | D·S | — |
| main | b6_main_s103 | shape (transformation-matched) | 3.41 | -0.066 | -0.152 | 0.218 | 0.70 | 1.00 | 0.99 / 0.40 | -0.00 | 0.01 | 0.0% | 0.0% | ··· | cycle gain |
| main | b6_main_s103 | loudness | 5.22 | +0.625 | -0.029 | 0.653 | 0.04 | 1.00 | 1.00 / 0.96 | -0.01 | 0.01 | 0.0% | 0.0% | DL· | — |
| main | b6_main_s103 | shape (loudness-null) | 5.34 | +0.069 | +0.310 | 0.378 | 0.82 | 1.00 | 0.99 / 0.68 | +0.00 | 0.01 | 0.0% | 0.0% | D·S | — |
| main | b6_main_s103 | PC2 | 5.05 | +0.371 | +0.164 | 0.535 | 0.31 | 1.00 | 1.00 / 0.92 | -0.00 | 0.01 | 0.0% | 0.0% | D·· | — |
| main | b6_main_s104 | shape (transformation-matched) | 2.32 | -0.047 | -0.136 | 0.184 | 0.74 | 1.00 | 0.98 / 0.31 | -0.00 | 0.01 | 0.0% | 0.0% | ··· | effect, cycle gain |
| main | b6_main_s104 | loudness | 5.51 | -0.634 | +0.021 | 0.655 | 0.03 | 1.00 | 1.00 / 0.89 | -0.01 | 0.01 | 0.0% | 0.0% | DL· | — |
| main | b6_main_s104 | shape (loudness-null) | 3.7 | +0.025 | +0.331 | 0.356 | 0.93 | 1.00 | 0.97 / 0.69 | -0.09 | 0.01 | 0.0% | 0.4% | ··· | envelope vs ideal |
| main | b6_main_s104 | PC2 | 3.84 | +0.026 | -0.364 | 0.390 | 0.93 | 1.00 | 0.99 / 0.76 | +0.00 | 0.01 | 0.0% | 0.0% | D·S | — |
| tau0 | b6_tau0_s200 | shape (transformation-matched) | 1.8 | +0.019 | +0.008 | 0.027 | 0.29 | 1.00 | 0.94 / 0.06 | -0.00 | 0.01 | 0.0% | 0.0% | ··· | effect, cycle gain |
| tau0 | b6_tau0_s201 | shape (transformation-matched) | 1.36 | -0.016 | -0.009 | 0.025 | 0.37 | 1.00 | 0.89 / 0.09 | -0.01 | 0.01 | 0.0% | 0.0% | ··· | effect, cycle corr, cycle gain |
| tau0 | b6_tau0_s202 | shape (transformation-matched) | 1.36 | -0.016 | -0.008 | 0.024 | 0.33 | 1.00 | 0.89 / 0.06 | -0.01 | 0.01 | 0.0% | 0.0% | ··· | effect, cycle corr, cycle gain |
| untrained | b6_untrained_s300 | shape (transformation-matched) | 2.84 | +0.001 | -0.320 | 0.321 | 1.00 | 0.82 | -0.08 / -0.08 | -0.05 | 0.01 | 100.0% | 100.0% | ··· | monotonicity, cycle corr, envelope vs ideal, cycle gain, grid artefacts |
| untrained | b6_untrained_s301 | shape (transformation-matched) | 1.46 | -0.045 | -0.021 | 0.066 | 0.32 | 0.42 | 0.33 / 0.04 | -0.03 | 0.01 | 100.0% | 100.0% | ··· | effect, monotonicity, cycle corr, cycle gain, grid artefacts |
| untrained | b6_untrained_s302 | shape (transformation-matched) | 0.903 | -0.671 | +0.107 | 0.778 | 0.14 | 1.00 | 0.73 / 0.15 | -0.02 | 0.01 | 100.0% | 38.7% | ··· | cycle corr, cycle gain, grid artefacts |

## A4: b6_main_s103 with the zero-mean step off (exploration sets; confirm3 not reopened)

| direction | set | mean removed: D L S (fails) | mean kept: D L S (fails) | frames with > 10% of ω² in the mean, max over α (raw) | Δs removed / kept | env − ideal removed / kept |
|---|---|---|---|---|---|---|
| loudness | explore | DL· (—) | DL· (—) | 2.7% | -0.001 / -0.003 | -0.00 / -0.00 |
| loudness | matched-energy (exploration) | DL· (—) | DL· (—) | 0.0% | -0.029 / -0.029 | -0.01 / -0.01 |
| shape (loudness-null) | explore | D·S (—) | D·S (—) | 16.4% | +0.343 / +0.340 | +0.00 / +0.00 |
| shape (loudness-null) | matched-energy (exploration) | D·S (—) | D·S (—) | 0.0% | +0.310 / +0.309 | +0.00 / +0.00 |

Checks: |cos| between recomputed and stored loudness-null / PC2 directions: 100 1.0000 / 1.0000, 101 1.0000 / 1.0000, 102 1.0000 / 1.0000, 103 1.0000 / 1.0000, 104 1.0000 / 1.0000, 200 1.0000 / 1.0000, 201 1.0000 / 1.0000, 202 1.0000 / 1.0000, 300 1.0000 / 1.0000, 301 1.0000 / 1.0000, 302 1.0000 / 1.0000.

Wall-clock 136 min.
