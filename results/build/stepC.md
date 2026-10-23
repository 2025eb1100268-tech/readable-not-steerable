# [BUILD] Step B: shape-noise dose on M2-SF-style content (exploratory)

Definitions and predictions: results/build/PREDICTIONS.md entry 8. Criteria: entry 2 plus the grid-artefact criterion. Calibrated dose = ridge R² of s from the content input's log shell powers (results/build/stepB_calibration.md). Shape rule: PC2 (primary, frozen); the projected-out rule and loudness-null PCA are secondary, chosen after seeing results.

## Per dose (counts over 3 seeds; all criteria)

| τ | content shape R² | loudness dial (symmetry) | shape dial: PC2 / projected-out / loudness-null | both dials: PC2 / projected-out / loudness-null | median σ_shape explore: PC2 / projected-out / loudness-null | ω val MSE (median) | rendering triggers |
|---|---|---|---|---|---|---|---|
| 0.5 | 0.294 | 3 | 1 / 0 / 3 | 1 / 0 / 3 | 0.58 / 0.45 / 0.96 | 0.0392 | 3 of 3 |

## Every model

| τ | seed | ω MSE / k≥10 kept / triggers | loudness: D L S, σ_shape, effect (explore), artefact frames | PC2: D L S, σ_shape explore / matched | projected-out: D L S, σ_shape | loudness-null: D L S, σ_shape | cos² null vs PC2 | loudness gradient: median cos(frame, mean) |
|---|---|---|---|---|---|---|---|---|
| 0.5 | 0 | 0.0394 / 0.47 / > 2× VICReg AE, > 0.02, < 50% small-scale power | DL·, 0.00, 0.59, 0.0% / 0.0% | D·S, 0.83 / 0.74 | D··, 0.45 / 0.48 | D·S, 1.00 / 0.88 | 0.91 | 0.93 |
| 0.5 | 1 | 0.0392 / 0.51 / > 2× VICReg AE, > 0.02 | DL·, 0.01, 0.62, 0.0% / 0.0% | D··, 0.58 / 0.61 | D··, 0.46 / 0.49 | D·S, 0.96 / 0.96 | 0.77 | 0.94 |
| 0.5 | 2 | 0.0390 / 0.46 / > 2× VICReg AE, > 0.02, < 50% small-scale power | DL·, 0.01, 0.51, 0.0% / 0.0% | DL·, 0.14 / 0.11 | ···, 0.43 / 0.47 | D·S, 0.86 / 0.85 | 0.52 | 0.94 |

## M2-SW reference (full whitening; analysis only; Step A's loudness direction)

| seed | PC2 | projected-out | loudness-null | both dials: PC2 / projected-out / loudness-null | cos² null vs PC2 | loudness gradient cos |
|---|---|---|---|---|---|---|

## Shape-rule directions, every metric — explore

| τ | seed | rule | σ_d | Δa | Δs | effect | σ_shape | monot. | cycle corr / gain | env − ideal | shared | artefact frames | fails |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.5 | 0 | PC2 | 4.75 | +0.058 | +0.278 | 0.336 | 0.83 | 1.00 | 0.96 / 0.71 | -0.01 | 0.02 | 0.0% | — |
| 0.5 | 0 | projected-out | 5.25 | -0.641 | +0.530 | 1.171 | 0.45 | 1.00 | 0.91 / 0.56 | +0.00 | 0.03 | 0.4% | — |
| 0.5 | 0 | loudness-null | 4.63 | +0.000 | +0.320 | 0.321 | 1.00 | 1.00 | 0.96 / 0.76 | -0.01 | 0.02 | 0.0% | — |
| 0.5 | 0 | loudness | 4.96 | +0.591 | -0.002 | 0.593 | 0.00 | 1.00 | 1.00 / 0.85 | -0.01 | 0.01 | 0.0% | — |
| 0.5 | 1 | PC2 | 4.75 | +0.327 | -0.453 | 0.780 | 0.58 | 1.00 | 0.90 / 0.56 | +0.00 | 0.03 | 0.0% | — |
| 0.5 | 1 | projected-out | 4.87 | +0.581 | -0.501 | 1.082 | 0.46 | 1.00 | 0.91 / 0.57 | +0.00 | 0.04 | 0.0% | — |
| 0.5 | 1 | loudness-null | 4.9 | -0.011 | +0.301 | 0.313 | 0.96 | 1.00 | 0.94 / 0.64 | +0.00 | 0.04 | 0.0% | — |
| 0.5 | 1 | loudness | 5.66 | +0.614 | -0.004 | 0.618 | 0.01 | 1.00 | 1.00 / 0.89 | -0.01 | 0.01 | 0.0% | — |
| 0.5 | 2 | PC2 | 4.91 | -0.466 | -0.076 | 0.543 | 0.14 | 1.00 | 0.99 / 0.87 | -0.01 | 0.01 | 0.0% | — |
| 0.5 | 2 | projected-out | 5.95 | -0.668 | +0.502 | 1.170 | 0.43 | 1.00 | 0.86 / 0.48 | -0.00 | 0.04 | 0.0% | cycle corr, cycle gain |
| 0.5 | 2 | loudness-null | 5.39 | -0.056 | +0.352 | 0.408 | 0.86 | 1.00 | 0.96 / 0.63 | +0.00 | 0.02 | 0.0% | — |
| 0.5 | 2 | loudness | 4.63 | +0.505 | -0.003 | 0.508 | 0.01 | 1.00 | 1.00 / 0.79 | -0.00 | 0.01 | 0.0% | — |

## Shape-rule directions, every metric — matched

| τ | seed | rule | σ_d | Δa | Δs | effect | σ_shape | monot. | cycle corr / gain | env − ideal | shared | artefact frames | fails |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.5 | 0 | PC2 | 4.75 | +0.103 | +0.288 | 0.391 | 0.74 | 1.00 | 0.98 / 0.87 | +0.00 | 0.02 | 0.0% | — |
| 0.5 | 0 | projected-out | 5.25 | -0.590 | +0.553 | 1.143 | 0.48 | 1.00 | 0.94 / 0.61 | +0.00 | 0.23 | 0.0% | — |
| 0.5 | 0 | loudness-null | 4.63 | +0.047 | +0.327 | 0.373 | 0.88 | 1.00 | 0.99 / 0.94 | +0.00 | 0.01 | 0.0% | — |
| 0.5 | 0 | loudness | 4.96 | +0.586 | -0.027 | 0.613 | 0.04 | 1.00 | 1.00 / 0.85 | -0.00 | 0.01 | 0.0% | — |
| 0.5 | 1 | PC2 | 4.75 | +0.285 | -0.451 | 0.736 | 0.61 | 1.00 | 0.97 / 0.74 | +0.00 | 0.01 | 0.0% | — |
| 0.5 | 1 | projected-out | 4.87 | +0.522 | -0.501 | 1.023 | 0.49 | 1.00 | 0.96 / 0.66 | +0.00 | 0.02 | 0.0% | — |
| 0.5 | 1 | loudness-null | 4.9 | -0.012 | +0.319 | 0.332 | 0.96 | 1.00 | 0.98 / 0.87 | +0.00 | 0.03 | 0.0% | — |
| 0.5 | 1 | loudness | 5.66 | +0.611 | -0.015 | 0.626 | 0.02 | 1.00 | 1.00 / 0.93 | -0.01 | 0.01 | 0.0% | — |
| 0.5 | 2 | PC2 | 4.91 | -0.499 | -0.063 | 0.563 | 0.11 | 1.00 | 1.00 / 0.94 | -0.00 | 0.01 | 0.0% | — |
| 0.5 | 2 | projected-out | 5.95 | -0.542 | +0.482 | 1.023 | 0.47 | 1.00 | 0.89 / 0.41 | +0.00 | 0.27 | 0.0% | cycle corr, cycle gain |
| 0.5 | 2 | loudness-null | 5.39 | +0.055 | +0.321 | 0.375 | 0.85 | 1.00 | 0.98 / 0.73 | -0.00 | 0.08 | 0.0% | — |
| 0.5 | 2 | loudness | 4.63 | +0.515 | -0.024 | 0.539 | 0.04 | 1.00 | 1.00 / 0.81 | -0.01 | 0.01 | 0.0% | — |
