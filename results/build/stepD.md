# [BUILD] Step B: shape-noise dose on M2-SF-style content (exploratory)

Definitions and predictions: results/build/PREDICTIONS.md entry 8. Criteria: entry 2 plus the grid-artefact criterion. Calibrated dose = ridge R² of s from the content input's log shell powers (results/build/stepB_calibration.md). Shape rule: PC2 (primary, frozen); the projected-out rule and loudness-null PCA are secondary, chosen after seeing results.

## Per dose (counts over 3 seeds; all criteria)

| τ | content shape R² | loudness dial (symmetry) | shape dial: PC2 / projected-out / loudness-null | both dials: PC2 / projected-out / loudness-null | median σ_shape explore: PC2 / projected-out / loudness-null | ω val MSE (median) | rendering triggers |
|---|---|---|---|---|---|---|---|
| 0.2 | 0.611 | 3 | 2 / 0 / 3 | 2 / 0 / 3 | 0.90 / 0.46 / 0.88 | 0.0223 | 3 of 3 |

## Every model

| τ | seed | ω MSE / k≥10 kept / triggers | loudness: D L S, σ_shape, effect (explore), artefact frames | PC2: D L S, σ_shape explore / matched | projected-out: D L S, σ_shape | loudness-null: D L S, σ_shape | cos² null vs PC2 | loudness gradient: median cos(frame, mean) |
|---|---|---|---|---|---|---|---|---|
| 0.2 | 0 | 0.0223 / 0.56 / > 2× VICReg AE, > 0.02 | DL·, 0.00, 0.66, 0.0% / 0.0% | D·S, 0.96 / 0.98 | ···, 0.51 / 0.55 | D·S, 0.99 / 0.95 | 0.88 | 0.97 |
| 0.2 | 1 | 0.0216 / 0.56 / > 2× VICReg AE, > 0.02 | DL·, 0.00, 0.53, 0.0% / 0.0% | D·S, 0.90 / 0.79 | D··, 0.40 / 0.45 | D·S, 0.87 / 0.82 | 0.99 | 0.97 |
| 0.2 | 2 | 0.0230 / 0.55 / > 2× VICReg AE, > 0.02 | DL·, 0.00, 0.62, 0.0% / 0.0% | D··, 0.69 / 0.87 | D··, 0.46 / 0.53 | D·S, 0.88 / 0.86 | 0.95 | 0.96 |

## Spatial-mean offset diagnostic (entry 10; not a criterion): fraction of frames whose decoded ω has more than 10% of ω² in its spatial mean, per α (−2, −1.5, −1, −0.5, 0, 0.5, 1, 1.5, 2)

| τ | seed | direction | explore | matched |
|---|---|---|---|---|
| 0.2 | 0 | loudness | 0% 0% 0% 0% 0% 0% 0% 0% 0% | 0% 0% 0% 0% 0% 0% 0% 0% 0% |
| 0.2 | 0 | PC2 | 0% 0% 0% 0% 0% 0% 0% 0% 0% | 0% 0% 0% 0% 0% 0% 0% 0% 0% |
| 0.2 | 0 | projected-out | 10% 1% 0% 0% 0% 0% 0% 0% 0% | 0% 0% 0% 0% 0% 0% 0% 0% 0% |
| 0.2 | 0 | loudness-null | 4% 0% 0% 0% 0% 0% 0% 0% 0% | 0% 0% 0% 0% 0% 0% 0% 0% 0% |
| 0.2 | 1 | loudness | 0% 0% 0% 0% 0% 0% 0% 0% 0% | 0% 0% 0% 0% 0% 0% 0% 0% 0% |
| 0.2 | 1 | PC2 | 0% 0% 0% 0% 0% 0% 0% 1% 2% | 0% 0% 0% 0% 0% 0% 0% 0% 0% |
| 0.2 | 1 | projected-out | 1% 0% 0% 0% 0% 0% 0% 0% 0% | 1% 0% 0% 0% 0% 0% 0% 0% 0% |
| 0.2 | 1 | loudness-null | 2% 0% 0% 0% 0% 0% 0% 0% 0% | 0% 0% 0% 0% 0% 0% 0% 0% 0% |
| 0.2 | 2 | loudness | 0% 0% 0% 0% 0% 0% 0% 2% 5% | 0% 0% 0% 0% 0% 0% 0% 0% 0% |
| 0.2 | 2 | PC2 | 0% 0% 0% 0% 0% 0% 0% 0% 1% | 0% 0% 0% 0% 0% 0% 0% 0% 0% |
| 0.2 | 2 | projected-out | 1% 0% 0% 0% 0% 0% 0% 0% 0% | 0% 0% 0% 0% 0% 0% 0% 0% 0% |
| 0.2 | 2 | loudness-null | 1% 0% 0% 0% 0% 0% 0% 0% 0% | 0% 0% 0% 0% 0% 0% 0% 0% 0% |

## M2-SW reference (full whitening; analysis only; Step A's loudness direction)

| seed | PC2 | projected-out | loudness-null | both dials: PC2 / projected-out / loudness-null | cos² null vs PC2 | loudness gradient cos |
|---|---|---|---|---|---|---|

## Shape-rule directions, every metric — explore

| τ | seed | rule | σ_d | Δa | Δs | effect | σ_shape | monot. | cycle corr / gain | env − ideal | shared | artefact frames | fails |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.2 | 0 | PC2 | 3.29 | -0.015 | +0.350 | 0.365 | 0.96 | 1.00 | 0.95 / 0.65 | +0.00 | 0.01 | 0.0% | — |
| 0.2 | 0 | projected-out | 5.07 | -0.526 | +0.539 | 1.064 | 0.51 | 1.00 | 0.82 / 0.42 | +0.00 | 0.02 | 0.0% | cycle corr, cycle gain |
| 0.2 | 0 | loudness-null | 3.1 | -0.003 | +0.428 | 0.431 | 0.99 | 1.00 | 0.96 / 0.75 | +0.00 | 0.01 | 0.0% | — |
| 0.2 | 0 | loudness | 5.1 | +0.660 | +0.001 | 0.661 | 0.00 | 1.00 | 0.99 / 0.88 | -0.00 | 0.01 | 0.0% | — |
| 0.2 | 1 | PC2 | 5.7 | +0.039 | -0.336 | 0.376 | 0.90 | 1.00 | 0.97 / 0.76 | +0.00 | 0.01 | 0.0% | — |
| 0.2 | 1 | projected-out | 6.37 | -0.771 | +0.523 | 1.294 | 0.40 | 1.00 | 0.96 / 0.69 | -0.00 | 0.01 | 0.0% | — |
| 0.2 | 1 | loudness-null | 5.69 | -0.049 | +0.329 | 0.378 | 0.87 | 1.00 | 0.97 / 0.74 | +0.00 | 0.01 | 0.0% | — |
| 0.2 | 1 | loudness | 5.9 | +0.532 | -0.001 | 0.534 | 0.00 | 1.00 | 1.00 / 0.76 | -0.00 | 0.01 | 0.0% | — |
| 0.2 | 2 | PC2 | 5.15 | +0.184 | -0.407 | 0.591 | 0.69 | 1.00 | 0.98 / 0.74 | +0.00 | 0.01 | 0.0% | — |
| 0.2 | 2 | projected-out | 5.48 | -0.598 | +0.512 | 1.111 | 0.46 | 1.00 | 0.95 / 0.65 | -0.00 | 0.01 | 0.0% | — |
| 0.2 | 2 | loudness-null | 5.17 | -0.046 | +0.347 | 0.394 | 0.88 | 1.00 | 0.98 / 0.79 | +0.00 | 0.01 | 0.0% | — |
| 0.2 | 2 | loudness | 5.75 | -0.619 | +0.003 | 0.622 | 0.00 | 1.00 | 1.00 / 0.91 | -0.01 | 0.01 | 0.0% | — |

## Shape-rule directions, every metric — matched

| τ | seed | rule | σ_d | Δa | Δs | effect | σ_shape | monot. | cycle corr / gain | env − ideal | shared | artefact frames | fails |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.2 | 0 | PC2 | 3.29 | -0.008 | +0.373 | 0.381 | 0.98 | 1.00 | 0.97 / 0.79 | +0.00 | 0.01 | 0.0% | — |
| 0.2 | 0 | projected-out | 5.07 | -0.463 | +0.558 | 1.021 | 0.55 | 1.00 | 0.93 / 0.44 | +0.00 | 0.04 | 0.0% | cycle gain |
| 0.2 | 0 | loudness-null | 3.1 | -0.022 | +0.455 | 0.477 | 0.95 | 1.00 | 0.97 / 0.87 | +0.00 | 0.01 | 0.0% | — |
| 0.2 | 0 | loudness | 5.1 | +0.687 | -0.022 | 0.709 | 0.03 | 1.00 | 0.99 / 0.91 | -0.02 | 0.01 | 0.0% | — |
| 0.2 | 1 | PC2 | 5.7 | -0.078 | -0.296 | 0.374 | 0.79 | 1.00 | 0.99 / 0.83 | -0.00 | 0.01 | 0.0% | — |
| 0.2 | 1 | projected-out | 6.37 | -0.549 | +0.452 | 1.001 | 0.45 | 1.00 | 0.97 / 0.65 | -0.00 | 0.01 | 0.0% | — |
| 0.2 | 1 | loudness-null | 5.69 | +0.066 | +0.293 | 0.359 | 0.82 | 1.00 | 0.99 / 0.82 | -0.00 | 0.01 | 0.0% | — |
| 0.2 | 1 | loudness | 5.9 | +0.504 | -0.013 | 0.516 | 0.02 | 1.00 | 1.00 / 0.75 | -0.02 | 0.01 | 0.0% | — |
| 0.2 | 2 | PC2 | 5.15 | +0.057 | -0.384 | 0.441 | 0.87 | 1.00 | 0.99 / 0.87 | +0.00 | 0.01 | 0.0% | — |
| 0.2 | 2 | projected-out | 5.48 | -0.420 | +0.475 | 0.895 | 0.53 | 1.00 | 0.97 / 0.65 | +0.00 | 0.01 | 0.0% | — |
| 0.2 | 2 | loudness-null | 5.17 | +0.056 | +0.334 | 0.391 | 0.86 | 1.00 | 0.99 / 0.93 | +0.00 | 0.01 | 0.0% | — |
| 0.2 | 2 | loudness | 5.75 | -0.660 | +0.032 | 0.692 | 0.05 | 1.00 | 1.00 / 0.92 | -0.01 | 0.01 | 0.0% | — |
