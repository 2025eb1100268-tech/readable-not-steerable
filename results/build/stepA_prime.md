# [BUILD] Step A′: shape dial after projecting out the symmetry-loudness direction (M2-SW; exploratory)

Procedure and predictions: results/build/PREDICTIONS.md entry 7. Criteria: entry 2 plus the grid-artefact criterion. D L S = dial / loudness / shape on both datasets (all criteria); in brackets, entry 2's criteria alone.

| seed | new rule: D L S | σ_shape explore / matched | effect explore / matched | artefact frames | cos² with PC2 / with loudness | z-variance share | PC2 rule: D L S | loudness (Step A) | both dials, new rule | both dials, PC2 rule |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | D·· (D··) | 0.54 / 0.76 | 0.909 / 0.551 | 0.0% / 0.4% | 0.75 / 0.00 | 0.30 | D·S | DL· | no (no) | **yes** (yes) |
| 1 | ··· (D··) | 0.54 / 0.64 | 0.933 / 0.672 | 11.3% / 18.0% | 0.80 / 0.00 | 0.31 | ··· | DL· | no (no) | no (no) |
| 2 | ··· (···) | 0.50 / 0.58 | 1.064 / 0.820 | 2.7% / 0.8% | 0.49 / 0.00 | 0.35 | ··· | ··· | no (no) | no (no) |

Both dials valid in the same seed (all criteria): new rule 0 of 3, PC2 rule 1 of 3. Shape dial (all criteria): new rule 0 of 3, PC2 rule 1 of 3.

### New-rule direction, every metric — explore

| seed | σ_d | Δa | Δs | Δlog P | effect | σ_shape | monot. | cycle corr / gain | env − ideal | shared | artefact frames | fails |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 4.98 | -0.416 | +0.492 | +0.545 | 0.909 | 0.54 | 1.00 | 0.95 / 0.66 | -0.00 | 0.01 | 0.0% | — |
| 1 | 5.23 | -0.428 | +0.505 | +0.475 | 0.933 | 0.54 | 1.00 | 0.93 / 0.67 | -0.00 | 0.01 | 11.3% | grid artefacts |
| 2 | 5.6 | -0.537 | +0.528 | +0.471 | 1.064 | 0.50 | 0.85 | 0.91 / 0.62 | +0.00 | 0.01 | 2.7% | monotonicity, grid artefacts |

### New-rule direction, every metric — matched

| seed | σ_d | Δa | Δs | Δlog P | effect | σ_shape | monot. | cycle corr / gain | env − ideal | shared | artefact frames | fails |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 4.98 | -0.134 | +0.416 | +0.551 | 0.551 | 0.76 | 1.00 | 0.96 / 0.61 | -0.00 | 0.01 | 0.4% | — |
| 1 | 5.23 | -0.239 | +0.433 | +0.418 | 0.672 | 0.64 | 1.00 | 0.96 / 0.67 | -0.00 | 0.08 | 18.0% | grid artefacts |
| 2 | 5.6 | -0.342 | +0.478 | +0.456 | 0.820 | 0.58 | 1.00 | 0.96 / 0.63 | -0.00 | 0.02 | 0.8% | — |

