# [BUILD] v3 confirmation (PREREGISTRATION_v3.md)

**Verdict:** PARTIALLY SUPPORTED: both dials are valid in 3 of 5 fresh main-model seeds (the pass rule needs 4): the design produces both dials in most, but not reliably in all, training runs. Both controls behave as pre-registered (tau = 0: shape dial in 0 of 3 seeds; untrained: 0 of 3).

Main model: both dials valid (all criteria, both sets) in **3 of 5** seeds → **partial**. τ = 0 control: shape dial in 0 of 3 seeds (within its rule). Untrained control: shape dial in 0 of 3 seeds (within its rule).

## Per model

| group | run | loudness dial | shape dial (loudness-null) | both | PC2: shape dial / both (secondary) | σ_shape loudness / shape, confirm3 | σ_shape loudness / shape, matched-energy-confirm | rendering ω (confirm3) | rendering ω (v1 val) |
|---|---|---|---|---|---|---|---|---|---|
| main | b6_main_s100 | yes | yes | **yes** | yes / yes | 0.00 / 0.94 | 0.03 / 0.94 | 0.0227 | 0.0227 |
| main | b6_main_s101 | yes | yes | **yes** | no / no | 0.00 / 0.98 | 0.04 / 0.78 | 0.0210 | 0.0212 |
| main | b6_main_s102 | yes | no | **no** | yes / yes | 0.00 / 0.41 | 0.05 / 0.46 | 0.0227 | 0.0224 |
| main | b6_main_s103 | yes | yes | **yes** | no / no | 0.00 / 1.00 | 0.05 / 0.81 | 0.0223 | 0.0222 |
| main | b6_main_s104 | yes | no | **no** | yes / yes | 0.00 / 0.96 | 0.03 / 0.96 | 0.0233 | 0.0231 |
| tau0 | b6_tau0_s200 | yes | no | **no** | no / no | 0.00 / 0.82 | 0.01 / 0.31 | 0.0161 | 0.0159 |
| tau0 | b6_tau0_s201 | yes | no | **no** | no / no | 0.00 / 0.36 | 0.00 / 0.70 | 0.0142 | 0.0143 |
| tau0 | b6_tau0_s202 | yes | no | **no** | no / no | 0.00 / 0.77 | 0.00 / 0.74 | 0.0154 | 0.0156 |
| untrained | b6_untrained_s300 | no | no | **no** | no / no | 0.07 / 0.35 | 0.22 / 0.94 | 16.2425 | — |
| untrained | b6_untrained_s301 | no | no | **no** | no / no | 0.12 / 0.66 | 0.15 / 0.33 | 1.8679 | — |
| untrained | b6_untrained_s302 | no | no | **no** | no / no | 0.12 / 0.25 | 0.16 / 0.15 | 2.1476 | — |

Rendering reference (not a gate): Stage 2 VICReg vorticity autoencoders on the same confirm3 frames: s2_ae_vorticity_vicreg_k0_lr0.001_s0: 0.0098, s2_ae_vorticity_vicreg_k0_lr0.001_s1: 0.0100, s2_ae_vorticity_vicreg_k0_lr0.001_s2: 0.0097.

## Every direction, every metric — confirm3 (spatial mean of decoded ω removed before measuring)

| group | run | direction | σ_d | Δa | Δs | effect | σ_shape | monot. | cycle corr / gain | env − ideal | shared | artefact frames | pre-removal mean > 10% (max over α) | D L S | fails |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| main | b6_main_s100 | loudness | 5.5 | -0.572 | -0.001 | 0.573 | 0.00 | 1.00 | 1.00 / 0.86 | -0.00 | 0.00 | 0.0% | 0.8% | DL· | — |
| main | b6_main_s100 | shape (loudness-null) | 5.12 | -0.025 | +0.355 | 0.380 | 0.94 | 1.00 | 0.96 / 0.70 | -0.00 | 0.00 | 0.0% | 1.2% | D·S | — |
| main | b6_main_s100 | PC2 | 5.12 | -0.117 | +0.390 | 0.507 | 0.77 | 1.00 | 0.96 / 0.67 | +0.00 | 0.00 | 0.0% | 1.2% | D·S | — |
| main | b6_main_s101 | loudness | 5.49 | +0.568 | +0.002 | 0.570 | 0.00 | 1.00 | 1.00 / 0.84 | -0.00 | 0.00 | 0.0% | 0.4% | DL· | — |
| main | b6_main_s101 | shape (loudness-null) | 5.22 | +0.007 | -0.386 | 0.392 | 0.98 | 1.00 | 0.99 / 0.90 | -0.00 | 0.00 | 0.0% | 2.3% | D·S | — |
| main | b6_main_s101 | PC2 | 5.16 | +0.258 | -0.443 | 0.701 | 0.63 | 1.00 | 0.98 / 0.79 | +0.00 | 0.01 | 0.0% | 2.0% | D·· | — |
| main | b6_main_s102 | loudness | 5.5 | +0.572 | +0.003 | 0.574 | 0.00 | 1.00 | 0.99 / 0.79 | -0.00 | 0.00 | 0.0% | 0.0% | DL· | — |
| main | b6_main_s102 | shape (loudness-null) | 2.11 | +0.059 | +0.040 | 0.099 | 0.41 | 0.36 | -0.21 / -0.05 | -0.05 | 0.00 | 3.5% | 1.6% | ··· | effect, monotonicity, cycle corr, cycle gain, grid artefacts |
| main | b6_main_s102 | PC2 | 4.47 | -0.016 | +0.351 | 0.368 | 0.96 | 1.00 | 0.98 / 0.80 | -0.00 | 0.00 | 0.0% | 0.0% | D·S | — |
| main | b6_main_s103 | loudness | 5.22 | +0.626 | +0.001 | 0.627 | 0.00 | 1.00 | 1.00 / 0.92 | -0.00 | 0.00 | 0.0% | 2.0% | DL· | — |
| main | b6_main_s103 | shape (loudness-null) | 5.34 | +0.001 | +0.325 | 0.326 | 1.00 | 1.00 | 0.95 / 0.60 | +0.00 | 0.00 | 0.0% | 23.0% | D·S | — |
| main | b6_main_s103 | PC2 | 5.05 | +0.312 | +0.179 | 0.491 | 0.36 | 1.00 | 0.99 / 0.85 | -0.00 | 0.00 | 0.0% | 0.4% | D·· | — |
| main | b6_main_s104 | loudness | 5.51 | -0.645 | -0.001 | 0.646 | 0.00 | 1.00 | 1.00 / 0.89 | -0.00 | 0.00 | 0.4% | 0.0% | DL· | — |
| main | b6_main_s104 | shape (loudness-null) | 3.7 | -0.014 | +0.334 | 0.348 | 0.96 | 1.00 | 0.95 / 0.58 | +0.00 | 0.00 | 0.0% | 1.6% | D·S | — |
| main | b6_main_s104 | PC2 | 3.84 | +0.066 | -0.367 | 0.433 | 0.85 | 1.00 | 0.96 / 0.65 | -0.00 | 0.00 | 0.0% | 0.0% | D·S | — |
| tau0 | b6_tau0_s200 | loudness | 6.37 | -0.778 | +0.004 | 0.781 | 0.00 | 1.00 | 0.99 / 0.89 | -0.00 | 0.00 | 0.0% | 0.8% | DL· | — |
| tau0 | b6_tau0_s200 | shape (loudness-null) | 3.94 | +0.004 | +0.019 | 0.023 | 0.82 | 1.00 | 0.63 / 0.14 | -0.03 | 0.00 | 0.0% | 0.4% | ··· | effect, cycle corr, cycle gain |
| tau0 | b6_tau0_s200 | PC2 | 4.35 | +0.108 | -0.012 | 0.120 | 0.10 | 1.00 | 0.42 / 0.04 | -0.00 | 0.00 | 0.0% | 0.0% | ··· | effect, cycle corr, cycle gain |
| tau0 | b6_tau0_s201 | loudness | 4.95 | +0.793 | -0.001 | 0.794 | 0.00 | 1.00 | 0.98 / 0.91 | -0.00 | 0.00 | 0.0% | 0.8% | DL· | — |
| tau0 | b6_tau0_s201 | shape (loudness-null) | 3.82 | -0.014 | -0.008 | 0.023 | 0.36 | 0.47 | 0.57 / 0.08 | -0.00 | 0.01 | 0.0% | 0.8% | ··· | effect, monotonicity, cycle corr, cycle gain |
| tau0 | b6_tau0_s201 | PC2 | 3.91 | -0.158 | +0.010 | 0.169 | 0.06 | 1.00 | -0.33 / -0.03 | -0.00 | 0.01 | 0.4% | 0.8% | ··· | effect, cycle corr, cycle gain |
| tau0 | b6_tau0_s202 | loudness | 6.09 | -0.838 | +0.003 | 0.841 | 0.00 | 1.00 | 0.99 / 0.98 | -0.00 | 0.00 | 0.0% | 0.0% | DL· | — |
| tau0 | b6_tau0_s202 | shape (loudness-null) | 3.36 | -0.004 | +0.013 | 0.017 | 0.77 | 1.00 | 0.70 / 0.07 | -0.00 | 0.01 | 0.0% | 0.0% | ··· | effect, cycle corr, cycle gain |
| tau0 | b6_tau0_s202 | PC2 | 3.79 | +0.210 | -0.016 | 0.226 | 0.07 | 1.00 | -0.72 / -0.06 | -0.00 | 0.00 | 0.0% | 0.0% | ··· | cycle corr, cycle gain |
| untrained | b6_untrained_s300 | loudness | 3 | -1.247 | -0.092 | 1.339 | 0.07 | 0.78 | 0.56 / 6.93 | -0.01 | 0.01 | 100.0% | 100.0% | ··· | monotonicity, cycle corr, cycle gain, grid artefacts |
| untrained | b6_untrained_s300 | shape (loudness-null) | 2.85 | -0.458 | -0.243 | 0.700 | 0.35 | 0.53 | 0.51 / 3.32 | -0.12 | 0.01 | 100.0% | 100.0% | ··· | monotonicity, cycle corr, envelope vs ideal, cycle gain, grid artefacts |
| untrained | b6_untrained_s300 | PC2 | 0.221 | -0.154 | +0.042 | 0.195 | 0.21 | 1.00 | -0.21 / -0.11 | -0.01 | 0.00 | 100.0% | 99.2% | ··· | effect, cycle corr, cycle gain, grid artefacts |
| untrained | b6_untrained_s301 | loudness | 2.15 | +0.352 | -0.049 | 0.401 | 0.12 | 0.51 | 0.44 / 1.16 | -0.02 | 0.00 | 100.0% | 97.7% | ··· | monotonicity, cycle corr, grid artefacts |
| untrained | b6_untrained_s301 | shape (loudness-null) | 2.15 | -0.080 | -0.158 | 0.238 | 0.66 | 0.67 | 0.37 / 0.57 | +0.00 | 0.00 | 100.0% | 100.0% | ··· | monotonicity, cycle corr, grid artefacts |
| untrained | b6_untrained_s301 | PC2 | 0.172 | -0.096 | +0.062 | 0.158 | 0.39 | 1.00 | 0.38 / 0.06 | -0.03 | 0.00 | 100.0% | 99.6% | ··· | effect, cycle corr, cycle gain, grid artefacts |
| untrained | b6_untrained_s302 | loudness | 1.24 | +0.801 | -0.111 | 0.912 | 0.12 | 0.98 | 0.57 / 1.20 | -0.03 | 0.00 | 100.0% | 52.3% | ··· | cycle corr, grid artefacts |
| untrained | b6_untrained_s302 | shape (loudness-null) | 0.953 | +0.139 | -0.047 | 0.186 | 0.25 | 0.57 | 0.15 / 0.13 | -0.16 | 0.01 | 100.0% | 96.9% | ··· | effect, monotonicity, cycle corr, envelope vs ideal, cycle gain, grid artefacts |
| untrained | b6_untrained_s302 | PC2 | 0.172 | +0.018 | -0.007 | 0.025 | 0.29 | 1.00 | -0.29 / -0.06 | -0.01 | 0.01 | 100.0% | 41.0% | ··· | effect, cycle corr, cycle gain, grid artefacts |

## Every direction, every metric — matched-energy-confirm (spatial mean of decoded ω removed before measuring)

| group | run | direction | σ_d | Δa | Δs | effect | σ_shape | monot. | cycle corr / gain | env − ideal | shared | artefact frames | pre-removal mean > 10% (max over α) | D L S | fails |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| main | b6_main_s100 | loudness | 5.5 | -0.555 | +0.019 | 0.575 | 0.03 | 1.00 | 1.00 / 0.87 | -0.01 | 0.01 | 0.0% | 0.0% | DL· | — |
| main | b6_main_s100 | shape (loudness-null) | 5.12 | +0.021 | +0.346 | 0.367 | 0.94 | 1.00 | 0.99 / 0.87 | -0.00 | 0.01 | 0.0% | 0.0% | D·S | — |
| main | b6_main_s100 | PC2 | 5.12 | -0.057 | +0.371 | 0.428 | 0.87 | 1.00 | 0.99 / 0.82 | -0.00 | 0.01 | 0.0% | 0.0% | D·S | — |
| main | b6_main_s101 | loudness | 5.49 | +0.574 | -0.025 | 0.599 | 0.04 | 1.00 | 1.00 / 0.89 | -0.01 | 0.01 | 0.0% | 0.0% | DL· | — |
| main | b6_main_s101 | shape (loudness-null) | 5.22 | -0.100 | -0.362 | 0.462 | 0.78 | 1.00 | 0.99 / 0.97 | +0.00 | 0.01 | 0.0% | 0.0% | D·S | — |
| main | b6_main_s101 | PC2 | 5.16 | +0.141 | -0.417 | 0.557 | 0.75 | 1.00 | 0.99 / 0.83 | +0.00 | 0.01 | 0.0% | 0.4% | D·S | — |
| main | b6_main_s102 | loudness | 5.5 | +0.588 | -0.032 | 0.620 | 0.05 | 1.00 | 1.00 / 0.79 | -0.01 | 0.01 | 0.0% | 0.0% | DL· | — |
| main | b6_main_s102 | shape (loudness-null) | 2.11 | +0.051 | +0.043 | 0.094 | 0.46 | 0.43 | -0.19 / -0.04 | -0.15 | 0.01 | 4.7% | 2.3% | ··· | effect, monotonicity, cycle corr, envelope vs ideal, cycle gain, grid artefacts |
| main | b6_main_s102 | PC2 | 4.47 | +0.112 | +0.321 | 0.433 | 0.74 | 1.00 | 0.99 / 0.90 | -0.00 | 0.01 | 0.0% | 0.0% | D·S | — |
| main | b6_main_s103 | loudness | 5.22 | +0.617 | -0.029 | 0.646 | 0.05 | 1.00 | 1.00 / 0.95 | -0.01 | 0.01 | 0.0% | 0.0% | DL· | — |
| main | b6_main_s103 | shape (loudness-null) | 5.34 | +0.069 | +0.302 | 0.371 | 0.81 | 1.00 | 0.99 / 0.68 | -0.00 | 0.01 | 0.0% | 0.0% | D·S | — |
| main | b6_main_s103 | PC2 | 5.05 | +0.369 | +0.162 | 0.531 | 0.30 | 1.00 | 1.00 / 0.92 | -0.00 | 0.01 | 0.0% | 0.0% | D·· | — |
| main | b6_main_s104 | loudness | 5.51 | -0.628 | +0.022 | 0.650 | 0.03 | 1.00 | 1.00 / 0.89 | -0.01 | 0.01 | 0.0% | 0.0% | DL· | — |
| main | b6_main_s104 | shape (loudness-null) | 3.7 | +0.015 | +0.336 | 0.351 | 0.96 | 1.00 | 0.97 / 0.69 | -0.09 | 0.01 | 0.4% | 0.4% | ··· | envelope vs ideal |
| main | b6_main_s104 | PC2 | 3.84 | +0.044 | -0.367 | 0.412 | 0.89 | 1.00 | 0.99 / 0.77 | -0.00 | 0.01 | 0.0% | 0.0% | D·S | — |
| tau0 | b6_tau0_s200 | loudness | 6.37 | -0.785 | +0.007 | 0.792 | 0.01 | 1.00 | 0.99 / 0.86 | -0.00 | 0.01 | 0.0% | 0.0% | DL· | — |
| tau0 | b6_tau0_s200 | shape (loudness-null) | 3.94 | +0.070 | +0.032 | 0.102 | 0.31 | 1.00 | 0.86 / 0.24 | -0.11 | 0.01 | 1.6% | 0.0% | ··· | effect, cycle corr, envelope vs ideal, cycle gain, grid artefacts |
| tau0 | b6_tau0_s200 | PC2 | 4.35 | +0.118 | -0.035 | 0.152 | 0.23 | 1.00 | 0.78 / 0.07 | -0.01 | 0.01 | 0.0% | 0.0% | ··· | effect, cycle corr, cycle gain |
| tau0 | b6_tau0_s201 | loudness | 4.95 | +0.812 | -0.003 | 0.815 | 0.00 | 1.00 | 0.98 / 0.91 | -0.00 | 0.01 | 0.0% | 0.0% | DL· | — |
| tau0 | b6_tau0_s201 | shape (loudness-null) | 3.82 | -0.011 | -0.024 | 0.035 | 0.70 | 1.00 | 0.93 / 0.12 | -0.01 | 0.01 | 0.0% | 0.4% | ··· | effect, cycle gain |
| tau0 | b6_tau0_s201 | PC2 | 3.91 | -0.164 | +0.029 | 0.193 | 0.15 | 1.00 | 0.00 / 0.00 | -0.01 | 0.01 | 0.0% | 0.4% | ··· | effect, cycle corr, cycle gain |
| tau0 | b6_tau0_s202 | loudness | 6.09 | -0.849 | +0.003 | 0.852 | 0.00 | 1.00 | 0.99 / 0.95 | -0.00 | 0.01 | 0.0% | 0.0% | DL· | — |
| tau0 | b6_tau0_s202 | shape (loudness-null) | 3.36 | +0.007 | +0.021 | 0.029 | 0.74 | 1.00 | 0.91 / 0.09 | -0.03 | 0.01 | 0.0% | 0.0% | ··· | effect, cycle gain |
| tau0 | b6_tau0_s202 | PC2 | 3.79 | +0.200 | -0.029 | 0.228 | 0.13 | 1.00 | -0.28 / -0.02 | -0.00 | 0.01 | 0.0% | 0.0% | ··· | cycle corr, cycle gain |
| untrained | b6_untrained_s300 | loudness | 3 | -0.637 | -0.180 | 0.818 | 0.22 | 0.45 | 0.65 / 1.10 | -0.03 | 0.01 | 100.0% | 100.0% | ··· | monotonicity, cycle corr, grid artefacts |
| untrained | b6_untrained_s300 | shape (loudness-null) | 2.85 | -0.020 | -0.319 | 0.339 | 0.94 | 0.72 | 0.13 / 0.09 | -0.05 | 0.01 | 100.0% | 100.0% | ··· | monotonicity, cycle corr, cycle gain, grid artefacts |
| untrained | b6_untrained_s300 | PC2 | 0.221 | -0.265 | +0.102 | 0.367 | 0.28 | 1.00 | 0.21 / 0.03 | -0.02 | 0.01 | 100.0% | 100.0% | ··· | cycle corr, cycle gain, grid artefacts |
| untrained | b6_untrained_s301 | loudness | 2.15 | -0.109 | -0.019 | 0.128 | 0.15 | 0.25 | -0.26 / -0.05 | +0.00 | 0.01 | 100.0% | 100.0% | ··· | effect, monotonicity, cycle corr, cycle gain, grid artefacts |
| untrained | b6_untrained_s301 | shape (loudness-null) | 2.15 | +0.249 | -0.125 | 0.374 | 0.33 | 0.53 | -0.46 / -0.08 | +0.00 | 0.01 | 100.0% | 100.0% | ··· | monotonicity, cycle corr, cycle gain, grid artefacts |
| untrained | b6_untrained_s301 | PC2 | 0.172 | -0.048 | +0.036 | 0.084 | 0.43 | 1.00 | -0.30 / -0.01 | -0.00 | 0.01 | 100.0% | 100.0% | ··· | effect, cycle corr, cycle gain, grid artefacts |
| untrained | b6_untrained_s302 | loudness | 1.24 | +0.712 | -0.133 | 0.844 | 0.16 | 0.95 | 0.69 / 0.12 | -0.02 | 0.01 | 100.0% | 87.9% | ··· | cycle corr, cycle gain, grid artefacts |
| untrained | b6_untrained_s302 | shape (loudness-null) | 0.953 | +0.225 | -0.040 | 0.265 | 0.15 | 0.53 | -0.45 / -0.08 | -0.13 | 0.01 | 100.0% | 100.0% | ··· | monotonicity, cycle corr, envelope vs ideal, cycle gain, grid artefacts |
| untrained | b6_untrained_s302 | PC2 | 0.172 | +0.030 | -0.013 | 0.044 | 0.31 | 1.00 | -0.49 / -0.01 | -0.00 | 0.01 | 100.0% | 67.6% | ··· | effect, cycle corr, cycle gain, grid artefacts |

Checkpoints verified against the frozen file: b6_main_s100 69d99574ec77, b6_main_s101 646f70864c77, b6_main_s102 05dd55ee2ce0, b6_main_s103 dda90e05d7da, b6_main_s104 5476785553d2, b6_tau0_s200 9f9ba703e485, b6_tau0_s201 41ab0ca15049, b6_tau0_s202 627e81cfaba7, b6_untrained_s300 d7beae1566b0, b6_untrained_s301 b99b0558d454, b6_untrained_s302 00e1bd3c9c16.

Wall-clock 116 min.
