# [BUILD] Stage 2: M2-SF, separate branches, scale-free content, one rendered field (exploratory)

Definitions and predictions: results/build/PREDICTIONS.md entries 2 and 4. The content encoder sees the input divided by its RMS, so loudness can reach the decoder only through the dials: M2-SF's loudness dial exists by construction and is a design property, not a finding. The finding is whether a label-free shape dial emerges. One rendered field (omega; psi is derived from it): no head-consistency criterion and no omega-psi contrast candidate.

Dial: effect ≥ 0.2, monotonicity ≥ 0.9, cycle corr ≥ 0.9 and gain in [0.5, 2.0], shared pattern ≤ 0.5, envelope ≥ ideal-dial envelope − 0.05 at every α, head inconsistency ≤ 1.5× its α = 0 value; on both explore and matched-energy; scored on the vorticity head. Loudness σ_shape ≤ 0.2, shape ≥ 0.7. A model produces a dial if ≥ 2 of 3 seeds have one among PC1–8 of the dial subspace.

## Summary

| model | seeds with a shape dial (PC1–8 / PC1–4) | seeds with a loudness dial (PC1–8) | produces a shape dial? | best shape dial loads on the ω−ψ contrast (cos² ≥ 0.5)? | contrast direction is a shape dial (seeds) | supervised s (dial / full) is a shape dial (seeds) |
|---|---|---|---|---|---|---|
| M2-SF | 0 / 0 of 3 | 3 of 3 | no | n/a (no contrast direction) | n/a | 0 / 0 |

## Rendering (val normalized MSE; trigger: > 2× the VICReg AE, > 0.02, or < 50% of the true k ≥ 10 power)

| model | seed | ω head MSE | ×plain AE | ×VICReg AE | ω k≥10 power kept | ψ head MSE | ×plain AE | ×VICReg AE | ψ k≥10 power | triggers | content gain | max dial std / median content std | r(top PC dial-ω, top PC dial-ψ) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M2-SF | 0 | 0.01818 | 15.2 | 1.74 | 0.48 | 0.00074 | 24.1 | 0.21 | 0.55 | vorticity: < 50% small-scale power | nan | 4.787 | nan |
| M2-SF | 1 | 0.01902 | 15.9 | 1.82 | 0.44 | 0.00072 | 23.3 | 0.20 | 0.51 | vorticity: < 50% small-scale power | nan | 3.716 | nan |
| M2-SF | 2 | 0.01876 | 15.7 | 1.79 | 0.45 | 0.00081 | 26.1 | 0.22 | 0.53 | vorticity: < 50% small-scale power | nan | 5.099 | nan |

Reference autoencoders (mean of 3 seeds): vorticity: plain 0.00119 (k≥10 0.93), VICReg 0.01047 (k≥10 0.65); streamfunction: plain 0.00003 (k≥10 1.57), VICReg 0.00362 (k≥10 9.83).

## Every candidate dial — explore (vorticity head; ψ head Δa / Δs alongside)

| model | seed | dir | σ_d | Δa | Δs | Δlog P | effect | σ_shape | monot. | cycle corr / gain | env − ideal (min) | shared | head ratio | ψ head Δa / Δs | cos² contrast | D L S | fails |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M2-SF | 0 | PC1 | 7.43 | -1.062 | +0.007 | -1.053 | 1.068 | 0.01 | 1.00 | 0.99 / 0.96 | -0.01 | 0.01 | — | — | — | DL· | — |
| M2-SF | 0 | PC2 | 1.73 | -0.040 | +0.006 | -0.028 | 0.046 | 0.13 | 1.00 | -0.59 / -0.02 | -0.00 | 0.01 | — | — | — | ··· | effect, cycle corr, cycle gain |
| M2-SF | 0 | PC3 | 0.864 | -0.014 | -0.001 | -0.017 | 0.016 | 0.08 | 1.00 | 0.52 / 0.01 | +0.00 | 0.09 | — | — | — | ··· | effect, cycle corr, cycle gain |
| M2-SF | 0 | PC4 | 0.399 | +0.001 | -0.001 | -0.005 | 0.002 | 0.66 | 1.00 | -0.04 / -0.00 | -0.00 | 0.01 | — | — | — | ··· | effect, cycle corr, cycle gain |
| M2-SF | 0 | PC5 | 0.25 | +0.006 | +0.002 | +0.010 | 0.007 | 0.22 | 1.00 | 0.60 / 0.01 | -0.00 | 0.02 | — | — | — | ··· | effect, cycle corr, cycle gain |
| M2-SF | 0 | PC6 | 0.223 | -0.002 | -0.001 | -0.003 | 0.002 | 0.23 | 1.00 | 0.39 / 0.00 | +0.00 | 0.01 | — | — | — | ··· | effect, cycle corr, cycle gain |
| M2-SF | 0 | PC7 | 0.0844 | +0.004 | +0.001 | +0.007 | 0.005 | 0.24 | 1.00 | -0.02 / -0.00 | +0.00 | 0.01 | — | — | — | ··· | effect, cycle corr, cycle gain |
| M2-SF | 0 | PC8 | 0.0701 | -0.002 | -0.000 | -0.002 | 0.002 | 0.17 | 1.00 | 0.04 / 0.00 | +0.00 | 0.01 | — | — | — | ··· | effect, cycle corr, cycle gain |
| M2-SF | 0 | supervised s (dial) | 1.09 | +0.142 | +0.006 | +0.160 | 0.147 | 0.04 | 0.63 | -0.18 / -0.01 | -0.00 | 0.02 | — | — | — | ··· | effect, monotonicity, cycle corr, cycle gain |
| M2-SF | 0 | supervised s (full) | 0.677 | +0.004 | +0.004 | +0.014 | 0.008 | 0.48 | 0.98 | 0.62 / 0.00 | -0.00 | 0.01 | — | — | — | ··· | effect, cycle corr, cycle gain |
| M2-SF | 1 | PC1 | 6.36 | -1.030 | +0.007 | -0.987 | 1.037 | 0.01 | 1.00 | 0.99 / 0.97 | -0.00 | 0.01 | — | — | — | DL· | — |
| M2-SF | 1 | PC2 | 1.86 | +0.068 | -0.007 | +0.037 | 0.075 | 0.10 | 1.00 | -0.55 / -0.02 | -0.00 | 0.02 | — | — | — | ··· | effect, cycle corr, cycle gain |
| M2-SF | 1 | PC3 | 1.44 | +0.043 | -0.003 | +0.033 | 0.046 | 0.07 | 1.00 | 0.30 / 0.02 | +0.00 | 0.02 | — | — | — | ··· | effect, cycle corr, cycle gain |
| M2-SF | 1 | PC4 | 0.346 | +0.003 | +0.001 | +0.007 | 0.004 | 0.25 | 1.00 | 0.33 / 0.00 | -0.00 | 0.05 | — | — | — | ··· | effect, cycle corr, cycle gain |
| M2-SF | 1 | PC5 | 0.225 | -0.005 | -0.000 | -0.006 | 0.005 | 0.07 | 1.00 | 0.49 / 0.01 | -0.00 | 0.02 | — | — | — | ··· | effect, cycle corr, cycle gain |
| M2-SF | 1 | PC6 | 0.171 | -0.004 | -0.002 | -0.010 | 0.006 | 0.31 | 1.00 | 0.20 / 0.00 | -0.00 | 0.01 | — | — | — | ··· | effect, cycle corr, cycle gain |
| M2-SF | 1 | PC7 | 0.131 | -0.006 | -0.001 | -0.008 | 0.007 | 0.10 | 1.00 | 0.28 / 0.00 | +0.00 | 0.02 | — | — | — | ··· | effect, cycle corr, cycle gain |
| M2-SF | 1 | PC8 | 0.0862 | +0.008 | +0.001 | +0.011 | 0.009 | 0.11 | 1.00 | -0.00 / -0.00 | -0.00 | 0.01 | — | — | — | ··· | effect, cycle corr, cycle gain |
| M2-SF | 1 | supervised s (dial) | 0.926 | +0.011 | +0.009 | +0.040 | 0.021 | 0.44 | 1.00 | 0.62 / 0.01 | -0.00 | 0.01 | — | — | — | ··· | effect, cycle corr, cycle gain |
| M2-SF | 1 | supervised s (full) | 1.08 | -0.001 | +0.008 | +0.026 | 0.009 | 0.87 | 1.00 | 0.79 / 0.01 | -0.00 | 0.01 | — | — | — | ··· | effect, cycle corr, cycle gain |
| M2-SF | 2 | PC1 | 7.33 | -1.035 | +0.015 | -0.998 | 1.050 | 0.01 | 1.00 | 0.99 / 0.94 | -0.01 | 0.01 | — | — | — | DL· | — |
| M2-SF | 2 | PC2 | 1.86 | -0.061 | +0.015 | -0.010 | 0.076 | 0.19 | 1.00 | 0.13 / 0.00 | -0.00 | 0.03 | — | — | — | ··· | effect, cycle corr, cycle gain |
| M2-SF | 2 | PC3 | 0.978 | +0.039 | -0.000 | +0.035 | 0.039 | 0.01 | 1.00 | 0.35 / 0.02 | -0.00 | 0.02 | — | — | — | ··· | effect, cycle corr, cycle gain |
| M2-SF | 2 | PC4 | 0.574 | +0.017 | +0.001 | +0.025 | 0.018 | 0.08 | 1.00 | 0.07 / 0.00 | -0.00 | 0.11 | — | — | — | ··· | effect, cycle corr, cycle gain |
| M2-SF | 2 | PC5 | 0.439 | +0.000 | -0.002 | -0.005 | 0.002 | 0.85 | 1.00 | 0.76 / 0.01 | +0.00 | 0.04 | — | — | — | ··· | effect, cycle corr, cycle gain |
| M2-SF | 2 | PC6 | 0.25 | +0.000 | +0.000 | +0.000 | 0.000 | 0.32 | 1.00 | 0.68 / 0.00 | +0.00 | 0.18 | — | — | — | ··· | effect, cycle corr, cycle gain |
| M2-SF | 2 | PC7 | 0.176 | -0.004 | -0.000 | -0.007 | 0.004 | 0.09 | 1.00 | 0.24 / 0.00 | -0.00 | 0.02 | — | — | — | ··· | effect, cycle corr, cycle gain |
| M2-SF | 2 | PC8 | 0.102 | -0.004 | -0.001 | -0.006 | 0.005 | 0.21 | 1.00 | 0.14 / 0.00 | +0.00 | 0.01 | — | — | — | ··· | effect, cycle corr, cycle gain |
| M2-SF | 2 | supervised s (dial) | 1.25 | +0.084 | +0.007 | +0.115 | 0.091 | 0.07 | 0.52 | 0.01 / 0.00 | -0.00 | 0.01 | — | — | — | ··· | effect, monotonicity, cycle corr, cycle gain |
| M2-SF | 2 | supervised s (full) | 1.75 | +0.204 | +0.001 | +0.244 | 0.205 | 0.01 | 0.47 | -0.18 / -0.02 | -0.00 | 0.02 | — | — | — | ··· | monotonicity, cycle corr, cycle gain |

## Every candidate dial — matched (vorticity head; ψ head Δa / Δs alongside)

| model | seed | dir | σ_d | Δa | Δs | Δlog P | effect | σ_shape | monot. | cycle corr / gain | env − ideal (min) | shared | head ratio | ψ head Δa / Δs | cos² contrast | D L S | fails |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M2-SF | 0 | PC1 | 7.43 | -1.169 | +0.021 | -1.129 | 1.190 | 0.02 | 1.00 | 1.00 / 0.99 | -0.01 | 0.01 | — | — | — | DL· | — |
| M2-SF | 0 | PC2 | 1.73 | -0.055 | +0.012 | -0.029 | 0.067 | 0.18 | 1.00 | -0.62 / -0.02 | -0.00 | 0.01 | — | — | — | ··· | effect, cycle corr, cycle gain |
| M2-SF | 0 | PC3 | 0.864 | -0.011 | -0.002 | -0.014 | 0.013 | 0.16 | 1.00 | 0.35 / 0.00 | -0.00 | 0.04 | — | — | — | ··· | effect, cycle corr, cycle gain |
| M2-SF | 0 | PC4 | 0.399 | -0.003 | -0.001 | -0.010 | 0.004 | 0.26 | 1.00 | -0.76 / -0.00 | -0.00 | 0.01 | — | — | — | ··· | effect, cycle corr, cycle gain |
| M2-SF | 0 | PC5 | 0.25 | +0.010 | +0.001 | +0.013 | 0.011 | 0.10 | 1.00 | 0.79 / 0.03 | -0.00 | 0.02 | — | — | — | ··· | effect, cycle corr, cycle gain |
| M2-SF | 0 | PC6 | 0.223 | -0.004 | -0.000 | -0.004 | 0.004 | 0.07 | 1.00 | 0.91 / 0.01 | +0.00 | 0.01 | — | — | — | ··· | effect, cycle gain |
| M2-SF | 0 | PC7 | 0.0844 | +0.006 | +0.001 | +0.009 | 0.007 | 0.15 | 1.00 | -0.58 / -0.01 | -0.00 | 0.02 | — | — | — | ··· | effect, cycle corr, cycle gain |
| M2-SF | 0 | PC8 | 0.0701 | -0.003 | -0.000 | -0.003 | 0.003 | 0.04 | 1.00 | 0.00 / 0.00 | +0.00 | 0.02 | — | — | — | ··· | effect, cycle corr, cycle gain |
| M2-SF | 0 | supervised s (dial) | 1.09 | +0.125 | +0.014 | +0.160 | 0.139 | 0.10 | 0.82 | -0.11 / -0.00 | -0.00 | 0.04 | — | — | — | ··· | effect, monotonicity, cycle corr, cycle gain |
| M2-SF | 0 | supervised s (full) | 0.677 | +0.004 | +0.004 | +0.016 | 0.008 | 0.52 | 1.00 | 0.17 / 0.00 | -0.00 | 0.02 | — | — | — | ··· | effect, cycle corr, cycle gain |
| M2-SF | 1 | PC1 | 6.36 | -1.130 | +0.028 | -1.055 | 1.158 | 0.02 | 1.00 | 1.00 / 1.00 | -0.00 | 0.01 | — | — | — | DL· | — |
| M2-SF | 1 | PC2 | 1.86 | +0.100 | -0.023 | +0.042 | 0.123 | 0.19 | 1.00 | 0.25 / 0.01 | -0.01 | 0.01 | — | — | — | ··· | effect, cycle corr, cycle gain |
| M2-SF | 1 | PC3 | 1.44 | +0.038 | -0.003 | +0.029 | 0.040 | 0.07 | 1.00 | -0.61 / -0.01 | -0.00 | 0.02 | — | — | — | ··· | effect, cycle corr, cycle gain |
| M2-SF | 1 | PC4 | 0.346 | +0.004 | +0.001 | +0.006 | 0.004 | 0.16 | 1.00 | -0.88 / -0.00 | -0.00 | 0.03 | — | — | — | ··· | effect, cycle corr, cycle gain |
| M2-SF | 1 | PC5 | 0.225 | -0.007 | -0.000 | -0.009 | 0.007 | 0.02 | 1.00 | 0.82 / 0.02 | +0.00 | 0.02 | — | — | — | ··· | effect, cycle corr, cycle gain |
| M2-SF | 1 | PC6 | 0.171 | -0.008 | -0.002 | -0.013 | 0.009 | 0.18 | 1.00 | 0.40 / 0.01 | -0.00 | 0.01 | — | — | — | ··· | effect, cycle corr, cycle gain |
| M2-SF | 1 | PC7 | 0.131 | -0.007 | -0.001 | -0.009 | 0.008 | 0.08 | 1.00 | 0.95 / 0.03 | -0.00 | 0.01 | — | — | — | ··· | effect, cycle gain |
| M2-SF | 1 | PC8 | 0.0862 | +0.011 | +0.000 | +0.013 | 0.012 | 0.04 | 1.00 | -0.39 / -0.01 | -0.00 | 0.01 | — | — | — | ··· | effect, cycle corr, cycle gain |
| M2-SF | 1 | supervised s (dial) | 0.926 | +0.017 | +0.011 | +0.047 | 0.028 | 0.40 | 1.00 | 0.88 / 0.02 | -0.00 | 0.02 | — | — | — | ··· | effect, cycle corr, cycle gain |
| M2-SF | 1 | supervised s (full) | 1.08 | -0.000 | +0.012 | +0.032 | 0.012 | 0.97 | 1.00 | 0.92 / 0.02 | -0.00 | 0.03 | — | — | — | ··· | effect, cycle gain |
| M2-SF | 2 | PC1 | 7.33 | -1.147 | +0.033 | -1.079 | 1.180 | 0.03 | 1.00 | 1.00 / 0.98 | -0.05 | 0.01 | — | — | — | DL· | — |
| M2-SF | 2 | PC2 | 1.86 | -0.101 | +0.030 | -0.029 | 0.131 | 0.23 | 1.00 | 0.50 / 0.02 | -0.00 | 0.01 | — | — | — | ··· | effect, cycle corr, cycle gain |
| M2-SF | 2 | PC3 | 0.978 | +0.041 | -0.001 | +0.032 | 0.042 | 0.02 | 1.00 | 0.74 / 0.02 | -0.00 | 0.01 | — | — | — | ··· | effect, cycle corr, cycle gain |
| M2-SF | 2 | PC4 | 0.574 | +0.022 | +0.000 | +0.029 | 0.022 | 0.02 | 1.00 | -0.97 / -0.02 | -0.00 | 0.05 | — | — | — | ··· | effect, cycle corr, cycle gain |
| M2-SF | 2 | PC5 | 0.439 | -0.000 | -0.001 | -0.004 | 0.002 | 0.90 | 1.00 | 0.72 / 0.02 | -0.00 | 0.01 | — | — | — | ··· | effect, cycle corr, cycle gain |
| M2-SF | 2 | PC6 | 0.25 | +0.001 | -0.000 | +0.000 | 0.002 | 0.14 | 1.00 | 0.63 / 0.00 | -0.00 | 0.06 | — | — | — | ··· | effect, cycle corr, cycle gain |
| M2-SF | 2 | PC7 | 0.176 | -0.005 | -0.001 | -0.008 | 0.006 | 0.13 | 1.00 | 0.81 / 0.02 | -0.00 | 0.02 | — | — | — | ··· | effect, cycle corr, cycle gain |
| M2-SF | 2 | PC8 | 0.102 | -0.003 | -0.001 | -0.006 | 0.003 | 0.18 | 1.00 | 0.65 / 0.01 | +0.00 | 0.01 | — | — | — | ··· | effect, cycle corr, cycle gain |
| M2-SF | 2 | supervised s (dial) | 1.25 | +0.058 | +0.016 | +0.105 | 0.074 | 0.22 | 0.28 | 0.61 / 0.04 | -0.01 | 0.01 | — | — | — | ··· | effect, monotonicity, cycle corr, cycle gain |
| M2-SF | 2 | supervised s (full) | 1.75 | +0.131 | +0.023 | +0.203 | 0.154 | 0.15 | 0.23 | 0.27 / 0.02 | -0.02 | 0.01 | — | — | — | ··· | effect, monotonicity, cycle corr, cycle gain |

## Prominence (E1-style: best of the top-3 slow directions, |r| with log Z₀ / log E₀, explore inviscid trajectories)

| model | seed | dial (8) | content (120) | full (128) |
|---|---|---|---|---|
| M2-SF | 0 | 0.63 / 0.99 | 0.07 / 0.61 | 0.70 / 0.98 |
| M2-SF | 1 | 0.64 / 0.96 | 0.11 / 0.52 | 0.55 / 0.96 |
| M2-SF | 2 | 0.62 / 0.98 | 0.10 / 0.58 | 0.66 / 0.96 |
| vorticity AE (reference) | 0 | | | 0.78 / 0.79 |
| vorticity AE (reference) | 1 | | | 0.80 / 0.94 |
| vorticity AE (reference) | 2 | | | 0.85 / 0.93 |
| streamfunction AE (reference) | 0 | | | 0.42 / 0.90 |
| streamfunction AE (reference) | 1 | | | 0.44 / 0.88 |
| streamfunction AE (reference) | 2 | | | 0.74 / 0.90 |
| vorticity AE + VICReg (reference) | 0 | | | 0.57 / 0.54 |
| vorticity AE + VICReg (reference) | 1 | | | 0.44 / 0.46 |
| vorticity AE + VICReg (reference) | 2 | | | 0.56 / 0.57 |
| streamfunction AE + VICReg (reference) | 0 | | | 0.38 / 0.57 |
| streamfunction AE + VICReg (reference) | 1 | | | 0.34 / 0.58 |
| streamfunction AE + VICReg (reference) | 2 | | | 0.28 / 0.54 |

## Ridge probes (fit v1 train, α on v1 val, scored on explore): R² (time-partialled R²); log ν: single frame (pair)

| model | seed | latent | log E | log Z | log P | s | log(P/Z) | log ν |
|---|---|---|---|---|---|---|---|---|
| M2-SF | 0 | content | 0.643 (0.311) | 0.131 (0.087) | 0.268 (0.212) | 0.915 (0.766) | 0.787 (0.591) | 0.096 (0.169) |
| M2-SF | 0 | dial | 0.998 (0.995) | 0.943 (0.938) | 0.798 (0.758) | 0.954 (0.852) | 0.760 (0.560) | 0.190 (0.642) |
| M2-SF | 0 | full | 0.999 (0.997) | 0.977 (0.967) | 0.864 (0.831) | 0.979 (0.911) | 0.813 (0.654) | 0.249 (0.682) |
| M2-SF | 1 | content | 0.623 (0.281) | 0.144 (0.087) | 0.252 (0.206) | 0.877 (0.677) | 0.733 (0.531) | 0.106 (0.167) |
| M2-SF | 1 | dial | 0.997 (0.993) | 0.954 (0.945) | 0.826 (0.774) | 0.948 (0.831) | 0.793 (0.583) | 0.264 (0.661) |
| M2-SF | 1 | full | 0.998 (0.995) | 0.975 (0.966) | 0.869 (0.829) | 0.973 (0.890) | 0.827 (0.654) | 0.276 (0.681) |
| M2-SF | 2 | content | 0.619 (0.268) | 0.130 (0.070) | 0.239 (0.184) | 0.878 (0.704) | 0.745 (0.536) | 0.086 (0.126) |
| M2-SF | 2 | dial | 0.997 (0.993) | 0.962 (0.957) | 0.852 (0.808) | 0.961 (0.877) | 0.807 (0.621) | 0.268 (0.653) |
| M2-SF | 2 | full | 0.998 (0.995) | 0.980 (0.973) | 0.886 (0.852) | 0.980 (0.917) | 0.840 (0.683) | 0.281 (0.673) |
| vorticity AE (reference) | 0 | pooled | 0.958 (0.890) | 0.867 (0.806) | 0.760 (0.688) | 0.946 (0.793) | 0.790 (0.573) | 0.337 (0.566) |
| vorticity AE (reference) | 1 | pooled | 0.960 (0.895) | 0.863 (0.800) | 0.755 (0.679) | 0.947 (0.794) | 0.789 (0.583) | 0.333 (0.563) |
| vorticity AE (reference) | 2 | pooled | 0.938 (0.842) | 0.843 (0.769) | 0.752 (0.680) | 0.937 (0.754) | 0.772 (0.546) | 0.332 (0.575) |
| streamfunction AE (reference) | 0 | pooled | 0.984 (0.958) | 0.912 (0.870) | 0.744 (0.678) | 0.959 (0.822) | 0.741 (0.522) | 0.164 (0.711) |
| streamfunction AE (reference) | 1 | pooled | 0.987 (0.965) | 0.926 (0.892) | 0.763 (0.702) | 0.959 (0.818) | 0.740 (0.519) | 0.175 (0.713) |
| streamfunction AE (reference) | 2 | pooled | 0.982 (0.951) | 0.900 (0.853) | 0.733 (0.666) | 0.957 (0.815) | 0.735 (0.516) | 0.156 (0.695) |
| vorticity AE + VICReg (reference) | 0 | pooled | 0.982 (0.952) | 0.973 (0.961) | 0.942 (0.924) | 0.970 (0.883) | 0.910 (0.816) | 0.403 (0.643) |
| vorticity AE + VICReg (reference) | 1 | pooled | 0.983 (0.956) | 0.976 (0.966) | 0.938 (0.923) | 0.970 (0.878) | 0.914 (0.823) | 0.418 (0.657) |
| vorticity AE + VICReg (reference) | 2 | pooled | 0.984 (0.959) | 0.975 (0.963) | 0.942 (0.927) | 0.975 (0.894) | 0.923 (0.842) | 0.414 (0.656) |
| streamfunction AE + VICReg (reference) | 0 | pooled | 0.985 (0.962) | 0.945 (0.915) | 0.772 (0.723) | 0.971 (0.869) | 0.727 (0.514) | 0.143 (0.400) |
| streamfunction AE + VICReg (reference) | 1 | pooled | 0.986 (0.965) | 0.946 (0.920) | 0.776 (0.724) | 0.971 (0.866) | 0.735 (0.517) | 0.156 (0.416) |
| streamfunction AE + VICReg (reference) | 2 | pooled | 0.986 (0.964) | 0.953 (0.928) | 0.782 (0.731) | 0.970 (0.860) | 0.726 (0.502) | 0.149 (0.392) |
