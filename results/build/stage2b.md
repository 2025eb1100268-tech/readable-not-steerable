# [BUILD] Stage 2b: M2-SW (spectrally whitened content) and the M2-SF attribution seed, omega-only loss (exploratory)

Definitions and predictions: results/build/PREDICTIONS.md entries 2 and 5. M2-SW's content encoder sees the spectrally whitened input (dials/whiten.py: RMS-normalized, every shell k = 1..21 set to a fixed reference power, gain capped at 1000, the mean and k > 21 zeroed), so loudness and (where the cap does not bind) spectral shape can reach the decoder only through the dials: shape information in the dials is a design property. The finding is whether label-free analysis of the dials yields separate loudness and shape directions that each steer cleanly and pass realism. PCA of the dial vector is the primary method; the effect-space directions ES1-4 are secondary. Both models use the omega-only loss (psi derived and reported). One rendered field: no head-consistency criterion.

Dial: effect ≥ 0.2, monotonicity ≥ 0.9, cycle corr ≥ 0.9 and gain in [0.5, 2.0], shared pattern ≤ 0.5, envelope ≥ ideal-dial envelope − 0.05 at every α, head inconsistency ≤ 1.5× its α = 0 value; on both explore and matched-energy; scored on the vorticity head. Loudness σ_shape ≤ 0.2, shape ≥ 0.7. A model produces a dial if ≥ 2 of 3 seeds have one among PC1–8 of the dial subspace.

## Summary

| model | seeds with a shape dial (PC1–8 / PC1–4) | seeds with a loudness dial (PC1–8) | produces a shape dial? | best shape dial loads on the ω−ψ contrast (cos² ≥ 0.5)? | contrast direction is a shape dial (seeds) | supervised s (dial / full) is a shape dial (seeds) |
|---|---|---|---|---|---|---|
| M2-SW | 2 / 2 of 3 | 0 of 3 | **yes** | n/a (no contrast direction) | n/a | 0 / 0 |
| M2-SF, omega-only loss | 0 / 0 of 1 | 1 of 1 | n/a (1 seed) | n/a (no contrast direction) | n/a | 0 / 0 |

## Effect-space method (secondary): top-4 right-singular vectors of the frame-averaged Jacobian of the decoded log shell spectrum with respect to the dial vector

| model | seed | singular values (top 4) | shape dials: PCA (PC1–8) / effect space (ES1–4) | loudness dials: PCA / effect space | cos² best PCA vs best ES shape dial | methods agree? | each ES's best-matching PC (cos²) |
|---|---|---|---|---|---|---|---|
| M2-SW | 0 | 3.12, 0.556, 0.353, 0.269 | PC2 / none | none / none | — | no | ES1→PC4 (0.49), ES2→PC2 (0.46), ES3→PC3 (0.68), ES4→PC1 (0.91) |
| M2-SW | 1 | 2.94, 0.562, 0.308, 0.258 | none / none | none / none | — | yes | ES1→PC7 (0.41), ES2→PC2 (0.55), ES3→PC1 (0.69), ES4→PC3 (0.53) |
| M2-SW | 2 | 3.78, 0.426, 0.3, 0.199 | PC2 / none | none / ES3 | — | no | ES1→PC6 (0.43), ES2→PC1 (0.38), ES3→PC1 (0.34), ES4→PC3 (0.51) |
| M2-SF, omega-only loss | 0 | 2.77, 0.41, 0.183, 0.0446 | none / none | PC1 / none | — | yes | ES1→PC7 (0.53), ES2→PC1 (0.88), ES3→PC4 (0.44), ES4→PC4 (0.37) |

M2-SW: effect-space shape dial in 0 of 3 seeds (produces: no); loudness in 1; methods agree in every seed: no.

M2-SF, omega-only loss: effect-space shape dial in 0 of 1 seeds (produces: n/a (1 seed)); loudness in 0; methods agree in every seed: yes.

## Rendering (val normalized MSE; trigger: > 2× the VICReg AE, > 0.02, or < 50% of the true k ≥ 10 power)

| model | seed | ω head MSE | ×plain AE | ×VICReg AE | ω k≥10 power kept | ψ head MSE | ×plain AE | ×VICReg AE | ψ k≥10 power | triggers | content gain | max dial std / median content std | r(top PC dial-ω, top PC dial-ψ) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M2-SW | 0 | 0.03991 | 33.4 | 3.81 | 0.52 | 0.00744 | 241.1 | 2.06 | 0.55 | vorticity: > 2× VICReg AE, > 0.02; streamfunction: > 2× VICReg AE | nan | 3.637 | nan |
| M2-SW | 1 | 0.03933 | 32.9 | 3.76 | 0.51 | 0.00698 | 226.1 | 1.93 | 0.55 | vorticity: > 2× VICReg AE, > 0.02 | nan | 5.049 | nan |
| M2-SW | 2 | 0.04082 | 34.2 | 3.90 | 0.48 | 0.00820 | 265.5 | 2.26 | 0.52 | vorticity: > 2× VICReg AE, > 0.02, < 50% small-scale power; streamfunction: > 2× VICReg AE | nan | 3.662 | nan |
| M2-SF, omega-only loss | 0 | 0.01525 | 12.8 | 1.46 | 0.52 | 0.00119 | 38.4 | 0.33 | 0.60 | none | nan | 4.957 | nan |

Reference autoencoders (mean of 3 seeds): vorticity: plain 0.00119 (k≥10 0.93), VICReg 0.01047 (k≥10 0.65); streamfunction: plain 0.00003 (k≥10 1.57), VICReg 0.00362 (k≥10 9.83).

## Every candidate dial — explore (vorticity head; ψ head Δa / Δs alongside)

| model | seed | dir | σ_d | Δa | Δs | Δlog P | effect | σ_shape | monot. | cycle corr / gain | env − ideal (min) | shared | head ratio | ψ head Δa / Δs | cos² contrast | D L S | fails |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M2-SW | 0 | PC1 | 6.24 | -0.845 | +0.310 | -0.326 | 1.154 | 0.27 | 1.00 | 0.95 / 0.78 | -0.00 | 0.01 | — | — | — | D·· | — |
| M2-SW | 0 | PC2 | 4.67 | -0.133 | +0.405 | +0.707 | 0.539 | 0.75 | 1.00 | 0.98 / 0.79 | -0.01 | 0.01 | — | — | — | D·S | — |
| M2-SW | 0 | PC3 | 3.24 | -0.062 | -0.026 | +0.072 | 0.088 | 0.30 | 1.00 | 0.96 / 0.63 | -0.01 | 0.00 | — | — | — | ··· | effect |
| M2-SW | 0 | PC4 | 1.45 | -0.005 | -0.034 | -0.153 | 0.039 | 0.87 | 1.00 | 0.88 / 0.30 | -0.01 | 0.01 | — | — | — | ··· | effect, cycle corr, cycle gain |
| M2-SW | 0 | PC5 | 1.36 | -0.014 | +0.024 | +0.035 | 0.038 | 0.62 | 1.00 | 0.88 / 0.08 | -0.00 | 0.21 | — | — | — | ··· | effect, cycle corr, cycle gain |
| M2-SW | 0 | PC6 | 0.873 | -0.005 | +0.027 | +0.065 | 0.031 | 0.85 | 1.00 | 0.84 / 0.08 | -0.00 | 0.01 | — | — | — | ··· | effect, cycle corr, cycle gain |
| M2-SW | 0 | PC7 | 0.557 | +0.005 | +0.011 | +0.032 | 0.016 | 0.68 | 1.00 | -0.15 / -0.00 | -0.00 | 0.01 | — | — | — | ··· | effect, cycle corr, cycle gain |
| M2-SW | 0 | PC8 | 0.302 | +0.001 | +0.016 | +0.045 | 0.017 | 0.95 | 1.00 | 0.02 / 0.00 | -0.00 | 0.01 | — | — | — | ··· | effect, cycle corr, cycle gain |
| M2-SW | 0 | ES1 | 2.07 | -0.079 | -0.152 | -0.548 | 0.231 | 0.66 | 1.00 | 0.97 / 0.63 | -0.06 | 0.01 | — | — | — | ··· | envelope vs ideal |
| M2-SW | 0 | ES2 | 3.56 | -0.217 | +0.277 | +0.360 | 0.494 | 0.56 | 1.00 | 0.92 / 0.52 | -0.01 | 0.01 | — | — | — | D·· | — |
| M2-SW | 0 | ES3 | 2.91 | +0.133 | +0.094 | +0.223 | 0.227 | 0.41 | 0.98 | 0.96 / 0.60 | -0.00 | 0.01 | — | — | — | D·· | — |
| M2-SW | 0 | ES4 | 6 | -0.711 | +0.305 | -0.202 | 1.016 | 0.30 | 1.00 | 0.94 / 0.65 | -0.00 | 0.01 | — | — | — | D·· | — |
| M2-SW | 0 | supervised s (dial) | 3 | -0.146 | +0.157 | +0.221 | 0.303 | 0.52 | 1.00 | 0.90 / 0.20 | -0.00 | 0.01 | — | — | — | ··· | cycle corr, cycle gain |
| M2-SW | 0 | supervised s (full) | 2.98 | -0.144 | +0.155 | +0.219 | 0.299 | 0.52 | 1.00 | 0.90 / 0.20 | -0.00 | 0.01 | — | — | — | ··· | cycle corr, cycle gain |
| M2-SW | 1 | PC1 | 6.75 | +0.796 | -0.221 | +0.452 | 1.016 | 0.22 | 1.00 | 0.97 / 0.80 | -0.01 | 0.01 | — | — | — | D·· | — |
| M2-SW | 1 | PC2 | 4.98 | +0.198 | -0.457 | -0.665 | 0.655 | 0.70 | 1.00 | 0.96 / 0.75 | -0.01 | 0.01 | — | — | — | D·· | — |
| M2-SW | 1 | PC3 | 3.45 | -0.106 | -0.007 | +0.095 | 0.112 | 0.06 | 1.00 | 0.95 / 0.66 | -0.01 | 0.01 | — | — | — | ··· | effect |
| M2-SW | 1 | PC4 | 1.94 | -0.003 | -0.000 | -0.029 | 0.003 | 0.12 | 1.00 | 0.77 / 0.11 | -0.00 | 0.33 | — | — | — | ··· | effect, cycle corr, cycle gain |
| M2-SW | 1 | PC5 | 1.43 | -0.040 | +0.051 | +0.100 | 0.090 | 0.56 | 1.00 | 0.64 / 0.07 | -0.00 | 0.03 | — | — | — | ··· | effect, cycle corr, cycle gain |
| M2-SW | 1 | PC6 | 1.27 | +0.002 | +0.035 | +0.091 | 0.037 | 0.95 | 1.00 | 0.86 / 0.15 | -0.00 | 0.01 | — | — | — | ··· | effect, cycle corr, cycle gain |
| M2-SW | 1 | PC7 | 0.496 | -0.018 | +0.035 | +0.054 | 0.053 | 0.65 | 1.00 | 0.76 / 0.08 | -0.00 | 0.00 | — | — | — | ··· | effect, cycle corr, cycle gain |
| M2-SW | 1 | PC8 | 0.208 | -0.001 | +0.000 | +0.001 | 0.001 | 0.07 | 1.00 | 0.02 / 0.00 | +0.00 | 0.16 | — | — | — | ··· | effect, cycle corr, cycle gain |
| M2-SW | 1 | ES1 | 2.16 | -0.065 | +0.228 | +0.479 | 0.292 | 0.78 | 1.00 | 0.90 / 0.46 | -0.05 | 0.02 | — | — | — | ··· | cycle corr, cycle gain |
| M2-SW | 1 | ES2 | 4.12 | -0.136 | +0.118 | +0.236 | 0.254 | 0.47 | 0.82 | 0.89 / 0.43 | -0.02 | 0.01 | — | — | — | ··· | monotonicity, cycle corr, cycle gain |
| M2-SW | 1 | ES3 | 5.89 | -0.595 | +0.464 | +0.121 | 1.059 | 0.44 | 0.88 | 0.78 / 0.50 | -0.00 | 0.07 | — | — | — | ··· | monotonicity, cycle corr, cycle gain |
| M2-SW | 1 | ES4 | 4.67 | -0.145 | -0.157 | -0.381 | 0.302 | 0.52 | 1.00 | 0.98 / 0.84 | -0.00 | 0.01 | — | — | — | D·· | — |
| M2-SW | 1 | supervised s (dial) | 2.2 | -0.151 | +0.261 | +0.387 | 0.412 | 0.63 | 1.00 | 0.90 / 0.31 | -0.01 | 0.02 | — | — | — | ··· | cycle corr, cycle gain |
| M2-SW | 1 | supervised s (full) | 2.21 | -0.148 | +0.262 | +0.391 | 0.410 | 0.64 | 1.00 | 0.90 / 0.31 | -0.01 | 0.02 | — | — | — | ··· | cycle corr, cycle gain |
| M2-SW | 2 | PC1 | 6.03 | -0.889 | +0.397 | -0.200 | 1.286 | 0.31 | 1.00 | 0.94 / 0.74 | -0.00 | 0.01 | — | — | — | D·· | — |
| M2-SW | 2 | PC2 | 5.13 | -0.003 | -0.356 | -0.785 | 0.359 | 0.99 | 1.00 | 0.97 / 0.80 | -0.01 | 0.00 | — | — | — | D·S | — |
| M2-SW | 2 | PC3 | 3.5 | -0.166 | +0.025 | +0.037 | 0.191 | 0.13 | 1.00 | 0.96 / 0.58 | -0.01 | 0.02 | — | — | — | ··· | effect |
| M2-SW | 2 | PC4 | 2.34 | +0.016 | -0.015 | -0.085 | 0.031 | 0.49 | 1.00 | 0.92 / 0.22 | -0.00 | 0.21 | — | — | — | ··· | effect, cycle gain |
| M2-SW | 2 | PC5 | 1.42 | +0.001 | +0.027 | +0.099 | 0.027 | 0.98 | 1.00 | 0.67 / 0.06 | -0.00 | 0.04 | — | — | — | ··· | effect, cycle corr, cycle gain |
| M2-SW | 2 | PC6 | 0.727 | -0.008 | +0.051 | +0.143 | 0.060 | 0.86 | 1.00 | 0.82 / 0.18 | -0.00 | 0.00 | — | — | — | ··· | effect, cycle corr, cycle gain |
| M2-SW | 2 | PC7 | 0.436 | +0.008 | -0.034 | -0.082 | 0.042 | 0.81 | 1.00 | 0.38 / 0.03 | -0.00 | 0.01 | — | — | — | ··· | effect, cycle corr, cycle gain |
| M2-SW | 2 | PC8 | 0.23 | -0.003 | +0.005 | +0.008 | 0.007 | 0.64 | 1.00 | 0.01 / 0.00 | -0.00 | 0.01 | — | — | — | ··· | effect, cycle corr, cycle gain |
| M2-SW | 2 | ES1 | 1.69 | +0.003 | +0.193 | +0.567 | 0.196 | 0.98 | 1.00 | 0.97 / 0.65 | -0.03 | 0.01 | — | — | — | ··· | effect |
| M2-SW | 2 | ES2 | 4.95 | +0.511 | -0.393 | -0.240 | 0.905 | 0.43 | 0.95 | 0.93 / 0.54 | -0.00 | 0.01 | — | — | — | D·· | — |
| M2-SW | 2 | ES3 | 4.91 | +0.500 | -0.016 | +0.423 | 0.515 | 0.03 | 1.00 | 0.99 / 0.80 | +0.00 | 0.01 | — | — | — | DL· | — |
| M2-SW | 2 | ES4 | 3.95 | -0.139 | +0.180 | +0.114 | 0.319 | 0.56 | 1.00 | 0.92 / 0.38 | -0.00 | 0.01 | — | — | — | ··· | cycle gain |
| M2-SW | 2 | supervised s (dial) | 1.93 | -0.098 | +0.192 | +0.386 | 0.290 | 0.66 | 1.00 | 0.96 / 0.26 | -0.00 | 0.00 | — | — | — | ··· | cycle gain |
| M2-SW | 2 | supervised s (full) | 1.91 | -0.097 | +0.184 | +0.368 | 0.281 | 0.65 | 1.00 | 0.96 / 0.25 | -0.00 | 0.00 | — | — | — | ··· | cycle gain |
| M2-SF, omega-only loss | 0 | PC1 | 7.22 | -0.933 | +0.011 | -0.925 | 0.943 | 0.01 | 1.00 | 0.99 / 0.95 | -0.01 | 0.01 | — | — | — | DL· | — |
| M2-SF, omega-only loss | 0 | PC2 | 2.22 | +0.081 | -0.014 | +0.039 | 0.095 | 0.15 | 1.00 | -0.12 / -0.00 | -0.00 | 0.01 | — | — | — | ··· | effect, cycle corr, cycle gain |
| M2-SF, omega-only loss | 0 | PC3 | 0.732 | -0.008 | +0.000 | -0.006 | 0.008 | 0.03 | 1.00 | 0.34 / 0.01 | -0.00 | 0.01 | — | — | — | ··· | effect, cycle corr, cycle gain |
| M2-SF, omega-only loss | 0 | PC4 | 0.582 | +0.002 | -0.000 | -0.005 | 0.003 | 0.04 | 1.00 | 0.52 / 0.02 | -0.00 | 0.02 | — | — | — | ··· | effect, cycle corr, cycle gain |
| M2-SF, omega-only loss | 0 | PC5 | 0.402 | +0.005 | +0.001 | +0.010 | 0.006 | 0.12 | 1.00 | 0.74 / 0.01 | -0.00 | 0.01 | — | — | — | ··· | effect, cycle corr, cycle gain |
| M2-SF, omega-only loss | 0 | PC6 | 0.206 | -0.000 | +0.001 | +0.001 | 0.001 | 0.66 | 1.00 | 0.34 / 0.00 | -0.00 | 0.03 | — | — | — | ··· | effect, cycle corr, cycle gain |
| M2-SF, omega-only loss | 0 | PC7 | 0.189 | +0.007 | -0.001 | +0.006 | 0.008 | 0.18 | 1.00 | 0.38 / 0.01 | +0.00 | 0.01 | — | — | — | ··· | effect, cycle corr, cycle gain |
| M2-SF, omega-only loss | 0 | PC8 | 0.0777 | -0.002 | +0.000 | -0.002 | 0.002 | 0.18 | 1.00 | 0.01 / 0.00 | +0.00 | 0.01 | — | — | — | ··· | effect, cycle corr, cycle gain |
| M2-SF, omega-only loss | 0 | ES1 | 1.48 | +0.322 | -0.028 | +0.305 | 0.350 | 0.08 | 1.00 | 0.88 / 0.33 | -0.02 | 0.03 | — | — | — | ··· | cycle corr, cycle gain |
| M2-SF, omega-only loss | 0 | ES2 | 6.77 | -0.589 | -0.013 | -0.584 | 0.602 | 0.02 | 0.88 | 0.91 / 0.57 | -0.00 | 0.07 | — | — | — | ··· | monotonicity |
| M2-SF, omega-only loss | 0 | ES3 | 1.45 | +0.073 | -0.011 | +0.037 | 0.084 | 0.13 | 1.00 | 0.67 / 0.03 | +0.00 | 0.01 | — | — | — | ··· | effect, cycle corr, cycle gain |
| M2-SF, omega-only loss | 0 | ES4 | 1.39 | -0.012 | +0.005 | -0.005 | 0.018 | 0.30 | 1.00 | 0.49 / 0.01 | -0.00 | 0.01 | — | — | — | ··· | effect, cycle corr, cycle gain |
| M2-SF, omega-only loss | 0 | supervised s (dial) | 1.23 | -0.057 | +0.011 | -0.040 | 0.068 | 0.16 | 1.00 | 0.82 / 0.02 | -0.00 | 0.01 | — | — | — | ··· | effect, cycle corr, cycle gain |
| M2-SF, omega-only loss | 0 | supervised s (full) | 0.0787 | -0.000 | +0.000 | -0.000 | 0.000 | 0.09 | 1.00 | 0.47 / 0.00 | +0.00 | 0.07 | — | — | — | ··· | effect, cycle corr, cycle gain |

## Every candidate dial — matched (vorticity head; ψ head Δa / Δs alongside)

| model | seed | dir | σ_d | Δa | Δs | Δlog P | effect | σ_shape | monot. | cycle corr / gain | env − ideal (min) | shared | head ratio | ψ head Δa / Δs | cos² contrast | D L S | fails |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M2-SW | 0 | PC1 | 6.24 | -0.572 | +0.269 | -0.172 | 0.841 | 0.32 | 1.00 | 0.94 / 0.53 | -0.01 | 0.01 | — | — | — | D·· | — |
| M2-SW | 0 | PC2 | 4.67 | +0.103 | +0.328 | +0.678 | 0.431 | 0.76 | 1.00 | 0.99 / 0.87 | -0.00 | 0.01 | — | — | — | D·S | — |
| M2-SW | 0 | PC3 | 3.24 | -0.208 | +0.127 | +0.089 | 0.334 | 0.38 | 1.00 | 0.97 / 0.70 | -0.01 | 0.01 | — | — | — | D·· | — |
| M2-SW | 0 | PC4 | 1.45 | -0.078 | -0.013 | -0.145 | 0.091 | 0.14 | 1.00 | 0.90 / 0.38 | -0.06 | 0.01 | — | — | — | ··· | effect, cycle corr, envelope vs ideal, cycle gain |
| M2-SW | 0 | PC5 | 1.36 | +0.009 | +0.021 | +0.033 | 0.030 | 0.71 | 1.00 | 0.85 / 0.05 | -0.00 | 0.09 | — | — | — | ··· | effect, cycle corr, cycle gain |
| M2-SW | 0 | PC6 | 0.873 | +0.058 | +0.014 | +0.072 | 0.072 | 0.19 | 1.00 | 0.93 / 0.11 | -0.00 | 0.01 | — | — | — | ··· | effect, cycle gain |
| M2-SW | 0 | PC7 | 0.557 | +0.013 | +0.012 | +0.032 | 0.025 | 0.49 | 1.00 | -0.67 / -0.02 | -0.00 | 0.02 | — | — | — | ··· | effect, cycle corr, cycle gain |
| M2-SW | 0 | PC8 | 0.302 | +0.016 | +0.014 | +0.043 | 0.030 | 0.45 | 1.00 | 0.32 / 0.02 | -0.00 | 0.01 | — | — | — | ··· | effect, cycle corr, cycle gain |
| M2-SW | 0 | ES1 | 2.07 | -0.212 | -0.123 | -0.517 | 0.335 | 0.37 | 1.00 | 0.96 / 0.63 | -0.16 | 0.01 | — | — | — | ··· | envelope vs ideal |
| M2-SW | 0 | ES2 | 3.56 | -0.033 | +0.248 | +0.361 | 0.281 | 0.88 | 1.00 | 0.97 / 0.70 | -0.00 | 0.03 | — | — | — | D·S | — |
| M2-SW | 0 | ES3 | 2.91 | +0.316 | -0.043 | +0.197 | 0.359 | 0.12 | 1.00 | 0.99 / 0.78 | -0.01 | 0.01 | — | — | — | DL· | — |
| M2-SW | 0 | ES4 | 6 | -0.445 | +0.249 | -0.089 | 0.694 | 0.36 | 1.00 | 0.90 / 0.39 | -0.02 | 0.01 | — | — | — | ··· | cycle corr, cycle gain |
| M2-SW | 0 | supervised s (dial) | 3 | -0.008 | +0.151 | +0.231 | 0.159 | 0.95 | 1.00 | 0.76 / 0.14 | -0.00 | 0.01 | — | — | — | ··· | effect, cycle corr, cycle gain |
| M2-SW | 0 | supervised s (full) | 2.98 | -0.009 | +0.150 | +0.229 | 0.159 | 0.94 | 1.00 | 0.76 / 0.14 | -0.00 | 0.01 | — | — | — | ··· | effect, cycle corr, cycle gain |
| M2-SW | 1 | PC1 | 6.75 | +0.612 | -0.210 | +0.324 | 0.822 | 0.26 | 1.00 | 0.96 / 0.63 | -0.01 | 0.02 | — | — | — | D·· | — |
| M2-SW | 1 | PC2 | 4.98 | +0.092 | -0.394 | -0.561 | 0.486 | 0.81 | 1.00 | 0.99 / 0.82 | -0.00 | 0.02 | — | — | — | D·S | — |
| M2-SW | 1 | PC3 | 3.45 | -0.229 | +0.138 | +0.131 | 0.368 | 0.38 | 1.00 | 0.98 / 0.82 | -0.01 | 0.01 | — | — | — | D·· | — |
| M2-SW | 1 | PC4 | 1.94 | -0.001 | -0.012 | -0.026 | 0.013 | 0.93 | 1.00 | 0.90 / 0.04 | -0.00 | 0.32 | — | — | — | ··· | effect, cycle gain |
| M2-SW | 1 | PC5 | 1.43 | +0.043 | +0.022 | +0.093 | 0.065 | 0.34 | 1.00 | 0.10 / 0.01 | -0.01 | 0.03 | — | — | — | ··· | effect, cycle corr, cycle gain |
| M2-SW | 1 | PC6 | 1.27 | +0.040 | +0.010 | +0.072 | 0.050 | 0.20 | 1.00 | 0.84 / 0.11 | -0.02 | 0.01 | — | — | — | ··· | effect, cycle corr, cycle gain |
| M2-SW | 1 | PC7 | 0.496 | -0.009 | +0.027 | +0.045 | 0.037 | 0.75 | 1.00 | 0.78 / 0.07 | -0.00 | 0.01 | — | — | — | ··· | effect, cycle corr, cycle gain |
| M2-SW | 1 | PC8 | 0.208 | -0.001 | +0.001 | +0.000 | 0.002 | 0.32 | 1.00 | 0.49 / 0.00 | -0.00 | 0.09 | — | — | — | ··· | effect, cycle corr, cycle gain |
| M2-SW | 1 | ES1 | 2.16 | +0.095 | +0.175 | +0.428 | 0.270 | 0.65 | 1.00 | 0.91 / 0.47 | -0.13 | 0.03 | — | — | — | ··· | envelope vs ideal, cycle gain |
| M2-SW | 1 | ES2 | 4.12 | -0.220 | +0.180 | +0.127 | 0.400 | 0.45 | 1.00 | 0.87 / 0.50 | -0.01 | 0.04 | — | — | — | ··· | cycle corr, cycle gain |
| M2-SW | 1 | ES3 | 5.89 | -0.323 | +0.410 | +0.260 | 0.733 | 0.56 | 1.00 | 0.75 / 0.31 | -0.02 | 0.21 | — | — | — | ··· | cycle corr, cycle gain |
| M2-SW | 1 | ES4 | 4.67 | -0.406 | +0.024 | -0.248 | 0.430 | 0.06 | 1.00 | 0.99 / 0.97 | -0.01 | 0.01 | — | — | — | DL· | — |
| M2-SW | 1 | supervised s (dial) | 2.2 | -0.003 | +0.213 | +0.370 | 0.216 | 0.99 | 1.00 | 0.91 / 0.27 | -0.06 | 0.03 | — | — | — | ··· | envelope vs ideal, cycle gain |
| M2-SW | 1 | supervised s (full) | 2.21 | -0.002 | +0.215 | +0.373 | 0.217 | 0.99 | 1.00 | 0.91 / 0.27 | -0.07 | 0.03 | — | — | — | ··· | envelope vs ideal, cycle gain |
| M2-SW | 2 | PC1 | 6.03 | -0.664 | +0.360 | -0.097 | 1.024 | 0.35 | 1.00 | 0.92 / 0.50 | -0.00 | 0.01 | — | — | — | D·· | — |
| M2-SW | 2 | PC2 | 5.13 | -0.107 | -0.311 | -0.681 | 0.418 | 0.74 | 1.00 | 0.99 / 0.96 | -0.00 | 0.01 | — | — | — | D·S | — |
| M2-SW | 2 | PC3 | 3.5 | -0.274 | +0.136 | +0.050 | 0.410 | 0.33 | 1.00 | 0.99 / 0.74 | -0.01 | 0.01 | — | — | — | D·· | — |
| M2-SW | 2 | PC4 | 2.34 | -0.014 | -0.019 | -0.094 | 0.034 | 0.58 | 1.00 | 0.95 / 0.16 | -0.02 | 0.14 | — | — | — | ··· | effect, cycle gain |
| M2-SW | 2 | PC5 | 1.42 | +0.041 | +0.020 | +0.094 | 0.062 | 0.33 | 1.00 | 0.69 / 0.06 | -0.01 | 0.01 | — | — | — | ··· | effect, cycle corr, cycle gain |
| M2-SW | 2 | PC6 | 0.727 | +0.020 | +0.047 | +0.125 | 0.066 | 0.70 | 1.00 | 0.82 / 0.09 | -0.00 | 0.01 | — | — | — | ··· | effect, cycle corr, cycle gain |
| M2-SW | 2 | PC7 | 0.436 | -0.008 | -0.031 | -0.073 | 0.038 | 0.80 | 1.00 | 0.27 / 0.02 | -0.00 | 0.01 | — | — | — | ··· | effect, cycle corr, cycle gain |
| M2-SW | 2 | PC8 | 0.23 | -0.003 | +0.005 | +0.007 | 0.007 | 0.64 | 1.00 | -0.04 / -0.00 | -0.00 | 0.03 | — | — | — | ··· | effect, cycle corr, cycle gain |
| M2-SW | 2 | ES1 | 1.69 | +0.088 | +0.178 | +0.493 | 0.267 | 0.67 | 1.00 | 0.96 / 0.64 | -0.07 | 0.01 | — | — | — | ··· | envelope vs ideal |
| M2-SW | 2 | ES2 | 4.95 | +0.382 | -0.357 | -0.231 | 0.739 | 0.48 | 0.98 | 0.98 / 0.65 | -0.00 | 0.01 | — | — | — | D·· | — |
| M2-SW | 2 | ES3 | 4.91 | +0.562 | -0.090 | +0.342 | 0.651 | 0.14 | 1.00 | 0.99 / 0.81 | -0.00 | 0.01 | — | — | — | DL· | — |
| M2-SW | 2 | ES4 | 3.95 | +0.038 | +0.094 | +0.122 | 0.132 | 0.71 | 1.00 | 0.92 / 0.27 | -0.01 | 0.01 | — | — | — | ··· | effect, cycle gain |
| M2-SW | 2 | supervised s (dial) | 1.93 | -0.015 | +0.181 | +0.345 | 0.196 | 0.92 | 1.00 | 0.95 / 0.19 | -0.00 | 0.01 | — | — | — | ··· | effect, cycle gain |
| M2-SW | 2 | supervised s (full) | 1.91 | -0.015 | +0.174 | +0.330 | 0.189 | 0.92 | 1.00 | 0.95 / 0.18 | -0.00 | 0.01 | — | — | — | ··· | effect, cycle gain |
| M2-SF, omega-only loss | 0 | PC1 | 7.22 | -0.946 | +0.020 | -0.907 | 0.966 | 0.02 | 1.00 | 1.00 / 0.96 | -0.03 | 0.01 | — | — | — | DL· | — |
| M2-SF, omega-only loss | 0 | PC2 | 2.22 | +0.082 | -0.026 | +0.019 | 0.109 | 0.24 | 1.00 | 0.79 / 0.03 | -0.00 | 0.01 | — | — | — | ··· | effect, cycle corr, cycle gain |
| M2-SF, omega-only loss | 0 | PC3 | 0.732 | -0.007 | -0.000 | -0.005 | 0.007 | 0.01 | 1.00 | 0.20 / 0.00 | -0.00 | 0.04 | — | — | — | ··· | effect, cycle corr, cycle gain |
| M2-SF, omega-only loss | 0 | PC4 | 0.582 | +0.008 | +0.001 | +0.002 | 0.009 | 0.11 | 1.00 | 0.90 / 0.05 | -0.00 | 0.01 | — | — | — | ··· | effect, cycle gain |
| M2-SF, omega-only loss | 0 | PC5 | 0.402 | +0.005 | -0.000 | +0.009 | 0.005 | 0.01 | 1.00 | 0.90 / 0.01 | -0.00 | 0.01 | — | — | — | ··· | effect, cycle gain |
| M2-SF, omega-only loss | 0 | PC6 | 0.206 | +0.001 | +0.001 | +0.002 | 0.002 | 0.49 | 1.00 | 0.62 / 0.01 | -0.00 | 0.01 | — | — | — | ··· | effect, cycle corr, cycle gain |
| M2-SF, omega-only loss | 0 | PC7 | 0.189 | +0.004 | -0.002 | +0.005 | 0.005 | 0.30 | 0.98 | 0.77 / 0.04 | -0.00 | 0.01 | — | — | — | ··· | effect, cycle corr, cycle gain |
| M2-SF, omega-only loss | 0 | PC8 | 0.0777 | -0.001 | +0.000 | -0.001 | 0.002 | 0.24 | 1.00 | 0.54 / 0.00 | -0.00 | 0.01 | — | — | — | ··· | effect, cycle corr, cycle gain |
| M2-SF, omega-only loss | 0 | ES1 | 1.48 | +0.290 | -0.025 | +0.294 | 0.315 | 0.08 | 1.00 | 0.92 / 0.30 | -0.10 | 0.08 | — | — | — | ··· | envelope vs ideal, cycle gain |
| M2-SF, omega-only loss | 0 | ES2 | 6.77 | -0.629 | -0.006 | -0.575 | 0.635 | 0.01 | 0.95 | 0.96 / 0.64 | +0.00 | 0.13 | — | — | — | DL· | — |
| M2-SF, omega-only loss | 0 | ES3 | 1.45 | +0.070 | -0.016 | +0.030 | 0.086 | 0.19 | 1.00 | 0.93 / 0.07 | -0.00 | 0.03 | — | — | — | ··· | effect, cycle gain |
| M2-SF, omega-only loss | 0 | ES4 | 1.39 | -0.007 | +0.010 | +0.011 | 0.017 | 0.57 | 1.00 | 0.87 / 0.03 | -0.00 | 0.01 | — | — | — | ··· | effect, cycle corr, cycle gain |
| M2-SF, omega-only loss | 0 | supervised s (dial) | 1.23 | -0.047 | +0.013 | -0.029 | 0.060 | 0.21 | 1.00 | 0.94 / 0.03 | -0.00 | 0.02 | — | — | — | ··· | effect, cycle gain |
| M2-SF, omega-only loss | 0 | supervised s (full) | 0.0787 | -0.000 | +0.000 | -0.000 | 0.000 | 0.12 | 1.00 | 0.50 / 0.00 | +0.00 | 0.14 | — | — | — | ··· | effect, cycle corr, cycle gain |

## Prominence (E1-style: best of the top-3 slow directions, |r| with log Z₀ / log E₀, explore inviscid trajectories)

| model | seed | dial (8) | content (120) | full (128) |
|---|---|---|---|---|
| M2-SW | 0 | 0.96 / 0.98 | 0.04 / 0.30 | 0.69 / 0.92 |
| M2-SW | 1 | 0.82 / 0.96 | 0.10 / 0.24 | 0.72 / 0.81 |
| M2-SW | 2 | 0.59 / 0.95 | 0.08 / 0.35 | 0.73 / 0.91 |
| M2-SF, omega-only loss | 0 | 0.75 / 0.96 | 0.14 / 0.33 | 0.72 / 0.85 |
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
| M2-SW | 0 | content | 0.114 (0.054) | 0.008 (0.000) | 0.045 (0.019) | 0.198 (0.182) | 0.141 (0.123) | 0.013 (0.014) |
| M2-SW | 0 | dial | 0.976 (0.945) | 0.947 (0.928) | 0.963 (0.956) | 0.988 (0.952) | 0.939 (0.879) | 0.471 (0.777) |
| M2-SW | 0 | full | 0.977 (0.948) | 0.951 (0.934) | 0.965 (0.956) | 0.988 (0.953) | 0.940 (0.881) | 0.469 (0.776) |
| M2-SW | 1 | content | 0.111 (0.022) | 0.013 (0.002) | 0.055 (0.033) | 0.195 (0.154) | 0.142 (0.129) | 0.020 (0.020) |
| M2-SW | 1 | dial | 0.966 (0.935) | 0.935 (0.915) | 0.959 (0.947) | 0.977 (0.916) | 0.918 (0.836) | 0.442 (0.739) |
| M2-SW | 1 | full | 0.968 (0.938) | 0.941 (0.923) | 0.961 (0.949) | 0.978 (0.919) | 0.919 (0.838) | 0.441 (0.743) |
| M2-SW | 2 | content | 0.107 (0.049) | 0.012 (0.003) | 0.057 (0.031) | 0.199 (0.162) | 0.147 (0.122) | 0.006 (0.006) |
| M2-SW | 2 | dial | 0.976 (0.944) | 0.964 (0.949) | 0.966 (0.957) | 0.975 (0.901) | 0.954 (0.902) | 0.453 (0.769) |
| M2-SW | 2 | full | 0.977 (0.947) | 0.970 (0.956) | 0.968 (0.958) | 0.976 (0.905) | 0.955 (0.903) | 0.451 (0.773) |
| M2-SF, omega-only loss | 0 | content | 0.072 (0.200) | 0.078 (0.105) | 0.196 (0.259) | 0.098 (0.551) | 0.192 (0.507) | 0.143 (0.180) |
| M2-SF, omega-only loss | 0 | dial | 0.997 (0.992) | 0.944 (0.957) | 0.856 (0.828) | 0.943 (0.864) | 0.833 (0.662) | 0.334 (0.655) |
| M2-SF, omega-only loss | 0 | full | 0.997 (0.993) | 0.978 (0.975) | 0.901 (0.874) | 0.975 (0.916) | 0.867 (0.728) | 0.363 (0.677) |
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
