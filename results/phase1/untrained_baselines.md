# Phase 1 untrained-encoder baselines (val split; fit on train)

| target | untrained val R², time × family partialled (pre-registered), mean ± std over 3 seeds | time-only partialled | plain R² | bootstrap SD | R² a trained model needs |
|---|---|---|---|---|---|
| log_Z (S1) | 0.762 ± 0.027 | 0.782 | 0.811 | 0.019 | 0.881 |
| log_P (S1) | 0.443 ± 0.031 | 0.511 | 0.570 | 0.033 | 0.643 |
| log1p_vortex_count (S1) | 0.580 ± 0.008 | 0.899 | 0.897 | 0.033 | 0.780 |
| ow_vort_frac (S1) | 0.606 ± 0.015 | 0.844 | 0.844 | 0.041 | 0.803 |
| log_eps (S1) | 0.236 ± 0.008 | 0.259 | 0.296 | 0.048 | 0.436 |
| log_E | 0.951 ± 0.007 | 0.981 | 0.982 | 0.004 | — |
| log_max_abs_w | 0.638 ± 0.034 | 0.758 | 0.778 | 0.027 | — |
| log_kurtosis | 0.723 ± 0.024 | 0.850 | 0.849 | 0.035 | — |
| ow_strain_frac | 0.596 ± 0.019 | 0.599 | 0.605 | 0.045 | — |
| time | 0.204 ± 0.011 | — | 0.204 | 0.024 | — |
| log ν, frame pair (S2) | 0.486 ± 0.049 | — | — | — | 0.786 |
| log ν, single frame (unscored) | 0.089 ± 0.018 | — | — | — | — |

| invariant (inviscid, fit train → score val) | untrained max \|r\| over top-3, mean ± std | 95% CI (seed 0) | \|r\| a trained model needs |
|---|---|---|---|
| log_E | 0.938 ± 0.007 | 0.92–0.96 | 1.038 (unreachable) |
| log_Z | 0.233 ± 0.009 | 0.13–0.48 | 0.800 |
| log_C4 (approx.) | 0.560 ± 0.007 | 0.40–0.69 | 0.800 |
