# Phase 3 val preview (decides nothing; alpha chosen on the same val set)

## Readout: raw

| quantity | untrained | jepa_spatial | pixel | jepa_pooled | static_vicreg | nonphys_jepa_spatial |
|---|---|---|---|---|---|---|
| log_Z | 0.762 ± 0.027 | 0.921 ± 0.014 | 0.901 ± 0.007 | 0.862 ± 0.013 | 0.977 ± 0.003 | 0.948 ± 0.006 |
| log_P | 0.443 ± 0.031 | 0.723 ± 0.028 | 0.731 ± 0.004 | 0.635 ± 0.037 | 0.903 ± 0.018 | 0.787 ± 0.010 |
| log1p_vortex_count | 0.580 ± 0.008 | 0.764 ± 0.005 | 0.517 ± 0.020 | 0.661 ± 0.014 | 0.851 ± 0.005 | 0.805 ± 0.012 |
| ow_vort_frac | 0.606 ± 0.015 | 0.620 ± 0.036 | 0.438 ± 0.011 | 0.528 ± 0.007 | 0.772 ± 0.053 | 0.625 ± 0.015 |
| log_eps | 0.236 ± 0.008 | 0.235 ± 0.005 | 0.247 ± 0.002 | 0.193 ± 0.009 | 0.258 ± 0.021 | 0.245 ± 0.028 |
| log_E | 0.951 ± 0.007 | 0.979 ± 0.001 | 0.961 ± 0.005 | 0.929 ± 0.008 | 0.980 ± 0.002 | 0.987 ± 0.000 |
| log_max_abs_w | 0.638 ± 0.034 | 0.778 ± 0.035 | 0.733 ± 0.012 | 0.697 ± 0.026 | 0.905 ± 0.009 | 0.778 ± 0.015 |
| log_kurtosis | 0.723 ± 0.024 | 0.733 ± 0.035 | 0.513 ± 0.013 | 0.628 ± 0.015 | 0.889 ± 0.036 | 0.703 ± 0.019 |
| ow_strain_frac | 0.596 ± 0.019 | 0.558 ± 0.021 | 0.498 ± 0.016 | 0.452 ± 0.017 | 0.751 ± 0.025 | 0.580 ± 0.040 |
| time (plain R²) | 0.204 ± 0.011 | 0.271 ± 0.005 | 0.287 ± 0.008 | 0.242 ± 0.008 | 0.319 ± 0.039 | 0.313 ± 0.014 |
| log_nu pair (S2) | 0.486 ± 0.049 | 0.413 ± 0.033 | 0.692 ± 0.001 | 0.259 ± 0.039 | 0.675 ± 0.027 | 0.497 ± 0.020 |
| log_nu single | 0.089 ± 0.018 | 0.137 ± 0.011 | 0.189 ± 0.010 | 0.102 ± 0.022 | 0.255 ± 0.018 | 0.151 ± 0.025 |
| slow |r| log_E | 0.938 ± 0.007 | 0.769 ± 0.029 | 0.889 ± 0.015 | 0.661 ± 0.030 | 0.694 ± 0.097 | 0.699 ± 0.073 |
| slow |r| log_Z | 0.233 ± 0.009 | 0.441 ± 0.035 | 0.849 ± 0.067 | 0.491 ± 0.045 | 0.527 ± 0.036 | 0.428 ± 0.054 |
| slow |r| log_C4 | 0.560 ± 0.007 | 0.244 ± 0.038 | 0.582 ± 0.020 | 0.294 ± 0.059 | 0.584 ± 0.161 | 0.362 ± 0.033 |
| slow |r| kurtosis (traj mean) | 0.407 ± 0.008 | 0.416 ± 0.058 | 0.368 ± 0.001 | 0.364 ± 0.017 | 0.410 ± 0.066 | 0.419 ± 0.128 |
| slow |r| OW fraction (traj mean) | 0.695 ± 0.005 | 0.664 ± 0.045 | 0.636 ± 0.003 | 0.562 ± 0.035 | 0.499 ± 0.034 | 0.575 ± 0.178 |
| MLP log_Z | 0.832 ± 0.011 | 0.946 ± 0.011 | 0.964 ± 0.004 | 0.899 ± 0.015 | 0.991 ± 0.001 | 0.966 ± 0.004 |
| MLP log_P | 0.482 ± 0.011 | 0.736 ± 0.035 | 0.898 ± 0.006 | 0.668 ± 0.039 | 0.913 ± 0.024 | 0.790 ± 0.016 |
| MLP log1p_vortex_count | 0.608 ± 0.022 | 0.774 ± 0.013 | 0.731 ± 0.003 | 0.695 ± 0.017 | 0.883 ± 0.008 | 0.802 ± 0.018 |
| MLP ow_vort_frac | 0.648 ± 0.013 | 0.628 ± 0.028 | 0.562 ± 0.019 | 0.560 ± 0.012 | 0.786 ± 0.047 | 0.627 ± 0.008 |
| MLP log_eps | 0.218 ± 0.013 | 0.217 ± 0.009 | 0.269 ± 0.010 | 0.162 ± 0.010 | 0.243 ± 0.029 | 0.226 ± 0.034 |

- **jepa_spatial vs control (val preview)**: S1 fails (2/5 count; OW fraction does not count), S2 fails (pair R² 0.413 vs 0.486), S3 fails (Z |r| 0.441 vs 0.233) → not supported
- **pixel vs control (val preview)**: S1 fails (2/5 count; OW fraction does not count), S2 fails (pair R² 0.692 vs 0.486), S3 holds (Z |r| 0.849 vs 0.233) → not supported
- **jepa_pooled vs control (val preview)**: S1 fails (0/5 count; OW fraction does not count), S2 fails (pair R² 0.259 vs 0.486), S3 fails (Z |r| 0.491 vs 0.233) → not supported
- **static_vicreg vs control (val preview)**: S1 fails (3/5 count; OW fraction does not count), S2 fails (pair R² 0.675 vs 0.486), S3 fails (Z |r| 0.527 vs 0.233) → not supported
- **nonphys_jepa_spatial vs control (val preview)**: S1 fails (3/5 count; OW fraction does not count), S2 fails (pair R² 0.497 vs 0.486), S3 fails (Z |r| 0.428 vs 0.233) → not supported

## Readout: shift_avg

| quantity | untrained | jepa_spatial | pixel | jepa_pooled | static_vicreg | nonphys_jepa_spatial |
|---|---|---|---|---|---|---|
| log_Z | 0.791 ± 0.015 | 0.922 ± 0.013 | 0.922 ± 0.012 | 0.864 ± 0.013 | 0.977 ± 0.003 | 0.949 ± 0.005 |
| log_P | 0.491 ± 0.007 | 0.726 ± 0.026 | 0.768 ± 0.016 | 0.637 ± 0.035 | 0.904 ± 0.018 | 0.791 ± 0.010 |
| log1p_vortex_count | 0.608 ± 0.011 | 0.764 ± 0.005 | 0.608 ± 0.034 | 0.664 ± 0.015 | 0.851 ± 0.005 | 0.806 ± 0.014 |
| ow_vort_frac | 0.643 ± 0.007 | 0.621 ± 0.039 | 0.537 ± 0.011 | 0.531 ± 0.007 | 0.773 ± 0.053 | 0.628 ± 0.017 |
| log_eps | 0.240 ± 0.008 | 0.235 ± 0.005 | 0.267 ± 0.005 | 0.195 ± 0.009 | 0.257 ± 0.021 | 0.246 ± 0.027 |
| log_E | 0.961 ± 0.006 | 0.980 ± 0.001 | 0.972 ± 0.004 | 0.930 ± 0.008 | 0.980 ± 0.003 | 0.987 ± 0.000 |
| log_max_abs_w | 0.673 ± 0.011 | 0.780 ± 0.035 | 0.781 ± 0.014 | 0.699 ± 0.025 | 0.905 ± 0.010 | 0.781 ± 0.012 |
| log_kurtosis | 0.763 ± 0.012 | 0.735 ± 0.036 | 0.639 ± 0.030 | 0.633 ± 0.015 | 0.889 ± 0.036 | 0.707 ± 0.017 |
| ow_strain_frac | 0.637 ± 0.025 | 0.559 ± 0.019 | 0.552 ± 0.020 | 0.455 ± 0.019 | 0.751 ± 0.025 | 0.584 ± 0.039 |
| time (plain R²) | 0.220 ± 0.003 | 0.271 ± 0.003 | 0.332 ± 0.001 | 0.241 ± 0.009 | 0.319 ± 0.038 | 0.313 ± 0.014 |
| log_nu pair (S2) | 0.601 ± 0.017 | 0.441 ± 0.031 | 0.778 ± 0.005 | 0.275 ± 0.039 | 0.680 ± 0.027 | 0.524 ± 0.021 |
| log_nu single | 0.090 ± 0.017 | 0.138 ± 0.012 | 0.219 ± 0.025 | 0.101 ± 0.023 | 0.254 ± 0.018 | 0.154 ± 0.025 |
| slow |r| log_E | 0.942 ± 0.007 | 0.785 ± 0.010 | 0.888 ± 0.005 | 0.680 ± 0.008 | 0.682 ± 0.110 | 0.668 ± 0.153 |
| slow |r| log_Z | 0.229 ± 0.006 | 0.436 ± 0.035 | 0.886 ± 0.046 | 0.499 ± 0.031 | 0.511 ± 0.027 | 0.422 ± 0.099 |
| slow |r| log_C4 | 0.565 ± 0.003 | 0.257 ± 0.067 | 0.569 ± 0.019 | 0.279 ± 0.083 | 0.561 ± 0.175 | 0.385 ± 0.019 |
| slow |r| kurtosis (traj mean) | 0.414 ± 0.006 | 0.397 ± 0.074 | 0.371 ± 0.005 | 0.367 ± 0.023 | 0.426 ± 0.063 | 0.380 ± 0.140 |
| slow |r| OW fraction (traj mean) | 0.701 ± 0.006 | 0.624 ± 0.092 | 0.641 ± 0.007 | 0.556 ± 0.070 | 0.513 ± 0.023 | 0.529 ± 0.189 |

- **jepa_spatial vs control (val preview)**: S1 fails (2/5 count; OW fraction does not count), S2 fails (pair R² 0.441 vs 0.601), S3 fails (Z |r| 0.436 vs 0.229) → not supported
- **pixel vs control (val preview)**: S1 fails (2/5 count; OW fraction does not count), S2 fails (pair R² 0.778 vs 0.601), S3 holds (Z |r| 0.886 vs 0.229) → not supported
- **jepa_pooled vs control (val preview)**: S1 fails (0/5 count; OW fraction does not count), S2 fails (pair R² 0.275 vs 0.601), S3 fails (Z |r| 0.499 vs 0.229) → not supported
- **static_vicreg vs control (val preview)**: S1 fails (3/5 count; OW fraction does not count), S2 fails (pair R² 0.680 vs 0.601), S3 fails (Z |r| 0.511 vs 0.229) → not supported
- **nonphys_jepa_spatial vs control (val preview)**: S1 fails (3/5 count; OW fraction does not count), S2 fails (pair R² 0.524 vs 0.601), S3 fails (Z |r| 0.422 vs 0.229) → not supported
