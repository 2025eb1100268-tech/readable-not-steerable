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
