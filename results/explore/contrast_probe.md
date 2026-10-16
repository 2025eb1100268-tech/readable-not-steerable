# [EXPLORE] Contrast probe (POST HOC): does the decoder read enstrophy from the map's spatial variation?

PREDICTIONS entry 7, designed after the imagination probe. Per channel, μ = spatial mean of the 8×8 map. contrast: m → μ + (1+ε)(m − μ); mean: m → (1+ε)μ + (m − μ); global: m → (1+ε)m; applied to every context map. Slopes of the rendered field's log Z, log E, log P per unit ε (512 explore inviscid anchors).

| run | renders | intervention | d log Z / dε | d log E / dε | d log P / dε | log Z / log E slope ratio |
|---|---|---|---|---|---|---|
| s2_imaginer_k4_lr0.001_s0 | velocity (t+k) | contrast | +2.014 | +2.007 | +2.089 | 1.00 |
| s2_imaginer_k4_lr0.001_s0 | velocity (t+k) | mean | +0.302 | +0.218 | +0.448 | 1.39 |
| s2_imaginer_k4_lr0.001_s0 | velocity (t+k) | global | +2.325 | +2.232 | +2.543 | 1.04 |
| s2_imaginer_k4_lr0.001_s1 | velocity (t+k) | contrast | +1.865 | +1.858 | +1.932 | 1.00 |
| s2_imaginer_k4_lr0.001_s1 | velocity (t+k) | mean | +0.105 | +0.059 | +0.179 | 1.76 |
| s2_imaginer_k4_lr0.001_s1 | velocity (t+k) | global | +1.981 | +1.928 | +2.134 | 1.03 |
| s2_imaginer_k4_lr0.001_s2 | velocity (t+k) | contrast | +1.917 | +1.891 | +2.035 | 1.01 |
| s2_imaginer_k4_lr0.001_s2 | velocity (t+k) | mean | +0.084 | +0.054 | +0.134 | 1.55 |
| s2_imaginer_k4_lr0.001_s2 | velocity (t+k) | global | +2.006 | +1.949 | +2.180 | 1.03 |
| s2_ae_velocity_k0_lr0.001_s0 | velocity | contrast | +2.064 | +2.060 | +2.102 | 1.00 |
| s2_ae_velocity_k0_lr0.001_s0 | velocity | mean | +0.088 | +0.077 | +0.112 | 1.15 |
| s2_ae_velocity_k0_lr0.001_s0 | velocity | global | +2.155 | +2.138 | +2.222 | 1.01 |
| s2_ae_velocity_k0_lr0.001_s1 | velocity | contrast | +2.113 | +2.110 | +2.141 | 1.00 |
| s2_ae_velocity_k0_lr0.001_s1 | velocity | mean | +0.068 | +0.050 | +0.106 | 1.37 |
| s2_ae_velocity_k0_lr0.001_s1 | velocity | global | +2.183 | +2.161 | +2.250 | 1.01 |
| s2_ae_velocity_k0_lr0.001_s2 | velocity | contrast | +2.073 | +2.072 | +2.103 | 1.00 |
| s2_ae_velocity_k0_lr0.001_s2 | velocity | mean | +0.101 | +0.088 | +0.127 | 1.14 |
| s2_ae_velocity_k0_lr0.001_s2 | velocity | global | +2.177 | +2.161 | +2.237 | 1.01 |
| s2_ae_vorticity_k0_lr0.001_s0 | vorticity | contrast | +2.169 | +2.160 | +2.209 | 1.00 |
| s2_ae_vorticity_k0_lr0.001_s0 | vorticity | mean | +0.088 | +0.061 | +0.146 | 1.45 |
| s2_ae_vorticity_k0_lr0.001_s0 | vorticity | global | +2.259 | +2.221 | +2.356 | 1.02 |
| s2_ae_vorticity_k0_lr0.001_s1 | vorticity | contrast | +2.169 | +2.163 | +2.200 | 1.00 |
| s2_ae_vorticity_k0_lr0.001_s1 | vorticity | mean | +0.105 | +0.071 | +0.176 | 1.48 |
| s2_ae_vorticity_k0_lr0.001_s1 | vorticity | global | +2.275 | +2.234 | +2.376 | 1.02 |
| s2_ae_vorticity_k0_lr0.001_s2 | vorticity | contrast | +2.165 | +2.157 | +2.199 | 1.00 |
| s2_ae_vorticity_k0_lr0.001_s2 | vorticity | mean | +0.110 | +0.069 | +0.190 | 1.58 |
| s2_ae_vorticity_k0_lr0.001_s2 | vorticity | global | +2.276 | +2.227 | +2.388 | 1.02 |

Checks (P7.1 on the imaginers' contrast row, P7.2 on their mean row, P7.3 = both for the vorticity autoencoder):

- s2_imaginer_k4_lr0.001_s0: contrast: |dlogZ| >= 1.5 |dlogE|: no; mean: |dlogE| > |dlogZ|: no
- s2_imaginer_k4_lr0.001_s1: contrast: |dlogZ| >= 1.5 |dlogE|: no; mean: |dlogE| > |dlogZ|: no
- s2_imaginer_k4_lr0.001_s2: contrast: |dlogZ| >= 1.5 |dlogE|: no; mean: |dlogE| > |dlogZ|: no
- s2_ae_velocity_k0_lr0.001_s0: contrast: |dlogZ| >= 1.5 |dlogE|: no; mean: |dlogE| > |dlogZ|: no
- s2_ae_velocity_k0_lr0.001_s1: contrast: |dlogZ| >= 1.5 |dlogE|: no; mean: |dlogE| > |dlogZ|: no
- s2_ae_velocity_k0_lr0.001_s2: contrast: |dlogZ| >= 1.5 |dlogE|: no; mean: |dlogE| > |dlogZ|: no
- s2_ae_vorticity_k0_lr0.001_s0: contrast: |dlogZ| >= 1.5 |dlogE|: no; mean: |dlogE| > |dlogZ|: no
- s2_ae_vorticity_k0_lr0.001_s1: contrast: |dlogZ| >= 1.5 |dlogE|: no; mean: |dlogE| > |dlogZ|: no
- s2_ae_vorticity_k0_lr0.001_s2: contrast: |dlogZ| >= 1.5 |dlogE|: no; mean: |dlogE| > |dlogZ|: no
