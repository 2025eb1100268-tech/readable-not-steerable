# [EXPLORE] Imagination probe: does the pixel predictor use its enstrophy direction?

Exploratory (PREDICTIONS entry 6, step 2). Shift both context maps along a slow direction's pooled-latent pattern by δ standard deviations of that slow coordinate, decode the frame at t + k, and measure the rendered field. Slopes are per unit δ (mean over 512 explore inviscid anchors). Random baseline: 20 random directions with the enstrophy pattern's norm.

| run | k | direction (slow dir #, \|r\|) | d log Z / dδ | d log E / dδ | d log P / dδ |
|---|---|---|---|---|---|
| s2_imaginer_k4_lr0.001_s0 | 4 | enstrophy direction (2, 0.70) | -0.0073 | -0.0076 | -0.0065 |
| s2_imaginer_k4_lr0.001_s0 | 4 | energy direction (1, 0.90) | +0.0155 | +0.0241 | -0.0019 |
| s2_imaginer_k4_lr0.001_s0 | 4 | random: median / 95th pct of \|slope\| | 0.0049 / 0.0167 | 0.0044 / 0.0146 | 0.0068 / 0.0222 |
| s2_imaginer_k4_lr0.001_s1 | 4 | enstrophy direction (2, 0.79) | +0.0256 | +0.0266 | +0.0247 |
| s2_imaginer_k4_lr0.001_s1 | 4 | energy direction (1, 0.88) | +0.0665 | +0.0793 | +0.0443 |
| s2_imaginer_k4_lr0.001_s1 | 4 | random: median / 95th pct of \|slope\| | 0.0068 / 0.0196 | 0.0058 / 0.0146 | 0.0099 / 0.0361 |
| s2_imaginer_k4_lr0.001_s2 | 4 | enstrophy direction (2, 0.90) | -0.0040 | -0.0019 | -0.0067 |
| s2_imaginer_k4_lr0.001_s2 | 4 | energy direction (1, 0.87) | +0.0247 | +0.0420 | -0.0064 |
| s2_imaginer_k4_lr0.001_s2 | 4 | random: median / 95th pct of \|slope\| | 0.0109 / 0.0237 | 0.0089 / 0.0263 | 0.0130 / 0.0380 |

P6.1 checks per run:

- s2_imaginer_k4_lr0.001_s0: enstrophy dir: |dlogZ| >= 1.5 |dlogE|: no; enstrophy dir: |dlogZ| > random p95: no; energy dir: |dlogE| > |dlogZ|: yes; holds: no
- s2_imaginer_k4_lr0.001_s1: enstrophy dir: |dlogZ| >= 1.5 |dlogE|: no; enstrophy dir: |dlogZ| > random p95: yes; energy dir: |dlogE| > |dlogZ|: yes; holds: no
- s2_imaginer_k4_lr0.001_s2: enstrophy dir: |dlogZ| >= 1.5 |dlogE|: yes; enstrophy dir: |dlogZ| > random p95: no; energy dir: |dlogE| > |dlogZ|: yes; holds: no
