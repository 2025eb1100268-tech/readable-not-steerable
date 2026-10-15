# [EXPLORE] Stage 1: does the pixel model's enstrophy lead survive?

Exploratory, not a finding. Gate and definitions committed beforehand (results/explore/PREDICTIONS.md, entries 1–2). Slow directions fitted on v1 train+val inviscid runs as pre-registered; scored on explore (378 inviscid runs) and matched-energy (300 inviscid runs, all at E₀ = 0.2893). Wording rule: these statistics say which conserved quantities are prominent in a latent, not that a model discovered conservation.

## Kill gate (pooled latent, as committed)

| condition | pixel | untrained | gap [95% CI] | rule | outcome |
|---|---|---|---|---|---|
| (a) | 0.829 | 0.418 | +0.411 [+0.326, +0.514] | gap ≥ 0.1 and CI > 0 (explore) | pass |
| (b) | 0.979 | 0.177 | +0.802 [+0.718, +0.827] | pixel ≥ 0.5, gap > 0, CI > 0 (matched-energy) | pass |
| (b_first16_frames) | 0.976 | 0.170 | +0.806 [+0.720, +0.826] | same as (b), frames 0–15 only (unscored robustness check) | pass |

**Gate: the lead survives; Stage 2 may start (with your approval).**

Context (unscored): best-of-top-3 enstrophy |r|, other pairs.

| set | comparison | first | second | gap [95% CI] |
|---|---|---|---|---|
| explore | jepa_spatial vs untrained | 0.353 | 0.418 | -0.065 [-0.18, +0.08] |
| explore | pixel vs jepa_spatial | 0.829 | 0.353 | +0.476 [+0.42, +0.53] |
| matched | jepa_spatial vs untrained | 0.895 | 0.177 | +0.719 [+0.64, +0.74] |
| matched | pixel vs jepa_spatial | 0.979 | 0.895 | +0.083 [+0.07, +0.10] |

## Every direction, every seed (pooled latent)

Directions 1–3, slowest first. |r| between the direction's trajectory-mean projection and each quantity.

| set | model | seed | log Z₀ | log E₀ | partial r log Z₀ \| log E₀, family | traj-mean log P (not conserved) | traj-mean log kurtosis | traj-mean log max\|ω\| | held-out λ |
|---|---|---|---|---|---|---|---|---|---|
| explore | untrained | 0 | 0.42 / 0.03 / 0.18 | 0.94 / 0.16 / 0.03 | 0.31 / 0.19 / 0.29 | 0.37 / 0.22 / 0.25 | 0.55 / 0.10 / 0.03 | 0.59 / 0.01 / 0.14 | 0.00 / 0.01 / 0.02 |
| explore | untrained | 1 | 0.42 / 0.08 / 0.06 | 0.94 / 0.10 / 0.04 | 0.28 / 0.32 / 0.07 | 0.38 / 0.11 / 0.07 | 0.55 / 0.29 / 0.09 | 0.58 / 0.23 / 0.03 | 0.00 / 0.02 / 0.03 |
| explore | untrained | 2 | 0.41 / 0.04 / 0.07 | 0.93 / 0.14 / 0.14 | 0.31 / 0.34 / 0.09 | 0.37 / 0.25 / 0.08 | 0.56 / 0.08 / 0.04 | 0.59 / 0.15 / 0.01 | 0.00 / 0.01 / 0.01 |
| explore | pixel | 0 | 0.44 / 0.86 / 0.13 | 0.88 / 0.06 / 0.21 | 0.31 / 0.94 / 0.12 | 0.30 / 0.85 / 0.12 | 0.51 / 0.30 / 0.07 | 0.58 / 0.40 / 0.04 | 0.00 / 0.01 / 0.01 |
| explore | pixel | 1 | 0.45 / 0.87 / 0.14 | 0.90 / 0.04 / 0.16 | 0.31 / 0.93 / 0.14 | 0.32 / 0.83 / 0.09 | 0.51 / 0.30 / 0.12 | 0.59 / 0.41 / 0.06 | 0.00 / 0.01 / 0.01 |
| explore | pixel | 2 | 0.44 / 0.42 / 0.75 | 0.89 / 0.17 / 0.09 | 0.31 / 0.58 / 0.66 | 0.31 / 0.59 / 0.58 | 0.51 / 0.10 / 0.30 | 0.58 / 0.29 / 0.28 | 0.00 / 0.00 / 0.01 |
| explore | jepa_spatial | 0 | 0.37 / 0.06 / 0.06 | 0.03 / 0.72 / 0.33 | 0.13 / 0.62 / 0.14 | 0.24 / 0.71 / 0.17 | 0.24 / 0.45 / 0.41 | 0.05 / 0.17 / 0.32 | 0.01 / 0.01 / 0.01 |
| explore | jepa_spatial | 1 | 0.33 / 0.27 / 0.05 | 0.40 / 0.69 / 0.20 | 0.33 / 0.33 / 0.14 | 0.58 / 0.41 / 0.14 | 0.55 / 0.29 / 0.36 | 0.15 / 0.27 / 0.24 | 0.01 / 0.01 / 0.02 |
| explore | jepa_spatial | 2 | 0.05 / 0.36 / 0.10 | 0.81 / 0.08 / 0.29 | 0.51 / 0.10 / 0.17 | 0.65 / 0.20 / 0.08 | 0.52 / 0.20 / 0.25 | 0.30 / 0.07 / 0.25 | 0.00 / 0.01 / 0.01 |
| matched | untrained | 0 | 0.21 / 0.15 / 0.22 | — (E₀ fixed) | 0.21 / 0.15 / 0.22 | 0.20 / 0.14 / 0.22 | 0.30 / 0.02 / 0.04 | 0.10 / 0.13 / 0.18 | 0.08 / 0.20 / 0.15 |
| matched | untrained | 1 | 0.00 / 0.05 / 0.02 | — (E₀ fixed) | 0.00 / 0.05 / 0.02 | 0.01 / 0.05 / 0.02 | 0.13 / 0.36 / 0.12 | 0.02 / 0.06 / 0.07 | 0.10 / 0.08 / 0.19 |
| matched | untrained | 2 | 0.25 / 0.19 / 0.18 | — (E₀ fixed) | 0.25 / 0.19 / 0.18 | 0.25 / 0.19 / 0.17 | 0.29 / 0.14 / 0.08 | 0.15 / 0.22 / 0.18 | 0.08 / 0.12 / 0.22 |
| matched | pixel | 0 | 0.68 / 0.98 / 0.96 | — (E₀ fixed) | 0.68 / 0.98 / 0.96 | 0.68 / 0.97 / 0.95 | 0.01 / 0.22 / 0.16 | 0.59 / 0.90 / 0.86 | 0.18 / 0.01 / 0.02 |
| matched | pixel | 1 | 0.40 / 0.98 / 0.92 | — (E₀ fixed) | 0.40 / 0.98 / 0.92 | 0.41 / 0.98 / 0.91 | 0.09 / 0.22 / 0.07 | 0.37 / 0.91 / 0.81 | 0.30 / 0.00 / 0.03 |
| matched | pixel | 2 | 0.75 / 0.97 / 0.98 | — (E₀ fixed) | 0.75 / 0.97 / 0.98 | 0.76 / 0.97 / 0.98 | 0.10 / 0.28 / 0.21 | 0.68 / 0.91 / 0.90 | 0.22 / 0.01 / 0.01 |
| matched | jepa_spatial | 0 | 0.90 / 0.90 / 0.06 | — (E₀ fixed) | 0.90 / 0.90 / 0.06 | 0.90 / 0.89 / 0.07 | 0.05 / 0.08 / 0.06 | 0.79 / 0.79 / 0.06 | 0.01 / 0.02 / 0.06 |
| matched | jepa_spatial | 1 | 0.92 / 0.51 / 0.69 | — (E₀ fixed) | 0.92 / 0.51 / 0.69 | 0.91 / 0.50 / 0.68 | 0.08 / 0.03 / 0.04 | 0.81 / 0.44 / 0.58 | 0.01 / 0.07 / 0.04 |
| matched | jepa_spatial | 2 | 0.86 / 0.87 / 0.51 | — (E₀ fixed) | 0.86 / 0.87 / 0.51 | 0.85 / 0.86 / 0.51 | 0.07 / 0.02 / 0.13 | 0.77 / 0.75 / 0.41 | 0.03 / 0.02 / 0.09 |

## Every direction, every seed (map_meanstd latent) — secondary, unscored

Directions 1–3, slowest first. |r| between the direction's trajectory-mean projection and each quantity.

| set | model | seed | log Z₀ | log E₀ | partial r log Z₀ \| log E₀, family | traj-mean log P (not conserved) | traj-mean log kurtosis | traj-mean log max\|ω\| | held-out λ |
|---|---|---|---|---|---|---|---|---|---|
| explore | untrained | 0 | 0.41 / 0.18 / 0.17 | 0.95 / 0.08 / 0.08 | 0.31 / 0.26 / 0.12 | 0.39 / 0.31 / 0.11 | 0.58 / 0.21 / 0.11 | 0.60 / 0.02 / 0.03 | 0.00 / 0.01 / 0.01 |
| explore | untrained | 1 | 0.41 / 0.22 / 0.18 | 0.96 / 0.01 / 0.08 | 0.27 / 0.32 / 0.11 | 0.40 / 0.30 / 0.06 | 0.57 / 0.15 / 0.40 | 0.59 / 0.08 / 0.23 | 0.00 / 0.01 / 0.02 |
| explore | untrained | 2 | 0.40 / 0.22 / 0.11 | 0.95 / 0.04 / 0.12 | 0.31 / 0.32 / 0.05 | 0.39 / 0.33 / 0.00 | 0.59 / 0.14 / 0.17 | 0.60 / 0.10 / 0.09 | 0.00 / 0.01 / 0.01 |
| explore | pixel | 0 | 0.41 / 0.91 / 0.06 | 0.90 / 0.00 / 0.18 | 0.28 / 0.98 / 0.19 | 0.34 / 0.83 / 0.17 | 0.54 / 0.29 / 0.07 | 0.58 / 0.44 / 0.08 | 0.00 / 0.00 / 0.01 |
| explore | pixel | 1 | 0.43 / 0.89 / 0.08 | 0.91 / 0.01 / 0.15 | 0.31 / 0.96 / 0.17 | 0.34 / 0.82 / 0.14 | 0.54 / 0.27 / 0.06 | 0.59 / 0.44 / 0.06 | 0.00 / 0.00 / 0.01 |
| explore | pixel | 2 | 0.45 / 0.90 / 0.02 | 0.91 / 0.03 / 0.15 | 0.35 / 0.97 / 0.27 | 0.32 / 0.84 / 0.22 | 0.52 / 0.31 / 0.09 | 0.60 / 0.41 / 0.15 | 0.00 / 0.00 / 0.01 |
| explore | jepa_spatial | 0 | 0.11 / 0.32 / 0.16 | 0.68 / 0.43 / 0.30 | 0.41 / 0.12 / 0.20 | 0.66 / 0.15 / 0.08 | 0.59 / 0.04 / 0.27 | 0.27 / 0.15 / 0.27 | 0.01 / 0.01 / 0.01 |
| explore | jepa_spatial | 1 | 0.21 / 0.27 / 0.09 | 0.83 / 0.36 / 0.13 | 0.33 / 0.15 / 0.28 | 0.55 / 0.46 / 0.06 | 0.56 / 0.51 / 0.23 | 0.43 / 0.18 / 0.25 | 0.01 / 0.01 / 0.01 |
| explore | jepa_spatial | 2 | 0.33 / 0.12 / 0.16 | 0.14 / 0.86 / 0.25 | 0.15 / 0.46 / 0.23 | 0.35 / 0.63 / 0.01 | 0.33 / 0.63 / 0.24 | 0.02 / 0.42 / 0.30 | 0.01 / 0.01 / 0.02 |
| matched | untrained | 0 | 0.24 / 0.31 / 0.17 | — (E₀ fixed) | 0.24 / 0.31 / 0.17 | 0.23 / 0.30 / 0.16 | 0.34 / 0.21 / 0.22 | 0.11 / 0.20 / 0.08 | 0.06 / 0.10 / 0.09 |
| matched | untrained | 1 | 0.10 / 0.48 / 0.16 | — (E₀ fixed) | 0.10 / 0.48 / 0.16 | 0.10 / 0.47 / 0.16 | 0.14 / 0.10 / 0.32 | 0.07 / 0.45 / 0.04 | 0.08 / 0.08 / 0.09 |
| matched | untrained | 2 | 0.26 / 0.40 / 0.03 | — (E₀ fixed) | 0.26 / 0.40 / 0.03 | 0.26 / 0.40 / 0.03 | 0.34 / 0.17 / 0.01 | 0.13 / 0.30 / 0.03 | 0.05 / 0.07 / 0.22 |
| matched | pixel | 0 | 0.45 / 0.99 / 0.82 | — (E₀ fixed) | 0.45 / 0.99 / 0.82 | 0.44 / 0.99 / 0.82 | 0.01 / 0.24 / 0.11 | 0.38 / 0.92 / 0.74 | 0.27 / 0.00 / 0.04 |
| matched | pixel | 1 | 0.47 / 0.99 / 0.89 | — (E₀ fixed) | 0.47 / 0.99 / 0.89 | 0.48 / 0.99 / 0.89 | 0.13 / 0.23 / 0.06 | 0.45 / 0.92 / 0.79 | 0.25 / 0.00 / 0.05 |
| matched | pixel | 2 | 0.95 / 0.99 / 0.69 | — (E₀ fixed) | 0.95 / 0.99 / 0.69 | 0.95 / 0.99 / 0.69 | 0.25 / 0.22 / 0.05 | 0.89 / 0.92 / 0.58 | 0.04 / 0.00 / 0.06 |
| matched | jepa_spatial | 0 | 0.90 / 0.73 / 0.46 | — (E₀ fixed) | 0.90 / 0.73 / 0.46 | 0.89 / 0.73 / 0.44 | 0.06 / 0.04 / 0.07 | 0.79 / 0.63 / 0.42 | 0.02 / 0.04 / 0.08 |
| matched | jepa_spatial | 1 | 0.53 / 0.93 / 0.88 | — (E₀ fixed) | 0.53 / 0.93 / 0.88 | 0.52 / 0.92 / 0.88 | 0.03 / 0.10 / 0.06 | 0.48 / 0.83 / 0.78 | 0.10 / 0.01 / 0.02 |
| matched | jepa_spatial | 2 | 0.88 / 0.78 / 0.57 | — (E₀ fixed) | 0.88 / 0.78 / 0.57 | 0.87 / 0.76 / 0.56 | 0.01 / 0.12 / 0.09 | 0.76 / 0.71 / 0.47 | 0.02 / 0.05 / 0.10 |

Gate statistics with this readout (unscored): (a) gap +0.494 [+0.412, +0.585]; (b) pixel 0.992 vs 0.394, gap +0.598 [+0.548, +0.658].

## 1c. What the enstrophy-tracking direction is (pooled latent)

Per seed: the top-3 direction with the largest |r| with log Z₀, regressed on standardized descriptors at t = 0 (log E₀ is constant on matched-energy and dropped there). Note log Z₀ = log E₀ + log(Z₀/E₀) exactly.

| set | model | seed | dir | \|r\| log Z₀ | R² | β log E₀ | β log(Z₀/E₀) | β high-k slope (k 8–20) | β log(P₀/Z₀) | β log kurtosis₀ | r log E₀ | r log(Z₀/E₀) | r high-k slope (k 8–20) | r log(P₀/Z₀) | r log kurtosis₀ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| explore | untrained | 0 | 1 | 0.42 | 0.90 | +1.10 | -0.04 | +0.14 | +0.16 | -0.11 | +0.94 | -0.83 | +0.25 | -0.79 | +0.55 |
| explore | untrained | 1 | 1 | 0.42 | 0.91 | +1.09 | -0.09 | +0.12 | +0.19 | -0.13 | +0.94 | -0.84 | +0.24 | -0.80 | +0.55 |
| explore | untrained | 2 | 1 | 0.41 | 0.90 | +1.08 | -0.03 | +0.15 | +0.15 | -0.11 | +0.93 | -0.83 | +0.26 | -0.78 | +0.56 |
| explore | pixel | 0 | 2 | 0.86 | 0.93 | +2.15 | +2.19 | +0.05 | +0.19 | -0.05 | -0.06 | +0.43 | -0.01 | +0.43 | -0.30 |
| explore | pixel | 1 | 2 | 0.87 | 0.92 | +2.16 | +2.26 | +0.03 | +0.12 | -0.04 | -0.04 | +0.42 | -0.04 | +0.41 | -0.30 |
| explore | pixel | 2 | 3 | 0.75 | 0.65 | +1.87 | +1.60 | -0.13 | +0.20 | -0.18 | +0.09 | +0.23 | -0.20 | +0.18 | -0.30 |
| explore | jepa_spatial | 0 | 1 | 0.37 | 0.23 | +0.90 | +1.35 | -0.14 | -0.50 | -0.09 | +0.03 | +0.12 | -0.29 | +0.05 | -0.24 |
| explore | jepa_spatial | 1 | 1 | 0.33 | 0.48 | +0.77 | +1.53 | -0.04 | -0.41 | -0.20 | -0.40 | +0.57 | -0.28 | +0.50 | -0.55 |
| explore | jepa_spatial | 2 | 2 | 0.36 | 0.22 | +0.89 | +1.23 | -0.15 | -0.44 | -0.09 | +0.08 | +0.07 | -0.28 | +0.00 | -0.20 |
| matched | untrained | 0 | 3 | 0.22 | 0.06 | — | +0.37 | -0.10 | -0.03 | -0.09 | — | +0.22 | +0.19 | +0.22 | -0.05 |
| matched | untrained | 1 | 2 | 0.05 | 0.16 | — | +0.93 | +0.42 | -1.20 | -0.39 | — | +0.05 | +0.07 | +0.04 | -0.36 |
| matched | untrained | 2 | 1 | 0.25 | 0.20 | — | +0.52 | -0.13 | -0.07 | -0.37 | — | +0.25 | +0.21 | +0.24 | -0.30 |
| matched | pixel | 0 | 2 | 0.98 | 0.96 | — | +0.47 | -0.35 | +0.84 | -0.01 | — | +0.98 | +0.87 | +0.97 | +0.18 |
| matched | pixel | 1 | 2 | 0.98 | 0.97 | — | +0.63 | -0.30 | +0.64 | -0.01 | — | +0.98 | +0.88 | +0.97 | +0.18 |
| matched | pixel | 2 | 3 | 0.98 | 0.97 | — | +0.59 | -0.30 | +0.68 | -0.03 | — | +0.98 | +0.88 | +0.97 | +0.16 |
| matched | jepa_spatial | 0 | 1 | 0.90 | 0.85 | — | +1.22 | -0.05 | -0.24 | -0.17 | — | +0.90 | +0.81 | +0.89 | +0.01 |
| matched | jepa_spatial | 1 | 1 | 0.92 | 0.87 | — | +1.47 | +0.30 | -0.81 | -0.15 | — | +0.92 | +0.86 | +0.90 | +0.04 |
| matched | jepa_spatial | 2 | 2 | 0.87 | 0.79 | — | +1.18 | +0.07 | -0.34 | -0.19 | — | +0.87 | +0.79 | +0.85 | -0.02 |

corr(log E₀, log Z₀) over explore inviscid runs: 0.381.
