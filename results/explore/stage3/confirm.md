# Stage 3 confirmation (PREREGISTRATION_v2.md)

**Verdict: partially supported** (H1 holds; H2 fails).

Scored on `confirm` inviscid trajectories; slow directions fitted on v1 train+val inviscid runs; pooled latent. E1 = seed-mean best-of-top-3 |r| with log Z₀. CIs: 1,000 paired trajectory-bootstrap resamples.

| group | seeds used | E1 [95% CI] | E4 a/b | log Z R² | log P R² |
|---|---|---|---|---|---|
| untrained (seeds 3-5) | 3 | 0.380 [0.30, 0.46] | -6878.39 | 0.738 | 0.410 |
| AE velocity (fresh) | 3 | 0.860 [0.84, 0.88] | 0.83 | 0.862 | 0.700 |
| AE velocity + VICReg (fresh) | 3 | 0.522 [0.47, 0.57] | 0.90 | 0.933 | 0.799 |
| AE vorticity (fresh) | 3 | 0.799 [0.78, 0.82] | 1.38 | 0.806 | 0.694 |
| AE vorticity + VICReg (fresh) | 3 | 0.650 [0.61, 0.69] | 1.07 | 0.955 | 0.912 |
| untrained | 3 | 0.376 [0.29, 0.45] | 13.98 | 0.737 | 0.407 |
| AE velocity | 3 | 0.813 [0.79, 0.83] | 0.81 | 0.855 | 0.698 |
| AE velocity + VICReg | 3 | 0.468 [0.41, 0.53] | 1.19 | 0.934 | 0.804 |
| AE vorticity | 3 | 0.813 [0.80, 0.83] | 1.00 | 0.799 | 0.698 |
| AE vorticity + VICReg | 3 | 0.551 [0.50, 0.61] | 1.19 | 0.959 | 0.918 |
| AE streamfunction | 3 | 0.558 [0.52, 0.60] | 3.13 | 0.861 | 0.657 |
| AE streamfunction + VICReg | 3 | 0.390 [0.34, 0.45] | 1.13 | 0.917 | 0.719 |
| imaginer k=4 | 3 | 0.815 [0.79, 0.84] | 0.87 | 0.941 | 0.830 |
| imaginer k=4 + VICReg | 3 | 0.500 [0.45, 0.55] | 0.71 | 0.937 | 0.782 |
| summarizer k=4 | 3 | 0.332 [0.29, 0.40] | 0.88 | 0.907 | 0.725 |

**H1 (primary)**: E1 ≥ 0.7 and ≥ control + 0.25, CI > 0, for every group:

- AE velocity (fresh) vs untrained (seeds 3-5): E1 0.860, gap +0.480 [+0.390, +0.565] → pass
- AE vorticity (fresh) vs untrained (seeds 3-5): E1 0.799, gap +0.418 [+0.360, +0.486] → pass

**H2 (primary)**: plain − VICReg ≥ 0.2, CI > 0, for every pair:

- AE velocity (fresh) − AE velocity + VICReg (fresh): +0.338 [+0.289, +0.390] → pass
- AE vorticity (fresh) − AE vorticity + VICReg (fresh): +0.149 [+0.105, +0.193] → fail

**H3 (secondary)**: E4 a/b AE vorticity 1.00 vs AE streamfunction 3.13 → pass

**H4 (secondary)**: plain imaginer − imaginer + VICReg ≥ 0.2, CI > 0:

- imaginer k=4 − imaginer k=4 + VICReg: +0.315 [+0.272, +0.358] → pass

Reported (no verdict): the H1 and H2 rules applied to the Stage 2 seeds 0–2:

- AE velocity vs untrained: E1 0.813, gap +0.437 [+0.345, +0.530] → pass
- AE vorticity vs untrained: E1 0.813, gap +0.437 [+0.354, +0.517] → pass
- AE velocity − AE velocity + VICReg: +0.345 [+0.285, +0.412] → pass
- AE vorticity − AE vorticity + VICReg: +0.262 [+0.207, +0.313] → pass

Reported: imaginer k=4 − summarizer k=4 E1 gap +0.483 [+0.42, +0.52].

Excluded runs: none.
