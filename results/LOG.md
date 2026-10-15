# Run log

Every run, its config and outcome, including failed runs. Commands are run from the repo root
with the project venv (`.venv\Scripts\python.exe`).

## Before the project: planning pilots (2026-10-13/14, not reproducible from this repo)

About 70 throwaway CPU runs made while reviewing the brief (toy solvers, untrained CNNs, ridge
probes; up to 768 trajectories, mostly 64², resolution checks at 128²/256²). They shaped the
pre-registration; none of their numbers is a result of this project. Main findings:

- Untrained encoders (Kaiming init, z-scored latents) reach time-partialled ridge R² of
  0.86–0.92 (log enstrophy), 0.79–0.85 (log palinstrophy), 0.72–0.80 (vortex count),
  0.23–0.70 (Okubo–Weiss fraction), 0.27–0.31 (log dissipation). PyTorch's default init gives
  baselines 0.1–0.4 lower and a nearly constant latent (std ≈ 0.2–0.4% of its mean).
- Slow-direction analysis on untrained encoders: held-out |r| with energy 0.98–1.00; with
  enstrophy 0.62–0.74 (Kaiming) / 0.72–0.90 (default init). Data had inviscid energy spread
  10–20× and ran into thermalization.
- Per-sample normalization (instance norm; group/layer norm with bias-free convs) erased
  energy (log-E R² 0.01–0.02).
- Strided CNN + global pooling is not shift-invariant: 4-px shifts move the pooled latent
  about as far as a different flow; 8-px shifts ~0.
- A flipped advection sign passed every solver test in the brief; only an analytic-RHS
  test, a vortex-pair rotation test and a stored-frame residual / ν-recovery test caught it.
- An idealized single-frame log-spectrum reads log ν at R² 0.77–0.81 (frame pair 0.86).
- Copying the current pooled latent (untrained features) scored normalized MSE ≈ 0.02 at
  k = 1; spatial map 0.11–0.20.
- R² against a narrow held-out band's own variance reached −4 to −29 for log ν.
- Raw palinstrophy / dissipation: 1% of frames held 65–99% of the sum of squares.

## 2026-10-14 (before 17:16) — environment
- `py -3.13 -m venv .venv`; `pip install torch --index-url https://download.pytorch.org/whl/cu128`;
  `pip install numpy scipy scikit-learn matplotlib pyyaml pytest`; pinned in `requirements.txt`
- outcome: torch 2.11.0+cu128, CUDA 12.8, NVIDIA GeForce RTX 3060 visible

## 2026-10-14 17:16 — Phase 0 hardware benchmark
- config: `configs/phase0_benchmark.yaml`
- command: `python -m common.benchmark --config configs/phase0_benchmark.yaml`
- git: `no-commit+dirty`
- device: NVIDIA GeForce RTX 3060
- wall-clock: 26.5 s
- outcome: ok
  - encoder fwd+bwd ms/frame (GPU fp32 / GPU bf16 / CPU 8 threads): 0.4214 / 0.2131 / 6.31
  - solver: cuda 64^2 B=256 float64: 15.0 ms/step; cuda 64^2 B=1024 float64: 55.6 ms/step; cuda 128^2 B=256 float64: 63.4 ms/step; cpu 64^2 B=256 float64: 103.9 ms/step
  - results: results/phase0/benchmark.json

## 2026-10-14 (17:16–17:20) — Phase 0 unit tests (FAILED)
- command: `python -m pytest`
- outcome: 15 passed, 1 failed (`test_time_stepping_is_fourth_order`). Cause: test-helper bug —
  with t_end = 0.5 and dt = 0.04 the helper rounded 12.5 steps to 12 and compared solutions at
  different times. Fixed in `solver/checks.py::convergence` (steps now land exactly on t_end).

## 2026-10-14 (17:16–17:20) — Phase 0 unit tests
- command: `python -m pytest`
- outcome: 16 passed in 8.7 s

## 2026-10-14 17:20 — Phase 0 solver validation
- config: `configs/phase0_solver_validation.yaml`
- command: `python -m solver.validate --config configs/phase0_solver_validation.yaml`
- git: `no-commit+dirty`
- device: cpu
- wall-clock: 16.6 s
- outcome: all checks pass
  - inviscid drift E / Z / int w^4 (T=8): 2.6e-06 / 4.0e-05 / 0.33
  - dE/dt vs -2 nu Z: 6.2e-07
  - convergence orders: 4.13, 4.04, 4.03
  - figure: results/solver_validation.png

## 2026-10-14 17:21 — Phase 0 solver validation (re-run after figure layout fix; identical numbers)
- config: `configs/phase0_solver_validation.yaml`
- command: `python -m solver.validate --config configs/phase0_solver_validation.yaml`
- git: `no-commit+dirty`
- device: cpu
- wall-clock: 16.9 s
- outcome: all checks pass
  - inviscid drift E / Z / int w^4 (T=8): 2.6e-06 / 4.0e-05 / 0.33
  - dE/dt vs -2 nu Z: 6.2e-07
  - convergence orders: 4.13, 4.04, 4.03
  - figure: results/solver_validation.png

## 2026-10-14 17:34 — Phase 0 pilot calibration
- config: `configs/phase0_pilot.yaml`
- command: `python -m solver.pilot --config configs/phase0_pilot.yaml`
- git: `no-commit+dirty`
- device: cuda
- wall-clock: 541.3 s
- outcome: ok (figure later redrawn from the saved JSON with `--plot-only` after a legend/tick layout fix)
  - windows (T_w: share >5% tail; untrained slow |r| logE/logZ): 0.32: 0.00, 0.92/0.31; 0.64: 0.00, 0.93/0.31; 0.96: 0.07, 0.93/0.31; 1.28: 0.20, 0.93/0.31; 1.92: 0.51, 0.93/0.31; 2.56: 0.70, 0.93/0.30; 3.2: 0.80, 0.93/0.30
  - resolution (nu: P err p95): 0.0005: 0.354; 0.001: 0.239; 0.0015: 0.167; 0.002: 0.117; 0.003: 0.060; 0.005: 0.020
  - results: results/phase0/pilot.json
  - figure: results/phase0_pilot.png

## 2026-10-14 (after the pilot) — Phase 0 unit tests
- command: `python -m pytest`
- outcome: 17 passed in 10.4 s (adds the 64↔128 resampling test)

## 2026-10-14 18:21 — Phase 1 data generation
- config: `configs/phase1_data.yaml`
- command: `python -m data.generate --config configs/phase1_data.yaml`
- git: `no-commit+dirty`
- device: cuda
- wall-clock: 307.6 s
- outcome: ok
  - frames: 108000
  - split counts (trajectories): {'train': 1295, 'val': 279, 'test_id': 276, 'test_ood': 150}
  - OOD Re band: 286–389
  - inviscid max drift E / Z / C4: 4.8e-09 / 1.1e-06 / 0.097
  - input scale (train velocity std): 0.9548
  - sha256 velocity.npy: ef3fbb8d3c5ac41b…

## 2026-10-14 18:26 — Phase 1 untrained baselines (train/val only) — superseded
- note: run with control seeds 100–102; superseded by the 18:39 run with seeds 0–2 (the control
  must be exactly the trained models' starting weights, which use seeds 0–2)
- config: `configs/phase1_untrained_baselines.yaml`
- command: `python -m probes.untrained_baselines --config configs/phase1_untrained_baselines.yaml`
- git: `no-commit+dirty`
- device: cuda
- wall-clock: 92.9 s
- outcome: ok
  - S1 partial R² (Z, P, count, OW, eps): 0.770, 0.495, 0.900, 0.851, 0.242
  - S2 log-nu pair R²: 0.467
  - S3 untrained max|r| E / Z: 0.938 / 0.264
  - results: results/phase1/untrained_baselines.json

## 2026-10-14 18:39 — Phase 1 untrained baselines (train/val only)
- config: `configs/phase1_untrained_baselines.yaml`
- command: `python -m probes.untrained_baselines --config configs/phase1_untrained_baselines.yaml`
- git: `no-commit+dirty`
- device: cuda
- wall-clock: 40.8 s
- outcome: ok
  - S1 partial R² (Z, P, count, OW, eps): 0.782, 0.511, 0.899, 0.844, 0.259
  - S2 log-nu pair R²: 0.486
  - S3 untrained max|r| E / Z: 0.938 / 0.233
  - results: results/phase1/untrained_baselines.json

## 2026-10-14 ~18:45 — Phase 1 ad-hoc check: initial-condition family (train/val only)
- command: scratch script (not in repo), control seed 0; run with the user's OK after an
  interrupted first attempt
- outcome: family alone (after time partialling) explains 0.11 (log Z), 0.10 (log P),
  0.79 (log1p vortex count), 0.76 (OW fraction), 0.05 (log eps) of val variance. Untrained
  OW-fraction R² is 0.72 within blob runs but 0.08 within spectral runs.

## 2026-10-14 ~18:50 — Phase 1 ad-hoc check: S1 baselines partialling time x family (train/val only)
- command: scratch script (not in repo), control seeds 0–2; frame-index one-hot crossed with family
- outcome: untrained val R² 0.762 (log Z), 0.443 (log P), 0.580 (log1p count), 0.606 (OW fraction),
  0.236 (log eps); bars for a trained model 0.881 / 0.643 / 0.780 / 0.803 / 0.436.
  If this definition is adopted it moves into `probes/untrained_baselines.py` and is re-run there.

## 2026-10-14 18:45 — Phase 1 untrained baselines (train/val only)
- config: `configs/phase1_untrained_baselines.yaml`
- command: `python -m probes.untrained_baselines --config configs/phase1_untrained_baselines.yaml`
- git: `no-commit+dirty`
- device: cuda
- wall-clock: 49.4 s
- outcome: ok
  - S1 partial R² (Z, P, count, OW, eps): 0.762, 0.443, 0.580, 0.606, 0.236
  - S2 log-nu pair R²: 0.486
  - S3 untrained max|r| E / Z: 0.938 / 0.233
  - results: results/phase1/untrained_baselines.json

## 2026-10-14 18:49 — Phase 2 training: smoke_jepa_spatial
- config: `configs/phase2/smoke_jepa_spatial.yaml`
- command: `python -m train.train --config configs/phase2/smoke_jepa_spatial.yaml`
- git: `f7237f2+dirty`
- device: NVIDIA GeForce RTX 3060
- wall-clock: 42.1 s
- outcome: collapsed (fails the pre-registered no-collapse rule)
  - final val loss / copy baseline: 14.2172
  - effective rank / latent std median: 2.3 / 0.150
  - steps: 244
  - peak GPU memory GB: 4.51
  - results: results/phase2/smoke_jepa_spatial.json

## 2026-10-14 18:50 — Phase 2 training: smoke_pixel
- config: `configs/phase2/smoke_pixel.yaml`
- command: `python -m train.train --config configs/phase2/smoke_pixel.yaml`
- git: `f7237f2+dirty`
- device: NVIDIA GeForce RTX 3060
- wall-clock: 44.4 s
- outcome: collapsed (fails the pre-registered no-collapse rule)
  - final val loss / copy baseline: 1.1566
  - effective rank / latent std median: 5.3 / 0.010
  - steps: 244
  - peak GPU memory GB: 4.51
  - results: results/phase2/smoke_pixel.json

## 2026-10-14 19:03 — Phase 2b non-physical dataset
- config: `configs/phase2b_nonphys_data.yaml`
- command: `python -m data.nonphysical --config configs/phase2b_nonphys_data.yaml`
- git: `1671301+dirty`
- device: cuda
- wall-clock: 271.7 s
- outcome: ok
  - enstrophy retained at last frame (median / p10): 0.62 / 0.18
  - median frame-to-frame change: 0.027 (physical data: 0.038)
  - sha256 velocity.npy: 0e1f6387e537bb96…

## 2026-10-14 19:20 — Phase 2 training: jepa_spatial_lr0.0003_s0
- config: `configs/phase2/jepa_spatial_lr0.0003_s0.yaml`
- command: `python -m train.train --config configs/phase2/jepa_spatial_lr0.0003_s0.yaml`
- git: `b18f7ea+dirty`
- device: NVIDIA GeForce RTX 3060
- wall-clock: 1619.2 s
- outcome: ok
  - final val loss / copy baseline: 1.1159
  - effective rank / latent std median: 42.5 / 0.999
  - steps: 9760
  - peak GPU memory GB: 4.51
  - results: results/phase2/jepa_spatial_lr0.0003_s0.json

## 2026-10-14 19:45 — Phase 2 training: jepa_spatial_lr0.001_s0
- config: `configs/phase2/jepa_spatial_lr0.001_s0.yaml`
- command: `python -m train.train --config configs/phase2/jepa_spatial_lr0.001_s0.yaml`
- git: `b18f7ea+dirty`
- device: NVIDIA GeForce RTX 3060
- wall-clock: 1472.4 s
- outcome: ok
  - final val loss / copy baseline: 0.5901
  - effective rank / latent std median: 57.2 / 1.019
  - steps: 9760
  - peak GPU memory GB: 4.51
  - results: results/phase2/jepa_spatial_lr0.001_s0.json

## 2026-10-14 20:12 — Phase 2 training: pixel_lr0.0003_s0
- config: `configs/phase2/pixel_lr0.0003_s0.yaml`
- command: `python -m train.train --config configs/phase2/pixel_lr0.0003_s0.yaml`
- git: `b18f7ea+dirty`
- device: NVIDIA GeForce RTX 3060
- wall-clock: 1622.8 s
- outcome: ok
  - final val loss / copy baseline: 0.0800
  - effective rank / latent std median: 4.3 / 0.006
  - steps: 9760
  - peak GPU memory GB: 4.51
  - results: results/phase2/pixel_lr0.0003_s0.json

## 2026-10-14 20:39 — Phase 2 training: pixel_lr0.001_s0
- config: `configs/phase2/pixel_lr0.001_s0.yaml`
- command: `python -m train.train --config configs/phase2/pixel_lr0.001_s0.yaml`
- git: `b18f7ea+dirty`
- device: NVIDIA GeForce RTX 3060
- wall-clock: 1654.0 s
- outcome: ok
  - final val loss / copy baseline: 0.0348
  - effective rank / latent std median: 4.1 / 0.006
  - steps: 9760
  - peak GPU memory GB: 4.51
  - results: results/phase2/pixel_lr0.001_s0.json

## 2026-10-14 20:39 — Phase 2 selection: jepa_spatial
- config: `-`
- command: `python -m train.select --model jepa_spatial --runs jepa_spatial_lr0.0003_s0 jepa_spatial_lr0.001_s0`
- git: `b18f7ea+dirty`
- outcome: selected jepa_spatial_lr0.001_s0 (lr 0.001)
  - candidates (run: metric, collapsed): jepa_spatial_lr0.0003_s0: 1.1159, False; jepa_spatial_lr0.001_s0: 0.5901, False
  - configs written: jepa_spatial_lr0.001_s1.yaml, jepa_spatial_lr0.001_s2.yaml

## 2026-10-14 20:39 — Phase 2 selection: pixel
- config: `-`
- command: `python -m train.select --model pixel --runs pixel_lr0.0003_s0 pixel_lr0.001_s0`
- git: `b18f7ea+dirty`
- outcome: selected pixel_lr0.001_s0 (lr 0.001)
  - candidates (run: metric, collapsed): pixel_lr0.0003_s0: 0.0800, False; pixel_lr0.001_s0: 0.0348, False
  - configs written: pixel_lr0.001_s1.yaml, pixel_lr0.001_s2.yaml

## 2026-10-14 21:04 — Phase 2 training: jepa_spatial_lr0.001_s1
- config: `configs/phase2/jepa_spatial_lr0.001_s1.yaml`
- command: `python -m train.train --config configs/phase2/jepa_spatial_lr0.001_s1.yaml`
- git: `b18f7ea+dirty`
- device: NVIDIA GeForce RTX 3060
- wall-clock: 1496.2 s
- outcome: ok
  - final val loss / copy baseline: 0.6175
  - effective rank / latent std median: 58.0 / 1.021
  - steps: 9760
  - peak GPU memory GB: 4.51
  - results: results/phase2/jepa_spatial_lr0.001_s1.json

## 2026-10-14 21:31 — Phase 2 training: jepa_spatial_lr0.001_s2
- config: `configs/phase2/jepa_spatial_lr0.001_s2.yaml`
- command: `python -m train.train --config configs/phase2/jepa_spatial_lr0.001_s2.yaml`
- git: `b18f7ea+dirty`
- device: NVIDIA GeForce RTX 3060
- wall-clock: 1594.4 s
- outcome: ok
  - final val loss / copy baseline: 0.5853
  - effective rank / latent std median: 59.4 / 1.015
  - steps: 9760
  - peak GPU memory GB: 4.51
  - results: results/phase2/jepa_spatial_lr0.001_s2.json

## 2026-10-14 21:59 — Phase 2 training: pixel_lr0.001_s1
- config: `configs/phase2/pixel_lr0.001_s1.yaml`
- command: `python -m train.train --config configs/phase2/pixel_lr0.001_s1.yaml`
- git: `b18f7ea+dirty`
- device: NVIDIA GeForce RTX 3060
- wall-clock: 1681.2 s
- outcome: ok
  - final val loss / copy baseline: 0.0389
  - effective rank / latent std median: 4.1 / 0.006
  - steps: 9760
  - peak GPU memory GB: 4.51
  - results: results/phase2/pixel_lr0.001_s1.json

## 2026-10-14 22:27 — Phase 2 training: pixel_lr0.001_s2
- config: `configs/phase2/pixel_lr0.001_s2.yaml`
- command: `python -m train.train --config configs/phase2/pixel_lr0.001_s2.yaml`
- git: `b18f7ea+dirty`
- device: NVIDIA GeForce RTX 3060
- wall-clock: 1692.6 s
- outcome: ok
  - final val loss / copy baseline: 0.0324
  - effective rank / latent std median: 4.5 / 0.006
  - steps: 9760
  - peak GPU memory GB: 4.51
  - results: results/phase2/pixel_lr0.001_s2.json

## 2026-10-15 01:51 — Phase 2 training: jepa_pooled_lr0.0003_s0
- config: `configs/phase2/jepa_pooled_lr0.0003_s0.yaml`
- command: `python -m train.train --config configs/phase2/jepa_pooled_lr0.0003_s0.yaml`
- git: `f0fc5cb+dirty`
- device: NVIDIA GeForce RTX 3060
- wall-clock: 1357.1 s
- outcome: ok
  - final val loss / copy baseline: 0.7915
  - effective rank / latent std median: 81.6 / 1.045
  - steps: 9760
  - peak GPU memory GB: 4.51
  - results: results/phase2/jepa_pooled_lr0.0003_s0.json

## 2026-10-15 02:14 — Phase 2 training: jepa_pooled_lr0.001_s0
- config: `configs/phase2/jepa_pooled_lr0.001_s0.yaml`
- command: `python -m train.train --config configs/phase2/jepa_pooled_lr0.001_s0.yaml`
- git: `f0fc5cb+dirty`
- device: NVIDIA GeForce RTX 3060
- wall-clock: 1346.5 s
- outcome: ok
  - final val loss / copy baseline: 0.5260
  - effective rank / latent std median: 98.7 / 1.059
  - steps: 9760
  - peak GPU memory GB: 4.51
  - results: results/phase2/jepa_pooled_lr0.001_s0.json

## 2026-10-15 02:14 — Phase 2 selection: jepa_pooled
- config: `-`
- command: `python -m train.select --model jepa_pooled --runs jepa_pooled_lr0.0003_s0 jepa_pooled_lr0.001_s0`
- git: `f0fc5cb+dirty`
- outcome: selected jepa_pooled_lr0.001_s0 (lr 0.001)
  - candidates (run: metric, collapsed): jepa_pooled_lr0.0003_s0: 0.7915, False; jepa_pooled_lr0.001_s0: 0.5260, False
  - configs written: jepa_pooled_lr0.001_s1.yaml, jepa_pooled_lr0.001_s2.yaml

## 2026-10-15 02:36 — Phase 2 training: jepa_pooled_lr0.001_s1
- config: `configs/phase2/jepa_pooled_lr0.001_s1.yaml`
- command: `python -m train.train --config configs/phase2/jepa_pooled_lr0.001_s1.yaml`
- git: `f0fc5cb+dirty`
- device: NVIDIA GeForce RTX 3060
- wall-clock: 1364.8 s
- outcome: ok
  - final val loss / copy baseline: 0.5345
  - effective rank / latent std median: 97.2 / 1.062
  - steps: 9760
  - peak GPU memory GB: 4.51
  - results: results/phase2/jepa_pooled_lr0.001_s1.json

## 2026-10-15 02:59 — Phase 2 training: jepa_pooled_lr0.001_s2
- config: `configs/phase2/jepa_pooled_lr0.001_s2.yaml`
- command: `python -m train.train --config configs/phase2/jepa_pooled_lr0.001_s2.yaml`
- git: `f0fc5cb+dirty`
- device: NVIDIA GeForce RTX 3060
- wall-clock: 1340.2 s
- outcome: ok
  - final val loss / copy baseline: 0.5288
  - effective rank / latent std median: 98.2 / 1.057
  - steps: 9760
  - peak GPU memory GB: 4.51
  - results: results/phase2/jepa_pooled_lr0.001_s2.json

## 2026-10-15 03:19 — Phase 2 training: static_vicreg_lr0.0003_s0
- config: `configs/phase2/static_vicreg_lr0.0003_s0.yaml`
- command: `python -m train.train --config configs/phase2/static_vicreg_lr0.0003_s0.yaml`
- git: `f0fc5cb+dirty`
- device: NVIDIA GeForce RTX 3060
- wall-clock: 1227.9 s
- outcome: ok
  - final val invariance loss: 0.0038
  - effective rank / latent std median: 58.7 / 1.033
  - steps: 9760
  - peak GPU memory GB: 4.51
  - results: results/phase2/static_vicreg_lr0.0003_s0.json

## 2026-10-15 03:40 — Phase 2 training: static_vicreg_lr0.001_s0
- config: `configs/phase2/static_vicreg_lr0.001_s0.yaml`
- command: `python -m train.train --config configs/phase2/static_vicreg_lr0.001_s0.yaml`
- git: `f0fc5cb+dirty`
- device: NVIDIA GeForce RTX 3060
- wall-clock: 1236.2 s
- outcome: ok
  - final val invariance loss: 0.0020
  - effective rank / latent std median: 70.7 / 1.027
  - steps: 9760
  - peak GPU memory GB: 4.51
  - results: results/phase2/static_vicreg_lr0.001_s0.json

## 2026-10-15 03:40 — Phase 2 selection: static_vicreg
- config: `-`
- command: `python -m train.select --model static_vicreg --runs static_vicreg_lr0.0003_s0 static_vicreg_lr0.001_s0`
- git: `f0fc5cb+dirty`
- outcome: selected static_vicreg_lr0.001_s0 (lr 0.001)
  - candidates (run: metric, collapsed): static_vicreg_lr0.0003_s0: 0.0038, False; static_vicreg_lr0.001_s0: 0.0020, False
  - configs written: static_vicreg_lr0.001_s1.yaml, static_vicreg_lr0.001_s2.yaml

## 2026-10-15 04:01 — Phase 2 training: static_vicreg_lr0.001_s1
- config: `configs/phase2/static_vicreg_lr0.001_s1.yaml`
- command: `python -m train.train --config configs/phase2/static_vicreg_lr0.001_s1.yaml`
- git: `f0fc5cb+dirty`
- device: NVIDIA GeForce RTX 3060
- wall-clock: 1238.3 s
- outcome: ok
  - final val invariance loss: 0.0024
  - effective rank / latent std median: 68.2 / 1.038
  - steps: 9760
  - peak GPU memory GB: 4.51
  - results: results/phase2/static_vicreg_lr0.001_s1.json

## 2026-10-15 04:22 — Phase 2 training: static_vicreg_lr0.001_s2
- config: `configs/phase2/static_vicreg_lr0.001_s2.yaml`
- command: `python -m train.train --config configs/phase2/static_vicreg_lr0.001_s2.yaml`
- git: `f0fc5cb+dirty`
- device: NVIDIA GeForce RTX 3060
- wall-clock: 1246.2 s
- outcome: ok
  - final val invariance loss: 0.0021
  - effective rank / latent std median: 67.6 / 1.042
  - steps: 9760
  - peak GPU memory GB: 4.51
  - results: results/phase2/static_vicreg_lr0.001_s2.json

## 2026-10-15 04:48 — Phase 2 training: nonphys_jepa_spatial_lr0.0003_s0
- config: `configs/phase2/nonphys_jepa_spatial_lr0.0003_s0.yaml`
- command: `python -m train.train --config configs/phase2/nonphys_jepa_spatial_lr0.0003_s0.yaml`
- git: `f0fc5cb+dirty`
- device: NVIDIA GeForce RTX 3060
- wall-clock: 1549.8 s
- outcome: ok
  - final val loss / copy baseline: 1.2231
  - effective rank / latent std median: 41.3 / 1.007
  - steps: 9760
  - peak GPU memory GB: 4.51
  - results: results/phase2/nonphys_jepa_spatial_lr0.0003_s0.json

## 2026-10-15 05:13 — Phase 2 training: nonphys_jepa_spatial_lr0.001_s0
- config: `configs/phase2/nonphys_jepa_spatial_lr0.001_s0.yaml`
- command: `python -m train.train --config configs/phase2/nonphys_jepa_spatial_lr0.001_s0.yaml`
- git: `f0fc5cb+dirty`
- device: NVIDIA GeForce RTX 3060
- wall-clock: 1550.1 s
- outcome: ok
  - final val loss / copy baseline: 0.5116
  - effective rank / latent std median: 55.6 / 1.011
  - steps: 9760
  - peak GPU memory GB: 4.51
  - results: results/phase2/nonphys_jepa_spatial_lr0.001_s0.json

## 2026-10-15 05:14 — Phase 2 selection: nonphys_jepa_spatial
- config: `-`
- command: `python -m train.select --model nonphys_jepa_spatial --runs nonphys_jepa_spatial_lr0.0003_s0 nonphys_jepa_spatial_lr0.001_s0`
- git: `f0fc5cb+dirty`
- outcome: selected nonphys_jepa_spatial_lr0.001_s0 (lr 0.001)
  - candidates (run: metric, collapsed): nonphys_jepa_spatial_lr0.0003_s0: 1.2231, False; nonphys_jepa_spatial_lr0.001_s0: 0.5116, False
  - configs written: nonphys_jepa_spatial_lr0.001_s1.yaml, nonphys_jepa_spatial_lr0.001_s2.yaml

## 2026-10-15 05:39 — Phase 2 training: nonphys_jepa_spatial_lr0.001_s1
- config: `configs/phase2/nonphys_jepa_spatial_lr0.001_s1.yaml`
- command: `python -m train.train --config configs/phase2/nonphys_jepa_spatial_lr0.001_s1.yaml`
- git: `f0fc5cb+dirty`
- device: NVIDIA GeForce RTX 3060
- wall-clock: 1549.1 s
- outcome: ok
  - final val loss / copy baseline: 0.6227
  - effective rank / latent std median: 56.8 / 1.017
  - steps: 9760
  - peak GPU memory GB: 4.51
  - results: results/phase2/nonphys_jepa_spatial_lr0.001_s1.json

## 2026-10-15 06:04 — Phase 2 training: nonphys_jepa_spatial_lr0.001_s2
- config: `configs/phase2/nonphys_jepa_spatial_lr0.001_s2.yaml`
- command: `python -m train.train --config configs/phase2/nonphys_jepa_spatial_lr0.001_s2.yaml`
- git: `f0fc5cb+dirty`
- device: NVIDIA GeForce RTX 3060
- wall-clock: 1482.3 s
- outcome: ok
  - final val loss / copy baseline: 0.6644
  - effective rank / latent std median: 56.5 / 1.006
  - steps: 9760
  - peak GPU memory GB: 4.51
  - results: results/phase2/nonphys_jepa_spatial_lr0.001_s2.json

## 2026-10-15 15:46 — Phase 3 probes, val preview, raw readout only (CRASHED after writing outputs)
- config: `configs/phase3_val_raw.yaml`
- command: `python -m probes.phase3_val --config configs/phase3_val_raw.yaml`
- git: `e35cf1f+dirty`
- outcome: crashed at the final console print (UnicodeEncodeError: stdout redirected to a file used cp1252, which cannot encode the arrow in the table); results JSON and MD were already written and are complete, but this log entry was not written by the script (added by hand)
  - started / crashed: 15:36:50 / ~15:44
  - results: results/phase3/val_preview_raw.json, results/phase3/val_preview_raw.md
  - fix: sys.stdout.reconfigure(encoding="utf-8") at the start of main() in probes/phase3_val.py and probes/final_evaluation.py (output only; no computation changed)

## 2026-10-15 15:46 — Phase 3 probes, val preview, full (STOPPED by hand)
- config: `configs/phase3_val.yaml`
- command: `python -m probes.phase3_val --config configs/phase3_val.yaml`
- git: `e35cf1f+dirty`
- outcome: stopped ~2 min after starting (15:44:48), before any output, because it would have hit the same encoding crash; relaunched after the fix

## 2026-10-15 17:55 — Phase 3 probes, val preview
- config: `configs/phase3_val.yaml`
- command: `python -m probes.phase3_val --config configs/phase3_val.yaml`
- git: `e35cf1f+dirty`
- device: NVIDIA GeForce RTX 3060
- wall-clock: 7758.5 s
- outcome: ok
  - preview verdicts (raw readout): {'jepa_spatial': 'not supported', 'pixel': 'not supported', 'jepa_pooled': 'not supported', 'static_vicreg': 'not supported', 'nonphys_jepa_spatial': 'not supported'}
  - results: results/phase3/val_preview.json

## 2026-10-15 18:26 — Phase 4 views (val; unscored) — superseded
- note: superseded by the 18:29 run. Traversal readout changed from nearest neighbours in the
  full 32-dim PCA space (the other directions dominated the distance, so several steps returned
  the same trajectory) to the frames nearest each step along the slow coordinate, and extended
  from the slowest direction to the top 3. Embedding views unchanged.
- config: `configs/phase4_views.yaml`
- command: `python -m probes.phase4_views --config configs/phase4_views.yaml`
- git: `04b3a17+dirty`
- device: NVIDIA GeForce RTX 3060
- wall-clock: 50.7 s
- outcome: ok
  - figures: results/phase4_pca.png, results/phase4_tsne.png, results/phase4_traversals.png
  - results: results/phase4/views.json

## 2026-10-15 18:29 — Phase 4 views (val; unscored) — superseded
- note: superseded by the next run, identical except the embedding figures are saved with a tight
  bounding box (colorbar labels were clipped).
- config: `configs/phase4_views.yaml`
- command: `python -m probes.phase4_views --config configs/phase4_views.yaml`
- git: `04b3a17+dirty`
- device: NVIDIA GeForce RTX 3060
- wall-clock: 53.1 s
- outcome: ok
  - figures: results/phase4_pca.png, results/phase4_tsne.png, results/phase4_traversals_untrained.png, results/phase4_traversals_jepa_spatial.png, results/phase4_traversals_pixel.png
  - results: results/phase4/views.json

## 2026-10-15 18:32 — Phase 4 views (val; unscored)
- config: `configs/phase4_views.yaml`
- command: `python -m probes.phase4_views --config configs/phase4_views.yaml`
- git: `04b3a17+dirty`
- device: NVIDIA GeForce RTX 3060
- wall-clock: 55.7 s
- outcome: ok
  - figures: results/phase4_pca.png, results/phase4_tsne.png, results/phase4_traversals_untrained.png, results/phase4_traversals_jepa_spatial.png, results/phase4_traversals_pixel.png
  - results: results/phase4/views.json

## 2026-10-15 19:22 — FINAL test evaluation (single test-set touch)
- config: `configs/final_evaluation.yaml`
- command: `python -m probes.final_evaluation --config configs/final_evaluation.yaml`
- git: `c2c3a79+dirty`
- device: NVIDIA GeForce RTX 3060
- wall-clock: 2553.8 s
- outcome: verdict: not supported
  - note (added by hand): the tree was clean at c2c3a79 when the run started (18:39:49); "+dirty"
    is this run's own output in results/final/, written before the log call. Run once, no rerun.
    Code path dry-run beforehand on train/val only (test splits swapped for val; no test rows read).
  - results: results/final/final_evaluation.json
  - tables: results/final/final_evaluation.md

## 2026-10-15 19:58 — [EXPLORE] data generation: explore
- config: `configs/explore/data_explore.yaml`
- command: `python -m data.generate --config configs/explore/data_explore.yaml`
- git: `c11e97a+dirty`
- device: cuda
- wall-clock: 62.9 s
- outcome: ok
  - frames: 54000
  - split counts (trajectories): {'excluded_re_band': 74, 'explore': 1006}
  - inviscid trajectories: 378
  - inviscid max drift E / Z / C4: 7.7e-09 / 1.2e-06 / 0.080
  - inviscid top-shell enstrophy share at last frame (median / share > 5%): 0.008 / 0.061
  - input scale (copied from data/generated/v1): 0.9548
  - sha256 velocity.npy: 81b741f57720b176…

## 2026-10-15 19:59 — [EXPLORE] data generation: confirm (sealed; counts and checksums only)
- config: `configs/explore/data_confirm.yaml`
- command: `python -m data.generate --config configs/explore/data_confirm.yaml`
- git: `c11e97a+dirty`
- device: cuda
- wall-clock: 64.3 s
- outcome: ok
  - frames: 54000
  - split counts (trajectories): {'confirm': 1005, 'excluded_re_band': 75}
  - inviscid trajectories: 378
  - input scale (copied from data/generated/v1): 0.9548
  - sha256 velocity.npy: ab1e009e41f14f15…

## 2026-10-15 19:59 — [EXPLORE] data generation: matched-energy
- config: `configs/explore/data_matched_energy.yaml`
- command: `python -m data.generate --config configs/explore/data_matched_energy.yaml`
- git: `c11e97a+dirty`
- device: cuda
- wall-clock: 6.6 s
- outcome: ok
  - frames: 7200
  - split counts (trajectories): {'matched': 300}
  - inviscid trajectories: 300
  - inviscid max drift E / Z / C4: 1.6e-09 / 3.7e-08 / 0.077
  - inviscid top-shell enstrophy share at last frame (median / share > 5%): 0.015 / 0.167
  - input scale (copied from data/generated/v1): 0.9548
  - sha256 velocity.npy: d450453f12e7d14c…

## 2026-10-15 20:42 — [EXPLORE] Stage 0: spatial-map readouts of the existing models (explore set)
- config: `configs/explore/stage0.yaml`
- command: `python -m explore.stage0 --config configs/explore/stage0.yaml`
- git: `bf9be1a+dirty`
- device: NVIDIA GeForce RTX 3060
- wall-clock: 2109.0 s
- outcome: ok
  - JEPA enstrophy |r| by readout: pooled 0.353, map_meanstd 0.303, map_pca512 0.295
  - results: results/explore/stage0.json, results/explore/stage0.md, results/explore/stage0_readouts.png

## 2026-10-15 20:45 — [EXPLORE] Stage 0 diagnostic: PCA-512 vs the spatial-mean subspace
- config: `configs/explore/stage0.yaml`
- command: `inline script (see results/explore/stage0_pca_diagnostic.md)`
- git: `bf9be1a+dirty`
- device: NVIDIA GeForce RTX 3060
- outcome: ok
  - share of spatial-mean subspace inside top-512 PCs (seed 0): untrained 0.101, jepa_spatial 0.488, pixel 0.083, static_vicreg 0.643
  - results: results/explore/stage0_pca_diagnostic.md

## 2026-10-15 20:53 — [EXPLORE] Stage 1: does the pixel model's enstrophy lead survive? (kill gate)
- config: `configs/explore/stage1.yaml`
- command: `python -m explore.stage1 --config configs/explore/stage1.yaml`
- git: `153f20c+dirty`
- device: NVIDIA GeForce RTX 3060
- wall-clock: 88.9 s
- outcome: gate passed: lead survives
  - (a) explore gap: +0.411 [+0.326, +0.514]
  - (b) matched pixel / control: 0.979 / 0.177, gap CI [+0.718, +0.827]
  - (b) frames 0–15 (unscored): 0.976 / 0.170
  - results: results/explore/stage1.md, results/explore/stage1.png, results/explore/stage1.json.gz

## 2026-10-15 21:06 — [EXPLORE] regression check: Phase 2 cosine path unchanged: regress_smoke_pixel
- config: `C:/Users/Manav/AppData/Local/Temp/claude/c--Users-Manav-geometric-scout/e52bf240-1420-472a-8838-5dba35b0a4a9/scratchpad/regress_smoke_pixel.yaml`
- command: `python -m train.train --config C:/Users/Manav/AppData/Local/Temp/claude/c--Users-Manav-geometric-scout/e52bf240-1420-472a-8838-5dba35b0a4a9/scratchpad/regress_smoke_pixel.yaml`
- git: `6207b9f+dirty`
- device: NVIDIA GeForce RTX 3060
- wall-clock: 51.6 s
- outcome: ok
  - final val loss / copy baseline: 1.1566
  - effective rank / latent std median: 5.3 / 0.010
  - steps: 244
  - peak GPU memory GB: 4.51
  - results: C:/Users/Manav/AppData/Local/Temp/claude/c--Users-Manav-geometric-scout/e52bf240-1420-472a-8838-5dba35b0a4a9/scratchpad/runs/regress_smoke_pixel.json

## 2026-10-15 21:07 — [EXPLORE] Stage 2 smoke (code check, 460 steps): smoke_stage2_pixel_vicreg_k32
- config: `C:/Users/Manav/AppData/Local/Temp/claude/c--Users-Manav-geometric-scout/e52bf240-1420-472a-8838-5dba35b0a4a9/scratchpad/smoke_stage2_pixel_vicreg.yaml`
- command: `python -m train.train --config C:/Users/Manav/AppData/Local/Temp/claude/c--Users-Manav-geometric-scout/e52bf240-1420-472a-8838-5dba35b0a4a9/scratchpad/smoke_stage2_pixel_vicreg.yaml`
- git: `6207b9f+dirty`
- device: NVIDIA GeForce RTX 3060
- wall-clock: 94.5 s
- outcome: ok
  - final val loss / copy baseline: 0.1993
  - effective rank / latent std median: 28.8 / 0.956
  - steps: 460
  - peak GPU memory GB: 4.51
  - results: C:/Users/Manav/AppData/Local/Temp/claude/c--Users-Manav-geometric-scout/e52bf240-1420-472a-8838-5dba35b0a4a9/scratchpad/runs/smoke_stage2_pixel_vicreg_k32.json
  - stop reason / lr reductions: step cap / 0
  - diagnostics: pred_over_true_energy_k_ge_5: 0.308, copy_over_true_energy_k_ge_5: 1.844, pred_over_true_energy_k_ge_10: 2.124, copy_over_true_energy_k_ge_10: 0.702

## 2026-10-15 21:09 — [EXPLORE] Stage 2 smoke (code check, 460 steps): smoke_stage2_jepa_spatial_k32
- config: `C:/Users/Manav/AppData/Local/Temp/claude/c--Users-Manav-geometric-scout/e52bf240-1420-472a-8838-5dba35b0a4a9/scratchpad/smoke_stage2_jepa_spatial.yaml`
- command: `python -m train.train --config C:/Users/Manav/AppData/Local/Temp/claude/c--Users-Manav-geometric-scout/e52bf240-1420-472a-8838-5dba35b0a4a9/scratchpad/smoke_stage2_jepa_spatial.yaml`
- git: `6207b9f+dirty`
- device: NVIDIA GeForce RTX 3060
- wall-clock: 88.3 s
- outcome: ok
  - final val loss / copy baseline: 0.5524
  - effective rank / latent std median: 13.1 / 0.558
  - steps: 460
  - peak GPU memory GB: 4.51
  - results: C:/Users/Manav/AppData/Local/Temp/claude/c--Users-Manav-geometric-scout/e52bf240-1420-472a-8838-5dba35b0a4a9/scratchpad/runs/smoke_stage2_jepa_spatial_k32.json
  - stop reason / lr reductions: step cap / 0
  - diagnostics: pred_over_target_map_variance: 0.590
