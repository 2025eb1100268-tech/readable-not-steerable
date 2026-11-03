# geometric-scout

Does a JEPA-style world model, trained only to predict 2D Navier–Stokes velocity fields, organize
its latent space around the quantities used to describe fluids (energy, enstrophy, vortex
structure, viscosity) better than an untrained encoder of the same architecture? The success
criteria are pre-registered in `PREREGISTRATION.md` (written before any training); results go in
`REPORT.md`; every run is logged in `results/LOG.md`.

## Paper, checkpoints and licences

This repository accompanies the paper "Readable Is Not Steerable: Pre-Registered Tests of Physical
Structure in Learned Fluid Representations". Its source and build are in `paper/` (`paper/BUILD_REPORT.md`
describes the build), and `results/paper/` holds the paper-materials pack. The commit history is part of
the record: each pre-registration was committed before the data it governs were opened.

The 50 scored checkpoints (Study 2: 39; Study 3: 11) are attached to the v1.0 release. Their SHA-256
values were committed before evaluation (`configs/explore/stage3_confirm.yaml`,
`configs/build/v3_frozen.json`); `results/paper/tools/checkpoint_manifest.txt` lists each file, its size
and the check against the frozen value.

- **Code** (Python modules and scripts, configurations, tests, build tools): MIT, see `LICENSE`.
- **Reports, results, figures and the paper** (the Markdown reports and pre-registrations, `results/`,
  `paper/`): CC BY 4.0, see `LICENSE-CC-BY-4.0.txt`.
- The TMLR style files in `paper/` (`tmlr.sty`, `tmlr.bst`, `fancyhdr.sty`) keep their own licences
  (`paper/tmlr/`).

## Setup (Windows, Python 3.13, NVIDIA GPU)

```
py -3.13 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

All commands below run from the repo root with `.venv\Scripts\python.exe`. Each run reads one
YAML file in `configs/` and appends to `results/LOG.md`.

## Phase 0: solver

| What | Command | Output |
|---|---|---|
| Unit tests (solver, diagnostics, initial conditions) | `python -m pytest` | ~10 s |
| Hardware benchmark | `python -m common.benchmark --config configs/phase0_benchmark.yaml` | `results/phase0/benchmark.json` |
| Solver validation | `python -m solver.validate --config configs/phase0_solver_validation.yaml` | `results/solver_validation.png`, `results/phase0/solver_validation.json` |
| Data-design pilot | `python -m solver.pilot --config configs/phase0_pilot.yaml` | `results/phase0_pilot.png`, `results/phase0/pilot.json` |

## Layout

```
common/    configs, seeding, environment capture, results/LOG.md, benchmark
solver/    ns2d.py (pseudo-spectral vorticity solver, IF-RK4), diagnostics.py,
           initial_conditions.py, checks.py (validation experiments), validate.py, pilot.py
probes/    slow_features.py (unsupervised invariant discovery, Phase 4 protocol)
tests/     unit tests
configs/   one YAML per run
results/   LOG.md, figures, metrics JSON
data/, models/, train/   Phases 1–2
```

## Conventions

Domain [0, 2π]², ω = ∂v/∂x − ∂u/∂y, u = ∂ψ/∂y, v = −∂ψ/∂x, E = ½⟨|u|²⟩, Z = ½⟨ω²⟩,
P = ½⟨|∇ω|²⟩, so dE/dt = −2νZ for unforced flow. 2/3-rule dealiasing (|k_i| ≤ n/3).
