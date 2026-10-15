# [EXPLORE] Brief: which conserved quantities are prominent in a latent, and does imagining the future change that?

The investigator's brief for this exploratory line, committed verbatim (2026-10-15), followed by
the changes agreed before any work started. Where the two differ, the agreed changes govern.
**Framing rule (agreed):** nothing in this line of work may claim a model "discovered
conservation". The supportable claim is about which conserved quantities are *prominent in the
latent*.

---

## Original brief (verbatim)

## Context

The confirmatory study is closed as **not supported** (see `REPORT.md`). I'm not reopening it. This is a new, **exploratory** line of work built on one unplanned result from that study:

> The pixel predictor (the model that decodes the actual future frame) was the only model whose unsupervised slow directions tracked enstrophy: |r| 0.82 vs 0.51 for the untrained control (gap CI +0.15 to +0.50). The JEPA, which predicts an abstract latent summary, did worse than the control (0.36).

**New question:** Does *rendering* the future, as opposed to summarizing it, make a network organize its latent space around conserved quantities? And does that effect get stronger when the future is harder to picture, i.e. at longer prediction horizons?

Read this whole brief, then reply with a plan and wait for my go-ahead. Push back on anything here that's wrong.

## Hard rules

1. **Everything in Stages 0–2 is exploratory.** Label it so in every file, figure title and log line. Nothing from Stages 0–2 is a "finding". Only Stage 3 can produce a claim.
2. **The old test sets are retired.** Do not read `test-ID` or `test-OOD` again for anything.
3. **Fresh data for everything new.** Generate new datasets with the existing generator, using seed ranges that provably don't overlap the original data. Add a test that checks for overlap (seeds and initial-condition hashes).
   - `explore/` set: used freely in Stages 0–2.
   - `confirm/` set: generated now, checksummed, **never opened** until Stage 3. Add a guard in code that refuses to load it unless a flag set only by the Stage 3 script is present.
4. **Write predictions before running.** Before each stage, append to `results/explore/PREDICTIONS.md` what you expect to see and why. Don't edit past entries.
5. **Report everything you run.** Every horizon, every direction, every seed, including the ones that look bad. No "best of" tables without the full table beside them.
6. **Same infrastructure as before:** same encoder architecture, same probe code, same slow-direction method, same partialling, same bootstrap. Only change what this brief says to change. Reuse, don't rewrite.
7. **Stop after each stage** with a 5-line summary and the key numbers. Ask me before going past a kill gate, adding a model family, or exceeding the compute budget.
8. All new outputs go under `results/explore/`. Log every run in `results/LOG.md` with an `[EXPLORE]` tag.

**Compute budget:** about 14 GPU-hours on the RTX 3060 for Stages 0–2. Tell me up front if the plan doesn't fit.

## Data

Generate, with the original generator and settings:
- `explore/`: 1,000 trajectories, with the inviscid share raised so there are **at least 300 inviscid trajectories**. S3-style statistics were unstable (about ±0.2) on 74 trajectories, and this stage leans on them.
- `confirm/`: same recipe and size, sealed as in rule 3.
- `matched-energy/` (exploration only): 300 inviscid trajectories whose initial fields are rescaled so **every trajectory starts at the same kinetic energy**, while enstrophy varies through spectral shape (vary the spectral peak and slope). Verify in a test that initial-energy spread is near zero and initial-enstrophy spread is wide.

## Stage 0 — Housekeeping (existing checkpoints only, no training)

**0a. Spatial-map probes (old decision 3a).** On `explore/`, probe the full 8×8×128 spatial maps of all six existing models, not just the pooled vector. Use ridge on a fixed reduction (e.g. PCA to 512 dims fitted on train) so all models get the same readout. Report the S1 targets, ν, and the slow-direction analysis next to the pooled results.

Purpose: if the JEPA's maps carry physics that pooling hid, the "imaginer vs summarizer" contrast might be a readout artefact. I need to know that before betting on it.

## Stage 1 — Does the lead survive? (existing checkpoints only, no training)

**1a. Replication.** Using the existing pixel and untrained checkpoints (3 seeds each), with slow directions fitted on the original train+val exactly as pre-registered, score on the inviscid trajectories in `explore/`. Report |r| with initial enstrophy for each of the top-3 directions separately, plus the max.

**1b. Is it just energy?** In 2D decaying flow, energy and enstrophy are correlated across trajectories. The pixel model also tracks energy strongly (|r| 0.92). Check whether its enstrophy direction is an energy direction in disguise:
- partial correlation of each slow direction with log enstrophy, controlling for log energy;
- the same slow-direction analysis on `matched-energy/`, where energy can't carry the signal.

**1c. Why the imaginer keeps it (old decision 3c).** Correlate the pixel model's enstrophy direction with the field's spectral slope, spectral centroid and vorticity-PDF width. Plain-language summary of what the direction is.

**Kill gate (write it in PREDICTIONS.md before running Stage 1):** the lead survives only if, on `explore/`, the pixel model beats the untrained control on enstrophy |r| by ≥ 0.1 with the trajectory-bootstrap CI excluding zero, **and** the partial correlation controlling for energy stays ≥ 0.5 for the pixel model. If either fails, stop, report, and don't start Stage 2.

## Stage 2 — Make imagining hard (new training)

Train **direct** predictors (from frames t−1, t to frame t+k) at horizons **k ∈ {1, 4, 16, 32}**, 3 seeds each, for:

1. **Imaginer:** the pixel predictor, same architecture and loss as before.
2. **Summarizer at the same horizons:** the spatial JEPA, same setup as before but single-horizon. This is the critical control. If the JEPA also finds enstrophy at long horizons, the ingredient is *horizon*, not *rendering*.
3. **Fake-physics imaginer:** the pixel predictor trained on the non-physical dataset, at the single horizon where the real imaginer looks best (decided on `explore/` after models 1 and 2 finish, and logged before training). This tells us whether rendering *real* physics matters, or rendering any dynamics.

**Constraints to handle and tell me about in your plan:**
- Inviscid trajectories are cut at 24 frames, so k = 32 can't use them, and k = 16 barely can. Propose how to handle this (e.g. train long horizons on viscous runs only and report that difference, or regenerate longer viscous windows). Don't silently change the training distribution between horizons.
- Use the same epochs and lr rule for every model. Fix the earlier under-training issue: extend the lr grid to {1e-3, 3e-3} and train until val loss plateaus (patience-based stop, logged), with the same rule for all models.
- At long horizons, MSE rewards blurry conditional means. Log the energy spectrum of predicted vs true frames at each horizon so we can see how much the imaginer is actually picturing versus averaging.
- Log each model's val loss relative to the copy-current-frame baseline at its horizon.

**Exploratory readouts, all on `explore/` and `matched-energy/`:**
- **Primary curve:** enstrophy slow-direction |r| (and the energy-partialled version) **vs horizon**, imaginer and summarizer on the same plot, untrained control as a flat line, mean ± seed std with bootstrap CIs.
- Secondary: S1 probes, ν pair R², Okubo–Weiss fraction, all vs horizon.
- Traversals along the enstrophy-tracking direction at the best horizon.

## Stage 3 — Confirmation (only if I approve after Stage 2)

If Stage 2 shows a pattern, write `PREREGISTRATION_v2.md` with specific, numeric hypotheses drawn from it: which model, which horizon, which statistic, which threshold, which control. Commit it. Then open `confirm/` once, run the frozen pipeline once, and write `REPORT_v2.md` with the verdict. Same discipline as the first study.

If Stage 2 shows no pattern, write `results/explore/SUMMARY.md` saying so plainly, and stop.

## Threats to keep in view (put them in your plan)

- An enstrophy direction that is really amplitude or spectral slope (1b and 1c exist for this).
- Horizon effects confounded with training-distribution changes (the inviscid-length constraint).
- Blurry long-horizon predictions: the model may "imagine" less as k grows, not more.
- Forking paths: Stages 0–2 try many things. That's fine because they're exploratory, but it's why only Stage 3 counts.

---

## Agreed changes (2026-10-15, before any work)

1. **Training distribution (Stage 2).** One common anchor set for every Stage 2 model and horizon:
   viscous trajectories only, input frames t ∈ [1, 31] (so t + 32 ≤ 63). Identical inputs across
   horizons; only the target moves. Plus a **bridge run**: pixel predictor at k = 4 on the
   original training distribution (inviscid included, all valid t), 3 seeds, to measure what the
   distribution change does. The bridge is not to be cut.
2. **Learning rate.** Selected once per family at k = 4 from {1e-3, 3e-3} (seed 0, label-free:
   val loss / copy baseline), applied to every horizon. Listed as a threat.
3. **Schedule.** Same rule and cap for every Stage 2 model: warmup, then constant lr; lr × 0.3 on a
   val plateau; stop after two reductions without improvement or at the step cap. JEPA EMA momentum
   tied to the lr: m = 1 − 0.004 · lr / lr_peak.
4. **Stage 0a readouts.** All six models get three readouts: pooled latent; ridge on a PCA-512
   reduction of the flattened 8×8×128 map (PCA fitted on train); per-channel spatial mean ⊕ spatial
   std of the map (384 dims).
5. **Kill gate (final; supersedes the brief's).** The lead survives only if both hold:
   (a) on explore inviscid runs, the pixel model beats the untrained control on enstrophy |r|
   (best of top-3 slow directions, mean over seeds) by ≥ 0.1, trajectory-bootstrap 95% CI on the
   gap excluding zero;
   (b) on matched-energy, the pixel model's best-of-top-3 |r| with log Z₀ is ≥ 0.5 and exceeds the
   untrained control's, CI on the gap excluding zero.
   The partial r on explore (controlling for log E₀ and family) is reported alongside, unscored.
6. **1c descriptors.** Scale-free: vorticity kurtosis, high-k spectral slope, P₀/Z₀, plus a small
   regression. (The vorticity-PDF width is √(2Z) by definition, so it is not used.)
7. **Specificity.** Next to every slow-direction result, report |r| with non-conserved statistics of
   similar cross-trajectory spread (e.g. trajectory-mean palinstrophy).
8. **Pixel + VICReg family (added).** The pixel predictor with VICReg variance/covariance terms on
   its pooled latent, at the single horizon chosen for the fake-physics imaginer (same pre-logged
   rule), 3 seeds. Without it a Stage 2 difference cannot be attributed to rendering rather than
   regularization.
9. **Data.** Non-overlap test on initial-condition seeds and initial-field hashes; explore drops
   viscous runs inside the retired OOD band (Re 286–389); matched-energy uses spectral fields at
   E* = 0.289 (median initial energy of the original spectral inviscid train/val runs), peak and
   slope varied, Z₀ kept inside the original range [2, 8].
10. **Stage 3** runs on the frozen Stage 2 checkpoints (no retraining).
11. **Budget.** 18 GPU-hours for Stages 0–2, nothing cut. Stop and ask before exceeding 18.
12. **Budget raised (2026-10-15, after the first Stage 2 run).** The first selection run reached the
    12,000-step cap in 40.6 min (about 0.2 s per step including evaluations), so the 35 Stage 2 runs
    project to about 25 GPU-hours in total. The investigator chose to raise the budget to
    **26 GPU-hours** with no change to the protocol (no lower cap, no fewer seeds). Stop and ask
    before exceeding 26.
