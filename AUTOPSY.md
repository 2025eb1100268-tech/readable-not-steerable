# [BUILD] Autopsy of the v3 confirmation (exploratory)

**Status: exploratory and post hoc. Nothing here changes REPORT_v3's verdict (partially supported: both
dials in 3 of 5 seeds).**

- No training and no new data.
- `confirm3` and `matched-energy-confirm` were not reopened. Directions, g and sealed-set scores are read
  from the Stage 5 outputs (`258d6af`).
- Every new measurement uses exploration data only: v1 train, the 256 explore optimization frames, and
  256 frames each of `explore` and the exploration matched-energy set. The v3 criteria apply, with the
  mean removed.
- Procedure and predictions: `results/build/PREDICTIONS.md` entry 11 (`1cbc2e7`), committed before any
  computation. Script: `dials/autopsy.py` (`195a4c3`). Outputs: `results/build/autopsy/` (`3c55d66`).

## Bottom line

- **Seed 102: the investigator's hypothesis does not hold. Shape is not encoded along the loudness
  gradient.**
  - cos²(g, PC2) is 0.000, and projecting out g removes none of PC2's shape effect (−0.00, both
    linear and steered).
  - What differs is the variance structure of the z-scored data. After the projection, the top
    direction is not PC2's projection: their variances are 2.39 and 1.48. In the other four seeds
    the two are within 0.15 of each other.
  - So the rule picked a direction with less than half of PC2's raw spread (σ_d 2.11 against 4.47).
  - This points at the z-scoring step, not the null-space projection. It was not tested by rerunning
    the rule without z-scoring.
- **Seed 104: also not the loudness gradient.** cos²(g, PC2) is 0.001, and the projection removes
  0.02 of PC2's shape effect.
  - The direction is nearly PC2 (|cos| 0.96). It misses one criterion: the envelope on matched-energy
    data.
  - The miss reproduces on the exploration matched-energy set at the same size (−0.09; tolerance
    −0.05). It is a property of the direction, not chance in the sealed set. PC2 has +0.00 there.
- **The union "loudness-null OR PC2" is not lenient on the controls.** It finds a shape dial in 0 of 3
  τ = 0 seeds and 0 of 3 untrained seeds.
  - In every main seed, one of the two directions is a valid shape dial on all four sets: loudness-null
    in 100, 101 and 103, and PC2 in 102 and 104. This is a post hoc reading.
- **The transformation-matched (TM) rule does not fix it.** It finds a shape dial in 0 of 5 main seeds,
  including 102 and 104, and in 0 of 6 controls.
  - The directions it picks are clean but weak. On explore their change is mostly shape (σ_shape
    0.89–0.97), but too small (effect 0.12–0.17; the criterion is 0.2).
  - Re-encoding recovers only 30–49% of the commanded move (cycle gain; the criterion is ≥ 0.5).
- **Seed 103 without mean removal: the same classification on the exploration sets.** The mean offset
  also appears there (16.4% of frames on explore, against 23.0% on `confirm3`). Whether 103 passes on
  `confirm3` without the removal remains unanswered, by design.

## A1: seed 102 and the loudness gradient

In the z-scored dial space of the rule:

- g is the frame-mean gradient of a.
- h is the frame-mean gradient of s.
- "Removed" is the fraction of PC2's shape effect lost when its g component is projected out.
- The recomputed g matched the stored g in every seed (|cos| 1.0000).

| seed | cos²(g, PC2) | cos²(g, h) | removed: linear / steered | projected z-variance: top / 2nd / along PC2 | σ_d loudness-null / PC2 |
|---|---|---|---|---|---|
| 100 | 0.009 | 0.28 | +0.07 / +0.08 | 2.91 / 0.85 / 2.90 | 5.12 / 5.12 |
| 101 | 0.057 | 0.29 | +0.17 / +0.18 | 2.44 / 1.44 / 2.43 | 5.22 / 5.16 |
| **102** | **0.000** | 0.23 | **−0.00 / −0.00** | 2.39 / 1.61 / **1.48** | **2.11** / 4.47 |
| 103 | 0.117 | 0.50 | −0.66 / −0.70 | 2.32 / 1.90 / 2.17 | 5.34 / 5.05 |
| 104 | 0.001 | 0.19 | +0.02 / +0.01 | 3.08 / 1.06 / 2.94 | 3.70 / 3.84 |

Two more observations from this table:

- In seed 103, projecting out g *increases* PC2's shape effect by about two-thirds. There PC2 mixes in
  loudness that works against shape. This fits PC2 failing 103 as "mostly loudness" (σ_shape 0.36).
- On the exploration sets, the stored 102 direction fails again with the same profile as on the sealed
  sets:
  - effect 0.09;
  - monotonicity 0.37;
  - cycle correlation −0.21;
  - artefacts in 3.9% / 3.5% of frames.

## A3: transformation-matched rule

How the rule works:

- **The family.** The family is fixed-energy log-k tilt and curvature of decoded vorticity in shells
  1–21: the content noise's own distortion family.
- **The objective.** J_T is the fraction of the steered change this family cannot explain.
- **The search.** It is the symmetry rule's search: Adam from PC1–8 and 4 random starts.
- **Scoring.** Each direction found is scored with the full v3 harness, on both exploration sets.

| group | J_T of TM direction | TM effect, explore / matched | TM cycle gain, explore / matched | TM σ_shape, explore / matched | TM shape dial |
|---|---|---|---|---|---|
| main (5) | 0.102–0.111 | 0.12–0.17 / 0.18–0.24 | 0.35–0.49 / 0.30–0.45 | 0.89–0.97 / 0.62–0.74 | 0 of 5 |
| τ = 0 (3) | 0.344–0.410 | 0.008–0.014 / 0.024–0.027 | 0.03–0.05 / 0.06–0.09 | — | 0 of 3 |
| untrained (3) | 0.819–0.866 | 0.30–1.05 / 0.07–0.78 | — | — | 0 of 3 (grid artefacts in 100% of frames) |

What the table shows:

- **Cycle gain fails in every main seed, on both sets.** Effect fails on explore in all five, and on
  matched-energy in 3 of 5.
- **The rule is indifferent to effect size.** J_T is a ratio, so minimizing it rewards a change that is
  *well explained*, not one that is large. The TM directions have the lowest spread of any rule (σ_d
  2.3–3.4, against about 5 for loudness-null and PC2).
- **Why re-encoding recovers so little of the move was not analysed.**
- **J_T itself separates the groups cleanly.** It is about 0.10 in main models, 0.34–0.41 at τ = 0 and
  0.82–0.87 untrained.
  - It also flags the failed 102 direction: J_T 0.69, against 0.19–0.35 for the other loudness-null
    directions.
  - It does not flag 104 (0.27).
  - Read this as a post hoc diagnostic, not a dial rule.
- **The family can partly mimic a rescaling.** A pure ×0.5 rescale gives J_T 0.22. This did not confuse
  the rule:
  - the TM directions barely move loudness (|Δa| ≤ 0.015 on explore);
  - the stored loudness dials score J_T 0.62–0.63.

## A4: seed 103 with and without mean removal (exploration sets)

| direction | set | mean removed | mean kept | frames with > 10% of ω² in the mean (worst α) | Δs removed / kept |
|---|---|---|---|---|---|
| loudness | explore | loudness dial | loudness dial | 2.7% | −0.001 / −0.003 |
| loudness | matched-energy (expl.) | loudness dial | loudness dial | 0.0% | −0.029 / −0.029 |
| shape (loudness-null) | explore | shape dial | shape dial | 16.4% | +0.343 / +0.340 |
| shape (loudness-null) | matched-energy (expl.) | shape dial | shape dial | 0.0% | +0.310 / +0.309 |

## Predictions against outcomes (entry 11)

| | prediction | p | outcome |
|---|---|---|---|
| PA.1 | Investigator's hypothesis holds in 102 | 20% | **no** (removed −0.00) |
| PA.2 | 102 keeps PC2's shape effect, but PC2 is not the top direction after z-scoring | 65% | **yes** (−0.00; 1.48 against 2.39) |
| PA.3 | (A2, no prediction) | — | union: 0 of 3 τ = 0, 0 of 3 untrained |
| PA.4 | TM finds a shape dial in 102 / in 104 | 60% / 55% | **no / no** |
| PA.5 | TM gives both dials in ≥ 4 of 5 seeds | 50% | **no** (0 of 5) |
| PA.6 | TM finds no shape dial in the controls | 85% | **yes** (0 of 6), but it found none in the main seeds either |
| PA.7 | loudness-null fails again in 102 / passes in 104 | 90% / 55% | **yes / no** (104 misses the envelope again, −0.09) |
| PA.8 | 103's classification is unchanged without mean removal | 65% | **yes** (exploration sets) |

- My confident calls held: PA.6 and PA.7a.
- I was over-optimistic about the TM rule: PA.4 and PA.5 were all wrong.

## Caveats

- **Post hoc.** Everything is post hoc and was chosen after seeing the Stage 5 failures. The v3 models
  have now informed rule design, so they cannot confirm any revised rule.
- **Exploration data only.** Exploration sets come from the same distributions as the sealed sets, but
  they are not the sealed sets.
- **The z-scoring explanation for 102 is inferred, not tested.** It comes from the variance table; the
  rule was not rerun without z-scoring.
- **Grid-edge fits.** In 30,781 of 5,586,944 TM family fits (0.55%), the coarse grid's edge (|t| = 8)
  was reached. The fits were not broken down by model.
- **A 37-second code check ran before the full run.** It used seed 102 only, with 2 restarts × 3
  steps and 32 frames, and wrote no outputs. Its printed classifications (for example, TM "dial, not
  shape") are a code check, not a result.

## What a v4 would pre-register

These are candidates; the choices are the investigator's.

1. **Fresh main and control seeds, and new sealed sets.** Generate and seal new versions of `confirm3`
   and the matched-energy set before training. The v3 models and sealed sets are spent for
   confirmation.
2. **One shape rule, fixed in advance, after an exploration-only check on the v3 models:**
   - (a) **Loudness-null without z-scoring** (projection and PCA in raw latent space). This targets
     the 102 mechanism, but has not been tested anywhere.
   - (b) **The union "loudness-null OR PC2", declared as the rule.** Post hoc on v3 it gives 5 of 5
     main seeds and 0 of 6 controls. It is two attempts per seed, so the controls carry the weight.
   - (c) **The TM rule as it stands: not recommended** (0 of 5). An effect-size term would make it a
     new, untested rule.
3. **The same criteria, including the envelope tolerance that 104 missed.** Nothing is relaxed after
   the fact.
4. **Pre-registered diagnostics, reported but not gated,** so that a failing seed can be explained
   without a post hoc analysis:
   - the A1 quantities (cos²(g, PC2), removed fraction, projected variances);
   - J_T of every scored direction;
   - classification with the mean kept as well as removed.
5. **Seed count and pass threshold, set with the interval in mind:**
   - 3 of 5 has a 95% interval of 15–95%; 8 of 10 has 44–97%.
   - With "≥ 4 of 5", a design that works 80% of the time passes with probability 0.74; at 60% it
     passes with 0.34.
   - Main training costs about 0.77 GPU-hours per seed.

## Compute

| item | GPU-hours |
|---|---|
| Full autopsy run (11 models; 19:22–21:38 on 2026-10-29) | 2.27 |
| Code check (seed 102, truncated) | 0.01 |

The estimate in entry 11 was about 2 GPU-hours. This is reported separately from the v3 budget (8.8 of
10 GPU-hours used).
