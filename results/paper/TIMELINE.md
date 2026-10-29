# Paper materials: process timeline

This file covers every pre-registration, amendment, agreed change, prediction entry, sealed-set
creation and opening, and deviation or incident, with dates and commits.

- All times are IST, from commit timestamps or `results/LOG.md` entry headers. A log header gives the
  time the entry was written, usually the end of the run.
- Commits are local (branch `master`).

**Types:**
- PREREG: pre-registration committed
- AMEND: amendment
- AGREED: change agreed with the investigator before the work it governs
- PRED: prediction entry
- SEAL: sealed set created
- OPEN: sealed or single-use set opened
- DEV: deviation or incident
- RUN: key run
- REPORT: report committed

## 1. Pre-registrations at a glance

| | file | committed | commit | approved by the investigator | amendments | verdict | report |
|---|---|---|---|---|---|---|---|
| Study 1 | `PREREGISTRATION.md` | 2026-10-14 18:46 | `f7237f2` | yes (§2, 2026-10-14) | A1, 2026-10-14 18:53 (`1671301`) | not supported | `REPORT.md` `31eba06` |
| v2 | `PREREGISTRATION_v2.md` | 2026-10-17 17:32 | `8797966` | yes (commit message) | none | partially supported | `REPORT_v2.md` `a88ddee` |
| v3 | `PREREGISTRATION_v3.md` | 2026-10-27 20:16 | `781b2ad` | yes (commit message) | none | partially supported | `REPORT_v3.md` `aff659d` |

**Note for the process section:** the v2 and v3 files were committed with their status line unchanged
("DRAFT for the investigator's review. It becomes binding when committed."). Each states that it binds
on commit, and the commit messages record the approval. The seal code (`data/seal.py`) requires the file
to be committed and unmodified.

## 2. Sealed and single-use data

| set | created | for | commit | opened | by | outputs | status now |
|---|---|---|---|---|---|---|---|
| v1 test-ID (276 traj.) and test-OOD (150) | 2026-10-14 18:21 (Phase 1) | `PREREGISTRATION.md` §2.7: evaluated exactly once by one scripted run | `f7237f2` | 2026-10-15 18:39:49–19:22, once | `probes/final_evaluation.py` | `e8d619e` | retired 2026-10-15 (explore brief, rule 2; `c11e97a`) |
| `confirm_v2` (1,005 traj., 378 inviscid) | 2026-10-15 19:59 | `PREREGISTRATION_v2.md` (code seal) | `d8c94eb` | 2026-10-17 17:33 (UNSEAL logged), run 17:33–17:48, once | `explore/stage3_confirm.py` | `a88ddee` | retired 2026-10-17 (build brief, rule 2; `08c1511`) |
| `confirm3` (1,007 traj., 378 inviscid) | 2026-10-17 18:26 | `PREREGISTRATION_v3.md` (code seal) | `8e8bd1b` | 2026-10-29 16:52 (UNSEAL logged), run 16:52–18:48, once | `dials/v3_confirm.py` | `258d6af` | spent |
| `matched-energy-confirm` (300 inviscid) | 2026-10-27 20:05 | `PREREGISTRATION_v3.md` (code seal) | `26921ea`, `525b5e0` | same run as `confirm3` | same | `258d6af` | spent |

- The exploration sets (`explore`, `matched-energy`; `d8c94eb`) were used freely in the exploratory
  line, the build and the autopsy.
- The autopsy did not reopen any sealed set.

## 3. Chronology

### Study 1 (2026-10-13 to 10-15)

| when | type | event | commit / source |
|---|---|---|---|
| 2026-10-13/14 | DEV | About 70 CPU planning pilots before the project; summarized at the top of the log; not reproducible from the repo | LOG (top) |
| 2026-10-14 | AGREED | §2 clarifications approved before data generation and training: S1 rule with the halving branch and mandatory Okubo–Weiss; time × family partialling; the S3 guard; the spatial JEPA as main model; controls; readouts | `PREREGISTRATION.md` §2 |
| 2026-10-14 18:21 | RUN | Phase 1 data: 2,000 trajectories including the test splits | LOG 18:21 |
| 2026-10-14 18:46 | PREREG | `PREREGISTRATION.md` committed, before any model was trained | `f7237f2` |
| 2026-10-14 18:49–18:50 | RUN | Two 1-epoch smoke runs | LOG |
| 2026-10-14 18:53 | AMEND | A1: the no-collapse selection rule applies only to collapse-prone objectives (not the pixel predictor) | `1671301` |
| 2026-10-14 19:04 | — | Phase 2b data, probe and final-evaluation code committed before any results | `b18f7ea` |
| 2026-10-15 15:36–15:44 | DEV | Phase 3 raw-only val run crashed at its final console print (cp1252 encoding), after writing complete outputs. Fix is output-only (stdout encoding); the parallel full run was stopped by hand and relaunched. | LOG 15:46; `REPORT.md` §10 |
| 2026-10-15 | DEV | Val results seen before test. Afterwards, the only changes were the encoding fix and the unscored Phase 4 traversal readout. | `REPORT.md` §6.9 |
| 2026-10-15 18:26–18:32 | DEV | Phase 4 traversal readout changed after its first run (unscored figure): two superseded runs; Phase 4 views were only loosely specified | LOG; `REPORT.md` §10 |
| 2026-10-15 18:39 | — | Final-evaluation config committed before the test run; code path dry-run on train/val (no test rows) | `c2c3a79` |
| 2026-10-15 18:39–19:22 | OPEN | Single test evaluation (run once). The log shows `c2c3a79+dirty` only because of the run's own output files. | `e8d619e`; LOG 19:22 |
| 2026-10-15 19:32 | REPORT | `REPORT.md`: not supported | `31eba06` |

### Exploratory line and v2 (2026-10-15 to 10-17)

| when | type | event | commit / source |
|---|---|---|---|
| 2026-10-15 19:52 | AGREED, PRED | Exploratory brief (verbatim), agreed changes 1–11 (incl. the final kill gate), entry 1 (Stage 0/1 predictions), all before any work. The test sets are retired. | `c11e97a` |
| 2026-10-15 19:58–19:59 | SEAL | `explore`, `confirm_v2` (sealed) and `matched-energy` generated; non-overlap tests | `d8c94eb` |
| 2026-10-15 20:03 | PRED | Entry 2: readout-dimension correction, matched-energy check (before Stage 0) | `bf9be1a` |
| 2026-10-15 20:42–20:53 | RUN | Stage 0 (spatial maps), Stage 1 kill gate passed | `153f20c`, `4b9be90` |
| 2026-10-15 20:59, 21:01 | PRED | Entries 3–4: Stage 2 definitions and predictions; plateau counting | `d8c75fa`, `6207b9f` |
| 2026-10-15 21:13 | DEV | Stage 2 launch aborted: driver read the run list with Windows line endings; no training happened, and the runs restarted from scratch | LOG 21:13 |
| 2026-10-15 21:57–21:59 | DEV, AGREED | Paused at a projected ~25 GPU-h against an 18 cap; change 12: budget 26, protocol unchanged | LOG; `0cd168f` |
| 2026-10-16 11:33 | PRED, AGREED | Entry 5, change 13: 8 of 15 imaginer-family runs diverged at lr 3e-3. All were rerun at 1e-3; budget 28. Before any Stage 2 evaluation. | `33ba39b`, `7c41a5e` |
| 2026-10-16 16:39 | PRED | Entry 6: revised plan before any Stage 2 evaluation. The pixel + VICReg and fake-physics imaginers were dropped; render-target autoencoders and the imagination probe were added. | `63d27e9` |
| 2026-10-16 19:46 | PRED (PH) | Entry 7: contrast probe, designed after seeing the imagination probe (labelled post hoc) | `9114f97` |
| 2026-10-17 00:23 | — | Stage 2 summary with candidate Stage 3 hypotheses | `244f625` |
| 2026-10-17 03:16 | RUN | Stage 3 script dry run on `explore` (confirm still sealed) | LOG 03:16 |
| 2026-10-17 03:19 | AGREED | Change 14, before any Stage 3 training or unsealing: fresh seeds 3–5 for H1/H2; H4 (pixel + VICReg) added; budget 30. Supersedes "no retraining" (change 10). | `63e0ad2` |
| 2026-10-17 03:35–06:31 | RUN | 12 fresh autoencoders trained blind, none diverged; paused by the investigator | `f765960`; LOG 06:31 |
| 2026-10-17 16:13–17:21 | RUN | 3 pixel + VICReg runs trained blind, none diverged (the log at 06:31 had said "tomorrow"; they ran the same day) | `cbfe138` |
| 2026-10-17 17:31 | RUN | Confirmation script dry run on a stand-in config (Stage 2 models only; confirm sealed) | LOG 17:31 |
| 2026-10-17 17:32 | PREREG | `PREREGISTRATION_v2.md`, frozen script and config (39 checkpoint SHA-256 values) | `8797966` |
| 2026-10-17 17:33–17:48 | OPEN | `confirm_v2` unsealed and evaluated once. The log shows `8797966+dirty` because unsealing appends to `LOG.md` during the run. | LOG 17:33, 17:47 |
| 2026-10-17 17:49 | REPORT | `REPORT_v2.md`: partially supported (H1 holds, H2 fails) | `a88ddee` |

### Dial build (2026-10-17 to 10-25)

| when | type | event | commit / source |
|---|---|---|---|
| 2026-10-17 18:24 | AGREED, PRED | Build brief (verbatim), agreed changes 1–10 (budget 15 GPU-h), entry 1 (Stage 0), before any build code. `confirm_v2` retired. | `08c1511` |
| 2026-10-17 18:26 | SEAL | `confirm3` generated, sealed for `PREREGISTRATION_v3.md`; seals name their own pre-registration; disjointness test extended | `8e8bd1b` |
| 2026-10-17 18:32–18:40 | DEV | The Stage 0 harness ran three times. At 18:37, two diagnostics (cycle gain, watermark) were added after inspecting decoded changes; 18:40 was a formatting fix. Classification was identical in all three. | LOG 18:45; `ae98ffa` |
| 2026-10-17 | RUN | Stage 0 diagnostic: the absolute envelope test rejects ideal dials of the proposed strength | `474855f`; LOG 18:55 |
| 2026-10-18 00:28 | PRED, AGREED | Entries 2–3, approved before any Stage 1 training: <br>• effect floor raised 0.1 → 0.2 after Stage 0; <br>• envelope redefined relative to an ideal dial; <br>• cycle gain added; <br>• head-consistency absolute bar dropped by the investigator. | `5ef729a` |
| 2026-10-18 00:33 | DEV | Stage 1 readout script fix (titles, detach); no definitions changed | `4f08bf3` |
| 2026-10-18 02:59 | RUN | Stage 1 (M1-spec, M1-G) | `4ae7259` |
| 2026-10-21 20:41 | PRED | Entry 4: M2-SF (variant approved by the investigator) | `fae76b7` |
| 2026-10-21 22:05 | RUN | Stage 2 (M2-SF) | `5ecd024` |
| 2026-10-22 17:49 | PRED | Entry 5: M2-SW, M2-SF ω-only (investigator's specification) | `087866c` |
| 2026-10-22 18:15–18:20 | DEV | M2-SW runs crashed at start-up (a non-deterministic CUDA median), before any training step. The median was moved to the CPU with the same value, and the runs were rerun unchanged. | `f497d6d`; LOG 18:20 |
| 2026-10-22 19:42 | RUN | Stage 2b | `c9c3e6c` |
| 2026-10-22 23:13 | PRED, DEV | Entry 6, investigator's specification: <br>• grid-artefact criterion, calibrated after seeing data on one known-bad case (M2-SW seed 1); <br>• applied from then on and retroactively (M2-SF's loudness dials withdrawn); <br>• effect-space SVD dropped. | `2382399` |
| 2026-10-22 23:26–23:32 | RUN, PRED | Step A; entry 7; Step A′ | `f40a279`, `e4448b0`, `6c9e29b` |
| 2026-10-22 23:40 | PRED, AGREED | Entry 8, investigator's specification: <br>• the brief's Stage 3 (VICReg-on-dials dose curve) replaced by a content-side shape-noise dose; <br>• PC2 primary and frozen for exploration; <br>• loudness-null a secondary rule. | `afd31a1` |
| 2026-10-23 ~03:41 | DEV | Power outage during `b3_sn_t0_s2` (~0.3 GPU-h lost); rerun from scratch, deterministic | `00a79a9`; LOG |
| 2026-10-23 06:20 | RUN | Step B (12 runs) | `5234145` |
| 2026-10-23 21:53 | PRED, AGREED | Entry 9: budget 15 → 18 GPU-h (investigator); decoder rule | `fa7e95d` |
| 2026-10-24 00:37 | RUN | Step C (resize-convolution, τ = 0.5) | `f0f6c1f` |
| 2026-10-25 01:39 | PRED, AGREED | Entry 10: budget cap 18.5; the v3 dose rule (investigator's proposal), before any τ = 0.2 training | `a7e4b32` |
| 2026-10-25 04:40 | RUN | Step D (τ = 0.2) | `6a3420d` |
| 2026-10-25 04:42 | REPORT | `BUILD_REPORT.md` (exploratory) with a draft v3 for the investigator's decision. Two departures were proposed openly: <br>• loudness-null (chosen after seeing results) as the primary shape rule, with PC2 secondary; <br>• the brief's rendering trigger fires and is not applied. | `70ec339` |

### v3 confirmation and autopsy (2026-10-27 to 10-29)

| when | type | event | commit / source |
|---|---|---|---|
| 2026-10-27 | AGREED | Investigator's decisions: 5 main seeds; τ = 0 control with the resize decoder (3 new runs); random-FiLM untrained control (3); a new sealed matched-energy set; pass ≥ 4 of 5; budget 10 GPU-h | `PREREGISTRATION_v3.md` §2, §5, §8 |
| 2026-10-27 20:05 | SEAL | `matched-energy-confirm` generated and sealed for `PREREGISTRATION_v3.md`; v3 configs, untrained-control builder and Stage 5 script committed | LOG 20:05; `26921ea`, `525b5e0` (20:14) |
| 2026-10-27 | RUN | Stage 5 dry run on stand-ins (exploration sets, 64 frames, shortened search); no sealed data, no v3 model existed | `PREREGISTRATION_v3.md` §6 |
| 2026-10-27 20:16:40 | PREREG | `PREREGISTRATION_v3.md` committed | `781b2ad` |
| 2026-10-27 20:16–20:17 | RUN | Untrained controls built (`781b2ad+dirty`) | LOG; `2733e33` |
| 2026-10-27 ~20:40, ~21:57 | DEV | Two power outages during the first training run (≈ 0.4 and ≤ 0.1 GPU-h lost); nothing written, rerun from scratch; sealed sets untouched | `2733e33`, `d8312f7`; LOG |
| 2026-10-27 22:14 | DEV | Training paused by the investigator (expected power cut); ≈ 0.2 GPU-h discarded. Total lost ≈ 0.7; no partial checkpoint was used. | `0433882` |
| 2026-10-29 00:56–07:08 | RUN | 5 main and 3 τ = 0 models trained, none diverged, not evaluated | LOG; `REPORT_v3.md` |
| 2026-10-29 07:12 | — | 11 checkpoint SHA-256 values frozen | `fc3b703` |
| 2026-10-29 16:52–18:48 | OPEN | `confirm3` and `matched-energy-confirm` unsealed and scored once. The log shows `fc3b703+dirty`, caused only by the unsealing log append; the frozen file and every checksum were verified. | LOG 16:52, 18:48 |
| 2026-10-29 18:49 | — | Stage 5 outputs committed as written | `258d6af` |
| 2026-10-29 18:51 | REPORT | `REPORT_v3.md`: partially supported (3 of 5); no amendments | `aff659d` |
| 2026-10-29 19:18 | PRED (PH) | Entry 11: exploratory autopsy procedure and predictions, before any computation | `1cbc2e7` |
| 2026-10-29 | DEV | The investigator asked for item 4 (seed 103 without mean removal) to be rerun. Rerunning it on `confirm3` would reopen a sealed set, so it was done on the exploration sets instead (stated in entry 11). | `1cbc2e7`; `AUTOPSY.md` |
| 2026-10-29 19:21 | — | Autopsy script; a 37 s truncated code check (seed 102) wrote no outputs | `195a4c3`; `AUTOPSY.md` |
| 2026-10-29 19:22–21:38 | RUN | Autopsy (exploration data only; 2.27 GPU-h, outside the v3 budget) | `3c55d66` |
| 2026-10-29 21:43 | REPORT | `AUTOPSY.md` (post hoc; the verdict is unchanged) | `41b1213` |
| 2026-10-30 | — | Investigator closes the confirmatory line: no further training or analysis | this pack |

## 4. Budgets

| line | initial | changes | used | src |
|---|---|---|---|---|
| Study 1 | — | — | 11.5 h logged | `REPORT.md` §9 |
| Exploratory line + v2 | 14 (brief) → 18 (change 11) | 26 (change 12), 28 (change 13), 30 (change 14) | 29.6 | `results/explore/BRIEF.md`; `REPORT_v2.md` |
| Build | 15 (change 10) | 18 (entry 9), 18.5 (entry 10) | ≈ 17.8 | `results/build/BRIEF.md`; `PREDICTIONS.md`; `BUILD_REPORT.md` |
| v3 | 10 | none | ≈ 8.8 | `PREREGISTRATION_v3.md` §8; `REPORT_v3.md` |
| Autopsy | ≈ 2 (estimate, separate) | — | 2.27 | entry 11; `AUTOPSY.md` |

Every raise was the investigator's decision and was recorded before the work it covered.
