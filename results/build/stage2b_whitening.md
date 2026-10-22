# [BUILD] Stage 2b: whitening check for M2-SW's content input (before any Stage 2b training; exploratory)

4096 explore frames (seed 0); reference = median RMS-normalized shell power of the data/generated/v1 train frames train/train.py samples. Cap G_max = 1000.

## Cap justification
- Noise floor, empirical: the power the stored fields carry where the solver puts none (shells k ≥ 31, beyond the square dealiasing mask's corners), per mode, scaled to each shell's mode count; 99th percentile over frames. It lies above simulated float32 rounding (uniform ±½ ulp of each stored value) by a median factor of 2, so it is the one used (the larger of the two per shell). Floor / ref_k: k=1 6.3e-18, k=10 3.6e-13, k=15 2.0e-11, k=21 9.7e-10.
- Largest gain that keeps the amplified floor ≤ 1% of ref_k: min over k = 3212 (k = 21). The cap 1000 keeps it ≤ 0.10% of ref_k in every shell.
- Required gains (to reach ref_k) over explore: median 0.96; 99th percentile per shell up to 53268; maximum 1.6e+06 (shells already at the storage floor).

## What the cap costs (shape leakage into content)
- Capped (frame, shell) pairs: 1.22%; frames with any capped shell: 5.6% (the smoothest viscous frames, high k only). Per shell k = 10, 15, 18, 21: 0.3%, 1.3%, 2.9%, 5.6%.

## Artefact checks on the whitened inputs
- Power beyond k = 21 / power in shells 1–21: max over frames 7.1e-32 (zeroed by construction).
- Uncapped shells: whitened power equals ref_k to within 1.3e-13 (relative).
- Amplified noise floor / whitened power, per frame: median 7.1e-16, max 5.8e-10.
- Scale invariance: max |W(3.7 x) − W(x)| / max |W(x)| = 2.9e-05.

ref_k (k = 1..21): 0.929, 0.311, 0.182, 0.0802, 0.0115, 0.00466, 0.0015, 0.000687, 0.000358, 0.000112, 6.72e-05, 2.83e-05, 1.63e-05, 7.1e-06, 3.09e-06, 2.05e-06, 9.53e-07, 4.7e-07, 2.58e-07, 1.33e-07, 1.08e-07
