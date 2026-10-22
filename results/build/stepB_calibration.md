# [BUILD] Step B: shape-noise calibration (before any Step B training; exploratory)

sigma1 (tilt) = 5.1589, sigma2 (curvature) = 2.3639: train-set std of the coefficients of each frame's log shell power on [1, phi1, phi2] (shells 1..21; the train frames train.py samples).

Calibrated dose: ridge from the content input's 21 log shell powers to s, fitted on 8610 v1 train frames (alpha on a held-out 20%), R^2 on 4939 v1 val frames, evaluation draws (hash-fixed per frame).

| tau | content shape R^2 |
|---|---|
| 0 | 0.966 |
| 0.5 | 0.294 |
| 1 | 0.181 |
| 2 | 0.066 |
