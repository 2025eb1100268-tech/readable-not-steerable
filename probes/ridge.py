"""Ridge linear probes (primary metric) with optional time partialling.

Pre-registered protocol:
- Features z-scored with statistics of the probe-fit (train) set; target centred on its train mean.
- Ridge strength alpha chosen on the validation set from ALPHAS (the only label-based selection).
- Plain R^2 on an evaluation set: 1 - SSE / sum (y - mean(y_eval))^2.
- Time-partialled R^2: targets and features are residualized on a one-hot of the frame index
  (i.e. minus per-frame-index means computed on the train set), the ridge is fitted on the train
  residuals, and on the evaluation set
      R^2_partial = 1 - SSE(time-only + latent) / SSE(time-only),
  where the time-only model predicts the train per-frame-index mean. This is the variance not
  explained by "how late is it" that the latent explains.
"""
from __future__ import annotations

import numpy as np

ALPHAS = np.logspace(-3, 7, 41)


def _frame_means(values: np.ndarray, frames: np.ndarray, n_index: int) -> np.ndarray:
    values2 = values.reshape(len(values), -1)
    sums = np.zeros((n_index, values2.shape[1]))
    np.add.at(sums, frames, values2)
    counts = np.bincount(frames, minlength=n_index).astype(float)
    glob = values2.mean(0)
    means = np.where(counts[:, None] > 0, sums / np.maximum(counts, 1)[:, None], glob[None])
    return means.reshape((n_index,) + values.shape[1:])


class RidgeProbe:
    def __init__(self, alphas=ALPHAS, partial: bool = False, n_frame_index: int = 64):
        self.alphas = np.asarray(alphas, float)
        self.partial = partial
        self.n_index = n_frame_index

    def _prep(self, x, frames):
        if self.partial:
            x = x - self.mx[frames]
        return (x - self.mu) / self.sd

    def _time(self, frames, n):
        return self.my[frames] if self.partial else np.full(n, self.ybar)

    def fit(self, x_tr, y_tr, f_tr, x_va, y_va, f_va):
        """Prediction = baseline (train mean, or train per-frame-index mean if partial)
        + ridge on z-scored (residualized) features fitted to y - baseline."""
        x_tr = np.asarray(x_tr, float)
        y_tr = np.asarray(y_tr, float)
        self.ybar = float(y_tr.mean())
        if self.partial:
            self.mx = _frame_means(x_tr, f_tr, self.n_index)
            self.my = _frame_means(y_tr, f_tr, self.n_index)
            xr = x_tr - self.mx[f_tr]
        else:
            xr = x_tr
        yr = y_tr - self._time(f_tr, len(y_tr))
        self.mu, self.sd = xr.mean(0), xr.std(0) + 1e-12
        xs = (xr - self.mu) / self.sd
        s, v = np.linalg.eigh(xs.T @ xs)
        proj = v.T @ (xs.T @ yr)
        coefs = v @ (proj[:, None] / (s[:, None] + self.alphas[None, :]))  # (d, n_alpha)
        pred_va = self._time(f_va, len(y_va))[:, None] + self._prep(np.asarray(x_va, float), f_va) @ coefs
        sse = ((np.asarray(y_va, float)[:, None] - pred_va) ** 2).sum(0)
        self.best = int(np.argmin(sse))
        self.alpha = float(self.alphas[self.best])
        self.coef = coefs[:, self.best]
        self.alpha_at_edge = self.best in (0, len(self.alphas) - 1)
        return self

    def predict(self, x, frames):
        x = np.asarray(x, float)
        return self._time(frames, len(x)) + self._prep(x, frames) @ self.coef

    def score(self, x, y, frames) -> dict:
        y = np.asarray(y, float)
        pred = self.predict(x, frames)
        out = {"r2": float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())}
        if self.partial:
            base = self._time(frames, len(y))
            out["r2_partial"] = float(1 - ((y - pred) ** 2).sum() / ((y - base) ** 2).sum())
        return out


def bootstrap_sd(y, pred, base, traj, n_boot: int = 200, seed: int = 0) -> float:
    """Trajectory-level bootstrap SD of 1 - SSE(pred)/SSE(base)."""
    rng = np.random.default_rng(seed)
    ids, inv = np.unique(traj, return_inverse=True)
    sse_p = np.bincount(inv, weights=(y - pred) ** 2)
    sse_b = np.bincount(inv, weights=(y - base) ** 2)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(ids), len(ids))
        vals.append(1 - sse_p[pick].sum() / sse_b[pick].sum())
    return float(np.std(vals))
