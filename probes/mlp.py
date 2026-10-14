"""Secondary MLP probes (PREREGISTRATION.md §5.2): one hidden layer of 256 GELU units on the
same z-scored, time x family residualized latents as the ridge probes; AdamW, early stopping on
val. Reported next to the linear probes; never used for the pass/fail criteria."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from data.dataset import VISCOUS_ONLY, Dataset
from probes.ridge import _frame_means


def mlp_probe(ds: Dataset, lat: np.ndarray, name: str, eval_split: str, seed: int = 0, device: str = "cuda",
              max_epochs: int = 200, patience: int = 10) -> float:
    y = ds.target(name)
    tr, va, ev = ds.rows(["train"]), ds.rows(["val"]), ds.rows([eval_split])
    if name in VISCOUS_ONLY:
        tr, va, ev = (r[~ds.inviscid[r]] for r in (tr, va, ev))
    pidx, n_idx = ds.partial_index(), 2 * ds.n_frame_max
    mx, my = _frame_means(lat[tr].astype(float), pidx[tr], n_idx), _frame_means(y[tr], pidx[tr], n_idx)
    res = lambda r: (lat[r] - mx[pidx[r]], y[r] - my[pidx[r]])  # noqa: E731
    (xtr, ytr), (xva, yva), (xev, yev) = res(tr), res(va), res(ev)
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-12
    ys = ytr.std()
    t = lambda a: torch.tensor(a, dtype=torch.float32, device=device)  # noqa: E731
    xtr_t, xva_t, xev_t = t((xtr - mu) / sd), t((xva - mu) / sd), t((xev - mu) / sd)
    ytr_t, yva_t = t(ytr / ys), t(yva / ys)
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(lat.shape[1], 256), nn.GELU(), nn.Linear(256, 1)).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    gen = torch.Generator(device=device).manual_seed(seed)
    best, best_state, bad = np.inf, None, 0
    for _ in range(max_epochs):
        perm = torch.randperm(len(xtr_t), generator=gen, device=device)
        for i in range(0, len(perm), 2048):
            b = perm[i:i + 2048]
            loss = ((net(xtr_t[b]).squeeze(1) - ytr_t[b]) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        with torch.no_grad():
            v = float(((net(xva_t).squeeze(1) - yva_t) ** 2).mean())
        if v < best - 1e-6:
            best, best_state, bad = v, {k: p.clone() for k, p in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    with torch.no_grad():
        pred = net(xev_t).squeeze(1).cpu().numpy() * ys
    return float(1 - ((yev - pred) ** 2).sum() / (yev**2).sum())
