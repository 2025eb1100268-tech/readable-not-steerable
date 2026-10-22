"""[BUILD] Stage 2b: effect-space identification of dials (secondary method; PREDICTIONS entry 5). Exploratory.

J = the Jacobian of the decoded field's log shell energy spectrum (vorticity field, shells k = 1..21)
with respect to the raw dial vector, averaged over frames; its SVD's right-singular vectors are
directions in dial space ordered by how much they change the output spectrum per unit of dial
movement. No labels and no data-variance ordering: knobs are found by what they do to the output.
The primary, pre-registered method stays PCA of the dial vector.
"""
from __future__ import annotations

import numpy as np
import torch

from dials import fields
from solver.ns2d import SpectralGrid


def effect_space(model, x_t: torch.Tensor, n_top: int = 4) -> dict:
    """model: a branch_ae Model (dial vector -> one rendered omega). x_t: scaled input frames."""
    h = model.head
    with torch.no_grad():
        content, g0 = model.enc(x_t.float()), h.dial(x_t.float())
    g = g0.detach().clone().requires_grad_(True)
    grid = SpectralGrid(x_t.shape[-1], x_t.device)
    w = h.dec(content, g).float() * model.ae_scales["vorticity"]
    log_spec = torch.log(fields.shell_spectrum(fields.vorticity_hat(w, "vorticity", grid), grid))  # (B, K_MAX)
    jac = torch.stack([torch.autograd.grad(log_spec[:, k].sum(), g, retain_graph=k < log_spec.shape[1] - 1)[0].double().mean(0)
                       for k in range(log_spec.shape[1])])  # (K_MAX, n_dial): mean over frames of d log E_k / d g
    u, s, vh = torch.linalg.svd(jac, full_matrices=False)
    return {"jacobian": jac.cpu().numpy(), "singular_values": s.cpu().numpy(), "left": u.cpu().numpy()[:, :n_top],
            "directions": vh.cpu().numpy()[:n_top]}


def cos2(a: np.ndarray, b: np.ndarray) -> float:
    return float((a @ b) ** 2 / ((a @ a) * (b @ b)))
