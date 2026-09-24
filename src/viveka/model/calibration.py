"""Post-hoc temperature scaling.

Given held-out logits z and labels y, find the scalar T > 0 minimising

    BCE(sigmoid(z / T), y)

This does not change the ranking of predictions (so accuracy / AUROC are
unchanged); it only stretches or squashes the probabilities so that a reported
0.9 actually means ~90% empirical frequency.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def fit_temperature(
    logits: np.ndarray | torch.Tensor,
    labels: np.ndarray | torch.Tensor,
    max_iter: int = 200,
    init: float = 1.0,
) -> float:
    z = torch.as_tensor(np.asarray(logits), dtype=torch.float64).flatten()
    y = torch.as_tensor(np.asarray(labels), dtype=torch.float64).flatten()
    if z.numel() == 0:
        return float(init)

    # Optimise log T so that T stays positive.
    log_t = nn.Parameter(torch.tensor(float(np.log(init)), dtype=torch.float64))
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(z / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    t = float(log_t.exp().item())
    if not np.isfinite(t) or t <= 0:
        return float(init)
    return t
