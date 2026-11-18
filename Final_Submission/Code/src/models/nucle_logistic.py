"""Sparse multinomial Logistic Regression for NUCLE S/L/P/M/D features.

``fit_logistic`` fits on a sparse PyTorch COO matrix and returns learned
weights, intercept, and explicit LBFGS stopping diagnostics. Hyperparameter
selection and validation reporting stay in ``src/training/nucle_benchmark.py``.
"""
from __future__ import annotations

from collections import Counter

import torch
import torch.nn.functional as F

from src.data.constants import TARGET_LABELS

LABELS = tuple(TARGET_LABELS)
SEED = 20261013
LR_MAX_ITER = 1000
LR_MAX_EVAL = 1250
LR_TOLERANCE_GRAD = 1e-6
LR_TOLERANCE_CHANGE = 1e-9


def balanced_weights(y: list[int]) -> torch.Tensor:
    """Compute inverse-frequency weights using the fixed label order."""
    counts = Counter(y)
    return torch.tensor(
        [len(y) / (len(LABELS) * counts.get(i, 1)) for i in range(len(LABELS))],
        dtype=torch.float32,
    )


def fit_logistic(
    x: torch.Tensor,
    y: list[int],
    c_value: float,
    balanced: bool,
    max_iter: int = LR_MAX_ITER,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Fit sparse multinomial LR with L2 regularization and LBFGS.

    The objective is mean cross entropy plus ``||W||²/(2*C*n)``. The return
    values are coefficient matrix, intercept and stopping diagnostics.
    """
    if x.layout != torch.sparse_coo:
        raise ValueError("Logistic regression expects a sparse COO feature matrix")
    if x.shape[0] != len(y) or not y:
        raise ValueError("Feature rows and non-empty targets must be aligned")
    if c_value <= 0 or max_iter < 1:
        raise ValueError("C and max_iter must be positive")

    torch.manual_seed(SEED)
    n, d = x.shape
    labels = torch.tensor(y, dtype=torch.long)
    weights = balanced_weights(y) if balanced else None
    w = torch.nn.Parameter(torch.zeros(d, len(LABELS), dtype=torch.float32))
    b = torch.nn.Parameter(torch.zeros(len(LABELS), dtype=torch.float32))
    optimizer = torch.optim.LBFGS(
        [w, b], lr=1.0, max_iter=max_iter, max_eval=LR_MAX_EVAL,
        tolerance_grad=LR_TOLERANCE_GRAD,
        tolerance_change=LR_TOLERANCE_CHANGE,
        line_search_fn="strong_wolfe",
    )
    eval_count = 0

    def closure():
        nonlocal eval_count
        optimizer.zero_grad()
        logits = torch.sparse.mm(x, w) + b
        loss = F.cross_entropy(logits, labels, weight=weights, reduction="mean")
        objective = loss + 0.5 * w.square().sum() / (c_value * n)
        objective.backward()
        eval_count += 1
        return objective

    optimizer.step(closure)
    optimizer_iters = int(optimizer.state[w].get("n_iter", 0))
    optimizer.zero_grad()
    final_objective = F.cross_entropy(
        torch.sparse.mm(x, w) + b, labels, weight=weights
    ) + 0.5 * w.square().sum() / (c_value * n)
    final_objective.backward()
    grad_inf_norm = max(float(w.grad.abs().max()), float(b.grad.abs().max()))
    final_loss = float(final_objective.detach())
    optimizer.zero_grad()

    gradient_stop = grad_inf_norm <= LR_TOLERANCE_GRAD
    change_stop = optimizer_iters < max_iter and eval_count < LR_MAX_EVAL
    converged = gradient_stop or change_stop
    termination = (
        "gradient_tolerance" if gradient_stop else
        "change_tolerance" if change_stop else
        "iteration_limit" if optimizer_iters >= max_iter else
        "evaluation_limit"
    )
    diagnostics = {
        "iterations": optimizer_iters,
        "iterations_max": max_iter,
        "closure_evaluations": eval_count,
        "max_evaluations": LR_MAX_EVAL,
        "final_objective": final_loss,
        "gradient_inf_norm": grad_inf_norm,
        "tolerance_grad": LR_TOLERANCE_GRAD,
        "tolerance_change": LR_TOLERANCE_CHANGE,
        "termination": termination,
        "converged": converged,
    }
    return w.detach(), b.detach(), diagnostics
