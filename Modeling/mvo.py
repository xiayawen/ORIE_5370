"""Mean-variance optimization helpers used by both the OLS and IPO pipelines.

We work in three regimes that all admit closed-form, differentiable solutions
(so PyTorch can backprop through them without a QP layer):

* ``unconstrained``     : ``S = R^n``, ``z*(y) = (1/δ) V^{-1} y``
* ``equality``          : ``S = {z : 1^T z = 1}`` (fully invested)
* ``equality_neutral``  : ``S = {z : 1^T z = 0}`` (market neutral)

For the inequality-constrained "long-only with caps" regime we follow Butler &
Kwon §3.3 and use the heuristic of solving the analytical equality-constrained
problem and then clipping (still differentiable approximately via
straight-through, but we evaluate it as a plain in-sample → out-of-sample post
processing step).

All functions accept either ``np.ndarray`` or ``torch.Tensor`` inputs and
dispatch accordingly.
"""

from __future__ import annotations

from typing import Tuple
import numpy as np

try:
    import torch
    _TORCH = True
except Exception:  # pragma: no cover
    _TORCH = False


# ---------------------------------------------------------------------------
# numpy implementations
# ---------------------------------------------------------------------------

def mvo_unconstrained_np(y: np.ndarray, V: np.ndarray, delta: float) -> np.ndarray:
    return np.linalg.solve(V, y) / delta


def mvo_equality_np(y: np.ndarray, V: np.ndarray, delta: float, sum_to: float = 1.0) -> np.ndarray:
    """Solve  min -z^T y + (delta/2) z^T V z  s.t.  1^T z = sum_to.

    Closed form via Lagrangian:
        z* = (1/delta) V^{-1} (y - lambda * 1),
        lambda = (1^T V^{-1} y - delta * sum_to) / (1^T V^{-1} 1)
    """
    n = y.shape[0]
    ones = np.ones(n)
    Vinv_y = np.linalg.solve(V, y)
    Vinv_1 = np.linalg.solve(V, ones)
    lam = (ones @ Vinv_y - delta * sum_to) / (ones @ Vinv_1)
    return (Vinv_y - lam * Vinv_1) / delta


def mvo_cost_np(z: np.ndarray, y: np.ndarray, V: np.ndarray, delta: float) -> float:
    return float(-z @ y + 0.5 * delta * z @ V @ z)


# ---------------------------------------------------------------------------
# torch implementations  (differentiable in y; V, delta treated as constants)
# ---------------------------------------------------------------------------

if _TORCH:

    def mvo_unconstrained_t(y: "torch.Tensor", V: "torch.Tensor", delta: float) -> "torch.Tensor":
        return torch.linalg.solve(V, y.unsqueeze(-1)).squeeze(-1) / delta

    def mvo_equality_t(
        y: "torch.Tensor", V: "torch.Tensor", delta: float, sum_to: float = 1.0
    ) -> "torch.Tensor":
        n = y.shape[-1]
        ones = torch.ones(n, dtype=y.dtype, device=y.device)
        # Stack RHS so we do one solve.
        rhs = torch.stack([y, ones], dim=-1)            # (n, 2)
        sol = torch.linalg.solve(V, rhs)                 # (n, 2)
        Vinv_y = sol[..., 0]
        Vinv_1 = sol[..., 1]
        lam = (ones @ Vinv_y - delta * sum_to) / (ones @ Vinv_1)
        return (Vinv_y - lam * Vinv_1) / delta

    def mvo_cost_t(
        z: "torch.Tensor", y: "torch.Tensor", V: "torch.Tensor", delta: float
    ) -> "torch.Tensor":
        # z, y: (n,) ; V: (n, n)
        return -(z @ y) + 0.5 * delta * (z @ V @ z)


# ---------------------------------------------------------------------------
# Convenience: dispatch based on input type
# ---------------------------------------------------------------------------

def mvo_solve(y, V, delta: float, region: str = "equality", sum_to: float = 1.0):
    """Dispatch to the right backend based on input type.

    ``region`` is one of ``unconstrained``, ``equality`` (long+short, fully
    invested) or ``equality_neutral`` (sum-to-zero, market neutral).
    """
    is_torch = _TORCH and isinstance(y, torch.Tensor)
    if region == "unconstrained":
        return mvo_unconstrained_t(y, V, delta) if is_torch else mvo_unconstrained_np(y, V, delta)
    if region == "equality":
        return (
            mvo_equality_t(y, V, delta, sum_to=sum_to)
            if is_torch
            else mvo_equality_np(y, V, delta, sum_to=sum_to)
        )
    if region == "equality_neutral":
        return (
            mvo_equality_t(y, V, delta, sum_to=0.0)
            if is_torch
            else mvo_equality_np(y, V, delta, sum_to=0.0)
        )
    raise ValueError(f"unknown region {region!r}")


def mvo_cost(z, y, V, delta: float):
    is_torch = _TORCH and isinstance(z, torch.Tensor)
    return mvo_cost_t(z, y, V, delta) if is_torch else mvo_cost_np(z, y, V, delta)
