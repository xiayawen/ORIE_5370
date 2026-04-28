"""Predictive models used inside the IPO / OLS pipelines.

Each model implements the same minimal interface:

    model = ModelClass(d_in, ...)
    yhat = model(X)        # X: (n, d) torch tensor -> (n,) tensor
    list(model.parameters())  # for gradient-based IPO training

For OLS-style training we expose ``fit_ols(model, X, y)`` which sets the model
parameters in closed form (or by sklearn) so that downstream "predict-then-
optimize" benchmarks share the same predictor classes as IPO.

Models implemented:

* ``LinearPredictor``     — ``f(x) = W x + b``
* ``PolynomialPredictor`` — ``f(x) = W φ(x) + b``, with ``φ`` adding pairwise
                            interactions and squared terms
* ``RidgePredictor``      — same as Linear, but trained with L2 regularisation
* ``KernelRidgePredictor``— ``f(x) = Σ α_i K(x, x_i)`` with RBF kernel
                            (anchor points are a random subsample of training
                            features, fixed up-front so the model is finite-
                            dim and differentiable)
* ``MLPPredictor``        — single hidden layer NN with tanh activation

All models output a 1-D tensor of length ``n`` representing predicted next-
period asset returns for the cross-section.
"""

from __future__ import annotations

from typing import Optional
import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Feature transforms (used by Polynomial / Kernel predictors)
# ---------------------------------------------------------------------------

def polynomial_features(x: torch.Tensor, degree: int = 2, interactions: bool = True) -> torch.Tensor:
    """Append squared and (optionally) interaction terms to ``x``.

    Output dimension: ``d`` (linear) + ``d`` (squared) + ``d*(d-1)/2``
    (interactions, if enabled).
    """
    d = x.shape[-1]
    feats = [x]
    if degree >= 2:
        feats.append(x * x)
        if interactions and d > 1:
            i_idx, j_idx = torch.triu_indices(d, d, offset=1)
            feats.append(x[..., i_idx] * x[..., j_idx])
    return torch.cat(feats, dim=-1)


def rbf_kernel(X: torch.Tensor, Z: torch.Tensor, gamma: float) -> torch.Tensor:
    """Gaussian RBF kernel matrix ``K[i, j] = exp(-gamma * ||X_i - Z_j||^2)``."""
    XX = (X * X).sum(-1, keepdim=True)
    ZZ = (Z * Z).sum(-1, keepdim=True).T
    sq = XX + ZZ - 2 * X @ Z.T
    return torch.exp(-gamma * sq.clamp(min=0))


# ---------------------------------------------------------------------------
# Predictor classes
# ---------------------------------------------------------------------------

class LinearPredictor(nn.Module):
    def __init__(self, d_in: int):
        super().__init__()
        self.linear = nn.Linear(d_in, 1, bias=True)
        # Initialize small to avoid extreme initial portfolio weights.
        nn.init.zeros_(self.linear.bias)
        nn.init.normal_(self.linear.weight, std=0.01)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.linear(X).squeeze(-1)


class PolynomialPredictor(nn.Module):
    def __init__(self, d_in: int, degree: int = 2, interactions: bool = True):
        super().__init__()
        self.degree = degree
        self.interactions = interactions
        with torch.no_grad():
            d_out = polynomial_features(torch.zeros(1, d_in), degree, interactions).shape[-1]
        self.linear = nn.Linear(d_out, 1, bias=True)
        nn.init.zeros_(self.linear.bias)
        nn.init.normal_(self.linear.weight, std=0.01)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.linear(polynomial_features(X, self.degree, self.interactions)).squeeze(-1)


class RidgePredictor(LinearPredictor):
    """Identical to ``LinearPredictor``; the L2 penalty is applied at training time."""


class KernelRidgePredictor(nn.Module):
    """Finite-dimensional kernel ridge regressor.

    ``anchors`` (a fixed random subsample of training rows, of size ``M``) are
    set once when the model is constructed. The trainable parameters are the
    ``M`` mixture coefficients ``α``.
    """

    def __init__(self, anchors: torch.Tensor, gamma: float = 1.0):
        super().__init__()
        self.register_buffer("anchors", anchors.clone())
        self.gamma = float(gamma)
        self.alpha = nn.Parameter(torch.zeros(anchors.shape[0]))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        K = rbf_kernel(X, self.anchors, self.gamma)
        return K @ self.alpha + self.bias


class MLPPredictor(nn.Module):
    """Single hidden layer NN. Matches the project outline's specification."""

    def __init__(self, d_in: int, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.net(X).squeeze(-1)


# ---------------------------------------------------------------------------
# OLS fitting (closed form / sklearn) for the predict-then-optimize baseline
# ---------------------------------------------------------------------------

def stack_panel(X_list: list[np.ndarray], y_list: list[np.ndarray]):
    """Stack a list of per-month (X_t, y_t) pairs into one big regression."""
    return np.vstack(X_list), np.concatenate(y_list)


def fit_ols(model: nn.Module, X_list: list[np.ndarray], y_list: list[np.ndarray], alpha: float = 0.0):
    """Fit ``model`` parameters in closed form using stacked OLS / Ridge.

    Supports ``LinearPredictor``, ``RidgePredictor`` and ``PolynomialPredictor``.
    For ``KernelRidgePredictor`` and ``MLPPredictor`` the closed-form OLS does
    not apply; in those cases we fall back to a small sklearn-style optimizer.
    """
    X, y = stack_panel(X_list, y_list)

    if isinstance(model, (LinearPredictor, RidgePredictor)):
        d = X.shape[1]
        Xb = np.hstack([X, np.ones((X.shape[0], 1))])
        A = Xb.T @ Xb + alpha * np.block([
            [np.eye(d), np.zeros((d, 1))],
            [np.zeros((1, d)), np.zeros((1, 1))],
        ])
        b = Xb.T @ y
        theta = np.linalg.solve(A, b)
        with torch.no_grad():
            model.linear.weight.copy_(torch.tensor(theta[:d], dtype=model.linear.weight.dtype).reshape(1, d))
            model.linear.bias.copy_(torch.tensor([theta[d]], dtype=model.linear.bias.dtype))
        return

    if isinstance(model, PolynomialPredictor):
        Xt = torch.tensor(X, dtype=torch.float32)
        Phi = polynomial_features(Xt, model.degree, model.interactions).numpy()
        d = Phi.shape[1]
        Xb = np.hstack([Phi, np.ones((Phi.shape[0], 1))])
        A = Xb.T @ Xb + alpha * np.block([
            [np.eye(d), np.zeros((d, 1))],
            [np.zeros((1, d)), np.zeros((1, 1))],
        ])
        b = Xb.T @ y
        theta = np.linalg.solve(A, b)
        with torch.no_grad():
            model.linear.weight.copy_(torch.tensor(theta[:d], dtype=model.linear.weight.dtype).reshape(1, d))
            model.linear.bias.copy_(torch.tensor([theta[d]], dtype=model.linear.bias.dtype))
        return

    if isinstance(model, KernelRidgePredictor):
        Xt = torch.tensor(X, dtype=model.anchors.dtype)
        K = rbf_kernel(Xt, model.anchors, model.gamma).numpy()
        # Add bias column.
        Kb = np.hstack([K, np.ones((K.shape[0], 1))])
        d = K.shape[1]
        A = Kb.T @ Kb + alpha * np.block([
            [np.eye(d), np.zeros((d, 1))],
            [np.zeros((1, d)), np.zeros((1, 1))],
        ])
        b = Kb.T @ y
        theta = np.linalg.solve(A, b)
        with torch.no_grad():
            model.alpha.copy_(torch.tensor(theta[:d], dtype=model.alpha.dtype))
            model.bias.copy_(torch.tensor([theta[d]], dtype=model.bias.dtype))
        return

    if isinstance(model, MLPPredictor):
        # No closed form: fall back to plain MSE training. This is the
        # standard "predict-then-optimize" baseline for the NN predictor.
        Xt = torch.tensor(X, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.float32)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=alpha)
        loss_fn = torch.nn.MSELoss()
        for _ in range(2000):
            opt.zero_grad()
            pred = model(Xt)
            loss = loss_fn(pred, yt)
            loss.backward()
            opt.step()
        return

    raise TypeError(f"fit_ols: unsupported model {type(model).__name__}")


def make_kernel_anchors(X_list: list[np.ndarray], n_anchors: int = 200, seed: int = 0) -> torch.Tensor:
    """Pick a random subsample of training rows to serve as RBF kernel anchors.

    Anchors are drawn uniformly across all (date, asset) pairs in the training
    set so that the kernel basis covers the input distribution.
    """
    X = np.vstack(X_list)
    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[0], size=min(n_anchors, X.shape[0]), replace=False)
    return torch.tensor(X[idx], dtype=torch.float32)
