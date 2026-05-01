"""Train all (predictor × training-paradigm) combinations and persist them.

Two training paradigms are compared:

* **OLS plug-in** (``fit_ols``): fit predictor by least-squares on stacked
  cross-sections, then plug ``ŷ`` into the downstream MVO program.
* **IPO** (``train_ipo``): minimise the realized MVO cost
  ``(1/T) Σ_t [-z*(ŷ_t)^T y_t + (δ/2) z*(ŷ_t)^T V_t z*(ŷ_t)]`` end-to-end
  via PyTorch backprop, where ``z*`` is the closed-form equality-constrained
  solution from ``mvo.mvo_solve``.

This is the project's central comparison (linear vs. nonlinear, OLS vs. IPO).
We sweep validation hyperparameters (ridge λ, NN width, kernel γ) on the
validation period and pick the configuration with lowest realized MVO cost
on the validation set.

The trained model state-dicts are saved into ``results/models/`` so that
``evaluate.py`` can backtest them without re-training.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from build_dataset import OUT_DIR, load_panel, COV_METHOD
from ipo_models import (
    LinearPredictor, PolynomialPredictor, RidgePredictor,
    LassoPredictor, ElasticNetPredictor,
    KernelRidgePredictor, MLPPredictor,
    fit_ols, make_kernel_anchors,
)
from mvo import mvo_solve, mvo_cost


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
MODELS_DIR = RESULTS / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Globals chosen to match the project outline (§3.4 sample splitting),
# adjusted for the available data window (price cache starts in 2000).
TRAIN_END = np.datetime64("2017-01-01")
VAL_END = np.datetime64("2019-01-01")     # validation: 2017–2018
# test starts after VAL_END

DELTA = 50.0         # risk-aversion parameter; matches Butler & Kwon §4
REGION = "equality"  # 1^T z = 1 (fully invested portfolio)
SUM_TO = 1.0

DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# Data slicing
# ---------------------------------------------------------------------------

def slice_period(panel: dict, t0: np.datetime64, t1: np.datetime64) -> dict:
    dates = panel["dates"].to_numpy() if hasattr(panel["dates"], "to_numpy") else np.asarray(panel["dates"])
    mask = (dates >= t0) & (dates < t1)
    idx = np.flatnonzero(mask)
    return {
        "dates": dates[idx],
        "X": [panel["X"][i] for i in idx],
        "y": [panel["y"][i] for i in idx],
        "V": [panel["V"][i] for i in idx],
        "tickers_per_t": [panel["tickers_per_t"][i] for i in idx],
        "feature_names": panel["feature_names"],
    }


# ---------------------------------------------------------------------------
# Realized MVO cost evaluation (numpy, used as the validation metric)
# ---------------------------------------------------------------------------

def realized_mvo_cost(model: nn.Module, ds: dict, delta: float = DELTA, region: str = REGION) -> float:
    model.eval()
    costs = []
    with torch.no_grad():
        for X, y, V in zip(ds["X"], ds["y"], ds["V"]):
            Xt = torch.tensor(X, dtype=torch.float32, device=DEVICE)
            yhat = model(Xt).cpu().numpy().astype(np.float64)
            try:
                z = mvo_solve(yhat, V, delta, region=region, sum_to=SUM_TO)
            except np.linalg.LinAlgError:
                continue
            costs.append(mvo_cost(z, y, V, delta))
    return float(np.mean(costs))


# ---------------------------------------------------------------------------
# IPO training loop
# ---------------------------------------------------------------------------

def train_ipo(
    model: nn.Module,
    ds_train: dict,
    ds_val: dict,
    epochs: int = 60,
    lr: float = 5e-3,
    weight_decay: float = 0.0,
    l1_penalty: float = 0.0,
    delta: float = DELTA,
    region: str = REGION,
    patience: int = 10,
    verbose: bool = False,
) -> dict:
    """Decision-focused training using closed-form ``z*`` and PyTorch autograd.

    ``weight_decay`` is the L2 penalty applied via Adam (proportional to the
    full parameter vector). ``l1_penalty`` is the explicit L1 penalty applied
    only to the predictor's weight matrix (via the ``linear.weight`` parameter
    when present), used by ``LassoPredictor`` and ``ElasticNetPredictor``.

    Returns a dict with the best validation cost and a list of (epoch, train,
    val) curves for inspection.
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Resolve the weight tensor used for the L1 penalty (linear / ridge / lasso /
    # elastic-net all expose .linear.weight; for kernel/NN we leave l1=0).
    weight_param = getattr(getattr(model, "linear", None), "weight", None)

    # Pre-cast tensors once.
    Xtr = [torch.tensor(X, dtype=torch.float32, device=DEVICE) for X in ds_train["X"]]
    ytr = [torch.tensor(y, dtype=torch.float32, device=DEVICE) for y in ds_train["y"]]
    Vtr = [torch.tensor(V, dtype=torch.float32, device=DEVICE) for V in ds_train["V"]]

    best_val = float("inf")
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    history = []
    bad_epochs = 0

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        running = 0.0
        n_seen = 0
        for i in perm.tolist():
            X, y, V = Xtr[i], ytr[i], Vtr[i]
            opt.zero_grad()
            yhat = model(X)
            try:
                z = mvo_solve(yhat, V, delta, region=region, sum_to=SUM_TO)
            except RuntimeError:
                continue
            loss = mvo_cost(z, y, V, delta)
            if l1_penalty > 0 and weight_param is not None:
                loss = loss + l1_penalty * weight_param.abs().sum()
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            running += float(loss.detach())
            n_seen += 1

        train_cost = running / max(n_seen, 1)
        val_cost = realized_mvo_cost(model, ds_val, delta=delta, region=region)
        history.append({"epoch": epoch, "train": train_cost, "val": val_cost})

        if val_cost < best_val - 1e-6:
            best_val = val_cost
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        if verbose:
            print(f"  epoch {epoch:3d}  train={train_cost:+.4f}  val={val_cost:+.4f}")
        if bad_epochs >= patience:
            break

    model.load_state_dict(best_state)
    return {"best_val": best_val, "history": history}


# ---------------------------------------------------------------------------
# Model factories — match the predictor table in the project outline
# ---------------------------------------------------------------------------

def model_factory(name: str, d_in: int, anchors: torch.Tensor | None = None,
                  hidden: int = 16, gamma: float = 1.0, degree: int = 2) -> nn.Module:
    if name == "linear":
        return LinearPredictor(d_in)
    if name == "polynomial":
        return PolynomialPredictor(d_in, degree=degree, interactions=True)
    if name == "ridge":
        return RidgePredictor(d_in)
    if name == "lasso":
        return LassoPredictor(d_in)
    if name == "elasticnet":
        return ElasticNetPredictor(d_in)
    if name == "kernel":
        if anchors is None:
            raise ValueError("kernel model requires anchors")
        return KernelRidgePredictor(anchors, gamma=gamma)
    if name == "nn":
        return MLPPredictor(d_in, hidden=hidden)
    raise ValueError(name)


# ---------------------------------------------------------------------------
# Hyperparameter sweeps for each model family
# ---------------------------------------------------------------------------

OLS_RIDGE_ALPHAS = [0.0, 1.0, 10.0, 100.0, 1000.0]
OLS_LASSO_ALPHAS = [1e-4, 1e-3, 1e-2, 1e-1]
OLS_ELASTICNET_L1RATIO = [0.25, 0.5, 0.75]      # convex combinations of L1/L2
IPO_RIDGE_WD = [0.0, 1e-4, 1e-3, 1e-2]
IPO_LASSO_L1 = [1e-5, 1e-4, 1e-3]
IPO_ELASTICNET = [(1e-4, 1e-4), (1e-3, 1e-4)]    # (l1_penalty, weight_decay)
IPO_NN_HIDDEN = [8, 16, 32]
KERNEL_GAMMA = [0.1, 0.5, 1.0]
KERNEL_M = 200

# Order in which we run models. Linear / ridge / lasso / elasticnet first
# because their training is fast; kernel / nn last. The same ordering is used
# in evaluate.py and make_figures.py for consistent column / row layout.
MODEL_FAMILIES = ["linear", "ridge", "lasso", "elasticnet", "polynomial", "kernel", "nn"]


def run_all(panel: dict, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    ds_train = slice_period(panel, panel["dates"][0], TRAIN_END)
    ds_val = slice_period(panel, TRAIN_END, VAL_END)
    ds_test = slice_period(panel, VAL_END, panel["dates"][-1] + np.timedelta64(1, "D"))

    n_tr, n_va, n_te = len(ds_train["X"]), len(ds_val["X"]), len(ds_test["X"])
    print(f"sample split: train={n_tr}  val={n_va}  test={n_te}  (months)")

    d_in = ds_train["X"][0].shape[1]
    anchors = make_kernel_anchors(ds_train["X"], n_anchors=KERNEL_M, seed=seed)

    summary: dict[str, dict] = {}

    # ----------- OLS plug-in baselines -----------
    for name in MODEL_FAMILIES:
        if name == "ridge":
            best = None
            for alpha in OLS_RIDGE_ALPHAS:
                m = model_factory(name, d_in)
                fit_ols(m, ds_train["X"], ds_train["y"], alpha=alpha)
                v = realized_mvo_cost(m, ds_val)
                if best is None or v < best[1]:
                    best = (alpha, v, m)
            alpha, _, m = best
            summary[f"OLS_{name}"] = {"alpha": alpha}
        elif name == "lasso":
            best = None
            for alpha in OLS_LASSO_ALPHAS:
                m = model_factory(name, d_in)
                fit_ols(m, ds_train["X"], ds_train["y"], alpha=alpha)
                v = realized_mvo_cost(m, ds_val)
                if best is None or v < best[1]:
                    best = (alpha, v, m)
            alpha, _, m = best
            summary[f"OLS_{name}"] = {"alpha": alpha}
        elif name == "elasticnet":
            best = None
            for alpha in OLS_LASSO_ALPHAS:
                for l1r in OLS_ELASTICNET_L1RATIO:
                    m = model_factory(name, d_in)
                    fit_ols(m, ds_train["X"], ds_train["y"], alpha=alpha, l1_ratio=l1r)
                    v = realized_mvo_cost(m, ds_val)
                    if best is None or v < best[1]:
                        best = ((alpha, l1r), v, m)
            (alpha, l1r), _, m = best
            summary[f"OLS_{name}"] = {"alpha": alpha, "l1_ratio": l1r}
        elif name == "kernel":
            best = None
            for g in KERNEL_GAMMA:
                m = model_factory(name, d_in, anchors=anchors, gamma=g)
                fit_ols(m, ds_train["X"], ds_train["y"], alpha=1e-2)
                v = realized_mvo_cost(m, ds_val)
                if best is None or v < best[1]:
                    best = (g, v, m)
            g, _, m = best
            summary[f"OLS_{name}"] = {"gamma": g}
        elif name == "nn":
            best = None
            for h in IPO_NN_HIDDEN:
                m = model_factory(name, d_in, hidden=h)
                fit_ols(m, ds_train["X"], ds_train["y"], alpha=1e-4)
                v = realized_mvo_cost(m, ds_val)
                if best is None or v < best[1]:
                    best = (h, v, m)
            h, _, m = best
            summary[f"OLS_{name}"] = {"hidden": h}
        else:
            m = model_factory(name, d_in)
            fit_ols(m, ds_train["X"], ds_train["y"], alpha=0.0 if name == "linear" else 1e-2)
            summary[f"OLS_{name}"] = {}

        torch.save(m.state_dict(), MODELS_DIR / f"OLS_{name}.pt")
        if name == "kernel":
            torch.save(m.anchors, MODELS_DIR / f"OLS_{name}.anchors.pt")
        summary[f"OLS_{name}"]["val_cost"] = realized_mvo_cost(m, ds_val)
        summary[f"OLS_{name}"]["test_cost"] = realized_mvo_cost(m, ds_test)
        print(f"OLS-{name:10s}  val={summary[f'OLS_{name}']['val_cost']:+.4f}  "
              f"test={summary[f'OLS_{name}']['test_cost']:+.4f}")

    # ----------- IPO (decision-focused) variants -----------
    for name in MODEL_FAMILIES:
        print(f"\n=== training IPO-{name} ===")
        best = None
        # ``grid`` entries are (family_name, weight_decay, l1_penalty, extras).
        if name in ("linear", "polynomial"):
            grid = [(name, 0.0, 0.0, {})]
        elif name == "ridge":
            grid = [("ridge", wd, 0.0, {}) for wd in IPO_RIDGE_WD]
        elif name == "lasso":
            grid = [("lasso", 0.0, l1, {}) for l1 in IPO_LASSO_L1]
        elif name == "elasticnet":
            grid = [("elasticnet", wd, l1, {}) for (l1, wd) in IPO_ELASTICNET]
        elif name == "kernel":
            grid = [("kernel", 1e-3, 0.0, {"gamma": g}) for g in KERNEL_GAMMA]
        elif name == "nn":
            grid = [("nn", 1e-4, 0.0, {"hidden": h}) for h in IPO_NN_HIDDEN]
        else:
            grid = []

        for nm, wd, l1, extras in grid:
            kwargs = {}
            if nm == "kernel":
                kwargs["anchors"] = anchors
                kwargs.update(extras)
            elif nm == "nn":
                kwargs.update(extras)
            elif nm == "polynomial":
                kwargs["degree"] = 2
            m = model_factory(nm, d_in, **kwargs)

            t0 = time.time()
            info = train_ipo(m, ds_train, ds_val, epochs=80, lr=5e-3,
                             weight_decay=wd, l1_penalty=l1, patience=12)
            dt = time.time() - t0
            print(f"  wd={wd}  l1={l1}  extras={extras}  val={info['best_val']:+.4f}  "
                  f"({dt:.1f}s)")
            if best is None or info["best_val"] < best[0]:
                best = (info["best_val"], m, wd, l1, extras)

        _, m, wd, l1, extras = best
        summary[f"IPO_{name}"] = {"weight_decay": wd, "l1_penalty": l1, **extras}
        torch.save(m.state_dict(), MODELS_DIR / f"IPO_{name}.pt")
        if name == "kernel":
            torch.save(m.anchors, MODELS_DIR / f"IPO_{name}.anchors.pt")
        summary[f"IPO_{name}"]["val_cost"] = realized_mvo_cost(m, ds_val)
        summary[f"IPO_{name}"]["test_cost"] = realized_mvo_cost(m, ds_test)
        print(f"IPO-{name:10s}  val={summary[f'IPO_{name}']['val_cost']:+.4f}  "
              f"test={summary[f'IPO_{name}']['test_cost']:+.4f}")

    out = {
        "delta": DELTA, "region": REGION,
        "split": {"train_end": str(TRAIN_END), "val_end": str(VAL_END)},
        "feature_names": panel["feature_names"],
        "models": summary,
    }
    with open(RESULTS / "training_summary.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nwrote {RESULTS / 'training_summary.json'}")
    return out


def main() -> None:
    panel = load_panel(OUT_DIR / f"panel_{COV_METHOD}.npz")
    run_all(panel)


if __name__ == "__main__":
    main()
