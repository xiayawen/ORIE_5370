"""Out-of-sample backtest and performance metrics for every trained model.

For each ``(predictor, training-paradigm)`` combination saved by ``train.py``
into ``results/models/`` we:

1. Reconstruct the predictor and load its state-dict.
2. On the held-out test cross-sections, predict ``ŷ_t`` and compute portfolio
   weights ``z*_t = arg min_z c(z, ŷ_t) s.t. 1^T z = 1`` using the analytical
   equality-constrained solution from ``mvo.py``.
3. Realize the portfolio return ``r_t = z*_t · y_t`` and the realized MVO cost
   ``c(z*_t, y_t)``.
4. Aggregate to performance summary (annualized return / vol / Sharpe, max
   drawdown, 5%-VaR, mean cost) and stack the per-month series for plotting.
5. Compute a paired bootstrap dominance ratio against the OLS-linear plug-in
   baseline (Butler & Kwon §4.3): the fraction of bootstrap resamples in
   which the candidate model has lower mean MVO cost than the baseline.

Outputs (written to ``results/``):

* ``per_month.csv``      — wide table of per-rebalance returns per model
* ``per_month_cost.csv`` — wide table of per-rebalance MVO costs per model
* ``performance.csv``    — one-row-per-model summary metrics
* ``coefficients.csv``   — linear / ridge feature coefficients (for the
                           interpretation section of the report)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch

from build_dataset import OUT_DIR, load_panel
from ipo_models import (
    LinearPredictor, PolynomialPredictor, RidgePredictor,
    LassoPredictor, ElasticNetPredictor,
    KernelRidgePredictor, MLPPredictor,
)
from mvo import mvo_solve, mvo_cost
from train import (
    DELTA, REGION, SUM_TO, TRAIN_END, VAL_END,
    slice_period, model_factory, MODELS_DIR, RESULTS,
    IPO_NN_HIDDEN, KERNEL_GAMMA,
)


PER_YEAR_REBALANCES = 12  # monthly portfolio rebalancing


# ---------------------------------------------------------------------------
# Loading saved predictors back from disk
# ---------------------------------------------------------------------------

def _load_anchors(name: str) -> torch.Tensor:
    return torch.load(MODELS_DIR / f"{name}.anchors.pt", weights_only=True)


def load_model(name: str, summary: dict, d_in: int) -> torch.nn.Module:
    """Reinstantiate a model and load its trained state-dict.

    ``summary`` is the per-model dict from ``results/training_summary.json``
    that records the chosen hyperparameters (kernel γ, NN width, ...).
    """
    family = name.split("_", 1)[1]
    cfg = summary["models"][name]
    if family == "kernel":
        anchors = _load_anchors(name)
        gamma = float(cfg.get("gamma", 1.0))
        m = KernelRidgePredictor(anchors, gamma=gamma)
    elif family == "nn":
        hidden = int(cfg.get("hidden", 16))
        m = MLPPredictor(d_in, hidden=hidden)
    elif family == "polynomial":
        m = PolynomialPredictor(d_in, degree=2, interactions=True)
    elif family == "ridge":
        m = RidgePredictor(d_in)
    elif family == "lasso":
        m = LassoPredictor(d_in)
    elif family == "elasticnet":
        m = ElasticNetPredictor(d_in)
    else:
        m = LinearPredictor(d_in)
    state = torch.load(MODELS_DIR / f"{name}.pt", weights_only=True)
    m.load_state_dict(state)
    m.eval()
    return m


# ---------------------------------------------------------------------------
# Per-month backtest (returns, costs, weights, predictions)
# ---------------------------------------------------------------------------

def backtest(model: torch.nn.Module, ds: dict,
             delta: float = DELTA, region: str = REGION) -> dict:
    """Run one model over the held-out cross-sections.

    Returns a dict with arrays of per-month realized return, realized MVO cost,
    weight L1-norm, predictive R², and the dates they correspond to.
    """
    rets, costs, wL1, r2 = [], [], [], []
    yhat_all, y_all = [], []
    dates_kept = []

    for d, X, y, V in zip(ds["dates"], ds["X"], ds["y"], ds["V"]):
        Xt = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            yhat = model(Xt).cpu().numpy().astype(np.float64)
        try:
            z = mvo_solve(yhat, V, delta, region=region, sum_to=SUM_TO)
        except np.linalg.LinAlgError:
            continue
        rets.append(float(z @ y))
        costs.append(float(mvo_cost(z, y, V, delta)))
        wL1.append(float(np.abs(z).sum()))
        # Cross-sectional R² for context (this is *prediction* R², not
        # decision quality — included only for the §V comparison).
        ssr = float(np.sum((y - yhat) ** 2))
        sst = float(np.sum((y - y.mean()) ** 2))
        r2.append(1.0 - ssr / sst if sst > 0 else np.nan)
        yhat_all.append(yhat)
        y_all.append(y)
        dates_kept.append(d)

    return {
        "dates": np.array(dates_kept),
        "ret": np.array(rets),
        "cost": np.array(costs),
        "weight_l1": np.array(wL1),
        "r2": np.array(r2),
    }


# ---------------------------------------------------------------------------
# Performance summary metrics
# ---------------------------------------------------------------------------

def annualized_return(rets: np.ndarray) -> float:
    return float(np.mean(rets) * PER_YEAR_REBALANCES)


def annualized_vol(rets: np.ndarray) -> float:
    return float(np.std(rets, ddof=1) * np.sqrt(PER_YEAR_REBALANCES))


def sharpe_ratio(rets: np.ndarray) -> float:
    sd = np.std(rets, ddof=1)
    if sd == 0 or not np.isfinite(sd):
        return float("nan")
    return float(np.mean(rets) / sd * np.sqrt(PER_YEAR_REBALANCES))


def max_drawdown(rets: np.ndarray) -> float:
    """Maximum drawdown of the cumulative-return curve, as a positive fraction."""
    eq = np.cumprod(1.0 + rets)
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / peak
    return float(np.max(dd)) if len(dd) else float("nan")


def value_at_risk(rets: np.ndarray, q: float = 0.05) -> float:
    """Empirical lower-tail VaR (positive number = magnitude of the loss)."""
    if not len(rets):
        return float("nan")
    return float(-np.quantile(rets, q))


def summary_metrics(name: str, bt: dict) -> dict:
    rets = bt["ret"]
    return {
        "model": name,
        "n_months": len(rets),
        "mean_cost": float(np.mean(bt["cost"])) if len(bt["cost"]) else float("nan"),
        "ann_return": annualized_return(rets),
        "ann_vol": annualized_vol(rets),
        "sharpe": sharpe_ratio(rets),
        "max_drawdown": max_drawdown(rets),
        "var_5pct": value_at_risk(rets, 0.05),
        "weight_l1_mean": float(np.mean(bt["weight_l1"])) if len(bt["weight_l1"]) else float("nan"),
        "r2_mean": float(np.nanmean(bt["r2"])) if len(bt["r2"]) else float("nan"),
    }


# ---------------------------------------------------------------------------
# Bootstrap dominance vs. OLS-linear baseline
# ---------------------------------------------------------------------------

def pairwise_dominance_matrix(cost_dict: dict[str, np.ndarray],
                              n_boot: int = 5000, seed: int = 0) -> pd.DataFrame:
    """Bootstrap probability that row-model has lower cost than column-model.

    Diagonal entries are 0.5 by convention. The matrix is the formal
    "hypothesis testing" the proposal §V calls for: for every (OLS_X, IPO_Y)
    pair we report ``P(cost_row < cost_col)`` over ``n_boot`` paired iid
    bootstrap resamples on the test-period cost differences.
    """
    names = sorted(cost_dict.keys())
    n_models = len(names)
    n = min(len(c) for c in cost_dict.values())
    rng = np.random.default_rng(seed)
    boot_idx = rng.integers(0, n, size=(n_boot, n))

    M = np.full((n_models, n_models), 0.5)
    for i, ni in enumerate(names):
        ci = cost_dict[ni][:n]
        for j, nj in enumerate(names):
            if i == j:
                continue
            cj = cost_dict[nj][:n]
            diffs = (ci - cj)[boot_idx]              # (n_boot, n)
            wins = (diffs.mean(axis=1) < 0).mean()
            M[i, j] = float(wins)

    return pd.DataFrame(M, index=names, columns=names)


def bootstrap_dominance(cand: np.ndarray, base: np.ndarray,
                        n_boot: int = 5000, seed: int = 0) -> dict:
    """Paired stationary bootstrap dominance ratio.

    Returns the fraction of bootstrap resamples for which the candidate model
    has *lower* mean realized MVO cost than the baseline (Butler & Kwon §4.3
    use this as their empirical significance test).

    A simple non-overlapping iid bootstrap is sufficient for monthly data.
    """
    rng = np.random.default_rng(seed)
    n = min(len(cand), len(base))
    if n == 0:
        return {"prob_dominates": float("nan"), "mean_diff": float("nan")}
    cand = cand[:n]
    base = base[:n]
    diffs = cand - base  # negative diff means candidate is better (lower cost)
    wins = 0
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if diffs[idx].mean() < 0:
            wins += 1
    return {
        "prob_dominates": wins / n_boot,
        "mean_diff": float(diffs.mean()),
    }


# ---------------------------------------------------------------------------
# Coefficient / interpretation table for linear models
# ---------------------------------------------------------------------------

def linear_coefficients(name: str, summary: dict, d_in: int,
                        feature_names: list[str]) -> dict | None:
    """Extract linear / ridge weights so we can show them in the report.

    Returns ``None`` for non-linear predictors.
    """
    family = name.split("_", 1)[1]
    if family not in ("linear", "ridge", "lasso", "elasticnet"):
        return None
    m = load_model(name, summary, d_in)
    w = m.linear.weight.detach().numpy().reshape(-1)
    b = float(m.linear.bias.detach().numpy().reshape(-1)[0])
    out = {f"w_{f}": float(v) for f, v in zip(feature_names, w)}
    out["bias"] = b
    out["model"] = name
    return out


# ---------------------------------------------------------------------------
# Main: stitch everything together
# ---------------------------------------------------------------------------

def main() -> None:
    panel = load_panel(OUT_DIR / "panel.npz")
    with open(RESULTS / "training_summary.json") as f:
        summary = json.load(f)

    ds_test = slice_period(
        panel, VAL_END, panel["dates"].to_numpy()[-1] + np.timedelta64(1, "D")
    )
    print(f"test sample: {len(ds_test['X'])} months, "
          f"{ds_test['dates'][0]} to {ds_test['dates'][-1]}")

    d_in = panel["X"][0].shape[1]
    feature_names = panel["feature_names"]

    model_names = sorted(summary["models"].keys())
    backtests: dict[str, dict] = {}
    rows: list[dict] = []

    for name in model_names:
        m = load_model(name, summary, d_in)
        bt = backtest(m, ds_test)
        backtests[name] = bt
        rows.append(summary_metrics(name, bt))
        print(f"  {name:18s}  "
              f"cost={rows[-1]['mean_cost']:+.4f}  "
              f"sharpe={rows[-1]['sharpe']:+.2f}  "
              f"MDD={rows[-1]['max_drawdown']:.2%}")

    perf = pd.DataFrame(rows).set_index("model").sort_index()

    # Bootstrap dominance against the OLS-linear baseline.
    base_cost = backtests.get("OLS_linear", {}).get("cost", None)
    if base_cost is not None and len(base_cost):
        dom = []
        for name, bt in backtests.items():
            d = bootstrap_dominance(bt["cost"], base_cost)
            dom.append({"model": name, **d})
        dom_df = pd.DataFrame(dom).set_index("model")
        perf = perf.join(dom_df)
    else:
        print("warning: no OLS_linear backtest found; skipping dominance test")

    out_perf = RESULTS / "performance.csv"
    perf.to_csv(out_perf)
    print(f"wrote {out_perf}")

    # Pairwise bootstrap dominance matrix (proposal §V hypothesis testing).
    cost_dict = {n: bt["cost"] for n, bt in backtests.items()}
    pw = pairwise_dominance_matrix(cost_dict)
    pw.to_csv(RESULTS / "pairwise_dominance.csv")
    print(f"wrote {RESULTS / 'pairwise_dominance.csv'}")

    # Per-month tables for plotting.
    dates_ref = backtests[model_names[0]]["dates"]
    per_ret = pd.DataFrame({n: bt["ret"] for n, bt in backtests.items()},
                           index=pd.to_datetime(dates_ref))
    per_cost = pd.DataFrame({n: bt["cost"] for n, bt in backtests.items()},
                            index=pd.to_datetime(dates_ref))
    per_ret.index.name = "date"
    per_cost.index.name = "date"
    per_ret.to_csv(RESULTS / "per_month.csv")
    per_cost.to_csv(RESULTS / "per_month_cost.csv")
    print(f"wrote {RESULTS / 'per_month.csv'}")
    print(f"wrote {RESULTS / 'per_month_cost.csv'}")

    # Linear coefficients for the report's interpretation table.
    coef_rows = []
    for name in model_names:
        c = linear_coefficients(name, summary, d_in, feature_names)
        if c is not None:
            coef_rows.append(c)
    if coef_rows:
        pd.DataFrame(coef_rows).set_index("model").to_csv(RESULTS / "coefficients.csv")
        print(f"wrote {RESULTS / 'coefficients.csv'}")


if __name__ == "__main__":
    main()
