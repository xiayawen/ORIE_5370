# Nonlinear Prediction Models in Integrated Mean-Variance Portfolio Optimization

This folder contains the code that completes the empirical part of the project.
It builds on the data preparation already done in `Project/get_data.ipynb` and
`Project/analysis_data.ipynb` (S&P 500 daily prices in `Project/price_cache/`,
and FF5 / Momentum / Industry-49 factor files in `Project/factor data/`).

> When pushed to the team repo, the contents of this folder are intended to sit
> alongside `Project/get_data.ipynb` (i.e. they expect to find `price_cache/`,
> `factor data/` as siblings, or symlinks pointing to them). Existing files in
> `Project/` are **not** modified.

## File layout

```
Project_extension/
  build_dataset.py        # Build sp500_filtered.csv + IPO panel from price_cache
  mvo.py                  # Mean-variance optimization helpers (analytical)
  ipo_models.py           # Linear / Polynomial / Ridge / Kernel / NN predictors
  train.py                # Trains OLS plug-in + IPO variants on training/val sets
  evaluate.py             # Out-of-sample backtest + performance metrics + bootstrap
  make_figures.py         # Plots and tables for the report
  run_all.py              # Orchestrator: runs the full pipeline end-to-end
  data_cache/             # Generated panel data (sp500_filtered.csv, panel.npz)
  results/                # CSVs of metrics, regression coefficients, etc.
  figures/                # PNG/PDF figures
  report/                 # Methodology / Results / Discussion write-up
```

## How to run

```bash
cd Project_extension
# Symlink (or copy) the team data into this folder if not already siblings:
ln -s ../Project/price_cache .
ln -s "../Project/factor data" "factor data"

# Build the panel dataset (~30 s on a laptop)
python build_dataset.py

# Run the full pipeline (build models, evaluate, make figures).
# Reuses data_cache/panel.npz if it already exists.
python run_all.py --skip-data           # if panel.npz is already built
python run_all.py                       # full pipeline from raw price cache
python run_all.py --skip-data --skip-train   # rerun only evaluation + plots
```

`run_all.py` runs `train.py`, then `evaluate.py`, then `make_figures.py`. It
also sets ``KMP_DUPLICATE_LIB_OK=TRUE`` and pins OMP/MKL threads to 1 — on
Anaconda + PyTorch (macOS) we observed deadlocks in multi-threaded BLAS
during the MLP plug-in training, and the OpenMP runtime conflict abort. If
you invoke ``train.py`` directly, set those vars yourself first.

## What is implemented

The empirical setup follows Butler & Kwon (2022, arXiv:2102.09287) and extends
their linear IPO framework to **nonlinear** predictive models trained against the
realized mean-variance cost (the central question of our project outline).

Five predictors are compared, each evaluated under two training paradigms:

| Predictor | OLS (predict-then-optimize) | IPO (decision-focused) |
|---|---|---|
| Linear `f(x)=Wx+b` | ✓ (closed-form OLS) | ✓ (closed-form analytical IPO) |
| Polynomial `f(x)=W φ(x)+b` | ✓ | ✓ (gradient IPO) |
| Ridge `‖W‖²` regularised | ✓ | ✓ (gradient IPO) |
| Kernel ridge `Σ α_i K(x,x_i)` | ✓ | ✓ (gradient IPO) |
| Neural net 1 hidden layer | ✓ | ✓ (gradient IPO) |

The downstream MVO program is the equality-constrained one
(`1ᵀ z = 1`, fully invested) used by Butler & Kwon §2.4.

## Performance metrics

For each model we report (out-of-sample, on the test period):

- Realized MVO cost `c(z*,y) = -zᵀy + (δ/2) zᵀ V z`
- Annualized return / volatility / Sharpe ratio
- Maximum drawdown, 5%-VaR
- Predictive R² (on returns, for context)
- Bootstrap dominance ratio against the OLS-linear baseline

## Computational notes

- Covariance matrices are estimated by 60-day rolling sample covariance with
  Ledoit-Wolfe linear shrinkage. This keeps them well-conditioned for the
  cross-sections used here (n ≈ 50–100 stocks).
- Gradient-based IPO uses PyTorch with an analytical, equality-constrained
  closed-form for `z*(ŷ)` so the full forward/backward pass is differentiable
  without needing `cvxpylayers`.
- All experiments fit on a laptop; full pipeline runtime is roughly 5 minutes.
