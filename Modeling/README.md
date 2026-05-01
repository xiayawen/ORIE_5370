# Nonlinear Prediction Models in Integrated Mean-Variance Portfolio Optimization

This folder contains the code that completes the empirical part of the project.
It builds on the data preparation already done in `Project/get_data.ipynb` and
`Project/analysis_data.ipynb` (S&P 500 daily prices in `Project/price_cache/`,
and FF5 / Momentum / Industry-49 factor files in `Project/factor data/`).

> When pushed to the team repo, the contents of this folder are intended to sit
> alongside `Project/get_data.ipynb` (i.e. they expect to find `price_cache/`,
> `factor data/` as siblings, or symlinks pointing to them). Existing files in
> `Project/` are **not** modified.

## Headline result

Best out-of-sample model on the 71-month test period (2019–2024):
**IPO Elastic Net**, Sharpe 1.18, max drawdown 9.5%, mean realised
MVO cost −0.0011. Two ingredients drive the gain over the OLS-linear
baseline: (i) L1 regularisation on the predictor (Lasso / Elastic Net)
and (ii) end-to-end decision-focused training. Both contribute
independently and combine multiplicatively. See `report/results.md`
for the full table and `report/conclusion.md` for the discussion.

## File layout

```
Data/
  get_data.ipynb # Download / prepare S&P 500 price data
  analysis_data.ipynb # Clean and inspect price + factor data
  price_cache_manager.py # Reusable yfinance cache manager
  price_cache/ # Cached daily OHLCV files for S&P 500 stocks
  factor data/ # FF5, Momentum, and Industry-49 daily factor files
  sp500_filtered.csv # Filtered S&P 500 universe used for modeling
  
Modeling/
  build_dataset.py            # Build sp500_filtered.csv + IPO panel from price_cache
  mvo.py                      # Mean-variance optimization helpers (analytical)
  ipo_models.py               # Linear / Ridge / Lasso / ElasticNet / Polynomial / Kernel / NN
  train.py                    # OLS plug-in + IPO training, hyperparameter sweeps
  evaluate.py                 # OOS backtest + summary metrics + pairwise bootstrap
  make_figures.py             # Plots and tables for the report
  run_all.py                  # Orchestrator (--skip-data / --skip-train)
  data_cache/                 # Generated panel (sp500_filtered.csv, panel.npz)  [gitignored]
  results/                    # Performance / coefficients / pairwise dominance CSVs
    performance.csv           # one row per (paradigm, predictor) summary
    pairwise_dominance.csv    # 14 × 14 bootstrap dominance probabilities
    per_month.csv             # OOS realized return per model per month
    per_month_cost.csv        # OOS realized MVO cost per model per month
    coefficients.csv          # Linear / regularised-linear feature weights
    training_summary.json     # Hyperparameters chosen on the validation set
  figures/                    # 6 PNGs that illustrate the results
  report/
    methodology.md            # § III data + § IV models / training
    literature_review.md      # § II prior work
    results.md                # § V results, § VI discussion
    conclusion.md             # § VII conclusion + future work

Theoretical Foundation/
  Closed-Form Structure for Nonlinear IPO Extensions.*
  # Analytical appendix for the MVO layer,
  # nonlinear IPO extensions, and covariance estimation
```

## How to run

```bash
cd Modeling
# Symlink (or copy) the team data into this folder if not already siblings:
ln -s ../Project/price_cache .
ln -s "../Project/factor data" "factor data"

# Build the panel dataset (~30 s on a laptop)
python build_dataset.py

# Run the full pipeline (build models, evaluate, make figures).
# Reuses data_cache/panel.npz if it already exists.
python run_all.py --skip-data                  # if panel.npz is already built
python run_all.py                              # full pipeline from raw price cache
python run_all.py --skip-data --skip-train     # rerun only evaluation + plots
```

`run_all.py` runs `train.py`, then `evaluate.py`, then `make_figures.py`. It
also sets ``KMP_DUPLICATE_LIB_OK=TRUE`` and pins OMP/MKL threads to 1 — on
Anaconda + PyTorch (macOS) we observed deadlocks in multi-threaded BLAS
during the MLP plug-in training, and the OpenMP runtime conflict abort. If
you invoke ``train.py`` directly, set those vars yourself first.

## What is implemented

The empirical setup follows Butler & Kwon (2022, arXiv:2102.09287) and extends
their linear IPO framework to **regularised and nonlinear** predictive models
trained against the realised mean-variance cost (the central question of our
project outline § IV).

Seven predictors are compared, each evaluated under two training paradigms
(14 model × paradigm cells in total):

| Predictor | Functional form | OLS (predict-then-optimize) | IPO (decision-focused) |
|---|---|---|---|
| Linear      | `f(x) = W x + b` | closed-form OLS | analytical equality-constrained IPO |
| Ridge       | same, with `λ‖W‖²` | closed-form | gradient IPO with `weight_decay` |
| Lasso       | same, with `λ‖W‖₁` | sklearn `Lasso` | gradient IPO with explicit L1 term |
| Elastic Net | same, with `λ₁‖W‖₁ + λ₂‖W‖²` | sklearn `ElasticNet` | gradient IPO with L1 + L2 |
| Polynomial  | `f(x) = W φ(x) + b`, φ adds squares + interactions | closed-form OLS | gradient IPO |
| Kernel ridge | `Σ α_i K(x, xᵢ)`, RBF kernel, M = 200 anchors | closed-form | gradient IPO |
| MLP (1 hidden) | `W₂ tanh(W₁ x + b₁) + b₂` | mini-batch MSE | gradient IPO |

The downstream MVO program is the equality-constrained one (`1ᵀ z = 1`, fully
invested) used by Butler & Kwon § 2.4. Hyperparameters (ridge λ, lasso α,
elastic-net mix, kernel γ, NN hidden width, IPO weight decay, IPO L1
penalty) are swept on a validation period and chosen by lowest realised
MVO cost.

## Theoretical foundation

The project also includes a theoretical appendix that supports the empirical
IPO extensions. The appendix shows that the equality-constrained MVO decision
layer remains closed-form because the optimal portfolio is affine in the
predicted return vector, `z*(ŷ) = B_t ŷ + a_t`. This allows IPO models to
backpropagate through the portfolio decision without using a differentiable QP
solver.

The appendix also clarifies the optimization structure of the nonlinear
extensions. Polynomial and fixed-anchor kernel predictors are nonlinear in
features but linear in trainable parameters, so they preserve a quadratic IPO
structure. Ridge IPO admits a closed-form estimator; Lasso and Elastic Net are
convex but nonsmooth because of the L1 penalty; neural-network IPO is nonconvex
and trained by gradient descent.

Finally, the appendix documents the covariance estimator used in the empirical
pipeline: a 60-day rolling daily covariance matrix with Ledoit-Wolf-style
shrinkage toward a scaled identity target, rescaled by 21 to match the monthly
return horizon used in the MVO objective.

## Performance metrics

For each model we report (out-of-sample, on the test period):

- Realised MVO cost `c(z*,y) = -zᵀy + (δ/2) zᵀ V z` — the IPO objective
- Annualised return / volatility / Sharpe ratio (×√12)
- Maximum drawdown, 5%-VaR (empirical lower-tail quantile)
- Mean per-month gross exposure `‖z‖₁` (weight-stability proxy)
- Predictive R² on returns (for the cost-vs-prediction-quality plot)
- Paired iid bootstrap (B = 5000) dominance probability against
  OLS-linear, *and* the full 14 × 14 pairwise dominance matrix for
  proposal § V hypothesis testing

## Computational notes

- Covariance matrices are estimated by 60-day rolling sample covariance with
  Ledoit-Wolf linear shrinkage. This keeps them well-conditioned for the
  cross-sections used here (n ≈ 100 stocks).
- Gradient-based IPO uses PyTorch with an analytical, equality-constrained
  closed-form for `z*(ŷ)`, so the full forward/backward pass is differentiable
  without needing `cvxpylayers`.
- The L1 penalty in IPO Lasso / Elastic Net training is applied as an explicit
  `λ‖W‖₁` term on the predictor weight; Adam handles the non-differentiability
  at zero via subgradient.
- All experiments fit on a laptop; the full pipeline (fresh data build →
  train all 14 models → evaluate → figures) runs in roughly **4 minutes**
  with the env vars `KMP_DUPLICATE_LIB_OK=TRUE` and `OMP_NUM_THREADS=1`
  set automatically by `run_all.py`.

## Reproducing the report

Once `python run_all.py --skip-data` finishes, the four files in `report/`
already cite the actual numbers from the latest CSVs in `results/` and embed
the figures from `figures/`. Concatenating them in this order gives the
final write-up:

```
report/literature_review.md   # § II
report/methodology.md         # § III + § IV
report/results.md             # § V + § VI
report/conclusion.md          # § VII + future work
```
