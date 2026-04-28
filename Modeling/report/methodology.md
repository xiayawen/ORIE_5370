# Methodology

This note documents the empirical implementation that accompanies the project
outline *Nonlinear Prediction Models in Integrated Mean-Variance Portfolio
Optimization*. The code lives in `Project_extension/` and is invoked end-to-end
via `python run_all.py`.

## 1. Data

We use the S&P 500 daily price universe collected by the teammate's
`Project/get_data.ipynb`: one CSV per ticker in `price_cache/`, plus the
Fama–French 5-factor, momentum, and 49-industry-portfolio daily files in
`factor data/`. Filtering rules (`build_dataset.build_sp500_filtered`):

* Keep tickers whose first available price falls on/before **2005-01-01** and
  whose last price extends to **2024-01-01** or later, so each name spans the
  full evaluation window.
* From the filtered set, retain the **top 100 tickers by median dollar
  volume** to keep the cross-section tractable for repeated covariance
  inversions (the project outline §3.1 used 300; we cut it for runtime, but
  the code's `UNIVERSE_SIZE` constant exposes the choice).
* Compute monthly arithmetic returns from end-of-month closing prices and
  realized next-month returns by compounding daily returns within the month.

### Predictive features

For stock $i$ on month-end $t$ we build the cross-sectional feature vector

$$
x_{i,t} = \big[\text{mom}_{12,2},\; \text{mom}_{1m},\; r_{i,t-1},\; \sigma^{\,60}_{i,t},\; \log\overline{\text{DV}}^{\,60}_{i,t},\; \text{Mkt-RF}_t,\; \text{SMB}_t,\; \text{HML}_t,\; \text{RMW}_t,\; \text{CMA}_t,\; \text{Mom}_t\big].
$$

The asset-specific block is z-scored cross-sectionally each month and missing
values are imputed by the cross-sectional median (project outline §3.3). The
common-factor block is broadcast across the cross-section so every asset sees
the same factor realisation that month.

### Sample splitting

* **Training:** earliest available rebalance through **2017-01-01** (≈12 yrs).
* **Validation:** **2017-01-01 → 2019-01-01** (used for hyperparameter
  selection and IPO early stopping).
* **Test:** **2019-01-01** through the latest available date (≈5 yrs).

This deviates from the outline's 1990–2010 / 2011–2016 / 2017–2023 split
because the price cache only goes back to 2005; the relative train/val/test
proportions are preserved.

### Covariance estimation

For each rebalance month $t$ we estimate $V_t$ from the trailing 60 trading-
day returns of the surviving cross-section using the Ledoit–Wolf (2004)
linear shrinkage estimator (`build_dataset.ledoit_wolf_shrinkage`). The daily
covariance is rescaled by 21 to match the units of the realized monthly
return that enters the MVO cost. Shrinkage keeps $V_t$ well-conditioned
even when $n \approx 100$ and the sample window is short — directly
addressing the "covariance not invertible" question raised in the
outline's *Unsolved Questions* section.

## 2. Predictive models

Five predictors $f(x;\theta)$ from the outline §IV are implemented in
`ipo_models.py`:

| Model | Functional form | Trained parameters |
|---|---|---|
| Linear | $W x + b$ | $W \in \mathbb{R}^{1\times d}, b$ |
| Polynomial | $W\,\phi(x) + b$ with $\phi(x) = [x,\, x^2,\, \{x_i x_j\}_{i<j}]$ | $W,b$ on the lifted features |
| Ridge | $Wx + b$ with $\ell_2$ penalty $\lambda\|W\|^2$ | $W,b$ |
| Kernel | $\sum_{m=1}^M \alpha_m K(x, x_m)$, RBF kernel, $M=200$ random anchors | $\alpha,b$ |
| MLP | $W_2\,\tanh(W_1 x + b_1) + b_2$, single hidden layer | $W_1,b_1,W_2,b_2$ |

Each model is trained under two paradigms:

1. **OLS plug-in.** Closed-form least squares for linear / ridge / polynomial /
   kernel; mini-batch MSE for the MLP. Predictions are then plugged into the
   downstream MVO program — i.e. the classical "predict-then-optimize" recipe.
2. **IPO (decision-focused).** End-to-end gradient training on the realized
   MVO cost
   $$\mathcal{L}(\theta) = \frac{1}{T}\sum_{t=1}^{T}\Big[-z^*(\hat y_t)^\top y_t + \tfrac{\delta}{2}\, z^*(\hat y_t)^\top V_t\, z^*(\hat y_t)\Big],$$
   where $z^*$ is the **closed-form** equality-constrained solution
   $$z^*(\hat y_t) = \tfrac{1}{\delta}\,V_t^{-1}\Big(\hat y_t - \lambda_t \mathbf{1}\Big),\quad \lambda_t = \frac{\mathbf{1}^\top V_t^{-1} \hat y_t - \delta}{\mathbf{1}^\top V_t^{-1} \mathbf{1}},$$
   so the entire forward / backward pass is differentiable in PyTorch
   without invoking `cvxpylayers` (`mvo.mvo_equality_t`). We use Adam with
   gradient clipping and early stop on the validation MVO cost.

Hyperparameters (ridge $\lambda$, kernel $\gamma$, NN hidden width, IPO
weight decay) are swept over the grid in `train.py` and selected by lowest
validation realized MVO cost.

## 3. Downstream optimization

We solve the equality-constrained MVO

$$\min_{z\in\mathbb{R}^n}\; -z^\top y + \tfrac{\delta}{2}\, z^\top V z\quad\text{s.t.}\quad \mathbf{1}^\top z = 1,$$

with $\delta = 50$ matching Butler & Kwon §4. Long-only / leverage caps are
deferred to the *Discussion* — they would require either iterative
projection (still differentiable but slower) or a QP layer.

## 4. Evaluation

`evaluate.py` walks each test-period rebalance, computes $\hat y_t$, solves
$z^*_t$, and records (i) the realized return $r_t = z_t^{*\top} y_t$,
(ii) the realized MVO cost $c(z^*_t, y_t)$, (iii) the gross exposure
$\|z^*_t\|_1$ as a stability proxy, and (iv) the cross-sectional predictive
$R^2$ for context.

Performance metrics reported in `results/performance.csv`:

* Mean realized MVO cost (the IPO objective)
* Annualized return / volatility / Sharpe ratio (×$\sqrt{12}$)
* Maximum drawdown of the cumulative-return curve
* 5%-VaR (empirical lower-tail quantile)
* Mean cross-sectional predictive $R^2$
* Bootstrap dominance probability against the OLS-linear baseline
  (paired iid bootstrap on the per-month cost differences, $B = 5000$)

## 5. Files produced

```
data_cache/panel.npz            per-month {X_t, y_t, V_t}
results/training_summary.json   chosen hyperparameters per model
results/models/*.pt             trained state-dicts
results/per_month.csv           OOS per-rebalance return per model
results/per_month_cost.csv      OOS per-rebalance MVO cost per model
results/performance.csv         summary metrics + bootstrap dominance
results/coefficients.csv        linear / ridge feature weights
figures/cum_returns.png         cumulative growth of \$1
figures/cost_bar.png            mean MVO cost
figures/sharpe_bar.png          Sharpe by model
figures/cost_vs_r2.png          decision quality vs prediction quality
figures/dominance_heatmap.png   bootstrap dominance vs OLS-linear
```

## 6. Caveats relative to the outline

* **Universe size**: 100, not 300. The covariance solve is $O(n^3)$ per
  rebalance and IPO sweeps a few hundred rebalances; 100 keeps the full
  pipeline under five minutes on a laptop. Increasing `UNIVERSE_SIZE` in
  `build_dataset.py` is a one-line change.
* **Sample window**: starts in 2005 (price cache limit), not 1990.
* **Long-only / turnover constraints**: omitted; gross exposure is reported
  as a stability proxy. This is the "transaction cost / turnover control"
  question flagged in the outline's *Unsolved Questions* and is a natural
  follow-up.
* **Industry-49 dummies**: ingested in the panel build step but currently
  not appended to the feature matrix — adding them is one line in
  `build_dataset.COMMON_FEATURES`.
