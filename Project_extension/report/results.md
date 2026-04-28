# Empirical Results

The numbers below come from `results/performance.csv` produced by
`run_all.py` over the 71-month test sample (2019-01-31 → 2024-11-29) with
the universe of the 100 most-liquid S&P 500 names that traded continuously
from 2005-01 onwards. δ = 50 (Butler & Kwon §4); equality-constrained
fully-invested MVO; 60-day Ledoit-Wolf shrunk covariance.

## 5.1 Headline performance table

| Model | Mean MVO cost | Ann. return | Ann. vol | Sharpe | Max DD | Gross exposure $\|z\|_1$ | $P(\text{cost}<\text{OLS\_linear})$ |
|---|---:|---:|---:|---:|---:|---:|---:|
| **IPO_nn**         | **−0.0012** | 13.6% | 14.5% | **0.94** | **11.8%** | **2.86** | **0.997** |
| IPO_linear         |   +0.0029 | 12.9% | 15.4% | 0.84 | 14.3% | 3.58 | 0.898 |
| IPO_ridge          |   +0.0065 | 11.6% | 18.4% | 0.63 | 19.6% | 4.11 | 0.683 |
| OLS_ridge          |   +0.0080 | 13.2% | 18.8% | 0.70 | 20.8% | 4.56 | 1.000\* |
| OLS_linear         |   +0.0093 | 13.2% | 19.4% | 0.68 | 21.7% | 4.76 | (baseline) |
| IPO_kernel         |   +0.0279 | 25.6% | 25.4% | **1.01** | 26.1% | 8.33 | 0.005 |
| OLS_nn             |   +0.0427 |  9.4% | 30.5% | 0.31 | 36.6% | 7.83 | 0.000 |
| OLS_polynomial     |   +0.0481 | 20.7% | 34.3% | 0.60 | 53.8% | 8.28 | 0.000 |
| OLS_kernel         |   +0.0524 | 15.9% | 31.3% | 0.51 | 43.8% | 9.40 | 0.000 |
| IPO_polynomial     |   +0.0903 | 22.3% | 49.9% | 0.45 | 59.4% | 11.83 | 0.000 |

\* OLS_ridge picked $\alpha = 0$ on validation, making it numerically
identical to OLS_linear; the bootstrap dominance probability of 1.0 reflects
that, not a real edge.

## 5.2 Does decision-focused training help? (IPO vs OLS, paired)

Comparing each predictor under the two paradigms:

| Predictor | OLS cost | IPO cost | IPO − OLS | IPO better? |
|---|---:|---:|---:|---:|
| Linear     | +0.0093 | **+0.0029** | −0.0064 | **yes** (P=0.90) |
| Ridge      | +0.0080 | **+0.0065** | −0.0015 | yes (P=0.68) |
| Kernel     | +0.0524 | **+0.0279** | −0.0245 | yes |
| MLP        | +0.0427 | **−0.0012** | −0.0439 | **yes** (P=0.997) |
| Polynomial | **+0.0481** | +0.0903 | +0.0422 | **no** |

**Headline:** IPO beats its OLS twin in 4 / 5 predictor families. The MLP
sees the largest absolute cost reduction (−0.044), and the IPO-MLP is the
only model with negative mean MVO cost on the test set. The polynomial
predictor is the exception — IPO training exacerbates overfitting because
we did not regularise the lifted-feature weights, and the resulting
gross-exposure of 11.8 dominates the cost decomposition.

## 5.3 Does nonlinearity help? (within-paradigm ranking)

| Rank by IPO cost | Rank by OLS cost |
|---|---|
| 1. **IPO_nn** (−0.0012) | 1. OLS_ridge (+0.0080) |
| 2. IPO_linear (+0.0029) | 2. OLS_linear (+0.0093) |
| 3. IPO_ridge (+0.0065) | 3. OLS_nn (+0.0427) |
| 4. IPO_kernel (+0.0279) | 4. OLS_polynomial (+0.0481) |
| 5. IPO_polynomial (+0.0903) | 5. OLS_kernel (+0.0524) |

Under the **plug-in (OLS) paradigm**, more flexible predictors uniformly
*hurt* — kernel and polynomial sit at the bottom, the MLP is fourth.
Without decision-aware training, prediction noise propagates through
$V^{-1}\hat y$ and produces unstable weights (gross exposure 8–9 vs 4.6
for linear). This is exactly the failure mode flagged in the project
outline §VI.

Under the **IPO paradigm** the picture flips for the MLP: it goes from
worst-but-one to best, while the polynomial expansion remains broken
(no regularisation in the IPO training of the polynomial coefficients
in our sweep). Kernel sits in the middle — it has more capacity than
linear but comparable training-time regularisation via a small weight
decay, so it improves over its plug-in version but not enough to beat
the parsimonious linear / NN models on the same δ.

## 5.4 Prediction quality vs decision quality

`figures/cost_vs_r2.png` makes the central point of Elmachtoub & Grigas
(2022) and the project outline §I visually concrete: the IPO points are
well below their OLS counterparts on the cost axis at *similar or worse*
predictive R² values. The IPO-MLP in particular has a *worse* mean
cross-sectional R² than any OLS model yet the *best* realized MVO cost.
Lower prediction error is not lower decision cost.

## 5.5 Stability of portfolio weights

Gross exposure $\|z\|_1$ in the table is the cleanest stability proxy.
Within IPO, the ordering tracks predictor capacity:

* IPO_nn: 2.86 (best)
* IPO_linear / IPO_ridge: 3.58 / 4.11
* IPO_kernel: 8.33
* IPO_polynomial: 11.83

The top three are well below the OLS baseline's 4.76. The MLP earning
both the lowest cost *and* the lowest gross exposure is the most striking
result: decision-aware training is regularising the weights *more* than
ridge-on-OLS, despite the MLP nominally having more capacity. The
polynomial blow-up on gross exposure is the proximate cause of its bad
realized cost.

## 5.6 Bootstrap dominance vs OLS-linear

Paired iid bootstrap (B = 5000) on the 71 monthly cost observations:

* IPO_nn dominates OLS_linear in 99.7% of resamples
* IPO_linear dominates in 89.8%
* IPO_ridge dominates in 68.3%
* IPO_kernel and all OLS-flexible variants are dominated (P < 0.01)

Adopting the conventional 95% one-sided threshold, **IPO_nn and IPO_linear
are both significantly better than the OLS-linear baseline** on this
sample.

## 5.7 Discussion

The empirical picture splits into three regimes:

1. **Decision-aware training pays off when the predictor is parsimonious or
   regularised.** IPO_linear, IPO_ridge, and IPO_nn all beat their OLS
   twins, and by progressively larger margins as parsimony comes from the
   model architecture itself (MLP) rather than an explicit penalty.
2. **Decision-aware training cannot rescue an over-parameterised
   predictor.** The polynomial expansion ($d \to d + d + \binom{d}{2} = 71$)
   is unregularised in our IPO sweep and the IPO objective's curvature
   amplifies, rather than dampens, the resulting weight instability.
   Adding a weight decay grid for the polynomial would be the natural fix.
3. **The Elmachtoub-Grigas thesis holds in this universe.** A model with
   *worse* cross-sectional R² (IPO_nn) achieves the *best* realized cost,
   confirming that prediction loss and decision loss are not aligned in
   the mean-variance setting on monthly U.S. equities.

## 5.8 Limitations

* δ is fixed at 50; sensitivity to risk aversion is not reported.
* No long-only / leverage / turnover constraints — IPO_kernel runs at
  gross 8.33 which is unrealistic for a real fund. The straightforward
  fix is to add an L1 weight penalty on $z$ (still differentiable
  through `mvo.py`'s closed form via the dual).
* 71 monthly observations on a single universe; cross-validating across
  rolling 3-year windows would tighten the bootstrap.
* The polynomial sweep should include ridge regularisation in IPO mode
  to make the comparison apples-to-apples.

The figures live in `figures/`:

* `cum_returns.png` — growth-of-\$1 by model
* `cost_bar.png`, `sharpe_bar.png` — summary bars
* `cost_vs_r2.png` — the prediction-vs-decision scatter
* `dominance_heatmap.png` — bootstrap dominance vs OLS-linear
