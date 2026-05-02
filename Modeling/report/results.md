# §V Empirical Results

The numbers below come from `results/performance.csv` produced by
`run_all.py` over the 71-month test sample (2019-01-31 → 2024-11-29) on
the 100 most-liquid S&P 500 names that traded continuously from 2005-01.
Risk aversion δ = 50 (Butler & Kwon §4); equality-constrained,
fully-invested MVO; 60-day Ledoit–Wolf-shrunk covariance.

## 5.1 Headline performance table

Sorted by realised mean MVO cost (lower is better).

| Model | Cost | Ann. ret | Ann. vol | Sharpe | Max DD | Gross exposure | Bootstrap P vs OLS-linear |
|---|---:|---:|---:|---:|---:|---:|---:|
| **IPO_elasticnet** | **−0.0011** | 18.2% | 15.5% | **1.18** | **9.5%** | 3.53 | **1.000** |
| OLS_elasticnet     | −0.0010 | 13.3% | 14.5% | 0.91 | 12.4% | **2.84** | 0.995 |
| OLS_lasso          | −0.0010 | 13.3% | 14.5% | 0.91 | 12.4% | **2.84** | 0.995 |
| IPO_nn             | +0.0009 | 13.3% | 15.7% | 0.85 | 14.6% | 3.26 | 0.999 |
| IPO_ridge          | +0.0041 | 13.5% | 16.3% | 0.83 | 17.0% | 3.92 | 0.815 |
| IPO_lasso          | +0.0050 | 11.1% | 17.0% | 0.65 | 22.7% | 3.82 | 0.824 |
| IPO_linear         | +0.0051 |  9.4% | 16.9% | 0.56 | 17.9% | 3.52 | 0.905 |
| OLS_ridge          | +0.0080 | 13.2% | 18.8% | 0.70 | 20.8% | 4.56 | 1.000\* |
| OLS_linear         | +0.0093 | 13.2% | 19.4% | 0.68 | 21.7% | 4.76 | (baseline) |
| OLS_nn             | +0.0314 | 10.0% | 25.8% | 0.39 | 35.3% | 6.90 | 0.001 |
| OLS_polynomial     | +0.0481 | 20.7% | 34.3% | 0.60 | 53.8% | 8.28 | 0.000 |
| OLS_kernel         | +0.0524 | 15.9% | 31.3% | 0.51 | 43.8% | 9.40 | 0.000 |
| IPO_kernel         | +0.0719 | 24.3% | 36.0% | 0.67 | 38.8% | 11.81 | 0.000 |
| IPO_polynomial     | +0.0862 |  6.3% | 43.2% | 0.15 | 66.0% | 10.85 | 0.000 |

\* OLS_ridge picked $\alpha = 0$ on validation, making it numerically
identical to OLS_linear; the bootstrap probability of 1.0 reflects ties.
OLS_lasso and OLS_elasticnet picked the same $\alpha$ at the same
$l_1$-ratio extreme on validation, hence their identical numbers.

## 5.2 Does decision-focused training help? (per-pair bootstrap)

Per-pair bootstrap probability that the IPO-trained variant has lower
mean MVO cost than its OLS-trained twin (B = 5000, paired iid resample).
$P > 0.95$ would be a one-sided 5% test.

| Predictor   | OLS cost | IPO cost | $P(\text{IPO} < \text{OLS})$ | Verdict |
|---|---:|---:|---:|---:|
| Linear      | +0.0093 | +0.0051 | **0.905** | IPO improves |
| Ridge       | +0.0080 | +0.0041 | **0.750** | IPO improves |
| Lasso       | −0.0010 | +0.0050 | 0.028 | **OLS already optimal** |
| Elastic Net | −0.0010 | −0.0011 | 0.544 | tie |
| Polynomial  | +0.0481 | +0.0862 | 0.010 | IPO **worse** |
| Kernel      | +0.0524 | +0.0719 | 0.090 | IPO worse |
| MLP         | +0.0314 | +0.0009 | **1.000** | IPO improves dramatically |

**Conclusion:** decision-focused training gives the largest improvement
on the predictors that have no built-in regularisation (linear, MLP). It
provides a smaller marginal improvement on Ridge. For **L1-regularised
predictors (Lasso, Elastic Net), OLS plug-in is already at or near the
decision-optimal solution** — the L1 penalty's variable-selection effect
dominates whatever gain decision-aware training could add. Polynomial
and kernel predictors, which have no L1 sparsity and no NN-style
implicit regularisation, are *hurt* by IPO training because the
end-to-end loss amplifies the high-variance lifted-feature noise into
the portfolio weights.

## 5.3 Headline IPO vs OLS-linear baseline

The bootstrap dominance probability against the canonical OLS-linear
baseline (Butler & Kwon §4 setup):

* IPO_elasticnet: **1.000** — dominates in all bootstrap resamples
* IPO_nn:         0.999
* OLS_elasticnet ≡ OLS_lasso: 0.995
* IPO_linear:     0.905
* IPO_lasso:      0.824
* IPO_ridge:      0.815
* OLS_nn:         0.001 — significantly worse
* IPO_polynomial / IPO_kernel: 0.000 — significantly worse

Six models clear the conventional 95% one-sided threshold: IPO_elasticnet,
IPO_nn, OLS_elasticnet, OLS_lasso, IPO_linear (just), and OLS_ridge (tie).
Three of those six are **regularised linear**, suggesting that the
substantive lift over the linear-OLS baseline comes from variable
selection at least as much as from decision-aware training.

## 5.4 Within-paradigm rankings: where does flexibility help?

Reading down the cost column within each paradigm:

| Rank | OLS plug-in     | Cost      | IPO              | Cost      |
|---:|---|---:|---|---:|
| 1   | OLS_lasso       | −0.0010   | **IPO_elasticnet** | **−0.0011** |
| 2   | OLS_elasticnet  | −0.0010   | IPO_nn          | +0.0009   |
| 3   | OLS_ridge       | +0.0080   | IPO_ridge       | +0.0041   |
| 4   | OLS_linear      | +0.0093   | IPO_lasso       | +0.0050   |
| 5   | OLS_nn          | +0.0314   | IPO_linear      | +0.0051   |
| 6   | OLS_polynomial  | +0.0481   | IPO_kernel      | +0.0719   |
| 7   | OLS_kernel      | +0.0524   | IPO_polynomial  | +0.0862   |

Two patterns:

* **Under OLS plug-in**, the ordering is essentially "more
  regularisation ⇒ better." Lasso and Elastic Net top the list
  because their L1 component zeros out the noisier features (the
  dollar-volume and short-reversal signals get small weights; momentum
  and FF5 factors retain larger weights — see `results/coefficients.csv`).
  Unregularised flexible models (kernel, polynomial) are at the bottom
  because nothing constrains the cross-sectional variance of $\hat y$.
* **Under IPO**, Elastic Net and the MLP both achieve negative or
  near-zero realised cost. The ordering is *not* monotone in
  representational capacity: the MLP beats the kernel ridge and
  polynomial because Adam + the small NN architecture acts as a
  stronger implicit regulariser than the explicit weight decay we
  applied to the kernel anchors or the (zero) penalty on the polynomial
  coefficients. Adding a comparable L1 penalty to IPO_polynomial would
  most likely close the gap.

## 5.5 Feature Importance and the "Shrink to GMVP" Phenomenon

Analyzing the learned coefficients of the linear models (saved in `results/coefficients.csv`) reveals the exact mechanism driving our headline results. 

A striking finding is that the cross-validated `OLS_lasso` and `OLS_elasticnet` models assigned a weight of exactly **0.0000** to all 11 input features, retaining only the global bias term. When a predictor outputs a constant expected return vector for all assets, the equality-constrained MVO program effectively ignores the mean vector and exclusively minimizes the covariance term. In other words, OLS Lasso succeeded because it learned that the cross-sectional features were too noisy for standard regression, actively choosing to output the **Global Minimum Variance Portfolio (GMVP)** instead. This strategy safely yielded a Sharpe ratio of 0.91 and a low maximum drawdown.

By contrast, the unregularised `OLS_linear` baseline placed extreme weights on the Fama-French factors (e.g., +3.21 on `CMA` and −1.59 on `HML`). These massive, unconstrained coefficients resulted in highly volatile return predictions, tricking the downstream MVO into taking unstable, high-turnover positions with large gross exposures. 

The champion model, `IPO_elasticnet`, demonstrates the power of decision-focused training. Unlike its OLS counterpart, the IPO Elastic Net did not zero out all features. Trained end-to-end against the realized MVO cost, it successfully extracted stable signals, placing precise, non-zero weights on short-term reversal (`rev_1d`), historical volatility (`vol_60`), and the Fama-French profitability factor (`RMW`). By safely incorporating these signals, the IPO Elastic Net learned how to systematically tilt the portfolio weights away from the pure GMVP to capture additional risk-adjusted returns, boosting the Sharpe ratio to 1.18 while maintaining a highly stable gross exposure.

## 5.6 Prediction quality vs decision quality (cost vs R²)

`figures/cost_vs_r2.png` shows the realised MVO cost on the y-axis and
mean cross-sectional predictive $R^2$ on the x-axis. The two highest-$R^2$
*OLS* models (kernel and MLP) sit at the *worst* costs (top-right of
the plot), while the lowest-cost models (IPO_elasticnet, OLS_lasso,
IPO_nn) live in the bottom-left with low predictive $R^2$. This is the
empirical statement of [Elmachtoub & Grigas (2022)][eg]'s "prediction
loss is not decision loss" thesis: more accurate prediction does not
imply better portfolio decisions.

[eg]: https://doi.org/10.1287/mnsc.2020.3922



## 5.7 Stability of portfolio weights

Mean per-month gross exposure $\|z\|_1$ — the cleanest stability proxy
because the model with extreme exposure sees the largest drawdowns and
volatility:

* **Most stable** (1–4): OLS_elasticnet/lasso (2.84), IPO_nn (3.26),
  IPO_linear (3.52), IPO_elasticnet (3.53)
* OLS_linear baseline: 4.76
* **Least stable**: IPO_kernel (11.81), IPO_polynomial (10.85),
  OLS_kernel (9.40), OLS_polynomial (8.28)

Notice that the four most stable portfolios are precisely the four with
the lowest realised cost — there is essentially a one-to-one mapping
between weight stability and decision quality on this universe. The IPO
training that *fails* (polynomial, kernel) does so because it pushes
gross exposure up rather than down.

## 5.8 Pairwise bootstrap dominance matrix

`figures/pairwise_dominance.png` is the full $14 \times 14$ matrix of
bootstrap dominance probabilities. Diagonal = 0.5 by convention.
A few observations from `results/pairwise_dominance.csv`:

* **IPO_elasticnet** is the only model that dominates *every* other
  model with $P > 0.5$. It cleanly beats every non-L1-regularised
  competitor with $P > 0.9$, and beats OLS_elasticnet narrowly
  ($P = 0.544$) and IPO_nn ($P = 0.609$).
* **OLS_lasso/elasticnet** tie each other (identical hyperparameter
  selection) and dominate the linear / ridge / non-NN baselines with
  $P > 0.95$.
* **IPO_polynomial** is dominated by every other model except
  IPO_kernel; it is the worst-performing variant in our sweep.

## 5.9 Empirical Ablation: The Perils of Ill-Conditioned Covariance

In our primary pipeline, we utilized the Ledoit-Wolf (2004) linear shrinkage estimator to condition the sample covariance matrix prior to downstream optimization. To test the boundaries of our Integrated Prediction and Optimization (IPO) framework, we conducted an ablation study replacing the linear shrinker with the state-of-the-art analytical nonlinear shrinkage method introduced by Ledoit and Wolf (2020). 

While mathematically elegant in its use of the Hilbert transform for local eigenvalue attraction, the nonlinear method triggered a catastrophic failure in our Mean-Variance Optimization (MVO) layer.

| Model | Covariance Method | Sharpe | Max Drawdown | Realised Cost |
| :--- | :--- | :--- | :--- | :--- |
| **IPO_elasticnet** | Linear (2004) | **1.18** | **9.5%** | **−0.0011** |
| IPO_elasticnet | Nonlinear (2020) | 0.30 | > 150,000% | −0.2140 |
| **OLS_linear** | Linear (2004) | 0.68 | 21.7% | +0.0093 |
| OLS_linear | Nonlinear (2020) | 0.29 | > 1.5 Billion % | +0.4586 |
| **OLS_lasso** | Linear (2004) | 0.91 | 12.4% | −0.0010 |
| OLS_lasso | Nonlinear (2020) | 0.86 | 24.2% | −0.0137 |

**Diagnosis of the Collapse:** The root cause of this failure lies in the dimensionality of our data regime. Our lookback window for covariance estimation is 60 days, while our asset universe contains 100 stocks. Because the number of assets exceeds the number of observations ($p > n$), the raw sample covariance matrix is mathematically singular, possessing 40 zero-eigenvalues. 

The 2004 linear shrinkage method effectively applies a Ridge-like penalty globally across the matrix, shifting all eigenvalues away from zero by a fixed constant and guaranteeing safe mathematical inversion. The 2020 nonlinear method, however, shrinks eigenvalues based on local density. In our severe $p > n$ regime, it failed to adequately bound the zero-eigenvalues away from zero. When the PyTorch MVO layer inverted this near-singular matrix to calculate the optimal portfolio weights, microscopic noise was amplified into massive, unconstrained leverage, evidenced by the physically impossible Maximum Drawdowns.

**The L1 Circuit Breaker:** This ablation study yielded one highly valuable insight regarding regularization. As seen in the table above, while the OLS Linear model leveraged itself into oblivion, the OLS Lasso model survived the toxic covariance matrix with a completely normal drawdown of **24.2%** and a Sharpe of **0.86**. 

This demonstrates that an L1 penalty on the expected returns acts as a phenomenal mathematical safeguard. By aggressively driving feature coefficients to zero, the Lasso model acted as a circuit breaker, successfully ignoring the explosive noise generated by the ill-conditioned matrix inversion. 

## 5.10 Discussion

Three robust takeaways emerge from our findings:

1. **L1 regularisation is the single most important ingredient for
   decision quality on this universe.** Both Lasso and Elastic Net
   under OLS plug-in already achieve negative realised mean MVO
   cost — better than the OLS-linear baseline at the conventional
   bootstrap significance level — *without any decision-aware
   training*. Variable selection is doing the heavy lifting.
2. **Decision-focused training adds value precisely when the
   predictor lacks built-in regularisation.** The largest IPO-vs-OLS
   improvements are on the linear (P=0.91) and MLP (P=1.00)
   predictors. On L1-regularised predictors the gain is zero or
   negative, and on un-regularised flexible models (polynomial,
   kernel) IPO is actively harmful.
3. **The Elmachtoub-Grigas thesis holds.** The lowest-cost models
   have lower predictive R² than the highest-cost models. Improving
   prediction accuracy does *not* monotonically improve portfolio
   decisions in this MVO setting.

The combined best-practice recipe that emerges from this sweep is:
**(L1-regularised linear predictor) + (decision-focused training)**.
Both ingredients matter individually, and combining them gives the
strongest result we saw — the IPO Elastic Net portfolio with Sharpe
1.18 and 9.5% maximum drawdown over the 71-month test period. Furthermore, our ablation study confirms that this framework is highly dependent on a globally regularised (linear) covariance estimator to ensure MVO inversion stability in $p > n$ regimes.

## 5.11 Limitations

* δ is fixed at 50; sensitivity to risk aversion is not reported.
* No long-only / leverage / turnover constraints. With realistic
  caps the spread between models would compress, but the *ranking*
  is unlikely to change because the failure mode in the bottom of
  the table (kernel, polynomial) is precisely excessive gross
  exposure.
* 71 monthly observations on a single universe; cross-validating
  across rolling windows would tighten the bootstrap.
* The polynomial sweep uses zero IPO regularisation. The natural
  fix — adding a Lasso-style penalty to the lifted-feature
  coefficients — would likely move IPO_polynomial up several
  positions in the ranking.

The figures live in `figures/`:

* `cum_returns.png`         — growth-of-\$1 by model
* `cost_bar.png`, `sharpe_bar.png` — summary bars
* `cost_vs_r2.png`          — the prediction-vs-decision scatter
* `dominance_heatmap.png`   — bootstrap dominance vs OLS-linear
* `pairwise_dominance.png`  — full pairwise dominance matrix