# Nonlinear and Regularised Prediction Models in Integrated Mean-Variance Portfolio Optimisation

> Course project for **ORIE 5370 — Optimization in Finance**.
> Empirical extension of Butler & Kwon (2022), *Integrating prediction in
> mean-variance portfolio optimization* ([arXiv:2102.09287](https://arxiv.org/abs/2102.09287)).

## Abstract

Portfolio construction traditionally follows a two-stage *predict-then-optimise*
paradigm in which expected asset returns are estimated separately from the
downstream mean-variance optimisation (MVO). Recent work in decision-focused
learning argues that such separation is sub-optimal: minimising prediction
error does not necessarily minimise downstream decision cost. This project
investigates whether **nonlinear and regularised** predictive models, trained
end-to-end against realised mean-variance cost (the *integrated prediction
and optimisation*, IPO, paradigm), improve portfolio decisions relative to
a linear OLS plug-in baseline. We evaluate seven predictor families
(linear, ridge, Lasso, elastic net, polynomial, kernel ridge, neural network)
under both training paradigms on a 2005–2024 panel of S&P 500 equities, with
2019–2024 held out for evaluation. The best model — an elastic-net predictor
trained end-to-end against MVO cost — achieves an annualised Sharpe ratio of
**1.18** with a **9.5%** maximum drawdown over the 71-month test period, and
beats the OLS-linear baseline in 100% of paired bootstrap resamples.

## 1. Research question

The proposal poses a single empirical question:

> Do **nonlinear** prediction models improve **decision quality** in
> integrated mean-variance portfolio optimisation compared to linear models?

This study addresses three sub-questions:

1. Does decision-focused (IPO) training reduce out-of-sample mean-variance
   cost relative to the OLS plug-in baseline?
2. How does prediction accuracy relate to downstream portfolio performance?
3. Under what predictive-model classes does increased representational
   flexibility help, and under what classes does it hurt?

## 2. Methodology

We solve the equality-constrained mean-variance program  
$\min_{z \in \mathbb{R}^n}\; -z^{\top}y + \frac{\delta}{2}z^{\top}Vz \quad \text{s.t.}\quad \mathbf{1}^{\top}z=1$

with risk-aversion parameter $\delta = 50$ (matching Butler & Kwon § 4) and
a 60-day Ledoit–Wolf-shrunk sample covariance $V$. Two training paradigms
are compared:

- **OLS plug-in (predict-then-optimise).** Predictor $\hat y = f(x;\theta)$
  is fit by least-squares (Lasso / elastic net via coordinate descent;
  neural network by mini-batch MSE), then $\hat y$ is plugged into the MVO
  program above.

- **IPO (decision-focused).** $f$ is trained end-to-end on the realised mean-variance cost

  $\mathcal{L}(\theta)=\frac{1}{T}\sum_t\left[-z^*(\hat y_t)^{\top}y_t\right.$

  $\left.+\frac{\delta}{2}z^*(\hat y_t)^{\top}V_tz^*(\hat y_t)\right]$,

  using the closed-form differentiable solution

  $z^*(\hat y_t)=\frac{1}{\delta}V_t^{-1}\left(\hat y_t-\lambda_t\mathbf{1}\right)$,

  so the entire pipeline is autodifferentiable in PyTorch without a QP layer.

Seven predictor families are evaluated under both paradigms (14 model ×
paradigm combinations in total):

| Family | Functional form | Penalty (training) |
|---|---|---|
| Linear | $Wx+b$ | none |
| Ridge | $Wx+b$ | $\lambda\lVert W\rVert_2^2$ |
| Lasso | $Wx+b$ | $\lambda\lVert W\rVert_1$ |
| Elastic Net | $Wx+b$ | $\lambda_1\lVert W\rVert_1+\lambda_2\lVert W\rVert_2^2$ |
| Polynomial | $W\phi(x)+b$, $\phi$ adds squares + interactions | none |
| Kernel ridge | $\sum_m \alpha_m K(x,x_m)$, RBF kernel, $M=200$ | $\lambda\lVert\alpha\rVert_2^2$ |
| MLP (1 hidden layer) | $W_2\tanh(W_1x+b_1)+b_2$ | $\lambda\lVert\theta\rVert_2^2$ |

Hyperparameters (ridge $\lambda$, Lasso $\alpha$, elastic-net mix, kernel
$\gamma$, NN hidden width, IPO weight decay, IPO L1 penalty) are selected
on a validation period by minimum realised MVO cost.

### Analytical structure of the IPO extensions

The equality-constrained MVO layer is closed-form for every predictor because
the optimal portfolio depends only on the predicted return vector:  
$z_t^*(\hat y_t)=B_t\hat y_t+a_t$, where  
$B_t=\frac{1}{\delta}\left[V_t^{-1}-\frac{V_t^{-1}\mathbf{1}\mathbf{1}^{\top}V_t^{-1}}{\mathbf{1}^{\top}V_t^{-1}\mathbf{1}}\right]$ and  
$a_t=\frac{V_t^{-1}\mathbf{1}}{\mathbf{1}^{\top}V_t^{-1}\mathbf{1}}$.

This affine structure lets us characterize the IPO estimator for each
predictor class.

For a parameter-linear predictor, $\hat y_t=X_t\beta$, substituting
$z_t^*(\hat y_t)=B_tX_t\beta+a_t$ into the realised MVO cost gives  
$\mathcal L(\beta)=\frac{1}{2}\beta^\top H\beta-g^\top\beta+\text{penalty}(\beta)$,  
where  
$H=\delta\sum_t X_t^\top B_t^\top V_tB_tX_t$ and  
$g=\sum_t X_t^\top B_t^\top(y_t-\delta V_ta_t)$.

Therefore, the Ridge IPO estimator has the closed-form solution  
$\hat\beta_{\text{Ridge IPO}}=(H+\lambda I)^{-1}g$.

For Lasso IPO, the $L_1$ penalty makes the objective convex but nonsmooth:

$$\hat\beta_{\text{Lasso IPO}}=\arg\min_{\beta}\,\frac{1}{2}\beta^\top H\beta-g^\top\beta+\lambda\|\beta\|_1$$

It does not have a simple matrix-inverse solution in general, but it is
characterized by the KKT condition  
$0\in H\hat\beta-g+\lambda\partial\|\hat\beta\|_1$.

Equivalently, componentwise,  
$(H\hat\beta-g)_j=-\lambda\text{sign}(\hat\beta_j)$ if $\hat\beta_j\neq0$, and  
$|(H\hat\beta-g)_j|\leq\lambda$ if $\hat\beta_j=0$.

Elastic Net IPO has the same nonsmooth structure after adding the $L_2$ term:

$$\hat\beta_{\text{EN IPO}}=\arg\min_{\beta}\,\frac{1}{2}\beta^\top(H+\lambda_2 I)\beta-g^\top\beta+\lambda_1\|\beta\|_1$$

with KKT condition  
$0\in(H+\lambda_2I)\hat\beta-g+\lambda_1\partial\|\hat\beta\|_1$.

When $\lambda_1=0$, Elastic Net reduces to Ridge IPO and has the closed form  
$\hat\beta=(H+\lambda_2I)^{-1}g$.

Polynomial IPO also has a closed-form structure because the predictor is
nonlinear in the features but linear in the parameters. If
$\hat y_t=\Phi_t\beta$ and $\Phi_t=\phi(X_t)$, then replacing $X_t$ by
$\Phi_t$ gives  
$H_{\phi}=\delta\sum_t\Phi_t^\top B_t^\top V_tB_t\Phi_t$ and  
$g_{\phi}=\sum_t\Phi_t^\top B_t^\top(y_t-\delta V_ta_t)$,  
so the polynomial IPO solution is  
$\hat\beta_{\text{Poly IPO}}=H_{\phi}^{-1}g_{\phi}$.

With an $L_2$ penalty, this becomes  
$\hat\beta_{\text{Poly Ridge IPO}}=(H_{\phi}+\lambda I)^{-1}g_{\phi}$.

Finite-kernel IPO has the same structure when the anchor points are fixed. With
$\hat y_t=\tilde K_t\tilde\alpha$, the IPO objective is quadratic in
$\tilde\alpha$. With kernel-ridge regularization matrix $D$, the closed-form
solution is  
$\hat{\tilde\alpha}_{\text{Kernel IPO}}=(H_K+\lambda D)^{-1}g_K$,  
where  
$H_K=\delta\sum_t\tilde K_t^\top B_t^\top V_tB_t\tilde K_t$ and  
$g_K=\sum_t\tilde K_t^\top B_t^\top(y_t-\delta V_ta_t)$.

Finally, neural-network IPO does not admit a closed-form estimator because
$f(X_t;\theta)$ is nonlinear in $\theta$. In this case we solve

$$\hat\theta_{\text{NN IPO}}=\arg\min_{\theta}\frac{1}{T}\sum_t\left[-(B_tf(X_t;\theta)+a_t)^\top y_t+\frac{\delta}{2}(B_tf(X_t;\theta)+a_t)^\top V_t(B_tf(X_t;\theta)+a_t)\right]$$

by gradient descent. The gradient is computed through the differentiable MVO
layer, using  
$\frac{\partial z_t^*}{\partial\hat y_t}=B_t$.

Thus, Ridge, polynomial, and fixed-anchor kernel IPO have closed-form
matrix-inverse solutions; Lasso and Elastic Net have convex nonsmooth KKT
characterizations; and neural-network IPO is nonconvex but differentiable
through the same closed-form MVO layer.

The resulting analytical classification is summarized below:

| Model | Nonlinear in inputs? | Linear in parameters? | Closed-form IPO estimator? |
|---|---:|---:|---:|
| Linear | No | Yes | Yes |
| Ridge | No | Yes | Yes |
| Lasso | No | Yes | No, due to $L_1$ nonsmoothness |
| Elastic Net | No | Yes | No, unless $\lambda_1=0$ |
| Polynomial | Yes | Yes | Yes |
| Finite Kernel Ridge | Yes | Yes, if anchors fixed | Yes |
| Neural Network | Yes | No | No |

The theoretical IPO structure maps to the implemented model classes as follows:

| Model | Predictor class | Analytical structure | Implementation |
|---|---|---|---|
| Linear/Ridge | Parameter-linear | Quadratic IPO objective | OLS closed-form; IPO Adam |
| Lasso/Elastic Net | Parameter-linear + $L_1$ | Convex but nonsmooth | sklearn / Adam with $L_1$ penalty |
| Polynomial | Nonlinear features, parameter-linear | Quadratic after feature expansion | Feature lift + linear layer |
| Finite Kernel | Fixed-anchor nonlinear features | Quadratic in kernel weights | Fixed anchors + trainable weights |
| Neural Network | Nonlinear in parameters | Nonconvex IPO objective | Backprop through MVO layer |

## 3. Data

Daily prices for S&P 500 constituents (2005–2024) are merged with the
Fama–French five-factor model, the momentum factor, and the 49-industry
return factors. The investable universe at each rebalance is restricted
to the 100 most-liquid names that traded continuously over the sample.
Cross-sectional features include momentum, short-term reversal, realised
volatility, and log dollar volume, z-scored within each rebalance month.
Portfolios are rebalanced monthly, with realised next-month returns
constructed by compounding intra-month daily returns. The sample is
partitioned into

- **Training:** earliest available month → 2017-01 (≈ 12 years, 141 months)
- **Validation:** 2017-01 → 2019-01 (24 months, used for hyperparameter
  selection and IPO early-stopping)
- **Test:** 2019-01 → 2024-11 (71 months, strictly held out)

## 4. Headline results

Out-of-sample test-period statistics for the four best- and four
worst-performing models. Realised MVO cost is the IPO objective (lower is
better). The "Bootstrap P" column reports the probability that the model
has lower mean cost than the OLS-linear baseline (paired iid bootstrap,
B = 5000); "Gross" is the mean per-month gross exposure (sum of
absolute portfolio weights).

| Model               | MVO cost | Ann. ret | Ann. vol | Sharpe | Max DD | Gross | Bootstrap P |
|---------------------|---------:|---------:|---------:|-------:|-------:|------:|------------:|
| **IPO Elastic Net** | **−0.0011** | 18.2% | 15.5% | **1.18** | **9.5%** | 3.53 | **1.000** |
| OLS Elastic Net     | −0.0010 | 13.3% | 14.5% | 0.91 | 12.4% | 2.84 | 0.995 |
| OLS Lasso           | −0.0010 | 13.3% | 14.5% | 0.91 | 12.4% | 2.84 | 0.995 |
| IPO MLP             | +0.0009 | 13.3% | 15.7% | 0.85 | 14.6% | 3.26 | 0.999 |
| OLS Linear (baseline) | +0.0093 | 13.2% | 19.4% | 0.68 | 21.7% | 4.76 | (baseline) |
| OLS Polynomial      | +0.0481 | 20.7% | 34.3% | 0.60 | 53.8% | 8.28 | 0.000 |
| OLS Kernel          | +0.0524 | 15.9% | 31.3% | 0.51 | 43.8% | 9.40 | 0.000 |
| IPO Polynomial      | +0.0862 |  6.3% | 43.2% | 0.15 | 66.0% | 10.85 | 0.000 |

Three robust empirical findings emerge:

1. **L1 regularisation is the single most important ingredient.** Both
   Lasso and Elastic Net under OLS plug-in already achieve negative
   realised MVO cost, beating the OLS-linear baseline at the 99.5%
   bootstrap level *without any decision-aware training*. The L1
   penalty's variable-selection effect dominates the gain.
2. **Decision-focused (IPO) training adds value precisely when the
   predictor lacks built-in regularisation.** The largest IPO-vs-OLS
   improvements are on the linear predictor (P = 0.91) and the neural
   network (P = 1.00). On L1-regularised predictors the marginal IPO
   improvement is statistically zero. On unregularised flexible models
   (polynomial, kernel) IPO is actively harmful: end-to-end training
   amplifies high-variance lifted-feature noise into the portfolio
   weights, and gross exposure climbs to 10–12.
3. **The Elmachtoub–Grigas (2022) thesis holds in this setting.** Models
   with *worse* mean cross-sectional predictive $R^2$ achieve the *best*
   realised mean-variance cost, while the highest-$R^2$ predictors
   (kernel and MLP under OLS) sit at the worst costs. Improving
   prediction accuracy does not monotonically improve portfolio
   decisions.

The combined best-practice recipe that emerges from the sweep is therefore
**(L1-regularised linear predictor) + (decision-focused training)** — both
ingredients contribute independently, and combining them produces the
strongest out-of-sample model in our suite.

## 5. Repository structure


```
ORIE_5370/
├── README.md                  This document
├── proposal.pdf               Original project proposal
├── Project/                   Stage 1 — data preparation
│   ├── get_data.ipynb         S&P 500 OHLCV scrape, filter to ≥ 2005-01 history
│   ├── analysis_data.ipynb    Feature engineering and exploratory MVO sketch
│   ├── price_cache_manager.py yfinance cache utilities
│   ├── price_cache/           Per-ticker daily OHLCV CSVs (≈ 500 names)
│   └── factor data/           FF5, momentum, and 49-industry daily factor files
├── Modeling/                  Stage 2 — empirical IPO pipeline
│   ├── README.md              Stage-2 documentation and run instructions
│   ├── build_dataset.py       Build per-month (X, y, V) panel with shrinkage covariance
│   ├── mvo.py                 Differentiable equality-constrained MVO solver
│   ├── ipo_models.py          Seven predictor families
│   ├── train.py               OLS plug-in and IPO training with hyperparameter sweeps
│   ├── evaluate.py            Out-of-sample backtest and bootstrap inference
│   ├── make_figures.py        Plots and tables
│   ├── run_all.py             End-to-end orchestrator
│   ├── figures/               Six PNG figures referenced in the report
│   ├── results/               Performance, coefficient, and pairwise dominance tables
│   └── report/
│       ├── literature_review.md   §II — prior work
│       ├── methodology.md         §III–IV — data, predictors, training, evaluation
│       ├── results.md             §V–VI — empirical results and discussion
│       └── conclusion.md          §VII — conclusions and future work
└── Theoretical Foundation/    Analytical support for nonlinear IPO extensions
    ├── Closed-Form Structure for Nonlinear IPO Extensions.*
    └── Nonlinear Features with Parameter-Linear Structure.*
```

## 6. Reproducing the results

```bash
git clone https://github.com/xiayawen/ORIE_5370.git
cd ORIE_5370/Modeling

# Symlink the team data files from the sibling Project/ folder.
ln -s ../Project/price_cache .
ln -s "../Project/factor data" "factor data"

# Build the panel dataset (~30 s on a laptop).
python build_dataset.py

# Run the full pipeline: train → evaluate → figures (~4 minutes).
python run_all.py --skip-data
```

`run_all.py` automatically sets `KMP_DUPLICATE_LIB_OK=TRUE` and pins
OpenMP / MKL threading to a single thread, which avoids a deadlock we
observed in Anaconda + PyTorch on macOS during the MLP training stage.

The four files in `Modeling/report/` cite numbers directly from the CSV
artefacts in `Modeling/results/` and embed the figures in
`Modeling/figures/`. Concatenating the four files in the order
`literature_review.md → methodology.md → results.md → conclusion.md`
yields the final write-up.

## 7. Limitations and future work

The empirical setting has several deliberate simplifications, each of
which constitutes a natural extension:

- **Universe size.** We use the 100 most-liquid S&P 500 names; the
  proposal envisaged 300, and a longer historical window would tighten
  the small-sample bootstrap. Extending to a CRSP-scale 3,000-name
  universe is computationally feasible but was outside scope.
- **Risk model.** The covariance $\hat\Sigma$ is held fixed at the
  Ledoit–Wolf shrinkage estimator. Joint decision-focused estimation of
  $(\hat\mu, \hat\Sigma)$ is a natural follow-up.
- **Constraints.** The MVO program is equality-constrained only.
  Long-only, leverage, and turnover constraints would compress the
  spread between models on raw cost but possibly amplify it on
  net-of-cost return.
- **Distributional robustness.** Even with IPO, point estimates of
  $\hat\mu$ are fragile under fat tails. A distributionally-robust IPO
  in the spirit of Costa & Iyengar (2023) is a natural bridge to the
  *end-to-end distributionally robust portfolio construction* literature.
- **Polynomial regularisation.** The polynomial sweep uses zero IPO
  regularisation. Adding a Lasso-style penalty to the lifted-feature
  coefficients would likely move IPO Polynomial up several positions in
  the ranking.

## 8. References

- Markowitz, H. (1952). Portfolio selection. *The Journal of Finance*, 7(1), 77–91.
- DeMiguel, V., Garlappi, L., & Uppal, R. (2009). Optimal versus naive
  diversification: How inefficient is the 1/N portfolio strategy?
  *The Review of Financial Studies*, 22(5), 1915–1953.
- Bertsimas, D., & Kallus, N. (2020). From predictive to prescriptive
  analytics. *Management Science*, 66(3), 1025–1044.
- Donti, P. L., Amos, B., & Kolter, J. Z. (2017). Task-based end-to-end
  model learning in stochastic optimization. *NeurIPS*, 30.
- Elmachtoub, A. N., & Grigas, P. (2022). Smart "predict, then optimize."
  *Management Science*, 68(1), 9–26.
- Butler, A., & Kwon, R. H. (2022). Integrating prediction in mean-variance
  portfolio optimization. arXiv:2102.09287.
- Gu, S., Kelly, B., & Xiu, D. (2020). Empirical asset pricing via machine
  learning. *The Review of Financial Studies*, 33(5), 2223–2273.
- Krauss, C., Do, X. A., & Huck, N. (2017). Deep neural networks,
  gradient-boosted trees, random forests: Statistical arbitrage on the
  S&P 500. *European Journal of Operational Research*, 259(2), 689–702.
- Fischer, T., & Krauss, C. (2018). Deep learning with long short-term
  memory networks for financial market predictions. *European Journal
  of Operational Research*, 270(2), 654–669.
- Ledoit, O., & Wolf, M. (2004). A well-conditioned estimator for
  large-dimensional covariance matrices. *Journal of Multivariate
  Analysis*, 88(2), 365–411.
