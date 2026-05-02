# §II Literature Review

This project sits at the intersection of three literatures: **classical
mean-variance portfolio construction**, **decision-focused (integrated)
learning**, and **nonlinear machine-learning approaches to asset return
prediction**. The proposal motivates the central research question by
arguing that the third stream has not been systematically evaluated under
the second stream's training paradigm. We organise the prior work along
the same three axes.

## 2.1 Predict-then-optimize portfolio construction

The classical mean-variance framework of [Markowitz (1952)][markowitz]
forms the foundation of modern portfolio theory. Practitioners typically
estimate expected returns separately from the optimisation, plug the
estimates into a quadratic program, and obtain weights. This
two-stage paradigm dominates both academic research and industry
applications.

[Elton & Gruber (1995)][elton] codify the now-standard pipeline of
covariance estimation followed by mean-variance optimisation. Because
the optimisation is highly sensitive to estimation error in the mean
vector, [DeMiguel, Garlappi & Uppal (2009)][demiguel] famously show that
the naive equal-weight (1/N) rule is competitive with — and often
strictly dominates — optimised portfolios out-of-sample on a wide range
of empirical tests, attributing the gap to estimation error rather than
to a defect in mean-variance theory itself.

[markowitz]: https://doi.org/10.2307/2975974
[elton]: https://www.amazon.com/Modern-Portfolio-Theory-Investment-Analysis/dp/0470050829
[demiguel]: https://doi.org/10.1093/rfs/hhn099

## 2.2 Decision-focused learning and integrated optimization

A separate literature in operations research and machine learning has
shown that the predict-then-optimize separation can be sub-optimal even
when predictions are unbiased. [Bertsimas & Kallus (2020)][bertsimas]
introduce the *prescriptive* framework, demonstrating that minimising
the conditional expectation error of a forecaster does not in general
minimise the downstream decision cost. [Donti, Amos & Kolter (2017)]
[donti] formalise this in the stochastic-programming setting and show
empirically that maximum-likelihood training is dominated by direct
end-to-end training on the operational cost.

The most directly relevant work is [Elmachtoub & Grigas (2022)][elmachtoub],
the so-called "Smart Predict-then-Optimize" (SPO) framework, which
constructs a convex surrogate of the realised decision loss and proves
consistency results for linear predictors. Their experiments show that
models with *higher* prediction MSE can produce *lower* downstream
decision cost — a counter-intuitive but central observation that we
visualise empirically in §V (`figures/cost_vs_r2.png`).

[Butler & Kwon (2022)][butler] specialise the SPO idea to mean-variance
portfolio optimisation. They derive a closed-form linear "Integrated
Prediction and Optimisation" (IPO) estimator, whose first-order
condition jointly solves the regression and portfolio problems, and
they show on a U.S. equity universe that IPO modestly but consistently
beats OLS plug-in estimation on realised mean-variance cost. Our
project takes this paper as the baseline and asks whether the linear
restriction of the predictor is essential or whether more flexible
predictors can extract additional decision-relevant signal.

[bertsimas]: https://doi.org/10.1287/mnsc.2018.3253
[donti]: https://papers.nips.cc/paper/2017/hash/3fc2c60b5782f641f76bcefc39fb2392-Abstract.html
[elmachtoub]: https://doi.org/10.1287/mnsc.2020.3922
[butler]: https://arxiv.org/abs/2102.09287

## 2.3 Nonlinear machine learning in asset prediction

Parallel to developments in decision-focused learning, a growing
literature applies nonlinear machine-learning methods to asset return
prediction. [Gu, Kelly & Xiu (2020)][gukelly] benchmark a wide suite
of models — generalised linear, regularised linear, principal
components regression, partial least squares, random forests, gradient
boosting, and feed-forward neural networks — on a large U.S. equity
panel. They report that neural networks and tree-based models
achieve substantially higher cross-sectional $R^2$ on monthly returns
than the linear factor models that dominate the academic asset-pricing
literature.

[Krauss, Do & Huck (2017)][krauss] and [Fischer & Krauss (2018)]
[fischer] specifically study neural-network return classifiers on
S&P 500 daily returns and document statistically significant
out-of-sample predictability that translates into trading-rule profits.

The common thread in this stream is that the predictive performance of
nonlinear models, evaluated on standard $R^2$ or directional-accuracy
metrics, is consistently better than that of linear factor regressions.
What this literature does *not* settle is whether the additional
predictive accuracy translates into better portfolio decisions — the
question §2.2 raises. Both lines of work also typically rely on
separate prediction and optimisation stages, leaving the
representational-flexibility-versus-decision-quality interaction
unexplored.

[gukelly]: https://doi.org/10.1093/rfs/hhaa009
[krauss]: https://doi.org/10.1016/j.ejor.2016.10.031
[fischer]: https://doi.org/10.1016/j.ejor.2017.11.054

## 2.4 Covariance estimation in high dimensions

A critical vulnerability in the classical mean-variance framework is the inversion of the covariance matrix. When the number of assets $p$ is large relative to the observation window $n$ (the $p > n$ regime), the sample covariance matrix is mathematically singular and its inverse amplifies estimation noise into extreme portfolio weights. 

To resolve this, Ledoit & Wolf (2004) introduced linear shrinkage, which globally pulls the sample covariance matrix towards a well-conditioned target (such as a scaled identity matrix), guaranteeing mathematical invertibility. This approach remains the industry standard for stable portfolio optimisation. 

More recently, Ledoit & Wolf (2020) introduced *analytical nonlinear shrinkage*, a state-of-the-art methodology that uses the Hilbert transform of the limiting spectral density to apply local, non-linear attraction to the sample eigenvalues. While mathematically optimal under large-dimensional asymptotics, the interplay between these advanced nonlinear covariance estimators and integrated prediction models (IPO) remains unexplored. Our project addresses this gap directly via an empirical ablation study.

[ledoit2004]: https://doi.org/10.1016/S0047-259X(03)00096-4
[ledoit2020]: https://doi.org/10.1214/19-AOS1921

## 2.5 Where this project fits

We treat Butler & Kwon (2022) as the linear IPO benchmark, and we
augment its predictor class with the regularised linear models
(Lasso and Elastic Net), polynomial expansions, kernel ridge, and
single-hidden-layer neural networks that the proposal §IV specifies.
Each predictor is trained in two paradigms — OLS plug-in and
end-to-end IPO. Furthermore, we test this pipeline across both linear 
and nonlinear covariance shrinkage regimes. 

Evaluated on realised mean-variance cost, risk-adjusted return, drawdown, 
and bootstrap dominance against the linear-OLS baseline, the empirical 
question is therefore: does the flexibility documented in §2.3 produce 
decision-quality gains in the sense of §2.2, or does it amplify the 
estimation-error problem flagged in §2.1 and §2.4?
