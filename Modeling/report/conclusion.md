# §VII Conclusion

This project asked whether nonlinear prediction models improve decision
quality in the integrated prediction-and-optimisation framework relative
to the linear baseline of [Butler & Kwon (2022)][butler] on a U.S. equity
mean-variance portfolio problem. Across seven predictor families — linear,
ridge, Lasso, Elastic Net, polynomial, kernel ridge, single-hidden-layer
neural network — and two training paradigms (OLS plug-in and end-to-end
IPO), we find several robust empirical patterns.

[butler]: https://arxiv.org/abs/2102.09287

**1. L1 regularisation is the single most important ingredient.** Both
Lasso and Elastic Net under OLS plug-in already achieve negative realised
mean MVO cost on the test set, beating the OLS-linear baseline at the
99.5% bootstrap level *without any decision-aware training*. Variable
selection — zeroing out the noisy cross-sectional features — is doing
most of the heavy lifting in this universe.

**2. Decision-focused training adds value precisely when the predictor
lacks built-in regularisation.** The IPO-trained linear and single-
hidden-layer neural-network models beat their OLS twins with bootstrap
probabilities 0.91 and 1.00 respectively, while the IPO improvement
over OLS for the L1-regularised predictors is statistically zero
(Lasso) or marginal (Elastic Net). For un-regularised flexible
predictors (polynomial, kernel), end-to-end IPO training is actively
*harmful*: the realised cost gets *worse* than the OLS plug-in because
the IPO loss amplifies high-variance lifted-feature noise into the
portfolio weights.

**3. The best-performing model combines both ingredients.** The IPO-
trained Elastic Net achieves Sharpe 1.18 with a 9.5% maximum drawdown
over the 71-month test period — the lowest realised cost, lowest
drawdown, and highest Sharpe of any model we evaluate. It dominates
every other model in the bootstrap pairwise comparison.

**4. The Elmachtoub-Grigas thesis holds in this setting.** Models with
*worse* mean cross-sectional predictive $R^2$ (IPO Elastic Net, OLS
Lasso, IPO MLP) achieve the *best* realised mean-variance cost, while
the highest-$R^2$ OLS predictors (kernel, MLP) sit at the *worst* costs.
This quantitatively confirms the central claim of decision-focused
learning: prediction loss and decision loss are not aligned in the
mean-variance portfolio problem.

**5. Stable covariance conditioning is a strict prerequisite.** Our empirical
ablation study reveals that in $p > n$ regimes, replacing globally regularised
linear shrinkage (Ledoit-Wolf 2004) with local analytical nonlinear shrinkage
(Ledoit-Wolf 2020) causes catastrophic downstream MVO failure due to near-zero
eigenvalues. However, L1-regularised predictors (Lasso) uniquely survive this
ill-conditioning, acting as a mathematical circuit breaker against explosive
inversion noise.

## Practical implications

For a practitioner deciding between predict-then-optimize and IPO on a
similar mean-variance portfolio problem:

* **The default starting point should be an L1-regularised linear
  predictor** (Lasso or Elastic Net). On its own it already beats the
  OLS-linear baseline; layered with IPO training (Elastic Net + IPO)
  it gives the strongest result we observed. Both stages are cheap.
* **Use IPO when the predictor has no built-in regularisation.** If
  interpretability or pipeline simplicity rules out L1, the linear /
  ridge / NN families benefit from end-to-end decision-focused
  training. The closed-form gradient through the equality-constrained
  MVO solution makes this engineering-cheap.
* **Condition your covariance globally in $p > n$ regimes.** Do not use
  local nonlinear shrinkage methods for covariance matrices unless you
  apply a strict positive-definite floor or use an L1-regularised predictor
  to safeguard against inversion explosions. The classic Ledoit-Wolf (2004)
  linear shrinker remains the safest default for MVO.
* **Avoid unregularised polynomial / kernel expansions.** Both are
  bottom of the table under both training paradigms. The IPO
  objective's curvature combined with the lifted feature space
  amplifies rather than dampens estimation noise; the resulting
  weights are extreme. Adding an L1 penalty on the lifted features
  would be the natural fix.

## Limitations and future work

Our empirical setting has well-known limitations that the discussion
of [§VI of the proposal][outline] flagged in advance:

[outline]: ./methodology.md

* **Universe size and composition.** We use the 100 most-liquid S&P 500
  stocks; the proposal envisaged 300, and a longer historical window
  would improve the small-sample bootstrap. Extending to the full
  3,000-name CRSP universe of [Gu, Kelly & Xiu (2020)][gukelly] would
  also stress-test whether the NN's regularisation effect persists at
  scale.
* **Risk model.** While we explored both 2004 linear and 2020 analytical
  nonlinear shrinkage estimators, we held the covariance matrix fixed prior
  to the MVO layer. Joint estimation of $(\hat\mu, \hat\Sigma)$
  with a decision-focused loss is a natural extension and is one of the
  *Unsolved Questions* listed in our outline.
* **Constraints.** The MVO program is equality-constrained only; long-
  only / leverage / turnover constraints are absent. Practitioners would
  almost certainly cap gross exposure at 1 or 2, which would compress
  the visible spread between models on raw cost but possibly amplify the
  spread on after-cost return. Adding a transaction-cost-aware IPO loss
  is the most directly publishable follow-up.
* **Regime stability.** Our test period (2019-2024) contains the COVID
  shock, the 2022 inflation regime, and the 2023-24 AI-led equity
  rally. A formal regime-conditional analysis (e.g. splitting the
  test set by VIX terciles) would clarify whether the IPO advantage is
  state-dependent.
* **Distributional robustness.** Even with IPO, point estimates of
  $\hat\mu$ are fragile under fat tails. A distributionally-robust IPO
  in the spirit of [Costa & Iyengar (2023)][costa] would be a natural
  bridge to the *Distributionally Robust End-to-End Portfolio
  Construction* line of work cited in our reading list.

[gukelly]: https://doi.org/10.1093/rfs/hhaa009
[costa]: https://arxiv.org/abs/2206.05134

## Summary

The headline of the project is short. **The decision-quality gain over
the OLS-linear baseline comes from two independent ingredients:
(i) L1-style variable selection in the predictor, and (ii) end-to-end
decision-focused training.** The two combine multiplicatively: the
IPO-trained Elastic Net predictor achieves Sharpe 1.18 with a 9.5%
maximum drawdown — better than any model in the sweep that uses only
one of the two ingredients. Pure flexibility (kernel, polynomial)
without either ingredient is harmful, and these gains strictly depend
on a well-conditioned covariance matrix foundation.

We therefore answer the central question of the proposal — *Do
nonlinear prediction models improve decision quality?* — with a
nuance: **regularised models do, unregularised flexible models do not,
and the regularisation matters at least as much as the nonlinearity.**
The single most useful empirical conclusion is that the venerable
combination of L1 variable selection and decision-aware training is
hard to beat on this universe.