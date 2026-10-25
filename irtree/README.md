# irtree

**Bayesian IRTree estimation via hybrid Gibbs/MH MCMC.**

An IRTree models an observed response category as the endpoint of a *sequence of
latent decisions*, each governed by its own trait, rather than a single ability
dimension. This package fits such models by MCMC. Probit nodes use Albert–Chib
data augmentation, so their parameters are drawn by exact **Gibbs** steps:
traits, difficulties, GRM thresholds, discriminations under the default
truncated-Normal prior, 3PL guessing (Béguin–Glas), covariate effects and missing
covariate values. **Metropolis–Hastings** is used only where no closed-form
conditional exists: every parameter of logit, GPCM and custom nodes,
discriminations under a LogNormal prior, non-conjugate variances and LKJ
correlations. Extra moves (a Cowles threshold step, a location move and
interweaving between covariate effects and the parameters they shift) keep the
chains mixing.

It was built for the *IRTree Model of Disengagement* (jointly estimating
test-taking effort and ability from rapid-guessing patterns), but supports
arbitrary trees, node models, covariates, parameter sharing, and missing data.

---

## Install

```bash
pip install -e .              # core: numpy + scipy only
pip install -e ".[full]"      # + pandas tables, matplotlib plots, formula covariates
pip install -e ".[dev]"       # + test/notebook/lint tooling
```

Requires Python ≥ 3.10. The sampler itself depends only on **numpy** and
**scipy**. `pandas` (summary tables and `irtree.load_data`), `matplotlib`
(trace/posterior plots) and `patsy` (covariate formulas, the `formula` extra) are
optional and imported only when used.

---

## 60-second example

```python
import numpy as np
import irtree

# Y is a [persons x items] matrix of observed categories (1, 2, 3 here):
#   1 = rapid guess, 2 = engaged-incorrect, 3 = engaged-correct
spec  = irtree.disengagement_tree(binary=True)          # 2-node tree
data  = irtree.expand_data(Y, spec, categories_start=1)  # -> pseudo-responses
nodes = [irtree.ProbitRaschNode(),   # node 1: disengagement
         irtree.ProbitRaschNode()]   # node 2: ability

fit = irtree.irtree_fit(data, spec, nodes, n_iter=2000, n_chains=4, seed=42)

fit.summary()        # posterior EAP, SD, 95% intervals per parameter
fit.diagnostics()    # R-hat, ESS, Geweke, Heidelberger–Welch
fit.plot_trace()     # (needs matplotlib)
fit.to_logit()       # the summary on the logistic-equivalent scale (D = 1.702)
```

Warm-up is automatic by default (`warmup_mode="evidence"`): each MH block adapts
until its acceptance rate stabilizes, and draws are kept once every block has
settled. A run needs `n_iter >= 1600` in this mode; for a short diagnostic run use
`warmup_mode="fixed", n_warmup=...`.

### A richer model

```python
# 2PL ability node, person covariates on disengagement, estimated correlation
spec  = irtree.disengagement_tree(binary=True)
data  = irtree.expand_data(Y, spec, categories_start=1, X=X, X_names=["female", "escs"])
nodes = [irtree.ProbitRaschNode(person_covariates=["female", "escs"]),  # covariates on node 1
         irtree.ProbitTwoPLNode()]                                      # 2PL ability

priors = irtree.build_default_priors(spec, nodes)
priors.set("theta.correlation", irtree.LKJPrior(eta=2.0))   # estimate rho

fit = irtree.irtree_fit(data, spec, nodes, priors=priors, n_iter=3000, n_chains=4, seed=42)
```

Data with person and item ids can be loaded from data frames or CSV files with
`irtree.load_data`, which aligns the responses and covariate tables by id.

---

## What's in the box

| Capability | Options |
|---|---|
| **Trees** | disengagement (dichotomous / trichotomous), distractor, ERS, any mapping matrix |
| **Node models** | Rasch, 2PL, 3PL (probit or logit), GRM, GPCM, PCM, mixed-format (binary and polytomous items in one node), custom likelihood; one slope per item or one shared slope per node |
| **Priors** | Normal, LogNormal, TruncNormal, Beta, InvGamma, HalfNormal, LKJ, InverseWishart; defaults per parameter, with the θ variance fixed at 1 where slopes are free (identification) |
| **Covariates** | person (X), item (Q), person×item (W), entering any node; missing values drawn as parameters each iteration |
| **Sharing / correlation** | share θ, b, a, c or covariate effects across nodes; correlated θ (all dimensions or chosen blocks), correlated b or a across nodes |
| **Diagnostics** | R-hat, ESS, MCSE, ACF, Geweke, Heidelberger–Welch; HTML report (`fit.report()`) |
| **Model comparison** | WAIC, LOO-CV |
| **Logit scale** | `fit.to_logit()`: probit estimates rescaled to their logistic equivalents, per parameter role |

---

## Correctness

This package is tested along **three axes** (see `docs/TESTING.md`):

1. **Module correctness** — each module checked by its output (against a
   hand/known answer) and its invariants (properties that must always hold).
   Includes unit checks of every Gibbs draw against its analytic full
   conditional, re-derived independently from the model.
2. **Recovery** — simulate from known parameters, fit, verify credible-interval
   coverage across a covering set of canonical trees and known-hard interactions.
3. **External validity** — reproduce published results (Verbal Aggression against
   an R IRTree implementation; a PISA 2022 disengagement study against its
   published estimates), including a probit→logit scale-bridging check.

```bash
pytest -m fast                          # quick checks (seconds)
pytest -m slow                          # recovery and long-chain checks (minutes)
pytest                                  # everything
jupyter nbconvert --execute notebooks/tests/*.ipynb   # the inspectable form
```

The test **notebooks** (`notebooks/tests/`) are the primary, human-readable
artifact: they print the analytic target beside the observed value so you can
confirm agreement by eye. The **pytest** files run the same kind of checks
headless for CI.

---

## Documentation

- `docs/architecture.md` — the design reference (sampler architecture, module map)
- `docs/full_conditionals.md` — full derivation of every Gibbs/MH full conditional
- `docs/TESTING.md` — the testing framework (three axes, tiers, coverage logic)
- `docs/questionnaire.md` — the deterministic model-specification questionnaire
- `CHANGELOG.md` — what changed in each version

---

## Citation

If you use this package, please cite:

> Mehou, J. M., Leventhal, B. C., Hunsberger, J., & Bao, Y. (2026). An IRTree
> approach to investigating examinee-level predictors of disengagement on the
> PISA. *International Journal of Testing*, 26(2), 81–103.
> https://doi.org/10.1080/15305058.2025.2609068

and the foundational model:

> Leventhal, B. C., & Pastor, D. (2024). An illustration of an IRTree model for
> disengagement. *Educational and Psychological Measurement*, 84(4), 810–834.

---

## License

MIT — see [LICENSE](LICENSE).
