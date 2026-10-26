# Changelog

All notable changes to `irtree` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project aims to follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- `center_covariates(M, columns=None, names=None, missing_codes=(99,), scale=False)`:
  centers (optionally standardizes) X, Q or W arrays over their observed
  entries before `expand_data`; constant columns such as an intercept and
  missing entries are left unchanged. Returns the values with the means and
  scales used.
- `scale_covariates(M, columns=None, names=None, missing_codes=(99,))`:
  divides by the standard deviation without centering, for covariates that
  are already centered.
- `load_data(center=..., scale=...)`: centering and scaling of the aligned
  covariates, independent of each other (each `True` for every grain, or a
  dict such as `{"X": True, "Q": ["load"]}`). Off by default.

### Changed
- Repository layout: the package files sit at the repository root (they were
  nested in an `irtree/` folder in the first commit, so GitHub did not find the
  CI workflow). Project URLs in `pyproject.toml` and the README point to
  `github.com/Juste-mehou/irtree-0.4.0`; the README shows the CI status badge.
- **Trait correlations are estimated by default.** The default
  `theta.correlation` prior is `LKJPrior(eta=1)` over all D latent dimensions
  (was `FixedCorrelation(rho=0)`). Correlations are scale-free, so they stay
  identified when σ² = 1 fixes the scale of a dimension; LKJ samples the
  correlation matrix alone and keeps that constraint. Independent traits:
  `priors.set("theta.correlation", FixedCorrelation(rho=0.0))`. Consequence:
  a model with D ≥ 2 now has one Metropolis-Hastings block (the correlation)
  by default, so it is no longer pure Gibbs unless traits are set independent.
- **σ² = 1 only where every node on the dimension has free slopes.** A
  slope-free node (Rasch, GRM, PCM) already fixes the scale through a = 1, so a
  dimension that mixes it with a 2PL/3PL/GPCM node now keeps σ² estimated
  (previously fixed at 1 as well, one restriction too many). The
  InverseWishart warning and `to_logit()` follow the same rule: on such a
  dimension θ carries the factor 1.702 and probit slopes are unchanged.

### Fixed
- Declaring `Constraints(theta_corr=...)` without setting a correlation prior
  silently kept the block's correlations at 0 (no chain column, no warning).
  With the new default the block is estimated.

## [0.4.0] - 2026-10-24

### Added
- `common_discrimination=True` at probit 2PL/3PL nodes: one slope shared by all
  the node's items, drawn by exact Gibbs from the latent regression pooled
  over every cell (one MH proposal under a LogNormal prior). Previously the
  setting was honoured only at logit/GPCM nodes and silently ignored at probit
  nodes. A node-wide slope inside an `a_shared` or `a_corr` group, or with an
  estimated slope variance, is rejected with an explicit error.
- `fit.to_logit()`: the posterior summary on the logistic-equivalent scale
  (D = 1.702), with the factor applied to each parameter and caveats in
  `df.attrs["notes"]`. The factor goes on whichever quantity carries the
  latent scale: on dimensions whose nodes are all probit without slopes
  (Rasch, GRM without a), θ, b, thresholds and covariate effects × 1.702 and
  variances × 1.702²; elsewhere probit slopes × 1.702 and the θ metric is
  kept. Correlations, guessing and logit nodes are unchanged.
  (`posterior.logit_scale_factors` gives the per-column factors.)
- Test harness: `analytic_kappa_marginal`, the exact posterior of an
  item-covariate coefficient with b integrated out.

### Fixed
- Each b-group's variance column now holds that group's variance. The chain
  stored `params["b.variance"]` indexed by the group's first node instead of
  the group, so with a shared group listed before a singleton (e.g.
  `b_shared=[[2, 3]]`) the `b.variance_*` columns were swapped or misread.
- Every prior `Priors.set` accepts is now honoured, and priors the sampler
  cannot honour are rejected at `set()` with the supported alternatives
  (found in the docs and questionnaire audits):
  - a non-zero prior mean on `b` was ignored by the probit Gibbs draws
    (node-level, pooled, mixed binary items, b-correlation groups) while the
    MH steps and the location move used it; it is now used everywhere,
    including the b-group variance draw and the Σ_b update;
  - bounded `NormalPrior`/`BetaPrior` priors on a two-member correlation
    crashed (`NameError: lkj_eta`) for θ and were never updated for b- and
    a-correlation groups; they are now proper MH targets;
  - bounds on θ, b, β, κ, γ priors were accepted with a warning promising a
    truncated draw that no Gibbs path made; now rejected;
  - a non-zero θ prior mean (it fixes the location) is rejected;
  - an estimated variance in `X_missing`/`Q_missing`/`W_missing` crashed at
    run time; now rejected at `set()`;
  - a non-Beta prior on `c` was silently replaced by Beta(5, 17); a
    `HalfNormalPrior` on `a` was read as LogNormal(0, 0.5) (probit) or
    Normal(1, 0.5) (logit); a HalfNormal variance on b/β/κ/γ was read as a
    fixed variance of 1 (b) or crashed (β/κ/γ); all now rejected.
- `gibbs_a` with `lower=None` truncated the draw to the upper half of its
  conditional (no current sampler path used an unbounded `a` prior here).

### Changed
- `posterior_summary` (and `fit.summary()`) report n_eff as the
  rank-normalized bulk ESS across chains, the same number as the
  diagnostics, instead of treating the stacked chains as one sequence.
- `Priors.free_params(constraints=None)` returns the chain column names of
  the estimated hyperparameters (`sigma2_1`, `b.variance_1&3`, `rho_1_2`,
  `beta.variance_1`, ...), computed by the chain-layout rules
  (`mcmc_sampler.hyperparameter_column_names`); `priors.summary(constraints)`
  lists them, and its "Shared?" column is renamed "Same prior" (it meant one
  type-level prior, not a shared parameter).
- Estimated covariate-group variances are stored in the chain
  (`beta.variance_{group}`, `kappa.variance_{group}`, `gamma.variance_{group}`)
  and rescaled by `to_logit()` (factor squared).
- `irtree_fit(categories_start=...)` defaults to 0, like `expand_data` and
  `load_data` (it defaulted to 1). Out-of-range codes raise with a hint.
- Docstrings no longer call GRM models pure Gibbs (the Cowles step is MH).
- `docs/TESTING.md` §5: the apparent collapse of item-covariate effects with
  free b was a two-block Gibbs mixing failure, not non-identifiability; κ is
  identified through b's prior. The identifiability probes now separate code,
  identifiability and mixing (start-dependence plus the exact posterior).
- Diagnostics are vectorized over parameters: rank normalization, split R-hat,
  bulk/tail ESS (FFT autocovariance) and MCSE are computed for all parameters
  at once, and the Geweke sweep re-tests only the parameters still failing.
  Results are identical to 0.3.0 (relative difference ~1e-14); a
  `convergence_report` on 4 chains × 1400 draws × 2094 parameters went from
  22.8 s to 5.5 s.
- `posterior_summary` is vectorized (means, SDs, quantiles, HDIs, n_eff):
  identical output, 19.7 s to 3.0 s on the same fit.
- HTML report: one trace/ACF figure is built once and redrawn per parameter
  (no `tight_layout` or `bbox_inches="tight"` re-renders), traces longer than
  800 points are drawn as a per-pixel min/max envelope (every excursion stays
  visible), and tick locators are fixed. A full report went from ~97 s to
  ~18 s.
- `plot_trace`, `plot_posterior`, `plot_acf` and the report no longer force
  the matplotlib backend to Agg, so they no longer change the user's backend
  (inline plots in notebooks keep working after a report).
- `docs/TESTING.md` §9.3: corrected the probit→logit table (it mixed the two
  identification conventions: slopes × 1.702 *and* θ × 1.702 with b
  unchanged), and widened the tolerance for rare-outcome nodes after a
  simulation showed 5–15% gaps there.
- `docs/full_conditionals.md` v5 and `docs/architecture.md` v10, rebuilt
  against the code: corrected GRM threshold conditional (prior truncated to
  the allowed interval), truncated-Normal slope default, the latent view,
  logit/GPCM/custom MH and mixed probit+logit groups, mixing moves, θ
  correlation blocks, componentwise a-correlation Gibbs, current API names.
- Lint cleanup of the whole package and test suite under the project's
  ruff configuration (E, F, I, UP, B, NPY): modern type annotations
  (`list[str]`, `X | None`), sorted imports, unused imports, variables and
  dead loops and helpers removed, exceptions chained (`raise ... from`),
  `zip(..., strict=True)` wherever the lengths must match,
  one statement per line in the tests. `ruff check` and `pyflakes` pass. B023
  is exempted in `mcmc_sampler.py` with the reason stated in the file (its
  iteration-loop closures are always called within the same iteration).

## [0.3.0] - 2026-10-23

The interface is still pre-1.0: names and defaults may change in minor
releases until 1.0.0 (planned with the JSS submission).

### Fixed
- Test harness: the GRM simulator (`_harness._draw_grm`) drew categories as a
  sum of independent Bernoulli trials with logistic probabilities. It now uses
  the node's link (probit by default, matching `GRMNode`) and one uniform per
  cell, so the simulated category probabilities are the GRM's.
- Test harness: `simulate_irtree(node_types=[..., 'gpcm'])` fell through to the
  binary branch and failed on the [I, M-1] step parameters. It now draws from
  the GPCM (`_harness._draw_gpcm`).

### Added
- `fast` / `slow` test markers: `pytest -m fast` runs the quick checks in
  seconds; unmarked tests are treated as fast.
- Full test suite along three axes (module correctness, recovery, external
  validity); see `docs/TESTING.md`. Delivered as inspectable notebooks
  (`notebooks/tests/`) mirrored by headless `pytest` files (`tests/`).
- Hierarchical variance estimation: an `InvGammaPrior` in a coefficient prior's
  `variance` slot is drawn by Gibbs; on a bounded prior (discrimination `a`) it
  is rejected unless `force=True`, then estimated by an adapted MH block.
- **MH updates at logit, GPCM and custom nodes.** Every parameter of a
  non-augmented node now has a proper update against its full conditional:
  - θ: person-wise MH per latent dimension. The target combines the
    conditional prior given the person's other dimensions, the Gaussian latent
    term of every probit node on the dimension, and each logit node's row
    log-likelihood. Probit-only dimensions are drawn by Gibbs given the MH
    dimensions (`gibbs_theta_given`).
  - b: item-wise MH per b-group, with the current hierarchical variance.
    Covers scalar difficulties, thresholds and mixed binary/polytomous parts.
  - Covariate coefficients (β, κ, γ): preconditioned random-walk MH for any
    group with a logit member.
  - Missing covariates (X, Q, W) used by a logit node: vectorized MH
    imputation.
  - Discrimination `a` and guessing `c` at logit 2PL/3PL nodes: item-wise MH
    (log scale for `a`, logit scale for `c`).
  - Proposals are preconditioned by the Gaussian precision plus the logit
    information bound (¼ a² per cell), with one adapted step per block.
- **GPCM discriminations** are estimated (one `a_i` per item, or one shared
  slope with `common_discrimination=True`). `PCMNode` (`"pcm"` in
  `make_node`), or `GPCMNode(discrimination=False)`, fixes `a_i = 1`.
- **Cowles (1996) threshold step** for GRM thresholds. Thresholds are proposed
  with Z integrated out (ordered random walk on the first threshold and the
  log gaps, with its Jacobian), then Z is redrawn. It is followed by the exact
  truncated-prior Gibbs step given Z.
- **Location move** (Liu & Sabatti, 2000): θ_d, the b-groups on dimension d
  and the GRM latent responses shift by a common constant drawn from its exact
  conditional.
- **ASIS interweaving** (Yu & Meng, 2011) for κ against b and β against θ.
- **θ correlation blocks:** `Constraints(theta_corr=...)` is now used. The θ
  correlation is block-diagonal over the declared blocks (LKJ: Fisher-z or
  vine per block; inverse-Wishart: one draw per block). Dimensions outside
  every block are independent.
- **Shared parameters across node types:**
  - Thresholds can be shared by GPCM and GRM nodes, and by several GRM nodes
    (one pooled update from all members).
  - b can be shared across mixed-format nodes.
  - b- and a-correlation groups can contain logit or GPCM nodes.
- **Estimated a-group variances** under an LKJ or fixed `a.correlation`, with
  the default `a` prior: Half-Normal(0, 0.5) on the sd. The ρ updates use
  slopes centred at the prior mean.
- Exact log-space truncated-Normal sampler (`truncnorm_draw`), accurate in
  far tails.
- `covariate_column_names()`, one source for coefficient column names (chain
  layout and posterior).
- `DiagnosticReport.geweke_alpha` (the Geweke significance the report was
  built with).
- Top-level exports: `irtree.LogitTwoPLNode`, `irtree.LogitThreePLNode`,
  `irtree.make_node`, `irtree.NODE_TYPES` and `irtree.load_data` (loaded on
  first access, so `import irtree` still needs only numpy and scipy).
- Published references for the example trees (`disengagement_tree`,
  `ers_tree`, `distractor_tree`).

### Fixed
- `log_prob` returned `NaN` in irrelevant cells; the likelihood mask now uses
  `np.where` (not `T * logprob`), which was poisoning the log-likelihood trace and
  freezing pure-MH nodes at 0% acceptance.
- Shared `b` / `a` / `c` across nodes were drawn once per node (last write wins),
  silently discarding most of the data and widening intervals ~50%; now pooled.
- 2PL-with-covariates used inconsistent offset conventions between `sample_Z` and
  `log_prob` (offset added before vs. after `a`-scaling); now consistent.
- `gibbs_a` ignored the covariate offset in its regressor, so `a` absorbed the
  covariate signal; the offset is now included.
- GRM/GPCM `log_prob` crashed on `int(NaN)` (→ `INT_MIN` indexing) in irrelevant
  cells; casts are now NaN-safe.
- Missing `import warnings` crashed any `b_corr` model before sampling.
- Theta ownership: pure-MH (Logit) nodes had their `theta` overwritten each
  iteration by a Gibbs draw built on an all-zero `Z`; MH nodes now own their θ.
- **Partially missing covariate rows:** observed values in such rows are kept.
  The whole row used to be zeroed, and those zeros were treated as data.
- **Covariate regression and imputation** use the latent regression
  Z = a (core + x'β + q'κ + w'γ) + ε on each node's latent mask:
  - imputed Q and W values enter the likelihood and every update;
  - each node uses its own discriminations;
  - the X imputation removes q'κ;
  - 3PL nodes contribute only their U = 0 cells.
  A 2PL model with missing X and an item covariate diverged (κ ~ 1e17).
- The coefficient prior mean is honoured (it was taken as 0).
- **GRM thresholds:** the Gibbs update multiplied the prior by a
  pseudo-likelihood built from Z, which is not in the model (threshold RMSE
  0.137 → 0.056). A duplicated `gibbs_thresholds_grm` definition had silently
  overridden the corrected one.
- **3PL statistics:** θ, pooled b and correlated b used every relevant cell,
  including guess cells with Z = 0 (probit 3PL: b bias −3.64 → +0.05,
  c 0.12 → 0.21).
- **Mixed GRM nodes:**
  - θ statistics dropped the binary difficulties;
  - the first iteration crashed with the default priors.
- **Correlated b** used only each group's first node.
- **Correlated a:**
  - the update used only each group's first node, every relevant cell and no
    offset;
  - it fell back to independent draws on failure;
  - ρ was evaluated on uncentred slopes, pushing it toward ±1.
- **Pooled probit a draw:** the raw covariate offset was subtracted from Z
  (valid only when a = 1).
- **WAIC/LOO** dropped named, pooled and column-selected covariates, and
  skipped mixed nodes. They now rebuild the linear predictor the chain used,
  and a node with missing columns raises an error.
- A fixed θ variance (`NormalPrior(variance=v)` on θ) was ignored (σ² stayed
  at 1).
- **Logit 3PL** nodes went through the probit 3PL Gibbs path (slope bias
  −0.7 → −0.01, c 0.05 → 0.20).
- `sample_U_3pl` ignored the covariate offset.
- `Constraints.validate()` was never called, so the documented sharing rules
  were not enforced. `run_chain` now calls it.
- `CustomNode` was treated as probit-augmentable, so its `log_prob` was never
  used for θ/b.
- **Adaptation blocks:**
  - HalfNormal σ² blocks were keyed by node instead of by dimension;
  - the pooled shared-`a` MH block never adapted;
  - evidence mode could keep zero draws when a block never proposed.
- Pure-Gibbs chains in evidence mode kept every draw from iteration 0. They
  now burn in (`n_warmup` clamped to [2·M·W, n_iter − min_keep·thin_by], 600
  by default).

### Changed
- De-looped the hot paths for large speedups with identical output:
  `gibbs_theta_correlated` (batched over persons, ~37×), the missing-data draws
  (vectorized), `gibbs_thresholds_grm` (single-pass groupby). Overall ~3× on the
  standard model, ~8× on covariate models.
- Consolidated the three identical variance draws into one
  `gibbs_variance_invgamma`, with `gibbs_sigma2` / `gibbs_b_variance` as aliases.
- Removed dead `gibbs_theta` / `gibbs_theta_grm`; θ is drawn only via the
  cross-node path.
- `__init__.py` now imports the full public API directly (no `try/except`
  guards), so a broken submodule fails loudly at import time.
- **Identification default:** on a latent dimension where some node has free
  discriminations, `build_default_priors` fixes the θ variance at 1 (shown in
  `summary()`; a type-level `priors.set("theta", ...)` replaces it).
- Covariate coefficients β, κ, γ are drawn from each node's latent residual
  by `gibbs_coef_pooled`.
- **`MCMCResult.log_posterior_trace` is renamed `log_lik_trace`.** It holds
  the data log-likelihood, without prior terms. The progress callback keyword
  `logpost` is renamed `loglik`, and the progress line prints `loglik`.
- **Geweke cutoff:** the multi-chain Geweke line in `console_summary` uses
  the report's `geweke_alpha` (the same cutoff as `convergence_report`).
- A shared guessing group (`c_shared`) containing a logit 3PL node is
  rejected by `Constraints.validate()` (NotImplementedError). Shared guessing
  is supported between probit 3PL nodes only.
- The inverse-Wishart default scale stays 0.01·I (I was tested and biased
  small variances upward).

### Removed
- `ProbitRaschWithCovariates` (a convenience subclass equivalent to
  `ProbitRaschNode(person_covariates=True, item_covariates=True,
  pi_covariates=True)`). Declare covariates via the constructor arguments on any
  node instead (the mechanism is per-instance, not per-subclass).
- `gibbs_beta`, `gibbs_kappa`, `gibbs_gamma` and their `_pooled` wrappers
  (replaced by `gibbs_coef_pooled`).
- `diagnostics.GEWEKE_FAIL` (use `GEWEKE_ALPHA` / `geweke_alpha`).
- `posterior._pointwise_log_lik` (unused, with the same column-lookup bug).

## [0.2.0]

### Added
- Albert–Chib data-augmented Gibbs sampler for probit IRTree nodes.
- Node models: Rasch, 2PL, 3PL, GRM, GPCM, mixed-format, logit and probit links.
- Trees: dichotomous/trichotomous disengagement, distractor, ERS, custom.
- Priors with sensible per-parameter defaults; parameter sharing and estimated
  latent correlations via `Constraints`.
- Convergence diagnostics (R-hat, ESS, MCSE, ACF, Geweke, Heidelberger–Welch)
  and model comparison (WAIC, LOO-CV).
- High-level `irtree_fit` wrapper: parallel chains, warm start, unified result.
