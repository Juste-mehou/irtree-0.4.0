---
title: "irtree Architecture — Consolidated Reference v10 (package 0.4)"
format: pdf
pdf-engine: xelatex
output: word_document
editor_options:
  markdown:
    wrap: 72
---

# irtree Python Package — Architecture & Structure

## Consolidated Reference (v10 — reconciled against the 0.4 code)

A pip-installable Python package for Bayesian IRTree estimation via MCMC.

This document is the single reference for the package architecture: what each
module does, why the sampler is built the way it is, and how the pieces fit.
Where a decision was made and later superseded, only the current decision is
shown; a short rationale is kept where the history explains the design.

> **What changed from v9.** v9 described the code before most of the 0.3.0
> work. This revision was rebuilt against the code module by module:
>
> - **Defaults.** The slope prior is a truncated Normal (exact Gibbs), not
>   LogNormal (MH). θ has variance 1 on dimensions with free slopes
>   (identification); Rasch-type dimensions estimate it (§2.6).
> - **Sampler structure.** No `SamplerBlock`/`mh_step` objects: `run_chain` is
>   one loop over nine stages (§9.10), built on one *latent view* per augmented
>   node. Non-augmented nodes (logit 2PL/3PL, GPCM/PCM, custom) are updated by
>   vectorized per-unit MH, including groups that mix probit and logit members.
> - **GRM thresholds.** The exact draw is the prior truncated to the allowed
>   interval (not Uniform, and the `b.poly` prior is used, not "initialisation
>   only"), preceded by a Cowles step with Z integrated out (§4.3).
> - **Mixing moves.** Location move on the θ/b ridge and interweaving (ASIS)
>   for covariate effects (§3.9).
> - **Correlations.** θ correlation blocks (`Constraints(theta_corr=...)`);
>   pooled statistics over every node of a group; componentwise a-correlation
>   Gibbs (§5).
> - **Covariates.** One coefficient sampler (`gibbs_coef_pooled`) keyed by
>   group; missingness per feature; κ is identified through b's prior (§9.4.1).
> - **Warm-up.** Evidence-based freezing of MH step sizes (§3.8).
> - **New modules and features.** `ingest.load_data`, `progress`,
>   `make_node`, `common_discrimination`, `fit.to_logit()`, vectorized
>   diagnostics and a faster HTML report (§10).
> - **API corrections.** No `Hyperprior` marker (a `PriorSpec` goes directly in
>   the variance slot), no `theta.variance` key, no `trait_cov`, no
>   `IndependentTraits` class.
>
> The companion `docs/full_conditionals.md` (v5) is the authoritative source
> for every full-conditional formula; this document points to it rather than
> repeating the algebra. `docs/TESTING.md` specifies the testing framework.

------------------------------------------------------------------------

# 1. Design Philosophy

The package is organized around the mathematical objects that fully define
an IRTree analysis:

| \# | Object | Description |
|----|----|----|
| 1 | **The Tree** | The mapping matrix M [K × R]: K observed response categories × R internal nodes. Entry M[k, r] is the pseudo-response value at node r on the path to category k; NaN when node r is not on that path. Nodes may share a latent trait (`node_to_latent`). |
| 2 | **The Node Models** | One `NodeModel` per node, giving Pr(Y\*\_pir = y │ θ, item parameters). Nodes can differ (Rasch, 2PL, 3PL, GRM, GPCM/PCM, logit or probit, custom). |
| 3 | **The Priors** | Priors on every parameter: θ (with σ² per latent dimension), b (with an estimated variance per b-group), a, c, correlations, covariate coefficients (β, κ, γ) and missing covariate values. |
| 4 | **Constraints** *(optional)* | Which nodes share b, a, c or γ; which b-groups, a-groups or θ dimensions are correlated. A modelling choice, kept separate from the tree. |
| 5 | **Covariates** *(optional)* | Person matrix X [P, K_x], item matrix Q [I, K_q], person×item tensor W [P, I, K_w]. Arrays live in `PreparedData`; nodes declare which columns they use. |

The package is modular: each layer is independently testable and
replaceable. The sampler is IRTree-aware (it knows the node structure for
caching and pooling) but model-agnostic: it dispatches on node properties
(`augmentable`, `has_discrimination`, `has_guessing`, `is_mixed`), never on
class identity.

**Stack:** Python 3.10+ · NumPy · SciPy. Optional: pandas (tables,
`load_data`), matplotlib (plots, report), patsy (covariate formulas). No C
extensions, no Cython, no Numba. Pure Python + NumPy throughout.

------------------------------------------------------------------------

# 2. The Dominant Efficiency Decision: Sampler Architecture

One decision determines the efficiency of the package more than any other,
and it follows from the structure of the IRTree likelihood and the link
function at each node.

## 2.1 The problem with pure random-walk Metropolis-Hastings

A pure random-walk MH sampler (propose every block from a Gaussian, accept or
reject) is correct but the least efficient valid approach for this model
class:

-   Step sizes must be tuned during warm-up, and a fraction of every
    proposal is wasted on rejections.
-   Even at the optimal acceptance rate the chain random-walks, so
    successive draws are highly autocorrelated, especially for the weakly
    identified person parameters.
-   A joint proposal for P person traits moves each one by a tiny amount as
    P grows (the ℓ²/d scaling).

## 2.2 The solution: Albert-Chib data augmentation

Albert and Chib (1993) showed that for a probit model one can introduce a
latent continuous variable Z such that, given Z, every parameter with a
Normal prior has a Gaussian full conditional. This unlocks exact Gibbs
sampling: draws from the exact conditional posterior, with nothing to accept
or reject.

For a binary probit node r, on every relevant (person, item) cell:

```
Z_pir │ θ_pr, b_ir, a_ir  ~  Normal(a_ir (θ_pr − b_ir + offset_pir), 1)
Y*_pir = 1{ Z_pir > 0 }
```

Given Z, the observed pseudo-response carries no further information, so
every conjugate parameter has a standard full conditional:

| Parameter | Full conditional given Z | Update |
|----|----|----|
| Z_pir (latent) | Normal truncated to ℝ+ or ℝ− by Y\*\_pir | Exact Gibbs, vectorized |
| θ (person traits, any number of dimensions) | (Multivariate) Normal | Exact Gibbs, batched over persons |
| b_ir (difficulty) | Normal | Exact Gibbs, vectorized over items |
| σ²_d (trait variance, InvGamma prior) | InvGamma | Exact Gibbs |
| b-group variance | InvGamma | Exact Gibbs |
| a_ir (discrimination, truncated-Normal prior — the default) | Truncated Normal: Z is linear in a | Exact Gibbs |
| a_ir (LogNormal prior, opt-in) | No closed form | MH on log a |
| U_pir (3PL guessing indicator, Béguin-Glas) | Bernoulli | Exact Gibbs |
| c_ir (3PL guessing, Beta prior) | Beta | Exact Gibbs |
| GRM thresholds | Prior truncated to the allowed interval (after a Cowles step) | Exact Gibbs (+ MH move) |
| β, κ, γ (covariate effects) | Multivariate Normal | Exact Gibbs |
| Missing covariate features | Multivariate Normal | Exact Gibbs |
| Correlations (LKJ) | No closed form | MH (Fisher-z / vine) |

> **Net effect.** A tree of probit Rasch, 2PL, 3PL and GRM nodes with the
> default priors is pure Gibbs apart from the GRM Cowles step and any
> estimated correlation. MH enters only where no closed form exists: logit,
> GPCM and custom nodes; LogNormal slopes; HalfNormal variance priors; LKJ
> correlations; estimated slope variances.

## 2.3 Why probit is the default, and how logit fits in

The augmentation needs the probit link. For IRT the practical difference is
small: Φ(x) is within 0.0095 of the logistic CDF evaluated at 1.702·x for
every x (Haley, 1952). The gain in sampling efficiency is large; the loss in
interpretability is small, and `fit.to_logit()` reports any probit fit on
the logistic-equivalent scale (§11).

Both links are supported. Logit nodes are for users who need parameter
comparability with logit-based software; their parameters are updated by
MH against the node likelihood.

|   | Probit (default) | Logit / GPCM |
|----|----|----|
| θ, b update | Exact Gibbs | Vectorized per-person / per-item MH |
| a update | Exact Gibbs (truncated Normal) or MH (LogNormal) | MH on log a |
| c update | Exact Gibbs (Béguin-Glas) | MH on logit c |
| σ² update | Exact Gibbs (InvGamma) | Exact Gibbs (InvGamma) — θ is still Normal |
| Warm-up | Burn-in only for pure-Gibbs chains | Step sizes tuned, then frozen |
| Rescaling to logit | `fit.to_logit()` (logistic-equivalent, D = 1.702) | already logit |

The match between the links is weakest in the tails. For nodes with rare
outcomes (rapid guessing, say) a logit refit can differ from the rescaled
probit fit by 5–15%; for mid-range outcomes the gap is about 1–3%.

## 2.4 Conditional independence: why person and item updates are cheap

Given item parameters and Z, the persons' traits are conditionally
independent; given traits and Z, so are the items' parameters. The whole
P-vector θ update is therefore P independent draws (or P small MVN draws
across dimensions), done as a few batched NumPy/LAPACK calls with no loop
over persons. The same holds for items.

The same independence is what makes the MH updates of non-augmented nodes
cheap: the logit log-likelihood is a sum of row terms (persons) or column
terms (items), so all units are proposed at once and accepted one by one,
each on its own ratio (`_mh_diag_units`).

## 2.5 Full conditionals

Every formula lives in `docs/full_conditionals.md`. The structure common to
all of them:

-   **One latent view.** On the cells where Z is drawn (all relevant cells,
    or the U = 0 ability cells at a 3PL node),
    `Z = a (core + offset) + ε`, with core = θ − b (binary), θ (GRM) or
    θ − b on the binary items of a mixed node. Every Gibbs statistic is built
    by subtracting from Z what does not involve the block being updated
    (`_latent_view` / `_latent_resid` in `run_chain`).
-   **Discrimination weighting.** Precisions carry a², linear statistics
    carry a. Rasch is the a = 1 special case.
-   **Pooling.** A parameter shared by several nodes is drawn once from the
    sum of the member nodes' statistics, with the prior counted once.
-   **Residualization on the a-scale.** Functions that apply a themselves
    (`sample_Z`, `log_prob`, `recompute_node`, `gibbs_a`) receive the raw
    covariate offset; draws that isolate a core subtract a·offset.

## 2.6 Identification defaults

-   **Location.** θ has prior mean 0 on every dimension; shifting θ and b of
    a dimension together leaves the likelihood unchanged, so this pins the
    location.
-   **Scale.** On a dimension whose nodes all have free slopes,
    `Priors.from_spec` fixes σ² = 1 (the usual 2PL convention), flagged as
    automatic in `priors.summary()`. A dimension with at least one
    slope-free node (Rasch, GRM, PCM) already has its scale fixed by a = 1,
    so σ² is estimated there; fixing it too would be a second, substantive
    restriction. A user `priors.set("theta", ...)` replaces the default.
-   **Correlations.** Correlations are scale-free (Corr(θ₁/c₁, θ₂/c₂) =
    Corr(θ₁, θ₂)), so they stay identified under either scale convention.
    They are estimated by default (LKJ(1) over all D dimensions); LKJ samples
    R alone and keeps σ² = 1, whereas `InverseWishartPrior` samples the full
    Σ and reopens the scale on all-slope dimensions (warned).
-   **Sign.** An unbounded slope prior is rejected outright: it would leave
    the reflection (θ, a, b) → (−θ, −a, −b) unbroken.

------------------------------------------------------------------------

# 3. Secondary Efficiency Improvements

## 3.1 Partial likelihood caching

The log-likelihood factorizes across nodes,
`log L = Σ_r node_loglik_r`, so the `LikelihoodCache` keeps one scalar per
node.

**How the cache is refreshed.** Gibbs draws never read the likelihood, so the
cache is not recomputed after each Gibbs block. MH blocks recompute the nodes
they touch inside their accept/reject step. For augmented nodes the cache is
refreshed once per stored draw, immediately before the log-likelihood trace
is recorded, which makes the trace exact rather than one draw stale.

**Speedup, honestly stated.** Recomputing only the touched node gives an R×
saving only for strictly one-node-at-a-time updates. Shared, correlated and
pooled draws touch every member node, so the saving is smaller there; the
cache is still correct and cheaper than full recomputation.

## 3.2 Inner sums: `np.where` on the relevance mask

Node log-likelihoods sum over relevant cells as
`np.sum(np.where(T_r, log_prob_matrix, 0.0))`. Benchmarks favoured a
sequential mask over scattered fancy indexing (2–7× faster at realistic
sizes). The `np.where` form, not `T_r * log_prob_matrix`, is required:
`log_prob` returns NaN in irrelevant cells and `0 · NaN = NaN`. Gibbs
statistics use the index arrays `row_idx, col_idx` (or the latent masks),
because there enumerating cells is the task. `PreparedData` exports both
forms.

## 3.3 Conjugate InvGamma prior for variance components

The default variance hyperprior is `InvGamma(1, 1)`, which makes every
variance of an unbounded Normal prior (θ, b, β, κ, γ) an exact Gibbs draw.
A `HalfNormalPrior(scale)` on the SD is supported for users who think on
the SD scale; it breaks conjugacy, adds an MH block and emits a
`UserWarning` at `priors.set()` time:

```python
priors.set("theta", NormalPrior(mean=0, variance=HalfNormalPrior(scale=1.0)))
```

## 3.4 Parallel chains

`irtree_fit` runs chains in a `ProcessPoolExecutor` (`max_workers`, default
one per chain up to the CPU count). Each chain gets an independent child seed
from `np.random.SeedSequence.spawn()`.

## 3.5 Pre-allocated chain storage, float32 by default

The chain is pre-allocated as `[n_stored × d]` in `float32`
(`chain_dtype="float64"` for full precision). Summaries and diagnostics
convert to float64.

## 3.6 Warm starting

With `warm_start=True`, b at each node starts at the probit of the empirical
proportion of the harder outcome; θ starts at 0 and variances at 1.

## 3.7 Reproducibility

All randomness uses NumPy's Generator API; each chain owns its Generator and
its final state is stored in `MCMCResult.rng_state`.

## 3.8 Warm-up: evidence-based freezing

`warmup_mode="evidence"` (the default) lets the data decide when warm-up
ends. Each MH block adapts its step size by Robbins-Monro (§9.11) and is
frozen once its acceptance rate, measured over non-overlapping windows of
W = 100 iterations, has stayed within k = 2 standard errors of its target for
M = 3 consecutive windows. Draws are kept once every block has frozen. A
block that never settles is force-frozen at a cap of half the run, with at
least `min_keep` (default 1000) kept draws per chain. A pure-Gibbs chain has
no block to freeze and burns in for `n_warmup` iterations (at least 2·M·W in
evidence mode). A run needs `n_iter ≥ 1600` in this mode;
`warmup_mode="fixed", n_warmup=...` gives a fixed warm-up.

`IRTreeFit.n_warmup` and `MCMCResult.effective_warmup` report the warm-up
the data actually required; `MCMCResult.frozen_at` records each block's
freeze iteration.

## 3.9 Moves that improve mixing

Three moves leave the target invariant and only speed up mixing
(`full_conditionals.md` §5):

-   **Cowles step** for ordered GRM thresholds: a joint random walk on
    (τ₁, log gaps) with Z integrated out, then Z redrawn, so the thresholds
    are not confined to the narrow gap between neighbouring latent
    responses (Cowles, 1996).
-   **Location move**: θ on a dimension, the b of its b-groups and the GRM
    latent responses shifted by one constant drawn from its exact Gaussian
    conditional (Liu & Sabatti, 2000).
-   **Interweaving (ASIS)** for κ with b and β with θ: an exact draw in the
    centred parameterization interleaved with the non-centred one (Yu &
    Meng, 2011). It removes the slow crawl of κ along the ridge b − Qκ = const.

## 3.10 Fast diagnostics and report

The convergence diagnostics and the posterior summary are vectorized over
parameters (rank normalization, split R-hat, bulk/tail ESS with FFT
autocovariances, MCSE, the Geweke sweep), and the HTML report redraws one
reusable figure per parameter. On a 2094-parameter fit (4 chains × 1400
draws) this cut `convergence_report` from 22.8 s to 5.5 s, `posterior_summary`
from 19.7 s to 3.0 s and the full report from about 97 s to about 18 s, with
identical results (§10).

------------------------------------------------------------------------

# 4. Polytomous and 3PL Node Models

The package supports the two standard step-function families (Penfield,
2014): cumulative (GRM, augmentable) and adjacent-category (GPCM/PCM, not
augmentable).

## 4.1 GRM — cumulative probit (augmentable)

Samejima (1969) cumulative step functions Ψ_k(θ) = Φ(θ + offset − τ_k),
with category probabilities as differences of adjacent Ψ. Thresholds must
be ordered per item; `GRMNode.log_prob` returns −∞ for disordered
thresholds, so any MH proposal that violates order is rejected. The
threshold array has shape [I, M−1]. GRM nodes have no discrimination
(a = 1).

## 4.2 GPCM and PCM — adjacent-category logit (not augmentable)

Muraki (1992):

```
Pr(Y* = j) = exp(Σ_{k=1}^{j} a_i(θ + offset − b_ik)) / D,   j = 0..M−1
D = 1 + Σ_{m=1}^{M−1} exp(Σ_{k=1}^{m} a_i(θ + offset − b_ik))
```

Step difficulties need not be ordered. `GPCMNode` estimates one a_i per
item (or one per node with `common_discrimination=True`); `PCMNode` fixes
a = 1 (Masters, 1982). Every parameter is updated by MH (§9.10).

## 4.3 GRM updates

Given Z, the GRM latent model is `Z_pi ~ N(θ_p + offset_pi, 1)` with
category m iff τ_{i,m−1} < Z_pi ≤ τ_{i,m}.

-   **θ:** the statistic is Σ Z (no b term): the thresholds carry the
    location.
-   **Thresholds, exact draw** (`gibbs_thresholds_grm`): the density of Z
    does not involve τ, which enters only through the category indicators,
    so the full conditional is the **prior** N(μ₀, τ²) truncated to the
    interval every indicator allows,
    `L_k = max(τ_{k−1}, max{Z : Y* = k})`,
    `U_k = min(τ_{k+1}, min{Z : Y* = k+1})`. The prior comes from
    `priors.resolve("b.poly", r)` and τ² is the group's current variance.
-   **Cowles step first** (`_mh_ordered_thresholds`): the interval above is
    very narrow with many respondents, so each iteration first moves all
    thresholds of an item jointly with Z integrated out, redraws Z, then
    makes the exact draw.

Shared thresholds are pooled over every member node in both steps.

## 4.4 3PL — Béguin-Glas augmentation

```
Pr(Y*_pir = 1)  =  c_ir + (1 − c_ir) Φ(a_ir(θ_pr − b_ir + offset_pir))
```

A guessing-mode indicator U_pir ~ Bernoulli(c_ir) resolves the mixture
(Béguin & Glas, 2001): U = 1 forces Y\* = 1; U = 0 hands the response to the
2PL latent process. Given U and Z:

-   U_pir: 0 when Y\* = 0; Bernoulli(c / (c + (1 − c)Φ(·))) when Y\* = 1.
-   Z only on the U = 0 cells (the node's latent mask); θ, b and a use those
    cells only.
-   c_ir ~ Beta(α + n_guess, β + n_relevant − n_guess); for a `c_shared`
    group the counts are summed over members. Default prior Beta(5, 17).
-   a: exact truncated-Normal Gibbs under the default prior.

So a probit 3PL node with default priors is pure Gibbs. The cost is one
extra [P × I] indicator matrix per 3PL node. A logit 3PL node has no U; its
c is updated by MH on logit c, and it cannot join a `c_shared` group.

## 4.5 Mixed-format nodes

`GRMNode(pseudo_item_outcomes=...)` and `GPCMNode(pseudo_item_outcomes=...)`
mix binary and polytomous items in one node. All polytomous items must share
one M (no ragged threshold storage). Item parameters are a `NodeBParams`
(`b_binary [I_bin]`, `b_poly [I_poly, M−1]`, `bin_idx`, `poly_idx`,
`M_poly`). In a mixed GRM node the binary items follow the binary probit
path and the polytomous items the GRM path; θ pools both. Priors
`b.binary` and `b.poly` can be set separately and fall back to `b`.

------------------------------------------------------------------------

# 5. Correlated Latent Traits and Item Parameters

With D latent dimensions each person has θ_p = (θ_p1, …, θ_pD); D ≤ R when
nodes share a trait (`node_to_latent`).

## 5.1 Modes

| Mode | Specification | Update |
|----|----|----|
| **Independent** | `FixedCorrelation(rho=0.0)` | σ²_d separately; θ diagonal fast path |
| **Fixed correlation** | `FixedCorrelation(rho=scalar or matrix)` | Free variances estimated; correlation frozen |
| **Estimated (DRD)** (default, `LKJPrior(eta=1)`) | `LKJPrior(eta)`; a bounded Normal on ρ for two dimensions only | MH on the correlation |
| **Estimated (full Σ)** | `InverseWishartPrior(scale, df)` | Exact Gibbs for Σ |

All are set with `priors.set("theta.correlation", ...)`.

**Correlation blocks.** `Constraints(theta_corr=[[1, 2], [3, 4]])` makes the
correlation matrix block-diagonal: estimated within each block, zero across
blocks, dimensions in no block independent. Without it, all D dimensions
form one block. Validation: 1-based dimension indices, at least two per
block, no repeats, disjoint blocks.

## 5.2 Separation strategy and the Inverse-Wishart alternative

**DRD (default when estimated).** Σ = D R D with D the SDs and R the
correlation matrix keeps prior beliefs about variances and correlations
independent: InvGamma Gibbs for the variances, MH for R.

**Inverse-Wishart (opt-in).** One conjugate draw of each block's full Σ,
`InvWishart(Ψ₀ + S, ν₀ + N)`; the block's σ²_d are read from its diagonal.
The prior entangles variances and correlations.
`InverseWishartPrior.vague(G)` gives scale 0.01·I_G and df = G + 1.

|                        | DRD (LKJ / fixed)        | Inverse-Wishart        |
|------------------------|--------------------------|------------------------|
| Correlation update     | MH (Fisher-z or vine)    | Exact Gibbs            |
| Variance update        | Separate InvGamma Gibbs  | Subsumed into Σ        |
| Prior independence     | Variances ⊥ correlations | Entangled              |
| Default                | ✓                        | Opt-in                 |

## 5.3 Gibbs update for θ

Given Σ, θ_p | Z ~ MVN(Σ\*S_p, Σ\*) with Σ\* = (Σ⁻¹ + Λ_p)⁻¹ and Λ_p the
per-dimension a²-weighted counts. `gibbs_theta_correlated` batches the P
solves and takes an elementwise path when Σ is diagonal. When some
dimensions carry a logit/GPCM node, those are drawn by person-wise MH with
the conditional prior from Σ⁻¹, and the rest from the Gaussian conditional
given them (`gibbs_theta_given`).

## 5.4 FixedCorrelation validation

Validated at construction: a scalar ρ must lie in (−1, 1); a matrix must be
square, symmetric (tolerance 1e-10), have a unit diagonal and be positive
definite. Violations raise `ValueError` naming the failed check.

## 5.5 Estimated correlation: prior by dimension

-   **Two dimensions:** `LKJPrior(eta)` (η = 1 uniform on (−1, 1); η > 1
    shrinks toward 0), a bounded Normal
    (`NormalPrior(mean=0, variance=..., lower=-1, upper=1)`), or
    `InverseWishartPrior`.
-   **More than two:** `LKJPrior` or `InverseWishartPrior`; a bounded scalar
    prior raises `NotImplementedError`.

## 5.6 MH for ρ with two dimensions (Fisher-z)

Random walk on z = arctanh(ρ); the Jacobian 1 − ρ² enters the acceptance
ratio (evaluated at both points, it does not cancel; without it ρ is biased
toward ±1). The chain stores ρ. `_mh_rho_fisher_z`.

## 5.7 MH for a correlation matrix with more than two dimensions (vine)

R is parameterized by its G(G−1)/2 partial correlations (Lewandowski,
Kurowicka & Joe, 2009), under which the LKJ density factorizes; each partial
correlation is updated in turn by a Fisher-z random walk. The chain stores
the marginal pairwise correlations (`rho_1_2`, `rho_1_3`, …).
`_mh_correlation_vine`. A joint proposal over all partial correlations is a
possible future optimization; one-at-a-time is correct for all G.

## 5.8 Cross-node item-parameter correlation

Item properties are often consistent across nodes (an item hard at one node
is hard at another). Correlation groups are declared on b-groups and
a-groups:

```python
con = Constraints(b_corr=[[1, 2]])                  # correlate b of nodes 1 and 2
priors.set("b.correlation", InverseWishartPrior())  # or LKJPrior(eta), FixedCorrelation(rho)
con = Constraints(b_corr=[[1, 3], [2, 4]])          # two separate groups
con = Constraints(b_shared=[[1, 2]], b_corr=[[1, 3]])  # a shared group correlated with node 3
con = Constraints(a_corr=[[1, 2]])                  # slopes (nodes with discriminations)
```

Node indices are 1-based; a node stands for its b-group (or a-group). b
correlation needs scalar b (binary nodes). The priors are group-level and
cannot be set per node.

## 5.9 Updates for correlated item parameters

-   **b:** the statistics of each group are summed over **every** member
    node (U = 0 cells at 3PL nodes, a-scaled offset removed), then one MVN
    draw per item (`gibbs_b_correlated_from_stats`, batched). Logit groups:
    the probit groups are drawn given them, and each logit group by
    item-wise MH with the conditional prior from Σ_b.
-   **a:** componentwise exact Gibbs on the truncated MVN (each component's
    conditional prior from Σ_a, plus the member nodes' latent terms,
    truncated to a > 0; no rejection step). Logit groups: per-item MH on
    log a with that prior.
-   **Group variances under LKJ/fixed a.correlation:** estimated with a
    Half-Normal(0, 0.5) prior on the SD when the slopes use the package
    default prior (fixed at 0.5 they would push ρ toward ±1). The ρ update
    uses slopes centred at the prior mean.
-   **Σ_b, Σ_a:** one Inverse-Wishart draw, or the Fisher-z/vine MH.

> **Stated approximation.** The truncated MVN prior on the slopes has a
> normalizing constant Pr(a > 0) that depends on Σ_a. The a-correlation,
> correlated-variance and Inverse-Wishart Σ_a updates ignore it, so they
> target the untruncated hyperparameter posterior. The effect is small when
> the prior mean is several SDs above 0.

`gibbs_b_correlated_rng` (first node of each group only) and
`gibbs_a_correlated` (joint rejection sampler) are older forms kept for
compatibility with the test notebooks; `run_chain` does not use them.

------------------------------------------------------------------------

# 6. User-Facing Coding Conventions

The package is 0-based internally (NumPy convention) and accepts user input
in either convention.

| Choice | Mechanism | Internal storage |
|----|----|----|
| **Node and dimension indices** | User input is 1-based ("node 1", `node_to_latent=(1, 2, 2)`, `Constraints(b_shared=[[1, 3]])`, `priors.set(..., node=1)`); display is 1-based | 0-based |
| **Pseudo-response values in the mapping** | `IRTreeSpec(mapping, values_start=0 or 1)` | 0-based after shift |
| **Observed categories in the data** | `expand_data(Y, spec, categories_start=0 or 1)`; `irtree_fit(..., categories_start=1)` when passing a raw matrix | 0-based after shift |

The `IRTreeSpec` constructor rejects a mapping that mixes conventions across
nodes and names the nodes that disagree. A consistently 1-based mapping
without `values_start=1` fails contiguity validation with a message
suggesting `values_start=1` (likewise `categories_start` in `expand_data`).
Display follows the user's convention.

------------------------------------------------------------------------

# 7. Directory Structure

Standard src-layout for a pip-installable package. The modules mirror the
pipeline (tree → data → node models → priors/constraints → sampler →
diagnostics/posterior → fit).

```
irtree/                              # repository root
├── pyproject.toml
├── README.md
├── CHANGELOG.md
├── LICENSE
├── docs/
│   ├── architecture.md              # this document
│   ├── full_conditionals.md         # every full conditional, derived
│   ├── TESTING.md                   # the testing framework
│   └── questionnaire.md             # model-specification questionnaire
├── src/
│   └── irtree/
│       ├── __init__.py              # public API (63 names, __all__)
│       ├── tree_spec.py             # Layer 1: IRTreeSpec, factory trees
│       ├── data_prep.py             # Layer 2: expand_data, PreparedData, covariate indices
│       ├── ingest.py                # load_data: id-keyed data frames / CSV, formulas
│       ├── node_models.py           # Layer 3: NodeModel classes, make_node
│       ├── gibbs_samplers.py        # Layer 3.5: every exact Gibbs draw
│       ├── likelihood.py            # Layer 4: node log-likelihoods, LikelihoodCache
│       ├── priors.py                # Layer 5: prior classes, Priors container
│       ├── constraints.py           # Layer 5.5: sharing and correlation groups
│       ├── mcmc_sampler.py          # Layer 6: run_chain, chain layout, MH helpers
│       ├── mcmc_adapt.py            # Layer 7: Robbins-Monro, evidence warm-up
│       ├── progress.py              # progress reporting during a run
│       ├── diagnostics.py           # Layer 8: R-hat, ESS, MCSE, Geweke, H-W, ACF, report object
│       ├── posterior.py             # Layer 9: summaries, intervals, WAIC, LOO, logit rescaling
│       └── fit.py                   # irtree_fit, IRTreeFit, plots, HTML report
├── tests/                           # pytest: fast / slow markers (TESTING.md)
│   ├── _harness.py                  # analytic posteriors, simulators, assertions
│   ├── conftest.py
│   └── test_*.py                    # ~25 files, one per module or capability
└── notebooks/
    └── tests/                       # the inspectable test notebooks (00–50)
```

**src-layout rationale.** The package cannot be imported from the repository
root by accident; tests run against the installed package
(`pip install -e .`), so `import irtree` never resolves to a stray folder.

------------------------------------------------------------------------

# 8. pyproject.toml

Build backend `hatchling`; package version 0.4.0 (1.0.0 at the JSS
submission). The core runtime needs only NumPy and SciPy; everything else is
an optional extra.

| Extra | Contents | Used by |
|----|----|----|
| (core) | numpy ≥ 1.24, scipy ≥ 1.10 | the sampler |
| `tables` | pandas | `summary_df`, diagnostics tables, `load_data` |
| `plots` | matplotlib | `plot_*`, `report` |
| `formula` | pandas, patsy | `load_data(formulas=...)` |
| `full` | pandas, matplotlib, patsy | everything a user needs |
| `dev` | the above + pytest, pytest-cov, nbmake, jupyter, ruff | development |

pytest runs `tests/` with the markers `fast` and `slow`. Ruff checks rules
E, F, I, UP, B and NPY (line length not enforced; the code keeps hand-aligned
columns and is not auto-formatted).

------------------------------------------------------------------------

# 9. Module Descriptions

## 9.1 tree_spec.py — Tree Specification

Translates a conceptual tree into the mapping matrix.

**`IRTreeSpec`**, a frozen dataclass: the mapping M [K × R] (float64, NaN
off-path), category and node names, `node_to_latent` (1-based on input,
stored 0-based) and `values_start`.

| Name | Purpose |
|----|----|
| `IRTreeSpec(mapping, ...)` | The tree; validated at construction |
| `disengagement_tree(binary=True/False)` | Disengagement tree: dichotomous (2 nodes) or trichotomous (ability node with M = 3) |
| `distractor_tree()` | Distractor-selection tree |
| `ers_tree()` | Extreme-response-style tree: 3 nodes, 2 latent dimensions (the extremity trait drives both branches) |
| `dim_to_nodes` | Property: for each latent dimension, the nodes that use it. The canonical source for pooling θ |
| `n_latent_dims`, `n_nodes`, `n_categories`, `outcomes_per_node`, `all_nodes_binary` | Structural properties |
| `category_path(category)`, `relevant_categories(node)`, `is_binary_node(node)` | Path queries, in the user's convention |
| `summary()` | Readable mapping matrix |

The factory trees are examples; they do not belong to particular authors.
Common sources: dichotomous disengagement (Leventhal & Pastor, 2024),
trichotomous disengagement (Mehou et al., 2026), ERS (Böckenholt, 2012;
Leventhal, 2019), distractor (Wright et al., 2008).

**`ers_tree()`** — three structural nodes, two traits:

```
Node 1 (θ₁: endorsement direction)
├─ 0 → Node 2 (θ₂: extremity on the negative branch)
└─ 1 → Node 3 (θ₂: extremity on the positive branch, same θ)

node_to_latent = (1, 2, 2)    # 1-based input; stored (0, 1, 1)
dim_to_nodes   = [[0], [1, 2]]
```

Nodes 2 and 3 share θ₂ but keep separate b arrays, so the model can test
whether items are equally extremity-inducing on both branches; equality is
`Constraints(b_shared=[[2, 3]])`.

**Validation at construction:** mapping shape and NaN structure,
cross-node consistency of the value convention, per-node contiguity of
pseudo-response values, `node_to_latent` length and contiguous 1-based labels.

## 9.2 data_prep.py — Data Preprocessing

Given Y [P × I] and an `IRTreeSpec`, produces everything the sampler needs.
Key identity: `Y_star_r[p, i] = M[Y[p, i], r]`.

```python
PreparedData = NamedTuple(
    Y_star, T, row_idx, col_idx,          # per node: pseudo-responses, relevance masks, index arrays
    observed, n_missing,                  # response missingness
    n_persons, n_items, n_nodes,
    X, X_observed, X_feature_observed, n_missing_X,   # person covariates [P, K_x]
    Q, Q_observed, Q_feature_observed, n_missing_Q,   # item covariates [I, K_q]
    W, W_observed, W_feature_observed, n_missing_W,   # cell covariates [P, I, K_w]
    X_names, Q_names, W_names,            # covariate column names (for selection by name)
    person_ids, item_ids,                 # ids carried from load_data
)
```

| Function | Purpose |
|----|----|
| `expand_data(Y, spec, categories_start=0, missing_codes=(99,), X=None, Q=None, W=None, ..., X_names=None, Q_names=None, W_names=None, person_ids=None, item_ids=None)` | Returns `PreparedData`; validates Y and covariate shapes; detects missing responses and missing covariate entries |
| `summarize_prepared(data, spec)` | Relevance rates per node, missingness, covariate dimensions |
| `center_covariates(M, columns=None, names=None, missing_codes=(99,), scale=False)` | Centers (with `scale=True`, standardizes) the columns of X [P, K], Q [I, K] or W [P, I, K] over their observed entries; constant columns (an intercept) and missing entries are left as they are; returns `CenteredCovariates(values, means, scales, transformed)`. For arrays passed straight to `expand_data` |
| `scale_covariates(M, columns=None, names=None, missing_codes=(99,))` | Divides the selected columns by their standard deviation without centering them (for covariates already centered); same rules and return type |
| `resolve_covariate_indices(node_models, data, spec, constraints)` | Turns each node's covariate selection into column indices and groups (one β per dimension, κ per b-group, γ per `gamma_shared` group); validates identical declarations across shared parameters (`_group_and_validate`) |

**Missing responses.** `np.nan` or any code in `missing_codes`. The
relevance mask is `T_r = on_path & observed`, the single gate for every
downstream computation, so no module below `data_prep` needs to know about
missing responses. Valid under MCAR/MAR; under MNAR excluding cells biases
trait estimates (a modelling limitation). A message reports counts and items
above 10% missing.

**Missing covariates are per feature.** `*_feature_observed` records which
features of each unit (person, item or cell) are observed; only the missing
features are drawn each iteration, conditional on the observed ones
(`full_conditionals.md` §2.11.2). The working copies (`X_working` etc.) hold
the current draws, and every cell enters the likelihood (full imputation).

**Centering.** `expand_data` uses covariates as given. The trait mean and the
item-location mean are fixed, so an uncentered covariate shifts every trait
or location it explains and its coefficient trades off with them; centering
removes that and leaves each coefficient's meaning unchanged.
`center_covariates` does it for arrays, and `load_data(center=...)` for data
frames. Scaling is separate: `scale_covariates` and `load_data(scale=...)`
divide by the standard deviation without centering. Neither is applied by
default.

## 9.2a ingest.py — Loading Data Frames

`load_data(responses, *, spec, person_id, item_id, response_col,
responses_layout="auto", person_covariates=None, item_covariates=None,
cell_covariates=None, roles=None, formulas=None, center=False, scale=False,
...)` reads long or wide
responses (data frames or CSV paths), aligns the covariate tables by person
and item id, optionally expands covariate formulas with patsy
(`formulas={"X": "C(gender) + escs"}`), and returns a `PreparedData` that
carries the ids and covariate names. `center` and `scale` are independent,
each `True` (every non-constant column of X, Q and W) or a dict per grain
(`{"X": True, "Q": ["load"]}`): `center` subtracts the mean, `scale` divides by
the standard deviation, and both together standardize. Constant columns are
never changed. Requires pandas; patsy only for formulas.

## 9.3 node_models.py — Node Probability Functions

All node classes derive from `NodeModel`:

```python
class NodeModel(ABC):
    link: str                    # "probit", "logit" or "custom"
    has_discrimination: bool
    has_guessing: bool
    n_outcomes: int

    @property
    def augmentable(self) -> bool:      # derived from the link, stated once
        return self.link == "probit"

    @abstractmethod
    def log_prob(self, y_star, theta_r, b_r, a_r=None, c_r=None, offset=None):
        """[P × I] log Pr(Y*_r = y_star | ...). The raw offset is added to
        θ − b BEFORE the a-scaling: mean a_i (θ_p − b_i + offset_pi)."""
```

| Class | Link | Updated by | Notes |
|----|----|----|----|
| `ProbitRaschNode` | probit | Gibbs | Default binary node |
| `ProbitTwoPLNode` | probit | Gibbs (a: truncated Normal default; MH under LogNormal) | |
| `ProbitThreePLNode` | probit | Gibbs (Béguin-Glas U for c) | §4.4 |
| `GRMNode(n_outcomes=M)` / `GRMNode(pseudo_item_outcomes=...)` | probit | Gibbs + Cowles step | No discrimination; mixed format §4.5 |
| `LogitRaschNode`, `LogitTwoPLNode`, `LogitThreePLNode` | logit | MH | For comparability with logit software |
| `GPCMNode`, `PCMNode` | logit | MH | Adjacent-category; PCM has a = 1 |
| `CustomNode` | custom | MH on the user's `log_prob` | `augmentable` is always False |

**`make_node(kind, **kwargs)`** builds a node from a string (`NODE_TYPES`):
`"rasch"`/`"1pl"`, `"2pl"`, `"3pl"` (probit), `"logit-rasch"`, `"logit-2pl"`,
`"logit-3pl"`, `"grm"`, `"gpcm"`, `"pcm"`, `"custom"`, so a tree can be
specified one readable line per node.

**Covariates are declared per instance**:
`person_covariates`, `item_covariates`, `pi_covariates`, each `None`
(default), `True` (all columns) or a sequence of column names/indices.
Coefficients are keyed by group, not by node: nodes sharing a trait share
one β, nodes sharing b share one κ, and γ is pooled only with
`Constraints(gamma_shared=...)`. A covariate on a shared parameter must be
declared identically on every node that shares it.

**`common_discrimination=True`** gives a node one slope shared by all its
items: exact pooled Gibbs at probit 2PL/3PL nodes (one MH proposal under a
LogNormal prior), one MH proposal at logit/GPCM nodes. Rejected inside an
`a_shared` or `a_corr` group or with an estimated slope variance.

**Custom nodes.** Subclass `CustomNode`, implement `log_prob` (using
`offset` if the node has covariates), set `has_discrimination` /
`n_outcomes` as needed. θ (person by person), b (item by item), a,
coefficients and missing covariates the node uses are updated by MH against
this `log_prob`, together with every node that shares them. A custom node
cannot join the Gibbs path: augmented nodes are handled by probit-specific
draws that never call `log_prob`.

**Numerics.** Probit `log_prob` uses `scipy.special.log_ndtr`. Irrelevant
cells carry NaN pseudo-responses; GRM/GPCM casts use
`np.nan_to_num(..., nan=0)` before `astype(int)`, and `T_r` discards those
cells.

### 9.3.1 Mixed-format nodes: NodeBParams

`pseudo_item_outcomes` is an [I] array of the number of pseudo-response
values each item can produce at the node (2 for binary items, M ≥ 3 for
polytomous ones, one M for all polytomous items).

| Property | Meaning |
|----|----|
| `is_mixed` | binary and polytomous items both present |
| `bin_idx`, `poly_idx` | 0-based item positions of each kind |
| `M_poly` | M of the polytomous items |

`NodeBParams(b_binary, b_poly, bin_idx, poly_idx, M_poly)` is the item
parameter for a mixed node; homogeneous nodes use plain arrays.
`isinstance(params["b"][g], NodeBParams)` is the single dispatch test. Chain
columns are emitted in item order: `b_{g}_{i}` for binary items,
`b_{g}_{i}_tau{k}` for polytomous ones (homogeneous polytomous nodes use
`b_{g}_{i}_thresh_{k}`).

## 9.4 gibbs_samplers.py — Gibbs Samplers

Every exact Gibbs draw, grouped by augmentation family. MH steps live in
`mcmc_sampler.py`. There is no per-node θ sampler (§9.5).

| Group | Function | Purpose |
|----|----|----|
| Binary probit | `sample_Z(..., a=None, offset=None)` | Latent draw on the given cells; mean a(θ − b + offset), offset added before the a-scaling |
| | `gibbs_a(..., offset=None, common=False)` | Slopes, truncated Normal; regressor θ − b + offset; `common=True` pools all cells into one slope |
| | `gibbs_b(...)` | Difficulties, one node |
| GRM | `sample_Z_grm`, `gibbs_thresholds_grm`, `truncnorm_draw` | Latent draw; thresholds = prior truncated to the allowed interval; tail-stable truncated Normal |
| 3PL | `sample_U_3pl`, `gibbs_c_3pl` | Guessing indicator; Beta draw for c |
| Variances | `gibbs_variance_invgamma` (aliases `gibbs_sigma2`, `gibbs_b_variance`) | InvGamma(α + n/2, β + ½Σ(x − μ)²); used for σ², b-group, β/κ/γ variances |
| Cross-node θ | `reduce_suff_stats_to_dims`, `build_sigma`, `gibbs_theta_correlated`, `gibbs_theta_given` | Pool node statistics onto dimensions, Σ = DRD, batched MVN draw (or given the MH dimensions) |
| Cross-node b | `gibbs_b_correlated_from_stats` | Joint MVN draw of b across a b-correlation group from pooled statistics |
| Coefficients | `coef_gaussian_terms`, `gibbs_coef_pooled` | Gaussian statistics (G, h) and the draw of a β/κ/γ group, each node with its own a |
| Missing covariates | `missing_unit_stats`, `missing_terms`, `draw_from_terms`; `gibbs_X_missing`, `gibbs_Q_missing`, `gibbs_W_missing` | Per-unit statistics, per-pattern Gaussian terms, batched draws |
| Covariance | `gibbs_Sigma_invwishart` | Inverse-Wishart draw (θ blocks, Σ_b, Σ_a) |
| Kept for compatibility (not used by `run_chain`) | `compute_suff_stats`, `gibbs_b_correlated_rng`, `gibbs_a_correlated` | Older forms; the test notebooks still check them |

**Vectorization.** Statistics accumulate with `np.add.at` / `np.bincount`
over index arrays or masks; per-unit MVN draws are batched over the leading
axis (two LAPACK calls for P person solves); truncated draws use the
inverse CDF.

### 9.4.1 Covariate identifiability

Exact-Gibbs status is a statement about the draw, not about identification.

-   **β** is identified with θ, whose mean is fixed at 0. Centring X helps
    mixing.
-   **κ** shares the item axis with b: the likelihood depends on (b, κ) only
    through b − Qκ. κ is identified through b's prior (integrating b out
    gives a regression of the item locations on Q). The apparent "collapse"
    of κ in early runs was slow two-block Gibbs mixing along the ridge, not
    non-identifiability; interweaving (§3.9) removes it. `TESTING.md` §5 has
    the probes that separate code, identifiability and mixing.
-   **γ** varies over cells, so it is identified at every node.

## 9.5 θ Sampling — Where the Functions Live

There is one θ path, run every iteration whether or not a correlation is
estimated:

1.  `run_chain` builds each augmented node's per-person statistics from its
    latent view (`missing_unit_stats` on the residual
    Z − a(core − θ + offset), on the cells where Z was drawn);
2.  `reduce_suff_stats_to_dims` adds the statistics of nodes that share a
    dimension;
3.  `build_sigma` assembles Σ = DRD (or reads the Inverse-Wishart Σ);
4.  `gibbs_theta_correlated` draws all dimensions (diagonal fast path when
    independent), or `gibbs_theta_given` draws the Gibbs dimensions given the
    MH dimensions.

Dimensions with a logit/GPCM/custom node are drawn person by person by
`_mh_diag_units`, with the conditional prior from Σ⁻¹ and every node on the
dimension in the target. The correlation updates (`_mh_rho_fisher_z`,
`_mh_correlation_vine`) are private helpers in `mcmc_sampler.py`.

## 9.6 likelihood.py — Node-Level Cached Log-Likelihood

```python
@dataclass
class LikelihoodCache:
    node_logliks: ndarray          # [R]
    def update(self, r, value): ...
    @property
    def total(self) -> float: ...
```

| Function | Purpose |
|----|----|
| `log_node_lik(y_star, T_r, theta_r, b_r, node_model, a_r=None, c_r=None, offset=None)` | One node's log-likelihood, summed with `np.where(T_r, ·, 0)` |
| `init_cache(...)`, `recompute_node(...)` | Build the cache; recompute one node in place |

The cache serves the MH blocks, the stored log-likelihood trace and
WAIC/LOO.

## 9.7 priors.py — Prior Specification

### Three levels

Priors are built outside `irtree_fit()`: the user builds a `Priors` object
from the tree and node models, inspects it, changes what they care about and
passes it in. The package enumerates every parameter.

1.  **Package default** — nothing set.
2.  **Type level** — `priors.set("b", ...)` applies to every node.
3.  **Node level** — `priors.set("b", ..., node=2)` (1-based) overrides
    one node.

Resolution: node level → type level → default. A type-level `set()` also
removes the automatic node-level entries of that parameter (for example the
identification default θ ~ N(0, 1) on dimensions with free slopes).

### Estimated variances: a PriorSpec in the variance slot

Any prior's `variance` may be a number (fixed) or a `PriorSpec` on (0, ∞)
(estimated):

```python
priors.set("b", NormalPrior(mean=0, variance=4.0))                           # fixed
priors.set("b", NormalPrior(mean=0, variance=InvGammaPrior()))               # estimated, InvGamma(1, 1): Gibbs
priors.set("b", NormalPrior(mean=0, variance=InvGammaPrior(alpha=2, beta=1)))
priors.set("theta", NormalPrior(mean=0, variance=HalfNormalPrior(scale=1.0)))  # SD prior: MH, warns
```

Whether the estimate is Gibbs or MH depends on one question: is the
parameter's own prior an unbounded Normal?

-   **Unbounded Normal (θ, b, β, κ, γ):** InvGamma variance → exact Gibbs
    (`gibbs_variance_invgamma`); HalfNormal SD → MH.
-   **Bounded prior (slopes, `TruncNormalPrior`):** the truncation
    normalizer depends on the variance, so no conjugate draw exists.
    `_check_variance_conjugacy` raises `TypeError` unless `force=True`, which
    downgrades to a warning and estimates the variance by MH (target includes
    the normalizer; chain column `a.variance_{label}`).

### Prior classes

```python
NormalPrior(mean, variance, lower=None, upper=None)
TruncNormalPrior(mean, variance, lower=None, upper=None)   # slopes: conjugate with the latent regression
LogNormalPrior(mean, variance, lower=None, upper=None)     # slopes on the log scale: MH
BetaPrior(alpha, beta, lower=None, upper=None)             # guessing
InvGammaPrior(alpha=1.0, beta=1.0)                         # variance hyperprior (conjugate)
HalfNormalPrior(scale)                                     # SD hyperprior (MH)
LKJPrior(eta)                                              # correlation matrices
FixedCorrelation(rho)                                      # scalar or matrix
InverseWishartPrior(scale=None, df=None); InverseWishartPrior.vague(G)
```

### Parameter keys

| Key | Meaning |
|----|----|
| `theta`, `b`, `a`, `c` | Main parameters (per node) |
| `b.binary`, `b.poly` | Binary items and thresholds of a mixed node (fall back to `b`); `b.poly` is also the GRM threshold prior |
| `beta`, `kappa`, `gamma` | Covariate coefficients (per group; set by node) |
| `X_missing`, `Q_missing`, `W_missing` | Prior of missing covariate values (fixed mean and variance) |
| `theta.correlation`, `b.correlation`, `a.correlation` | Correlation priors (group level, no node override) |

### Package defaults

| Parameter | Default | Update |
|----|----|----|
| θ | Normal(0, σ²), σ² ~ InvGamma(1, 1) on dimensions with a slope-free node; σ² = 1 where every node has free slopes | Gibbs |
| b | Normal(0, τ²_g), τ²_g ~ InvGamma(1, 1) per b-group | Gibbs |
| a | TruncNormal(1, 0.5) on (0, ∞) | Gibbs |
| c | Beta(5, 17) | Gibbs (probit), MH (logit) |
| GRM thresholds (`b.poly`) | Normal(0, τ²_g), τ²_g estimated | Gibbs + Cowles step |
| β, κ, γ | Normal(0, τ²_g), τ²_g ~ InvGamma(1, 1) per group | Gibbs (MH with a logit member) |
| X/Q/W missing | Normal(0, 1) per feature | Gibbs (MH where a logit node uses it) |
| θ correlation | `LKJPrior(eta=1)` over all D dimensions (D ≥ 2) | MH (Fisher-z, vine for D > 2) |
| b, a correlations | `FixedCorrelation(rho=0.0)` (independent) | — |

### Validation at `set()` time

-   Support must match the parameter's domain (`TypeError` unless
    `force=True`); an unbounded slope prior is rejected with no override.
-   HalfNormal variance priors emit a `UserWarning` (conjugacy lost).
-   Correlation priors are checked against the group size (bounded scalar
    priors, Normal or Beta on (−1, 1), only for two members; Inverse-Wishart
    df ≥ G + 1). Bounded scalar priors work for θ, b- and a-correlation.

-   **Every accepted prior is honoured** (`_check_sampler_support`): location
    priors (θ, b, `b.binary`, `b.poly`, β, κ, γ) are unbounded Normals, θ's
    mean is 0, an estimated variance is InvGamma (HalfNormal also for θ);
    missing-covariate priors have a fixed mean and variance; slopes take a
    bounded Normal/TruncNormal (Gibbs) or an unbounded LogNormal (MH);
    guessing takes a Beta on (0, 1). Anything else raises at `set()` time
    with the supported alternatives, rather than running a different
    model. A non-zero b mean is used by every b update (node-level, pooled,
    mixed binary items, b-correlation groups, the b-group variance).

### Inspection

`priors.summary()` prints the main parameters per node, the estimated
variances, the covariate priors, the missing-covariate priors and the
correlation structure, with the level each came from ([1] default, [2] type,
[3] node) and a note on the automatic identification default.
`priors.free_params(constraints=None)` lists the estimated
hyperparameters by their chain column names (`sigma2_1`, `b.variance_1&3`,
`a.variance_2`, `rho_1_2`, `b.rho_1_2`, `beta.variance_1`), computed by the
same rules as the chain layout (`hyperparameter_column_names`); pass the
fit's `constraints` to get the shared-group labels. `priors.summary(constraints)`
prints the same names under TABLE 2.

## 9.8 progress.py — Progress Reporting

On by default (`progress=True`): one self-updating stderr line per chain
with iteration, elapsed time, ETA, the data log-likelihood at the current
state and the mean MH acceptance. It prints about ten times per run
(`print_every` overrides), uses a carriage return on a terminal and plain
lines in notebooks and logs, accepts a user callback, and never touches the
sampling. `ProgressReporter`, `mean_acceptance`.

## 9.9 constraints.py — Sharing and Correlation Groups

`Constraints` is separate from `IRTreeSpec` and `Priors` because sharing is
a modelling choice independent of the tree: the same tree can be fitted with
or without longitudinal item invariance.

```python
from irtree import Constraints, share_nodes

c = Constraints(
    b_shared=[share_nodes(1, 3), share_nodes(2, 4)],   # or [[1, 3], [2, 4]]
    a_shared=[[1, 2]],
    c_shared=[[2, 3]],          # probit 3PL nodes only
    gamma_shared=[[1, 2]],      # opt-in pooling of person×item effects
    b_corr=[[1, 2]],            # correlate b-groups
    a_corr=[[1, 2]],            # correlate a-groups
    theta_corr=[[1, 2]],        # θ correlation blocks (latent dimensions)
)
```

`share_nodes(*nodes)` is sugar for a list. `irtree_fit` calls
`constraints.validate(spec, node_models)`.

**Sampler-facing interface.** For each parameter type the object derives
0-based groups and lookups: `b_groups_for(R)` (e.g. `[[0, 2], [1, 3]]`),
`b_node_to_group_for(R)`, `b_representative_nodes(R)`, the same for a and c,
`b_corr_groups_for(R)` / `a_corr_groups_for(R)` (as group indices),
`theta_corr_groups_for(D)`. Without declarations every node is its own
group.

**Chain column naming.** A shared group's columns carry the group label, the
member nodes joined with `&`: `b_shared=[[1, 3]]` gives `b_1&3_1 … b_1&3_I`
and `b.variance_1&3`; θ on a shared dimension gives `theta_2&3_p` and
`sigma2_2&3`. The chain is narrower when sharing is active.

**Validation** (`validate`):

| Condition | Error |
|----|----|
| Indices in range; no node twice in one parameter's groups; groups of ≥ 2 nodes | `ValueError` |
| b-group members have the same `n_outcomes` and identical `pseudo_item_outcomes` | `ValueError` |
| a-group members have discriminations; c-group members have guessing (probit 3PL; logit 3PL in a c group is `NotImplementedError`) | `ValueError` |
| b_corr members have scalar b; no two members from one b-sharing group | `ValueError` |
| a_corr members have discriminations; no two members from one a-sharing group | `ValueError` |
| theta_corr blocks: 1-based dimensions, ≥ 2 per block, no repeats, disjoint | `ValueError` |

Covariate declarations on shared parameters are validated in `data_prep`
(§9.2). `IRTreeSpec`, `node_models`, `likelihood` and `gibbs_samplers` are
unaware of `Constraints`; it is consumed by the chain layout, the
initialization, `run_chain`, and `waic`/`loo_cv` (which take the same
`constraints`).

## 9.10 mcmc_sampler.py — Hybrid Gibbs/MH Sampler

`run_chain(data, node_models, priors, n_iter, n_warmup=0,
warmup_mode="evidence", min_keep=None, seed=None, warm_start=True,
chain_dtype="float32", thin_by=1, spec=None, constraints=None,
progress=True, ...)` runs one chain and returns an `MCMCResult`
(`chain`, `param_names`, `log_lik_trace`, `accept_rates`, `n_warmup`,
`n_iter`, `final_params`, `rng_state`, `warmup_mode`, `effective_warmup`,
`frozen_at`).

### One iteration

Every node is either augmented (probit: Rasch, 2PL, 3PL, GRM, mixed GRM; Z is
drawn) or not (logit 2PL/3PL, GPCM/PCM, custom; MH on the node likelihood).
The stages, in order:

1.  **Node loop.** Binary probit: `sample_Z` → `gibbs_b` (unless b is
    shared, correlated or has a logit member) → slopes by `gibbs_a`
    (bounded prior) or `_mh_discrimination` (LogNormal), unless shared or
    correlated. 3PL: `sample_U_3pl` → `sample_Z` on U = 0 → b, c, a as
    above. GRM: Cowles step → `sample_Z_grm` → `gibbs_thresholds_grm`
    (shared thresholds: Z only here). Mixed GRM: binary items as binary
    probit, polytomous items as GRM. Logit/GPCM: a and c by
    `_mh_logit_discrimination` / `_mh_logit_guessing` (θ and b in stages
    3–4).
2.  **Shared a and c**: one pooled draw per group (MH with every member's
    likelihood when a member is logit); uncorrelated a-group variances (MH).
3.  **Shared b**: pooled Gibbs (all probit); item-wise MH (`_mh_diag_units`)
    with a logit member; pooled Cowles step for threshold groups, then the
    GRM members' Z redrawn.
4.  **θ**: Gibbs (§9.5), or person-wise MH on dimensions with a
    logit/GPCM/custom node.
5.  **Location move** (§3.9).
6.  **σ²** (`gibbs_sigma2`, `_mh_sigma2_halfnormal`, or fixed); θ
    correlation per block (Inverse-Wishart Gibbs, or LKJ via
    `_mh_rho_fisher_z` / `_mh_correlation_vine`); b-group variances.
7.  **b-correlation groups**, then Σ_b.
8.  **a-correlation groups**, their variances, then Σ_a.
9.  **Covariates**: β, κ, γ per group (`gibbs_coef_pooled`, or
    `_mh_coefficient` with a logit member); interweaving for κ/b and β/θ;
    missing X, Q, W (Gibbs, or `_mh_missing_units` for units a logit node
    uses).

Then, at storage iterations: refresh the likelihood cache, record the
log-likelihood, write the chain row.

**The latent view.** Every Gibbs statistic is built from
`_latent_view(r)` → (mask, a, core) and `_latent_resid(r, include)` →
Z − a(core + included covariate terms), so masks (3PL guess cells, mixed
items), discriminations, offsets and node types are handled in one place.

### MH helpers (module-level, private)

| Helper | Block |
|----|----|
| `_mh_discrimination` | Probit slopes under a LogNormal prior (log scale; `common` for one slope per node) |
| `_mh_logit_discrimination`, `_mh_logit_guessing` | Logit/GPCM slopes (log scale), logit 3PL guessing (logit scale) |
| `_mh_diag_units` | Vectorized per-person / per-item random walk: logit θ and b, GPCM steps, binary b with logit members |
| `_mh_ordered_thresholds` | Cowles step (scales from `_threshold_proposal_scale`) |
| `_mh_coefficient` | Preconditioned random walk for a coefficient group with a logit member |
| `_mh_missing_units` | Per-unit random walk for missing covariates used by logit nodes |
| `_mh_sigma2_halfnormal` | σ under a HalfNormal prior (log scale) |
| `_mh_rho_fisher_z`, `_mh_correlation_vine` | Correlations (θ blocks, b- and a-correlation groups) |

Each target that moves on a transformed scale carries its Jacobian (log a,
logit c, log σ, Fisher-z), and every helper's docstring states it.

### The params dictionary

| Key | Indexed by | Contents |
|----|----|----|
| `theta` | latent dimension d | [P] |
| `sigma2`, `log_sigma` | d | trait variance (and its log under HalfNormal) |
| `corr_matrix` | — | θ correlation matrix, or the full Σ under Inverse-Wishart |
| `b` | b-group g | [I], [I, M−1] or `NodeBParams` |
| `b.variance` | g | estimated variance per b-group |
| `a`, `a.variance`, `a.variance.estimated` | a-group | slopes; group variance and whether it is estimated |
| `c` | c-group | guessing |
| `b.corr_matrix`, `a.corr_matrix` | correlation group | Σ (Inverse-Wishart) or R (LKJ/fixed) |
| `beta`, `kappa`, `gamma` | coefficient group | [K] vectors; variances in `beta.variance` etc. |
| `X_working`, `Q_working`, `W_working` | — | covariates with the current draws of the missing entries |

Lookups (`node_to_dim`, `b_node_to_g`, `a_node_to_g`, `c_node_to_g`) are
built once before the loop.

### Chain layout

Built by `_build_chain_layout`, in this order:

```
theta_{dim}_{p}          per latent dimension (dim label uses & for shared dims)
b_{group}_{i}            binary b; b_{group}_{i}_thresh_{k} (GRM/GPCM); b_{group}_{i}_tau{k} (mixed, polytomous items)
a_{group}_{i}            groups with free slopes
c_{group}_{i}            3PL groups
sigma2_{dim}
b.variance_{group}       b-groups with an estimated variance
a.variance_{group}       a-groups with an estimated variance
rho_{d1}_{d2}            within each θ correlation block (LKJ / bounded prior); or
Sigma_theta_{d1}_{d2}    each block's lower triangle (Inverse-Wishart)
Sigma_b_{tag}_{i}_{j}    or b.rho_{label1}_{label2}   per b-correlation group
Sigma_a_{tag}_{i}_{j}    or a.rho_{label1}_{label2}   per a-correlation group
beta_{group}_{cov}, kappa_{group}_{cov}, gamma_{group}_{cov}   (cov = column name or 1-based index)
beta.variance_{group}, kappa.variance_{group}, gamma.variance_{group}   estimated group variances
X_missing_{p}_{k}, Q_missing_{i}_{k}, W_missing_{p}_{i}_{k}    missing features only
```

All labels are 1-based. `fit.to_logit()` rebuilds this layout from the
fit's specification to match columns to roles; `waic`/`loo_cv` look columns
up by the same group labels, so they need the fit's `spec` and `constraints`.

## 9.11 mcmc_adapt.py — Step-Size Adaptation

Robbins-Monro on the log step size of every MH block:

```
log ℓ_{t+1} = log ℓ_t + γ_t (accept_t − target),   γ_t = C (t + t₀)^(−α)
```

with C = 1, t₀ = 10, α = 0.6 (both Robbins-Monro conditions hold for
α ∈ (0.5, 1]), and log ℓ clamped to [log 1e-10, log 100]. The target depends
on how many parameters the block proposes jointly (`target_accept`, Roberts
& Rosenthal, 2001): 0.441 for one, 0.352, 0.316, 0.285, 0.275 for two to
five, and 0.234 beyond. Vectorized per-unit blocks adapt one shared step on
the mean acceptance of their units.

Adaptation stops at the end of warm-up (evidence rule of §3.8, or
`n_warmup` in fixed mode), so the kept draws come from a fixed-kernel chain
and the ergodic theorem applies. Functions: `make_adapt_state`,
`adapt_step`, `make_schedule` (validates the conditions), `target_accept`,
`evidence_band_delta`; constants `EVIDENCE_WINDOW_W = 100`,
`EVIDENCE_BAND_K = 2`, `EVIDENCE_NEED_M = 3`,
`EVIDENCE_WARMUP_FRACTION = 0.5`, `EVIDENCE_MIN_KEEP = 1000`.

A chain of binary probit nodes with conjugate priors and no estimated
correlation has no MH block, so this module is not called; that chain only
burns in (GRM nodes add a Cowles MH step).

------------------------------------------------------------------------

# 10. Diagnostics Architecture

The diagnostics layer is designed for two users.

| Profile | Description | Implication |
|----|----|----|
| **A — MCMC-literate** | Knows R-hat and ESS; wants the numbers | Calls `fit.diagnostics()` and the raw functions |
| **B — IRT practitioner** | Deep model knowledge, little MCMC (SAS PROC MCMC, Mplus); thinks of convergence as yes/no | Reads the console verdict; looks further only when something is flagged |

**Design principle:** print a verdict, not a table. Name failures
explicitly. Let users pull detail on demand. Never call `plt.show()`.

## 10.1 Three layers of output

**Layer 1 — console summary**, printed when `irtree_fit` returns: one line
per concern, numbers only when something fails, adapted to the number of
chains (R-hat needs at least two):

```
✓ Convergence  (R-hat  ):   All R-hat within thresholds
✕ Stationarity (H-W    ):   16 parameters failed  [theta_1_237, ...]
✓ Stationarity (Geweke ):   All Geweke within thresholds
✕ Mixing       (ESS    ):   196 parameters ESS < 400  [call fit.diagnostics()]
✕ Precision    (MCSE/SD):   6 parameters MCSE/SD > 0.1
```

**Layer 2 — `fit.diagnostics()`** returns a `DiagnosticReport` (cached):
per-parameter R-hat, bulk and tail ESS, MCSE and MCSE/SD, Geweke z and p,
Geweke status and recommended burn-in, Heidelberger-Welch pass and discard,
ACF of the first chain, MH acceptance rates, status (PASS/WARN/FAIL)
and counts, plus `posterior_df` / `diagnostic_df` tables.

**Layer 3 — `fit.report(path, plot_params="default")`**, a self-contained
HTML file:

1.  Model summary
2.  Convergence (status counts, number flagged)
3.  Diagnostic table, with the PASS/WARN/FAIL legend
4.  Flagged parameters: every indicator with a plain-language
    interpretation, and that parameter's trace and ACF
5.  Posterior summary
6.  Trace and ACF plots for the first 50 parameters (`plot_params`: an
    integer or `"all"`)

Section 4 is the one for Profile B: three bad parameters are hard to find in
a 2000-row table and easy to read in their own section.

## 10.2 Status rules

-   **PASS** — stationary with no burn-in needed, adequate mixing.
-   **WARN** — usable but check: under-warmed (a burn-in is recommended),
    the two stationarity tests disagree, or mixing/precision is marginal
    (low ESS, high MCSE, R-hat slightly above 1).
-   **FAIL** — fails both stationarity tests, or R-hat exceeds its hard
    threshold.

## 10.3 Diagnostics by number of chains

| Diagnostic | 1 chain | ≥ 2 chains | Notes |
|----|----|----|----|
| R-hat | undefined (raises) | primary | Rank-normalized split R-hat (Vehtari et al., 2021) |
| Geweke | primary | supplementary | First 10% vs last 50%, spectral variances; a sweep over later starts gives the recommended burn-in |
| Heidelberger-Welch | primary | supplementary | Cramér–von Mises on the normalized cumulative sum, progressive discard up to 50% (no half-width test) |
| ESS bulk / tail | per chain | across chains | Rank-normalized bulk ESS; folded (|z − median|) ESS as the tail ESS |
| MCSE, MCSE/SD | always | always | SD/√ESS |
| ACF | always | always | Lags match the SAS PROC MCMC convention |

With one chain there is no between-chain variance, so R-hat would be 1 by
construction; at least two chains are strongly recommended.

## 10.4 Thresholds (module level, user-overridable)

```python
diagnostics.RHAT_WARN      = 1.01
diagnostics.RHAT_FAIL      = 1.05
diagnostics.ESS_WARN       = 400      # Vehtari et al. (2021)
diagnostics.GEWEKE_ALPHA   = 0.003    # |z| cutoff ≈ 2.97; one alpha drives every Geweke check
diagnostics.HW_ALPHA       = 0.05     # critical values HW_CRIT (0.45), HW_CRIT_01, HW_CRIT_10
diagnostics.MCSE_SD_WARN   = 0.10
diagnostics.ACF_LAG50_WARN = 0.10
```

## 10.5 Implementation

The per-parameter computations are vectorized over parameters in batches of
256: one rank normalization (`scipy.stats.rankdata`, axis 0), split R-hat,
FFT autocovariances (`scipy.fft` with `next_fast_len`), the initial
monotone sequence for the ACF sum, MCSE, and the Geweke sweep (which re-tests
only the parameters still failing). Results equal the earlier loop to
~1e-14. The HTML report builds one matplotlib `Figure` with its own Agg
canvas (pyplot untouched, so the user's backend is never changed), fixes the
layout once, and per parameter only updates the data, the titles and the
bars; traces longer than 800 points are drawn as a per-pixel min/max
envelope, which keeps every excursion visible.

## 10.6 SAS PROC MCMC validation cases

The diagnostics were designed against SAS PROC MCMC output for a 2-node
IRTree (30 items, N = 4246). They must flag: a 3PL a/c pair with ESS ≈ 22,
lag-1 ACF ≈ 0.998 and Geweke z > 6 (FAIL); a block of node-1 difficulties
with ESS ≈ 170 and lag-50 ACF ≈ 0.38 (WARN); ρ with ESS ≈ 148 (WARN); a
difficulty with a large H-W half-width (caught by MCSE/SD).

## 10.7 Public API

| Function / class | Purpose |
|----|----|
| `rhat(chains)` | R-hat; `ValueError` for one chain |
| `ess(chains)` | (bulk, tail) |
| `mcse(chains)` | (MCSE, MCSE/SD) |
| `geweke(chain, first=0.1, last=0.5)` | (z, p) |
| `geweke_sweep(chain, alpha=0.003, ...)`, `geweke_zcrit(alpha)`, `GewekeSweepResult` | Burn-in search; cutoff |
| `heidelberger_welch(chain, alpha=0.05, ...)` | (passed, n_discarded) |
| `acf(chain, lags=(1, 5, 10, 50))` | {lag: value} |
| `convergence_report(results, param_names=None, geweke_alpha=0.003)` | `DiagnosticReport` |
| `console_summary(report)` | Layer 1 |
| `STATUS_LEGEND` | The PASS/WARN/FAIL definitions |

------------------------------------------------------------------------

# 11. Posterior Summarization

| Function | Purpose |
|----|----|
| `posterior_summary(results, param_names=None, hdi_level=0.95)` | Mean, SD, 2.5/50/97.5 percentiles, HDI and n_eff per parameter (vectorized) |
| `summary_to_dataframe(summaries)` | pandas table |
| `credible_interval(results, level=0.95, method="hdi")` | HDI or equal-tailed intervals |
| `waic(results, data, node_models, spec=None, constraints=None)` | WAIC |
| `loo_cv(results, data, node_models, spec=None, constraints=None)` | PSIS-LOO with k̂ diagnostics |
| `compare_waic`, `compare_loo` | Differences with standard errors |
| `logit_scale_factors(param_names, data, spec, node_models, priors, constraints)` | Per-column logistic-equivalent factors (§2.3) |

WAIC and LOO rebuild the per-draw linear predictor from the chain (covariate
offsets included, missing covariates filled with each draw's imputed
values) using the sampler's own covariate index and offset function. Always
pass the `spec` and `constraints` used for the fit; `fit.waic()` and
`fit.loo()` do this automatically.

**Logistic-equivalent rescaling.** `fit.to_logit()` returns the posterior
summary with each parameter multiplied by its factor (and a `factor`
column). The factor 1.702 multiplies the linear predictor
a(θ − b + offset), so it goes on whichever quantity carries the latent
scale: on a dimension whose nodes are all probit without slopes, θ, b,
thresholds and covariate effects × 1.702 and variances × 1.702²; elsewhere
probit slopes × 1.702. Correlations, guessing and logit nodes are unchanged;
a location pooled across dimensions keeps the fitted metric (the notes then
report an implied common slope). The rescaling is linear with a positive
factor, so summaries convert exactly and n_eff is unchanged. Output is
labelled "logistic-equivalent (D = 1.702 rescaling)", never "logit
estimates".

**n_eff in the summary** is the rank-normalized bulk ESS across chains,
the same number as the diagnostics' bulk ESS.

------------------------------------------------------------------------

# 12. fit.py — Main User-Facing Wrapper

```python
import irtree

fit = irtree.irtree_fit(
    data,                       # PreparedData, or a raw [P × I] matrix (then categories_start applies)
    spec,
    node_models,
    priors=None,                # build_default_priors(spec, node_models) if None
    n_iter=2000,
    n_warmup=0,                 # pure-Gibbs burn-in / fixed-mode warm-up
    warmup_mode="evidence",     # or "fixed"
    min_keep=None,              # evidence mode: guaranteed kept draws per chain
    n_chains=4,
    seed=42,
    warm_start=True,
    chain_dtype="float32",
    thin_by=1,
    categories_start=0,         # only for a raw matrix (as in expand_data)
    max_workers=None,           # min(n_chains, cpu_count)
    constraints=None,
    X=None, Q=None, W=None,     # only for a raw matrix; otherwise inside PreparedData
    X_missing_codes=(99,), Q_missing_codes=(99,), W_missing_codes=(99,),
    progress=True,
    print_every=None,
)
```

| `IRTreeFit` member | Returns |
|----|----|
| `results`, `data`, `spec`, `node_models`, `priors`, `constraints`, `n_chains`, `n_iter`, `n_warmup` | The fit's inputs and outputs (`n_warmup` is the effective warm-up in evidence mode) |
| `diagnostics()` | `DiagnosticReport` (cached) |
| `summary(max_params=50)` | prints the posterior table |
| `summary_df()`, `coef()`, `credible_interval(level, method)` | table, posterior means, intervals |
| `to_logit()` | logistic-equivalent summary (§11) |
| `waic()`, `loo()` | model comparison |
| `plot_trace(params, max_params=12)`, `plot_acf(params, lags, max_params=12)`, `plot_posterior(params, max_params=16)` | matplotlib Figures (no `plt.show()`, backend untouched) |
| `report(path="irtree_report.html", plot_params="default")` | writes the HTML report |

------------------------------------------------------------------------

# 13. Data Flow

| Input | → | Output | Module |
|----|----|----|----|
| Tree description | → | `IRTreeSpec` | `tree_spec` |
| Y (+ X, Q, W) or data frames | → | `PreparedData` | `data_prep` / `ingest.load_data` |
| Node models + spec | → | `Priors` (defaults, identification) | `priors` |
| Sharing / correlation choices | → | `Constraints` | `constraints` |
| PreparedData + params | → | Z, U per augmented node | `gibbs_samplers` (`sample_Z`, `sample_Z_grm`, `sample_U_3pl`) |
| Latent views | → | θ, b, a, c, thresholds, β/κ/γ, missing covariates (exact draws) | `gibbs_samplers` |
| Node likelihoods | → | logit/GPCM/custom parameters, LogNormal slopes, correlations, Cowles step (MH) | `mcmc_sampler` `_mh_*` |
| All of the above | → | `MCMCResult` per chain | `mcmc_sampler.run_chain` |
| Chains | → | `DiagnosticReport` | `diagnostics` |
| Chains | → | Summaries, intervals, WAIC, LOO, logit rescaling | `posterior` |
| Everything | → | `IRTreeFit`, console verdict, plots, HTML report | `fit` |

------------------------------------------------------------------------

# 14. Parameter Representation

Parameters live in a named dictionary for the sampler (§9.10) and in a flat
chain row for storage (layout in §9.10).

```python
# R = 3 nodes; nodes 2 and 3 share θ dimension 2 and share b
params = {
    "theta":      [theta_dim1, theta_dim2],          # [P] each; length = n_latent_dims
    "sigma2":     [1.0, 1.0],                        # one per dimension
    "b":          [b_group_node1, b_group_nodes23],  # one per b-group
    "b.variance": [1.0, 1.0],                        # one per b-group
    ...
}
# A mixed-format node's b-group holds a NodeBParams instead of an array.
```

-   **θ-sharing is structural** (`IRTreeSpec.node_to_latent`):
    `params["theta"]` has `n_latent_dims` ≤ R entries; `node_to_dim[r]`
    maps nodes to dimensions.
-   **Item-parameter sharing is a modelling choice** (`Constraints`):
    `params["b"]` has `n_b_groups` ≤ R entries; `b_node_to_g[r]` maps nodes
    to groups. Unshared, every node is its own group.
-   **One variance per b-group**: nodes in a group share the array and its
    variance (`b.variance_1&3`).
-   **Coefficients per group**: `params["beta"][g]` for the g-th dimension
    carrying person covariates, likewise κ per b-group and γ per
    `gamma_shared` group.

------------------------------------------------------------------------

# 15. Testing

The testing framework is specified in `docs/TESTING.md`. In short:

-   **Three axes.** (1) Module correctness: each module checked by its
    output against a known answer and by its invariants, including every
    Gibbs draw against its analytic full conditional. (2) Recovery:
    simulate from known parameters and check credible-interval coverage
    over a covering set of trees and interactions. (3) External validity:
    reproduce published results (Verbal Aggression against an R IRTree
    implementation; the PISA 2022 disengagement study), including the
    probit→logit rescaling.
-   **Two tiers.** `pytest -m fast` (seconds, per commit) and
    `pytest -m slow` (recovery and long chains).
-   **Two forms.** The notebooks in `notebooks/tests/` (00 module outputs,
    10 unit conditionals, 20 invariants, 30 identifiability, 40–42 recovery
    tiers A–C, 50 external cross-check) print the analytic target beside the
    observed value; the pytest files in `tests/` run the same kind of checks
    headless (the notebook-to-pytest conversion is planned).
-   **Why coverage, not closeness.** Nearly every defect found while building
    the package was invisible to bitwise regression and visible only to a
    statistical check on a path that had never been run (a shared parameter
    drawn once per node gave intervals ~50% too wide with unbiased means).

------------------------------------------------------------------------

# 16. Key References

-   Albert, J. H. (1992). Bayesian estimation of normal ogive item response curves using Gibbs sampling. *Journal of Educational Statistics*, 17, 251–269.
-   Albert, J. H., & Chib, S. (1993). Bayesian analysis of binary and polychotomous response data. *Journal of the American Statistical Association*, 88, 669–679.
-   Béguin, A. A., & Glas, C. A. W. (2001). MCMC estimation and some model-fit analysis of multidimensional IRT models. *Psychometrika*, 66(4), 541–561.
-   Böckenholt, U. (2012). Modeling multiple response processes in judgment and choice. *Psychological Methods*, 17(4), 665–678.
-   Cowles, M. K. (1996). Accelerating Monte Carlo Markov chain convergence for cumulative-link generalized linear models. *Statistics and Computing*, 6(2), 101–111.
-   Gelman, A. (2006). Prior distributions for variance parameters in hierarchical models. *Bayesian Analysis*, 1(3), 515–534.
-   Geweke, J. (1992). Evaluating the accuracy of sampling-based approaches to calculating posterior moments. In *Bayesian Statistics 4* (pp. 169–193). Oxford University Press.
-   Haley, D. C. (1952). *Estimation of the dosage mortality relationship when the dose is subject to error* (Technical Report No. 15). Stanford University.
-   Heidelberger, P., & Welch, P. D. (1983). Simulation run length control in the presence of an initial transient. *Operations Research*, 31, 1109–1144.
-   Johnson, V. E., & Albert, J. H. (1999). *Ordinal Data Modeling*. Springer.
-   Klein Entink, R. H., Fox, J.-P., & van der Linden, W. J. (2009). A multivariate multilevel approach to the modeling of accuracy and speed of test takers. *Psychometrika*, 74(1), 21–48.
-   Leventhal, B. C., & Pastor, D. (2024). An illustration of an IRTree model for disengagement. *Educational and Psychological Measurement*, 84(4), 810–834.
-   Lewandowski, D., Kurowicka, D., & Joe, H. (2009). Generating random correlation matrices based on vines and extended onion method. *Journal of Multivariate Analysis*, 100, 1989–2001.
-   Liu, J. S., & Sabatti, C. (2000). Generalised Gibbs sampler and multigrid Monte Carlo for Bayesian computation. *Biometrika*, 87(2), 353–369.
-   Liu, J. S., Wong, W. H., & Kong, A. (1994). Covariance structure of the Gibbs sampler with applications to the comparisons of estimators and augmentation schemes. *Biometrika*, 81(1), 27–40.
-   Masters, G. N. (1982). A Rasch model for partial credit scoring. *Psychometrika*, 47, 149–174.
-   Mehou, J. M., Leventhal, B. C., Hunsberger, J., & Bao, Y. (2026). An IRTree approach to investigating examinee-level predictors of disengagement on the PISA. *International Journal of Testing*, 26(2), 81–103.
-   Muraki, E. (1992). A generalized partial credit model: Application of an EM algorithm. *Applied Psychological Measurement*, 16, 159–176.
-   Patz, R. J., & Junker, B. W. (1999). A straightforward approach to Markov chain Monte Carlo methods for item response models. *Journal of Educational and Behavioral Statistics*, 24, 146–178.
-   Penfield, R. D. (2014). An NCME instructional module on polytomous item response theory models. *Educational Measurement: Issues and Practice*, 33(1), 36–48.
-   Robbins, H., & Monro, S. (1951). A stochastic approximation method. *Annals of Mathematical Statistics*, 22(3), 400–407.
-   Roberts, G. O., Gelman, A., & Gilks, W. R. (1997). Weak convergence and optimal scaling of random walk Metropolis algorithms. *Annals of Applied Probability*, 7, 110–120.
-   Roberts, G. O., & Rosenthal, J. S. (2001). Optimal scaling for various Metropolis-Hastings algorithms. *Statistical Science*, 16(4), 351–367.
-   Samejima, F. (1969). Estimation of latent ability using a response pattern of graded scores. *Psychometrika Monograph Supplement*, No. 17.
-   Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Bürkner, P.-C. (2021). Rank-normalization, folding, and localization: An improved R̂ for assessing convergence of MCMC. *Bayesian Analysis*, 16, 667–718.
-   Wright, S. B., Matlen, B. J., Baym, C. L., Ferrer, E., & Bunge, S. A. (2008). Neural correlates of fluid reasoning in children and adults. *Frontiers in Human Neuroscience*, 1(8), 1–8.
-   Yu, Y., & Meng, X.-L. (2011). To center or not to center: That is not the question. *Journal of Computational and Graphical Statistics*, 20(3), 531–570.

------------------------------------------------------------------------

*Document version: v10 (package 0.4). Authoritative architecture reference
for the irtree package.*

*Earlier versions, in brief: v0.5 mixed-format nodes (`NodeBParams`),
`dim_to_nodes`, the 3-node ERS tree; v0.6–0.7 covariates (offset mechanism,
residualization, missing covariates as parameters); v0.8 cross-node item
correlation, `TruncNormalPrior`, `InverseWishartPrior`; v9 reconciliation of
the a-weighted conditionals, pooling of shared parameters and hierarchical
variances. v10 is described at the top of this document.*
