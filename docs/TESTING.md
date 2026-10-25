# Testing framework for `irtree`

One framework on three axes: (1) module correctness — each module checked by its
output and/or its invariants (some have both, some one, some none); (2) recovery
— a covering set of canonical trees and known-weak combinations; (3) external
validity against published results (Verbal Aggression vs an R IRTree package;
PISA 2022 disengagement vs the authors' published estimates). Delivered as
notebooks first (you run and inspect), mirrored as pytest second (others run headless).

This document defines *why* we test the way we do, and *how* the tests are
organized. It is meant to be read before writing any test, and updated whenever
a new class of bug teaches us a new premise.

---

## 1. The premise that shapes everything

Almost every real defect found while building this package was **invisible to
"does the output match yesterday" regression** and visible **only to a check
against ground truth on a code path that had never been run**. A partial list,
all silent under byte-diffing:

- GRM/GPCM `log_prob` crashed on the first call (`int(NaN) → INT_MIN` indexing).
- A `b_corr` model raised `NameError` before sampling (missing `import`).
- Pure-MH (Logit) nodes accepted **0%** of proposals — a frozen chain, no error.
- The log-posterior trace was **entirely `NaN`** (`0.0 * nan == nan` in the mask).
- Shared `b` / `a` / `c` were drawn once **per node**, the last overwriting the
  rest, silently discarding most of the data (credible intervals ~50% too wide).
- 2PL-with-covariates used **two incompatible offset conventions** between
  `sample_Z` and `log_prob`; nobody had ever run the combination.

The lesson is not "write more regression tests." It is:

> **A test suite for a sampler must (a) exercise every code path and (b) check
> statistical correctness against a known answer. Bitwise regression is a
> fast guard on *refactors*, never the correctness oracle.**

Everything below follows from that sentence.

---

## 2. Three axes of testing

Tests are organized by **what object is under test**, not by how end-to-end they
are. This matches how you actually debug — you ask "which part is wrong?", not
"which layer failed?". There are three axes.

### Axis 1 — Module correctness

Each module is tested by looking at two *kinds* of thing, and a module may have
**both, one, or neither**:

- **Output checks** — does the module produce the artifact it is supposed to,
  against a known or hand-computed answer? Deterministic, no chain. (The unit
  conditionals are exactly this for `gibbs_samplers`: a Gibbs draw *is* that
  module's output, and comparing it to the analytic posterior is an output
  check, no different in kind from checking `data_prep`'s pseudo-responses
  against a hand table.)
- **Invariant checks** — do the properties that must *always* hold, hold?
  No ground truth needed; you assert a rule that can never legitimately break
  (finiteness, cache consistency, reproducibility, "it runs at all").

The "both / one / none" split is the map of this axis:

| Module | Output check | Invariant check |
| --- | --- | --- |
| **tree_spec** | *skipped* — any matrix error surfaces in `data_prep` | — |
| **data_prep** | ✓ hand-computed `Y_star`, `T`, index sets | shape/dtype consistency |
| **node_models** | ✓ derived structure (`is_mixed`, `bin_idx`, `poly_idx`, `augmentable`) | `log_prob` is NaN *only* in irrelevant cells |
| **gibbs_samplers** | ✓ **the unit conditionals** — each draw == its analytic full conditional | — |
| **likelihood** | — (no standalone artifact) | ✓ `total == Σ` node scalars; recompute locality |
| **priors** | ✓ default assignments; conjugacy verdicts | — |
| **constraints** | ✓ grouping / `node_to_group` | ✓ validation rejects illegal models |
| **mcmc_adapt** | ✓ step-size moves the right direction | freezes after warmup |
| **diagnostics** | ✓ R-hat / ESS on constructed input | — |
| **posterior** | ✓ summaries / WAIC / LOO on constructed input; ✓ probit→logit conversion on known params (§9.3) | — |
| **mcmc_sampler** | **none** — pure orchestration, nothing hand-verifiable in isolation | finiteness, reproducibility, MH liveness (all require a running chain) |

Three populations fall out of that table, and naming them is the point of the
reframe:

1. **Both** (data_prep, node_models, constraints, mcmc_adapt) — a hand-verifiable
   artifact *and* an always-true property.
2. **One** — `gibbs_samplers` and priors/diagnostics/posterior have an output
   check but no standalone invariant; `likelihood` has invariants but no output.
3. **None** — `mcmc_sampler.run_chain` is orchestration. It produces nothing you
   can hand-verify by itself, so its correctness is testable *only* downstream:
   through the chain-level invariants (finiteness, reproducibility, MH
   acceptance) and through Axis 2. This is why a bug in the assembled sampler —
   the offset-convention mismatch, the shared-parameter double-draw — is invisible
   until a chain runs.

**The rule that keeps Axis 1 small:** give a module an output or invariant check
*only* when a bug there would be **silent** or **surface obscurely**. If a bug
surfaces loudly downstream, do not double-check it (hence `tree_spec` is skipped —
its errors reappear in `data_prep` against hand values).

Axis 1 is **fast and deterministic** (no chains, or tiny fixed draws) and runs on
every commit.

### Axis 2 — Recovery

Simulate from known parameters, fit, verify coverage. Run on the *fewest*
configurations that touch every capability: a covering set of canonical trees
plus the known-weak combinations (the interactions that were actually bugs). This
is **slow and stochastic** and runs nightly / pre-release. Detailed in §5.

### Axis 3 — External validity

Agreement with **already-published results** on **the same data** — the final
check on convention, scaling, and sign errors the internal math cannot
self-detect (a sampler can be internally consistent and still be estimating a
subtly different model than the literature). Two anchor datasets, chosen because
they validate *different* things and both have published numbers to hit:

1. **Verbal Aggression** (De Boeck & Wilson; Jeon & De Boeck IRTree work) — the
   canonical IRTree benchmark, a nested/linear tree on a small, public, widely
   re-analysed dataset. Validates the *core IRTree machinery* against an
   independent implementation.
2. **PISA 2022 disengagement** (Mehou, Leventhal, Hunsberger & Bao, 2026 — the
   authors' own study) — validates the *full feature set this package was built
   for*: a 2-node disengagement tree with a **2PL/GPCM** ability node, **examinee
   covariates** on the disengagement node, and a **freely-estimated latent
   correlation** between the two traits. Published EAP estimates, SDs, and 95%
   HPD intervals give exact targets.

Detailed in §9. Slow, brittle, a small number of golden comparisons — run before
a release or when a convention might have changed.

## 3. Speed follows the axis (no separate tier concept)

There is no independent "tier" dimension: **speed is a property of the axis.**

- **Axis 1 is fast** — no chains (output checks) or fixed moderate draw counts
  (the sampler's moment checks). Whole axis runs per commit, target under a
  minute. The one knob is `n_draws` for the moment-matching output checks
  (env-tunable, §8); its assertions scale with Monte-Carlo SE so fewer draws stay
  valid, just noisier.
- **Axis 2 is slow** — full chains. It has an internal fast/slow *of its own*:
  the same simulate→fit config runs either as a tiny **liveness smoke** (small
  `P`, few iterations, assert the chain runs, stays finite, moves the right way)
  or as a full **coverage** run (large `P`, thinned, K datasets, credible-interval
  coverage). That is a parameterization of one config — `(P, n_iter, n_datasets,
  mode)` — not a separate taxonomy. The smoke version can join the per-commit run;
  the coverage version is nightly.
- **Axis 3 is slow and manual** — run before a release or when conventions change.

So "fast per-commit" = all of Axis 1 + Axis 2 smoke; "nightly" = Axis 2 coverage
+ Axis 3. The `@pytest.mark.fast` / `@pytest.mark.slow` tags (§8) encode exactly
this, but they are a consequence of the axis, not a design axis themselves.

### 3.1 The intended workflow: notebooks first, pytest second

The suite is built and consumed in two phases, in this order:

1. **Author and run as notebooks (now).** Every check is first a runnable
   notebook cell so *you* execute it, read the intermediate values, and confirm
   with your own eyes that each module and each recovery does what it should. The
   notebook is the artifact; *seeing the numbers agree* is the goal, not a green
   check.
2. **Mirror as pytest (later).** Once the notebooks pass under your inspection,
   the same assertion logic — already in `_harness` (§8) — is exposed as `pytest`
   files so others run the identical checks headless, in CI. A thin wrapper, not
   a rewrite, because the correctness logic lives in `_harness`.

Notebooks *establish* trust (you, reading values); pytest *preserves* it
(everyone else, running a gate). Do not skip phase 1: a passing pytest you never
watched run is a checkmark you have no reason to believe.

## 4. Axis 1 in detail — module output and invariant checks

Each module below produces an artifact *before any sampling*. The table gives,
per module, whether to check it directly and the **cheapest sufficient check** —
governed by the §2 rule (check directly only when a bug is silent or surfaces
obscurely). The right-answer source differs per module and is chosen to be the
least work that still catches the bug:

| Module | Output | Check directly? | Right answer from | Cheapest sufficient check |
| --- | --- | --- | --- | --- |
| **tree_spec** | mapping matrix `M`, `R`, categories | **No** | — | Skipped: any matrix error surfaces in `data_prep`'s `Y_star`/`T` against hand values. Testing `M` alone is redundant. |
| **data_prep** | `Y_star`, `T`, `row_idx`/`col_idx`, dims | **Yes** | **hand-computed** | The foundation every downstream statistic is indexed by. One tiny `Y` (e.g. 3×3) per canonical tree; assert `Y_star` and `T` match the pseudo-response tables filled in by hand. Deterministic, no MCMC. |
| **node_models** | `is_mixed`, `bin_idx`, `poly_idx`, `augmentable` | **Partly** | **hand-computed** | `log_prob` numbers are already exercised by the unit conditionals (§4.1). Check only the *derived structure*: `pseudo_item_outcomes=[2,2,3]` → `is_mixed=True`, `bin_idx=[0,1]`, `poly_idx=[2]`. Pure logic; wrong values crash the sampler obscurely. |
| **priors** | resolved priors, defaults, conjugacy verdicts | **Yes — logic only** | **known decisions** | Not the numbers a prior returns (trivial). The *decisions*: `build_default_priors` assigns the documented default per parameter; `_check_variance_conjugacy` raises on a bounded prior with a `PriorSpec` variance and passes under `force=True`. These branch points were bugs. |
| **constraints** | groups, `node_to_group`, validation errors | **Yes — validation only** | **known errors** | Grouping arithmetic surfaces in recovery. The *validation* must be checked directly because a missing rejection is silent: assert it rejects a partial covariate declaration and a node in both a sharing and a correlation group. |
| **mcmc_adapt** | `AdaptState`, step-size updates | **Yes — small** | **known direction** | Feed a run of accept/reject; assert Robbins-Monro moves the step the right way and freezes after warmup. No chain. |
| **diagnostics** | R-hat, ESS | **Yes** | **constructed input** | Feed constructed chains with a known answer: identical chains → R-hat ≈ 1; deliberately divergent → R-hat large. No model fit. |
| **posterior** | summaries, WAIC, LOO | **Yes** | **constructed input** | Feed a hand-made chain array; assert mean/interval/WAIC match the arithmetic. Separates "computes the summary right" from "did the chain converge". |
| **likelihood** | cache scalars, `total` | **invariant only** | — | Cache-total-equals-sum and recompute-locality are its invariant checks; no standalone output. |
| **gibbs_samplers** | draws | **the unit conditionals (§4.1)** | — | Its output check is the conditionals; no standalone invariant. |

That is ~15–20 small deterministic checks. Their payoff: when a Layer-3 recovery
fails, the setup (data prep, priors, constraints, grouping) is already
known-good, so the failure is an estimation problem by elimination.

**What Axis 1 does *not* do.** It does not check that the modules *compose*
correctly into a working chain — that is Axis 2. A module can produce a correct
artifact in isolation and still be wired wrong downstream. Axis 1 buys clean
localization, not end-to-end correctness.

### 4.1 The sampler's output check — unit conditionals

(The unit conditionals are `gibbs_samplers`' output check from the Axis-1 table.
They are the largest single block of Axis 1, so they get their own detail.)

#### 4.1.1 The pattern

Every conjugate draw has the same test shape, which we have already used
repeatedly by hand during development. Generalized:

1. **Fix all conditioning variables** (θ, b, a, Z, T, priors) at arbitrary but
   valid values. Do *not* run a chain.
2. **Derive the analytic full conditional** independently, from the model, in
   the test — not by copying the implementation. For the Gaussian blocks this
   is a Normal–Normal / MVN Normal–Normal update; for variances an InvGamma; for
   `c` a Beta; for GRM thresholds the prior truncated by the latent responses
   on either side.
3. **Draw N times** (`N ≈ 3–5 × 10⁴`) from the function under test with fresh
   seeds.
4. **Compare** empirical mean and (co)variance to the analytic values, asserting
   the gap is within a small multiple of the Monte-Carlo standard error
   `sqrt(var / N)`. Typically `< 4 × SE`.

The independence of step 2 is the whole point: if the test re-derives the
posterior from the *model* and the function re-derives it from the *code*, a
match means both agree with the math. Copying the code's formula into the test
would only check the code against itself.

#### 4.1.2 The "brute-force design" trick

For the regression-style blocks (`beta`, `kappa`, `gamma`, and their pooled
forms), the cleanest independent posterior is the **full-design Normal–Normal**:
stack one design row per relevant cell, then

```
V⁻¹ = I/τ²  +  Dᵀ D
μ   = V (Dᵀ y)
```

where `D` is the `[N_rel × K]` design (`a_i · X_p` for beta under 2PL,
`a_i · Q_i` for kappa, `a_i · W_pi` for gamma) and `y` the residual on the
`a`-scale (`Z − a_i·(everything except the target)`). This closed form is what
the function *should* reproduce, computed without touching the implementation.

#### 4.1.3 What each function needs (the sampler-output checklist)

Grouped by augmentation family, matching the module's own organization.

**Binary probit augmentation**
- `sample_Z` — draws land in the correct half-line per `Y*`; conditional mean is
  `a_i·(θ−b+offset)`; truncated-normal moments match; **offset enters before the
  `a` scaling** (the convention bug — assert `sample_Z` and `log_prob` agree).
- `gibbs_a` — slope of `Z` on the ability predictor `(θ−b+offset)`; truncated
  at `lower`; **offset must be in the regressor** (its omission was the leak that
  starved every covariate). Rasch/`a=None` path bitwise-stable.
- `gibbs_b` — Normal conditional, precision `1/τ² + Σ_i a_i²` (weighted count,
  not bare `n_p`); **sign**: statistic is `Σ_p(θ_p − Z_pi)`, *not* `Σ(Z−θ)`
  (the v2 doc has this backwards — the test encodes the code's correct sign).

**Cross-node item parameters**
- `gibbs_b_correlated_from_stats` (used by `run_chain` for b-correlation groups)
  and the older `gibbs_a_correlated` / `gibbs_b_correlated_rng` — batched MVN
  draw matches the per-item analytic conditional; positive-orthant truncation for
  `a`; because these draw via Cholesky (not `multivariate_normal`), assert
  **moments**, not bitwise equality to a reference draw. (Correlated `a` groups
  are drawn componentwise inside `run_chain`.)

**Polytomous GRM**
- `sample_Z_grm` — latent draws respect the ordered-threshold intervals.
- `gibbs_thresholds_grm` — exact Gibbs given Z: each threshold's prior truncated
  to (max(previous threshold, largest Z in its lower category), min(next
  threshold, smallest Z in its upper category)); items or categories without
  data get the prior truncated by their neighbours. (`run_chain` precedes it
  with a Cowles step that integrates Z out.)

**3PL Béguin–Glas**
- `sample_U_3pl` — guessing-indicator probabilities.
- `gibbs_c_3pl` — Beta conditional `Beta(α + Σ guess, β + Σ(relevant−guess))`,
  pooled additively across member nodes for `c_shared`.

**Hyperprior variances**
- `gibbs_variance_invgamma` — InvGamma `(α + n/2, β + ½Σ(x−μ)²)`; verify the
  **`mean` argument** actually shifts the statistic (a zero-mean assumption
  would silently bias any non-zero-centred prior).
- `gibbs_sigma2`, `gibbs_b_variance` — thin aliases; assert they equal the
  generic with the right `mean`, and (regression) that they still return what
  the pre-consolidation functions did.

**Cross-node traits**
- `compute_suff_stats` + `reduce_suff_stats_to_dims` — the `(S_p, n_p)` it
  produces reproduce the 2PL statistics from §6.2 (`n_p = Σ_i a_i²`,
  `S_p = Σ_i a_i(Z + a_i b)`); Rasch reduces to counts.
- `build_sigma` — `Σ = D R D` assembly.
- `gibbs_theta_correlated` — MVN conditional; **diagonal fast path bitwise
  identical** to the dense path and to the former scalar draw; batched draw
  matches per-person analytic moments; bounded path stays in bounds.

**Covariate coefficients**
- `gibbs_coef_pooled(kind, nodes, residuals, masks, discriminations, design, τ², rng)`
  — one function for β (`kind='x'`), κ (`'q'`) and γ (`'w'`): §4.1.2
  brute-force match, with and without `a`, each node using its own `a` and its
  own latent mask (U = 0 cells at a 3PL node); pooling sums the statistics
  across the group's member nodes (the fix for the shared-parameter double-draw).
  The residual is on the `a` scale: `Z − a_i·(everything except the target)`.

**Missing covariate data**
- `gibbs_X/Q/W_missing` — each missing value's conditional is the Normal implied
  by its coefficient block; the vectorized/pattern-batched forms match the
  per-cell analytic conditional (moments for `W`, which batches the RNG).

**Covariance matrices**
- `gibbs_Sigma_invwishart` — InvWishart conditional `(ν + n, Ψ + Σ xxᵀ)`.

---

## 5. Identifiability probes — a first-class tool

When a coefficient does not recover there are three possible explanations, and
the probe is the tool that tells them apart:

1. **the draw is wrong** (code) — ruled out by the unit conditionals (§4.1);
2. **the model cannot separate it** from a parameter on the same axis
   (identifiability);
3. **the sampler cannot move** between the values the posterior allows
   (mixing).

Explanations 2 and 3 look alike in a single run: a coefficient stuck far from
its truth. They differ in what the *posterior* says, and that can be computed.

### 5.1 The case that taught this: `κ` vs `b`

Item-covariate effects (`κ`) appeared to collapse toward zero when the item
difficulties `b` were freely estimated. The first reading was confounding:
`b_i` and `Q_iᵀκ` are both per-item shifts, so only their sum is identified.
That holds **without a prior on `b`**. With `b_i ~ N(0, τ²_b)`, the model is a
random-effects regression of the item shifts on `Q`:

```
s_i = Q_iᵀκ − b_i       (the item shift the data pin down)
b_i ~ N(0, τ²_b)    ⇒   s_i | κ ~ N(Q_iᵀκ, τ²_b + 1/n_i)
```

so κ **is identified**, weakly when `τ²_b` is wide (its information comes from
the I item locations, not the P·I cells). Its exact posterior is a Normal
linear regression of `s` on `Q` (`_harness.analytic_kappa_marginal`).

What failed was the **sampler**. κ given b is pinned down by all P·I cells, and
b given κ by P cells each, so a two-block Gibbs sampler takes tiny steps along
the ridge `b − Qκ = const`. Started from three points, it stays at each:

```
# 1500 persons, 30 items, true κ = [0.50, -0.40], τ²_b = 4, 1200 iterations
start at 0       → κ ≈ [0.07, -0.04]    (the old "collapse")
start at truth   → κ ≈ [0.51, -0.39]
start at (2, 2)  → κ ≈ [1.81,  1.71]
exact posterior  → κ = [0.50, -0.33]  (sd 0.34)
```

The earlier prior sweep (tighter `τ²_b` → κ "re-emerges") was the alternation
catching up as the ridge shortened, not κ becoming identified. The package's
sampler adds an interweaving step (ASIS; Yu & Meng 2011) that moves κ and `b`
together along the ridge; with it, `run_chain` recovers κ with `b` free.

### 5.2 The general recipe (apply to every covariate/parameter pair)

For any coefficient `φ` (a covariate effect) and any structural parameter `ψ`
that shares an axis with it:

1. **Unit-test the draw of `φ` in isolation** (Axis 1) so a recovery failure
   cannot be blamed on the draw itself.
2. **Fix `ψ` at truth**, draw only `φ`. If `φ` recovers, the draw is fine.
3. **Free `ψ` and run the chain from several starting values.** If the answers
   depend on the start, the chain is not mixing (explanation 3), whatever the
   values look like.
4. **Compare with the exact posterior** of `φ` (`ψ` integrated out from the
   model, in `_harness`). If the exact posterior is concentrated near the truth,
   `φ` is identified and any failure is the sampler's. If it is flat along a
   direction, that direction is not identified (explanation 2).
5. **Confirm the package's sampler** reaches the exact posterior.

### 5.3 Applying the recipe across the covariate types

| Covariate | Axis it lives on | Structural parameter on the same axis | Identified with that parameter free? | Probe target `ψ` |
| --- | --- | --- | --- | --- |
| **β** (person) | person `p` | θ_p (person trait) | **Yes**, with `X` centred and a proper prior on θ: β is the regression of the person locations on `X`, with θ as residuals, and the information comes from P persons. | θ (expect recovery; guards against θ over-absorbing β) |
| **κ** (item) | item `i` | b_i (item difficulty) | **Yes, weakly**: through b's prior, from I item locations (§5.1). Without any prior on `b` only `b − Qκ` is identified. Slow to sample without interweaving. | b (expect the exact posterior near truth; two-block Gibbs start-dependent; `run_chain` recovers) |
| **γ** (person×item) | cell `(p,i)` | none free per cell — but γ competes with **both** θ_p and b_i | **Weakly**: `W_pi` must vary within persons *and* within items beyond what θ_p, b_i, β and κ explain. | θ and b jointly (fix both, then free each in turn) |

**Practical consequences.** κ's precision depends on the number of items and
on `τ²_b`, not on the number of persons, so item-covariate effects come with
wide intervals unless there are many items or `b` is constrained. With the
truth (including `b`) held fixed across simulated datasets, κ's posterior
centres on the regression of *those* item shifts on `Q`, so coverage of the
true κ can sit below 0.95 in a recovery study without any fault.

### 5.4 Why these are Axis 1 (mostly), not Axis 2

Steps 1–4 are conditional checks with fixed inputs and short alternating
chains — Axis 1, fast. Only step 5 fits the full model. The probe localizes a
finding to a specific parameter pair, and step 3 is what keeps a mixing
failure from being reported as a property of the model.

---

## 6. Axis 2 in detail — recovery as a covering set

A full crossing of the package's capabilities is thousands of configurations and
pointless — the bugs hide in a handful of *interactions*, not in every cell. So
Axis 2 is a **covering set**: the fewest fits such that every capability is
exercised at least once, and every *interaction that was actually a bug* is
exercised at least once.

### 6.1 The capability inventory

- **Tree shapes:** linear (disengagement), nested (distractor), shared-trait
  (ERS), shared-node
- **Node models:** Rasch, 2PL, 3PL, GRM, GPCM, mixed-format
- **Link:** probit (Gibbs), logit (MH)
- **Covariates:** none, person (β), item (κ), cell (γ)
- **Sharing:** none, shared-b, shared-a, shared-c, shared-θ
- **Correlation:** independent, estimated (LKJ), full-covariance (InvWishart)
- **Missing data:** none, X, Q, W
- **Variance:** conjugate Gibbs, HalfNormal MH, hierarchical-`a` MH

### 6.2 The three groups of recovery configs (~15 fits total)

**Group A — canonical baselines.** One fit per tree, simplest node model. Proves
each tree shape recovers at all. (~4 fits.)

| Config | Exercises | Note |
| --- | --- | --- |
| Dichotomous disengagement, Rasch, probit | linear tree, pure Gibbs | **primary integration test** — dissertation Model 1 |
| Distractor, Rasch | nested tree | |
| ERS, Rasch | shared-θ tree | |
| Trichotomous disengagement, GRM node | GRM augmentation | dissertation Model 2 |

**Group B — one capability added at a time**, on the simplest tree that supports
it. Isolates each capability so a failure names the capability. (~6–8 fits.)

| Config | Adds |
| --- | --- |
| Disengagement + 2PL node | discrimination Gibbs |
| Disengagement + 3PL node | guessing (Beta) + `a` MH |
| Disengagement + Logit node | person-wise θ and item-wise b MH, acceptance ≈ 0.44 per scalar block |
| Disengagement + person covariate (β) | covariate Gibbs, offset residualization |
| Two-dim tree + estimated θ-correlation | LKJ ρ, correlated θ path |
| Disengagement + HalfNormal trait variance | variance MH |

**Group C — the risky interactions only.** These are the combinations that were
*actually bugs* this session; they are non-negotiable. (~4–5 fits.)

| Config | The bug it guards | What it asserts |
| --- | --- | --- |
| ERS or shared-node + shared b (and a, c) | shared-parameter double-draw | recovery **and** interval width ≈ nominal (the bug widened them ~50%) |
| Disengagement + 2PL + person covariate | 2PL/covariate offset-convention mismatch | β and `a` both recover (the combination never run before) |
| Disengagement + **item** covariate + free b | apparent κ "collapse" (a two-block Gibbs mixing failure, §5.1) | κ recovered through b's prior; judged against the regression of the item shifts on Q (wide intervals by nature) |
| Trichotomous disengagement (GRM) + covariate | GRM augmentation meets covariates | recovery of the covariate under a polytomous node |
| Disengagement + covariate + missing X/Q/W | the Q/W residualization / caller-contract paths | recovery with missing covariate values imputed |

**The logic of the split:** Group A proves the trees work; Group B proves each
capability works in isolation; Group C proves the interactions that broke work
now. Anything outside this set is a combination whose parts are each covered by
A/B and whose interaction was never a failure — deliberately skipped, not
forgotten. When a new bug is found in an untested interaction, it becomes a new
Group-C row (and a questionnaire compatibility rule — see §8.3).

### 6.3 Recovery configs ARE the questionnaire's reachable leaves

Each recovery config is one path through the Deterministic Questionnaire, and
**Group C is exactly the set of questionnaire paths that trigger a compatibility
rule** (identifiability block, shared-parameter pooling, offset residualization).
So building the recovery matrix simultaneously proves that the questionnaire's
hard cases compile and estimate correctly. Maintain **one** list, not two: a
package capability reachable by no questionnaire path is a coverage hole; a
questionnaire path with no recovery config is an unverified promise. The
questionnaire's completeness and Axis 2's coverage are the same object.

### 6.4 How a recovery config is judged

Not by point-estimate closeness — that hides the shared-parameter "intervals too
wide" failure. By **credible-interval coverage**: over K simulated datasets, the
fraction of 95% intervals containing truth must fall in the binomial acceptance
band around 0.95 (`_harness.coverage_band`). Point closeness is reported for
diagnosis but not asserted. The `fast`-tier version of each config is the same
simulate→fit at tiny `P` with a *direction* assertion (recovers the right sign,
stays finite) instead of coverage — a liveness smoke test.

## 7. Deliverable format — notebooks first, pytest second

The recovery and identifiability layers are delivered as **runnable notebooks**,
not only as `pytest` files, because they double as dissertation-defensible
artifacts: each shows the simulation, the fit, and the recovery/coverage figure
inline. A reviewer can read the argument, not just a green check.

Notebook set, grouped by layer, numbered so they run in dependency order
(module outputs → conditionals → invariants → identifiability → recovery →
cross-check). Filled in one at a time; each is a self-contained argument you run
and read.

```
notebooks/tests/
  # --- Axis 1: fast, deterministic, no chains ---
  00_module_outputs.ipynb        # Axis 1 (output+invariant): data_prep hand-check, node structure,
                                  #          prior/constraint decisions, diagnostic math (§5)
  10_unit_conditionals.ipynb     # Axis 1 (sampler output): each gibbs_* vs analytic conditional (§4.1)
  20_invariants.ipynb            # Axis 1 (invariants): cache, reproducibility, NaN-freedom, a=None,
                                  #          MH liveness, GRM runs (§3 layer 2)
  30_identifiability.ipynb       # Identifiability probes: β/θ, κ/b, γ/(θ,b) + prior sweeps (§5)

  # --- Axis 2: slow, stochastic, full chains — the covering set (§8) ---
  40_recovery_tierA_canonical.ipynb   # one fit per tree (disengagement, distractor, ERS, GRM)
  41_recovery_tierB_capabilities.ipynb# 2PL, 3PL, Logit, person-cov, corr-θ, HalfNormal var
  42_recovery_tierC_interactions.ipynb# shared b/a/c, 2PL+cov, item-cov+free-b (κ via b's prior),
                                       # GRM+cov, missing+cov  — the was-a-bug set
  50_external_crosscheck.ipynb        # Axis 3 (§9): Verbal Aggression vs R (irtrees/psychotree);
                                       #   PISA 2022 vs published Mehou et al. (2026) Table 3;
                                       #   probit->logit scale bridging (D=1.702, per-role) vs both (§9.3)
```

Each notebook:
- states its premise and the exact model(s) it exercises at the top;
- for Axis 2, is **parameterized by tier** via a leading `TIER = "fast" | "slow"`
  cell that sets `(P, n_iter, n_datasets)` — the same notebook runs as a
  seconds-long liveness smoke test or a minutes-long coverage test (§8.4);
- ends with an explicit pass/fail cell (asserts, not just plots) so it can later
  be executed head-less via `nbmake` or `jupyter nbconvert --execute`.

**Phase 1 (now): you run them.** Execute top to bottom, read the intermediate
values, confirm the analytic target and the sampled/observed value agree with
your own eyes. The notebook is the artifact; the printed comparison is the
evidence.

**Phase 2 (later): pytest mirror for others.** Once a notebook passes under your
inspection, its assertions — already in `_harness` (§8) — are exposed as a
`pytest` file so others run the identical check headless. Three of these already
exist (`test_unit_conditionals`, `test_invariants`, `test_identifiability`) and
can be read back into notebooks 10/20/30 for phase-1 inspection; they import the
same `_harness`. The pure-`pytest` mirror is a thin wrapper, never a reimplementation.

---

## 8. Shared harness and pytest mirror

To avoid two copies of the correctness logic:

```
tests/
  conftest.py                # fixtures: seeded rng, tiny dims, env-tunable n_draws
  _harness.py                # THE shared logic, imported by notebooks and pytest alike:
                             #   analytic_normal_normal / analytic_theta_scalar / _mvn /
                             #     analytic_b_scalar / analytic_invgamma / analytic_beta_conjugate /
                             #     truncnorm_moments        (posteriors, re-derived from the model)
                             #   draw_many / assert_mean / assert_var / assert_cov /
                             #     assert_bitwise / assert_finite / coverage_band  (checks)
                             #   simulate_probit_cell_means / simulate_augmented_Z  (generators)
                             #   [to add for Axis 2] simulate_tree / fit_and_coverage /
                             #     identifiability_probe
  test_module_outputs.py     # Axis 1 output, @pytest.mark.fast   (diagnostics + posterior, known chains)  [done]
  test_unit_conditionals.py  # Axis 1 sampler output, @pytest.mark.fast                          [done]
  test_invariants.py         # Axis 1 invariants, @pytest.mark.fast                              [done]
  test_identifiability.py    # Probes, fast + @pytest.mark.slow sweeps                          [done]
  test_fit.py                # Axis 1 orchestration invariants, @pytest.mark.fast                 [done]
  test_recovery.py           # Axis 2, @pytest.mark.slow (parametrized over groups A/B/C)        [to write]
```

Rules the harness enforces so tests stay honest:

- **Analytic posteriors are re-derived in `_harness.py` from the model**, never
  imported from the sampler. A test may not call the function it is testing to
  compute its own expected value.
- **Every stochastic assertion is expressed against Monte-Carlo SE**, with the
  seed and `N` fixed, so "pass" has a defined false-positive rate rather than a
  hand-tuned tolerance.
- **Recovery is judged by coverage**, not point-estimate distance: over K
  datasets, the fraction of 95% credible intervals containing truth must lie in
  a binomial acceptance band around 0.95. Point closeness is reported but not
  asserted (it hides the shared-parameter "intervals too wide" failure mode).
- **Reproducibility is a test**: same seed → identical chain, so any future
  change to RNG-consumption order is caught deliberately, not discovered.

Per §4.1, these `pytest` files are **phase 2** — the headless mirror others run.
Phase 1 is you executing the equivalent notebooks and reading the values. The
`_harness` is written once and both front-ends import it, so the mirror is a
wrapper, never a second implementation of correctness.

---

## 9. Axis 3 in detail — external validity against published results

Axes 1 and 2 prove the sampler is *internally* correct and recovers *simulated*
truth. Neither can catch an error where the whole package is self-consistently
computing a subtly different model than the field's — a flipped sign convention,
a logit-vs-probit scale factor, a difficulty parameterised as `b` vs `-b`, a
correlation with the wrong reference. The only defence is to reproduce a result
someone else already published, on the same data, and land on their numbers.

Two anchors, deliberately chosen to validate different things.

### 9.1 Verbal Aggression — the canonical IRTree benchmark

**What it is.** The De Boeck & Wilson verbal-aggression data (24 items: 3
situations × 2 behaviour modes × ... , polytomous "no/perhaps/yes"), the dataset
every IRTree paper cuts its teeth on (De Boeck & Partchev; Jeon & De Boeck).
Small, public, and re-analysed enough that the expected structure is not in
dispute.

**What it validates.** The *core tree machinery* — mapping matrix, pseudo-response
expansion, node-wise likelihood, the sequential/nested decomposition — against an
independent implementation. This is the "did we build an IRTree at all, correctly"
check, isolated from the disengagement-specific features.

**How it is judged.** Fit the corresponding tree in this package and in an
established R implementation (`irtrees` / the De Boeck-Wilson `lme4`
formulation, or `psychotree`) on the identical data, and compare the node-level
item and person parameters. Because the two implementations use different priors
and estimators (Bayesian MCMC here vs. penalised ML there), the check is
**agreement within a tolerance**, not equality: parameter *rank order* and
*sign* must match exactly, and point estimates must agree within a stated band
(e.g. |Δ| < 0.1 on the logit scale for well-identified items, looser for extreme
ones). A rank correlation near 1 and no sign flips is the pass criterion; exact
values are reported, not asserted.

### 9.2 PISA 2022 disengagement — the full feature set, against the authors' own paper

**What it is.** The model of Mehou, Leventhal, Hunsberger & Bao (2026), fit to
the U.S.A. PISA 2022 science sample (2,201 examinees, 115 items). It exercises,
in one fit, nearly everything this package adds beyond a plain tree:

- a **2-node disengagement tree** (Stage 1 = rapid-guess, Stage 2 = ability);
- a **1PL** disengagement node with a single shared discrimination `a⁽¹⁾`;
- a **2PL / GPCM** ability node (2PL for dichotomous items, GPCM for the
  polytomous ones — the mixed-format path);
- **examinee covariates** entering the Stage-1 (disengagement) logit — gender,
  language, ESCS, grade, growth mindset, education expectation, belonging,
  family support, bullying;
- a **freely-estimated latent correlation** between disengagement θ⁽¹⁾ and
  ability θ⁽²⁾;
- **planned-missing** item data (cluster assignment) and imputed missing
  covariates.

**What it validates.** That the covariate regression, the 2PL/GPCM ability node,
the latent correlation, and the missing-data handling not only run (Axis 2) but
land on **numbers a peer-reviewed paper published** — the strongest possible
check that the feature set is not just self-consistent but *correct against the
world*.

**The published targets** (Mehou et al., 2026, Table 3; U.S.A. sample). These are
the golden values the package must reproduce, within MCMC tolerance:

| Parameter | Published EAP | SD | 95% HPD |
|---|---|---|---|
| a⁽¹⁾ (disengagement discrimination) | 2.957 | 0.107 | [2.756, 3.170] |
| β₀ intercept | 0.812 | 0.186 | [0.465, 1.186] |
| β₁ Female | −0.248 | 0.057 | [−0.361, −0.137] |
| β₂ Language of test | 0.141 | 0.072 | [0.003, 0.287] |
| β₃ ESCS | −0.028 | 0.030 | [−0.088, 0.031] |
| β₄ Growth mindset | 0.000 | 0.043 | [−0.086, 0.086] |
| β₅₁ Grade 10 | −1.274 | 0.081 | [−1.431, −1.114] |
| β₅₂ Grade 9 | 0.534 | 0.099 | [0.344, 0.735] |
| β₆ Education expectation | 0.017 | 0.080 | [−0.145, 0.167] |
| β₇ Sense of belonging | −0.055 | 0.053 | [−0.158, 0.048] |
| β₈ Family support | 0.043 | 0.056 | [−0.070, 0.151] |
| β₉ Bullying | −0.022 | 0.053 | [−0.128, 0.078] |
| ρ(θ⁽¹⁾, θ⁽²⁾) latent correlation | −0.522 | 0.031 | [−0.579, −0.459] |

**How it is judged.** The paper's model was fit in SAS (150,000 retained draws,
thin 15). Reproducing it here uses the same data, the same tree, the same
noninformative priors (Table 1 of the paper), and enough draws to make MCMC error
small. The pass criteria, in decreasing strictness:

1. **Sign and significance verdict match exactly** for every covariate — the
   substantive conclusions (Female −, Grade 10 −, Grade 9 +, Language +, the
   moderate-evidence directions, the null effects) must all replicate. This is
   the check that matters most and the cheapest to state.
2. **Each EAP falls inside the published 95% HPD**, and this package's own 95%
   interval overlaps the published one. Two independent Bayesian fits of the same
   model on the same data should not disagree beyond their combined uncertainty.
3. **The latent correlation ρ reproduces** to within ~0.05 (published −0.522) —
   the single most important structural number, and the one a convention error
   (sign, reference dimension) would most visibly break.
4. **a⁽¹⁾ reproduces** near 2.96 — a scale check on the discrimination.

Point estimates are reported in a side-by-side table (published vs. this
package); (1) and (3) are *asserted*, the rest reported with a pass/fail flag.

### 9.3 Probit→logit scale bridging — a validation claim AND a package feature

This package samples with the **probit** link (Albert-Chib augmentation is what
makes the Gibbs sampler exact). Most published IRTree work — including the PISA
anchor (§9.2), whose Eqs. 2–3 are logistic — reports **logit** parameters. Axis 3
therefore adds a *scale-bridging* step: fit probit here, convert the estimates to
the logit scale, and show they land on the logit reference. This serves two ends,
which must be kept distinct because they are different artifacts.

**(1) The validation claim (this section, Axis 3).** Converting the probit fit to
logit should reproduce (a) a from-scratch logit fit on the same data, and (b) the
published logit estimates. Passing this tells researchers something useful: *a
probit fit can be read as a logit fit after the known rescaling* — the link choice
does not change the substantive story.

**(2) The package feature (belongs in `posterior`/`fit`, with its own Axis-1
output check).** When a user fits with probit, the output *also reports* the
logistic-equivalent parameters, so they never convert by hand. This is output
formatting, not a test — but it needs a deterministic Axis-1 unit check that the
conversion function computes correctly (§4, "posterior" row extends to it).

**The conversion is per-parameter-role, not a global multiply.** This is the part
a naive implementation gets wrong. From the premises Φ(x) ≈ L(1.702·x) and
η = a(θ − b + offset), the logistic-equivalent linear predictor is 1.702·η, and
the factor 1.702 is carried by whichever quantity is free to carry the latent
scale. That depends on how the dimension is identified:

| Parameter | Dimension whose nodes are all probit **without** slopes (Rasch, GRM without a; θ variance estimated) | Any other dimension (θ variance fixed at 1, slopes free) |
|---|---|---|
| slope `a` | stays 1 | **× 1.702** at probit nodes, × 1 at logit nodes |
| `θ`, σ, latent-regression `β` | **× 1.702** (σ² × 1.702²) | unchanged |
| `b`, thresholds, `κ`, `γ` (everything inside `θ − b + offset`) | **× 1.702** | unchanged |
| correlations `ρ`, guessing `c` | unchanged | unchanged |
| covariance entries (Σ_θ, Σ_b, Σ_a) | product of the two factors | product of the two factors |

So a discrimination and a difficulty *at the same node* are treated differently,
and a Rasch difficulty is treated differently from a 2PL difficulty. A single
global multiply would corrupt every location parameter of a 2PL node, and "b
unchanged" would corrupt every Rasch difficulty (a Rasch logit program fixes
a = 1 and reports difficulties about 1.7 times the probit ones).

A location pooled across dimensions (shared b, pooled κ or γ) needs one metric,
so every dimension it links to a kept-metric dimension keeps its fitted metric
too, and the slope-free probit nodes there get the *implied* common slope 1.702
(reported in the output's notes). A slope shared between a probit and a logit
node has no single factor and is reported as NaN.

**The constant is 1.702, and this is not arbitrary.** Three independently-motivated
criteria all converge on it to four decimals:

| Criterion | Constant | max prob error |
|---|---|---|
| minimise the **max** probability gap between the probit and logit ICCs (Haley) | 1.7017 | 0.0095 |
| minimise the **integrated** squared probability gap | 1.7010 | 0.0095 |
| the **empirical** logit/probit coefficient ratio from refitting the same data | 1.7024 | 0.0095 |
| variance-matching (π/√3) | 1.8138 | 0.023 |
| slope-at-origin (4·φ(0)) | 1.5958 | 0.018 |

The three near 1.702 are the ones that make the two links agree *as
probabilities* — exactly the goal ("read probit as logit"). Variance-matching
(1.814) and the tangent/information constants optimise quantities the researcher
does not report and pay 2–3× the probability error; they are the wrong choice
here despite being "theoretically motivated" for other purposes.

**Is there a more accurate number?** No single constant makes probit and logit
*exactly* equal — they are different functions, so ~0.01 max probability error is
the floor for *any* constant, and 1.702 already sits on it. The only way to do
materially better is to abandon the constant: an optional **per-item ICC-matching
map** that finds, for each item, the logit `(a, b)` whose curve best matches the
probit curve by least squares over the ability range. That is exact-by-construction
per item and removes the ~1% residual — but it is a per-item optimisation rather
than a formula and does not extend cleanly to covariate `β`, so it is an optional
"exact" mode, not the default.

**Tolerance and labelling.** Because the 1.702 correspondence is approximate
(~1% in probability, more in the tails where the likelihoods weight
differently), the Axis-3 check budgets a discrepancy, not equality. For nodes
whose outcomes are mostly mid-range the gap is ~1–3%. For a node with rare
outcomes it is larger: on simulated logit data with a Rasch disengagement node
(P = 1500, I = 20, rapid-guess rates of roughly 3–20%), the converted probit fit
put the difficulties ~5% and σ² ~15% below a logit refit (β ~9% below, about
one posterior SD), while the 2PL ability node agreed within ~1–5%. The
converted values correlate ~1.00 with the refit, so the ordering and the
substantive story are preserved; the scale gap is the logistic's heavier tail.
The check therefore budgets ~5% on mid-range nodes and up to ~15% (or ~1–2
posterior SDs) on rare-outcome nodes, and
the feature must label its output **"logistic-equivalent (D = 1.702 rescaling),"
never "logistic estimates."** Even the exact per-item map cannot make the
*likelihoods* identical — two "equivalent" parameter sets still imply slightly
different standard errors — so the labelling caveat holds regardless of method.

**How it is judged (both references, per §9.1–9.2 choice of "both"):**
1. **Against a from-scratch logit refit on the same data** — the controlled check.
   Convert the probit fit with `fit.to_logit()` and confirm the converted
   values match the logit refit within the tolerance above. Any gap beyond that is a
   conversion bug, isolated from all other differences.
2. **Against the published PISA logit estimates (§9.2 Table 3)** — the in-the-wild
   check. The paper's β₁(Female) = −0.248 is a *logit* coefficient; this package's
   probit fit should give ≈ −0.248 / 1.702 ≈ −0.146 at the (slope-free)
   disengagement node, and `to_logit()` must bring it back near −0.248, within
   the rare-outcome tolerance above, since rapid guessing is rare. `ρ = −0.522` converts to itself. Reproducing the table after
   conversion is the strongest statement that the feature is correct.

### 9.4 Why this pair, and what it does NOT prove

The two anchors are complementary: Verbal Aggression validates the **tree core**
on neutral ground with an independent *estimator*; PISA validates the **full
disengagement feature set** against an independent *study* (the authors' own,
so the intended model is known exactly). Passing both means the machinery is a
correct IRTree *and* the disengagement-specific extensions land on published
science.

What Axis 3 does **not** prove: that the package is correct on models *no one has
published* — a novel tree, an untested covariate structure. Those are covered
only by Axes 1–2. External validity is a spot-check against known points, not a
proof over the whole model space; it is the last line of defence precisely
because it catches the errors that are invisible from the inside, but it catches
them only where someone has already been.

### 9.5 Practical notes

- **Reproducibility of the PISA fit** depends on access to the restricted-use
  merge of cognitive + questionnaire + timing data; the notebook documents the
  exact PISA 2022 variables and the RT-threshold rule (Normative Thresholds) used
  to derive the rapid-guess indicators, so the input is reconstructable.
- **The R comparison** (Verbal Aggression) is the more portable of the two and
  should be the default CI-able Axis-3 check; the PISA reproduction is a
  release-gate artifact, run manually.
- **The probit→logit conversion function itself** (the §9.3 feature) is testable
  with **no data at all** — build the chain layout of a known model and assert
  the per-role factors of the table above (`tests/test_logit_conversion.py`). That deterministic unit check is Axis 1 and
  runs on every commit; the Axis-3 notebook only adds *agreement with a real
  logit fit / the published table* on top of it.
- Both live in `50_external_crosscheck.ipynb`, split into a Verbal-Aggression
  section (independent-estimator agreement), a PISA section (published-number
  reproduction), and a scale-bridging section (probit fit → logit, vs both
  references), each ending in an explicit table and verdict.

## 10. What each check would have caught (validation of the design)

A cross-check that the architecture actually addresses the session's bugs:

| Bug | Caught by |
| --- | --- |
| GRM `int(NaN)` crash | Axis 1 invariant (`20_invariants`, GRM-runs) + Axis 1 output (`sample_Z_grm` first call) |
| `b_corr` `NameError` | Axis 1 invariant (config runs at all) / Axis 1 constraint check |
| Logit node 0% acceptance | Axis 1 invariant (MH-liveness) + Axis 2 group B (acceptance ≈ 0.44 per scalar block) |
| all-`NaN` log-posterior | Axis 1 invariant (finiteness) |
| shared `b`/`a`/`c` double-draw | Axis 1 output (pooled conditional pools stats) + Axis 2 group C (intervals too wide) |
| 2PL+covariate offset mismatch | Axis 1 output (`sample_Z` vs `log_prob` offset convention) + Axis 2 group C |
| `gibbs_a` ignores covariate offset | Axis 1 output (`gibbs_a` regressor includes offset) + Axis 2 group C |
| apparent κ collapse with free b | Identifiability probe (§5): start-dependence plus the exact posterior show a mixing failure, not non-identifiability; fixed by interweaving (ASIS) |
| data-prep mis-indexing (hypothetical) | Axis 1 output (`00_module_outputs`, hand-computed `Y_star`/`T`) |
| illegal model not rejected (hypothetical) | Axis 1 invariant (`00_module_outputs`, constraint validation) |
| convention/scale error (sign flip, logit-probit factor, ρ reference) | **Axis 3 only** (§9): reproduces published PISA / Verbal-Aggression numbers — invisible to Axes 1–2, which are internally self-consistent |
| probit→logit conversion wrong (global multiply, scaling `ρ`, or one rule for Rasch and 2PL nodes) | Axis 1 output (§9.3 feature: deterministic unit check on the conversion) + Axis 3 (converted fit reproduces the logit reference) |

Every silent failure maps to an axis-1 check, an axis-2 config, or (for a whole-package convention error) the axis-3 external check that would have made it loud. That mapping
is the acceptance criterion for the suite: a new bug class must extend the table,
and if it doesn't fit an existing axis, the architecture grows a new one.

This is not hypothetical for the whole table: the GRM-runs invariant **already
caught a regression on its first execution** — a NaN-safe cast fix that had been
lost from the delivered `node_models.py`. Bitwise regression would have blessed
the broken file (it matched an earlier broken snapshot); only a run-it-and-check
invariant flagged it.
