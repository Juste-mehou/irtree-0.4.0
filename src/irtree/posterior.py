"""
irtree.posterior
================
Layer 9 — Posterior summarization and model comparison (WAIC, IS-LOO).

PUBLIC API
==========
posterior_summary(results, param_names=None, hdi_level=0.95)
    Mean, SD, 2.5th/50th/97.5th percentiles, HDI and bulk ESS for every
    parameter.  Accepts a single MCMCResult or a list (chains are stacked).
    Returns a dict keyed by parameter name; summary_to_dataframe() turns it
    into a pandas DataFrame.

credible_interval(results, level=0.95, param_names=None, method='hdi')
    Highest-density interval (default) or equal-tail interval.
    HDI = shortest interval containing ``level`` posterior mass.
    Returns a dict mapping name → (lower, upper).

waic(results, data, node_models, spec=None, constraints=None)
    Watanabe-Akaike Information Criterion.
    waic = -2 × (lppd − p_waic)
    where lppd is the log pointwise predictive density and p_waic is
    the posterior variance of the log-likelihood (effective parameters).
    Returns WAICResult with waic, elpd_waic, p_waic, lppd, se_elpd.

loo_cv(results, data, node_models, spec=None, constraints=None)
    Leave-one-person-out cross-validation by importance sampling, with the
    Pareto k-hat diagnostic of Vehtari, Gelman & Gabry (2017).  The weights
    are NOT Pareto-smoothed (see loo_cv), so this is IS-LOO with a PSIS
    reliability diagnostic, not full PSIS-LOO.
    Returns LOOResult with elpd_loo, p_loo, k_hats, se_elpd, and
    a warning flag when any k_hat > 0.7.

    For waic and loo_cv, ``spec`` and ``constraints`` must be the ones used
    for the fit: they locate each node's columns (shared dimensions and
    parameter groups) and its covariates.

compare_waic(w1, w2), compare_loo(l1, l2)
    Differences in elpd with standard errors.

POINTWISE LOG-LIKELIHOOD
========================
Both WAIC and LOO use an [S × P] matrix of per-PERSON log-likelihoods: for
each draw, the log-probability of all of a person's pseudo-responses,
summed over items and nodes.

The matrix is built by ``_pointwise_log_lik_fast()``, which reads each
draw's parameters from the chain columns and rebuilds the covariate offset
with the sampler's own covariate index, column names and ``_node_offset``
(imputed missing covariates filled from the chain), so the linear
predictor is exactly the one the chain was drawn under.  A node whose
columns cannot be found raises an error instead of being skipped.

UNIT OF LOO CROSS-VALIDATION
=============================
LOO-CV is defined over persons (leave one person out at a time), not over
individual (person, item) responses.  This matches the IRTree sampling
design — within a person, responses across items are correlated through the
shared θ_p.  The per-person predictive density is:

    log p(y_p | y_{-p}) ≈ Σ_i Σ_r log p(Y*_{pir} | theta_{ps}, params_s)

summed over all items i and nodes r relevant to person p, evaluated at
each posterior draw s and importance-weighted.

REFERENCES
==========
Watanabe, S. (2010). Asymptotic equivalence of Bayes cross validation and
    widely applicable information criterion in singular learning theory.
    JMLR, 11, 3571–3594.

Vehtari, A., Gelman, A. & Gabry, J. (2017). Practical Bayesian model
    evaluation using leave-one-out cross-validation and WAIC.
    Statistics and Computing, 27(5), 1413–1432.

Vehtari, A., Simpson, D., Gelman, A., Yao, Y. & Gabry, J. (2024).
    Pareto smoothed importance sampling. Journal of Machine Learning
    Research, 25(72), 1–58.

Gelman, A., Hwang, J. & Vehtari, A. (2014). Understanding predictive
    information criteria for Bayesian models. Statistics and Computing,
    24(6), 997–1016.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

# ===========================================================================
# Result dataclasses
# ===========================================================================

@dataclass
class PosteriorSummary:
    """Summary statistics for a single parameter.

    Attributes
    ----------
    mean   : float — posterior mean
    sd     : float — posterior standard deviation
    q025   : float — 2.5th percentile
    q50    : float — 50th percentile (median)
    q975   : float — 97.5th percentile
    hdi_lo : float — lower bound of 95% highest-density interval
    hdi_hi : float — upper bound of 95% highest-density interval
    n_eff  : float — bulk effective sample size (from ESS computation)
    """
    mean:   float
    sd:     float
    q025:   float
    q50:    float
    q975:   float
    hdi_lo: float
    hdi_hi: float
    n_eff:  float

    def __repr__(self) -> str:
        return (f"PosteriorSummary(mean={self.mean:.3f}, sd={self.sd:.3f}, "
                f"95%HDI=[{self.hdi_lo:.3f}, {self.hdi_hi:.3f}])")


@dataclass
class WAICResult:
    """Output of waic().

    Attributes
    ----------
    waic      : float — WAIC = −2 × elpd_waic  (lower is better)
    elpd_waic : float — expected log pointwise predictive density (WAIC)
    p_waic    : float — effective number of parameters (posterior variance)
    lppd      : float — log pointwise predictive density (in-sample)
    se_elpd   : float — standard error of elpd_waic across observations
    pointwise : ndarray [N_obs] — per-observation elpd_waic contribution
    """
    waic:      float
    elpd_waic: float
    p_waic:    float
    lppd:      float
    se_elpd:   float
    pointwise: np.ndarray

    def __repr__(self) -> str:
        return (f"WAICResult(waic={self.waic:.2f}, elpd={self.elpd_waic:.2f}, "
                f"p_waic={self.p_waic:.2f}, se={self.se_elpd:.2f})")


@dataclass
class LOOResult:
    """Output of loo_cv().

    Attributes
    ----------
    elpd_loo  : float — expected log pointwise predictive density (LOO)
    p_loo     : float — effective number of parameters (LOO)
    se_elpd   : float — standard error of elpd_loo across persons
    k_hats    : ndarray [P] — Pareto k diagnostic, one per person
                k < 0.5: reliable; 0.5–0.7: OK; 0.7–1.0: warn; ≥ 1.0: bad
    n_warn    : int — number of persons with k_hat > 0.7
    pointwise : ndarray [P] — per-person elpd_loo contribution
    """
    elpd_loo:  float
    p_loo:     float
    se_elpd:   float
    k_hats:    np.ndarray
    n_warn:    int
    pointwise: np.ndarray

    @property
    def loo(self) -> float:
        """LOO-IC = −2 × elpd_loo  (lower is better, comparable to WAIC)."""
        return -2.0 * self.elpd_loo

    def __repr__(self) -> str:
        return (f"LOOResult(elpd_loo={self.elpd_loo:.2f}, p_loo={self.p_loo:.2f}, "
                f"se={self.se_elpd:.2f}, k_warn={self.n_warn}/{len(self.k_hats)})")


# ===========================================================================
# Internal helpers
# ===========================================================================

def _stack_chains(results) -> np.ndarray:
    """Stack one or more MCMCResult chains into a single [S_total, d] array.

    All chains must have the same d (number of parameters).  Chains are
    concatenated along axis 0 (samples stacked vertically).
    """
    from irtree.mcmc_sampler import MCMCResult

    if isinstance(results, MCMCResult):
        return results.chain.astype(np.float64)

    if not isinstance(results, list):
        results = [results]

    chains = [r.chain.astype(np.float64) for r in results]
    d_vals = [c.shape[1] for c in chains]
    if len(set(d_vals)) != 1:
        raise ValueError(
            f"All chains must have the same number of parameters d; "
            f"got d values {d_vals}."
        )
    return np.concatenate(chains, axis=0)


def _get_param_names(results) -> list[str]:
    """Extract parameter names from a result or list of results."""
    from irtree.mcmc_sampler import MCMCResult
    if isinstance(results, MCMCResult):
        return results.param_names
    if isinstance(results, list):
        return results[0].param_names
    return results.param_names


def _hdi_1d(samples: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    """Highest-density interval for a 1-D array of posterior samples.

    Algorithm (Chen & Shao 1999 / McElreath 2020):
      1. Sort the samples.
      2. The HDI is the shortest contiguous interval containing at least
         ``level`` fraction of the samples.
      3. Consider all n_intervals = n − n_in candidate intervals
         [x[i], x[i + n_in]] and return the one with minimum width.

    This is O(n log n) for the sort and O(n) for the scan.

    Parameters
    ----------
    samples : ndarray [S], posterior draws for one parameter
    level   : float, credible mass (default 0.95)

    Returns
    -------
    (lower, upper) : tuple[float, float]
    """
    x = np.sort(samples)
    n = len(x)
    n_in = int(np.floor(level * n))

    if n_in <= 0:
        return float(x[0]), float(x[-1])
    if n_in >= n:
        return float(x[0]), float(x[-1])

    n_intervals = n - n_in
    widths = x[n_in:n_in + n_intervals] - x[:n_intervals]
    idx = int(np.argmin(widths))
    return float(x[idx]), float(x[idx + n_in])


def _hdi_cols(chain: np.ndarray, level: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
    """_hdi_1d for every column of chain [S, d] (shortest interval per column)."""
    x = np.sort(chain, axis=0)
    n = x.shape[0]
    n_in = int(np.floor(level * n))
    if n_in <= 0 or n_in >= n:
        return x[0].copy(), x[-1].copy()
    n_intervals = n - n_in
    widths = x[n_in:n_in + n_intervals] - x[:n_intervals]
    idx = np.argmin(widths, axis=0)
    cols = np.arange(x.shape[1])
    return x[idx, cols], x[idx + n_in, cols]


def _eti_1d(samples: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    """Equal-tail credible interval."""
    alpha = 1.0 - level
    lo = float(np.percentile(samples, 100.0 * alpha / 2.0))
    hi = float(np.percentile(samples, 100.0 * (1.0 - alpha / 2.0)))
    return lo, hi


def _gpd_khat(excesses: np.ndarray) -> float:
    """Estimate the generalized-Pareto shape k from the largest weights' excesses.

    Method of moments: for a GPD with shape k, mean^2 / var = 1 - 2k, so
    k_mm = 0.5 (1 - mean^2 / var).  The estimate is then shrunk toward 0.5
    with a weight of 10 pseudo-observations, k = (n k_mm + 5) / (n + 10),
    the same shrinkage Vehtari et al. (2024) apply.  (Their base estimator is
    Zhang & Stephens' 2009 profile estimator, not the method of moments.)
    Returns 0.0 when fewer than 5 excesses or no spread.

    For k < 0.5: importance weights have finite variance → reliable LOO.
    For k ∈ [0.5, 0.7): estimates are noisier → mild caution.
    For k ∈ [0.7, 1.0): variance of weights is infinite → warn.
    For k ≥ 1.0: mean of weights is infinite → LOO unreliable.
    """
    n = len(excesses)
    if n < 5:
        return 0.0

    x = np.sort(excesses)
    x = x - x[0]  # shift minimum to 0

    xbar = float(np.mean(x))
    if xbar < 1e-14:
        return 0.0

    xvar = float(np.var(x, ddof=1))
    if xvar < 1e-14:
        return 0.0

    # Method-of-moments shape: k = 0.5 * (1 − mean²/var)
    k_mm = 0.5 * (1.0 - xbar ** 2 / xvar)

    # Regularization: shrink toward 0.5 with effective prior weight 10
    k_hat = (n * k_mm + 0.5 * 10.0) / (n + 10.0)
    return float(k_hat)


# ===========================================================================
# Pointwise log-likelihood builder
# ===========================================================================

def _pointwise_log_lik_fast(
    chain:       np.ndarray,
    param_names: list[str],
    data,
    node_models: list,
    spec=None,
    constraints=None,
) -> np.ndarray:
    """Fast vectorised per-person log-likelihood matrix [S, P].

    Handles both theta-sharing (via spec.node_to_latent) and item-parameter
    sharing (via constraints).  For a shared parameter, the representative
    node's chain columns are looked up rather than the node's own columns.

    Parameters
    ----------
    chain, param_names, data, node_models : as before
    spec        : IRTreeSpec or None — used to resolve theta column indices
                  when nodes share a latent dimension.
    constraints : Constraints or None — used to resolve b/a/c column indices
                  when nodes share item parameters.

    Returns
    -------
    log_lik_person : ndarray [S, P]
    """
    from irtree.constraints import NO_CONSTRAINTS

    S = chain.shape[0]
    P = data.n_persons
    I = data.n_items
    R = data.n_nodes

    name_to_col = {name: idx for idx, name in enumerate(param_names)}

    def _node_label(nodes_0based):
        return "&".join(str(r + 1) for r in sorted(nodes_0based))

    # ------------------------------------------------------------------
    # Build node → representative node lookups
    # ------------------------------------------------------------------

    # Theta: node r uses latent dimension d; chain columns are named
    # after the dimension index.  With no sharing d == r for all nodes.
    if spec is not None and spec.node_to_latent is not None:
        node_to_theta_dim = list(spec.node_to_latent)   # 0-based dim
    else:
        node_to_theta_dim = list(range(R))

    # b/a/c: node r uses the representative node of its sharing group.
    _con      = constraints if constraints is not None else NO_CONSTRAINTS
    b_groups  = _con.b_groups_for(R)
    b_n2g     = _con.b_node_to_group_for(R)
    a_groups  = _con.a_groups_for(R)
    a_n2g     = _con.a_node_to_group_for(R)
    c_groups  = _con.c_groups_for(R)
    c_n2g     = _con.c_node_to_group_for(R)

    # dim_to_nodes for theta label resolution
    # dim_to_nodes: the spec already computes this; the sampler reads the same
    # attribute (mcmc_sampler uses spec.dim_to_nodes directly).
    if spec is not None and getattr(spec, "dim_to_nodes", None) is not None:
        dim_to_nodes = spec.dim_to_nodes
    else:
        dim_to_nodes = [[r] for r in range(R)]

    # ------------------------------------------------------------------
    # Column resolver helpers
    # ------------------------------------------------------------------

    def _theta_cols(r):
        d    = node_to_theta_dim[r]
        label = _node_label(dim_to_nodes[d]) if spec is not None and spec.node_to_latent is not None \
                else str(d + 1)
        cols = [name_to_col.get(f"theta_{label}_{p}") for p in range(1, P+1)]
        return cols if all(c is not None for c in cols) else None

    def _b_cols(r, nm):
        """Chain columns of node r's b block, in the sampler's layout."""
        g_b         = b_n2g[r]
        group_label = _node_label(b_groups[g_b])
        if getattr(nm, "is_mixed", False):
            bin_set = set(np.asarray(nm.bin_idx).tolist())
            keys = []
            for i in range(I):
                if i in bin_set:
                    keys.append(f"b_{group_label}_{i + 1}")
                else:
                    keys.extend(f"b_{group_label}_{i + 1}_tau{k}"
                                for k in range(1, nm.M_poly))
        elif nm.n_outcomes > 2:
            keys = [f"b_{group_label}_{i}_thresh_{k}"
                    for i in range(1, I+1) for k in range(1, nm.n_outcomes)]
        else:
            keys = [f"b_{group_label}_{i}" for i in range(1, I+1)]
        cols = [name_to_col.get(k) for k in keys]
        return cols if all(c is not None for c in cols) else None

    def _b_value(nm, row):
        """Rebuild node b from one draw's b columns (plain array or NodeBParams)."""
        if getattr(nm, "is_mixed", False):
            from irtree.node_models import NodeBParams
            bin_set = set(np.asarray(nm.bin_idx).tolist())
            b_bin, b_poly, pos = [], [], 0
            for i in range(I):
                if i in bin_set:
                    b_bin.append(row[pos])
                    pos += 1
                else:
                    b_poly.append(row[pos:pos + nm.M_poly - 1])
                    pos += nm.M_poly - 1
            return NodeBParams(b_binary=np.asarray(b_bin), b_poly=np.asarray(b_poly),
                               bin_idx=nm.bin_idx, poly_idx=nm.poly_idx,
                               M_poly=nm.M_poly)
        if nm.n_outcomes > 2:
            return row.reshape(I, nm.n_outcomes - 1)
        return row

    def _a_cols(r):
        group_label = _node_label(a_groups[a_n2g[r]])
        cols  = [name_to_col.get(f"a_{group_label}_{i}") for i in range(1, I+1)]
        return cols if all(c is not None for c in cols) else None

    def _c_cols(r):
        group_label = _node_label(c_groups[c_n2g[r]])
        cols  = [name_to_col.get(f"c_{group_label}_{i}") for i in range(1, I+1)]
        return cols if all(c is not None for c in cols) else None

    # ------------------------------------------------------------------
    # Node parameter columns (a node that cannot be resolved is an error:
    # skipping it would silently drop its likelihood from WAIC / LOO)
    # ------------------------------------------------------------------
    node_cols = []
    for r, nm in enumerate(node_models):
        th_cols, bc = _theta_cols(r), _b_cols(r, nm)
        if th_cols is None or bc is None:
            raise ValueError(
                f"Node {r + 1}: cannot find its theta/b columns in the chain "
                f"(spec/constraints must be the ones used for the fit).")
        ac = _a_cols(r) if nm.has_discrimination else None
        cc = _c_cols(r) if nm.has_guessing else None
        if (nm.has_discrimination and ac is None) or (nm.has_guessing and cc is None):
            raise ValueError(f"Node {r + 1}: cannot find its a/c columns in the chain.")
        node_cols.append((th_cols, bc, ac, cc))

    # ------------------------------------------------------------------
    # Covariates: the sampler's own index, column names and offset function,
    # so the linear predictor here is the one the chain was drawn under.
    # Missing covariate values are filled with each draw's imputed values.
    # ------------------------------------------------------------------
    from irtree.data_prep import resolve_covariate_indices
    from irtree.mcmc_sampler import _node_offset, covariate_column_names

    cov_index = resolve_covariate_indices(node_models, data, spec=spec,
                                          constraints=constraints)
    coef_cols = {}
    for kind, blocks in covariate_column_names(cov_index, data).items():
        cols_k = []
        for names_g in blocks:
            cg = [name_to_col.get(n) for n in names_g]
            if any(c is None for c in cg):
                raise ValueError(
                    f"Cannot find covariate columns {names_g} in the chain.")
            cols_k.append(cg)
        coef_cols[kind] = cols_k
    any_cov = any(len(v) for v in coef_cols.values())

    def _missing_cols(prefix, units, feat_obs):
        out = []
        for u in units:
            u = tuple(np.atleast_1d(u))
            fm = np.where(~feat_obs[u])[0]
            tag = "_".join(str(int(x) + 1) for x in u)
            cs = [name_to_col.get(f"{prefix}_{tag}_{k + 1}") for k in fm]
            if any(c is None for c in cs):
                raise ValueError(f"Cannot find {prefix} columns for unit {tag}.")
            out.append((u, fm, cs))
        return out

    miss_X = (_missing_cols("X_missing", np.where(~data.X_observed)[0], data.X_feature_observed)
              if data.X is not None and data.n_missing_X > 0 else [])
    miss_Q = (_missing_cols("Q_missing", np.where(~data.Q_observed)[0], data.Q_feature_observed)
              if data.Q is not None and data.n_missing_Q > 0 else [])
    miss_W = (_missing_cols("W_missing", np.argwhere(~data.W_observed), data.W_feature_observed)
              if data.W is not None and data.n_missing_W > 0 else [])

    def _filled(M, miss, s):
        if M is None:
            return None
        if not miss:
            return M
        M = M.copy()
        for u, fm, cs in miss:
            M[u + (fm,)] = chain[s, cs]
        return M

    # ------------------------------------------------------------------
    # Main accumulation loop (draw-outer, so each draw's covariate arrays
    # are built once and shared by every node)
    # ------------------------------------------------------------------
    log_lik_person = np.zeros((S, P))
    for s in range(S):
        params_s = None
        if any_cov:
            params_s = {
                "beta":  [chain[s, c] for c in coef_cols["beta"]],
                "kappa": [chain[s, c] for c in coef_cols["kappa"]],
                "gamma": [chain[s, c] for c in coef_cols["gamma"]],
                "X_working": _filled(data.X, miss_X, s),
                "Q_working": _filled(data.Q, miss_Q, s),
                "W_working": _filled(data.W, miss_W, s),
            }
        for r, nm in enumerate(node_models):
            th_cols, bc, ac, cc = node_cols[r]
            offset_s = (_node_offset(r, params_s, cov_index, data)
                        if params_s is not None else None)
            lp = nm.log_prob(
                data.Y_star[r], chain[s, th_cols], _b_value(nm, chain[s, bc]),
                a_r=chain[s, ac] if ac is not None else None,
                c_r=chain[s, cc] if cc is not None else None,
                offset=offset_s)
            # np.where, not T * lp: irrelevant cells are NaN.
            log_lik_person[s] += np.where(data.T[r], lp, 0.0).sum(axis=1)

    return log_lik_person   # [S, P]


# ===========================================================================
# Public functions
# ===========================================================================

def posterior_summary(
    results,
    param_names: list[str] | None = None,
    hdi_level:   float = 0.95,
) -> dict[str, PosteriorSummary]:
    """Compute posterior summary statistics for all parameters.

    For each parameter column in the chain, computes:
      - Mean and SD
      - 2.5th, 50th, 97.5th percentiles (equal-tail 95% interval)
      - 95% highest-density interval (HDI)
      - Bulk effective sample size across chains (as in the diagnostics)

    Parameters
    ----------
    results      : MCMCResult or list[MCMCResult]
    param_names  : list[str] or None — override parameter names
    hdi_level    : float, credible mass for HDI (default 0.95)

    Returns
    -------
    dict[str, PosteriorSummary]
        One entry per parameter.  If pandas is available, also accessible
        as a DataFrame via ``summary_to_dataframe(result_dict)``.
    """
    chain = _stack_chains(results)
    names = param_names or _get_param_names(results)

    if len(names) != chain.shape[1]:
        raise ValueError(
            f"len(param_names)={len(names)} does not match chain width {chain.shape[1]}."
        )

    from irtree.diagnostics import _batch_diagnostics

    # All parameters at once (same numbers as the per-column definitions).
    means = chain.mean(axis=0)
    sds   = chain.std(axis=0, ddof=1)
    q025, q50, q975 = np.percentile(chain, [2.5, 50.0, 97.5], axis=0)
    hdi_lo, hdi_hi = _hdi_cols(chain, hdi_level)
    # n_eff: the rank-normalized bulk ESS across chains (Vehtari et al. 2021),
    # the same number the diagnostics report.  Chains of unequal length are
    # aligned on their last S_min draws.
    res_list = results if isinstance(results, list) else [results]
    if len(res_list) == 1:
        chains_3d = chain[:, None, :]
    else:
        per = [np.asarray(r.chain, dtype=np.float64) for r in res_list]
        s_min = min(c.shape[0] for c in per)
        chains_3d = np.stack([c[c.shape[0] - s_min:] for c in per], axis=1)
    n_eff = _batch_diagnostics(chains_3d, want_rhat=False, want_tail=False)["ess_bulk"]

    return {
        name: PosteriorSummary(
            mean=float(means[j]), sd=float(sds[j]),
            q025=float(q025[j]), q50=float(q50[j]), q975=float(q975[j]),
            hdi_lo=float(hdi_lo[j]), hdi_hi=float(hdi_hi[j]), n_eff=float(n_eff[j]),
        )
        for j, name in enumerate(names)
    }


def summary_to_dataframe(summaries: dict[str, PosteriorSummary]):
    """Convert the dict returned by posterior_summary() to a pandas DataFrame.

    Raises ImportError if pandas is not installed.
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "pandas is required for summary_to_dataframe(). "
            "Install with: pip install pandas"
        ) from exc

    rows = []
    for name, s in summaries.items():
        rows.append({
            "param":    name,
            "mean":     s.mean,
            "sd":       s.sd,
            "q2.5":     s.q025,
            "q50":      s.q50,
            "q97.5":    s.q975,
            "hdi_lo":   s.hdi_lo,
            "hdi_hi":   s.hdi_hi,
            "n_eff":    s.n_eff,
        })
    return pd.DataFrame(rows).set_index("param")


def credible_interval(
    results,
    level:       float = 0.95,
    param_names: list[str] | None = None,
    method:      str = "hdi",
) -> dict[str, tuple[float, float]]:
    """Compute credible intervals for all parameters.

    Parameters
    ----------
    results     : MCMCResult or list[MCMCResult]
    level       : float, credible mass (default 0.95)
    param_names : list[str] or None
    method      : 'hdi' (highest-density interval, default) or
                  'eti' (equal-tail interval, symmetric percentile-based)

    Returns
    -------
    dict[str, (lower, upper)]
        One (lower, upper) tuple per parameter.

    Notes
    -----
    HDI vs ETI:
      - The HDI is the shortest interval containing ``level`` mass.
        For symmetric, unimodal posteriors, HDI ≈ ETI.
        For skewed posteriors, HDI has strictly shorter width.
      - The ETI is the [alpha/2, 1−alpha/2] quantile interval.
        Simple to compute; the standard in many software packages.
    """
    if method not in ("hdi", "eti"):
        raise ValueError(f"method must be 'hdi' or 'eti'; got '{method}'.")
    if not (0 < level < 1):
        raise ValueError(f"level must be in (0, 1); got {level}.")

    chain = _stack_chains(results)
    names = param_names or _get_param_names(results)

    _interval = _hdi_1d if method == "hdi" else _eti_1d

    return {
        name: _interval(chain[:, idx], level)
        for idx, name in enumerate(names)
    }


def waic(
    results,
    data,
    node_models: list,
    spec=None,
    constraints=None,
) -> WAICResult:
    """Watanabe-Akaike Information Criterion (WAIC).

    DERIVATION (Watanabe 2010; Gelman, Hwang & Vehtari 2014):

        lppd     = Σ_p log( 1/S Σ_s p(y_p | theta_s) )
        p_waic   = Σ_p Var_s[ log p(y_p | theta_s) ]
        elpd_waic = lppd − p_waic
        WAIC     = −2 × elpd_waic

    where the sum is over persons p (LOO unit = person), S is the number
    of posterior draws, and theta_s is the full parameter vector at draw s.

    The variance term p_waic estimates the effective number of parameters
    consumed by fitting: a model that over-fits will have large per-person
    variance in log-likelihood across draws.

    Parameters
    ----------
    results     : MCMCResult or list[MCMCResult]
    data        : PreparedData from expand_data()
    node_models : list[NodeModel]
    spec        : IRTreeSpec used for the fit (shared latent dimensions)
    constraints : Constraints used for the fit (shared parameter groups),
                  or None.  A node whose columns cannot be found with these
                  raises ValueError.

    Returns
    -------
    WAICResult
        Fields: waic, elpd_waic, p_waic, lppd, se_elpd, pointwise [P].
    """
    chain      = _stack_chains(results)
    names      = _get_param_names(results)
    S, d       = chain.shape
    P          = data.n_persons

    # [S, P] log-likelihood matrix (per person, summed over items and nodes)
    log_lik_sp = _pointwise_log_lik_fast(chain, names, data, node_models,
                                         spec=spec, constraints=constraints)

    # lppd: log pointwise predictive density
    # lppd_p = log( 1/S Σ_s exp(log_lik_sp[:, p]) ) = logsumexp - log(S)
    lppd_terms = logsumexp(log_lik_sp, axis=0) - np.log(S)   # [P]
    lppd       = float(np.sum(lppd_terms))

    # p_waic: posterior variance of per-person log-likelihood
    p_waic_terms = np.var(log_lik_sp, axis=0, ddof=1)   # [P]
    p_waic       = float(np.sum(p_waic_terms))

    elpd_waic = lppd - p_waic
    waic_val  = -2.0 * elpd_waic

    # Pointwise elpd: lppd_p - p_waic_p (per person)
    pointwise = lppd_terms - p_waic_terms   # [P]

    # Standard error of elpd_waic across persons
    se_elpd = float(np.std(pointwise, ddof=1) * np.sqrt(P))

    return WAICResult(
        waic=float(waic_val),
        elpd_waic=float(elpd_waic),
        p_waic=float(p_waic),
        lppd=float(lppd),
        se_elpd=float(se_elpd),
        pointwise=pointwise.astype(np.float64),
    )


def loo_cv(
    results,
    data,
    node_models: list,
    spec=None,
    constraints=None,
) -> LOOResult:
    """Leave-one-person-out cross-validation by importance sampling (IS-LOO).

    DERIVATION (Vehtari, Gelman & Gabry 2017):

    The LOO predictive density for person p is:

        p(y_p | y_{-p}) ≈ harmonic mean over draws of p(y_p | theta_s)
                        = S / Σ_s (1/p(y_p | theta_s))
                        = exp( log(S) − logsumexp(−log_lik_p) )

    This is the importance-sampling LOO estimator with unnormalized weights
    r_s = p(theta_s | y_{-p}) / p(theta_s | y) ∝ 1/p(y_p | theta_s).

    The Pareto k diagnostic assesses the reliability of these importance
    weights.  Large k (> 0.7) means the tail of the weight distribution
    is too heavy for reliable estimation; those persons are difficult to
    predict given the rest of the data.

    NO PARETO SMOOTHING:
    The raw importance-sampling (harmonic mean) estimator is used; the
    largest weights are not replaced by order statistics of the fitted
    generalized Pareto, as full PSIS-LOO does.  k-hat is computed and
    reported as the reliability diagnostic, and its interpretation is the
    same (k > 0.7: the estimate for that person is unreliable), but with
    heavy-tailed weights the raw estimator has higher variance than PSIS.

    Parameters
    ----------
    results     : MCMCResult or list[MCMCResult]
    data        : PreparedData from expand_data()
    node_models : list[NodeModel]
    spec        : IRTreeSpec used for the fit (shared latent dimensions)
    constraints : Constraints used for the fit (shared parameter groups),
                  or None.  A node whose columns cannot be found with these
                  raises ValueError.

    Returns
    -------
    LOOResult
        Fields: elpd_loo, p_loo, se_elpd, k_hats [P], n_warn, pointwise [P].
    """
    chain  = _stack_chains(results)
    names  = _get_param_names(results)
    S, d   = chain.shape
    P      = data.n_persons

    # [S, P] log-likelihood matrix
    log_lik_sp = _pointwise_log_lik_fast(chain, names, data, node_models,
                                         spec=spec, constraints=constraints)

    # LOO predictive density per person: harmonic mean estimator
    # elpd_p = log S - logsumexp(-log_lik_sp[:, p])
    loo_terms = np.log(S) - logsumexp(-log_lik_sp, axis=0)   # [P]
    elpd_loo  = float(np.sum(loo_terms))

    # lppd for p_loo calculation
    lppd_terms = logsumexp(log_lik_sp, axis=0) - np.log(S)
    lppd       = float(np.sum(lppd_terms))
    p_loo      = float(lppd - elpd_loo)

    # Standard error across persons
    se_elpd = float(np.std(loo_terms, ddof=1) * np.sqrt(P))

    # Pareto k diagnostic per person
    k_hats = np.empty(P)
    for p in range(P):
        # Importance weights: w_s ∝ 1/p(y_p | theta_s) = exp(-log_lik)
        log_w    = -log_lik_sp[:, p]
        log_w_max = np.max(log_w)
        w        = np.exp(log_w - log_w_max)

        # Tail: top M = min(S//5, 3*sqrt(S)) weights
        M     = max(5, min(S // 5, int(3 * np.sqrt(S))))
        w_sorted  = np.sort(w)
        threshold = w_sorted[max(0, len(w_sorted) - M - 1)]
        w_tail    = w_sorted[-M:] - threshold

        k_hats[p] = _gpd_khat(w_tail)

    n_warn = int(np.sum(k_hats > 0.7))

    return LOOResult(
        elpd_loo=float(elpd_loo),
        p_loo=float(p_loo),
        se_elpd=float(se_elpd),
        k_hats=k_hats.astype(np.float64),
        n_warn=n_warn,
        pointwise=loo_terms.astype(np.float64),
    )


# ===========================================================================
# Convenience: compare two models
# ===========================================================================

def compare_waic(waic1: WAICResult, waic2: WAICResult) -> dict[str, float]:
    """Compare two models via WAIC.

    Returns a dict with:
      - 'delta_elpd': elpd_waic2 − elpd_waic1 (positive → model 2 better)
      - 'se_delta': standard error of the difference, from the two SEs
        (ignores the correlation between the models' pointwise values)
      - 'delta_waic': waic2 − waic1 = −2 × delta_elpd (positive → model 2
        worse)
    """
    delta_elpd = waic2.elpd_waic - waic1.elpd_waic
    # SE of difference: sqrt(SE1² + SE2²) (approximate, ignores correlation)
    se_delta = float(np.sqrt(waic1.se_elpd ** 2 + waic2.se_elpd ** 2))
    return {
        "delta_elpd": float(delta_elpd),
        "se_delta":   se_delta,
        "delta_waic": float(-2.0 * delta_elpd),
    }


def compare_loo(loo1: LOOResult, loo2: LOOResult) -> dict[str, float]:
    """Compare two models via LOO-CV.

    Returns delta_elpd, se_delta, delta_loo_ic (= −2 × delta_elpd).
    The se_delta is computed from the pointwise differences (Vehtari 2017).
    """
    if len(loo1.pointwise) == len(loo2.pointwise):
        # Pointwise SE: more accurate because it uses the correlation structure
        diff = loo2.pointwise - loo1.pointwise
        se_delta = float(np.std(diff, ddof=1) * np.sqrt(len(diff)))
    else:
        # Fallback: approximate
        se_delta = float(np.sqrt(loo1.se_elpd ** 2 + loo2.se_elpd ** 2))

    delta_elpd = loo2.elpd_loo - loo1.elpd_loo
    return {
        "delta_elpd":  float(delta_elpd),
        "se_delta":    se_delta,
        "delta_loo_ic": float(-2.0 * delta_elpd),
    }


# ===========================================================================
# Logistic-equivalent rescaling of a probit fit (fit.to_logit)
# ===========================================================================

#: Probit-to-logit constant: Phi(x) is within 0.0095 of L(1.702 x) for all x,
#: the smallest maximum gap any single constant achieves (Haley, 1952).
LOGIT_D = 1.702


def logit_scale_factors(param_names, data, spec, node_models, priors,
                        constraints=None):
    """Per-column multipliers that put a fit on the logistic-equivalent scale.

    Premise: a probit node's probability Phi(eta) is approximately
    L(1.702 eta), so its logistic-equivalent linear predictor is 1.702 times
    the probit one.  With eta = a (theta - b + offset), the factor 1.702 is
    carried by whichever quantity is free to carry the latent scale:

    * a latent dimension whose nodes are all probit **without** slopes
      (Rasch, GRM without a): a stays 1, so the theta metric is stretched.
      theta, b, thresholds, covariate effects (beta, kappa, gamma) are
      multiplied by s = 1.702, variances by s**2.
    * any other dimension (s = 1): the theta metric is kept, and every slope
      at node r is multiplied by k_r / s, with k_r = 1.702 at probit nodes
      and 1 at logit (and custom) nodes.

    Correlations, guessing parameters and imputed covariate values are
    unchanged; covariance entries take the product of the two factors.

    Returns
    -------
    factors : np.ndarray [n_params]
        Multiplier for each chain column; NaN where no single factor exists
        (a pooled parameter whose nodes would need different factors).
    notes : list[str]
        Human-readable caveats (implied slopes, custom nodes, NaN columns).
    """
    from irtree.constraints import NO_CONSTRAINTS
    from irtree.data_prep import resolve_covariate_indices
    from irtree.mcmc_sampler import (
        _build_chain_layout,
        _effective_priors,
        _node_label,
        _theta_corr_blocks,
    )

    R = data.n_nodes
    con = constraints if constraints is not None else NO_CONSTRAINTS
    eff_priors = _effective_priors(priors, constraints, R)
    cov_index = resolve_covariate_indices(node_models, data, spec=spec,
                                          constraints=constraints)
    names, offsets = _build_chain_layout(data, node_models, eff_priors, spec=spec,
                                         constraints=constraints, cov_index=cov_index)
    if list(names) != list(param_names):
        raise RuntimeError(
            "logit_scale_factors: the rebuilt chain layout does not match the "
            "fit's parameter names, so columns cannot be matched to roles.")

    notes = []
    dim_to_nodes = spec.dim_to_nodes
    D = len(dim_to_nodes)
    node_to_dim = {r: d for d, grp in enumerate(dim_to_nodes) for r in grp}
    a_nodes = set(eff_priors._param_nodes.get("a", []))

    link = [getattr(nm, "link", "custom") for nm in node_models]
    k = [LOGIT_D if lk == "probit" else 1.0 for lk in link]
    s = [LOGIT_D if all(link[r] == "probit" and r not in a_nodes for r in grp) else 1.0
         for grp in dim_to_nodes]
    # A location pooled across dimensions (shared b, pooled kappa/gamma) needs
    # one metric on all of them; keep the fitted metric (s = 1) on every
    # dimension it links to a kept-metric dimension, until nothing changes.
    # Their slope-free probit nodes then get the implied common slope 1.702.
    linked = [list(grp) for grp in con.b_groups_for(R)]
    linked += [list(grp) for grp in cov_index.q_groups + cov_index.w_groups]
    changed = True
    while changed:
        changed = False
        for grp in linked:
            dims = {node_to_dim[r] for r in grp}
            if len({s[d] for d in dims}) > 1:
                for d in dims:
                    s[d] = 1.0
                changed = True

    for r in range(R):
        if link[r] == "custom":
            notes.append(f"node {r + 1}: custom link, its slopes are reported as fitted.")
        elif link[r] == "probit" and r not in a_nodes and s[node_to_dim[r]] == 1.0:
            notes.append(
                f"node {r + 1}: probit without slopes on a dimension whose metric is "
                f"kept; its logistic-equivalent slope is the common value {LOGIT_D}.")

    factors = np.full(len(names), np.nan)
    ambiguous = []

    def _put(key, value):
        if key in offsets:
            lo, hi = offsets[key]
            factors[lo:hi] = value

    def _group_factor(kind, nodes, per_node):
        vals = {round(per_node(r), 12) for r in nodes}
        if len(vals) == 1:
            return vals.pop()
        ambiguous.append(f"{kind}_{_node_label(nodes)}")
        return np.nan

    # theta, sigma2 per dimension
    for d in range(D):
        _put(f"theta_{d}", s[d])
        _put(f"sigma2_{d}", s[d] ** 2)

    # b (locations / thresholds) and their hierarchical variance, per b-group
    b_groups = con.b_groups_for(R)
    b_fac = [_group_factor("b", grp, lambda r: s[node_to_dim[r]]) for grp in b_groups]
    for g, f in enumerate(b_fac):
        _put(f"b_{g}", f)
        _put(f"b_var_{g}", f ** 2)

    # a (slopes) and their variance, per a-group
    a_groups = con.a_groups_for(R)
    a_fac = [_group_factor("a", grp, lambda r: k[r] / s[node_to_dim[r]]) for grp in a_groups]
    for g, f in enumerate(a_fac):
        _put(f"a_{g}", f)
        _put(f"a_var_{g}", f ** 2)

    # guessing: a probability, unchanged
    for g in range(len(con.c_groups_for(R))):
        _put(f"c_{g}", 1.0)

    # theta correlation / covariance
    if "rho" in offsets:
        lo, hi = offsets["rho"]
        _blk, pairs, tri = _theta_corr_blocks(constraints, D)
        if hi - lo == len(tri) and len(tri) != len(pairs):
            factors[lo:hi] = [s[d1] * s[d2] for d1, d2 in tri]   # Sigma_theta
        else:
            factors[lo:hi] = 1.0                                  # rho

    # cross-node b / a correlation (rho) or covariance (Sigma) blocks
    for kind, grp_fac in (("b", b_fac), ("a", a_fac)):
        cgs = con.b_corr_groups_for(R) if kind == "b" else con.a_corr_groups_for(R)
        for idx, cg in enumerate(cgs):
            _put(f"{kind}_corr_rho_{idx}", 1.0)
            key = f"{kind}_corr_Sigma_{idx}"
            if key in offsets:
                lo, hi = offsets[key]
                factors[lo:hi] = [grp_fac[cg[i]] * grp_fac[cg[j]]
                                  for i in range(len(cg)) for j in range(i, len(cg))]

    # covariate effects live inside (theta - b + offset): theta metric of the node
    for kind, groups in (("beta", cov_index.x_groups), ("kappa", cov_index.q_groups),
                         ("gamma", cov_index.w_groups)):
        for g, grp in enumerate(groups):
            f_g = _group_factor(kind, grp, lambda r: s[node_to_dim[r]])
            _put(f"{kind}_{g}", f_g)
            _put(f"{kind}_var_{g}", f_g ** 2)

    # imputed covariate values: data, not parameters of the latent scale
    for key in offsets:
        if key.startswith(("X_missing_", "Q_missing_", "W_missing_")):
            _put(key, 1.0)

    if ambiguous:
        notes.append(
            "no single factor for " + ", ".join(ambiguous) + " (pooled across nodes "
            "whose logistic metrics differ); reported as NaN.")
    return factors, notes
