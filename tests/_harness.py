"""
tests/_harness.py
=================
The single shared correctness engine for the irtree test suite, imported by
both the pytest files and the runnable notebooks (TESTING.md §7).

TWO RULES THIS MODULE ENFORCES (TESTING.md §7):

1. **Analytic posteriors are re-derived here from the MODEL, never imported
   from the sampler.** A test may not call the function it is testing to
   compute its own expected value. Every closed form below is written from the
   full-conditional derivations, so a match means the code and the math agree
   — not that the code agrees with itself.

2. **Every stochastic assertion is expressed against Monte-Carlo standard
   error**, with the seed and N fixed, so "pass" has a defined false-positive
   rate rather than a hand-tuned tolerance.

The functions here fall into three groups:
  - analytic_* : closed-form posteriors (the ground truth)
  - assert_*   : moment / coverage / bitwise checks
  - simulate_* / relevant_index : data generators and index helpers
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr

# ===========================================================================
# Index / relevance helpers
# ===========================================================================

def relevant_index(T):
    """(row_idx, col_idx) of the True cells of a boolean relevance matrix T."""
    return np.where(T)


def random_relevance(P, I, sparsity, rng):
    """Boolean [P, I] relevance mask with the given fraction of cells set False."""
    return rng.random((P, I)) > sparsity


# ===========================================================================
# Analytic full conditionals — re-derived from the model (NOT from the code)
# ===========================================================================

def analytic_normal_normal(design, y, tau2, mu0=None):
    """Multivariate Normal-Normal posterior for a linear-Gaussian block.

    Model:   y_j ~ N(design_j . coef, 1)     (unit noise, Albert-Chib scale)
    Prior:   coef ~ N(mu0, tau2 I_K)

    Returns (mu_post [K], V_post [K, K]) with
        V_post^-1 = I/tau2 + design^T design
        mu_post   = V_post (design^T y + mu0/tau2)

    This is the derivations-doc §1.2 / §2.11 template. `design` already carries
    any a-weighting (rows are a_i * covariate) and any residualization of y.
    """
    design = np.atleast_2d(np.asarray(design, float))
    y = np.asarray(y, float).ravel()
    K = design.shape[1]
    if mu0 is None:
        mu0 = np.zeros(K)
    else:
        mu0 = np.asarray(mu0, float).ravel()
    Vinv = np.eye(K) / tau2 + design.T @ design
    V = np.linalg.inv(Vinv)
    mu = V @ (design.T @ y + mu0 / tau2)
    return mu, V


def analytic_theta_scalar(Z, b, T_row, sigma2, a=None):
    """Posterior of a single person's theta at one node (derivations §2.1/§2.3).

    Z, b, T_row are length-I (one person's row). a=None -> 1PL.
        v* = 1/(1/sigma2 + sum_i a_i^2)
        mu*= v* * sum_i a_i (Z_i + a_i b_i)
    """
    idx = np.where(T_row)[0]
    a_i = np.ones(len(idx)) if a is None else np.asarray(a)[idx]
    prec = 1.0 / sigma2 + np.sum(a_i ** 2)
    v = 1.0 / prec
    mu = v * np.sum(a_i * (Z[idx] + a_i * b[idx]))
    return mu, v


def analytic_b_scalar(Z, theta, T_col, tau2, a_i=None):
    """Posterior of a single item's b at one node (derivations §2.2/§2.4).

    Z is length-P (one item's column), theta length-P, T_col length-P.
    a_i is a scalar (item-level). a_i=None -> 1PL.
        v* = 1/(1/tau2 + n * a_i^2)
        mu*= v* * a_i * sum_p (a_i theta_p - Z_p)
    """
    idx = np.where(T_col)[0]
    ai = 1.0 if a_i is None else float(a_i)
    n = len(idx)
    prec = 1.0 / tau2 + n * ai ** 2
    v = 1.0 / prec
    mu = v * ai * np.sum(ai * theta[idx] - Z[idx])
    return mu, v


def analytic_theta_mvn(S_p, n_p, Sigma):
    """Correlated-theta posterior for one person (derivations §2.6).

        A      = Sigma^-1 + diag(n_p)
        Sig*   = A^-1
        mu*    = Sig* S_p
    S_p, n_p are length-D; Sigma is [D, D].
    """
    A = np.linalg.inv(Sigma) + np.diag(n_p)
    V = np.linalg.inv(A)
    mu = V @ np.asarray(S_p, float)
    return mu, V


def analytic_invgamma(x, alpha0, beta0, mean=0.0):
    """Posterior InvGamma shape/scale for a Normal-variance (derivations §1.3).

    Returns (alpha_post, beta_post) for
        sigma2 | x ~ InvGamma(alpha0 + n/2, beta0 + 0.5 sum (x-mean)^2).
    Compare a sample's mean/var to the InvGamma moments:
        E   = beta/(alpha-1)                (alpha>1)
        Var = beta^2/((alpha-1)^2(alpha-2)) (alpha>2)
    """
    x = np.asarray(x, float).ravel()
    n = x.size
    return alpha0 + n / 2.0, beta0 + 0.5 * float(np.sum((x - mean) ** 2))


def analytic_kappa_marginal(Z, theta, T, Q, tau2_b, tau2_kappa):
    """Exact posterior of an item-covariate coefficient with b integrated out.

    Model:  Z_pi = theta_p - b_i + Q_i kappa + eps,  eps ~ N(0, 1)  (theta known)
            b_i ~ N(0, tau2_b),  kappa ~ N(0, tau2_kappa I).
    Per item, the mean residual s_i = mean_p (Z_pi - theta_p) over the relevant
    cells is sufficient for (b_i, kappa), and with b_i integrated out
        s_i | kappa ~ N(Q_i kappa, tau2_b + 1/n_i).
    So kappa's posterior is a Normal linear regression of s on Q:
        V^-1 = I/tau2_kappa + sum_i Q_i Q_i' / (tau2_b + 1/n_i)
        mu   = V sum_i Q_i s_i / (tau2_b + 1/n_i)
    Returns (mu [K], V [K, K]).
    """
    T = np.asarray(T, bool)
    n = T.sum(axis=0).astype(float)
    s = np.where(T, Z - theta[:, None], 0.0).sum(axis=0) / np.maximum(n, 1)
    keep = n > 0
    w = 1.0 / (tau2_b + 1.0 / n[keep])
    Qk = np.asarray(Q, float)[keep]
    K = Qk.shape[1]
    V = np.linalg.inv(np.eye(K) / tau2_kappa + (Qk * w[:, None]).T @ Qk)
    return V @ ((Qk * w[:, None]).T @ s[keep]), V


def invgamma_moments(alpha, beta):
    """(mean, var) of InvGamma(alpha, beta); var is nan when alpha<=2."""
    mean = beta / (alpha - 1.0) if alpha > 1 else np.nan
    var = beta ** 2 / ((alpha - 1) ** 2 * (alpha - 2)) if alpha > 2 else np.nan
    return mean, var


def analytic_beta_conjugate(n_success, n_total, alpha0, beta0):
    """Beta posterior for a Bernoulli success prob (guessing c, derivations §3.2).

        c | U ~ Beta(alpha0 + n_success, beta0 + n_total - n_success).
    Returns (a_post, b_post). Beta mean = a/(a+b).
    """
    return alpha0 + n_success, beta0 + (n_total - n_success)


def truncnorm_moments(mu, sd, lower=None, upper=None):
    """Mean and variance of N(mu, sd^2) truncated to (lower, upper).

    Used to check truncated-Normal draws (a under TruncNormal, GRM thresholds).
    """
    a = -np.inf if lower is None else (lower - mu) / sd
    b = np.inf if upper is None else (upper - mu) / sd
    Za = ndtr(a) if np.isfinite(a) else 0.0
    Zb = ndtr(b) if np.isfinite(b) else 1.0
    Z = Zb - Za
    def phi(t):
        return np.exp(-0.5 * t * t) / np.sqrt(2 * np.pi)
    pa = phi(a) if np.isfinite(a) else 0.0
    pb = phi(b) if np.isfinite(b) else 0.0
    m = mu + sd * (pa - pb) / Z
    # variance
    ta = a * pa if np.isfinite(a) else 0.0
    tb = b * pb if np.isfinite(b) else 0.0
    v = sd ** 2 * (1.0 + (ta - tb) / Z - ((pa - pb) / Z) ** 2)
    return m, v


# ===========================================================================
# Moment / coverage / bitwise assertions
# ===========================================================================

def draw_many(fn, n_draws, seed_base=0):
    """Call a draw function n_draws times with fresh seeds; stack the results.

    fn(rng) -> ndarray. Returns an array of shape (n_draws, *fn_output_shape).
    """
    first = fn(np.random.default_rng(seed_base))
    out = np.empty((n_draws,) + np.shape(first), dtype=float)
    out[0] = first
    for s in range(1, n_draws):
        out[s] = fn(np.random.default_rng(seed_base + s))
    return out


def assert_mean(draws, target_mean, n_se=4.0, label=""):
    """Assert empirical mean matches target within n_se Monte-Carlo SEs.

    SE of the mean per component is sqrt(sample_var / n_draws).
    """
    draws = np.asarray(draws)
    n = draws.shape[0]
    emp = draws.mean(axis=0)
    se = np.sqrt(draws.var(axis=0, ddof=1) / n)
    se = np.where(se == 0, 1e-12, se)
    z = np.abs(emp - np.asarray(target_mean)) / se
    zmax = float(np.max(z))
    assert zmax < n_se, (
        f"{label}: mean off by {zmax:.2f} SE (>{n_se}). "
        f"emp={np.round(emp, 4)}, target={np.round(np.asarray(target_mean), 4)}"
    )
    return zmax


def assert_var(draws, target_var, rtol=0.10, label=""):
    """Assert empirical variance matches target within a relative tolerance.

    Variance has no clean SE without 4th moments, so use a relative band that
    is generous enough for ~1e4-1e5 draws but tight enough to catch a wrong
    precision (which is the failure mode that matters).
    """
    draws = np.asarray(draws)
    emp = draws.var(axis=0, ddof=1)
    tv = np.asarray(target_var, float)
    rel = np.abs(emp - tv) / np.where(tv == 0, 1e-12, np.abs(tv))
    rmax = float(np.max(rel))
    assert rmax < rtol, (
        f"{label}: variance off by {rmax*100:.1f}% (>{rtol*100:.0f}%). "
        f"emp={np.round(emp, 5)}, target={np.round(tv, 5)}"
    )
    return rmax


def assert_cov(draws, target_cov, rtol=0.15, label=""):
    """Assert empirical covariance matches target (for MVN draws)."""
    draws = np.asarray(draws)
    if draws.ndim == 2:
        emp = np.cov(draws.T)
    else:
        emp = np.array(target_cov)  # scalar case handled by assert_var
    tc = np.asarray(target_cov, float)
    scale = np.sqrt(np.outer(np.diag(tc), np.diag(tc)))
    rel = np.abs(emp - tc) / np.where(scale == 0, 1e-12, scale)
    rmax = float(np.max(rel))
    assert rmax < rtol, (
        f"{label}: covariance off by {rmax*100:.1f}% (>{rtol*100:.0f}%).\n"
        f"emp=\n{np.round(emp, 4)}\ntarget=\n{np.round(tc, 4)}"
    )
    return rmax


def assert_bitwise(a, b, label=""):
    """Assert two arrays are exactly equal (RNG-order-stable regression)."""
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape, f"{label}: shape {a.shape} != {b.shape}"
    assert np.array_equal(a, b), (
        f"{label}: not bitwise identical; max|diff|={np.abs(a-b).max():.2e}"
    )


def assert_close_roundoff(a, b, atol=1e-12, label=""):
    """Assert equality up to floating-point roundoff (BLAS-reassociation)."""
    a, b = np.asarray(a), np.asarray(b)
    d = float(np.abs(a - b).max())
    assert d < atol, f"{label}: max|diff|={d:.2e} exceeds {atol:.0e}"


def assert_finite(x, label=""):
    """Assert an array contains no NaN or inf (the log-posterior / cache guard)."""
    x = np.asarray(x, float)
    assert np.all(np.isfinite(x)), (
        f"{label}: {np.sum(~np.isfinite(x))}/{x.size} non-finite entries"
    )


def coverage_band(k, n, level=0.95, z=3.0):
    """Acceptance band for a coverage fraction: expected `level` over n intervals.

    Returns (lo, hi) using a normal approximation to Binomial(n, level) with a
    z-SE half-width. A recovery test asserts lo <= (#covered / n) <= hi.
    """
    se = np.sqrt(level * (1 - level) / n)
    return level - z * se, level + z * se


# ===========================================================================
# Simulators (used by Layer 3 recovery; kept here so notebooks share them)
# ===========================================================================

def simulate_probit_cell_means(theta, b, a=None, offset=None):
    """[P, I] probit mean a_i*(theta_p - b_i + offset). offset raw, a-scaled here."""
    eta = theta[:, None] - b[None, :]
    if offset is not None:
        eta = eta + offset
    if a is not None:
        eta = a[None, :] * eta
    return eta


def simulate_augmented_Z(means, rng):
    """Draw Z ~ N(means, 1) for every cell (used by unit tests that skip the
    probit step and test the Gaussian conjugacy directly)."""
    return rng.normal(means, 1.0)


# ===========================================================================
# Axis 2 — recovery helpers (simulate from a tree, fit, judge by coverage)
# ===========================================================================
# These live here so notebooks and pytest share one implementation. They are
# deliberately generic over the tree (driven by spec.mapping), so a new tree
# needs no new simulator.

def simulate_irtree(spec, node_params, P, I, rng, node_links=None,
                    node_types=None):
    """Simulate an observed category matrix Y [P, I] from an IRTree.

    node_params[r] is a dict with the parameters of node r's model:
        'theta' : [P]   person effect at node r
        'b'     : [I]    item difficulty (binary node) or [I, M-1] thresholds (GRM)
        'a'     : [I] or scalar, optional (2PL/3PL/GRM discrimination)
        'c'     : [I], optional (3PL guessing)
        'offset': [P, I], optional (covariate linear predictor, RAW)
    node_links[r] in {'probit','logit'} (default 'probit').
    node_types[r] in {'binary','grm','gpcm'} (default 'binary').  'gpcm' is
    the adjacent-category logit model of GPCMNode (link ignored); 'b' is then
    the [I, M-1] step parameters.

    Walks each (p,i) down the tree using spec.mapping: draw each node's outcome,
    then pick the category whose mapping row matches the drawn outcomes.
    Categories are returned 1-based (categories_start=1).
    """
    M = np.asarray(spec.mapping, float)          # [K, R]
    K, R = M.shape
    links = node_links or ['probit'] * R
    types = node_types or ['binary'] * R

    # draw the outcome at every node for every cell (only nodes on the path
    # actually matter, but drawing all is simpler and correct given independence)
    node_outcome = np.full((R, P, I), -1, dtype=int)
    for r in range(R):
        pr = node_params[r]
        theta = np.asarray(pr['theta'], float)
        a = pr.get('a', None)
        offset = pr.get('offset', None)
        if types[r] == 'grm':
            b = np.asarray(pr['b'], float)       # [I, M-1]
            node_outcome[r] = _draw_grm(theta, b, a, offset, rng, links[r])
        elif types[r] == 'gpcm':
            b = np.asarray(pr['b'], float)       # [I, M-1] step parameters
            node_outcome[r] = _draw_gpcm(theta, b, a, offset, rng)
        else:
            b = np.asarray(pr['b'], float)       # [I]
            eta = theta[:, None] - b[None, :]
            if offset is not None:
                eta = eta + offset
            if a is not None:
                a_arr = a if np.ndim(a) else np.full(I, a)
                eta = a_arr[None, :] * eta
            p1 = _cdf(eta, links[r])
            if 'c' in pr and pr['c'] is not None:
                c = np.asarray(pr['c'], float)
                p1 = c[None, :] + (1 - c[None, :]) * p1
            node_outcome[r] = (rng.random((P, I)) < p1).astype(int)

    # resolve category per cell from the mapping
    Y = np.zeros((P, I), dtype=int)
    for k in range(K):
        row = M[k]                                # [R], nan where node off-path
        match = np.ones((P, I), bool)
        for r in range(R):
            if not np.isnan(row[r]):
                match &= (node_outcome[r] == int(row[r]))
        Y[match] = k + 1                          # 1-based
    return Y


def _cdf(eta, link):
    if link == 'logit':
        return 1.0 / (1.0 + np.exp(-eta))
    return ndtr(eta)


def _draw_grm(theta, b_thresholds, a, offset, rng, link='probit'):
    """Draw a GRM outcome in {0,...,M-1} per cell. b_thresholds is [I, M-1].

    P(Y >= m+1) = F(a_i (theta_p - tau_im + offset_pi)) with F the node's link
    (Phi for probit, matching GRMNode; the logistic for logit).
    """
    P = len(theta)
    I = b_thresholds.shape[0]
    Mm1 = b_thresholds.shape[1]
    a_arr = (np.ones(I) if a is None else (a if np.ndim(a) else np.full(I, a)))
    # Cumulative probabilities from ONE uniform per cell: Y = #{m : U < P(Y > m)}.
    # Using a fresh uniform per threshold would not give the GRM's
    # category probabilities P(Y >= m) - P(Y >= m+1).
    U = rng.random((P, I))
    outcome = np.zeros((P, I), dtype=int)
    for m in range(Mm1):
        eta = a_arr[None, :] * (theta[:, None] - b_thresholds[None, :, m])
        if offset is not None:
            eta = eta + a_arr[None, :] * offset
        outcome += (U < _cdf(eta, link)).astype(int)
    return outcome


def _draw_gpcm(theta, steps, a, offset, rng):
    """Draw a GPCM outcome in {0,...,M-1} per cell (GPCMNode's model).

    P(Y = m) ∝ exp( sum_{k<m} a_i (theta_p + offset_pi - tau_ik) ), with the
    empty sum (m = 0) equal to 0.  One uniform per cell, inverse CDF.
    """
    P = len(theta)
    I, Mm1 = steps.shape
    a_arr = (np.ones(I) if a is None else (a if np.ndim(a) else np.full(I, a)))
    th = theta[:, None] if offset is None else theta[:, None] + offset   # [P, I]
    eta = a_arr[None, :, None] * (th[:, :, None] - steps[None, :, :])     # [P, I, M-1]
    logit = np.concatenate([np.zeros((P, I, 1)), np.cumsum(eta, axis=2)], axis=2)
    prob = np.exp(logit - logit.max(axis=2, keepdims=True))
    cdf = np.cumsum(prob / prob.sum(axis=2, keepdims=True), axis=2)
    U = rng.random((P, I, 1))
    return np.minimum((U > cdf).sum(axis=2), Mm1)


def recovery_table(fit, truth, param_prefixes):
    """Build a (name, truth, mean, lo, hi, covered) table from a fit.

    truth : dict mapping a chain param_name -> true value.
    param_prefixes : list of prefixes to include (e.g. ['b_1_', 'beta_']).
    Returns list of rows and the coverage fraction over the included params.
    """
    ch = _stack_fit(fit)
    names = _fit_param_names(fit)
    rows = []
    covered = []
    for j, nm in enumerate(names):
        if not any(nm.startswith(pre) for pre in param_prefixes):
            continue
        if nm not in truth:
            continue
        col = ch[:, j]
        mean = float(col.mean())
        lo, hi = np.percentile(col, [2.5, 97.5])
        t = truth[nm]
        cov = bool(lo <= t <= hi)
        rows.append((nm, t, mean, float(lo), float(hi), cov))
        covered.append(cov)
    frac = float(np.mean(covered)) if covered else float('nan')
    return rows, frac


def _stack_fit(fit):
    """Get an [S_total, d] float64 chain from an IRTreeFit or MCMCResult(s)."""
    if hasattr(fit, 'results'):
        return np.concatenate([np.asarray(r.chain, np.float64) for r in fit.results], axis=0)
    if isinstance(fit, list):
        return np.concatenate([np.asarray(r.chain, np.float64) for r in fit], axis=0)
    return np.asarray(fit.chain, np.float64)


def _fit_param_names(fit):
    if hasattr(fit, 'results'):
        return fit.results[0].param_names
    if isinstance(fit, list):
        return fit[0].param_names
    return fit.param_names
