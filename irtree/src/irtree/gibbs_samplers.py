"""
irtree.gibbs_samplers
=====================
Layer 3.5 — All exact Gibbs draws for IRTree model parameters.

ORGANIZATION
============
Functions are grouped by the augmentation scheme they belong to, then by the
parameter they draw:

    NODE-LEVEL: binary probit      sample_Z, gibbs_a, gibbs_b
    CROSS-NODE: item parameters    gibbs_b_correlated_from_stats
                                   (gibbs_a_correlated, gibbs_b_correlated_rng:
                                    kept for compatibility, not used by run_chain)
    NODE-LEVEL: polytomous GRM     sample_Z_grm, gibbs_thresholds_grm, truncnorm_draw
    NODE-LEVEL: 3PL Beguin-Glas    sample_U_3pl, gibbs_c_3pl
    HYPERPRIOR VARIANCES           gibbs_variance_invgamma (+ 2 aliases)
    CROSS-NODE: correlated traits  gibbs_theta_correlated, gibbs_theta_given
    COVARIATE COEFFICIENTS         coef_gaussian_terms, gibbs_coef_pooled
    MISSING COVARIATE DATA         missing_unit_stats, missing_terms,
                                   gibbs_X / Q / W_missing
    COVARIANCE MATRICES            gibbs_Sigma_invwishart

WHY GIBBS SAMPLERS ARE IN A SEPARATE FILE FROM MH SAMPLERS
===========================================================
Gibbs draws produce exact samples from a closed-form full conditional. There
is no accept/reject step, no step size to tune, and no autocorrelation from
the sampling mechanism itself. MH steps propose and decide — they carry step
sizes, acceptance ratios, and adaptation machinery. These are different
enough in structure and debugging behaviour that mixing them in one file
obscures which category a function belongs to when reading the sampler loop.
All MH steps live inline in mcmc_sampler.py.

FUNCTION INDEX
==============
Binary probit augmentation (Albert & Chib 1993; 2PL extension):
    sample_Z                — latent Z draw; Rasch mean (theta-b) or 2PL a(theta-b) always at node level.
    gibbs_a                 — discriminations, per node (truncated Normal prior)
    gibbs_b                 — item difficulties, per node

Polytomous GRM (Johnson & Albert 1999):
    sample_Z_grm            — latent draw for GRM nodes - always at node level
    gibbs_thresholds_grm    — exact draw of ordered thresholds given Z and Y*:
                              the prior truncated to the interval the
                              category indicators allow (run_chain precedes
                              it with a Cowles step, Z integrated out)
    truncnorm_draw          — tail-stable truncated Normal (log-space inverse CDF)

3PL Beguin-Glas (Beguin & Glas 2001):
    sample_U_3pl            — guessing-mode indicator draw
    gibbs_c_3pl             — guessing parameters given U (Beta conjugate)

Hyperprior variances (InvGamma conjugate — see section banner):
    gibbs_variance_invgamma — the single generic draw
    gibbs_sigma2            — alias: person-trait variance sigma^2_r
    gibbs_b_variance        — alias: coefficient-block variance tau^2
                              (called for b, beta, kappa, and gamma)

Cross-node item parameters (used when a_corr / b_corr groups are declared):
    gibbs_b_correlated_from_stats — joint MVN draw of b_i across the groups of
                              a b-corr group from pooled latent statistics
    gibbs_b_correlated_rng  — older form (first node of each group only);
                              computes its statistics, then calls the above
    gibbs_a_correlated      — older joint rejection draw for a; run_chain now
                              uses componentwise exact Gibbs (truncnorm_draw)


Cross-node trait sampling:
    compute_suff_stats         — older per-person S_p, n_p over every relevant
                                 cell (not used by run_chain: it cannot mask
                                 3PL guess cells or mixed GRM binary items)
    reduce_suff_stats_to_dims  — collapse node-indexed stats onto latent dims
    build_sigma                — assemble Sigma = D R D from sigmas and corr
    gibbs_theta_correlated     — MVN Gibbs draw for theta_p across dimensions
    gibbs_theta_given          — the same, for some dimensions given the rest

Covariate coefficients:
    coef_gaussian_terms — Gaussian sufficient statistics (G, h) of a shared
                         coefficient from the probit nodes' latent residuals
    gibbs_coef_pooled  — beta, kappa or gamma shared by a group of nodes,
                         from each node's latent residual, latent mask and
                         own discriminations

Missing covariate data (values drawn as parameters each iteration, from the
same latent residuals as gibbs_coef_pooled):
    missing_unit_stats / missing_terms / draw_from_terms
                     — shared per-unit Gaussian algebra (also the Gaussian
                       part of the MH imputation at logit nodes)
    gibbs_X_missing  — missing person-covariate features
    gibbs_Q_missing  — missing item-covariate features
    gibbs_W_missing  — missing person-by-item covariate features

Covariance matrices:
    gibbs_Sigma_invwishart  — InvWishart draw for a full covariance matrix

HOW THETA IS DRAWN
==================
There is no per-node theta sampler. Every Gibbs theta draw — correlated or
not — goes through the cross-node path.  run_chain builds each node's
per-person statistics from its latent view (missing_unit_stats on
E = Z - a(core - theta + offset), on the cells where Z was drawn), then

    reduce_suff_stats_to_dims -> build_sigma -> gibbs_theta_correlated
                                               (or gibbs_theta_given)

Dimensions with a logit/GPCM node are drawn by person-wise MH in run_chain.

The independent case is not a separate code path; it is the case where Sigma
is diagonal, which gibbs_theta_correlated detects and handles with an
elementwise fast path equal to independent per-dimension scalar draws.
One path for both cases means a parameter shared across nodes is drawn once,
from all its nodes' cells, rather than once per node with each draw seeing
only that node's cells and the last overwriting the rest.

2PL SUFFICIENT STATISTICS — DERIVATION
=======================================
For a 2PL node, Z_pi | theta_p, a_i, b_i ~ N(a_i(theta_p - b_i), 1).

**theta_p statistics** (built in run_chain from the latent view, consumed
by gibbs_theta_correlated). Expand (Z_pi - a_i(theta_p - b_i))^2 in theta_p.
Terms quadratic in theta_p contribute a_i^2 to the likelihood precision;
terms linear in theta_p contribute a_i(Z_pi + a_i b_i):

    n_p = sum_i a_i^2                      (Lambda_p diagonal entry)
    S_p = sum_i a_i (Z_pi + a_i b_i)       (sufficient statistic vector)

which gibbs_theta_correlated turns into

    Sigma_p* = (Sigma^-1 + Lambda_p)^-1
    mu_p*    = Sigma_p* S_p

Setting a_i = 1 everywhere recovers the Rasch statistics exactly, and a
diagonal Sigma factorises the matrix inverse into R scalar inversions.

**b_i full conditional** (gibbs_b). Let w_pi = Z_pi - a_i theta_p. Then
w_pi = -a_i b_i + eps, so b_i acts as a regression coefficient for -a_i:

    sigma_i^2* = 1 / (1/tau^2 + n_p a_i^2)
    mu_i*      = sigma_i^2* x a_i x sum_p (a_i theta_p - Z_pi)

Note the sign: the statistic is sum_p (theta_p - Z_pi), NOT sum_p (Z_pi -
theta_p). Setting a_i = 1 recovers the Rasch formula with n_p x 1 = n_persons.

CORRELATED TRAITS — DERIVATION
===============================
When traits are correlated the joint prior is theta_p ~ MVN(0, Sigma) with
Sigma = D R D (separation strategy: D = diag(sigma_1,...,sigma_R), R = the
correlation matrix). The MVN full conditional given Z at all nodes is

    Sigma_p* = (Sigma^-1 + Lambda_p)^-1
    mu_p*    = Sigma_p* S_p

with Lambda_p = diag(n_p1, ..., n_pR) and S_p the per-node sufficient
statistics (sum_i (Z_pir + b_ir) for binary/2PL nodes; sum_i Z_pir for GRM).

VECTORISATION
=============
All samplers use the precomputed index sets (row_idx / col_idx from
data_prep.PreparedData) to enumerate relevant (person, item) pairs, and
accumulate sufficient statistics with np.add.at / np.bincount rather than a
Python loop over persons or items. Per-unit MVN draws are batched over the
leading axis, so P person-solves or I item-solves run as two LAPACK calls
rather than P or I of them. Truncated-normal draws use inverse-CDF sampling
via scipy.special.ndtr / ndtri.

REFERENCES
==========
Albert, J.H. & Chib, S. (1993). JASA, 88(422), 669-679.
Beguin, A.A. & Glas, C.A.W. (2001). Psychometrika, 66(4), 541-561.
Johnson, V.E. & Albert, J.H. (1999). Ordinal Data Modeling. Springer, Ch. 4.
Lewandowski, D., Kurowicka, D. & Joe, H. (2009). J. Multivariate Analysis.
Patz, R.J. & Junker, B.W. (1999a). JEBS, 24(2), 146-178.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as _stats
from scipy.special import log_ndtr as _log_Phi
from scipy.special import ndtr as _Phi
from scipy.special import ndtri as _Phi_inv
from scipy.special import ndtri_exp as _Phi_inv_exp

_MAX_TRUNC_MVN_SWEEPS = 1000


# ===========================================================================
# NODE-LEVEL: Binary probit augmentation (Albert & Chib 1993; 2PL extension)
# ===========================================================================

def sample_Z(
    Y_star:  np.ndarray,
    theta:   np.ndarray,
    b:       np.ndarray,
    T:       np.ndarray,
    row_idx: np.ndarray,
    col_idx: np.ndarray,
    rng:     np.random.Generator,
    a:       np.ndarray | None = None,
    offset:  np.ndarray | None = None,
) -> np.ndarray:
    """Draw latent augmentation variables Z for binary probit nodes.

    For each relevant pair (p, i) with T[p, i] = True:

        Rasch/1PL (a=None):
            Z_pi | Y*=1  ~  TruncNormal(θ_p − b_i,        1, lower=0)
            Z_pi | Y*=0  ~  TruncNormal(θ_p − b_i,        1, upper=0)

        2PL (a provided):
            Z_pi | Y*=1  ~  TruncNormal(a_i(θ_p − b_i),  1, lower=0)
            Z_pi | Y*=0  ~  TruncNormal(a_i(θ_p − b_i),  1, upper=0)

    Uses inverse-CDF sampling, fully vectorised.

    Parameters
    ----------
    Y_star   : ndarray [P × I], binary pseudo-responses (0 or 1)
    theta    : ndarray [P],     person trait parameters
    b        : ndarray [I],     item difficulty parameters
    T        : ndarray [P × I], boolean relevance mask
    row_idx  : ndarray [N_rel], precomputed np.where(T)[0]
    col_idx  : ndarray [N_rel], precomputed np.where(T)[1]
    rng      : numpy Generator
    a        : ndarray [I] or None — discriminations (2PL / 3PL); at a 3PL
               node the caller passes the U = 0 cells in row_idx / col_idx
    offset   : ndarray [P × I] or None — raw covariate linear predictor
               (Xβ + Qκ + Wγ), added to θ_p − b_i before the a_i scaling

    Returns
    -------
    Z : ndarray [P × I]  — valid at T[p,i]=True; zero elsewhere.
    """
    P, I = Y_star.shape
    Z = np.zeros((P, I), dtype=np.float64)

    if len(row_idx) == 0:
        return Z

    # Linear predictor: Rasch = θ_p − b_i; 2PL = a_i(θ_p − b_i)
    # With covariates: offset[p,i] is added before truncation.
    eta = theta[row_idx] - b[col_idx]
    if offset is not None:
        eta = eta + offset[row_idx, col_idx]
    if a is not None:
        eta = a[                    col_idx] * eta

    y_pos = Y_star[row_idx, col_idx].astype(bool)

    # Phi(-eta) = CDF boundary at truncation point 0
    phi_neg_eta = _Phi(-eta)

    # Inverse-CDF sampling:
    #   Y*=1 → Z > 0 → U ~ Uniform(Phi(-eta), 1)
    #   Y*=0 → Z < 0 → U ~ Uniform(0, Phi(-eta))
    lo = np.where(y_pos, phi_neg_eta, 0.0)
    hi = np.where(y_pos, 1.0,         phi_neg_eta)

    U = rng.uniform(lo, hi)
    U = np.clip(U, 1e-14, 1.0 - 1e-14)

    Z[row_idx, col_idx] = eta + _Phi_inv(U)
    return Z

# ===========================================================================
# NODE-LEVEL: Polytomous GRM augmentation (Johnson & Albert 1999)
# ===========================================================================

def sample_Z_grm(
    Y_star:     np.ndarray,
    theta:      np.ndarray,
    thresholds: np.ndarray,
    T:          np.ndarray,
    row_idx:    np.ndarray,
    col_idx:    np.ndarray,
    n_outcomes: int,
    rng:        np.random.Generator,
    offset:     np.ndarray | None = None,
) -> np.ndarray:
    """Draw latent augmentation variables Z for polytomous probit GRM nodes.

    For M ordered categories with item-specific thresholds
    τ_{i,1} < τ_{i,2} < ... < τ_{i,M-1}, the latent variable model is:

        Z_pi ~ N(θ_p, 1)
        Y*_pi = m   iff   τ_{i,m} < Z_pi ≤ τ_{i,m+1}

    where τ_{i,0} = -∞ and τ_{i,M} = +∞.

    The conditional draw is:

        Z_pi | Y*=m, θ  ~  TruncNormal(θ_p, 1, lower=τ_{i,m}, upper=τ_{i,m+1})

    Key difference from binary sample_Z: mean is θ_p (not θ_p − b_i),
    because in the GRM item information lives in the thresholds.

    Parameters
    ----------
    Y_star     : ndarray [P × I], pseudo-responses in {0, ..., M-1}
    theta      : ndarray [P],     person trait parameters
    thresholds : ndarray [I × (M-1)], ordered thresholds per item
    T          : ndarray [P × I], boolean relevance mask
    row_idx    : ndarray [N_rel], precomputed np.where(T)[0]
    col_idx    : ndarray [N_rel], precomputed np.where(T)[1]
    n_outcomes : int, number of categories M
    rng        : numpy Generator
    offset     : ndarray [P × I] or None — raw covariate linear predictor,
                 added to the mean: Z_pi ~ N(θ_p + offset_pi, 1)

    Returns
    -------
    Z : ndarray [P × I]  — valid at T[p,i]=True; zero elsewhere.
    """
    P, I = Y_star.shape
    M = n_outcomes
    Z = np.zeros((P, I), dtype=np.float64)

    if len(row_idx) == 0:
        return Z

    m_vals = Y_star[row_idx, col_idx].astype(int)
    i_vals = col_idx

    lower = np.full(len(row_idx), -np.inf)
    upper = np.full(len(row_idx),  np.inf)

    mask_has_lower = m_vals > 0
    if np.any(mask_has_lower):
        lower[mask_has_lower] = thresholds[
            i_vals[mask_has_lower], m_vals[mask_has_lower] - 1
        ]

    mask_has_upper = m_vals < M - 1
    if np.any(mask_has_upper):
        upper[mask_has_upper] = thresholds[
            i_vals[mask_has_upper], m_vals[mask_has_upper]
        ]

    # GRM mean = θ_p + offset[p,i] (covariate adjustment, if any)
    mu = theta[row_idx]
    if offset is not None:
        mu = mu + offset[row_idx, col_idx]

    phi_lo = np.where(np.isfinite(lower), _Phi(lower - mu), 0.0)
    phi_hi = np.where(np.isfinite(upper), _Phi(upper - mu), 1.0)

    U = rng.uniform(phi_lo, phi_hi)
    U = np.clip(U, 1e-14, 1.0 - 1e-14)

    Z[row_idx, col_idx] = mu + _Phi_inv(U)
    return Z


# ===========================================================================
# NODE-LEVEL: 3PL Béguin-Glas augmentation (Béguin & Glas 2001)
# ===========================================================================

def sample_U_3pl(
    Y_star:  np.ndarray,
    theta:   np.ndarray,
    b:       np.ndarray,
    a:       np.ndarray,
    c:       np.ndarray,
    T:       np.ndarray,
    row_idx: np.ndarray,
    col_idx: np.ndarray,
    rng:     np.random.Generator,
    offset:  np.ndarray | None = None,
) -> np.ndarray:
    """Draw Béguin-Glas guessing-mode indicators U for 3PL nodes.

    U_pi ∈ {0, 1} indicates whether person p's response at item i was
    generated by the guessing process (U=1) or the ability process (U=0).

    Full conditional:
        If Y*_pi = 0: U_pi = 0 deterministically.
        If Y*_pi = 1: U_pi ~ Bernoulli(p*_pi) where
            p*_pi = c_i / [c_i + (1 − c_i) Φ(a_i(θ_p − b_i + offset_pi))]

    Parameters
    ----------
    Y_star   : ndarray [P × I], binary pseudo-responses (0 or 1)
    theta    : ndarray [P],     person trait parameters
    b        : ndarray [I],     item difficulty parameters
    a        : ndarray [I],     item discrimination parameters (> 0)
    c        : ndarray [I],     item guessing parameters ∈ (0, 1)
    T        : ndarray [P × I], boolean relevance mask
    row_idx  : ndarray [N_rel], precomputed np.where(T)[0]
    col_idx  : ndarray [N_rel], precomputed np.where(T)[1]
    rng      : numpy Generator
    offset   : ndarray [P × I] or None — covariate linear predictor, added to
               θ_p − b_i before the a-scaling (as in sample_Z).  Without it
               the indicators would be drawn under a model without covariates.

    Returns
    -------
    U : ndarray [P × I], dtype int8 — 0 or 1 at relevant cells; 0 elsewhere.
    """
    P, I = Y_star.shape
    U = np.zeros((P, I), dtype=np.int8)

    if len(row_idx) == 0:
        return U

    y_pos    = Y_star[row_idx, col_idx].astype(bool)
    pos_rows = row_idx[y_pos]
    pos_cols = col_idx[y_pos]

    if len(pos_rows) == 0:
        return U

    core    = theta[pos_rows] - b[pos_cols]
    if offset is not None:
        core = core + offset[pos_rows, pos_cols]
    eta     = a[pos_cols] * core
    phi_eta = _Phi(eta)
    c_i     = c[pos_cols]

    p_star = c_i / (c_i + (1.0 - c_i) * phi_eta)
    p_star = np.clip(p_star, 0.0, 1.0)

    U[pos_rows, pos_cols] = (rng.uniform(size=len(pos_rows)) < p_star).astype(np.int8)
    return U


# ---------------------------------------------------------------------------
# SINGLE-NODE DISCRIMINATION SAMPLER (TruncNormal prior path)
# ---------------------------------------------------------------------------

def gibbs_a(
    Z:           np.ndarray,
    theta:       np.ndarray,
    b:           np.ndarray,
    row_idx:     np.ndarray,
    col_idx:     np.ndarray,
    mean:        float,
    variance:    float,
    lower:       float | None,
    rng:         np.random.Generator,
    n_items:     int | None = None,
    offset:      np.ndarray | None = None,
    common:      bool = False,
) -> np.ndarray:
    """Exact Gibbs draw for per-item discrimination a_i at one node.

    With ``common=True`` the node has ONE slope a shared by all its items.
    It appears in every cell, so its full conditional pools every cell of
    the node: the same truncated Normal with the sums taken over all items,

        σ*² = 1 / (1/variance + Σ_p Σ_i d_pi²),
        μ*  = σ*² · (mean/variance + Σ_p Σ_i d_pi z_pi),

    and the single draw is returned repeated over the I items.

    Companion to :func:`gibbs_a_correlated` for the uncorrelated (single-node)
    case. Requires a bounded NormalPrior or TruncNormalPrior on 'a' — any
    prior for which the augmented likelihood stays conjugate. The Albert–Chib
    augmentation gives, at a 2PL node,

        z_{p,i} = a_i · d_{p,i} + ε,   d_{p,i} = θ_p − b_i,   ε ~ N(0, 1)

    which is *linear in a_i*, so each a_i has a Normal full conditional,
    truncated to [lower, ∞) by the prior's support:

        σ*²_i = 1 / (1/variance + Σ_p d_{p,i}²)
        μ*_i  = σ*²_i · (mean/variance + Σ_p d_{p,i} z_{p,i})
        a_i   ~  TruncNormal(μ*_i, σ*²_i, lower)

    The draw is fully vectorized across items via inverse-CDF sampling: all I
    items are drawn in one call, no per-item loop, no accept/reject.

    Only the cells enumerated by (row_idx, col_idx) contribute: at a 2PL
    node every relevant cell, at a 3PL node only the cells with U = 0 (where
    Z was drawn), so the caller passes the node's latent-mask index arrays. The sufficient statistics Σ d² and Σ d·z
    are accumulated over exactly those cells.

    Parameters
    ----------
    Z : ndarray [P, I]
        Current augmentation for this node.
    theta : ndarray [P]
        Person traits for this node's latent dimension.
    b : ndarray [I]
        Item difficulties for this node's b-group.
    row_idx, col_idx : ndarray
        (person, item) cells of the node's latent mask (all relevant cells,
        or the U = 0 cells at a 3PL node).
    mean, variance : float
        Prior mean μ₀ and variance τ² of the (truncated) Normal prior on a.
    lower : float or None
        Lower truncation bound (typically 0 for discrimination). None means
        unbounded below, i.e. an ordinary Normal full conditional.
    rng : np.random.Generator
    n_items : int, optional
        I. Inferred from ``Z.shape[1]`` when omitted.
    offset : ndarray [P, I] or None
        Raw covariate linear predictor; part of the regressor
        d_pi = θ_p − b_i + offset_pi.
    common : bool
        One slope shared by all items (see above).

    Returns
    -------
    a : ndarray [I]
        Updated discrimination parameters, one per item.
    """
    I = Z.shape[1] if n_items is None else n_items

    # a_i is the slope regressing Z on the ability predictor d_pi.  With
    # covariates the predictor is (theta_p - b_i + offset_pi), not just
    # (theta_p - b_i): the covariate effect is inside the a-scaled term
    # Z ~ N(a_i (theta - b + offset), 1), so it must be part of the regressor,
    # otherwise a absorbs the covariate signal.  offset=None -> plain 2PL.
    d_pi = theta[row_idx] - b[col_idx]
    if offset is not None:
        d_pi = d_pi + offset[row_idx, col_idx]
    sum_d2 = np.zeros(I, dtype=np.float64)
    sum_dz = np.zeros(I, dtype=np.float64)
    np.add.at(sum_d2, col_idx, d_pi ** 2)
    np.add.at(sum_dz, col_idx, d_pi * Z[row_idx, col_idx])
    if common:
        sum_d2 = np.array([sum_d2.sum()])
        sum_dz = np.array([sum_dz.sum()])

    sig2_star = 1.0 / (1.0 / variance + sum_d2)
    mu_star   = sig2_star * (mean / variance + sum_dz)
    sd_star   = np.sqrt(sig2_star)

    lo_std = (-np.inf if lower is None else lower - mu_star) / sd_star
    U = _Phi(lo_std) + (1.0 - _Phi(lo_std)) * rng.uniform(size=len(mu_star))
    U = np.clip(U, 1e-14, 1.0 - 1e-14)
    draw = mu_star + sd_star * _Phi_inv(U)
    return np.full(I, draw[0]) if common else draw

# ---------------------------------------------------------------------------
# SINGLE-NODE DIFFICULTY SAMPLER
# ---------------------------------------------------------------------------

def gibbs_b(
    Z:           np.ndarray,
    theta:       np.ndarray,
    tau2:        float,
    T:           np.ndarray,
    row_idx:     np.ndarray,
    col_idx:     np.ndarray,
    rng:         np.random.Generator,
    a:           np.ndarray | None = None,
    prior_lower: float | None = None,
    prior_upper: float | None = None,
    prior_mean:  float = 0.0,
) -> np.ndarray:
    """Gibbs draw for item difficulties b at a binary probit node.

    Full conditional:

    Rasch/1PL (a=None):
        Likelihood: Z_pi ~ N(θ_p − b_i, 1)
        σ²_i* = 1 / (1/τ² + n_persons_i)
        μ_i*  = σ²_i* × (μ_b/τ² + Σ_p (θ_p − Z_pi))

    2PL (a provided):
        Likelihood: Z_pi ~ N(a_i(θ_p − b_i), 1)
        Let w_pi = Z_pi − a_i·θ_p.  Then w_pi = −a_i·b_i + ε, so b_i
        is a regression coefficient for −a_i.  Per-item precision is
        n_p·a_i² (not n_p), where n_p = number of relevant persons.
        σ²_i* = 1 / (1/τ² + n_persons_i·a_i²)
        μ_i*  = σ²_i* × (μ_b/τ² + a_i × Σ_p (a_i·θ_p − Z_pi))

    under the prior b_i ~ N(μ_b, τ²) (``prior_mean`` = μ_b, default 0).

    When prior_lower or prior_upper is not None, the Normal draw is
    replaced by a truncated Normal draw using the inverse-CDF method.

    Parameters
    ----------
    Z       : ndarray [P × I]
    theta   : ndarray [P],     current person parameters
    tau2    : float > 0,       prior variance for b
    T, row_idx, col_idx : relevance mask and precomputed indices
    rng     : numpy Generator
    a       : ndarray [I] or None — discrimination parameters (2PL only)
    prior_lower : float or None — lower bound on b (truncated Normal)
    prior_upper : float or None — upper bound on b (truncated Normal)
    prior_mean  : float — prior mean μ_b of b (default 0)

    Returns
    -------
    b : ndarray [I]
    """
    I = Z.shape[1]

    if a is None:
        # --- Rasch/1PL ---
        n_persons = np.zeros(I)
        sum_diff  = np.zeros(I)
        np.add.at(n_persons, col_idx, 1.0)
        np.add.at(sum_diff,  col_idx, theta[row_idx] - Z[row_idx, col_idx])

        sigma2_star = 1.0 / (1.0 / tau2 + n_persons)
        mu_star     = sigma2_star * (prior_mean / tau2 + sum_diff)

    else:
        # --- 2PL ---
        a_i  = a[col_idx]
        a2_i = a_i ** 2
        diff = a_i * theta[row_idx] - Z[row_idx, col_idx]

        n_persons = np.zeros(I)
        sum_a2    = np.zeros(I)
        sum_d     = np.zeros(I)
        np.add.at(n_persons, col_idx, 1.0)
        np.add.at(sum_a2,    col_idx, a2_i)
        np.add.at(sum_d,     col_idx, diff)

        sigma2_star = 1.0 / (1.0 / tau2 + sum_a2)
        mu_star     = sigma2_star * (prior_mean / tau2 + a[np.arange(I)] * sum_d)

    if prior_lower is None and prior_upper is None:
        return rng.normal(mu_star, np.sqrt(sigma2_star))

    sd_star = np.sqrt(sigma2_star)
    lo_std  = (-np.inf if prior_lower is None else (prior_lower - mu_star) / sd_star)
    hi_std  = ( np.inf if prior_upper is None else (prior_upper - mu_star) / sd_star)
    p_lo = _Phi(lo_std)
    p_hi = _Phi(hi_std)
    U = p_lo + (p_hi - p_lo) * rng.uniform(size=I)
    U = np.clip(U, 1e-14, 1.0 - 1e-14)
    return mu_star + sd_star * _Phi_inv(U)

# ---------------------------------------------------------------------------
# SINGLE-NODE THRESHOLDS (grm) SAMPLER
# ---------------------------------------------------------------------------

def truncnorm_draw(mu, sd, lo, hi, rng):
    """Vectorized draw from N(mu, sd^2) truncated to (lo, hi).

    Inverse CDF computed in log space (log_ndtr / ndtri_exp) on the side of
    the mean the interval lies on, so intervals deep in either tail (where
    Phi(hi) - Phi(lo) underflows in linear space) are still drawn exactly.
    Infinite bounds are allowed.
    """
    mu, sd, lo, hi = np.broadcast_arrays(*(np.asarray(v, float) for v in (mu, sd, lo, hi)))
    a = (lo - mu) / sd
    b = (hi - mu) / sd
    flip = a > 0                     # upper tail: draw the mirror image
    a2 = np.where(flip, -b, a)
    b2 = np.where(flip, -a, b)
    la, lb = _log_Phi(a2), _log_Phi(b2)          # la <= lb <= log(1/2)
    v = rng.uniform(size=mu.shape)
    # log( Phi(a) + v (Phi(b) - Phi(a)) ), stably
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.exp(la - lb)
        logu = lb + np.log(r + v * (1.0 - r))
    x = _Phi_inv_exp(logu)
    x = np.clip(x, a2, b2)
    x = np.where(flip, -x, x)
    return mu + sd * x


def gibbs_thresholds_grm(
    Z:              np.ndarray,
    Y_star:         np.ndarray,
    T:              np.ndarray,
    row_idx:        np.ndarray,
    col_idx:        np.ndarray,
    n_outcomes:     int,
    rng:            np.random.Generator,
    old_thresholds: np.ndarray | None = None,
    mu_0:           float = 0.0,
    tau2:           float = 4.0,
) -> np.ndarray:
    """Exact Gibbs draw of GRM thresholds (Albert & Chib 1993).

    Model at a GRM node, for each relevant (p, i):

        Z_pi ~ N(theta_p + offset_pi, 1),
        Y*_pi = m  iff  tau_{i,m-1} < Z_pi <= tau_{i,m}
                        (tau_{i,-1} = -inf, tau_{i,M-1} = +inf; column k of
                         the threshold matrix is tau_{i,k}),
        tau_{i,k} ~ N(mu_0, tau2).

    The density of Z does not involve tau; tau enters only through the
    category indicators.  Hence, given Z, Y* and the item's other
    thresholds, the full conditional of tau_{i,k} is the PRIOR truncated to
    the interval every indicator allows:

        L = max( tau_{i,k-1},  max{Z_pi : Y*_pi = k} )
        U = min( tau_{i,k+1},  min{Z_pi : Y*_pi = k+1} )
        tau_{i,k} | rest ~ N(mu_0, tau2) truncated to (L, U).

    Thresholds are updated in order k = 0..K-1 (Gibbs within the item),
    vectorized over items: tau_{i,k-1} is the value just drawn and
    tau_{i,k+1} the current one.  An empty side contributes no bound, so an
    item or category with no data gets its prior truncated by its
    neighbours.

    Given Z, the latent responses enter tau's conditional only through the
    bounds; they are not noisy measurements of tau, so no Normal likelihood
    term built from them belongs in the draw.

    Parameters
    ----------
    Z, Y_star, T, row_idx, col_idx, n_outcomes, rng : as for sample_Z_grm
    old_thresholds : ndarray [I x (M-1)], current (ordered) thresholds.
                     None starts from evenly spaced values in (-1, 1).
    mu_0, tau2     : prior mean and (current) prior variance.

    Returns
    -------
    thresholds : ndarray [I x (M-1)]
    """
    M = n_outcomes
    I = Z.shape[1]
    K = M - 1
    if old_thresholds is None:
        tau = np.tile(np.linspace(-1.0, 1.0, K), (I, 1))
    else:
        tau = np.array(old_thresholds, dtype=np.float64, copy=True).reshape(I, K)

    z_rel = Z[row_idx, col_idx]
    y_rel = np.clip(Y_star[row_idx, col_idx].astype(int), 0, M - 1)
    flat = col_idx * M + y_rel
    max_im = np.full(I * M, -np.inf)
    min_im = np.full(I * M, np.inf)
    np.maximum.at(max_im, flat, z_rel)
    np.minimum.at(min_im, flat, z_rel)
    max_im = max_im.reshape(I, M)       # max Z in category m of item i
    min_im = min_im.reshape(I, M)       # min Z in category m of item i

    sd = np.sqrt(tau2)
    for k in range(K):
        L = max_im[:, k].copy()
        if k > 0:
            L = np.maximum(L, tau[:, k - 1])
        U = min_im[:, k + 1].copy()
        if k + 1 < K:
            U = np.minimum(U, tau[:, k + 1])
        ok = L < U
        draw = truncnorm_draw(np.full(I, mu_0), np.full(I, sd), L, U, rng)
        # L >= U cannot occur for a consistent state; keep the value if it does.
        tau[:, k] = np.where(ok, draw, tau[:, k])
    return tau


# ---------------------------------------------------------------------------
# SINGLE-NODE DIFFICULTY SAMPLER
# ---------------------------------------------------------------------------

def gibbs_c_3pl(
    U:           np.ndarray,
    Y_star:      np.ndarray,
    row_idx:     np.ndarray,
    col_idx:     np.ndarray,
    alpha_prior: float,
    beta_prior:  float,
    n_items:     int,
    rng:         np.random.Generator,
) -> np.ndarray:
    """Gibbs draw for guessing parameters c at a 3PL node (Béguin-Glas).

    Full conditional (Beta-Bernoulli conjugacy on U):

        c_i | U  ~  Beta(α_c + n_guess_i,  β_c + n_ir_i − n_guess_i)

    where
        n_guess_i = #{p : U_{pi} = 1}        (guessing-mode trials)
        n_ir_i    = #{p : T_r[p,i] = 1}      (all relevant trials)
        n_ir_i − n_guess_i = #{p : U_{pi} = 0}  (ability-mode trials)

    All relevant cells (T_r=1) contribute Bernoulli(c_i) trials, whether
    Y* = 0 or Y* = 1.  For Y*=0 cells, U = 0 deterministically; they still
    count as observed non-guessing trials that inform β.

    One Beta draw per item; no accept-reject.

    Parameters
    ----------
    U            : ndarray [P × I], guessing-mode indicators from sample_U_3pl
    Y_star       : ndarray [P × I], not used directly (U already encodes this)
    row_idx, col_idx : relevance indices (np.where(T_r))
    alpha_prior  : float, Beta prior α_c
    beta_prior   : float, Beta prior β_c
    n_items      : int, number of items I
    rng          : numpy Generator

    Returns
    -------
    c : ndarray [I]
    """
    # Count guessing trials (U=1) per item
    n_guess = np.zeros(n_items, dtype=np.float64)
    np.add.at(n_guess, col_idx, U[row_idx, col_idx].astype(np.float64))

    # Count total relevant trials per item
    n_relevant = np.zeros(n_items, dtype=np.float64)
    np.add.at(n_relevant, col_idx, 1.0)

    alpha_post = alpha_prior + n_guess
    beta_post  = beta_prior  + (n_relevant - n_guess)   # ability-mode + incorrect trials

    return rng.beta(alpha_post, beta_post)


# ===========================================================================
# HYPERPRIOR VARIANCES (conjugate InvGamma draws)
# ===========================================================================
# Every estimated variance hyperparameter in the package is the variance of a
# zero-centred Normal prior on a coefficient vector, so they all share ONE
# full conditional:
#
#     sigma^2 | x  ~  InvGamma(alpha + n/2,  beta + 0.5 * sum_i (x_i - mu)^2)
#
# This covers theta (person traits), b (item difficulties), and the covariate
# coefficient blocks beta / kappa / gamma.  gibbs_sigma2 and gibbs_b_variance
# are retained as named aliases because the sampler and its tests call them by
# those names, and because the alias makes the call site self-documenting.
#
# NOT covered: a hierarchical variance on a BOUNDED prior (e.g. the truncated
# Normal on discrimination a).  The truncation constant depends on sigma, so
# the InvGamma posterior does not close.  priors.py rejects that combination
# unless force=True, in which case an MH block is required instead.
# ===========================================================================

def gibbs_variance_invgamma(
    x:           np.ndarray,
    alpha_prior: float,
    beta_prior:  float,
    rng:         np.random.Generator,
    mean:        float = 0.0,
) -> float:
    """Exact Gibbs draw for the variance of a Normal coefficient vector.

    Full conditional (Normal-InvGamma conjugacy):

        sigma^2 | x  ~  InvGamma(alpha + n/2,  beta + 0.5 * sum_i (x_i - mu)^2)

    Parameters
    ----------
    x           : ndarray [n], the coefficient vector whose variance is drawn
                  (theta_r, b_g, beta_g, kappa_g, or gamma_g)
    alpha_prior : float, InvGamma shape alpha
    beta_prior  : float, InvGamma scale beta
    rng         : numpy Generator
    mean        : float, prior mean mu of the coefficients.  Defaults to 0.0,
                  which is the value used by every prior in the package that
                  carries an estimated variance.  Exposed because the
                  sufficient statistic is sum (x_i - mu)^2, not sum x_i^2, and
                  silently assuming mu = 0 would bias the draw for any
                  non-zero-centred prior.

    Returns
    -------
    variance : float > 0
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    dev = x if mean == 0.0 else (x - mean)

    alpha_post = alpha_prior + n / 2.0
    beta_post  = beta_prior  + 0.5 * float(np.sum(dev ** 2))

    return float(_stats.invgamma.rvs(
        a=alpha_post, scale=beta_post, random_state=rng,
    ))


def gibbs_sigma2(
    theta:       np.ndarray,
    alpha_prior: float,
    beta_prior:  float,
    rng:         np.random.Generator,
) -> float:
    """Gibbs draw for the person-trait variance sigma^2_r.

    Alias for ``gibbs_variance_invgamma`` with the person-trait vector.
    Works for Rasch, 2PL, and GRM nodes -- the prior structure
    theta_pr ~ N(0, sigma^2_r) is the same in all three.

    Note that sigma^2_r is not merely a by-product of the theta draw: it feeds
    back into it.  build_sigma() assembles Sigma = D R D from these variances,
    and gibbs_theta_correlated draws theta from that Sigma every iteration.
    Skipped only when theta.correlation carries an InverseWishartPrior, in
    which case Sigma is drawn whole and the variances are read off its
    diagonal.
    """
    return gibbs_variance_invgamma(theta, alpha_prior, beta_prior, rng)


def gibbs_b_variance(
    b:           np.ndarray,
    alpha_prior: float,
    beta_prior:  float,
    rng:         np.random.Generator,
) -> float:
    """Gibbs draw for a coefficient-block variance (tau^2).

    Alias for ``gibbs_variance_invgamma``.  Despite the name, this is the
    general-purpose conjugate variance draw and the sampler calls it for four
    different parameters: item difficulties b (once per b-sharing group) and
    the covariate coefficient blocks beta, kappa, and gamma (once per
    covariate group).  All four carry a zero-centred Normal prior, so the
    sufficient statistic sum_i x_i^2 is correct for each.
    """
    return gibbs_variance_invgamma(b, alpha_prior, beta_prior, rng)



# ===========================================================================
# CROSS-NODE ITEM-PARAMETER SAMPLERS
# ===========================================================================
# These functions draw item difficulties (and discriminations) jointly across
# nodes when a cross-node correlation prior is active.  The mathematics
# mirror gibbs_theta_correlated exactly, with items in the role of persons
# and b-groups in the role of latent dimensions.
#
# DERIVATION (matches §2.12 of full_conditionals_derivations.md)
# --------------------------------------------------------------
# For a b-correlation group of G b-sharing groups, define for item i:
#
#   b_i = (b_{i,g1}, ..., b_{i,gG})  ~  N_G(0, Σ_b)
#
# The likelihood at each group g_k (representative node r_k):
#   Rasch: Z_{p,i,r_k} | b_{i,g_k}, θ_{p,r_k} ~ N(θ_{p,r_k} − b_{i,g_k}, 1)
#   2PL:   Z_{p,i,r_k} | b, θ, a  ~ N(a_{i,r_k}(θ_{p,r_k} − b_{i,g_k}), 1)
#
# Sufficient statistics per group g_k:
#   S_{i,g_k} = Σ_p T_{r_k}[p,i] · (a_{i,r_k}·θ_{p,r_k} − Z_{p,i,r_k}) · a_{i,r_k}
#   n_{i,g_k} = Σ_p T_{r_k}[p,i] · a_{i,r_k}²          [Rasch: a=1]
#
# Full conditional (Normal-Normal conjugacy):
#   Λ_i   = diag(n_{i,g1}, ..., n_{i,gG})
#   Σ_i*  = (Σ_b⁻¹ + Λ_i)⁻¹
#   μ_i*  = Σ_i* × S_i
#   b_i | Z, θ, Σ_b  ~  N_G(μ_i*, Σ_i*)
# ===========================================================================

# ---------------------------------------------------------------------------
# CROSS-NODE DISCRIMINATION SAMPLER (TruncNormal prior path)
# ---------------------------------------------------------------------------

def gibbs_a_correlated(
    Z_list:        list,
    theta_list:    list,
    b_list:        list,
    T_list:        list,
    row_idx_list:  list,
    col_idx_list:  list,
    Sigma_a:       np.ndarray,
    mu_a:          np.ndarray,
    rng:           np.random.Generator,
    max_rejection: int = 200,
) -> list:
    """Gibbs draw for discrimination parameters across an a-correlation group.

    Requires a bounded NormalPrior or TruncNormalPrior on 'a' — any prior
    for which _a_prior_is_gibbs() returns True.  Both are conjugate with
    the augmented Z likelihood (linear in a), giving a truncated-MVN full
    conditional on the positive orthant.

    DERIVATION
    ----------
    Given z_{p,i,r} = a_{i,r}·d_{p,i,r} + ε with d_{p,i,r} = θ_{p,r} − b_{i,r}:

        S_{i,g_k}  = Σ_p T[p,i]·d_{p,i,r_k}·z_{p,i,r_k}
        n_{i,g_k}  = Σ_p T[p,i]·d_{p,i,r_k}²

        Λ_i   = diag(n_{i,g1}, ..., n_{i,gG})
        Σ_i*  = (Σ_a⁻¹ + Λ_i)⁻¹
        μ_i*  = Σ_i*(Σ_a⁻¹ μ_a + S_i)
        a_i   ~  TruncMVN_+(μ_i*, Σ_i*)

    Parameters
    ----------
    Z_list : list of ndarray [P, I]
    theta_list : list of ndarray [P]
    b_list : list of ndarray [I]
    T_list, row_idx_list, col_idx_list : matching index arrays.
    Sigma_a : ndarray [G, G]
        Current covariance matrix for a (raw scale).
    mu_a : ndarray [G]
        Prior mean vector (e.g., np.full(G, 1.0) for discrimination).
    rng : np.random.Generator
    max_rejection : int
        Maximum rejection draws per item before scalar fallback.

    Returns
    -------
    list of ndarray [I]
        Updated a arrays, one per group.
    """
    G = len(Z_list)
    I = Z_list[0].shape[1]

    Sigma_a_inv = np.linalg.inv(Sigma_a)
    prior_term  = Sigma_a_inv @ mu_a  # [G]

    S = np.zeros((I, G), dtype=np.float64)
    N = np.zeros((I, G), dtype=np.float64)

    for k in range(G):
        Z_k     = Z_list[k]
        theta_k = theta_list[k]
        b_k     = b_list[k]
        ri      = row_idx_list[k]
        ci      = col_idx_list[k]

        d_pi = theta_k[ri] - b_k[ci]
        np.add.at(S[:, k], ci, d_pi * Z_k[ri, ci])
        np.add.at(N[:, k], ci, d_pi ** 2)

    # Batched over items (see gibbs_theta_correlated / gibbs_b_correlated_rng).
    # The positive-orthant constraint is enforced by batched rejection: each
    # sweep redraws only the items still outside the orthant.  Items that never
    # land inside after max_rejection sweeps fall back to independent scalar
    # truncated Normals, exactly as before.
    #
    # NOTE: draws via Cholesky rather than rng.multivariate_normal's SVD, so
    # chains are statistically -- not bitwise -- equivalent to the old code.
    A = np.broadcast_to(Sigma_a_inv, (I, G, G)).copy()
    g_idx = np.arange(G)
    A[:, g_idx, g_idx] += N

    Sigma_post = np.linalg.inv(A)
    Sigma_post = 0.5 * (Sigma_post + np.swapaxes(Sigma_post, -1, -2))
    mu_post    = np.einsum('igh,ih->ig', Sigma_post, prior_term[None, :] + S)
    L          = np.linalg.cholesky(Sigma_post)

    new_a   = np.empty((I, G), dtype=np.float64)
    pending = np.arange(I)
    for _ in range(max_rejection):
        if pending.size == 0:
            break
        eps  = rng.standard_normal((pending.size, G))
        draw = mu_post[pending] + np.einsum('igh,ih->ig', L[pending], eps)
        ok   = np.all(draw > 0.0, axis=1)
        new_a[pending[ok]] = draw[ok]
        pending = pending[~ok]

    if pending.size:
        sd     = np.sqrt(np.diagonal(Sigma_post[pending], axis1=-2, axis2=-1))
        mu_f   = mu_post[pending]
        lo_std = (0.0 - mu_f) / sd
        p_lo   = _Phi(lo_std)
        u      = p_lo + (1.0 - p_lo) * rng.uniform(size=mu_f.shape)
        u      = np.clip(u, 1e-14, 1.0 - 1e-14)
        new_a[pending] = mu_f + sd * _Phi_inv(u)

    return [new_a[:, k].copy() for k in range(G)]


# ---------------------------------------------------------------------------
# CROSS-NODE DIFFICULTY SAMPLER
# ---------------------------------------------------------------------------

def gibbs_b_correlated_rng(
    Z_list:       list,
    theta_list:   list,
    b_list:       list,
    T_list:       list,
    row_idx_list: list,
    col_idx_list: list,
    a_list:       list,
    Sigma_b:      np.ndarray,
    rng:          np.random.Generator,
) -> list:
    """Joint MVN Gibbs draw for item difficulties across a b-correlation group.

    Draws b_i = (b_{i,g1}, ..., b_{i,gG}) for all items i simultaneously.

    Parameters
    ----------
    Z_list : list of ndarray [P, I]
        Augmented latent variables, one per group (representative node).
    theta_list : list of ndarray [P]
        Person traits, one per group.
    b_list : list of ndarray [I]
        Current item difficulties (not modified; new arrays returned).
    T_list : list of ndarray [P, I] bool
        Relevance masks, one per group.
    row_idx_list, col_idx_list : list of ndarray
        Pre-computed index pairs from np.where(T_r).
    a_list : list of ndarray [I] or None per entry
        Discrimination parameters.  None → Rasch (a=1 everywhere).
    Sigma_b : ndarray [G, G]
        Current full covariance matrix for b across the G groups.
    rng : np.random.Generator

    Returns
    -------
    list of ndarray [I]
        Updated b arrays, one per group in the correlation group.

    Notes
    -----
    When G == 1, reduces exactly to scalar gibbs_b (fast path).
    """
    G = len(Z_list)
    I = Z_list[0].shape[1]

    # --- Fast path: G=1 is scalar gibbs_b ---
    if G == 1:
        Z, theta = Z_list[0], theta_list[0]
        T = T_list[0]
        ri, ci = row_idx_list[0], col_idx_list[0]
        a    = a_list[0] if a_list is not None else None
        tau2 = float(Sigma_b[0, 0])
        return [gibbs_b(Z, theta, tau2, T, ri, ci, rng, a=a)]

    # --- General path ---
    S = np.zeros((I, G), dtype=np.float64)
    N = np.zeros((I, G), dtype=np.float64)

    for k in range(G):
        Z_k     = Z_list[k]
        theta_k = theta_list[k]
        ri      = row_idx_list[k]
        ci      = col_idx_list[k]
        a_k     = a_list[k] if a_list is not None else None

        if a_k is None:
            # Rasch: S_{i,k} = Σ_p T[p,i]·(θ_p − Z_{p,i}), N_{i,k} = count
            contrib_S = theta_k[ri] - Z_k[ri, ci]
            contrib_N = np.ones(len(ri), dtype=np.float64)
        else:
            # 2PL: S_{i,k} = Σ_p T[p,i]·a_i·(a_i·θ_p − Z_{p,i})
            a_i       = a_k[ci]
            contrib_S = a_i * (a_i * theta_k[ri] - Z_k[ri, ci])
            contrib_N = a_i ** 2

        np.add.at(S[:, k], ci, contrib_S)
        np.add.at(N[:, k], ci, contrib_N)

    return gibbs_b_correlated_from_stats(S, N, Sigma_b, rng)


def gibbs_b_correlated_from_stats(
    S:       np.ndarray,
    N:       np.ndarray,
    Sigma_b: np.ndarray,
    rng:     np.random.Generator,
) -> list:
    """Joint MVN draw of b_i = (b_i1..b_iG) from Gaussian statistics.

    For each item, b_i ~ N(0, Sigma_b) a priori and the latent likelihood
    contributes precision diag(N_i) and linear term S_i, so

        b_i | rest ~ N(A_i^{-1} S_i, A_i^{-1}),   A_i = Sigma_b^{-1} + diag(N_i).

    S, N : [I, G], already summed over every node of each b-group, on the
    cells where Z was drawn (run_chain builds them with _b_gauss_stats).
    """
    I, G = S.shape
    Sigma_b_inv = np.linalg.inv(Sigma_b)  # [G, G], computed once
    # Batched over items: the posterior precision Sigma_b^-1 + diag(N_i)
    # differs across items only in its diagonal, giving I distinct G x G
    # systems.  numpy's linalg broadcasts over leading axes, so all I
    # inversions and Cholesky factorisations run in two batched LAPACK calls
    # with no Python-level loop -- the same change made to
    # gibbs_theta_correlated (measured 37x there).
    #
    # NOTE: this draws via Cholesky rather than rng.multivariate_normal's SVD.
    # The target distribution is identical; the realised draws differ, so
    # chains are statistically -- not bitwise -- equivalent to the old code.
    A = np.broadcast_to(Sigma_b_inv, (I, G, G)).copy()
    g_idx = np.arange(G)
    A[:, g_idx, g_idx] += N

    Sigma_post = np.linalg.inv(A)
    Sigma_post = 0.5 * (Sigma_post + np.swapaxes(Sigma_post, -1, -2))
    mu_post    = np.einsum('igh,ih->ig', Sigma_post, S)
    L          = np.linalg.cholesky(Sigma_post)
    eps        = rng.standard_normal((I, G))
    new_b      = mu_post + np.einsum('igh,ih->ig', L, eps)

    return [new_b[:, k].copy() for k in range(G)]


# ===========================================================================
# CROSS-NODE: Correlated trait sampling
# ===========================================================================
# These functions implement the ONLY theta path in the package.  They are not
# reserved for correlated models: gibbs_theta_correlated is called every
# iteration regardless, and detects a diagonal Sigma to take an elementwise
# fast path that is bitwise identical to a per-node scalar draw.
# ===========================================================================

def compute_suff_stats(
    Z_list:       list[np.ndarray],
    b_list:       list[np.ndarray],
    T_list:       list[np.ndarray],
    row_idx_list: list[np.ndarray],
    col_idx_list: list[np.ndarray],
    is_polytomous: list[bool],
    a_list:       list[np.ndarray | None] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-person sufficient statistics S_p across all R nodes.

    For each person p and node r the sufficient statistic is:

        Binary/1PL:  S_{p,r} = Σ_{i: T_r[p,i]=1} (Z_{pir} + b_{ir})
        2PL:         S_{p,r} = Σ_{i: T_r[p,i]=1} a_i·(Z_{pir} + a_i·b_{ir})
        GRM:         S_{p,r} = Σ_{i: T_r[p,i]=1}  Z_{pir}

    and the per-person item count (used as likelihood precision) is:

        Binary/1PL:  n_{p,r} = number of relevant items
        2PL:         n_{p,r} = Σ_{i: T_r[p,i]=1} a_i²   (weighted count)
        GRM:         n_{p,r} = number of relevant items

    These match the full conditional derivations in gibbs_theta and
    the unified compute_suff_stats path.  Crucially, using weighted item counts for 2PL
    ensures the cross-node MVN formula Σ_p* = (Σ⁻¹ + Λ_p)⁻¹ remains
    correct, where Λ_p = diag(n_{p,1}, …, n_{p,R}).

    Parameters
    ----------
    Z_list        : list[ndarray [P × I]], one per node
    b_list        : list[ndarray [I]],     one per node (ignored for GRM)
    T_list        : list[ndarray [P × I]], one per node
    row_idx_list  : list[ndarray],         one per node
    col_idx_list  : list[ndarray],         one per node
    is_polytomous : list[bool],            True if the node has n_outcomes > 2
    a_list        : list[ndarray [I] or None] or None
                    Discrimination parameters per node; None entries → 1PL/Rasch

    Returns
    -------
    S : ndarray [P × R], sufficient statistics
    n : ndarray [P × R], effective item counts (precision weights)
    """
    R = len(Z_list)
    P = Z_list[0].shape[0]

    S = np.zeros((P, R), dtype=np.float64)
    n = np.zeros((P, R), dtype=np.float64)

    for r in range(R):
        Z_r      = Z_list[r]
        b_r      = b_list[r]
        row_idx  = row_idx_list[r]
        col_idx  = col_idx_list[r]
        grm      = is_polytomous[r]
        a_r      = (a_list[r] if a_list is not None else None)

        if len(row_idx) == 0:
            continue

        if grm:
            # GRM: S_{p,r} = Σ Z_pir,  n_{p,r} = count of relevant items
            np.add.at(S[:, r], row_idx, Z_r[row_idx, col_idx])
            np.add.at(n[:, r], row_idx, 1.0)

        elif a_r is None:
            # Rasch/1PL: S_{p,r} = Σ (Z_pir + b_ir),  n_{p,r} = count
            np.add.at(S[:, r], row_idx, Z_r[row_idx, col_idx] + b_r[col_idx])
            np.add.at(n[:, r], row_idx, 1.0)

        else:
            # 2PL: S_{p,r} = Σ a_i(Z_pir + a_i*b_ir),  n_{p,r} = Σ a_i²
            a_i  = a_r[col_idx]
            suff = a_i * (Z_r[row_idx, col_idx] + a_i * b_r[col_idx])
            np.add.at(S[:, r], row_idx, suff)
            np.add.at(n[:, r], row_idx, a_i ** 2)

    return S, n


def reduce_suff_stats_to_dims(
    S:            np.ndarray,
    n:            np.ndarray,
    dim_to_nodes: list,
) -> tuple[np.ndarray, np.ndarray]:
    """Sum node-width sufficient statistics down to latent-dimension width.

    ``compute_suff_stats`` always returns one column per physical node
    (width R), because T/row_idx/col_idx are stored per node. When
    ``spec.node_to_latent`` collapses several nodes onto one latent
    dimension, ``gibbs_theta_correlated`` needs sufficient statistics at
    dimension width (D) to match ``Sigma`` (D x D). This sums each
    dimension's member-node columns together.

    When every node has its own dimension (D == R, the default), this is
    a no-op relabeling: each output column equals its single input column.

    Parameters
    ----------
    S, n         : ndarray [P x R], from compute_suff_stats
    dim_to_nodes : list[list[int]], length D; entry d lists the 0-based
                   node indices sharing latent dimension d

    Returns
    -------
    S_d, n_d : ndarray [P x D]
    """
    P = S.shape[0]
    D = len(dim_to_nodes)
    S_d = np.zeros((P, D), dtype=np.float64)
    n_d = np.zeros((P, D), dtype=np.float64)
    for d, nodes in enumerate(dim_to_nodes):
        S_d[:, d] = S[:, nodes].sum(axis=1)
        n_d[:, d] = n[:, nodes].sum(axis=1)
    return S_d, n_d


def build_sigma(
    sigma2_list: np.ndarray,
    corr_matrix: np.ndarray,
) -> np.ndarray:
    """Construct the trait covariance matrix Σ = D R D.

    The separation strategy Σ = D R D, where D = diag(σ_1, …, σ_R) =
    diag(sqrt(σ²_1), …, sqrt(σ²_R)) and R is the correlation matrix,
    keeps prior beliefs about variances independent of correlations.

    Parameters
    ----------
    sigma2_list : ndarray [R], per-node variances σ²_r (NOT standard deviations)
    corr_matrix : ndarray [R × R], valid correlation matrix.  Pass np.eye(R)
                  for the independent case; the result is diag(sigma2_list).

    Returns
    -------
    Sigma : ndarray [R × R], full covariance matrix Σ = D R D
    """
    D = np.diag(np.sqrt(np.asarray(sigma2_list, dtype=np.float64)))
    return D @ corr_matrix @ D


def gibbs_theta_given(
    S:         np.ndarray,
    n:         np.ndarray,
    Sigma:     np.ndarray,
    theta_mat: np.ndarray,
    owned:     list[int],
    rng:       np.random.Generator,
) -> np.ndarray:
    """Gibbs draw of the owned theta dimensions given the others.

    theta_p ~ N(0, Sigma) jointly over D dimensions.  The owned dimensions G
    have Gaussian latent sufficient statistics (S, n, as for
    gibbs_theta_correlated); the remaining dimensions M are held at their
    current values theta_mat[:, M].  With Lam = Sigma^{-1}:

        prec_p = Lam_GG + diag(n_pG)
        rhs_p  = S_pG - Lam_GM theta_pM
        theta_pG | rest ~ N(prec_p^{-1} rhs_p, prec_p^{-1})

    Returns [P, |G|] in the order of ``owned``.
    """
    D = Sigma.shape[0]
    G = np.asarray(owned, dtype=np.intp)
    M = np.setdiff1d(np.arange(D), G)
    Lam = np.linalg.inv(np.atleast_2d(Sigma))
    P = S.shape[0]
    A = np.broadcast_to(Lam[np.ix_(G, G)], (P, len(G), len(G))).copy()
    k = np.arange(len(G))
    A[:, k, k] += n[:, G]
    rhs = S[:, G] - theta_mat[:, M] @ Lam[np.ix_(M, G)]
    V = np.linalg.inv(A)
    V = 0.5 * (V + np.swapaxes(V, -1, -2))
    mu = np.einsum('pab,pb->pa', V, rhs)
    L = np.linalg.cholesky(V)
    return mu + np.einsum('pab,pb->pa', L, rng.standard_normal((P, len(G))))


def gibbs_theta_correlated(
    S:           np.ndarray,
    n:           np.ndarray,
    Sigma:       np.ndarray,
    rng:         np.random.Generator,
    prior_lower: float | None = None,
    prior_upper: float | None = None,
) -> np.ndarray:
    """MVN Gibbs draw for theta_p across all D latent dimensions simultaneously.

    Full conditional (MVN conjugacy; see module docstring for derivation):

        Sigma_p* = (Sigma^-1 + Lambda_p)^-1
        mu_p*    = Sigma_p* @ S_p

    where Lambda_p = diag(n_{p,1}, ..., n_{p,D}) and S_p are sufficient stats
    from compute_suff_stats.

    VECTORISATION
    -------------
    Each person's posterior precision is Sigma^-1 + diag(n_p), which differs
    across persons only in its diagonal.  This gives P distinct D x D systems.
    NumPy's linalg routines broadcast over leading axes, so all P inversions
    and Cholesky factorisations are performed in two batched LAPACK calls with
    no Python-level loop over persons.

    Measured against a per-person Python loop (identical arithmetic, same
    RNG stream):

        P=1000,  D=2:  25.6 ms -> 0.69 ms   (37x)
        P=5000,  D=2:  131.6 ms -> 3.75 ms  (35x)
        P=1000,  D=3:  27.5 ms -> 1.26 ms   (22x)
        P=1000,  D=5:  29.1 ms -> 2.50 ms   (12x)

    The speedup narrows as D grows because the O(P*D^3) arithmetic begins to
    dominate the per-call overhead that batching eliminates.

    RNG STREAM COMPATIBILITY
    ------------------------
    ``rng.standard_normal((P, D))`` fills row-major, consuming the stream in
    exactly the order that P sequential ``rng.standard_normal(D)`` calls did.
    Unbounded draws therefore reproduce the per-person loop to
    floating-point roundoff (max |diff| ~ 1e-16, from BLAS ordering alone).

    The bounded path is NOT stream-compatible: it redraws only the persons
    still outside the bounds, whereas a per-person loop would draw each
    person until accepted.  Both target the same distribution.

    DIAGONAL FAST PATH
    ------------------
    When Sigma is diagonal (the uncorrelated case), (Sigma^-1 + Lambda_p)^-1
    is diagonal too, so the D x D linear algebra collapses to elementwise
    reciprocals.  This is detected automatically and is a further ~8x faster
    (P=1000, D=2: 0.69 ms -> 0.08 ms).  The draw is mathematically identical;
    only the arithmetic path differs.

    Parameters
    ----------
    S     : ndarray [P x D], from compute_suff_stats (+ reduce_suff_stats_to_dims)
    n     : ndarray [P x D], from compute_suff_stats (+ reduce_suff_stats_to_dims)
    Sigma : ndarray [D x D], from build_sigma
    rng   : numpy Generator
    prior_lower : float or None - element-wise lower bound on all theta dimensions
    prior_upper : float or None - element-wise upper bound on all theta dimensions

    Returns
    -------
    theta : ndarray [P x D]
    """
    S = np.asarray(S, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)
    P, D = S.shape

    Sigma = np.atleast_2d(np.asarray(Sigma, dtype=np.float64))
    if Sigma.shape != (D, D):
        raise ValueError(
            f"Sigma must be [{D} x {D}] to match S/n width; got {Sigma.shape}."
        )

    bounded = not (prior_lower is None and prior_upper is None)
    lo = -np.inf if prior_lower is None else float(prior_lower)
    hi =  np.inf if prior_upper is None else float(prior_upper)

    # --- Diagonal fast path -------------------------------------------------
    # Sigma diagonal  =>  (Sigma^-1 + Lambda_p)^-1 diagonal for every p.
    diag_Sigma = np.diag(Sigma)
    if np.count_nonzero(Sigma - np.diag(diag_Sigma)) == 0:
        if np.any(diag_Sigma <= 0.0):
            raise np.linalg.LinAlgError(
                "Sigma has a non-positive diagonal entry; covariance must be "
                "positive definite."
            )
        sigma2_star = 1.0 / (1.0 / diag_Sigma[np.newaxis, :] + n)   # [P x D]
        mu_star     = sigma2_star * S
        sd_star     = np.sqrt(sigma2_star)

        if not bounded:
            return mu_star + sd_star * rng.standard_normal((P, D))

        # Exact truncated draw per component via inverse CDF (no rejection
        # needed: independent components under a diagonal Sigma).
        lo_std = (lo - mu_star) / sd_star if prior_lower is not None else np.full((P, D), -np.inf)
        hi_std = (hi - mu_star) / sd_star if prior_upper is not None else np.full((P, D),  np.inf)
        p_lo = _Phi(lo_std)
        p_hi = _Phi(hi_std)
        U = p_lo + (p_hi - p_lo) * rng.uniform(size=(P, D))
        U = np.clip(U, 1e-14, 1.0 - 1e-14)
        return mu_star + sd_star * _Phi_inv(U)

    # --- General dense path -------------------------------------------------
    Sigma_inv = np.linalg.inv(Sigma)                    # [D x D], once

    # Posterior precision stack: A_p = Sigma^-1 + diag(n_p)
    A = np.broadcast_to(Sigma_inv, (P, D, D)).copy()    # copy: broadcast is read-only
    d_idx = np.arange(D)
    A[:, d_idx, d_idx] += n

    Sigma_p = np.linalg.inv(A)                          # batched, one call
    # Symmetrise: guards Cholesky against roundoff asymmetry from the inverse.
    Sigma_p = 0.5 * (Sigma_p + np.swapaxes(Sigma_p, -1, -2))

    mu = np.einsum('pij,pj->pi', Sigma_p, S)            # [P x D]
    L  = np.linalg.cholesky(Sigma_p)                    # batched, one call

    if not bounded:
        eps = rng.standard_normal((P, D))
        return mu + np.einsum('pij,pj->pi', L, eps)

    # Bounded: batched rejection, redrawing only the persons still outside.
    theta   = np.empty((P, D), dtype=np.float64)
    pending = np.arange(P)
    for _ in range(_MAX_TRUNC_MVN_SWEEPS):
        if pending.size == 0:
            break
        eps  = rng.standard_normal((pending.size, D))
        draw = mu[pending] + np.einsum('pij,pj->pi', L[pending], eps)
        ok   = np.all((draw >= lo) & (draw <= hi), axis=1)
        theta[pending[ok]] = draw[ok]
        pending = pending[~ok]

    if pending.size:
        # Fallback for persons whose posterior sits far outside the bounds:
        # independent per-component truncated Normals using the marginal sds.
        # Ignores within-person correlation for these few persons, but is
        # strictly better than returning an out-of-bounds draw.
        sd = np.sqrt(np.diagonal(Sigma_p[pending], axis1=-2, axis2=-1))
        m  = mu[pending]
        lo_std = (lo - m) / sd if prior_lower is not None else np.full_like(m, -np.inf)
        hi_std = (hi - m) / sd if prior_upper is not None else np.full_like(m,  np.inf)
        p_lo = _Phi(lo_std)
        p_hi = _Phi(hi_std)
        U = p_lo + (p_hi - p_lo) * rng.uniform(size=m.shape)
        U = np.clip(U, 1e-14, 1.0 - 1e-14)
        theta[pending] = m + sd * _Phi_inv(U)

    return theta


# ===========================================================================
# LATENT-REGRESSION VIEW: shared by the covariate coefficients and imputation
# ===========================================================================
#
# At every augmented (probit) node r the latent response satisfies
#
#     Z_pi = a_i * ( core_pi + x_p'beta + q_i'kappa + w_pi'gamma ) + eps,
#     eps ~ N(0, 1),
#
# on the cells where Z was actually drawn (the node's "latent mask"):
#
#     binary Rasch / 2PL   core = theta_p - b_i,  mask = T_r
#     3PL (Beguin-Glas)    core = theta_p - b_i,  mask = T_r & (U = 0)
#     GRM                  core = theta_p,        mask = T_r
#                          (thresholds carry the item location)
#     mixed GRM            core = theta_p - b_i on binary items, theta_p on
#                          polytomous items,      mask = T_r
#     logit / GPCM         no latent response: excluded (empty mask)
#
# and a_i = 1 when the node has no discrimination.  The caller (run_chain)
# builds, per node, the residual
#
#     E_pi = Z_pi - a_i * ( core_pi + every covariate term EXCEPT the target )
#          = a_i * (target term) + eps,
#
# so each target is a Normal linear regression with design rows a_i * d_pi
# (d = x_p, q_i or w_pi).  Every function below takes that (E, mask, a)
# triple, so the four node types reduce to one formula and a node never
# borrows another node's discrimination.
# ===========================================================================


def _mvn_from_precision(prec, rhs, rng):
    """Draw from N(prec^{-1} rhs, prec^{-1})."""
    K = prec.shape[0]
    if K == 1:
        v = 1.0 / float(prec[0, 0])
        return np.array([rng.normal(v * float(rhs[0]), np.sqrt(v))])
    V = np.linalg.solve(prec, np.eye(K))
    V = 0.5 * (V + V.T)
    return rng.multivariate_normal(V @ rhs, V)


def coef_gaussian_terms(
    kind:        str,
    group_nodes: list[int],
    resid_list:  list[np.ndarray | None],
    mask_list:   list[np.ndarray | None],
    a_list:      list[np.ndarray | None],
    design:      np.ndarray,
):
    """Gaussian sufficient statistics of a shared coefficient c.

    For every node r in the group and every cell of its mask,
    E_r[p,i] = a_ri d_pi' c + eps.  Returns

        G = sum_r sum_cells a_ri^2 d_pi d_pi'      [K, K]
        h = sum_r sum_cells a_ri d_pi E_r[p,i]     [K]

    so the log-likelihood is h'c - c'Gc/2 + const.  The design is reduced
    along the axis it does not vary on:

        kind='x' (beta):  d_pi = x_p,  design [P, K]
        kind='q' (kappa): d_pi = q_i,  design [I, K]
        kind='w' (gamma): d_pi = w_pi, design [P, I, K]

    Each node keeps its OWN discrimination vector (a_list[r], None = ones).
    A node whose resid or mask entry is None contributes nothing.  Passing
    resid entries of zeros returns G alone (used for MH preconditioning).
    """
    K = design.shape[-1]
    G = np.zeros((K, K))
    h = np.zeros(K)
    for r in group_nodes:
        E, M = resid_list[r], mask_list[r]
        if E is None or M is None or not np.any(M):
            continue
        a_r = a_list[r] if a_list is not None else None
        if kind == "w":
            ri, ci = np.where(M)
            a_c = np.ones(len(ri)) if a_r is None else a_r[ci]
            D = a_c[:, None] * design[ri, ci]          # [n_cells, K]
            G += D.T @ D
            h += D.T @ E[ri, ci]
            continue
        a1 = np.ones(M.shape[1]) if a_r is None else np.asarray(a_r, float)
        Mw = M.astype(np.float64)
        Ez = np.where(M, E, 0.0)
        if kind == "x":
            n = Mw @ (a1 ** 2)                          # [P]
            sv = Ez @ a1                                # [P]
        elif kind == "q":
            n = (a1 ** 2) * Mw.sum(axis=0)              # [I]
            sv = a1 * Ez.sum(axis=0)                    # [I]
        else:
            raise ValueError(f"kind must be 'x', 'q' or 'w'; got {kind!r}")
        G += design.T @ (n[:, None] * design)
        h += design.T @ sv
    return G, h


def gibbs_coef_pooled(
    kind:        str,
    group_nodes: list[int],
    resid_list:  list[np.ndarray | None],
    mask_list:   list[np.ndarray | None],
    a_list:      list[np.ndarray | None],
    design:      np.ndarray,
    tau2:        float,
    rng:         np.random.Generator,
    prior_mean:  np.ndarray | None = None,
) -> np.ndarray:
    """Exact Gibbs draw for a covariate coefficient shared by a group of nodes.

    Model, for every node r in the group and every cell in its latent mask:

        E_r[p,i] = a_ri * d_pi' c + eps,   eps ~ N(0, 1),   c ~ N(mu0, tau2 I)

    Full conditional, with (G, h) from coef_gaussian_terms:

        prec = I/tau2 + G,   rhs = mu0/tau2 + h
        c | rest ~ N(prec^{-1} rhs, prec^{-1})

    Each node keeps its OWN discrimination vector, so a group mixing Rasch
    and 2PL nodes, or 2PL nodes in different a-groups, is handled exactly.
    Nodes whose resid_list entry is None (non-augmented) contribute nothing;
    groups containing such nodes are updated by MH in run_chain instead.
    """
    K = design.shape[-1]
    G, h = coef_gaussian_terms(kind, group_nodes, resid_list, mask_list,
                               a_list, design)
    mu0 = np.zeros(K) if prior_mean is None else np.asarray(prior_mean, float)
    return _mvn_from_precision(np.eye(K) / tau2 + G, mu0 / tau2 + h, rng)


# ---------------------------------------------------------------------------
# Missing covariates: one algebra for X, Q and W
# ---------------------------------------------------------------------------
#
# A "unit" is what one missing row belongs to: a person (X), an item (Q) or
# a (person, item) cell (W).  Given everything else, units are independent.
# For unit u, node r contributes, over the unit's cells in the latent mask,
#
#     n_ru = sum a_ri^2,          S_ru = sum a_ri E_r[p,i],
#
# and, splitting the unit's covariate vector into observed (fixed) and
# missing (drawn) features, with the node's coefficient c_r at full width,
#
#     prec_u = I/v0 + sum_r n_ru c_r^mis c_r^mis'
#     rhs_u  = m0/v0 + sum_r c_r^mis (S_ru - n_ru v_u^obs' c_r^obs)
#     v_u^mis | rest ~ N(prec_u^{-1} rhs_u, prec_u^{-1})        (probit only)
#
# missing_unit_stats builds (n, S); missing_terms builds (prec, rhs) grouped
# by missingness pattern so every draw is a batched K_mis-dimensional solve.
# The same terms are the Gaussian part of the MH target used when a logit
# node also uses the covariate (run_chain, _mh_missing_units).
# ---------------------------------------------------------------------------


def missing_unit_stats(
    kind:       str,
    units:      np.ndarray,
    resid_list: list[np.ndarray | None],
    mask_list:  list[np.ndarray | None],
    a_list:     list[np.ndarray | None],
):
    """Per-unit (n_ru, S_ru) for every node; entries are None for skipped nodes.

    kind 'x': units are person indices [U];  'q': item indices [U];
    'w': (person, item) pairs [U, 2].
    """
    R = len(resid_list)
    n_list: list[np.ndarray | None] = [None] * R
    S_list: list[np.ndarray | None] = [None] * R
    for r in range(R):
        E, M = resid_list[r], mask_list[r]
        if E is None or M is None:
            continue
        I_r = M.shape[1]
        a_r = a_list[r] if a_list is not None else None
        a1 = np.ones(I_r) if a_r is None else np.asarray(a_r, float)
        Ez = np.where(M, E, 0.0)
        if kind == "x":
            n_list[r] = (M.astype(np.float64) @ (a1 ** 2))[units]
            S_list[r] = (Ez @ a1)[units]
        elif kind == "q":
            n_list[r] = ((a1 ** 2) * M.sum(axis=0))[units]
            S_list[r] = (a1 * Ez.sum(axis=0))[units]
        elif kind == "w":
            p, i = units[:, 0], units[:, 1]
            n_list[r] = (a1[i] ** 2) * M[p, i]
            S_list[r] = a1[i] * Ez[p, i]
        else:
            raise ValueError(f"kind must be 'x', 'q' or 'w'; got {kind!r}")
    return n_list, S_list


def missing_terms(
    n_list:         list[np.ndarray | None],
    S_list:         list[np.ndarray | None],
    coef_list:      list[np.ndarray | None],
    values:         np.ndarray,
    feat_obs:       np.ndarray,
    prior_mean:     float,
    prior_variance: float,
):
    """Gaussian full-conditional terms of the missing features, by pattern.

    values, feat_obs : [U, K] current covariate values / observed mask of the
                       units.  coef_list[r] is node r's coefficient at full
                       width (None = node does not use this covariate type).
    prior_variance   : np.inf gives the likelihood terms alone.

    Returns a list of (sel, miss_idx, prec [n, K_mis, K_mis], rhs [n, K_mis]),
    one per missingness pattern; sel indexes the units.
    """
    U, K = feat_obs.shape
    code = feat_obs.astype(np.int64) @ (1 << np.arange(K, dtype=np.int64))
    prior_prec = 0.0 if np.isinf(prior_variance) else 1.0 / prior_variance
    groups = []
    for pc in np.unique(code):
        sel = np.where(code == pc)[0]
        fo = feat_obs[sel[0]]
        miss_idx, obs_idx = np.where(~fo)[0], np.where(fo)[0]
        Km = len(miss_idx)
        if Km == 0:
            continue
        prec = np.broadcast_to(np.eye(Km) * prior_prec, (len(sel), Km, Km)).copy()
        rhs = np.full((len(sel), Km), prior_mean * prior_prec)
        for r, c in enumerate(coef_list):
            if c is None or n_list[r] is None:
                continue
            n = n_list[r][sel]
            if not np.any(n):
                continue
            S = S_list[r][sel].copy()
            if len(obs_idx):
                S -= n * (values[sel][:, obs_idx] @ c[obs_idx])
            cm = c[miss_idx]
            prec += n[:, None, None] * np.outer(cm, cm)[None]
            rhs += S[:, None] * cm[None, :]
        groups.append((sel, miss_idx, prec, rhs))
    return groups


def draw_from_terms(prec, rhs, rng):
    """Batched draws u_n ~ N(prec_n^{-1} rhs_n, prec_n^{-1})."""
    V = np.linalg.inv(prec)
    V = 0.5 * (V + np.swapaxes(V, -1, -2))
    mu = np.einsum('nab,nb->na', V, rhs)
    L = np.linalg.cholesky(V)
    return mu + np.einsum('nab,nb->na', L, rng.standard_normal(rhs.shape))


def _impute_gibbs(kind, units, resid_list, mask_list, a_list, coef_list,
                  values, feat_obs, prior_mean, prior_variance, rng):
    n, S = missing_unit_stats(kind, units, resid_list, mask_list, a_list)
    out = values.copy()
    for sel, miss_idx, prec, rhs in missing_terms(
            n, S, coef_list, values, feat_obs, prior_mean, prior_variance):
        out[sel[:, None], miss_idx[None, :]] = draw_from_terms(prec, rhs, rng)
    return out


def gibbs_X_missing(
    missing_persons:    np.ndarray,
    resid_list:         list[np.ndarray | None],
    mask_list:          list[np.ndarray | None],
    a_list:             list[np.ndarray | None],
    beta_list:          list[np.ndarray | None],
    X_working:          np.ndarray,
    X_feature_observed: np.ndarray,
    prior_mean:         float,
    prior_variance:     float,
    rng:                np.random.Generator,
) -> np.ndarray:
    """Gibbs draw for missing person-covariate features (probit nodes only).

    E_r = Z_r - a_r (core_r + Q kappa_r + W gamma_r) = a_ri X_p' beta_r + eps
    on node r's latent mask; see the block comment above for the formula.
    beta_list[r] is at FULL X width (zeros on unselected columns).
    """
    units = np.asarray(missing_persons, dtype=np.intp)
    if len(units) == 0:
        return X_working
    X_working[units] = _impute_gibbs(
        "x", units, resid_list, mask_list, a_list, beta_list,
        X_working[units], X_feature_observed[units], prior_mean, prior_variance, rng)
    return X_working


def gibbs_Q_missing(
    missing_items:      np.ndarray,
    resid_list:         list[np.ndarray | None],
    mask_list:          list[np.ndarray | None],
    a_list:             list[np.ndarray | None],
    kappa_list:         list[np.ndarray | None],
    Q_working:          np.ndarray,
    Q_feature_observed: np.ndarray,
    prior_mean:         float,
    prior_variance:     float,
    rng:                np.random.Generator,
) -> np.ndarray:
    """Gibbs draw for missing item-covariate features (probit nodes only).

    E_r = Z_r - a_r (core_r + X beta_r + W gamma_r) = a_ri Q_i' kappa_r + eps.
    """
    units = np.asarray(missing_items, dtype=np.intp)
    if len(units) == 0:
        return Q_working
    Q_working[units] = _impute_gibbs(
        "q", units, resid_list, mask_list, a_list, kappa_list,
        Q_working[units], Q_feature_observed[units], prior_mean, prior_variance, rng)
    return Q_working


def gibbs_W_missing(
    missing_cells:      np.ndarray,
    resid_list:         list[np.ndarray | None],
    mask_list:          list[np.ndarray | None],
    a_list:             list[np.ndarray | None],
    gamma_list:         list[np.ndarray | None],
    W_working:          np.ndarray,
    W_feature_observed: np.ndarray,
    prior_mean:         float,
    prior_variance:     float,
    rng:                np.random.Generator,
) -> np.ndarray:
    """Gibbs draw for missing person-by-item covariate features.

    E_r = Z_r - a_r (core_r + X beta_r + Q kappa_r) = a_ri W_pi' gamma_r + eps.
    A cell no augmented node uses gets a prior draw.
    """
    cells = np.asarray(missing_cells, dtype=np.intp).reshape(-1, 2)
    if len(cells) == 0:
        return W_working
    p, i = cells[:, 0], cells[:, 1]
    W_working[p, i] = _impute_gibbs(
        "w", cells, resid_list, mask_list, a_list, gamma_list,
        W_working[p, i], W_feature_observed[p, i], prior_mean, prior_variance, rng)
    return W_working


def gibbs_Sigma_invwishart(
    data_matrix: np.ndarray,
    mu0:         np.ndarray,
    Psi0:        np.ndarray,
    nu0:         int,
    rng:         np.random.Generator,
) -> np.ndarray:
    """Exact Gibbs draw for a covariance matrix Σ with Inverse-Wishart prior.

    Full conditional::

        Σ | data  ~  InvWishart(Ψ₀ + ScatterMatrix,  ν₀ + N)

    where N is the number of rows in data_matrix and
    ScatterMatrix = (data − μ₀)ᵀ (data − μ₀).

    This function is shared by all three correlation types:

    * ``b.correlation`` — data_matrix = b_matrix [I × G], μ₀ = 0
    * ``a.correlation`` — data_matrix = a_matrix [I × G], μ₀ = μ_a
      (log(a) for LogNormal prior; raw a for TruncNormal prior)
    * ``theta.correlation`` — data_matrix = theta_matrix [P × D], μ₀ = 0

    Parameters
    ----------
    data_matrix : ndarray [N, G]
        Observed parameter values (items or persons × group/dim).
    mu0 : ndarray [G]
        Prior mean.  Pass ``np.zeros(G)`` for b and θ.
    Psi0 : ndarray [G, G]
        Scale matrix Ψ₀ (symmetric positive definite).
    nu0 : int
        Prior degrees of freedom ν₀ ≥ G + 1.
    rng : np.random.Generator

    Returns
    -------
    ndarray [G, G]
        A draw from the posterior InvWishart.  Always positive definite.
    """
    N         = data_matrix.shape[0]
    residuals = data_matrix - mu0[np.newaxis, :]
    scatter   = residuals.T @ residuals
    Psi_post  = np.asarray(Psi0, dtype=np.float64) + scatter
    nu_post   = nu0 + N
    return _stats.invwishart.rvs(df=nu_post, scale=Psi_post,
                                 random_state=int(rng.integers(2**31)))


