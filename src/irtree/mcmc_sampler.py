"""
irtree.mcmc_sampler
===================
Layer 6 — Hybrid Gibbs/MH sampler for IRTree models.

ROLE IN THE PACKAGE
===================
This module is the orchestrating loop.  It knows which functions to call
and in what order, but contains no statistical math itself.  All Gibbs
draws live in gibbs_samplers.py; all MH steps are inline helpers here
(prefixed with _mh_) because they are only ever called from this loop and
have no use outside of it.

ONE ITERATION
=============
Every node is either augmented (probit: Rasch/1PL, 2PL, 3PL, GRM, mixed GRM;
a latent Z is drawn) or not (logit 2PL/3PL, GPCM/PCM; no Z, updated by MH
against the node likelihood).  At an augmented node, on the cells where Z is
drawn (its latent mask: every relevant cell, or the U = 0 ability cells at a
3PL node),

    Z = a (core + X beta + Q kappa + W gamma) + eps,   eps ~ N(0, 1),

with core = theta - b (binary), theta (GRM; thresholds carry the location)
or theta - b_binary on the binary items of a mixed GRM node.  Every Gibbs
statistic below is built from this "latent view" (_latent_view /
_latent_resid), so masks, discriminations, offsets and node types are
handled in one place.

1. Node loop (one node at a time)
     binary probit   sample_Z -> gibbs_b (unless b is shared/correlated/logit)
                     -> a: gibbs_a (bounded prior) or _mh_discrimination
                        (LogNormal), unless a is shared or correlated
     3PL             sample_U_3pl -> sample_Z on U = 0 -> gibbs_b, gibbs_c_3pl,
                     a as above (ability cells only)
     GRM             Cowles step (_mh_ordered_thresholds, Z integrated out)
                     -> sample_Z_grm -> gibbs_thresholds_grm (exact, given Z);
                     shared thresholds: Z only (step 3)
     mixed GRM       binary items as binary probit, polytomous items as GRM;
                     shared b: Z only (step 3)
     logit / GPCM    a and c by MH (_mh_logit_discrimination,
                     _mh_logit_guessing) unless a is shared or correlated;
                     theta and b are updated in steps 3-4
2. Shared a and c: pooled draw per group (all members; MH with every
   member's likelihood when a member is logit).  Estimated a-group
   variances of uncorrelated groups (MH; truncated-Normal likelihood).
3. Shared b: pooled Gibbs (binary, all probit); item-wise MH
   (_mh_diag_units) for groups with a logit member; pooled Cowles step for
   threshold groups with a GRM member (also mixed-format groups), then the
   GRM members' Z redrawn.
4. theta: Gibbs for dimensions whose nodes are all augmented
   (gibbs_theta_correlated, or gibbs_theta_given conditional on the MH
   dimensions under a correlated Sigma); person-wise MH (_mh_diag_units) for
   dimensions with a logit/GPCM node, with the conditional prior from
   Sigma^{-1} and every node on the dimension in the target.
5. Location move: theta_d, the b-groups on d and the GRM latent responses
   shifted by a common c drawn from its exact conditional (Liu & Sabatti
   2000).
6. sigma2 (gibbs_sigma2 or _mh_sigma2_halfnormal; fixed values are used as
   given), theta correlation within each Constraints(theta_corr=...) block
   (one block of all D dims by default; InvWishart Gibbs per block, LKJ via
   _mh_rho_fisher_z for 2 dims or _mh_correlation_vine for more),
   b.variance (InvGamma).
7. b-correlation groups: probit groups jointly given the logit groups
   (gibbs_b_correlated_from_stats / gibbs_theta_given), logit groups by
   item-wise MH with the conditional prior; then Sigma_b.
8. a-correlation groups: componentwise exact Gibbs (truncated to a > 0) or
   MH on log a for logit groups; estimated group variances (Half-Normal sd
   prior by default under LKJ/fixed); slopes centred at the prior mean for
   the rho update; then Sigma_a.
9. Covariates: beta, kappa, gamma per group (gibbs_coef_pooled when every
   member is augmented, _mh_coefficient otherwise); interweaving (ASIS)
   steps for kappa/b and beta/theta; missing X, Q, W per unit (Gibbs, or
   _mh_missing_units for units a logit node uses).

Warm-up: MH blocks adapt their step size (mcmc_adapt) and, in evidence mode,
draws are kept once every block has frozen; pure-Gibbs chains burn in for
n_warmup iterations (>= 2*M*W).

CHAIN LAYOUT
============
The chain is a [n_samples × d] float32 (or float64) array.  Columns are
laid out in the following canonical order, built by _build_chain_layout():

    theta_{dim_label}_{person}  for each latent dim d=0..D-1, p=1..P
                                dim_label uses & notation for shared dims
    b_{group_label}_{item}      for each b-sharing group g (binary nodes: [I])
    b_{group_label}_{item}_thresh_{k}   for homogeneous GRM/GPCM nodes
    b_{group_label}_{item}_tau{k}       for mixed-format nodes (polytomous items)
    a_{group_label}_{item}      for groups with free discriminations
    c_{group_label}_{item}      for 3PL groups only
    sigma2_{dim_label}          one scalar per latent dim
    b.variance_{group_label}    one scalar per b-group with InvGammaPrior variance
    a.variance_{group_label}    one scalar per a-group with an estimated variance
    rho_{d1}_{d2}               for d1<d2 (1-based) in the same theta_corr block,
                                if theta.correlation is LKJ/bounded
    Sigma_theta_{d1}_{d2}       each block's lower triangle (d1≤d2, 1-based) if
                                InverseWishartPrior
    Sigma_b_{tag}_{i}_{j}       for each b-corr group, lower triangle, if InverseWishartPrior
    b.rho_{label1}_{label2}     for each b-corr group pair, if LKJPrior
    Sigma_a_{tag}_{i}_{j}       for each a-corr group, lower triangle, if InverseWishartPrior
    a.rho_{label1}_{label2}     for each a-corr group pair, if LKJPrior
    beta_{group_label}_{cov}    per person-covariate group (shared theta)
    kappa_{group_label}_{cov}   per item-covariate group (shared b)
    gamma_{group_label}_{cov}   per person-by-item group (gamma_shared)
                                {cov} is the covariate name, or its 1-based
                                column index; see covariate_column_names()
    {beta|kappa|gamma}.variance_{group_label}   estimated coefficient-group variance
    X_missing_{p+1}_{k+1}   for missing persons, only for missing feature indices k
    Q_missing_{i+1}_{k+1}   for missing items,   only for missing feature indices k
    W_missing_{p+1}_{i+1}_{k+1}  for missing (p,i) cells, only for missing feature indices k

INLINE MH HELPERS
=================
MH steps live here as module-level private functions (prefixed _mh_):

    _mh_discrimination        — log-scale RW for probit a (LogNormal prior)
    _mh_logit_discrimination  — log-scale RW for logit/GPCM a, per item
    _mh_logit_guessing        — logit-scale RW for logit 3PL c
    _mh_sigma2_halfnormal     — log-scale RW for sigma^2 under a HalfNormal prior
    _mh_rho_fisher_z          — Fisher-z RW for a scalar rho (theta, b, a)
    _mh_correlation_vine      — vine partial-correlation RW, G > 2
    _mh_coefficient           — preconditioned RW for a covariate coefficient
                                group with a logit member
    _mh_missing_units         — per-unit RW for missing covariates at logit nodes
    _mh_diag_units            — vectorized per-person / per-item RW (logit
                                theta and b, binary b with logit members)
    _mh_ordered_thresholds    — Cowles step for ordered thresholds
                                (scales from _threshold_proposal_scale)

REFERENCES
==========
Albert, J.H. & Chib, S. (1993). JASA, 88(422), 669–679.
Béguin, A.A. & Glas, C.A.W. (2001). Psychometrika, 66(4), 541–561.
Cowles, M.K. (1996). Statistics and Computing, 6(2), 101–111.
Gelman, A. (2006). Bayesian Analysis, 1(3), 515–534.
Johnson, V.E. & Albert, J.H. (1999). Ordinal Data Modeling. Springer.
Klein Entink, R.H., Fox, J.-P., & van der Linden, W.J. (2009). Psychometrika, 74(1), 21–48.
Lewandowski, D., Kurowicka, D. & Joe, H. (2009). J. Multivariate Analysis.
Liu, J.S. & Sabatti, C. (2000). Biometrika, 87(2), 353–369.
Roberts, G.O., Gelman, A. & Gilks, W.R. (1997). Ann. Appl. Prob. 7, 110–120.
Yu, Y. & Meng, X.-L. (2011). JCGS, 20(3), 531–570.
"""

# B023 (closure uses a loop variable) is a false positive throughout this
# file: the helpers defined inside the iteration loop (and its per-group inner
# loops) read the current draws by closure and are called, directly or by the
# MH helpers they are passed to, within the same iteration. None is stored
# and called later, so none can see a stale loop variable.
# ruff: noqa: B023

from __future__ import annotations

import copy
import dataclasses
import math
import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy import stats as _stats
from scipy.special import ndtr as _Phi
from scipy.special import ndtri as _Phi_inv

import irtree.data_prep as _dp
import irtree.gibbs_samplers as _gibbs
import irtree.mcmc_adapt as _mcmc_adapt
from irtree.constraints import NO_CONSTRAINTS
from irtree.data_prep import PreparedData
from irtree.likelihood import init_cache, recompute_node
from irtree.mcmc_adapt import adapt_step, make_adapt_state, target_accept
from irtree.node_models import (
    GPCMNode,
    GRMNode,
    NodeBParams,
    NodeModel,
)
from irtree.priors import (
    BetaPrior,
    FixedCorrelation,
    HalfNormalPrior,
    InverseWishartPrior,
    InvGammaPrior,
    LKJPrior,
    LogNormalPrior,
    NormalPrior,
    Priors,
    PriorSpec,
    TruncNormalPrior,
)
from irtree.priors import _make_defaults as _make_default_priors
from irtree.progress import ProgressReporter, mean_acceptance

# ===========================================================================
# Public data classes
# ===========================================================================

@dataclass
class MCMCResult:
    """Output of a single MCMC chain.

    Attributes
    ----------
    chain : ndarray [n_samples × d]
        Posterior draws, float32 by default.  Rows are samples; columns are
        parameters in canonical order (see _build_chain_layout docstring).
    param_names : list[str]
        Column names corresponding to chain columns.  Length == d.
    log_lik_trace : ndarray [n_samples]
        Log-likelihood (sum of the R node terms, priors not included) at
        each retained sample, from the likelihood cache.  float64.
    accept_rates : dict[str, float]
        Empirical acceptance rate for each MH block, keyed by block name
        (e.g. ``'a_0'``, ``'theta_mh_0'``, ``'b_mh_0'``, ``'tau_mh_0'``,
        ``'beta_mh_0'``, ``'rho_0'``; the index is the node, dimension or
        group).  Empty for pure Gibbs chains.
    n_warmup : int
        Warm-up length used by the sampler: the fixed length in fixed mode;
        in evidence mode the cap for MH chains, or the burn-in for pure-Gibbs
        chains.  See ``effective_warmup`` for where storage actually began.
    n_iter : int
        Total iterations run (warmup + sampling).
    final_params : dict
        Final parameter state at end of chain.  Useful for warm-starting
        a continuation run.
    rng_state : dict
        State of the numpy Generator at end of chain.  Allows exact
        reproducible continuation.
    warmup_mode : str
        "fixed" or "evidence": the warm-up rule that produced the chain.
    effective_warmup : int
        Iteration at which storage began (fixed mode: n_warmup; evidence
        mode: the first iteration after every MH block froze, or the burn-in
        for a pure-Gibbs chain).
    frozen_at : dict[str, int]
        Freeze iteration of each MH block.  In evidence mode, a value equal
        to the cap means the block never met the stability test.
    """
    chain:               np.ndarray
    param_names:         list[str]
    log_lik_trace: np.ndarray
    accept_rates:        dict[str, float]
    n_warmup:            int
    n_iter:              int
    final_params:        dict
    rng_state:           dict
    #: "fixed" or "evidence" — which warmup rule produced this chain.
    warmup_mode:         str = "fixed"
    #: The iteration at which kept-sample storage began.  In fixed mode this is
    #: n_warmup; in evidence mode it is the first all-frozen iteration (so it
    #: reports the *actual* warmup length the data required).
    effective_warmup:    int = 0
    #: Per-MH-block freeze iteration {block_name: frozen_at_t}.  Empty for pure
    #: Gibbs chains.  In evidence mode a block whose value equals the cap never
    #: satisfied the stability test — a flag that the run was too short or that
    #: block mixes badly.
    frozen_at:           dict[str, int] = field(default_factory=dict)


@dataclass
class _NodeDispatch:
    """Internal per-node dispatch information built once at sampler init."""
    r:                int
    is_polytomous:    bool
    has_guessing:     bool
    has_discrimination: bool
    is_augmentable:   bool
    n_outcomes:       int     # 2 for binary, M for GRM
    is_mixed:         bool    # True for GRMNode/GPCMNode with mixed item formats


# ===========================================================================
# Chain layout builder
# ===========================================================================

def _node_label(nodes_0based) -> str:
    """Display label for a group of 0-based node indices.

    Single node  → "1"        (1-based)
    Shared group → "1&3"      (sorted, 1-based, joined with &)

    Used for all column names so that shared parameters are immediately
    identifiable in chain output, posterior summaries, and diagnostic tables.
    """
    return "&".join(str(r + 1) for r in sorted(nodes_0based))


def _theta_corr_blocks(constraints, D):
    """theta correlation blocks (0-based dims) and their within-block pairs.

    With Constraints(theta_corr=...) the correlation matrix is block-diagonal:
    estimated within each declared block, zero across blocks, dimensions in
    no block independent.  Without it, all D dimensions form one block (the
    joint correlation).  ``pairs`` lists the (d1, d2), d1 < d2, of every
    block in block order: the rho chain columns and adaptation keys follow
    it, so the single-block case reproduces the full D(D-1)/2 layout.
    ``tri`` lists each block's lower triangle including the diagonal
    (InverseWishart Sigma_theta columns).
    """
    con = constraints if constraints is not None else NO_CONSTRAINTS
    if getattr(con, "theta_corr", None):
        blocks = [sorted(b) for b in con.theta_corr_groups_for(D)]
    else:
        blocks = [list(range(D))] if D >= 2 else []
    pairs = [(b[i], b[j]) for b in blocks for i in range(len(b)) for j in range(i + 1, len(b))]
    tri = [(b[i], b[j]) for b in blocks for i in range(len(b)) for j in range(i, len(b))]
    return blocks, pairs, tri


def covariate_column_names(cov_index, data):
    """Chain column names of every covariate coefficient block.

    Returns {"beta": [names of group 0, ...], "kappa": [...], "gamma": [...]},
    in group order.  beta is keyed by person-covariate group (shared theta),
    kappa by item-covariate group (shared b), gamma by pi-covariate group
    (gamma_shared; singletons by default).  A column is named
    {prefix}_{group label}_{covariate name}, or a 1-based column index when
    the matrix has no names; the group label is the '&'-joined 1-based node
    list (e.g. beta_2&3_ses).

    Single source of truth for the chain layout and for posterior.py, which
    must find the same columns to rebuild the linear predictor.
    """
    def cols(prefix, nodes, idx_arr, names_tuple):
        return [f"{prefix}_{_node_label(nodes)}_"
                f"{names_tuple[k] if names_tuple is not None else k + 1}"
                for k in idx_arr]

    return {
        "beta":  [cols("beta", grp, cov_index.x_idx[g], data.X_names)
                  for g, grp in enumerate(cov_index.x_groups)],
        "kappa": [cols("kappa", grp, cov_index.q_idx[g], data.Q_names)
                  for g, grp in enumerate(cov_index.q_groups)],
        "gamma": [cols("gamma", grp, cov_index.w_group_idx[g], data.W_names)
                  for g, grp in enumerate(cov_index.w_groups)],
    }


def hyperparameter_column_names(spec, node_models, priors, constraints=None) -> list[str]:
    """Chain column names of every ESTIMATED hyperparameter, in chain order.

    The same names ``_build_chain_layout`` emits, computed without data:
    sigma2 of dimensions whose variance is estimated (InvGamma / HalfNormal,
    or read from an Inverse-Wishart Sigma), b.variance and a.variance per
    sharing group, the theta / b / a correlation (rho) or covariance (Sigma)
    columns, and the estimated covariate-group variances.  Used by
    ``Priors.free_params`` so the priors report the names the chain uses.
    """
    R = len(node_models)
    con = constraints if constraints is not None else NO_CONSTRAINTS
    eff = _effective_priors(priors, constraints, R)
    dim_to_nodes = spec.dim_to_nodes if spec is not None else [[r] for r in range(R)]
    D = len(dim_to_nodes)
    corr_spec = eff.resolve("theta.correlation", None)
    theta_iw = isinstance(corr_spec, InverseWishartPrior)
    out = []

    for d in range(D):
        var = getattr(eff.resolve("theta", dim_to_nodes[d][0]), "variance", None)
        if theta_iw or isinstance(var, PriorSpec):
            out.append(f"sigma2_{_node_label(dim_to_nodes[d])}")

    b_groups = con.b_groups_for(R)
    for g, rep in enumerate(con.b_representative_nodes(R)):
        pr = eff.resolve("b", rep)
        if isinstance(pr, NormalPrior) and isinstance(pr.variance, InvGammaPrior):
            out.append(f"b.variance_{_node_label(b_groups[g])}")

    a_groups = con.a_groups_for(R)
    for g, rep in enumerate(con.a_representative_nodes(R)):
        if rep is not None and isinstance(getattr(eff.resolve("a", rep), "variance", None), PriorSpec):
            out.append(f"a.variance_{_node_label(a_groups[g])}")

    if not isinstance(corr_spec, FixedCorrelation):
        _blk, pairs, tri = _theta_corr_blocks(constraints, D)
        if theta_iw:
            out += [f"Sigma_theta_{d1 + 1}_{d2 + 1}" for d1, d2 in tri]
        else:
            out += [f"rho_{d1 + 1}_{d2 + 1}" for d1, d2 in pairs]

    for kind, groups, cgs in (("b", b_groups, con.b_corr_groups_for(R)),
                              ("a", a_groups, con.a_corr_groups_for(R))):
        cspec = eff.type_level.get(f"{kind}.correlation", FixedCorrelation(rho=0.0))
        for cg in cgs:
            labels = [_node_label(groups[g]) for g in cg]
            if isinstance(cspec, InverseWishartPrior):
                tag = "&".join(labels)
                out += [f"Sigma_{kind}_{tag}_{i + 1}_{j + 1}"
                        for i in range(len(cg)) for j in range(i, len(cg))]
            elif not isinstance(cspec, FixedCorrelation):
                out += [f"{kind}.rho_{labels[i]}_{labels[j]}"
                        for i in range(len(cg)) for j in range(i + 1, len(cg))]

    gamma_struct = con.gamma_groups_for(R)
    for kind, flag, structure in (("beta", "has_person_covariates", dim_to_nodes),
                                  ("kappa", "has_item_covariates", b_groups),
                                  ("gamma", "has_pi_covariates", gamma_struct)):
        for grp in structure:
            if any(getattr(node_models[r], flag, False) for r in grp):
                if isinstance(getattr(eff.resolve(kind, grp[0]), "variance", None), InvGammaPrior):
                    out.append(f"{kind}.variance_{_node_label(grp)}")
    return out


def _effective_priors(priors: Priors, constraints, R: int) -> Priors:
    """The priors the sampler actually runs with.

    Under an LKJ (or fixed) a.correlation, each correlated a-group whose
    prior is the package default gets an estimated variance (Half-Normal(0,
    0.5) on its sd); see run_mcmc.  Returns ``priors`` itself when nothing
    changes, otherwise an adjusted deep copy.  Shared with fit.to_logit(),
    which rebuilds the chain layout and must see the same a.variance columns.
    """
    _a_corr_spec0 = priors.type_level.get("a.correlation", FixedCorrelation(rho=0.0))
    _con0 = constraints if constraints is not None else NO_CONSTRAINTS
    _a_corr_groups0 = _con0.a_corr_groups_for(R)
    if _a_corr_groups0 and not isinstance(_a_corr_spec0, InverseWishartPrior):
        _a_default = _make_default_priors()["a"]
        _a_groups0 = _con0.a_groups_for(R)
        _auto = [r_ for cg in _a_corr_groups0 for g_ in cg for r_ in _a_groups0[g_]
                 if priors.resolve("a", r_) == _a_default]
        if _auto:
            priors = copy.deepcopy(priors)
            for r_ in _auto:
                priors.node_level[("a", r_)] = TruncNormalPrior(
                    mean=1.0, variance=HalfNormalPrior(scale=0.5), lower=0.0)
    return priors


def _build_chain_layout(
    data:        PreparedData,
    node_models: list[NodeModel],
    priors:      Priors,
    spec=None,
    constraints=None,
    cov_index=None,
) -> tuple[list[str], dict[str, tuple[int, int]]]:
    """Build the canonical chain column names and slice offsets.

    Returns
    -------
    param_names : list[str]
        One name per chain column, in canonical order.
    offsets : dict[str, (start, stop)]
        Maps a logical parameter key to its column slice in the chain.

        Sharing / latent-dim keys (0-based indices):
          'theta_{d}'           latent dim d  → [P] theta columns
          'b_{g}'               b-group g     → [I] or [I*(M-1)] b columns
          'a_{g}'               a-group g     → [I] a columns (2PL/3PL only)
          'c_{g}'               c-group g     → [I] c columns (3PL only)
          'sigma2_{d}'          latent dim d  → 1 sigma2 column
          'b_var_{g}'           b-group g     → 1 b.variance column (InvGammaPrior only)

        Correlation keys:
          'rho'                 → one rho_{d1}_{d2} column per within-block pair
                                  (LKJ/bounded theta.corr; D*(D-1)/2 for one block)
                                  OR each block's lower-triangle Sigma_theta_{d1}_{d2}
                                  columns (InvWishart; D*(D+1)/2 for one block)
          'b_corr_Sigma_{idx}'  → G*(G+1)/2 Sigma_b columns (InvWishart b.corr group idx)
          'b_corr_rho_{idx}'    → G*(G-1)/2 b.rho columns   (LKJ b.corr group idx)
          'a_corr_Sigma_{idx}'  → same pattern for a.corr
          'a_corr_rho_{idx}'

        Covariate keys (0-based group index in key; 1-based labels in names):
          'beta_{g}', 'kappa_{g}', 'gamma_{g}'   → coefficient columns of group g
          'beta_var_{g}', ...   → 1 column: the group's estimated variance
          'X_missing_{p}'       → columns for missing features of person p (0-based)
          'Q_missing_{i}'       → columns for missing features of item i
          'W_missing_{p}_{i}'   → columns for missing features of cell (p,i)
    """
    P, I, R = data.n_persons, data.n_items, data.n_nodes
    names   = []
    offsets = {}

    # --- theta: [P] per latent dimension ---
    # When spec.node_to_latent is None, D == R and each node has its own dim.
    # When nodes share a latent variable, theta has fewer blocks than nodes.
    dim_to_nodes = spec.dim_to_nodes if spec is not None else [[r] for r in range(R)]
    D = len(dim_to_nodes)
    for d in range(D):
        start     = len(names)
        dim_label = _node_label(dim_to_nodes[d])
        for p in range(1, P + 1):
            names.append(f"theta_{dim_label}_{p}")
        offsets[f"theta_{d}"] = (start, len(names))

    # --- b: one block per sharing group (representative node names the columns) ---
    _con = constraints if constraints is not None else NO_CONSTRAINTS
    b_groups      = _con.b_groups_for(R)
    b_reps        = _con.b_representative_nodes(R)
    for g, (group, rep_r) in enumerate(zip(b_groups, b_reps, strict=True)):
        nm          = node_models[rep_r]
        start       = len(names)
        group_label = _node_label(group)

        # Mixed node: binary items get one column, polytomous items get M-1
        # columns each, emitted in item-index order (not group-membership order)
        # so the chain is interpretable item-by-item.
        if isinstance(nm, (GRMNode, GPCMNode)) and nm.is_mixed:
            bin_set = set(nm.bin_idx.tolist())
            for i in range(I):
                i1 = i + 1           # 1-based item label
                if i in bin_set:
                    names.append(f"b_{group_label}_{i1}")
                else:
                    for k in range(1, nm.M_poly):
                        names.append(f"b_{group_label}_{i1}_tau{k}")
        elif nm.n_outcomes > 2:
            M = nm.n_outcomes
            for i in range(1, I + 1):
                for k in range(1, M):
                    names.append(f"b_{group_label}_{i}_thresh_{k}")
        else:
            for i in range(1, I + 1):
                names.append(f"b_{group_label}_{i}")
        offsets[f"b_{g}"] = (start, len(names))

    # --- a: one block per sharing group among discrimination nodes ---
    a_nodes       = set(priors._param_nodes.get("a", []))
    a_groups      = _con.a_groups_for(R)
    a_reps        = _con.a_representative_nodes(R)
    for g, (group, rep_r) in enumerate(zip(a_groups, a_reps, strict=True)):
        if rep_r not in a_nodes:
            continue
        start       = len(names)
        group_label = _node_label(group)
        for i in range(1, I + 1):
            names.append(f"a_{group_label}_{i}")
        offsets[f"a_{g}"] = (start, len(names))

    # --- c: one block per sharing group among guessing nodes ---
    c_nodes       = set(priors._param_nodes.get("c", []))
    c_groups      = _con.c_groups_for(R)
    c_reps        = _con.c_representative_nodes(R)
    for g, (group, rep_r) in enumerate(zip(c_groups, c_reps, strict=True)):
        if rep_r not in c_nodes:
            continue
        start       = len(names)
        group_label = _node_label(group)
        for i in range(1, I + 1):
            names.append(f"c_{group_label}_{i}")
        offsets[f"c_{g}"] = (start, len(names))

    # --- sigma2: one scalar per latent dimension ---
    for d in range(D):
        start     = len(names)
        dim_label = _node_label(dim_to_nodes[d])
        names.append(f"sigma2_{dim_label}")
        offsets[f"sigma2_{d}"] = (start, start + 1)

    # --- b.variance: one column per b-sharing group with an estimated variance ---
    # Nodes in the same b-group share one b array → they share one b.variance.
    for g, rep_r in enumerate(b_reps):
        b_prior_r = priors.resolve("b", rep_r)
        if isinstance(b_prior_r, NormalPrior) and isinstance(b_prior_r.variance, InvGammaPrior):
            start = len(names)
            names.append(f"b.variance_{_node_label(b_groups[g])}")
            offsets[f"b_var_{g}"] = (start, start + 1)

    # --- a.variance: one column per a-group whose (bounded) discrimination
    # prior carries a forced hierarchical variance (MH-estimated, not Gibbs) ---
    for g, rep_r in enumerate(a_reps):
        if rep_r is None:
            continue
        a_prior_r = priors.resolve("a", rep_r)
        if isinstance(getattr(a_prior_r, "variance", None), PriorSpec):
            start = len(names)
            names.append(f"a.variance_{_node_label(a_groups[g])}")
            offsets[f"a_var_{g}"] = (start, start + 1)

    # --- rho: D(D-1)/2 scalars if theta.correlation is estimated ---
    corr_spec = priors.resolve("theta.correlation", None)
    if not isinstance(corr_spec, FixedCorrelation):
        start = len(names)
        _blk, _pairs, _tri = _theta_corr_blocks(constraints, D)
        if isinstance(corr_spec, InverseWishartPrior):
            # Covariance of each block: lower triangle including diagonal
            for d1, d2 in _tri:
                names.append(f"Sigma_theta_{d1+1}_{d2+1}")
        else:
            for d1, d2 in _pairs:
                names.append(f"rho_{d1+1}_{d2+1}")
        offsets["rho"] = (start, len(names))

    # --- b.corr_matrix: rho or Sigma columns for cross-node b-correlation ---
    b_corr_spec = priors.type_level.get("b.correlation", FixedCorrelation(rho=0.0))
    b_corr_groups_layout = _con.b_corr_groups_for(R)
    for cg_idx, cg in enumerate(b_corr_groups_layout):
        # Validate G-dependent rules now that group size is known
        _G_b = len(cg)
        priors._validate_corr_prior("b.correlation", b_corr_spec, G=_G_b)
        G_cg = len(cg)
        start = len(names)
        # One representative label per correlation group (sorted group indices)
        cg_labels = [_node_label(b_groups[g]) for g in cg]
        cg_tag = "&".join(cg_labels)
        if isinstance(b_corr_spec, InverseWishartPrior):
            for i in range(G_cg):
                for j in range(i, G_cg):
                    names.append(f"Sigma_b_{cg_tag}_{i+1}_{j+1}")
            offsets[f"b_corr_Sigma_{cg_idx}"] = (start, len(names))
        elif not isinstance(b_corr_spec, FixedCorrelation):
            # LKJPrior: off-diagonal pairs only
            for i in range(G_cg):
                for j in range(i + 1, G_cg):
                    names.append(f"b.rho_{cg_labels[i]}_{cg_labels[j]}")
            offsets[f"b_corr_rho_{cg_idx}"] = (start, len(names))
        # FixedCorrelation: no chain columns

    # --- a.corr_matrix: rho or Sigma columns for cross-node a-correlation ---
    a_corr_spec = priors.type_level.get("a.correlation", FixedCorrelation(rho=0.0))
    a_corr_groups_layout = _con.a_corr_groups_for(R)
    for cg_idx, cg in enumerate(a_corr_groups_layout):
        # Validate G-dependent rules now that group size is known
        _G_a = len(cg)
        priors._validate_corr_prior("a.correlation", a_corr_spec, G=_G_a)
        G_cg = len(cg)
        start = len(names)
        cg_labels = [_node_label(a_groups[g]) for g in cg]
        cg_tag = "&".join(cg_labels)
        if isinstance(a_corr_spec, InverseWishartPrior):
            for i in range(G_cg):
                for j in range(i, G_cg):
                    names.append(f"Sigma_a_{cg_tag}_{i+1}_{j+1}")
            offsets[f"a_corr_Sigma_{cg_idx}"] = (start, len(names))
        elif not isinstance(a_corr_spec, FixedCorrelation):
            for i in range(G_cg):
                for j in range(i + 1, G_cg):
                    names.append(f"a.rho_{cg_labels[i]}_{cg_labels[j]}")
            offsets[f"a_corr_rho_{cg_idx}"] = (start, len(names))

    # ------------------------------------------------------------------
    # Covariate coefficient blocks.
    # beta is keyed per latent-dimension group (shared theta), kappa per
    # b-sharing group (shared b), gamma per pi-covariate group (gamma_shared;
    # one group per node by default). Widths come from the resolved
    # selection (cov_index), NOT the full matrix, so a node that uses 2 of 5
    # X columns emits exactly 2 beta columns. Names come from
    # covariate_column_names() (shared with posterior.py):
    # {beta|kappa|gamma}_{group label}_{covariate name or 1-based column}.
    #
    # Offset keys:
    #   'beta_{g}'   g = index into cov_index.x_groups
    #   'kappa_{g}'  g = index into cov_index.q_groups
    #   'gamma_{g}'  g = index into cov_index.w_groups
    # ------------------------------------------------------------------
    if cov_index is not None:
        for kind, blocks in covariate_column_names(cov_index, data).items():
            for g, cols in enumerate(blocks):
                start = len(names)
                names.extend(cols)
                offsets[f"{kind}_{g}"] = (start, len(names))
        # Estimated coefficient-group variances (InvGamma hyperprior), one
        # column per group: {beta|kappa|gamma}.variance_{group label}.
        for kind, groups in (("beta", cov_index.x_groups), ("kappa", cov_index.q_groups),
                             ("gamma", cov_index.w_groups)):
            for g, grp in enumerate(groups):
                if isinstance(getattr(priors.resolve(kind, grp[0]), "variance", None),
                              InvGammaPrior):
                    offsets[f"{kind}_var_{g}"] = (len(names), len(names) + 1)
                    names.append(f"{kind}.variance_{_node_label(grp)}")

    # --- X_missing: one column per missing feature per missing person ---
    # Only emitted when data.n_missing_X > 0.
    # Column names: X_missing_{p+1}_{k+1}  (1-based; only missing feature indices)
    # Offset key:   "X_missing_{p}"         (0-based person index)
    if data.X is not None and data.n_missing_X > 0:
        missing_persons = np.where(~data.X_observed)[0]
        for p in missing_persons:
            start     = len(names)
            feat_miss = np.where(~data.X_feature_observed[p])[0]
            for k in feat_miss:
                names.append(f"X_missing_{p + 1}_{k + 1}")
            offsets[f"X_missing_{p}"] = (start, len(names))

    # --- Q_missing: one column per missing feature per missing item ---
    if data.Q is not None and data.n_missing_Q > 0:
        missing_items = np.where(~data.Q_observed)[0]
        for i in missing_items:
            start     = len(names)
            feat_miss = np.where(~data.Q_feature_observed[i])[0]
            for k in feat_miss:
                names.append(f"Q_missing_{i + 1}_{k + 1}")
            offsets[f"Q_missing_{i}"] = (start, len(names))

    # --- W_missing: one column per missing feature per missing (p,i) cell ---
    if data.W is not None and data.n_missing_W > 0:
        missing_cells = np.argwhere(~data.W_observed)
        for cell in missing_cells:
            p, i  = int(cell[0]), int(cell[1])
            start = len(names)
            feat_miss = np.where(~data.W_feature_observed[p, i])[0]
            for k in feat_miss:
                names.append(f"W_missing_{p + 1}_{i + 1}_{k + 1}")
            offsets[f"W_missing_{p}_{i}"] = (start, len(names))

    return names, offsets


# ===========================================================================
# Parameter initialisation (warm start)
# ===========================================================================

def _cov_matrices(params, data):
    """Current (X, Q, W) covariate arrays: the working copies when missing
    values are being imputed, otherwise the data.

    Under full imputation every missing entry holds its current
    draw, so all cells of the returned arrays enter the likelihood; no
    observed-cell mask is applied anywhere downstream.
    """
    X = params.get("X_working")
    Q = params.get("Q_working")
    if Q is None:
        Q = data.Q
    W = params.get("W_working")
    if W is None:
        W = data.W
    return X, Q, W


def _node_offset(r, params, cov_index, data, include=("x", "q", "w")):
    """Compute node r's covariate offset matrix [P, I], or None.

    offset[p, i] = X_p[x_idx] @ beta[x_group(r)]        (person, broadcast over i)
                 + Q_i[q_idx] @ kappa[q_group(r)]        (item, broadcast over p)
                 + W_{pi}[w_idx] @ gamma[w_group(r)]     (cell)

    X, Q and W are the CURRENT covariate values (see _cov_matrices): the
    imputed draws enter the likelihood exactly like observed values.

    ``include`` selects which of the three terms to add, so that the residual
    for one coefficient (or one covariate type's imputation) leaves out its
    own term: include=("q", "w") is "everything except beta", and so on.
    Returns None when the node uses none of the included types.

    This is the single source of truth for the covariate offset: the latent
    draws, the theta/b residualization, the coefficient draws, the imputation
    and the likelihood cache all call it.
    """
    if cov_index is None:
        return None

    P, I = data.n_persons, data.n_items
    X, Q, W = _cov_matrices(params, data)
    g_x = cov_index.x_node_to_group[r] if "x" in include else None
    g_q = cov_index.q_node_to_group[r] if "q" in include else None
    g_w = cov_index.w_node_to_group[r] if "w" in include else None

    has_x = g_x is not None and X is not None
    has_q = g_q is not None and Q is not None
    has_w = g_w is not None and W is not None

    if not (has_x or has_q or has_w):
        return None

    offset = np.zeros((P, I))
    if has_x:
        offset += (X[:, cov_index.x_idx[g_x]] @ params["beta"][g_x])[:, np.newaxis]
    if has_q:
        offset += (Q[:, cov_index.q_idx[g_q]] @ params["kappa"][g_q])[np.newaxis, :]
    if has_w:
        offset += W[:, :, cov_index.w_group_idx[g_w]] @ params["gamma"][g_w]
    return offset


def _init_params(
    data:        PreparedData,
    node_models: list[NodeModel],
    priors:      Priors,
    warm_start:  bool,
    rng:         np.random.Generator,
    spec=None,
    constraints=None,
    cov_index=None,
) -> dict:
    """Initialise all parameters to their starting values.

    Returns a dict with the following keys:

    Core IRT parameters
        'theta'        : list[ndarray [P]]    length = n_latent_dims; start at 0
        'b'            : list[ndarray]        length = n_b_groups
                         [I] for binary/1PL; [I, M-1] for GRM;
                         NodeBParams for mixed-format nodes.
                         warm_start=True initialises from empirical difficulty.
        'a'            : list[ndarray|None]   length = n_a_groups; None if no discrimination
        'c'            : list[ndarray|None]   length = n_c_groups; None if no guessing
        'sigma2'       : list[float]          length = n_latent_dims; start at 1.0
                         (run_chain then sets fixed variances to their value)
        'log_sigma'    : list[float|None]     non-None only under a HalfNormal
                                              variance prior (MH on log sd)
        'b.variance'   : list[float]          length = n_b_groups
                         Free InvGammaPrior entries start at 1.0; fixed entries
                         hold the constant from the prior specification.
        'a.variance'   : list[float]          length = n_a_groups; 1.0 when
                         estimated, else the a prior's fixed variance (0.5)
        'a.variance.estimated' : list[bool]   length = n_a_groups

    Correlation parameters
        'corr_matrix'  : ndarray [D × D], block-diagonal over the
                         Constraints(theta_corr=...) blocks (one block of all
                         D dimensions when none is given).
                         Correlation matrix R under LKJ / bounded theta.correlation.
                         Full covariance matrix Σ under InverseWishartPrior
                         (build_sigma() is bypassed in that case; per-dim sigma2
                         is kept consistent by reading the diagonal of Σ after each
                         InvWishart draw).
                         Starts at the identity when estimated; a fixed
                         scalar rho fills the off-diagonal within each block.
        'b.corr_matrix': list[ndarray [G×G]]  one per b-correlation group.
                         Identity at the start when estimated (LKJ or
                         InverseWishartPrior); compound-symmetry when
                         FixedCorrelation(scalar); the given matrix for a
                         fixed matrix.  Empty list when no b_corr is declared.
        'a.corr_matrix': list[ndarray [G×G]]  same structure for a-correlation groups.

    Covariate parameters (empty lists / None when no node uses the array)
        'beta'         : list[ndarray]  one per x-group (cov_index.x_groups);
                         width = the group's selected X columns; start at 0
        'kappa'        : list[ndarray]  one per q-group (cov_index.q_groups)
        'gamma'        : list[ndarray]  one per w-group (cov_index.w_groups)
        'beta.variance' / 'kappa.variance' / 'gamma.variance'
                       : list[float]    one per group, matching the
                         coefficient lists; start at 1.0
        'X_working'    : ndarray [P, K_x] or None (None when X is absent)
                         Observed cells frozen; missing cells start at 0.0
                         and are updated each iteration (Gibbs given Z, or
                         MH when a logit node uses the covariate).
        'Q_working'    : ndarray [I, K_q] or None (None when Q has no
                         missing cells); same rules as X_working.
        'W_working'    : ndarray [P, I, K_w] or None (None when W has no
                         missing cells); same rules as X_working.
    """
    P, I, R = data.n_persons, data.n_items, data.n_nodes

    # --- theta: one array per latent dimension ---
    dim_to_nodes = spec.dim_to_nodes if spec is not None else [[r] for r in range(R)]
    D             = len(dim_to_nodes)
    _con          = constraints if constraints is not None else NO_CONSTRAINTS
    theta = [np.zeros(P) for _ in range(D)]

    # --- b: one array per sharing group (representative node determines shape) ---
    b_reps_init   = _con.b_representative_nodes(R)
    b = []
    for rep_r in b_reps_init:
        nm = node_models[rep_r]
        r  = rep_r   # warm-start uses representative node's empirical proportions

        # ----------------------------------------------------------------
        # Mixed node: construct NodeBParams with separate binary and
        # polytomous arrays.
        # ----------------------------------------------------------------
        if isinstance(nm, (GRMNode, GPCMNode)) and nm.is_mixed:
            # Binary sub-group: scalar thresholds via probit warm-start
            if warm_start:
                y_r = data.Y_star[r][:, nm.bin_idx]
                T_r = data.T[r][:, nm.bin_idx]
                n_r  = np.sum(T_r, axis=0).astype(float)
                p0_r = np.sum((y_r == 0) & T_r, axis=0).astype(float)
                prop = np.where(n_r > 0, p0_r / np.maximum(n_r, 1), 0.5)
                prop = np.clip(prop, 0.05, 0.95)
                b_bin = _Phi_inv(prop)     # [I_bin]
            else:
                b_bin = np.zeros(len(nm.bin_idx))

            # Polytomous sub-group: evenly spaced quantile warm-start
            M_poly = nm.M_poly
            I_poly = len(nm.poly_idx)
            if warm_start:
                thresholds = np.zeros((I_poly, M_poly - 1))
                y_r_poly = data.Y_star[r][:, nm.poly_idx]
                T_r_poly = data.T[r][:, nm.poly_idx]
                for ii in range(I_poly):
                    mask = T_r_poly[:, ii]
                    y_i  = y_r_poly[mask, ii]
                    if len(y_i) > 0:
                        for k in range(M_poly - 1):
                            p_k = np.mean(y_i <= k)
                            p_k = np.clip(p_k, 0.01, 0.99)
                            thresholds[ii, k] = _Phi_inv(p_k)
                    for k in range(1, M_poly - 1):
                        thresholds[ii, k] = max(thresholds[ii, k],
                                                thresholds[ii, k - 1] + 0.1)
                b_poly = thresholds   # [I_poly, M_poly-1]
            else:
                b_poly = np.linspace(-1.5, 1.5, M_poly - 1)[None, :].repeat(I_poly, 0)

            b.append(NodeBParams(
                b_binary  = b_bin,
                b_poly    = b_poly,
                bin_idx   = nm.bin_idx,
                poly_idx  = nm.poly_idx,
                M_poly    = M_poly,
            ))

        # ----------------------------------------------------------------
        # Homogeneous polytomous (GRM/GPCM, uniform M)
        # ----------------------------------------------------------------
        elif nm.n_outcomes > 2:
            M = nm.n_outcomes
            if warm_start:
                # Space thresholds evenly based on empirical category proportions
                thresholds = np.zeros((I, M - 1))
                y_r = data.Y_star[r]
                T_r = data.T[r]
                for i in range(I):
                    mask = T_r[:, i]
                    y_i  = y_r[mask, i]
                    if len(y_i) > 0:
                        for k in range(M - 1):
                            p_k = np.mean(y_i <= k)
                            p_k = np.clip(p_k, 0.01, 0.99)
                            thresholds[i, k] = _Phi_inv(p_k)
                    # enforce ordering
                    for k in range(1, M - 1):
                        thresholds[i, k] = max(thresholds[i, k],
                                               thresholds[i, k - 1] + 0.1)
                b.append(thresholds)
            else:
                b.append(np.linspace(-1.5, 1.5, M - 1)[None, :].repeat(I, 0))
        else:
            if warm_start:
                # Proportion of 0-responses (the harder outcome) → probit scale
                y_r  = data.Y_star[r]
                T_r  = data.T[r]
                n_r  = np.sum(T_r, axis=0).astype(float)
                p0_r = np.sum((y_r == 0) & T_r, axis=0).astype(float)
                prop = np.where(n_r > 0, p0_r / np.maximum(n_r, 1), 0.5)
                prop = np.clip(prop, 0.05, 0.95)
                b.append(_Phi_inv(prop))
            else:
                b.append(np.zeros(I))

    # --- a: one entry per group; None if group has no discrimination ---
    a_nodes_set   = set(priors._param_nodes.get("a", []))
    a_reps_init   = _con.a_representative_nodes(R)
    a = [np.ones(I) if r in a_nodes_set else None
         for r in a_reps_init]

    # --- c: one entry per group; None if group has no guessing ---
    c_nodes_set   = set(priors._param_nodes.get("c", []))
    c_reps_init   = _con.c_representative_nodes(R)
    c = []
    for rep_r in c_reps_init:
        if rep_r in c_nodes_set:
            c_prior = priors.resolve("c", rep_r)
            a_c = getattr(c_prior, "alpha", 5.0)
            b_c = getattr(c_prior, "beta",  17.0)
            c.append(np.full(I, a_c / (a_c + b_c)))
        else:
            c.append(None)

    # --- sigma2: one per latent dimension ---
    # When nodes share a latent dim we use the prior from the first node
    # in that dim (all sharing nodes must have the same prior by design).
    sigma2    = [1.0 for _ in range(D)]
    log_sigma = []
    for d in range(D):
        r_representative = dim_to_nodes[d][0]
        th_prior = priors.resolve("theta", r_representative)
        if isinstance(getattr(th_prior, "variance", None), HalfNormalPrior):
            log_sigma.append(0.0)
        else:
            log_sigma.append(None)

    # --- b.variance: one estimated variance per b-sharing group ---
    # Nodes in the same group share one b array → one shared variance.
    # Free (InvGammaPrior) entries start at 1.0; fixed entries hold the constant.
    b_variance = []
    for rep_r in b_reps_init:
        b_prior_r = priors.resolve("b", rep_r)
        if isinstance(b_prior_r, NormalPrior):
            var = b_prior_r.variance
            b_variance.append(1.0 if isinstance(var, InvGammaPrior) else float(var))
        else:
            b_variance.append(1.0)

    # --- a.variance: hierarchical discrimination variance (MH-updated) ---
    # Only populated when the user has forced an estimated variance onto the
    # (bounded) discrimination prior.  A float variance leaves this at the
    # fixed value and the MH block never runs.  See priors._check_variance_
    # conjugacy: the truncation constant makes this non-conjugate, so unlike
    # b/beta/kappa/gamma it cannot use gibbs_variance_invgamma.
    a_variance = []
    a_variance_estimated = []
    for rep_r in a_reps_init:
        a_prior_r = priors.resolve("a", rep_r) if rep_r is not None else None
        var = getattr(a_prior_r, "variance", None)
        if isinstance(var, PriorSpec):
            a_variance.append(1.0)
            a_variance_estimated.append(True)
        else:
            a_variance.append(float(var) if var is not None else 0.5)
            a_variance_estimated.append(False)

    # --- correlation matrix: identity (independent default) ---
    # Size is D×D (one entry per latent dimension pair)
    corr_matrix = np.eye(D)
    corr_spec   = priors.resolve("theta.correlation", None)
    if isinstance(corr_spec, FixedCorrelation):
        rho = corr_spec.rho
        if np.isscalar(rho):
            # A scalar rho applies within each correlation block only.
            for d1, d2 in _theta_corr_blocks(constraints, D)[1]:
                corr_matrix[d1, d2] = corr_matrix[d2, d1] = float(rho)
        else:
            corr_matrix = np.asarray(rho)

    # --- Covariate coefficients ---
    # Keyed to match the parameter each explains: beta per latent-dimension
    # group (like theta), kappa per b-sharing group (like b), gamma per node.
    # Each vector's width is the group's SELECTED covariate count, not the
    # full matrix width. Initialized to zero (the prior's highest-density
    # region for centered covariates — no warm start for regression coefs).
    if cov_index is not None:
        beta  = [np.zeros(len(cov_index.x_idx[g])) for g in range(len(cov_index.x_groups))]
        kappa = [np.zeros(len(cov_index.q_idx[g])) for g in range(len(cov_index.q_groups))]
        gamma = [np.zeros(len(cov_index.w_group_idx[g])) for g in range(len(cov_index.w_groups))]
    else:
        beta, kappa, gamma = [], [], []

    # --- Covariate coefficient variance hyperparameters ---
    # Start at 1.0 (prior mode for InvGamma(1,1)); fixed-variance cases
    # hold the constant (never updated). Keyed the same way as the coefs
    # (beta per latent dim, kappa per b-group, gamma per gamma-group).
    beta_variance  = [1.0 for _ in beta]
    kappa_variance = [1.0 for _ in kappa]
    gamma_variance = [1.0 for _ in gamma]

    # --- X_working: working copy of person covariate matrix ---
    # Observed CELLS are never modified.  Missing cells are updated each
    # iteration by gibbs_X_missing and initialized here to 0.0 (the prior
    # mean for centered covariates).  Only the missing cells are seeded: the
    # observed features of a partially-missing person are data and are held
    # fixed by gibbs_X_missing while it draws the missing ones.
    if data.X is not None:
        X_working = data.X.copy()
        if data.n_missing_X > 0:
            X_working[~data.X_feature_observed] = 0.0
    else:
        X_working = None

    # --- b.corr_matrix: one covariance/correlation matrix per b-corr group ---
    b_corr_groups_init = _con.b_corr_groups_for(R)
    b_corr_matrices = []
    b_corr_spec_init = priors.type_level.get("b.correlation", FixedCorrelation(rho=0.0))
    for cg in b_corr_groups_init:
        G_cg = len(cg)
        if isinstance(b_corr_spec_init, FixedCorrelation):
            rho = b_corr_spec_init.rho
            if np.isscalar(rho):
                C = float(rho) * np.ones((G_cg, G_cg)) + (1 - float(rho)) * np.eye(G_cg)
            else:
                C = np.asarray(rho, dtype=np.float64)
            b_corr_matrices.append(C)
        else:
            # InverseWishartPrior or LKJPrior: start at identity
            b_corr_matrices.append(np.eye(G_cg))

    # --- a.corr_matrix: one covariance/correlation matrix per a-corr group ---
    a_corr_groups_init = _con.a_corr_groups_for(R)
    a_corr_matrices = []
    a_corr_spec_init = priors.type_level.get("a.correlation", FixedCorrelation(rho=0.0))
    for cg in a_corr_groups_init:
        G_cg = len(cg)
        if isinstance(a_corr_spec_init, FixedCorrelation):
            rho = a_corr_spec_init.rho
            if np.isscalar(rho):
                C = float(rho) * np.ones((G_cg, G_cg)) + (1 - float(rho)) * np.eye(G_cg)
            else:
                C = np.asarray(rho, dtype=np.float64)
            a_corr_matrices.append(C)
        else:
            a_corr_matrices.append(np.eye(G_cg))

    # --- Q_working: working copy of item covariates with missing items to be sampled ---
    Q_working: np.ndarray | None = None
    if data.Q is not None and data.n_missing_Q > 0:
        Q_working = data.Q.copy()
        # Seed missing cells only; observed features of a partially-missing
        # item are kept (see X_working above).
        Q_working[~data.Q_feature_observed] = 0.0

    # --- W_working: working copy of person×item covariates ---
    W_working: np.ndarray | None = None
    if data.W is not None and data.n_missing_W > 0:
        W_working = data.W.copy()
        # Seed missing feature values only; observed features of a
        # partially-missing (p, i) cell are kept (see X_working above).
        W_working[~data.W_feature_observed] = 0.0

    return dict(
        theta=theta,
        b=b,
        a=a,
        c=c,
        sigma2=sigma2,
        log_sigma=log_sigma,
        **{"b.variance": b_variance},
        **{"a.variance": a_variance},
        **{"a.variance.estimated": a_variance_estimated},
        corr_matrix=corr_matrix,
        **{"b.corr_matrix": b_corr_matrices},
        **{"a.corr_matrix": a_corr_matrices},
        beta=beta,
        kappa=kappa,
        gamma=gamma,
        **{"beta.variance": beta_variance},
        **{"kappa.variance": kappa_variance},
        **{"gamma.variance": gamma_variance},
        X_working=X_working,
        Q_working=Q_working,
        W_working=W_working,
    )


# ===========================================================================
# Adaptation state builder
# ===========================================================================

def _build_adapt_states(
    node_models: list[NodeModel],
    priors:      Priors,
    n_warmup:    int,
    R:           int,
    D:           int = 0,
    constraints=None,
    P:           int = 0,
    I:           int = 0,
    evidence:    bool = False,
    dim_to_nodes: list[list[int]] | None = None,
) -> dict:
    """Build an AdaptState for every MH block that will be used.

    Returns immediately with an empty dict when n_warmup == 0 (pure Gibbs
    chains need no adaptation).

    Returns a dict keyed by block name:
        'a_{r}'              one per node r with discrimination whose a prior
                             uses MH (LogNormal, or unbounded NormalPrior);
                             bounded NormalPrior and TruncNormalPrior use
                             Gibbs at probit nodes and are excluded
        'a_mh_{r}'           one per non-augmentable node r with discrimination
                             (logit 2PL/3PL, GPCM): item-wise MH on log a
        'c_mh_{r}'           one per non-augmentable node r with guessing
                             (logit 3PL): item-wise MH on logit c
        'log_sigma_{d}'      one per latent dim d whose first node has a
                             HalfNormal theta variance prior
        'rho_{idx}'          one per correlated pair inside the theta_corr
                             blocks when theta.correlation is LKJ or a bounded
                             scalar prior; absent under InverseWishartPrior
                             (Gibbs, no adaptation)
        'b_rho_{cg}_{idx}'   for b-correlation groups with LKJPrior
        'a_rho_{cg}_{idx}'   for a-correlation groups with LKJPrior

    run_chain adds the blocks that need the dimension, group and covariate
    maps it builds:
        'theta_mh_{d}'       person-wise MH, dims with a non-augmentable node
        'b_mh_{g}'           item-wise MH, b-groups with a non-augmentable
                             member (binary part for mixed nodes)
        'bpoly_mh_{g}'       item-wise MH, polytomous thresholds of a mixed
                             b-group with a non-augmentable member
        'bbin_mh_{g}'        binary part of a pooled mixed GRM b-group with a
                             non-augmentable member
        'tau_mh_{g}'         Cowles threshold step, b-groups with a GRM member
        '{beta,kappa,gamma}_mh_{g}'  coefficient groups with a
                             non-augmentable member
        '{X,Q,W}_missing_mh' missing covariate units used by a
                             non-augmentable node
        'a_shared_{g}'       pooled MH for a multi-node a-group (non-Gibbs a
                             prior or a non-augmentable member)
        'a_corr_mh_{g}'      a-groups in an a_corr group with a
                             non-augmentable member
        'a.variance_{g}'     estimated a-group variances (MH on log v)

    A chain with no MH block (probit nodes, InvGamma variances,
    InverseWishart or fixed correlations) gets an empty dict.
    """
    if n_warmup == 0:
        return {}

    states = {}

    # Discrimination a_r — only for nodes using MH (LogNormal or unbounded Normal).
    # Bounded Normal / TruncNormal nodes use exact Gibbs and need no adapt state.
    for r in priors._param_nodes.get("a", []):
        a_pr = priors.resolve("a", r)
        if not _a_prior_is_gibbs(a_pr):
            # a_i are updated PER ITEM (scalar), so tune to the scalar-optimal
            # acceptance ~0.44, not the joint-block 0.234.
            states[f"a_{r}"] = make_adapt_state(n_warmup, target=target_accept(1), evidence=evidence)

    # HalfNormal sigma2: one block per latent DIMENSION d, from the prior of
    # the dimension's first node (as in _init_params and the sigma2 update),
    # so nodes sharing a dimension share the block.
    _d2n = dim_to_nodes if dim_to_nodes is not None else [[r] for r in range(R)]
    for d, nodes_d in enumerate(_d2n):
        th_prior = priors.resolve("theta", nodes_d[0])
        if isinstance(getattr(th_prior, "variance", None), HalfNormalPrior):
            # scalar variance (HalfNormal) -> dim 1
            states[f"log_sigma_{d}"] = make_adapt_state(n_warmup, target=target_accept(1), evidence=evidence)

    # theta / b of non-augmented (logit / GPCM) nodes: per-person and
    # per-item MH blocks, registered in run_chain (theta_mh_{d}, b_mh_{g}),
    # where the dimension and b-group maps are known.

    # Correlation MH — theta
    corr_spec = priors.resolve("theta.correlation", None)
    if not isinstance(corr_spec, FixedCorrelation) and not isinstance(corr_spec, InverseWishartPrior):
        n_pairs = len(_theta_corr_blocks(constraints, D)[1])
        for idx in range(n_pairs):
            # each rho pair is proposed as a scalar (Fisher-z RW) -> dim 1
            states[f"rho_{idx}"] = make_adapt_state(n_warmup, target=target_accept(1), evidence=evidence)

    # b.correlation MH — LKJ or a bounded scalar prior (InvWishart is Gibbs,
    # Fixed has none)
    b_corr_spec = priors.type_level.get("b.correlation", FixedCorrelation(rho=0.0))
    if not isinstance(b_corr_spec, (FixedCorrelation, InverseWishartPrior)) \
            and constraints is not None:
        b_corr_groups_adapt = constraints.b_corr_groups_for(R)
        for cg_idx, cg in enumerate(b_corr_groups_adapt):
            G_cg = len(cg)
            n_pairs = G_cg * (G_cg - 1) // 2
            for idx in range(n_pairs):
                states[f"b_rho_{cg_idx}_{idx}"] = make_adapt_state(n_warmup, target=target_accept(1), evidence=evidence)

    # a.correlation MH — LKJ or a bounded scalar prior
    a_corr_spec = priors.type_level.get("a.correlation", FixedCorrelation(rho=0.0))
    if not isinstance(a_corr_spec, (FixedCorrelation, InverseWishartPrior)) \
            and constraints is not None:
        a_corr_groups_adapt = constraints.a_corr_groups_for(R)
        for cg_idx, cg in enumerate(a_corr_groups_adapt):
            G_cg = len(cg)
            n_pairs = G_cg * (G_cg - 1) // 2
            for idx in range(n_pairs):
                states[f"a_rho_{cg_idx}_{idx}"] = make_adapt_state(n_warmup, target=target_accept(1), evidence=evidence)

    # Discrimination / guessing on NON-augmentable (logit) nodes.  These are
    # estimated by MH directly against the node likelihood (there is no
    # augmented Z), so they get their own blocks, distinct from the augmentable
    # a_{r} path.  Per-item (or shared-scalar) updates → scalar-optimal target.
    for r in range(R):
        nm = node_models[r]
        if not nm.augmentable and nm.has_discrimination:
            states[f"a_mh_{r}"] = make_adapt_state(
                n_warmup, target=target_accept(1), evidence=evidence)
        if not nm.augmentable and nm.has_guessing:
            states[f"c_mh_{r}"] = make_adapt_state(
                n_warmup, target=target_accept(1), evidence=evidence)

    return states


# ===========================================================================
# Inline MH helpers
# ===========================================================================

def _mh_discrimination(
    a_current:  np.ndarray,
    Z:          np.ndarray,
    theta:      np.ndarray,
    b:          np.ndarray,
    T:          np.ndarray,
    row_idx:    np.ndarray,
    col_idx:    np.ndarray,
    log_prior_fn,
    step_size:  float,
    rng:        np.random.Generator,
    offset:     np.ndarray | None = None,
    common:     bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Log-normal random-walk MH for item discrimination a_r (2PL/3PL).

    ``common=True``: one slope shared by all items, so one proposal for all
    of them, accepted or rejected on the node's total augmented
    log-likelihood with the prior evaluated once.  The returned accept array
    then holds the same value for every item.

    PER-ITEM updates.  The a_i are conditionally independent given Z, so this
    proposes all I on the log scale but accepts/rejects EACH ITEM separately
    (a Metropolis-within-Gibbs sweep over the item axis).  Returns the updated
    a vector and a boolean accept array [I]; callers feed the mean of that
    array to ``adapt_step`` to tune the shared step toward the scalar-optimal
    ~0.44 acceptance (target_accept(1)), not the joint-block 0.234.

    Only called when the resolved prior for 'a' is LogNormalPrior, or a
    NormalPrior without bounds (i.e. _a_prior_is_gibbs() returns False).
    When a bounded NormalPrior or TruncNormalPrior is used, the sampler
    takes the scalar Gibbs path instead and this function is never reached.

    Proposal: log(a*) = log(a) + step_size * ε, ε ~ N(0, I)
    Target: augmented normal likelihood on Z + log-prior on log(a).

    Change of variables.  The chain moves in xi = log a with a SYMMETRIC
    proposal, so the correct target here is the density of xi, not of a.
    That target is assembled entirely in xi: the augmented likelihood is a
    likelihood (it does not transform), and ``log_prior_fn`` returns the
    prior IN xi (a Normal kernel on log a, with NO -log a term).  Under
    those two conditions, and only those, no explicit +（log a* - log a)
    Jacobian is needed — the Normal-on-xi prior already carries the
    |da/dxi| factor.  If ``_resolve_a_prior_fn`` is ever changed to return
    an a-scale density (i.e. to re-add -log a), this ratio MUST gain a
    +(Σ log a* - Σ log a) term or the chain will target posterior × (1/a).
    """
    I = len(a_current)

    log_a         = np.log(a_current)
    log_a_prop    = log_a + step_size * (rng.standard_normal() if common
                                         else rng.standard_normal(I))
    a_prop        = np.exp(log_a_prop)

    # Per-item augmented log-likelihood on Z (treated as observed).
    # Z_pi ~ N(a_i (theta_p - b_i + offset), 1).  Given Z, theta, b, the a_i
    # are conditionally INDEPENDENT across items (the sum factorises over the
    # item axis), so we accumulate the log-likelihood into one bin per item
    # and accept/reject each item on its own.  This is ~I-fold better mixing
    # than a single joint accept/reject, which is forced to a ~1/sqrt(I) step
    # by the 0.234 high-dim scaling and stalls all items on one bad component.
    pred = theta[row_idx] - b[col_idx]
    if offset is not None:
        pred = pred + offset[row_idx, col_idx]
    z_rel = Z[row_idx, col_idx]

    def _aug_loglik_peritem(a_vec: np.ndarray) -> np.ndarray:
        diff = z_rel - a_vec[col_idx] * pred
        out  = np.zeros(I, dtype=np.float64)
        np.add.at(out, col_idx, -0.5 * diff ** 2)   # sum into per-item bins
        return out

    ll_curr = _aug_loglik_peritem(a_current)
    ll_prop = _aug_loglik_peritem(a_prop)

    # log_prior_fn now returns a PER-ITEM vector [I] (see _resolve_a_prior_fn).
    lp_curr = np.asarray(log_prior_fn(a_current), dtype=np.float64)
    lp_prop = np.asarray(log_prior_fn(a_prop),    dtype=np.float64)

    if common:
        log_alpha = (ll_prop.sum() + lp_prop[0]) - (ll_curr.sum() + lp_curr[0])
        accept = np.full(I, math.log(max(rng.uniform(), 1e-300)) < log_alpha)
        return (a_prop if accept[0] else a_current), accept
    log_alpha = (ll_prop + lp_prop) - (ll_curr + lp_curr)          # [I]
    accept    = np.log(rng.uniform(size=I)) < log_alpha            # [I] bool
    a_new     = np.where(accept, a_prop, a_current)
    return a_new, accept


def _mh_sigma2_halfnormal(
    theta:     np.ndarray,
    scale:     float,
    log_sigma: float,
    step_size: float,
    rng:       np.random.Generator,
) -> tuple[float, bool]:
    """Random-walk MH for σ on log scale under HalfNormal(0, scale) prior.

    The chain moves in z = log(σ) with a symmetric proposal, so the target
    must be the density of z, which is the σ-scale posterior times the
    change-of-variables Jacobian |dσ/dz| = σ = exp(z), i.e. a "+ z" term.

    Log unnormalized target on z = log(σ) scale:
        log p(z | θ) = -P*z - Σθ²/(2*exp(2z)) - exp(2z)/(2*scale²)  +  z
                       └────────── likelihood ──────┘ └─ prior ─┘   └ Jacobian ┘

    The final "+ z" is the Jacobian.  Without it the chain targets
    posterior × (1/σ), which under a weak likelihood biases σ toward zero;
    with it the stationary mean matches a grid-integrated posterior.

    Returns
    -------
    log_sigma_new : float
    accepted : bool
    """
    P          = len(theta)
    sum_theta2 = float(np.sum(theta ** 2))

    def _log_target(lz: float) -> float:
        sigma2 = math.exp(2.0 * lz)
        return (-P * lz - sum_theta2 / (2.0 * sigma2)
                - sigma2 / (2.0 * scale ** 2) + lz)   # + lz = log|dσ/dz| Jacobian

    log_sigma_prop = log_sigma + step_size * rng.standard_normal()
    log_alpha      = _log_target(log_sigma_prop) - _log_target(log_sigma)

    if math.log(rng.uniform()) < log_alpha:
        return log_sigma_prop, True
    return log_sigma, False


def _mh_rho_fisher_z(
    theta_matrix:  np.ndarray,
    sigma2_list:   np.ndarray,
    rho_current:   float,
    log_prior_fn,
    step_size:     float,
    rng:           np.random.Generator,
) -> tuple[float, bool]:
    """Fisher-z MH for scalar correlation ρ ∈ (-1, 1).

    General implementation: accepts any [N × 2] data matrix and two variance
    values.  Called in two contexts:
        • theta.correlation (D=2, LKJ/bounded prior):
            data_matrix = theta_matrix [P × 2], variance_list = sigma2_list [2]
        • b.correlation or a.correlation (G=2, LKJ prior):
            data_matrix = b_matrix or a_matrix [I × 2], variance_list = tau2_list [2]

    Fisher z-transform: z = arctanh(ρ), mapping (-1,1) → ℝ.
    Symmetric RW on z; back-transform via tanh.

    Log-likelihood: bivariate Normal on rows of data_matrix with covariance
    Σ = D R D where D = diag(√σ₁, √σ₂), R = [[1,ρ],[ρ,1]].

    Change of variables.  The proposal is symmetric in z = arctanh(ρ), but
    the target and prior are densities in ρ, so the correct target in the
    working coordinate z carries |dρ/dz| = 1 − ρ².  This factor is evaluated
    at ρ* and at ρ and is NOT equal at the two points (that is the whole
    point of the move), so it does NOT cancel: the acceptance ratio must
    include +（log(1 − ρ*²) − log(1 − ρ²)).  Equivalently, viewing the map
    as an asymmetric proposal on ρ, this same term is the Hastings ratio
    q(ρ|ρ*)/q(ρ*|ρ).  Verified against a grid-integrated posterior: with the
    term, the stationary mean matches the grid; without it, ρ is biased
    toward ±1.
    """
    def _log_lik(rho: float) -> float:
        Sigma  = _gibbs.build_sigma(sigma2_list, np.array([[1.0, rho], [rho, 1.0]]))
        try:
            mvn   = _stats.multivariate_normal(mean=np.zeros(2), cov=Sigma)
            return float(np.sum(mvn.logpdf(theta_matrix)))
        except np.linalg.LinAlgError:
            return -np.inf

    z_current  = math.atanh(np.clip(rho_current, -0.9999, 0.9999))
    z_proposed = z_current + step_size * rng.standard_normal()
    rho_prop   = math.tanh(z_proposed)

    ll_curr = _log_lik(rho_current)
    ll_prop = _log_lik(rho_prop)
    lp_curr = float(log_prior_fn(rho_current))
    lp_prop = float(log_prior_fn(rho_prop))

    # Jacobian of the Fisher-z transform: log|dρ/dz| = log(1-ρ²).  It does
    # NOT cancel (proposal is symmetric in z, not ρ; the factor differs at
    # ρ* and ρ).  Floor 1-ρ² to avoid -inf when a proposal lands on ±1.
    _EPS = 1e-300
    log_jac = (math.log(max(1.0 - rho_prop ** 2, _EPS))
               - math.log(max(1.0 - rho_current ** 2, _EPS)))
    log_alpha = (ll_prop + lp_prop) - (ll_curr + lp_curr) + log_jac

    if math.log(rng.uniform()) < log_alpha:
        return rho_prop, True
    return rho_current, False


def _mh_correlation_vine(
    theta_matrix: np.ndarray,
    sigma2_list:  np.ndarray,
    C_current:    np.ndarray,
    eta:          float,
    step_sizes:   list[float],
    rng:          np.random.Generator,
) -> tuple[np.ndarray, list[bool]]:
    """Vine-decomposition MH for R×R correlation matrix (R > 2).

    Decomposes C into R(R-1)/2 partial correlations via Cholesky-based
    vine (Lewandowski et al. 2009).  Each partial correlation z_{jk} is
    updated one at a time via Fisher-z MH.

    LKJ log-density in the vine parameterization (eq. 16 of the paper),
    written with 1-based level index k = 1..R-1:
        log p ∝ Σ_{k=1}^{R-1} (η - 1 + (R-k-1)/2) × Σ_j log(1 - z_{jk}²)
    The loop below is 0-based (k = 0..R-2), so the exponent there is the same
    quantity shifted by one: (η - 1 + (R-k-2)/2).  These are identical, NOT an
    off-by-one.  Verified: under a flat likelihood this sampler reproduces the
    analytic LKJ pairwise-correlation variance 1/(2(η+(R-2)/2)+1) at every
    level; dropping the Fisher-z Jacobian inflates it (toward ±1), and using
    (R-k-1)/2 in the 0-based loop shrinks it.

    The chain stores marginal pairwise correlations (back-transformed from
    the Cholesky factor), not partial correlations.

    Returns
    -------
    C_new      : ndarray [R × R], updated correlation matrix
    accepted   : list[bool], one per partial correlation updated
    """
    R = C_current.shape[0]

    def _corr_to_partial(C: np.ndarray) -> np.ndarray:
        """Cholesky-based mapping from correlation → partial correlations."""
        L = np.linalg.cholesky(C)
        Z = np.zeros((R, R))
        for k in range(R):
            Z[k, k] = 1.0
            for j in range(k + 1, R):
                Z[j, k] = L[j, k] / np.sqrt(np.maximum(1.0 - np.sum(L[j, :k]**2), 1e-14))
                Z[j, k] = np.clip(Z[j, k], -0.9999, 0.9999)
        return Z

    def _partial_to_corr(Z: np.ndarray) -> np.ndarray:
        """Reverse mapping from partial correlations → correlation matrix."""
        L = np.zeros((R, R))
        for j in range(R):
            for k in range(j + 1):
                if k == 0:
                    L[j, k] = Z[j, k]
                else:
                    L[j, k] = Z[j, k] * np.sqrt(
                        np.maximum(1.0 - np.sum(L[j, :k]**2), 1e-14)
                    )
            # diagonal
        for j in range(R):
            diag = 1.0 - np.sum(L[j, :j]**2)
            L[j, j] = np.sqrt(np.maximum(diag, 1e-14))
        return L @ L.T

    def _lkj_log_prior(Z: np.ndarray) -> float:
        """LKJ log-density in partial-correlation parameterization."""
        log_p = 0.0
        for k in range(R - 1):
            exp = eta - 1.0 + (R - k - 2) / 2.0
            for j in range(k + 1, R):
                log_p += exp * math.log(max(1.0 - Z[j, k]**2, 1e-14))
        return log_p

    def _log_lik(C: np.ndarray) -> float:
        try:
            Sigma = _gibbs.build_sigma(sigma2_list, C)
            mvn   = _stats.multivariate_normal(mean=np.zeros(R), cov=Sigma)
            return float(np.sum(mvn.logpdf(theta_matrix)))
        except (np.linalg.LinAlgError, ValueError):
            return -np.inf

    Z_current  = _corr_to_partial(C_current)
    Z_proposed = Z_current.copy()
    accepted   = []

    idx = 0
    for k in range(R - 1):
        for j in range(k + 1, R):
            step = step_sizes[idx] if idx < len(step_sizes) else 0.1

            z_fish_curr = math.atanh(np.clip(Z_current[j, k], -0.9999, 0.9999))
            z_fish_prop = z_fish_curr + step * rng.standard_normal()
            Z_proposed[j, k] = math.tanh(z_fish_prop)

            C_prop = _partial_to_corr(Z_proposed)

            ll_curr = _log_lik(C_current) + _lkj_log_prior(Z_current)
            ll_prop = _log_lik(C_prop)    + _lkj_log_prior(Z_proposed)

            # Fisher-z Jacobian for the ONE partial correlation moved this step.
            # Proposal is symmetric in arctanh(z_{jk}); target/prior are in z_{jk};
            # |dz/d(arctanh z)| = 1 - z^2 differs at prop vs curr and does NOT
            # cancel.  Same fix as _mh_rho_fisher_z.
            log_jac = (math.log(max(1.0 - Z_proposed[j, k] ** 2, 1e-300))
                       - math.log(max(1.0 - Z_current[j, k] ** 2, 1e-300)))

            if math.log(max(rng.uniform(), 1e-300)) < (ll_prop - ll_curr) + log_jac:
                Z_current = Z_proposed.copy()
                C_current = C_prop
                accepted.append(True)
            else:
                Z_proposed[j, k] = Z_current[j, k]
                accepted.append(False)
            idx += 1

    return C_current, accepted


# ===========================================================================
# Per-node dispatch helper
# ===========================================================================

def _corr_log_prior_fn(spec):
    """Log prior density of a scalar correlation rho (two-member block).

    LKJPrior(eta): (eta - 1) log(1 - rho^2).  A bounded NormalPrior /
    TruncNormalPrior: the Normal kernel on (lower, upper) (default (-1, 1)).
    A bounded BetaPrior: the Beta density of (rho - lower)/(upper - lower).
    Outside the bounds the density is -inf, so the Fisher-z step rejects.
    """
    if isinstance(spec, LKJPrior):
        eta = float(spec.eta)
        return lambda rho: (eta - 1.0) * math.log(max(1.0 - rho * rho, 1e-14))
    lo = getattr(spec, "lower", None)
    hi = getattr(spec, "upper", None)
    lo = -1.0 if lo is None else max(float(lo), -1.0)
    hi = 1.0 if hi is None else min(float(hi), 1.0)
    if isinstance(spec, (NormalPrior, TruncNormalPrior)):
        if isinstance(spec.variance, PriorSpec):
            raise NotImplementedError(
                "A correlation prior needs a fixed variance (got an estimated one).")
        m, v = float(spec.mean), float(spec.variance)
        return lambda rho: (-0.5 * (rho - m) ** 2 / v) if lo < rho < hi else -np.inf
    if isinstance(spec, BetaPrior):
        al, be, w = float(spec.alpha), float(spec.beta), hi - lo

        def _lp(rho):
            x = (rho - lo) / w
            if not 0.0 < x < 1.0:
                return -np.inf
            return (al - 1.0) * math.log(x) + (be - 1.0) * math.log(1.0 - x)
        return _lp
    raise NotImplementedError(
        f"Unsupported correlation prior {type(spec).__name__}; use LKJPrior, a "
        f"bounded NormalPrior/BetaPrior (two members), InverseWishartPrior or "
        f"FixedCorrelation.")


def _a_prior_is_gibbs(prior) -> bool:
    """True when the prior on 'a' activates the exact Gibbs (truncated-Normal) path.

    Both NormalPrior and TruncNormalPrior with at least one bound set are
    conjugate with the augmented Z likelihood (linear in a).  Either class
    with lower and/or upper set is handled identically by the inverse-CDF
    Gibbs draw.  LogNormalPrior (unbounded, log-scale prior) always uses MH.
    """
    return isinstance(prior, (NormalPrior, TruncNormalPrior)) and prior.is_bounded


def _classify_nodes(node_models: list[NodeModel]) -> list[_NodeDispatch]:
    """Build per-node dispatch metadata from node model properties.

    The returned list drives all conditional branching in run_chain — no
    isinstance() checks appear inside the main loop.

    Note on has_discrimination: this flag is set False when has_guessing is
    True, even though 3PL nodes do have discrimination.  This prevents the
    2PL a-update path from firing for 3PL nodes; the 3PL branch handles a
    via its own _mh_discrimination call under has_guessing=True dispatch.

    Note on is_mixed: True only for GRMNode/GPCMNode instances whose
    pseudo_item_outcomes array contains both 2-outcome items (binary) and
    higher-outcome items (polytomous) — i.e., different items at the same
    node use different response formats.  Homogeneous nodes (all items same
    format) always have is_mixed=False.
    """
    dispatches = []
    for r, nm in enumerate(node_models):
        dispatches.append(_NodeDispatch(
            r=r,
            is_polytomous=nm.n_outcomes > 2,
            has_guessing=nm.has_guessing,
            has_discrimination=nm.has_discrimination and not nm.has_guessing,
            is_augmentable=nm.augmentable,
            n_outcomes=nm.n_outcomes,
            is_mixed=getattr(nm, "is_mixed", False),
        ))
    return dispatches


def _flatten_mixed_b(b_params: NodeBParams, I: int) -> np.ndarray:
    """Flatten NodeBParams into item-index order for chain storage.

    The chain layout emits columns in item order (0 to I-1), with binary
    items contributing one value and polytomous items contributing M_poly-1
    values.  This function produces that flat vector from a NodeBParams.

    Parameters
    ----------
    b_params : NodeBParams
    I : int
        Total number of items at this node.

    Returns
    -------
    ndarray, shape [I_bin + I_poly*(M_poly-1)]
    """
    bin_pos  = {int(idx): i for i, idx in enumerate(b_params.bin_idx)}
    poly_pos = {int(idx): i for i, idx in enumerate(b_params.poly_idx)}
    parts = []
    for item in range(I):
        if item in bin_pos:
            parts.append(b_params.b_binary[bin_pos[item]:bin_pos[item] + 1])
        else:
            pi = poly_pos[item]
            parts.append(b_params.b_poly[pi])
    return np.concatenate(parts)


# ===========================================================================
# Single-chain runner
# ===========================================================================

def run_chain(
    data:        PreparedData,
    node_models: list[NodeModel],
    priors:      Priors,
    n_iter:      int,
    n_warmup:    int       = 0,
    warmup_mode: str       = "evidence",
    min_keep:    int | None = None,
    seed:        int | None = None,
    warm_start:  bool      = True,
    chain_dtype: str       = "float32",
    thin_by:     int       = 1,
    spec=None,
    constraints=None,
    progress: bool = True,
    progress_callback=None,
    progress_min_interval: float = 0.5,
    progress_label: str = "",
    print_every: int | None = None,
) -> MCMCResult:
    """Run one MCMC chain.

    Parameters
    ----------
    data        : PreparedData from expand_data()
    node_models : list[NodeModel], one per IRTree node
    priors      : Priors from Priors.from_spec() or build_default_priors()
    n_iter      : total iterations (warmup + sampling)
    n_warmup    : iterations discarded before draws are kept.
                  MH chains adapt their step sizes during this period.
                  In warmup_mode="evidence" it is the hard CAP (n_warmup_max)
                  of the MH warm-up (0 = automatic).  For pure-Gibbs chains
                  (no MH block) it is the burn-in length: the given value
                  clamped to [2·M·W, n_iter - min_keep·thin_by], or 2·M·W
                  (600) when 0.
    warmup_mode : "evidence" (default) or "fixed".
                  "evidence" — each MH block freezes when its windowed
                               acceptance rate has stabilised (see mcmc_adapt),
                               or at the cap, whichever first.  n_warmup becomes
                               that cap (0 => auto: a fraction of n_iter, kept
                               below n_iter by min_keep).  Draws are kept only
                               once EVERY block has frozen (the first all-frozen
                               iteration), so the sampling phase is a true
                               fixed-kernel chain; per-block freeze times are in
                               the result's ``frozen_at``.  This is the default
                               because it both removes the guess of n_warmup AND
                               tunes the step sizes even when n_warmup is left at
                               0 — unlike "fixed", which never adapts at n_warmup
                               = 0.  A block freezes after M consecutive
                               stable windows of W iterations (no earlier
                               than M·W = 300 with the defaults).  Every
                               model needs n_iter >= 2·M·W + min_keep·thin_by
                               (~1600): room for a warm-up of up to 2·M·W
                               plus the kept draws.  Pure-Gibbs models have
                               no block to freeze: they burn in for n_warmup
                               iterations (clamped as above; 600 by default).
                  "fixed"    — every MH block adapts for exactly n_warmup
                               iterations, then freezes; draws t>=n_warmup are
                               kept.  Use it (with a small n_warmup) for
                               short diagnostic runs.
    min_keep    : evidence mode only — the guaranteed number of KEPT
                  (post-thinning) draws.  The warmup cap is held at or below
                  n_iter - min_keep*thin_by so that even a worst-case freeze at
                  the cap leaves this many samples for inference.  None uses the
                  package default (mcmc_adapt.EVIDENCE_MIN_KEEP).  Raise it for
                  high-dimensional joint blocks whose ESS/iter is low.
    seed        : integer seed for the numpy Generator (None → random)
    warm_start  : if True, initialise b from empirical node-level difficulty
                  (probit transform of observed 0-response proportion).
                  If False, initialise b to zero everywhere.
    chain_dtype : 'float32' (default, halves memory) or 'float64' (full precision)
    thin_by     : store every thin_by-th post-warmup sample (1 = no thinning)
    spec        : IRTreeSpec-like object with n_nodes, dim_to_nodes attributes.
                  If None, each node is treated as its own independent latent dim.
    constraints : Constraints object specifying b/a/c and coefficient
                  sharing groups, b/a correlation groups and theta
                  correlation blocks; validated against spec and node_models
                  before sampling.  If None, all parameters are independent
                  across nodes (equivalent to NO_CONSTRAINTS).
    progress    : print a progress line to stderr (default True).
    progress_callback : callable or None.  If given, it is called at each
                  report with keyword arguments t, n_iter, n_warmup, elapsed,
                  rate, eta, loglik (data log-likelihood at the current
                  state, no prior terms), accept (mean MH acceptance, None for
                  pure Gibbs), phase ("warmup" / "sample") and label, and the
                  stderr line is not printed.
    progress_min_interval : accepted for compatibility; not used (reports are
                  iteration-based, see print_every).
    progress_label : prefix for the progress line (irtree_fit uses it to tag
                  chains, e.g. "chain 1/4").
    print_every : report every print_every iterations and at the last one.
                  None (default) gives about ten reports per run
                  (every ceil(n_iter / 10) iterations).

    Notes on sampler paths activated by prior/constraints choices
    -------------------------------------------------------------
    TruncNormalPrior or bounded NormalPrior on 'a':
        Activates scalar truncated-Normal Gibbs for a nodes (per item,
        inverse-CDF draw).  Both NormalPrior(lower=0) and TruncNormalPrior(lower=0)
        are equivalent and use the same code path.  Much faster than
        _mh_discrimination; no n_warmup needed for those nodes.

    InverseWishartPrior on 'theta.correlation':
        For each theta_corr block (all D dimensions when no block is given)
        the per-dim sigma2 steps and the rho MH steps are replaced by one
        gibbs_Sigma_invwishart draw of the block covariance.
        params['corr_matrix'] stores the covariance Σ of the blocks; their
        sigma2 values are read from its diagonal after each draw, and
        dimensions outside every block keep their own sigma2 update.

    b_corr / a_corr in Constraints:
        For each b-correlation group, the per-node b updates are SKIPPED
        (avoiding a double update).  After the node loop the probit groups
        are drawn jointly (gibbs_b_correlated_from_stats, or
        gibbs_theta_given conditional on logit groups) and logit groups by
        item-wise MH with the conditional prior.  The Σ_b update follows:
        InverseWishartPrior → gibbs_Sigma_invwishart; LKJPrior →
        _mh_rho_fisher_z (G=2) or _mh_correlation_vine (G>2);
        FixedCorrelation → no update.  a-correlation groups: componentwise
        exact Gibbs (truncated to a > 0) or MH on log a for logit groups,
        estimated group variances under LKJ/fixed, then Σ_a.

    Returns
    -------
    MCMCResult
    """
    rng      = np.random.default_rng(seed)
    R        = data.n_nodes
    P        = data.n_persons
    I        = data.n_items
    dtype    = np.float32 if chain_dtype == "float32" else np.float64

    # --- Warmup mode resolution --------------------------------------------
    # "fixed":    n_warmup is a length; adapt for n_warmup iters, then freeze;
    #             keep draws with t >= n_warmup.
    # "evidence": each MH block freezes when its windowed acceptance rate has
    #             stabilised (mcmc_adapt), or at a CAP; n_warmup is that cap
    #             (0 => auto).  Keep draws only once every block has frozen.
    if warmup_mode not in ("fixed", "evidence"):
        raise ValueError(
            f"warmup_mode must be 'fixed' or 'evidence'; got {warmup_mode!r}."
        )
    _evidence = (warmup_mode == "evidence")
    _evidence_too_short = False   # resolved against the real guard, below
    if _evidence:
        # --- Evidence-warmup CAP policy ------------------------------------
        # The cap is the forced-freeze backstop, bounded so that even a freeze
        # AT the cap leaves a guaranteed kept-sample budget.  Because keep_start
        # is set by the SLOWEST block, the cap must be < n_iter by that budget.
        #
        #   floor_warmup = 2·M·W      : minimum room for a block to converge
        #   min_keep_pre = min_keep·thin_by : guaranteed pre-thin sampling iters
        #   cap_from_keep = n_iter − min_keep_pre     (hard inference floor)
        #   fraction cap  = ⌊WARMUP_FRACTION·n_iter⌋  (warmup ≤ half the run)
        #
        # The fraction governs on typical/long runs (so a complex block in a
        # long run gets proportionally more room — no fixed low ceiling); the
        # keep floor governs on short runs (protecting inference).  A
        # user-supplied n_warmup is honoured as the cap but clamped into
        # [floor_warmup, cap_from_keep] so it can never starve sampling.
        #
        # The run-length guard (n_iter big enough for warmup + kept budget) is
        # only MEANINGFUL when the model actually has MH blocks.  A pure-Gibbs
        # model (probit nodes, no MH block) has nothing to adapt, so the
        # guard RAISES only after the adapt states are built and something
        # needs adapting (see below).
        _M   = _mcmc_adapt.EVIDENCE_NEED_M
        _W   = _mcmc_adapt.EVIDENCE_WINDOW_W
        _frac = _mcmc_adapt.EVIDENCE_WARMUP_FRACTION
        floor_warmup = 2 * _M * _W
        mk           = int(min_keep) if min_keep is not None else _mcmc_adapt.EVIDENCE_MIN_KEEP
        if mk < 1:
            raise ValueError(f"min_keep must be >= 1; got {mk}.")
        min_keep_pre = mk * thin_by

        # Does the run have room for warmup + the guaranteed kept budget?
        # (Enforced only for models with MH blocks; see the deferred raise.)
        _evidence_too_short = n_iter < floor_warmup + min_keep_pre
        # Clamp cap_from_keep up to floor_warmup so n_warmup_eff stays a valid
        # positive schedule length even in the short-run case (which raises
        # below unless warmup_mode='fixed').
        cap_from_keep = max(n_iter - min_keep_pre, floor_warmup)
        if n_warmup and n_warmup > 0:
            n_warmup_eff = int(n_warmup)
            if n_warmup_eff > cap_from_keep and not _evidence_too_short:
                warnings.warn(
                    f"evidence warmup cap n_warmup={n_warmup_eff} would leave "
                    f"fewer than min_keep={mk} kept draws (post-thin); clamping "
                    f"the cap to {cap_from_keep} to protect the sampling phase.",
                    UserWarning, stacklevel=2,
                )
            n_warmup_eff = max(floor_warmup, min(n_warmup_eff, cap_from_keep))
        else:
            n_warmup_eff = max(min(int(_frac * n_iter), cap_from_keep), floor_warmup)
    else:
        n_warmup_eff = n_warmup

    # Adaptation is *attempted* up to this iteration; in evidence mode the block
    # self-freezes earlier on the stability test, so we keep calling adapt_step
    # every iteration and let the state decide.  In fixed mode this equals
    # n_warmup, reproducing the original gate exactly.
    adapt_until = n_warmup_eff if not _evidence else n_iter

    # --- Resolve per-node covariate selections once, up front ---
    # This validates names/indices and pooling consistency, and turns each
    # node's declared person/item/pi_covariates into concrete column indices
    # (pooled per latent dimension for beta, per b-group for kappa). Doing it
    # here — before layout, init, and the iteration loop — means a bad
    # covariate name or an inconsistent shared-dimension declaration fails
    # immediately rather than deep inside sampling.
    # Enforce the documented sharing / correlation rules (Constraints
    # conditions 1-11 and theta_corr blocks).  validate() existed but was
    # never called, so violations went unchecked.
    if constraints is not None and spec is not None:
        constraints.validate(spec, node_models)

    cov_index = _dp.resolve_covariate_indices(
        node_models, data, spec=spec, constraints=constraints,
    )

    # --- Build layout and preallocate chain ---
    # Under an LKJ (or fixed) a.correlation the a-group variances set the
    # scale D in Sigma_a = D R D.  Held fixed at the a prior's 0.5, they
    # force rho toward +-1 whenever the slopes vary less than that (the MVN
    # log-likelihood is then dominated by -N/2 log(1 - rho^2)).  When
    # a's prior is the package default, estimate each correlated group's
    # variance with a Half-Normal(0, 0.5) prior on its sd (positive density
    # at 0, so small slope variances are not pushed up).  Applied to a copy
    # of the priors; a user-set fixed or hierarchical variance is respected.
    priors = _effective_priors(priors, constraints, R)

    param_names, offsets = _build_chain_layout(
        data, node_models, priors, spec=spec, constraints=constraints,
        cov_index=cov_index,
    )
    d          = len(param_names)
    if _evidence:
        # keep_start (the first all-frozen iteration) is only known at runtime,
        # so allocate the worst case (keep from t=0) and trim via stored_idx.
        n_stored = math.ceil(n_iter / thin_by)
    else:
        n_sampling = n_iter - n_warmup_eff
        n_stored   = math.ceil(n_sampling / thin_by)
    chain      = np.empty((n_stored, d), dtype=dtype)
    log_lik_trace = np.empty(n_stored, dtype=np.float64)

    # --- Initialise parameters ---
    params   = _init_params(data, node_models, priors, warm_start, rng, spec=spec, constraints=constraints, cov_index=cov_index)
    dispatch       = _classify_nodes(node_models)
    dim_to_nodes = spec.dim_to_nodes if spec is not None else [[r] for r in range(R)]
    D              = len(dim_to_nodes)
    # node_to_latent_dim[r] = the latent dim index for node r
    if spec is not None and spec.node_to_latent is not None:
        node_to_dim = list(spec.node_to_latent)   # 0-based
    else:
        node_to_dim = list(range(R))

    # --- Build item-parameter group lookups from constraints ---
    _con          = constraints if constraints is not None else NO_CONSTRAINTS
    b_groups      = _con.b_groups_for(R)
    b_node_to_g   = _con.b_node_to_group_for(R)
    b_reps        = _con.b_representative_nodes(R)
    a_groups      = _con.a_groups_for(R)
    a_node_to_g   = _con.a_node_to_group_for(R)
    a_reps        = _con.a_representative_nodes(R)
    c_groups      = _con.c_groups_for(R)
    c_node_to_g   = _con.c_node_to_group_for(R)

    # --- Initialise Z (latent augmentation variables) ---
    Z = [np.zeros((P, I)) for _ in range(R)]

    # --- Precomputed per-group index sets for mixed nodes (A) ---
    # For each mixed GRMNode/GPCMNode, store row/col indices *within the
    # submatrix* (not the full [P,I] matrix) for the binary and polytomous
    # item groups.  Computed once here; consumed inside the Gibbs loop.
    # Full-matrix column is recovered as nm.bin_idx[col] or nm.poly_idx[col].
    bin_row_idx:  dict[int, np.ndarray] = {}
    bin_col_idx:  dict[int, np.ndarray] = {}
    poly_row_idx: dict[int, np.ndarray] = {}
    poly_col_idx: dict[int, np.ndarray] = {}

    for r, nm in enumerate(node_models):
        if isinstance(nm, (GRMNode, GPCMNode)) and nm.is_mixed:
            T_bin  = data.T[r][:, nm.bin_idx]
            T_poly = data.T[r][:, nm.poly_idx]
            bin_row_idx[r],  bin_col_idx[r]  = np.where(T_bin)
            poly_row_idx[r], poly_col_idx[r] = np.where(T_poly)

    # --- Initialise likelihood cache ---
    cache = init_cache(
        data=data,
        theta=[params["theta"][node_to_dim[r]] for r in range(R)],
        b=[params["b"][b_node_to_g[r]] for r in range(R)],
        node_models=node_models,
        a=[params["a"][a_node_to_g[r]] for r in range(R)],
        c=[params["c"][c_node_to_g[r]] for r in range(R)],
    )

    # --- Initialise adaptation states ---
    adapt_states = _build_adapt_states(node_models, priors, n_warmup_eff, R, D=D,
                                       constraints=_con, P=P, I=I,
                                       evidence=_evidence,
                                       dim_to_nodes=dim_to_nodes)

    # Covariate coefficient groups that include a non-augmented (logit/GPCM)
    # node are updated by MH (see _mh_coefficient); one block per group,
    # tuned to the block-size target acceptance.
    _coef_groups_by_kind = {}
    if cov_index is not None:
        _coef_groups_by_kind = {
            "beta":  (cov_index.x_groups, cov_index.x_idx),
            "kappa": (cov_index.q_groups, cov_index.q_idx),
            "gamma": (cov_index.w_groups, cov_index.w_group_idx),
        }
    _coef_mh = {
        kind: [any(not node_models[r].augmentable for r in grp) for grp in groups]
        for kind, (groups, _idx) in _coef_groups_by_kind.items()
    }
    if n_warmup_eff > 0:
        for kind, (groups, idx) in _coef_groups_by_kind.items():
            for g in range(len(groups)):
                if _coef_mh[kind][g]:
                    adapt_states[f"{kind}_mh_{g}"] = make_adapt_state(
                        n_warmup_eff, target=target_accept(len(idx[g])),
                        evidence=_evidence)

    # --- theta / b of non-augmented (logit, GPCM) nodes ---
    # A latent dimension with any non-augmented node is updated person by
    # person by MH (_theta_mh_dims); a b-group with any non-augmented member
    # item by item (_b_mh_groups).  Each target includes EVERY node sharing
    # the parameter and the current hierarchical prior.
    _theta_mh_dims = [d_ for d_ in range(len(dim_to_nodes))
                      if any(not node_models[r].augmentable for r in dim_to_nodes[d_])]
    _b_groups_chk = _con.b_groups_for(R)
    _b_mh_groups = [g for g, grp in enumerate(_b_groups_chk)
                    if any(not node_models[r].augmentable for r in grp)]
    _b_mh_set = set(_b_mh_groups)
    # GRM thresholds.  Every b-group with a GRM member gets
    # a Cowles step (thresholds with Z integrated out, then Z redrawn).  A
    # group of one GRM node does this inside the node loop; a shared group
    # (several GRM nodes, or GRM with GPCM) does it once after the loop from
    # all members' marginal likelihoods.
    def _is_grm(r_):
        return node_models[r_].augmentable and node_models[r_].n_outcomes > 2
    _tau_groups = [g for g, grp in enumerate(_b_groups_chk) if any(_is_grm(r_) for r_ in grp)]
    _tau_pooled = [g for g in _tau_groups
                   if len(_b_groups_chk[g]) > 1 or g in _b_mh_set]
    _tau_pooled_set = set(_tau_pooled)
    # Dimensions for the theta/b location move: every b-group used on the
    # dimension has all its members on it, and none is in a b-correlation
    # group (whose prior couples it to other groups).
    _n2d_chk = (list(spec.node_to_latent) if spec is not None and spec.node_to_latent is not None
                else list(range(R)))
    _bcorr_chk = {g for cg in _con.b_corr_groups_for(R) for g in cg}
    _bnode2g_chk = _con.b_node_to_group_for(R)
    _shift_dims, _shift_bgroups = [], {}
    for d_, nodes_ in enumerate(dim_to_nodes):
        gs_ = sorted({_bnode2g_chk[r_] for r_ in nodes_})
        if all(all(_n2d_chk[m_] == d_ for m_ in _b_groups_chk[g_]) and g_ not in _bcorr_chk
               for g_ in gs_):
            _shift_dims.append(d_)
            _shift_bgroups[d_] = gs_
    _tau_scale = {}
    for g in _tau_groups:
        grm_m_ = [r_ for r_ in _b_groups_chk[g] if _is_grm(r_)]
        nm_ = node_models[grm_m_[0]]
        if getattr(nm_, "is_mixed", False):
            _tau_scale[g] = _threshold_proposal_scale(
                [data.Y_star[r_] for r_ in grm_m_], [data.T[r_] for r_ in grm_m_],
                nm_.M_poly - 1, cols=nm_.poly_idx)
        else:
            _tau_scale[g] = _threshold_proposal_scale(
                [data.Y_star[r_] for r_ in _b_groups_chk[g]],
                [data.T[r_] for r_ in _b_groups_chk[g]], nm_.n_outcomes - 1)
    if n_warmup_eff > 0:
        for d_ in _theta_mh_dims:
            adapt_states[f"theta_mh_{d_}"] = make_adapt_state(
                n_warmup_eff, target=target_accept(1), evidence=_evidence)
        for g in _tau_pooled:
            grp_ = _b_groups_chk[g]
            if (getattr(node_models[grp_[0]], "is_mixed", False)
                    and any(not node_models[r_].augmentable for r_ in grp_)):
                adapt_states[f"bbin_mh_{g}"] = make_adapt_state(
                    n_warmup_eff, target=target_accept(1), evidence=_evidence)
        for g in _tau_groups:
            K_tau = (node_models[[r_ for r_ in _b_groups_chk[g] if _is_grm(r_)][0]].M_poly
                     if getattr(node_models[_b_groups_chk[g][0]], "is_mixed", False)
                     else node_models[_b_groups_chk[g][0]].n_outcomes) - 1
            adapt_states[f"tau_mh_{g}"] = make_adapt_state(
                n_warmup_eff, target=target_accept(K_tau), evidence=_evidence)
        for g in _b_mh_groups:
            if g in _tau_pooled_set:
                continue
            nm_g = node_models[_b_groups_chk[g][0]]
            if getattr(nm_g, "is_mixed", False):
                adapt_states[f"b_mh_{g}"] = make_adapt_state(
                    n_warmup_eff, target=target_accept(1), evidence=_evidence)
                adapt_states[f"bpoly_mh_{g}"] = make_adapt_state(
                    n_warmup_eff, target=target_accept(nm_g.M_poly - 1),
                    evidence=_evidence)
            else:
                adapt_states[f"b_mh_{g}"] = make_adapt_state(
                    n_warmup_eff, target=target_accept(max(nm_g.n_outcomes - 1, 1)),
                    evidence=_evidence)

    # Missing covariate units: persons (X), items (Q), cells (W).  When a
    # logit/GPCM node uses a covariate type with missing values, those units
    # are imputed partly by MH (see _impute_missing): one adapted step per
    # covariate type, tuned on the mean per-unit acceptance.
    _miss_units = {
        "x": (np.where(~data.X_observed)[0]
              if data.X is not None and data.n_missing_X > 0 else None),
        "q": (np.where(~data.Q_observed)[0]
              if data.Q is not None and data.n_missing_Q > 0 else None),
        "w": (np.argwhere(~data.W_observed)
              if data.W is not None and data.n_missing_W > 0 else None),
    }
    _miss_prior_name = {"x": "X_missing", "q": "Q_missing", "w": "W_missing"}
    if n_warmup_eff > 0 and cov_index is not None:
        _n2g = {"x": cov_index.x_node_to_group, "q": cov_index.q_node_to_group,
                "w": cov_index.w_node_to_group}
        _Kfull = {"x": None if data.X is None else data.X.shape[1],
                  "q": None if data.Q is None else data.Q.shape[1],
                  "w": None if data.W is None else data.W.shape[2]}
        for _c, _u in _miss_units.items():
            if _u is None:
                continue
            if any(not node_models[r].augmentable and _n2g[_c][r] is not None
                   for r in range(R)):
                adapt_states[f"{_c.upper()}_missing_mh"] = make_adapt_state(
                    n_warmup_eff, target=target_accept(_Kfull[_c]),
                    evidence=_evidence)

    # Burn-in for pure-Gibbs chains.  With no MH block there is no
    # freeze test to decide when the chain has left its starting values, so
    # the burn-in is a fixed number of iterations, under the same rule as the
    # MH warm-up: the user's n_warmup (if given) clamped into
    # [2*M*W, n_iter - min_keep*thin_by], otherwise the floor 2*M*W.  Gibbs
    # draws need no adaptation, but they still start from initial values, and
    # keeping iteration 0 onward mixed the transient into the posterior.
    if _evidence and not adapt_states:
        if n_warmup and n_warmup > 0:
            n_warmup_eff = max(floor_warmup, min(int(n_warmup), cap_from_keep))
        else:
            n_warmup_eff = floor_warmup

    # Evidence mode, the default, needs enough iterations for the warm-up
    # (MH) or burn-in (pure Gibbs) PLUS the guaranteed kept budget.
    if _evidence and _evidence_too_short:
        raise ValueError(
            f"warmup_mode='evidence' needs n_iter >= 2·M·W + min_keep·thin_by "
            f"= {floor_warmup} + {min_keep_pre} = {floor_warmup + min_keep_pre}; "
            f"got n_iter={n_iter}. Up to {floor_warmup} iterations are "
            f"reserved for warm-up/burn-in and at least min_keep={mk} draws "
            f"are kept. "
            f"Increase n_iter, lower min_keep, reduce thin_by, or pass "
            f"warmup_mode='fixed' (with n_warmup) for a short diagnostic run."
        )

    # --- Resolve prior parameters used each iteration ---
    def _resolve_sigma2_prior(r: int):
        th_prior = priors.resolve("theta", r)
        var = getattr(th_prior, "variance", None)
        if isinstance(var, InvGammaPrior):
            hp = var
            return hp.alpha, hp.beta, "invgamma"
        if isinstance(var, HalfNormalPrior):
            return var.scale, None, "halfnormal"
        if isinstance(var, (int, float)):
            return float(var), None, "fixed"
        return 1.0, None, "fixed"

    def _resolve_b_prior(r: int):
        b_prior = priors.resolve("b", r)
        var = getattr(b_prior, "variance", None)
        if isinstance(var, InvGammaPrior):
            return None, None   # b.variance[g] is free; updated in post-node loop
        if isinstance(var, (int, float)):
            return float(var), None
        return 1.0, None

    def _resolve_a_prior_fn(r: int):
        a_prior = priors.resolve("a", r)
        mu_a  = getattr(a_prior, "mean",     0.0)
        var_a = getattr(a_prior, "variance", 0.5)
        def log_prior_fn(a_vec):
            # The chain works in the coordinate xi = log a, and a LogNormal(mu,var)
            # prior on a IS a Normal(mu,var) prior on xi.  We therefore return the
            # prior density IN xi (a plain Normal kernel), NOT the a-scale LogNormal
            # density.  Because the proposal in _mh_discrimination is a symmetric
            # random walk in xi and the target is assembled in xi, no change-of-
            # variables Jacobian is needed here: the Normal-on-xi density already
            # equals (a-scale LogNormal density) x |da/dxi|.
            #
            # Subtracting Sum(log a) here as well would leave one uncancelled
            # -log a: the chain would target posterior x (1/a), biasing the
            # discriminations toward zero (under a flat likelihood it converges
            # to N(mu - var, var) instead of the prior N(mu, var)).
            # Returns a PER-ITEM vector [I]: the a_i are conditionally
            # independent given Z, so _mh_discrimination accepts/rejects each
            # item on its own.  Sum externally if a scalar prior is needed.
            return -0.5 * ((np.log(a_vec) - mu_a) ** 2) / var_a
        return log_prior_fn

    def _resolve_corr_spec():
        return priors.resolve("theta.correlation", None)

    # --- Pre-resolve per-latent-dim sigma2 constants ---
    # Use the representative node (first in dim) to resolve prior.
    sigma2_prior_type = []
    sigma2_alpha      = []
    sigma2_beta_prior = []
    halfnormal_scale  = []
    for d in range(D):
        r_rep = dim_to_nodes[d][0]
        al, bt, typ = _resolve_sigma2_prior(r_rep)
        sigma2_prior_type.append(typ)
        sigma2_alpha.append(al)
        sigma2_beta_prior.append(bt if bt is not None else 0.0)
        halfnormal_scale.append(al if typ == "halfnormal" else 1.0)
        # A fixed theta variance must be the value used: sigma2 was left at
        # its starting value 1.0, so NormalPrior(variance=v) on theta acted
        # as variance 1 for every v.
        if typ == "fixed":
            params["sigma2"][d] = float(al)
            if "log_sigma" in params and params["log_sigma"] is not None:
                params["log_sigma"][d] = 0.5 * math.log(float(al))

    # Per-b-group: is b.variance estimated (InvGammaPrior)? Use representative node's prior.
    b_tau2_is_free    = []   # list[bool], one per b-group
    tau2_alpha_priors = []   # list[float]
    tau2_beta_priors  = []   # list[float]
    for rep_r in b_reps:
        b_prior_r = priors.resolve("b", rep_r)
        var_r = getattr(b_prior_r, "variance", None)
        if isinstance(var_r, InvGammaPrior):
            b_tau2_is_free.append(True)
            tau2_alpha_priors.append(var_r.alpha)
            tau2_beta_priors.append(var_r.beta)
        else:
            b_tau2_is_free.append(False)
            tau2_alpha_priors.append(1.0)
            tau2_beta_priors.append(1.0)
    any_b_tau2_free = any(b_tau2_is_free)

    # --- b.corr / a.corr setup ---
    b_corr_spec       = priors.type_level.get("b.correlation", FixedCorrelation(rho=0.0))
    b_corr_groups     = _con.b_corr_groups_for(R)
    any_b_corr        = len(b_corr_groups) > 0
    # b_node_in_corr[r] = True if node r's b-group is in any b-correlation group.
    # Used inside the per-node loop to SKIP the scalar gibbs_b for correlated nodes
    # (the correlated-b block after the node loop replaces it).
    _b_corr_node_map  = _con.b_node_to_corr_group_for(R)   # list[int|None]
    b_node_in_b_corr  = [_b_corr_node_map[r] is not None for r in range(R)]
    # b_node_in_shared[r] = True if node r's b-group has >1 member node.
    # Such groups are drawn ONCE after the node loop from POOLED sufficient
    # statistics across all member nodes (see _gibbs_b_shared_pooled below).
    # Drawing inside the loop would run one draw per member node, each
    # conditioning on only that node's cells, with the last overwriting the
    # rest -- not a full conditional of the joint target.
    _b_groups_all     = _con.b_groups_for(R)
    b_multi_groups    = [g for g, grp in enumerate(_b_groups_all) if len(grp) > 1]
    b_node_in_shared  = [len(_b_groups_all[b_node_to_g[r]]) > 1 for r in range(R)]

    # a / c sharing groups: same rule as b.  A shared array must be drawn ONCE
    # from pooled sufficient statistics over its member nodes, not once per node.
    _a_groups_all     = _con.a_groups_for(R)
    a_multi_groups    = [g for g, grp in enumerate(_a_groups_all) if len(grp) > 1]
    a_node_in_shared  = [len(_a_groups_all[a_node_to_g[r]]) > 1 for r in range(R)]
    _c_groups_all     = _con.c_groups_for(R)
    c_multi_groups    = [g for g, grp in enumerate(_c_groups_all) if len(grp) > 1]
    c_node_in_shared  = [len(_c_groups_all[c_node_to_g[r]]) > 1 for r in range(R)]
    # (c_shared groups are probit-only; Constraints.validate enforces it.)
    # One slope for the whole node (common_discrimination); a-sharing and
    # a-correlation groups reject it in Constraints.validate.
    _common_a = [bool(getattr(node_models[r], "common_discrimination", False))
                 and node_models[r].has_discrimination for r in range(R)]
    for r in range(R):
        if _common_a[r] and params["a.variance.estimated"][a_node_to_g[r]]:
            raise ValueError(
                f"Node {r + 1} has common_discrimination=True (one slope) and an "
                f"estimated variance on its a prior.  A variance across slopes "
                f"needs more than one slope; give the a prior a fixed variance.")

    b_corr_is_iw      = isinstance(b_corr_spec, InverseWishartPrior)
    # MH-estimated correlation: LKJ (any G) or a bounded scalar prior (G = 2)
    b_corr_is_lkj     = not isinstance(b_corr_spec, (FixedCorrelation, InverseWishartPrior))

    a_corr_spec       = priors.type_level.get("a.correlation", FixedCorrelation(rho=0.0))
    a_corr_groups     = _con.a_corr_groups_for(R)
    any_a_corr        = len(a_corr_groups) > 0
    _a_corr_node_map  = _con.a_node_to_corr_group_for(R)
    a_node_in_a_corr  = [_a_corr_node_map[r] is not None for r in range(R)]
    a_corr_is_iw      = isinstance(a_corr_spec, InverseWishartPrior)
    a_corr_is_lkj     = not isinstance(a_corr_spec, (FixedCorrelation, InverseWishartPrior))

    def _a_group_var(g_):
        """Current variance of a-group g_: the estimated value when its prior
        variance is hierarchical, else the prior's fixed variance."""
        if g_ < len(params.get("a.variance.estimated", [])) and params["a.variance.estimated"][g_]:
            return float(params["a.variance"][g_])
        ap = priors.resolve("a", a_reps[g_])
        v = getattr(ap, "variance", 0.5)
        return float(v) if isinstance(v, (int, float)) else 0.5

    # Pre-resolve InvWishart params for each b-corr group (fills vague defaults)
    b_corr_iw_params = []  # list[(Psi0, nu0)] per b-corr group
    if b_corr_is_iw:
        for cg in b_corr_groups:
            G_cg = len(cg)
            Psi0, nu0 = b_corr_spec.resolved_params(G_cg)
            b_corr_iw_params.append((Psi0, nu0))

    a_corr_iw_params = []
    if a_corr_is_iw:
        for cg in a_corr_groups:
            G_cg = len(cg)
            Psi0, nu0 = a_corr_spec.resolved_params(G_cg)
            a_corr_iw_params.append((Psi0, nu0))

    # Pre-resolve LKJ eta for b/a correlation
    b_corr_lkj_eta = float(getattr(b_corr_spec, "eta", 1.0))
    a_corr_lkj_eta = float(getattr(a_corr_spec, "eta", 1.0))
    # Log prior of a scalar correlation (two-member groups / blocks); built
    # here so an unsupported prior fails before the first iteration.
    _b_rho_lp = _corr_log_prior_fn(b_corr_spec) if b_corr_is_lkj else None
    _a_rho_lp = _corr_log_prior_fn(a_corr_spec) if a_corr_is_lkj else None

    # Detect TruncNormal prior on a (unlocks Gibbs path for a)
    # Check representative node of each a-group
    # Pre-compute Gibbs-path flag per a-group (bounded Normal or TruncNormal).
    # When True: inverse-CDF Gibbs draw. When False: LogNormal MH.
    a_prior_is_truncnormal = []
    for rep_r in a_reps:
        if rep_r in priors._param_nodes.get("a", []):
            a_prior_is_truncnormal.append(_a_prior_is_gibbs(priors.resolve("a", rep_r)))
        else:
            a_prior_is_truncnormal.append(False)

    # Register adapt states for the pooled shared-a MH blocks.  A multi-node
    # a-group whose shared discrimination uses MH (LogNormal, i.e. not the
    # truncnormal Gibbs path) is updated by the pooled block under key
    # "a_shared_{g}", with the same per-item scalar target (~0.44) as the
    # per-node a-blocks, since the pooled update is also per-item.  Without
    # an adapt state the block would run at the fixed fallback step 0.3.
    for _cg in a_corr_groups:
        for g in _cg:
            if any(not node_models[r_].augmentable for r_ in _a_groups_all[g]):
                adapt_states[f"a_corr_mh_{g}"] = make_adapt_state(
                    n_warmup_eff, target=target_accept(1), evidence=_evidence)
    for g in a_multi_groups:
        _has_logit_a = any(not node_models[r_].augmentable for r_ in _a_groups_all[g])
        if not a_prior_is_truncnormal[g] or _has_logit_a:
            key = f"a_shared_{g}"
            if key not in adapt_states:
                adapt_states[key] = make_adapt_state(n_warmup_eff, target=target_accept(1), evidence=_evidence)

    corr_spec    = _resolve_corr_spec()
    is_estimated_corr = not isinstance(corr_spec, FixedCorrelation)
    using_invwishart_theta = isinstance(corr_spec, InverseWishartPrior)
    if using_invwishart_theta and any(
            any(node_models[r_].has_discrimination for r_ in nodes_)
            for nodes_ in dim_to_nodes):
        warnings.warn(
            "InverseWishartPrior on theta.correlation estimates the theta "
            "variances, so on dimensions with free discriminations the scale "
            "of theta and the slopes trade off (a -> c a, theta -> theta / c) "
            "and is identified only by the a prior.  Use LKJPrior (theta "
            "variances then follow the theta prior, fixed at 1 by default on "
            "those dimensions) or fix the slopes' scale another way.",
            UserWarning, stacklevel=2)

    _th_blocks, _th_pairs, _th_tri = _theta_corr_blocks(constraints, D)
    _th_free_dims = [d_ for d_ in range(D) if not any(d_ in b_ for b_ in _th_blocks)]

    def _theta_Sigma():
        """Covariance of theta_p used by every theta draw: D R D (LKJ/fixed),
        or the InverseWishart block draws with sigma2 on the dimensions
        outside every block."""
        if not using_invwishart_theta:
            return _gibbs.build_sigma(params["sigma2"], params["corr_matrix"])
        S_ = np.array(params["corr_matrix"], dtype=np.float64, copy=True)
        for d_ in _th_free_dims:
            S_[d_, :] = 0.0
            S_[:, d_] = 0.0
            S_[d_, d_] = params["sigma2"][d_]
        return S_

    if using_invwishart_theta:
        _theta_iw_Psi0, _theta_iw_nu0 = corr_spec.resolved_params(D)
        _theta_iw_block_params = [corr_spec.resolved_params(len(b_)) for b_ in _th_blocks]
        # Warn when per-dim sigma2 InvGammaPrior is silently overridden.
        # InverseWishartPrior subsumes all per-dimension variances into a
        # single Σ draw; the InvGammaPrior on theta.variance is never
        # called.  sigma2 is read from diag(Σ) instead.
        if any(t == "invgamma" for t in sigma2_prior_type):
            warnings.warn(
                "InverseWishartPrior on theta.correlation subsumes per-dimension "
                "sigma2 draws.  The InvGammaPrior specified on "
                "theta.variance will not be used; sigma2 values are read from "
                "the diagonal of the InvWishart draw instead.  To suppress this "
                "warning, set theta.variance to a fixed float (e.g. "
                "NormalPrior(mean=0, variance=1.0)) when using "
                "InverseWishartPrior on theta.correlation.",
                UserWarning,
                stacklevel=2,
            )

    # Warn when b.correlation InverseWishartPrior overrides b.variance InvGammaPrior.
    # b_tau2_is_free[g] is True when the b-group g has a free (InvGammaPrior) variance.
    if b_corr_is_iw and any_b_corr:
        # Which b-groups are in a correlation group?
        b_corr_group_set = set(g for cg in b_corr_groups for g in cg)
        if any(b_tau2_is_free[g] for g in b_corr_group_set
               if g < len(b_tau2_is_free)):
            warnings.warn(
                "InverseWishartPrior on b.correlation subsumes per-group "
                "b.variance draws for the correlated b-groups.  The InvGammaPrior "
                "(InvGamma) on b.variance for those groups will not be used; "
                "variances are read from the diagonal of the InvWishart draw "
                "instead.  To suppress this warning, set b.variance to a fixed "
                "float for correlated b-groups when using InverseWishartPrior "
                "on b.correlation.",
                UserWarning,
                stacklevel=2,
            )

    # Warn when a.correlation InverseWishartPrior is used alongside a
    # TruncNormalPrior that has its own estimated variance (InvGammaPrior).
    if a_corr_is_iw and any_a_corr:
        a_corr_group_set = set(g for cg in a_corr_groups for g in cg)
        for g in a_corr_group_set:
            if g < len(a_reps):
                a_spec = priors.resolve("a", a_reps[g])
                if _a_prior_is_gibbs(a_spec) and isinstance(
                    getattr(a_spec, "variance", None), InvGammaPrior
                ):
                    warnings.warn(
                        "InverseWishartPrior on a.correlation subsumes the "
                        "per-group a.variance draws for the correlated a-groups.  "
                        "The InvGammaPrior on a.variance for those groups "
                        "will not be used; variances are read from the diagonal of "
                        "the InvWishart draw instead.  To suppress this warning, "
                        "set a.variance to a fixed float for correlated a-groups "
                        "when using InverseWishartPrior on a.correlation.",
                        UserWarning,
                        stacklevel=2,
                    )
                    break

    lkj_eta = float(getattr(corr_spec, "eta", 1.0))   # vine (G > 2): LKJ only
    _theta_rho_lp = (_corr_log_prior_fn(corr_spec)
                     if is_estimated_corr and not isinstance(corr_spec, InverseWishartPrior)
                     else None)

    stored_idx = 0

    # Drop adaptation blocks that never run: a node's own slope block when its
    # a-group is shared or correlated (the pooled / correlated block updates
    # it instead).  A registered block that never steps never freezes, and
    # evidence mode then stores no draws at all.
    for r in range(R):
        if a_node_in_shared[r] or a_node_in_a_corr[r]:
            adapt_states.pop(f"a_mh_{r}", None)
            adapt_states.pop(f"a_{r}", None)

    # keep_start: the first iteration whose draw is kept.
    #   fixed    — known up front = n_warmup_eff.
    #   evidence — unknown until every MH block has frozen; set to the first
    #              all-frozen iteration (t_freeze_all) inside the loop.  Until
    #              then no draw is stored.
    if _evidence:
        keep_start = None
        # Pure Gibbs: nothing to freeze, so sampling starts after the fixed
        # burn-in resolved above.
        if not adapt_states:
            keep_start = n_warmup_eff
    else:
        keep_start = n_warmup_eff

    # -----------------------------------------------------------------------
    # Main iteration loop
    # -----------------------------------------------------------------------

    # Dimensions drawn by the unified Gibbs block: every node mapping to the
    # dimension must be augmentable.  Dimensions owned by MH nodes are updated
    # inside those nodes' own blocks.
    # Adaptation state for the a.variance MH blocks (one per a-group whose
    # discrimination prior carries a forced hierarchical variance).
    a_var_estimated = params.get("a.variance.estimated", [])
    for g, est in enumerate(a_var_estimated):
        if est:
            # scalar hierarchical discrimination variance -> dim 1
            adapt_states[f"a.variance_{g}"] = make_adapt_state(n_warmup_eff, target=target_accept(1), evidence=_evidence)

    gibbs_owned_dims = [
        d for d in range(D)
        if all(dispatch[_r].is_augmentable for _r in dim_to_nodes[d])
    ]

    _reporter = ProgressReporter(
        n_iter=n_iter, n_warmup=n_warmup, enabled=progress,
        callback=progress_callback, min_interval=progress_min_interval,
        label=progress_label, print_every=print_every,
    )
    _reporter.start()

    for t in range(n_iter):

        # Per-iteration state retained for the pooled a / c draws below.
        # ab_idx_by_node[r] = (row, col) of the cells that inform a at node r
        # (ability cells only for 3PL, all relevant cells for 2PL).
        U_by_node      = {}
        ab_idx_by_node = {}

        # Latent-regression view of each augmented node, shared by the
        # coefficient, imputation and logit-MH blocks below.  At node r, on
        # the cells where Z_r was drawn this iteration (its latent mask),
        #     Z_r = a_r (core_r + covariate offset) + eps.
        # The closures read params at call time, so they always see the
        # latest draws of every other block.
        _empty_idx = np.empty(0, dtype=np.intp)

        def _latent_view(r):
            """(mask, a, core) of node r's latent regression, or None."""
            d_info_r = dispatch[r]
            if not d_info_r.is_augmentable:
                return None
            th = params["theta"][node_to_dim[r]]
            b_v = params["b"][b_node_to_g[r]]
            if d_info_r.has_guessing:
                ri, ci = ab_idx_by_node.get(r, (_empty_idx, _empty_idx))
                mask = np.zeros((P, I), dtype=bool)
                mask[ri, ci] = True
            else:
                mask = data.T[r]
            if d_info_r.is_mixed:
                core = np.repeat(th[:, None], I, axis=1)
                core[:, node_models[r].bin_idx] -= np.asarray(b_v.b_binary)[None, :]
            elif d_info_r.is_polytomous:
                core = np.repeat(th[:, None], I, axis=1)
            else:
                core = th[:, None] - np.asarray(b_v)[None, :]
            a_cur = (params["a"][a_node_to_g[r]]
                     if a_node_to_g[r] is not None else None)
            return mask, a_cur, core

        def _latent_resid(r, include):
            """E_r = Z_r - a_r (core_r + included covariate terms), or None."""
            view = _latent_view(r)
            if view is None:
                return None
            _m, a_r_, core = view
            off = _node_offset(r, params, cov_index, data, include=include)
            pred = core if off is None else core + off
            return Z[r] - (pred if a_r_ is None else a_r_[None, :] * pred)

        def _masks():
            return [(v[0] if v is not None else None)
                    for v in (_latent_view(r) for r in range(R))]

        def _a_gauss_stats(members):
            """Gaussian statistics (n [I], s [I]) of a discrimination array
            shared by ``members``: on each augmented member's latent mask,
            Z = a d + eps with d = core + offset (core = theta - b), so
            n_i = sum d^2 and s_i = sum d Z.  Non-augmented members add 0."""
            n_ = np.zeros(I)
            s_ = np.zeros(I)
            for r in members:
                view = _latent_view(r)
                if view is None or view[1] is None:
                    continue
                mask, _a, core = view
                off = _node_offset(r, params, cov_index, data)
                d = core if off is None else core + off
                n_ += np.where(mask, d * d, 0.0).sum(axis=0)
                s_ += np.where(mask, d * Z[r], 0.0).sum(axis=0)
            return n_, s_

        def _b_gauss_stats(members):
            """Gaussian statistics (N [I], S [I]) of a binary b array shared by
            ``members``.  At each augmented member, on its latent mask (U = 0
            cells at a 3PL node),

                E = Z - a (theta + offset) = -a b + eps,

            so N_i = sum a_i^2 and S_i = sum a_i (a_i (theta + offset) - Z),
            and b | rest ~ N(S / (1/tau2 + N), 1 / (1/tau2 + N)) under a
            N(0, tau2) prior.  Non-augmented members contribute nothing.
            """
            N_ = np.zeros(I)
            S_ = np.zeros(I)
            for r in members:
                view = _latent_view(r)
                if view is None:
                    continue
                mask, a_r, _core = view
                b_cur = np.asarray(params["b"][b_node_to_g[r]], dtype=np.float64)
                E = _latent_resid(r, ("x", "q", "w"))          # Z - a(theta - b + off)
                E = E - (b_cur[None, :] if a_r is None else a_r[None, :] * b_cur[None, :])
                a1 = np.ones(I) if a_r is None else np.asarray(a_r, dtype=np.float64)
                N_ += (a1 ** 2) * mask.sum(axis=0)
                S_ += a1 * np.where(mask, -E, 0.0).sum(axis=0)
            return N_, S_


        # --- Per-node updates ---
        for d_info in dispatch:
            r = d_info.r
            T_r      = data.T[r]
            row_idx  = data.row_idx[r]
            col_idx  = data.col_idx[r]
            d        = node_to_dim[r]
            theta_r  = params["theta"][d]
            g_b      = b_node_to_g[r]
            g_a      = a_node_to_g[r]
            g_c      = c_node_to_g[r]
            b_r      = params["b"][g_b]
            a_r      = params["a"][g_a]
            c_r      = params["c"][g_c]

            # Resolve b.variance for this node from the per-b-group params list
            tau2_r = params["b.variance"][b_node_to_g[r]]

            # ----------------------------------------------------------
            # Covariate offset for this node
            # ----------------------------------------------------------
            # Computed before sample_Z so the augmented likelihood reflects
            # current covariate contributions. When no covariates are active
            # at this node, _node_offset returns None and all downstream
            # paths are identical to the pre-covariate code. Single source of
            # truth shared with the residualization offset below.
            _offset = _node_offset(r, params, cov_index, data)
            # _offset is the RAW covariate linear predictor.  Passed directly to
            # sample_Z / log_prob (they apply a_i themselves).  But wherever we
            # subtract it from Z to residualise for a Gibbs draw, Z lives on the
            # a-scale (Z ~ N(a_i(theta - b + offset), 1)), so we must subtract
            # a_i * offset.  _offset_a is that a-scaled version (== _offset for
            # Rasch nodes, where a_r is None).
            _offset_a = _offset
            if _offset is not None and a_r is not None:
                _offset_a = a_r[np.newaxis, :] * _offset

            # Z_resid: Z with the covariate offset subtracted.
            # gibbs_theta and gibbs_b expect:
            #   Z_pi ≈ theta_p - b_i + noise  (Rasch model)
            # When covariates are present:
            #   Z_pi ≈ theta_p - b_i + X_p@beta + Q_i@kappa + W_{pi}@gamma + noise
            # So gibbs_theta / gibbs_b must receive Z - offset, not Z itself.
            # This is computed lazily after sample_Z (so Z[r] is already drawn
            # with the current offset), and used only for the sufficient-statistic
            # updates. sample_Z and recompute_node always receive _offset directly.
            # Note: _z_resid is only used for gibbs_b and gibbs_theta (post-node).
            # It is NOT used for sample_Z — sample_Z draws from the correct
            # truncated Normal because it has already received the offset.

            # ----------------------------------------------------------
            # 3PL: Béguin-Glas two-layer augmentation (exact order from v3)
            # ----------------------------------------------------------
            # Only PROBIT 3PL nodes take the Beguin-Glas augmentation; a logit
            # 3PL has no latent response and falls through to the MH branch.
            if d_info.has_guessing and d_info.is_augmentable:
                # Layer 1: draw guessing-mode indicators U
                U = _gibbs.sample_U_3pl(
                    data.Y_star[r], theta_r, b_r, a_r, c_r,
                    T_r, row_idx, col_idx, rng, offset=_offset,
                )

                # Restrict to ability-process cells (U=0) for Z and b
                T_ab     = T_r & (U == 0)
                ri_ab, ci_ab = np.where(T_ab)
                ab_idx_by_node[r] = (ri_ab, ci_ab)

                # Layer 2: draw Z *only* on ability cells
                Z[r] = _gibbs.sample_Z(
                    data.Y_star[r], theta_r, b_r,
                    T_ab, ri_ab, ci_ab, rng, a=a_r,
                    offset=_offset,
                )

                # Update b on ability cells only
                # Use Z_resid = Z[r] - _offset so theta and b see residualized Z.
                _Z_for_b = Z[r] if _offset is None else Z[r] - _offset_a
                if not (b_node_in_b_corr[r] or b_node_in_shared[r]):
                    params["b"][g_b] = _gibbs.gibbs_b(
                        _Z_for_b, theta_r, tau2_r,
                        T_ab, ri_ab, ci_ab, rng, a=a_r,
                        prior_mean=float(getattr(priors.resolve("b", r), "mean", 0.0)),
                    )

                # Update c (Beta draw, uses full U)
                c_prior = priors.resolve("c", r)
                alpha_c = getattr(c_prior, "alpha", 5.0)
                beta_c  = getattr(c_prior, "beta", 17.0)
                U_by_node[r] = U          # retained for the pooled c draw
                if c_node_in_shared[r]:
                    params["c"][g_c] = params["c"][g_c]   # drawn post-loop
                else:
                    params["c"][g_c] = _gibbs.gibbs_c_3pl(
                        U, data.Y_star[r], row_idx, col_idx,
                        alpha_c, beta_c, I, rng,
                    )

                # a: Gibbs (TruncNormal prior) or MH (LogNormal prior)
                # Uses T_ab / ri_ab / ci_ab (ability cells only, U=0).
                # Skip when this node's a-group is shared or in an
                # a-correlation group (the pooled / correlated-a block after
                # the node loop replaces it).
                if a_r is not None and not (a_node_in_a_corr[r] or a_node_in_shared[r]):
                    g_a_cur = a_node_to_g[r]
                    if a_prior_is_truncnormal[g_a_cur]:
                        # Exact Gibbs: scalar truncated Normal per item,
                        # restricted to ability cells (T_ab / ri_ab / ci_ab).
                        _a_prior_tn = priors.resolve("a", r)
                        params["a"][g_a] = _gibbs.gibbs_a(
                            Z[r], params["theta"][d], params["b"][g_b],
                            ri_ab, ci_ab,
                            float(getattr(_a_prior_tn, "mean", 1.0)),
                            float(getattr(_a_prior_tn, "variance", 0.5)),
                            _a_prior_tn.lower, rng, n_items=I,
                            offset=_offset, common=_common_a[r],
                        )
                    else:
                        log_prior_fn = _resolve_a_prior_fn(r)
                        key_a = f"a_{r}"
                        step  = adapt_states[key_a].step_size if key_a in adapt_states else 0.3
                        a_new, accepted = _mh_discrimination(
                            a_r, Z[r], theta_r, params["b"][g_b],
                            T_ab, ri_ab, ci_ab, log_prior_fn, step, rng,
                            offset=_offset, common=_common_a[r],
                        )
                        params["a"][g_a] = a_new
                        if key_a in adapt_states and t < adapt_until:
                            # accepted is a per-item boolean array; tune the
                            # shared step on the mean per-item acceptance.
                            adapt_states[key_a] = adapt_step(
                                adapt_states[key_a], float(np.mean(accepted)), t)

                # sigma2 update stays in the unified per-dim block below
                # (theta is updated there for this dim)

                # Cache uses the *restricted* likelihood for consistency
                # [moved to storage-time refresh] cache for augmentable
                # nodes is recomputed once per stored sample.
            # ----------------------------------------------------------
            # Augmentable: Rasch / 1PL / 2PL / GRM Gibbs
            # ----------------------------------------------------------
            elif d_info.is_augmentable:

                # --------------------------------------------------
                # Mixed GRM: Albert-Chib for binary items,
                # Gibbs-threshold for polytomous items (B)
                # --------------------------------------------------
                if d_info.is_mixed and g_b in _tau_pooled_set:
                    # Shared mixed-format b: draw Z only; b_binary and b_poly
                    # are updated once after the node loop from all members.
                    nm_r = node_models[r]
                    _off_b = _offset[:, nm_r.bin_idx] if _offset is not None else None
                    Z[r][:, nm_r.bin_idx] = _gibbs.sample_Z(
                        data.Y_star[r][:, nm_r.bin_idx], theta_r, b_r.b_binary,
                        data.T[r][:, nm_r.bin_idx], bin_row_idx[r], bin_col_idx[r], rng,
                        offset=_off_b)
                    _off_p = _offset[:, nm_r.poly_idx] if _offset is not None else None
                    Z[r][:, nm_r.poly_idx] = _gibbs.sample_Z_grm(
                        data.Y_star[r][:, nm_r.poly_idx], theta_r, b_r.b_poly,
                        data.T[r][:, nm_r.poly_idx], poly_row_idx[r], poly_col_idx[r],
                        nm_r.M_poly, rng, offset=_off_p)

                elif d_info.is_mixed:
                    nm_r = node_models[r]   # must be GRMNode with is_mixed=True

                    # --- Binary subgroup: Albert-Chib ---
                    Y_star_bin = data.Y_star[r][:, nm_r.bin_idx]
                    T_bin      = data.T[r][:, nm_r.bin_idx]
                    Z_bin = _gibbs.sample_Z(
                        Y_star_bin, theta_r, b_r.b_binary,
                        T_bin, bin_row_idx[r], bin_col_idx[r], rng,
                        offset=_offset[:, nm_r.bin_idx] if _offset is not None else None,
                    )
                    # Write binary Z draws back into the full Z[r] matrix
                    Z[r][:, nm_r.bin_idx] = Z_bin

                    _off_bin = _offset[:, nm_r.bin_idx] if _offset is not None else None
                    _Z_bin_resid = Z_bin if _off_bin is None else Z_bin - _off_bin
                    _bbin_spec  = priors.resolve("b.binary", r)
                    # Resolve BEFORE casting: a hierarchical variance is a
                    # PriorSpec, and float() on it raised on the first
                    # iteration of every mixed GRM node with default priors.
                    _bbin_raw   = getattr(_bbin_spec, "variance", tau2_r)
                    _bbin_tau2  = (float(tau2_r) if isinstance(_bbin_raw, PriorSpec)
                                   else float(_bbin_raw))
                    params["b"][g_b].b_binary = _gibbs.gibbs_b(
                        _Z_bin_resid, theta_r, _bbin_tau2,
                        T_bin, bin_row_idx[r], bin_col_idx[r], rng,
                        prior_mean=float(getattr(_bbin_spec, "mean", 0.0)),
                    )

                    # --- Polytomous subgroup: Cowles step, Z, then exact Gibbs ---
                    Y_star_poly = data.Y_star[r][:, nm_r.poly_idx]
                    T_poly      = data.T[r][:, nm_r.poly_idx]
                    _bpm_spec = priors.resolve("b.poly", r)
                    _bpm_raw  = getattr(_bpm_spec, "variance", 4.0)
                    _bpm_tau2 = (float(params["b.variance"][g_b])
                                 if isinstance(_bpm_raw, PriorSpec) else float(_bpm_raw))
                    _bpm_mu0  = float(getattr(_bpm_spec, "mean", 0.0))
                    _bin_now  = params["b"][g_b].b_binary

                    def _clm(Tm, r=r, nm_r=nm_r, theta_r=theta_r, _offset=_offset,
                             _bin_now=_bin_now):
                        bp = NodeBParams(b_binary=_bin_now, b_poly=Tm,
                                         bin_idx=nm_r.bin_idx, poly_idx=nm_r.poly_idx,
                                         M_poly=nm_r.M_poly)
                        lp = nm_r.log_prob(data.Y_star[r], theta_r, bp, offset=_offset)
                        return np.where(data.T[r], lp, 0.0)[:, nm_r.poly_idx].sum(axis=0)
                    _key_t = f"tau_mh_{g_b}"
                    _st_t = adapt_states[_key_t].step_size if _key_t in adapt_states else 1.0
                    _new_t, _acc_t = _mh_ordered_thresholds(
                        params["b"][g_b].b_poly, _clm, _bpm_mu0, _bpm_tau2, _st_t, rng,
                        scale=_tau_scale[g_b])
                    params["b"][g_b].b_poly = _new_t
                    b_r = params["b"][g_b]
                    if _key_t in adapt_states and t < adapt_until:
                        adapt_states[_key_t] = adapt_step(
                            adapt_states[_key_t], float(np.mean(_acc_t)), t)
                    Z_poly = _gibbs.sample_Z_grm(
                        Y_star_poly, theta_r, b_r.b_poly,
                        T_poly, poly_row_idx[r], poly_col_idx[r],
                        nm_r.M_poly, rng,
                        offset=_offset[:, nm_r.poly_idx] if _offset is not None else None,
                    )
                    # Write polytomous Z draws back into the full Z[r] matrix
                    Z[r][:, nm_r.poly_idx] = Z_poly

                    _bp_spec  = priors.resolve("b.poly", r)
                    _bp_mu0   = float(getattr(_bp_spec, "mean",     0.0))
                    # Resolve BEFORE casting: when the threshold prior declares a
                    # hierarchical variance (e.g. InvGammaPrior), 'variance' is a
                    # PriorSpec, not a number, and float() would raise.
                    _bp_var_raw = getattr(_bp_spec, "variance", 4.0)
                    if isinstance(_bp_var_raw, PriorSpec):
                        # Current hierarchical variance of this node's b-GROUP
                        # (indexed by group, not by node).
                        _bp_tau2 = float(params["b.variance"][g_b])
                    else:
                        _bp_tau2 = float(_bp_var_raw)
                    params["b"][g_b].b_poly = _gibbs.gibbs_thresholds_grm(
                        Z_poly, Y_star_poly, T_poly,
                        poly_row_idx[r], poly_col_idx[r],
                        nm_r.M_poly, rng,
                        old_thresholds=b_r.b_poly,
                        mu_0=_bp_mu0,
                        tau2=_bp_tau2,
                    )

                # [moved to storage-time refresh] cache for augmentable
                # nodes is recomputed once per stored sample.

                # --------------------------------------------------
                # Homogeneous polytomous GRM
                # --------------------------------------------------
                elif d_info.is_polytomous:
                    _bp_spec  = priors.resolve("b.poly", r)
                    _bp_mu0   = float(getattr(_bp_spec, "mean",     0.0))
                    # Resolve BEFORE casting: a hierarchical threshold variance
                    # is a PriorSpec; use the b-GROUP's current value.
                    _bp_var_raw = getattr(_bp_spec, "variance", 4.0)
                    _bp_tau2 = (float(params["b.variance"][g_b])
                                if isinstance(_bp_var_raw, PriorSpec)
                                else float(_bp_var_raw))
                    _own_tau = g_b not in _tau_pooled_set
                    if _own_tau:
                        # Cowles step: thresholds with Z integrated out, then Z
                        # redrawn below given them.
                        def _cl(Tm, r=r, theta_r=theta_r, _offset=_offset, T_r=T_r):
                            lp = node_models[r].log_prob(
                                data.Y_star[r], theta_r, Tm, offset=_offset)
                            return np.where(T_r, lp, 0.0).sum(axis=0)
                        _key_t = f"tau_mh_{g_b}"
                        _st_t = adapt_states[_key_t].step_size if _key_t in adapt_states else 1.0
                        _new_t, _acc_t = _mh_ordered_thresholds(
                            params["b"][g_b], _cl, _bp_mu0, _bp_tau2, _st_t, rng,
                            scale=_tau_scale[g_b])
                        params["b"][g_b] = _new_t
                        if _key_t in adapt_states and t < adapt_until:
                            adapt_states[_key_t] = adapt_step(
                                adapt_states[_key_t], float(np.mean(_acc_t)), t)
                    Z[r] = _gibbs.sample_Z_grm(
                        data.Y_star[r], theta_r, params["b"][g_b],
                        T_r, row_idx, col_idx, d_info.n_outcomes, rng,
                        offset=_offset,
                    )
                    # theta updated in post-node-loop section (pooled across shared dims)
                    if _own_tau:
                        # Exact Gibbs step given Z: cheap local moves.
                        params["b"][g_b] = _gibbs.gibbs_thresholds_grm(
                            Z[r], data.Y_star[r], T_r, row_idx, col_idx,
                            d_info.n_outcomes, rng, old_thresholds=params["b"][g_b],
                            mu_0=_bp_mu0,
                            tau2=_bp_tau2,
                        )
                    # Shared thresholds are updated once after the node loop.

                else:
                    # Binary: Rasch (a=None) or 2PL (a=array)
                    Z[r] = _gibbs.sample_Z(
                        data.Y_star[r], theta_r, b_r,
                        T_r, row_idx, col_idx, rng, a=a_r,
                        offset=_offset,
                    )
                    # theta updated in post-node-loop section (pooled across shared dims)
                    _Z_resid_b = Z[r] if _offset is None else Z[r] - _offset_a
                    # Skip scalar gibbs_b when this node's b-group is shared or
                    # in a b-correlation group: the pooled / correlated-b block
                    # after the node loop replaces it (no double update).
                    if not (b_node_in_b_corr[r] or b_node_in_shared[r]):
                        params["b"][g_b] = _gibbs.gibbs_b(
                            _Z_resid_b, theta_r, tau2_r,
                            T_r, row_idx, col_idx, rng, a=a_r,
                            prior_mean=float(getattr(priors.resolve("b", r), "mean", 0.0)),
                        )

                    # 2PL: MH for discrimination (or Gibbs if TruncNormal prior)
                    # Skip when this node's a-group is shared or in an
                    # a-correlation group (replaced after the node loop).
                    ab_idx_by_node.setdefault(r, (row_idx, col_idx))
                    if d_info.has_discrimination and a_r is not None and not (a_node_in_a_corr[r] or a_node_in_shared[r]):
                        g_a_cur = a_node_to_g[r]
                        if a_prior_is_truncnormal[g_a_cur]:
                            # Exact Gibbs: scalar truncated Normal per item.
                            _a_prior_tn = priors.resolve("a", r)
                            # When the discrimination variance is a forced
                            # hierarchical parameter, its current MH-updated
                            # value lives in params["a.variance"]; the prior's
                            # own .variance slot holds the (non-float) PriorSpec.
                            _a_var_raw = getattr(_a_prior_tn, "variance", 0.5)
                            if isinstance(_a_var_raw, PriorSpec):
                                _a_var = float(params["a.variance"][g_a_cur])
                            else:
                                _a_var = float(_a_var_raw)
                            params["a"][g_a] = _gibbs.gibbs_a(
                                Z[r], params["theta"][d], params["b"][g_b],
                                row_idx, col_idx,
                                float(getattr(_a_prior_tn, "mean", 1.0)),
                                _a_var,
                                _a_prior_tn.lower, rng, n_items=I,
                                offset=_offset, common=_common_a[r],
                            )
                        else:
                            log_prior_fn = _resolve_a_prior_fn(r)
                            key_a = f"a_{r}"
                            step  = adapt_states[key_a].step_size if key_a in adapt_states else 0.3
                            a_new, accepted = _mh_discrimination(
                                a_r, Z[r], theta_r, params["b"][g_b],
                                T_r, row_idx, col_idx, log_prior_fn, step, rng,
                                offset=_offset, common=_common_a[r],
                            )
                            params["a"][g_a] = a_new
                            if key_a in adapt_states and t < adapt_until:
                                adapt_states[key_a] = adapt_step(
                                    adapt_states[key_a], float(np.mean(accepted)), t)

                # Likelihood cache uses current theta (not yet updated for this iter)
                # [moved to storage-time refresh] cache for augmentable
                # nodes is recomputed once per stored sample.

            # ----------------------------------------------------------
            # Pure MH block: logit / GPCM nodes
            # ----------------------------------------------------------
            else:
                # theta and b of non-augmented nodes are updated after the
                # node loop, person by person and item by item, against every
                # node that shares them (_theta_mh_dims / _b_mh_groups).  Only
                # this node's own a and c are updated here.

                # ----------------------------------------------------------
                # Discrimination a and guessing c for this non-augmentable
                # node (logit 2PL/3PL/GPCM-with-a).  Metropolis-within-Gibbs:
                # theta,b were just updated above with a,c fixed; now update a
                # (then c) against the node likelihood with theta,b fixed.
                # These are the ONLY places a_i / c_i move for a logit node —
                # the augmented-Z paths never touch a logit node.
                # ----------------------------------------------------------
                nm_r = node_models[r]
                # dispatch.has_discrimination is False for every guessing node
                # (it steers the probit 2PL path), so ask the node model.
                if node_models[r].has_discrimination and params["a"][g_a] is not None \
                        and not (a_node_in_shared[r] or a_node_in_a_corr[r]):
                    key_a  = f"a_mh_{r}"
                    step_a = adapt_states[key_a].step_size if key_a in adapt_states else 0.3
                    a_new, acc_a = _mh_logit_discrimination(
                        nm_r, data.Y_star[r], data.T[r],
                        params["theta"][d], params["b"][g_b],
                        params["a"][g_a], params["c"][g_c], _offset,
                        priors.resolve("a", r), step_a, rng,
                        common=bool(getattr(nm_r, "common_discrimination", False)),
                    )
                    params["a"][g_a] = a_new
                    if key_a in adapt_states and t < adapt_until:
                        adapt_states[key_a] = adapt_step(adapt_states[key_a], acc_a, t)

                if d_info.has_guessing and params["c"][g_c] is not None \
                        and not c_node_in_shared[r]:
                    key_c  = f"c_mh_{r}"
                    step_c = adapt_states[key_c].step_size if key_c in adapt_states else 0.3
                    c_new, acc_c = _mh_logit_guessing(
                        nm_r, data.Y_star[r], data.T[r],
                        params["theta"][d], params["b"][g_b],
                        params["a"][g_a], params["c"][g_c], _offset,
                        priors.resolve("c", r), step_c, rng,
                    )
                    params["c"][g_c] = c_new
                    if key_c in adapt_states and t < adapt_until:
                        adapt_states[key_c] = adapt_step(adapt_states[key_c], acc_c, t)

                # Refresh this node's cached log-likelihood with the updated
                # a / c so the stored log-likelihood trace and any later reader
                # are consistent.
                if node_models[r].has_discrimination or d_info.has_guessing:
                    cache.update(r, _log_node_lik_direct(
                        data.Y_star[r], data.T[r],
                        params["theta"][d], params["b"][g_b], nm_r,
                        params["a"][g_a], params["c"][g_c], offset=_offset,
                    ))

        # === Pooled a draw for multi-node sharing groups ===
        # z_pir = a_i * d_pir + eps  with d_pir = theta_pr - b_ir, so a shared
        # a_i is a regression coefficient informed by every member node:
        #   sigma*_i^2 = 1 / (1/var + sum_r sum_p d_pir^2)
        #   mu*_i      = sigma*_i^2 * (mean/var + sum_r sum_p d_pir * z_pir)
        # truncated to the prior's support.  For a LogNormal / unbounded Normal
        # prior the Gibbs form does not apply and a pooled MH step is used, whose
        # acceptance ratio sums the augmented log-likelihood over member nodes.
        for g in a_multi_groups:
            if a_node_in_a_corr[_a_groups_all[g][0]]:
                continue          # updated in the correlated-a block
            member_nodes = [r_m for r_m in _a_groups_all[g]
                            if dispatch[r_m].has_discrimination
                            and params["a"][g] is not None
                            and r_m in ab_idx_by_node]
            logit_members = [r_m for r_m in _a_groups_all[g]
                             if not dispatch[r_m].is_augmentable
                             and params["a"][g] is not None]
            if not member_nodes and not logit_members:
                continue
            rep = (member_nodes or logit_members)[0]

            def _parts():
                # Z = a (theta - b + offset) + eps on the latent mask, so the
                # regressor for a is d = theta - b + offset (raw offset).
                # Subtracting the raw offset from Z instead is right only
                # when a = 1.
                out = []
                for r_m in member_nodes:
                    ri, ci = ab_idx_by_node[r_m]
                    if len(ri) == 0:
                        continue
                    off_m = _node_offset(r_m, params, cov_index, data)
                    th_m  = params["theta"][node_to_dim[r_m]]
                    b_m   = params["b"][b_node_to_g[r_m]]
                    d_pi  = th_m[ri] - b_m[ci]
                    if off_m is not None:
                        d_pi = d_pi + off_m[ri, ci]
                    out.append((ri, ci, Z[r_m], d_pi))
                return out

            if logit_members:
                # Item-wise MH on xi = log a against EVERY member: the logit
                # members' column log-likelihood + the probit members'
                # Gaussian latent term + the natural-scale a prior + the
                # log-scale Jacobian (see _a_logprior_natural).
                parts   = _parts()
                offs_l  = {r_m: _node_offset(r_m, params, cov_index, data)
                           for r_m in logit_members}
                prior_a = priors.resolve("a", rep)
                a_curr  = np.asarray(params["a"][g], float)
                key_a   = f"a_shared_{g}"
                step    = adapt_states[key_a].step_size if key_a in adapt_states else 0.3

                def _target(a_vec):
                    out = _a_logprior_natural(a_vec, prior_a) + np.log(a_vec)
                    for ri, ci, Z_m, d_pi in parts:
                        diff = Z_m[ri, ci] - a_vec[ci] * d_pi
                        out = out + np.bincount(ci, weights=-0.5 * diff ** 2, minlength=I)
                    for r_m in logit_members:
                        lp = node_models[r_m].log_prob(
                            data.Y_star[r_m], params["theta"][node_to_dim[r_m]],
                            params["b"][b_node_to_g[r_m]], a_r=a_vec,
                            c_r=params["c"][c_node_to_g[r_m]], offset=offs_l[r_m])
                        out = out + np.where(data.T[r_m], lp, 0.0).sum(axis=0)
                    return out

                a_prop = np.exp(np.log(a_curr) + step * rng.standard_normal(I))
                acc = np.log(rng.uniform(size=I)) < (_target(a_prop) - _target(a_curr))
                params["a"][g] = np.where(acc, a_prop, a_curr)
                if key_a in adapt_states and t < adapt_until:
                    adapt_states[key_a] = adapt_step(
                        adapt_states[key_a], float(np.mean(acc)), t)
                for r_m in logit_members:
                    cache.update(r_m, _log_node_lik_direct(
                        data.Y_star[r_m], data.T[r_m],
                        params["theta"][node_to_dim[r_m]],
                        params["b"][b_node_to_g[r_m]], node_models[r_m],
                        params["a"][g], params["c"][c_node_to_g[r_m]],
                        offset=offs_l[r_m]))
                continue

            if a_prior_is_truncnormal[g]:
                _p  = priors.resolve("a", rep)
                m0  = float(getattr(_p, "mean", 1.0))
                v0  = float(getattr(_p, "variance", 0.5))
                low = getattr(_p, "lower", 0.0)
                sd2 = np.zeros(I, dtype=np.float64)
                sdz = np.zeros(I, dtype=np.float64)
                for ri, ci, Z_m, d_pi in _parts():
                    np.add.at(sd2, ci, d_pi ** 2)
                    np.add.at(sdz, ci, d_pi * Z_m[ri, ci])
                s2_a = 1.0 / (1.0 / v0 + sd2)
                mu_a = s2_a * (m0 / v0 + sdz)
                sd_a = np.sqrt(s2_a)
                lo_s = (-np.inf if low is None else (low - mu_a) / sd_a)
                U_a  = _Phi(lo_s) + (1.0 - _Phi(lo_s)) * rng.uniform(size=I)
                params["a"][g] = mu_a + sd_a * _Phi_inv(np.clip(U_a, 1e-14, 1 - 1e-14))
            else:
                parts   = _parts()
                a_curr  = params["a"][g]
                key_a   = f"a_shared_{g}"
                step    = adapt_states[key_a].step_size if key_a in adapt_states else 0.3
                a_prop  = np.exp(np.log(a_curr) + step * rng.standard_normal(I))
                lpf     = _resolve_a_prior_fn(rep)   # per-item [I]

                def _ll_peritem(a_vec):
                    # a_i conditionally independent across items even when the
                    # group pools several nodes: accumulate each node's cells
                    # into per-item bins, then accept/reject per item.
                    out = np.zeros(I, dtype=np.float64)
                    for ri, ci, Z_m, d_pi in parts:
                        diff = Z_m[ri, ci] - a_vec[ci] * d_pi
                        np.add.at(out, ci, -0.5 * diff ** 2)
                    return out

                log_alpha = (_ll_peritem(a_prop) + np.asarray(lpf(a_prop))) - \
                            (_ll_peritem(a_curr) + np.asarray(lpf(a_curr)))   # [I]
                acc = np.log(rng.uniform(size=I)) < log_alpha                 # [I]
                params["a"][g] = np.where(acc, a_prop, a_curr)
                if key_a in adapt_states and t < adapt_until:
                    adapt_states[key_a] = adapt_step(
                        adapt_states[key_a], float(np.mean(acc)), t)


        # === Pooled c draw for multi-node sharing groups ===
        # Beta-Bernoulli conjugacy pools additively over member nodes:
        #   c_i | U ~ Beta(alpha + sum_r n_guess_ir,
        #                  beta  + sum_r (n_relevant_ir - n_guess_ir))
        for g in c_multi_groups:
            member_nodes = [r_m for r_m in _c_groups_all[g] if r_m in U_by_node]
            if not member_nodes:
                continue
            _pc  = priors.resolve("c", member_nodes[0])
            a_c  = float(getattr(_pc, "alpha", 5.0))
            b_c  = float(getattr(_pc, "beta", 17.0))
            n_gs = np.zeros(I, dtype=np.float64)
            n_rl = np.zeros(I, dtype=np.float64)
            for r_m in member_nodes:
                ri, ci = data.row_idx[r_m], data.col_idx[r_m]
                np.add.at(n_gs, ci, U_by_node[r_m][ri, ci].astype(np.float64))
                np.add.at(n_rl, ci, 1.0)
            params["c"][g] = rng.beta(a_c + n_gs, b_c + (n_rl - n_gs))

        # === a.variance MH block (non-conjugate; see priors._check_variance_conjugacy) ===
        # Random walk on log(sigma^2) so the proposal respects positivity and
        # the Jacobian is d(sigma^2)/d(log sigma^2) = sigma^2, contributing
        # log(prop) - log(curr) to the acceptance ratio.  Target is the
        # truncated-Normal likelihood of the discriminations times the
        # InvGamma prior on the variance.
        for g, est in enumerate(a_var_estimated):
            if not est or (a_reps[g] is not None and a_node_in_a_corr[a_reps[g]]):
                continue          # correlated groups: updated in the a.corr block
            a_g = params["a"][g]
            if a_g is None or np.size(a_g) == 0:
                continue

            rep_r  = a_reps[g] if a_reps[g] is not None else 0
            aprior = priors.resolve("a", rep_r)
            mu_a   = float(getattr(aprior, "mean", 1.0))
            lo_a   = getattr(aprior, "lower", None)
            hi_a   = getattr(aprior, "upper", None)
            vprior = getattr(aprior, "variance", None)

            key   = f"a.variance_{g}"
            state = adapt_states[key]
            curr  = float(params["a.variance"][g])
            prop  = float(np.exp(math.log(curr) + state.step_size * rng.standard_normal()))

            lp_curr = (_log_trunc_normal_density(a_g, mu_a, curr, lo_a, hi_a)
                       + _log_variance_hyperprior(curr, vprior)
                       + math.log(curr))          # Jacobian
            lp_prop = (_log_trunc_normal_density(a_g, mu_a, prop, lo_a, hi_a)
                       + _log_variance_hyperprior(prop, vprior)
                       + math.log(prop))          # Jacobian

            accepted = math.log(max(rng.uniform(), 1e-300)) < (lp_prop - lp_curr)
            if accepted:
                params["a.variance"][g] = prop

            # Step-size adaptation is frozen after warmup, but acceptance is
            # tallied for the whole run so accept_rates reports the realised
            # rate rather than the warmup-only rate.
            if t < adapt_until:
                adapt_states[key] = adapt_step(adapt_states[key], accepted, t)
            else:
                st = adapt_states[key]
                adapt_states[key] = dataclasses.replace(
                    st,
                    n_accepted=st.n_accepted + int(accepted),
                    n_proposed=st.n_proposed + 1,
                )

        # === Pooled b draw for multi-node sharing groups ===
        # A shared b array must be drawn from ONE full conditional that pools
        # sufficient statistics over every member node:
        #
        #   sigma*_i^2 = 1 / (1/tau2 + sum_{r in g} sum_p T_r[p,i] a_ir^2)
        #   mu*_i      = sigma*_i^2 * sum_{r in g} a_ir sum_p (a_ir theta_pr - Z_pir)
        #
        # The per-node scalar draw is skipped for these groups (see
        # b_node_in_shared), so this is the only place they are updated.
        for g in b_multi_groups:
            if g in _b_mh_set or b_node_in_b_corr[_b_groups_all[g][0]]:
                continue          # item-wise MH below / correlated draw later
            member_nodes = _b_groups_all[g]
            if isinstance(params["b"][g], NodeBParams) or np.ndim(params["b"][g]) != 1:
                continue          # shared thresholds: pooled threshold block below
            tau2_g = params["b.variance"][g]
            # Pooled over every augmented member, on the cells where Z was
            # drawn (U = 0 at 3PL nodes; guess cells carry no Z) and with the
            # a-scaled offset.
            prec, stat = _b_gauss_stats(member_nodes)
            mu_b_g = float(getattr(priors.resolve("b", member_nodes[0]), "mean", 0.0))
            s2_b = 1.0 / (1.0 / tau2_g + prec)
            params["b"][g] = rng.normal(s2_b * (mu_b_g / tau2_g + stat), np.sqrt(s2_b))


        # === Logit/GPCM nodes: helpers for the person- and item-wise MH ===
        def _a_cur(r):
            return (params["a"][a_node_to_g[r]]
                    if a_node_to_g[r] is not None else None)

        def _masked_lp(r, off, theta=None, b=None):
            """Node r's [P, I] log-probabilities, zero on irrelevant cells."""
            lp = node_models[r].log_prob(
                data.Y_star[r],
                params["theta"][node_to_dim[r]] if theta is None else theta,
                params["b"][b_node_to_g[r]] if b is None else b,
                a_r=_a_cur(r), c_r=params["c"][c_node_to_g[r]], offset=off)
            return np.where(data.T[r], lp, 0.0)

        def _logit_info(nodes, axis):
            """1/4 sum a^2 over each person's (axis=1) or item's (axis=0)
            relevant cells: the logit Fisher-information bound."""
            out = np.zeros(P if axis == 1 else I)
            for r in nodes:
                a_r = _a_cur(r)
                a2 = np.ones(I) if a_r is None else np.asarray(a_r) ** 2
                Tw = data.T[r] * a2[None, :]
                out += 0.25 * Tw.sum(axis=axis)
            return out

        def _refresh_logit_cache(nodes):
            for r in nodes:
                cache.update(r, _log_node_lik_direct(
                    data.Y_star[r], data.T[r], params["theta"][node_to_dim[r]],
                    params["b"][b_node_to_g[r]], node_models[r], _a_cur(r),
                    params["c"][c_node_to_g[r]],
                    offset=_node_offset(r, params, cov_index, data)))

        def _b_prior(g, which):
            pr = priors.resolve(which, _b_groups_all[g][0])
            v = getattr(pr, "variance", 1.0)
            v = float(params["b.variance"][g]) if isinstance(v, PriorSpec) else float(v)
            return float(getattr(pr, "mean", 0.0)), v

        def _step(key):
            return adapt_states[key].step_size if key in adapt_states else 1.0

        def _adapt(key, acc):
            if key in adapt_states and t < adapt_until:
                adapt_states[key] = adapt_step(adapt_states[key], float(np.mean(acc)), t)

        # === Item-wise MH for b-groups with a logit/GPCM member ===========
        # Item i's target: prior N(mu, tau2_g) (current hierarchical variance)
        # + Gaussian latent term of each probit member (binary b only; exact
        # given Z) + each logit member's column log-likelihood.
        _all_items = np.arange(I)
        for g in _b_mh_groups:
            members = _b_groups_all[g]
            if b_node_in_b_corr[members[0]] or g in _tau_pooled_set:
                continue          # correlated-b block / pooled threshold block
            lm = [r for r in members if not dispatch[r].is_augmentable]
            pm = [r for r in members if dispatch[r].is_augmentable]
            offs = {r: _node_offset(r, params, cov_index, data) for r in lm}
            Hl = _logit_info(lm, axis=0)
            b_cur = params["b"][g]

            def col_ll(b_val, cols):
                tot = np.zeros(I)
                for r in lm:
                    tot += _masked_lp(r, offs[r], b=b_val).sum(axis=0)
                return tot[cols]

            if isinstance(b_cur, NodeBParams):
                nm_g = node_models[members[0]]

                def _mk(bb, bp):
                    return NodeBParams(b_binary=bb, b_poly=bp, bin_idx=nm_g.bin_idx,
                                       poly_idx=nm_g.poly_idx, M_poly=nm_g.M_poly)
                bi, pi_ = np.asarray(nm_g.bin_idx), np.asarray(nm_g.poly_idx)
                mu, v = _b_prior(g, "b.binary")
                cur = np.asarray(b_cur.b_binary, float)[:, None]
                new, acc = _mh_diag_units(
                    cur, np.full(cur.shape, 1 / v), np.full(cur.shape, mu / v),
                    1 / v + Hl[bi][:, None],
                    lambda vals: col_ll(_mk(vals[:, 0], b_cur.b_poly), bi),
                    _step(f"b_mh_{g}"), rng)
                _adapt(f"b_mh_{g}", acc)
                b_bin_new = new[:, 0]
                mu, v = _b_prior(g, "b.poly")
                cur = np.asarray(b_cur.b_poly, float)
                new, acc = _mh_diag_units(
                    cur, np.full(cur.shape, 1 / v), np.full(cur.shape, mu / v),
                    1 / v + Hl[pi_][:, None] + 0 * cur,
                    lambda vals: col_ll(_mk(b_bin_new, vals), pi_),
                    _step(f"bpoly_mh_{g}"), rng)
                _adapt(f"bpoly_mh_{g}", acc)
                params["b"][g] = _mk(b_bin_new, new)
            else:
                b_arr = np.asarray(b_cur, float)
                mu, v = _b_prior(g, "b")
                if b_arr.ndim == 1:
                    prec_q = np.full(I, 1 / v)
                    rhs_q = np.full(I, mu / v)
                    if pm:
                        # Probit member r: E = Z - a(theta + offset) = -a b + eps
                        E_neg = [None] * R
                        masks = [None] * R
                        for r in pm:
                            a_r = _a_cur(r)
                            E = _latent_resid(r, ("x", "q", "w"))
                            E = E - (b_arr[None, :] if a_r is None
                                     else a_r[None, :] * b_arr[None, :])
                            E_neg[r] = -E
                            masks[r] = _latent_view(r)[0]
                        n_l, S_l = _gibbs.missing_unit_stats(
                            "q", _all_items, E_neg, masks, [_a_cur(r) for r in range(R)])
                        for r in pm:
                            prec_q += n_l[r]
                            rhs_q += S_l[r]
                    new, acc = _mh_diag_units(
                        b_arr[:, None], prec_q[:, None], rhs_q[:, None],
                        (prec_q + Hl)[:, None],
                        lambda vals: col_ll(vals[:, 0], _all_items),
                        _step(f"b_mh_{g}"), rng)
                    params["b"][g] = new[:, 0]
                else:
                    new, acc = _mh_diag_units(
                        b_arr, np.full(b_arr.shape, 1 / v), np.full(b_arr.shape, mu / v),
                        1 / v + Hl[:, None] + 0 * b_arr,
                        lambda vals: col_ll(vals, _all_items),
                        _step(f"b_mh_{g}"), rng)
                    params["b"][g] = new
                _adapt(f"b_mh_{g}", acc)
            _refresh_logit_cache(lm)

        def _pooled_mixed_b(g, members, grm_m, lm, offs):
            """Shared b of mixed-format nodes (binary + polytomous items).

            Binary difficulties: at a mixed GRM member, on the binary columns,
            Z = theta - b + offset + eps, so the probit members give Gaussian
            statistics N_i = #cells, S_i = sum (theta + offset - Z).  All
            members probit: exact Gibbs draw.  Otherwise per-item MH whose
            target adds the GPCM members' column log-likelihood.
            Polytomous thresholds: pooled Cowles step (all members' marginal
            likelihoods), the GRM members' polytomous Z redrawn, then the
            exact pooled Gibbs step when every member is GRM.
            """
            nm0 = node_models[members[0]]
            bi, pi_ = np.asarray(nm0.bin_idx), np.asarray(nm0.poly_idx)
            bcur = params["b"][g]

            def mk(bb, bp):
                return NodeBParams(b_binary=np.asarray(bb), b_poly=np.asarray(bp),
                                   bin_idx=nm0.bin_idx, poly_idx=nm0.poly_idx,
                                   M_poly=nm0.M_poly)

            # --- binary difficulties ---
            mu_b, v_b = _b_prior(g, "b.binary")
            N_ = np.zeros(len(bi))
            S_ = np.zeros(len(bi))
            for r_ in grm_m:
                th = params["theta"][node_to_dim[r_]]
                pred = np.repeat(th[:, None], len(bi), axis=1)
                if offs[r_] is not None:
                    pred = pred + offs[r_][:, bi]
                Tb = data.T[r_][:, bi]
                N_ += Tb.sum(axis=0)
                S_ += np.where(Tb, pred - Z[r_][:, bi], 0.0).sum(axis=0)
            prec_b = 1.0 / v_b + N_
            rhs_b = mu_b / v_b + S_
            if not lm:
                b_bin = rng.normal(rhs_b / prec_b, 1.0 / np.sqrt(prec_b))
            else:
                def ll_bin(vals):
                    tot = np.zeros(I)
                    for r_ in lm:
                        tot += _masked_lp(r_, offs[r_], b=mk(vals[:, 0], bcur.b_poly)).sum(axis=0)
                    return tot[bi]
                new, acc = _mh_diag_units(
                    np.asarray(bcur.b_binary, float)[:, None], prec_b[:, None],
                    rhs_b[:, None], (prec_b + _logit_info(lm, axis=0)[bi])[:, None],
                    ll_bin, _step(f"bbin_mh_{g}"), rng)
                _adapt(f"bbin_mh_{g}", acc)
                b_bin = new[:, 0]

            # --- polytomous thresholds ---
            mu_t, var_t = _b_prior(g, "b.poly")

            def cl_poly(Tm):
                tot = np.zeros(I)
                for r_ in members:
                    tot += _masked_lp(r_, offs[r_], b=mk(b_bin, Tm)).sum(axis=0)
                return tot[pi_]

            new_t, acc_t = _mh_ordered_thresholds(
                bcur.b_poly, cl_poly, mu_t, var_t, _step(f"tau_mh_{g}"), rng,
                scale=_tau_scale[g])
            _adapt(f"tau_mh_{g}", acc_t)
            params["b"][g] = mk(b_bin, new_t)
            for r_ in grm_m:
                off_p = offs[r_][:, pi_] if offs[r_] is not None else None
                Z[r_][:, pi_] = _gibbs.sample_Z_grm(
                    data.Y_star[r_][:, pi_], params["theta"][node_to_dim[r_]], new_t,
                    data.T[r_][:, pi_], poly_row_idx[r_], poly_col_idx[r_],
                    nm0.M_poly, rng, offset=off_p)
            if not lm:
                Zs = np.concatenate([Z[r_][:, pi_] for r_ in grm_m], axis=0)
                Ys = np.concatenate([data.Y_star[r_][:, pi_] for r_ in grm_m], axis=0)
                Ts = np.concatenate([data.T[r_][:, pi_] for r_ in grm_m], axis=0)
                ri_s, ci_s = np.where(Ts)
                params["b"][g] = mk(b_bin, _gibbs.gibbs_thresholds_grm(
                    Zs, Ys, Ts, ri_s, ci_s, nm0.M_poly, rng,
                    old_thresholds=new_t, mu_0=mu_t, tau2=var_t))
            _refresh_logit_cache(lm)

        # === Shared GRM thresholds (items 14, 17) ==========================
        # One Cowles step per shared group: the target sums every member's
        # MARGINAL log-likelihood (GRM with Z integrated out, GPCM directly)
        # under the ordered N(mu0, tau2) prior; then each GRM member's Z is
        # redrawn given the new thresholds.  When every member is GRM, an
        # exact Gibbs step given all members' Z follows (bounds pooled by
        # stacking the members on the person axis).  Updating the shared
        # array once from every member is what makes it a draw from the
        # pooled conditional: per-member updates would let the last member
        # overwrite the others.
        for g in _tau_pooled:
            members = _b_groups_all[g]
            grm_m = [r_ for r_ in members if dispatch[r_].is_augmentable]
            lm = [r_ for r_ in members if not dispatch[r_].is_augmentable]
            mu_t, var_t = _b_prior(g, "b.poly")
            offs = {r_: _node_offset(r_, params, cov_index, data) for r_ in members}
            if isinstance(params["b"][g], NodeBParams):
                _pooled_mixed_b(g, members, grm_m, lm, offs)
                continue

            def _cl_pool(Tm, members=members, offs=offs):
                tot = np.zeros(I)
                for r_ in members:
                    tot += _masked_lp(r_, offs[r_], b=Tm).sum(axis=0)
                return tot

            new_t, acc_t = _mh_ordered_thresholds(
                params["b"][g], _cl_pool, mu_t, var_t, _step(f"tau_mh_{g}"), rng,
                scale=_tau_scale[g])
            params["b"][g] = new_t
            _adapt(f"tau_mh_{g}", acc_t)
            for r_ in grm_m:
                Z[r_] = _gibbs.sample_Z_grm(
                    data.Y_star[r_], params["theta"][node_to_dim[r_]], params["b"][g],
                    data.T[r_], data.row_idx[r_], data.col_idx[r_],
                    node_models[r_].n_outcomes, rng, offset=offs[r_])
            if not lm:
                Zs = np.concatenate([Z[r_] for r_ in grm_m], axis=0)
                Ys = np.concatenate([data.Y_star[r_] for r_ in grm_m], axis=0)
                Ts = np.concatenate([data.T[r_] for r_ in grm_m], axis=0)
                ri_s, ci_s = np.where(Ts)
                params["b"][g] = _gibbs.gibbs_thresholds_grm(
                    Zs, Ys, Ts, ri_s, ci_s, node_models[grm_m[0]].n_outcomes, rng,
                    old_thresholds=params["b"][g], mu_0=mu_t, tau2=var_t)
            _refresh_logit_cache(lm)

        # === Unified theta update for ALL latent dimensions (shared or independent) ===
        # Per-node latent statistics, summed per dimension (node_to_latent
        # sharing), then one joint Gibbs draw with the current Sigma.
        _a_by_node = [
            (params["a"][a_node_to_g[r]] if a_node_to_g[r] is not None else None)
            for r in range(R)
        ]

        # Per-node theta statistics from the latent view: on node r's latent
        # mask (U = 0 cells at a 3PL node),
        #     E = Z - a (core - theta + offset) = a theta + eps,
        # so n_pr = sum_i a_i^2 and S_pr = sum_i a_i E.  core is theta - b
        # (binary), theta (GRM) or theta - b_binary on the binary items of a
        # mixed GRM node.  The 3PL guess cells (Z = 0) are excluded, and the
        # binary items of a mixed GRM node keep their b_binary.
        S = np.zeros((P, R))
        n_mat = np.zeros((P, R))
        _persons = np.arange(P)
        for _rr in range(R):
            _view = _latent_view(_rr)
            if _view is None:
                continue
            _mask, _a_rr, _core = _view
            _th = params["theta"][node_to_dim[_rr]]
            _E = _latent_resid(_rr, ("x", "q", "w"))
            _E = _E + (_th[:, None] if _a_rr is None else _a_rr[None, :] * _th[:, None])
            _El = [None] * R
            _Ml = [None] * R
            _El[_rr], _Ml[_rr] = _E, _mask
            _nl, _Sl = _gibbs.missing_unit_stats("x", _persons, _El, _Ml,
                                                 [_a_rr if j == _rr else None for j in range(R)])
            n_mat[:, _rr] = _nl[_rr]
            S[:, _rr] = _Sl[_rr]
        # S, n_mat come back at node width (R); reduce to dimension width (D)
        # before the correlated draw, since Sigma is D x D. No-op when D == R.
        S, n_mat = _gibbs.reduce_suff_stats_to_dims(S, n_mat, dim_to_nodes)

        Sigma  = _theta_Sigma()
        _Lam_theta = np.linalg.inv(np.atleast_2d(Sigma))
        _need_cond = bool(_theta_mh_dims) and bool(gibbs_owned_dims) and D > 1 and \
            np.any(_Lam_theta[np.ix_(_theta_mh_dims, gibbs_owned_dims)] != 0)
        if _need_cond:
            # Probit-only dimensions drawn GIVEN the current values of the
            # MH-owned dimensions (correlated Sigma), not from a joint draw
            # that would treat them as unknown.
            _th_now = np.column_stack([params["theta"][d_] for d_ in range(D)])
            _draw = _gibbs.gibbs_theta_given(S, n_mat, Sigma, _th_now,
                                             gibbs_owned_dims, rng)
            for k_, d_ in enumerate(gibbs_owned_dims):
                params["theta"][d_] = _draw[:, k_]
        else:
            theta_matrix = _gibbs.gibbs_theta_correlated(S, n_mat, Sigma, rng)
            # Assign back to per-dimension theta arrays -- but ONLY for dimensions
            # whose nodes are all augmentable.
            #
            # A non-augmented (logit/GPCM) node has no Z, so its dimension has
            # no Gibbs statistics from that node; such dimensions (even when
            # shared with probit nodes) are drawn by the person-wise MH below,
            # whose target includes every node on the dimension.
            for d in gibbs_owned_dims:
                params["theta"][d] = theta_matrix[:, d]

        # === Person-wise MH for dimensions with a logit/GPCM node ===========
        # theta_pd's target: conditional prior given the person's other
        # dimensions (from Sigma^{-1}) + Gaussian latent term of every probit
        # node on d (exact given Z) + every logit node's row log-likelihood.
        _all_persons = np.arange(P)
        for d_ in _theta_mh_dims:
            nodes_d = dim_to_nodes[d_]
            lm = [r for r in nodes_d if not dispatch[r].is_augmentable]
            pm = [r for r in nodes_d if dispatch[r].is_augmentable]
            offs = {r: _node_offset(r, params, cov_index, data) for r in lm}
            th = params["theta"][d_]
            prec_q = np.full(P, _Lam_theta[d_, d_])
            rhs_q = np.zeros(P)
            for e_ in range(D):
                if e_ != d_ and _Lam_theta[d_, e_] != 0:
                    rhs_q -= _Lam_theta[d_, e_] * params["theta"][e_]
            if pm:
                # Probit node r: E = Z - a(core - theta + offset) = a theta + eps
                E_t = [None] * R
                masks = [None] * R
                for r in pm:
                    a_r = _a_cur(r)
                    E = _latent_resid(r, ("x", "q", "w"))
                    E_t[r] = E + (th[:, None] if a_r is None else a_r[None, :] * th[:, None])
                    masks[r] = _latent_view(r)[0]
                n_l, S_l = _gibbs.missing_unit_stats(
                    "x", _all_persons, E_t, masks, [_a_cur(r) for r in range(R)])
                for r in pm:
                    prec_q += n_l[r]
                    rhs_q += S_l[r]

            def row_ll(vals, lm=lm, offs=offs):
                tot = np.zeros(P)
                for r in lm:
                    tot += _masked_lp(r, offs[r], theta=vals[:, 0]).sum(axis=1)
                return tot

            new, acc = _mh_diag_units(
                th[:, None], prec_q[:, None], rhs_q[:, None],
                (prec_q + _logit_info(lm, axis=1))[:, None], row_ll,
                _step(f"theta_mh_{d_}"), rng)
            params["theta"][d_] = new[:, 0]
            _adapt(f"theta_mh_{d_}", acc)
            _refresh_logit_cache(lm)

        # Current theta of every dimension, for the correlation updates below.
        theta_matrix = np.column_stack([params["theta"][d_] for d_ in range(D)])

        # === Location move along the theta/b ridge =========================
        # Shifting theta_d, every difficulty/threshold of the b-groups on
        # dimension d, and the GRM latent responses by one constant c leaves
        # every likelihood term unchanged (theta - b, Z - tau and Z - theta are
        # invariant; binary Z ~ a(theta - b) is untouched).  Only the priors
        # depend on c, so drawing c from its exact Gaussian conditional and
        # applying the shift is a valid Gibbs step on a translation group
        # (Liu & Sabatti 2000).  It moves the common location the Gibbs and
        # Cowles steps can only cross slowly.  Applied when every member of
        # each b-group on d is on d (the shift then touches nothing else).
        for d_ in _shift_dims:
            nodes_d = dim_to_nodes[d_]
            prec_c = P * _Lam_theta[d_, d_]
            m_p = np.zeros(P)
            for e_ in range(D):
                if e_ != d_ and _Lam_theta[d_, e_] != 0:
                    m_p -= _Lam_theta[d_, e_] * params["theta"][e_] / _Lam_theta[d_, d_]
            lin_c = -_Lam_theta[d_, d_] * np.sum(params["theta"][d_] - m_p)
            parts = []
            for g_ in _shift_bgroups[d_]:
                bv = params["b"][g_]
                if isinstance(bv, NodeBParams):
                    for vals, which in ((bv.b_binary, "b.binary"), (bv.b_poly, "b.poly")):
                        mu_, v_ = _b_prior(g_, which)
                        prec_c += np.size(vals) / v_
                        lin_c -= np.sum(np.asarray(vals) - mu_) / v_
                else:
                    mu_, v_ = _b_prior(g_, "b.poly" if np.ndim(bv) == 2 else "b")
                    prec_c += np.size(bv) / v_
                    lin_c -= np.sum(np.asarray(bv) - mu_) / v_
            c_ = rng.normal(lin_c / prec_c, 1.0 / np.sqrt(prec_c))
            params["theta"][d_] = params["theta"][d_] + c_
            for g_ in _shift_bgroups[d_]:
                bv = params["b"][g_]
                if isinstance(bv, NodeBParams):
                    bv.b_binary = np.asarray(bv.b_binary) + c_
                    bv.b_poly = np.asarray(bv.b_poly) + c_
                else:
                    params["b"][g_] = np.asarray(bv) + c_
            for r_ in nodes_d:
                if not dispatch[r_].is_augmentable:
                    continue
                if dispatch[r_].is_mixed:
                    Z[r_][:, node_models[r_].poly_idx] += c_
                elif dispatch[r_].is_polytomous:
                    Z[r_] += c_
        if _shift_dims:
            theta_matrix = np.column_stack([params["theta"][d_] for d_ in range(D)])

        # === Sigma2 update per latent dimension (preserves HalfNormal MH exactly) ===
        # Runs *after* the new theta draw, as required by the HalfNormal target density.
        for d in range(D):
            if sigma2_prior_type[d] == "invgamma":
                params["sigma2"][d] = _gibbs.gibbs_sigma2(
                    params["theta"][d],
                    sigma2_alpha[d], sigma2_beta_prior[d], rng,
                )
            elif sigma2_prior_type[d] == "halfnormal":
                key_ls = f"log_sigma_{d}"
                step_s = adapt_states[key_ls].step_size if key_ls in adapt_states else 0.3
                ls_new, acc_s = _mh_sigma2_halfnormal(
                    params["theta"][d], halfnormal_scale[d],
                    params["log_sigma"][d], step_s, rng,
                )
                params["log_sigma"][d] = ls_new
                params["sigma2"][d]    = math.exp(2.0 * ls_new)
                if key_ls in adapt_states and t < adapt_until:
                    adapt_states[key_ls] = adapt_step(adapt_states[key_ls], acc_s, t)

        # --- Correlated theta update (only when needed and after Z is updated) ---
        corr_spec = priors.resolve("theta.correlation", None)
        is_estimated_corr = not isinstance(corr_spec, FixedCorrelation)

        if is_estimated_corr or (
            isinstance(corr_spec, FixedCorrelation) and
            not np.allclose(corr_spec.rho if np.isscalar(corr_spec.rho) else np.array(corr_spec.rho), 0.0)
        ):
            # theta was already drawn above with Sigma built from
            # params["corr_matrix"], so it accounts for correlation (estimated
            # or fixed-nonzero).  `theta_matrix` holds the current theta of
            # every dimension (after the MH dimensions and the location move),
            # so the correlation update below uses it directly.

            # Correlation / Sigma update (only if estimated)
            if is_estimated_corr:
                # One update per correlation block (theta_corr); without
                # theta_corr there is a single block of all D dimensions.
                if using_invwishart_theta:
                    # InvWishart: one exact draw per block's covariance; the
                    # block's sigma2 are read from its diagonal.
                    S_full = np.array(params["corr_matrix"], dtype=np.float64, copy=True)
                    for b_, (Psi0_b_, nu0_b_) in zip(_th_blocks, _theta_iw_block_params, strict=True):
                        S_b = _gibbs.gibbs_Sigma_invwishart(
                            theta_matrix[:, b_], np.zeros(len(b_)), Psi0_b_, nu0_b_, rng)
                        S_full[np.ix_(b_, b_)] = S_b
                        for k_, d_ in enumerate(b_):
                            params["sigma2"][d_] = float(S_b[k_, k_])
                    params["corr_matrix"] = S_full
                else:
                    C_full = np.array(params["corr_matrix"], dtype=np.float64, copy=True)
                    k0 = 0                       # offset of the block's pairs
                    for b_ in _th_blocks:
                        G_b = len(b_)
                        n_pb = G_b * (G_b - 1) // 2
                        keys_b = [f"rho_{k0 + j}" for j in range(n_pb)]
                        s2_b = [params["sigma2"][d_] for d_ in b_]
                        if G_b == 2:
                            _lkj_log_prior_r2 = _theta_rho_lp
                            step_rho = (adapt_states[keys_b[0]].step_size
                                        if keys_b[0] in adapt_states else 0.1)
                            rho_new, acc_rho = _mh_rho_fisher_z(
                                theta_matrix[:, b_], s2_b, C_full[b_[0], b_[1]],
                                _lkj_log_prior_r2, step_rho, rng,
                            )
                            C_full[b_[0], b_[1]] = C_full[b_[1], b_[0]] = rho_new
                            acc_list = [acc_rho]
                        else:
                            step_list = [adapt_states[k_].step_size if k_ in adapt_states
                                         else 0.1 for k_ in keys_b]
                            C_b, acc_list = _mh_correlation_vine(
                                theta_matrix[:, b_], s2_b, C_full[np.ix_(b_, b_)],
                                lkj_eta, step_list, rng,
                            )
                            C_full[np.ix_(b_, b_)] = C_b
                        for k_, acc in zip(keys_b, acc_list, strict=True):
                            if k_ in adapt_states and t < adapt_until:
                                adapt_states[k_] = adapt_step(adapt_states[k_], acc, t)
                        k0 += n_pb
                    params["corr_matrix"] = C_full

        # --- b.variance update: one InvGamma draw per free b-group ---
        if any_b_tau2_free:
            for g in range(len(b_groups)):
                if not b_tau2_is_free[g]:
                    continue
                b_g = params["b"][g]
                # For mixed nodes, stack binary and polytomous thresholds
                # into one flat vector — all are drawn from Normal(0, tau2).
                rep_g = b_groups[g][0]
                if isinstance(b_g, NodeBParams):
                    b_g_vals = np.concatenate([
                        b_g.b_binary - float(getattr(priors.resolve("b.binary", rep_g), "mean", 0.0)),
                        b_g.b_poly.ravel() - float(getattr(priors.resolve("b.poly", rep_g), "mean", 0.0)),
                    ])
                else:
                    which_g = "b.poly" if np.ndim(b_g) == 2 else "b"
                    b_g_vals = (b_g.ravel()
                                - float(getattr(priors.resolve(which_g, rep_g), "mean", 0.0)))
                params["b.variance"][g] = _gibbs.gibbs_b_variance(
                    b_g_vals, tau2_alpha_priors[g], tau2_beta_priors[g], rng,
                )

        # === b.corr joint draws (post-node-loop, after b.variance) =========
        # For each b-correlation group, replace the independently drawn b arrays
        # with a joint G-dimensional MVN draw conditioned on the current Sigma_b.
        if any_b_corr:
            for cg_idx, cg in enumerate(b_corr_groups):
                G_cg = len(cg)
                # Build Sigma_b: either the stored full covariance (InvWishart)
                # or D R D from per-group tau2 and the correlation matrix (LKJ/Fixed).
                if b_corr_is_iw:
                    Sigma_b = params["b.corr_matrix"][cg_idx]
                else:
                    # DRD: diagonal of sqrt(tau2) values, scaled by stored corr matrix
                    tau2_vec = np.array([params["b.variance"][g] for g in cg],
                                        dtype=np.float64)
                    D_b  = np.diag(np.sqrt(tau2_vec))
                    Sigma_b = D_b @ params["b.corr_matrix"][cg_idx] @ D_b

                # Statistics pooled over EVERY node of each b-group, on the
                # cells where Z was drawn (not the 3PL guess cells, where Z is
                # 0), with the a-scaled covariate offset.
                S_cg = np.zeros((I, G_cg))
                N_cg = np.zeros((I, G_cg))
                for k in range(G_cg):
                    N_cg[:, k], S_cg[:, k] = _b_gauss_stats(_b_groups_all[cg[k]])
                # b_i ~ N(mu, Sigma_b) with each group's own prior mean: draw
                # the centred b - mu (statistic S - N mu) and add mu back.
                mu_cg = np.array([float(getattr(priors.resolve("b", _b_groups_all[g_][0]),
                                                "mean", 0.0)) for g_ in cg])
                Sc_cg = S_cg - N_cg * mu_cg[None, :]
                logit_k = [k for k in range(G_cg) if cg[k] in _b_mh_set]
                if not logit_k:
                    new_b_arrays = _gibbs.gibbs_b_correlated_from_stats(
                        Sc_cg, N_cg, Sigma_b, rng)
                    for k in range(G_cg):
                        params["b"][cg[k]] = new_b_arrays[k] + mu_cg[k]
                else:
                    # Probit groups drawn jointly GIVEN the logit
                    # groups; then each logit group by item-wise MH whose
                    # prior is the conditional b_ik | b_i,-k from Sigma_b.
                    owned = [k for k in range(G_cg) if k not in logit_k]
                    B_now = np.column_stack([np.asarray(params["b"][g_], float) for g_ in cg])
                    if owned:
                        drawn = _gibbs.gibbs_theta_given(
                            Sc_cg, N_cg, Sigma_b, B_now - mu_cg[None, :], owned, rng)
                        for j_, k in enumerate(owned):
                            params["b"][cg[k]] = drawn[:, j_] + mu_cg[k]
                            B_now[:, k] = drawn[:, j_] + mu_cg[k]
                    Lam_b = np.linalg.inv(Sigma_b)
                    for k in logit_k:
                        g_ = cg[k]
                        lm = [r_ for r_ in _b_groups_all[g_] if not dispatch[r_].is_augmentable]
                        offs = {r_: _node_offset(r_, params, cov_index, data) for r_ in lm}
                        prec_q = Lam_b[k, k] + N_cg[:, k]
                        rhs_q = (Lam_b[k, k] * mu_cg[k] + S_cg[:, k]
                                 - sum(Lam_b[k, h] * (B_now[:, h] - mu_cg[h])
                                       for h in range(G_cg) if h != k))

                        def col_ll_corr(vals, lm=lm, offs=offs):
                            tot = np.zeros(I)
                            for r_ in lm:
                                tot += _masked_lp(r_, offs[r_], b=vals[:, 0]).sum(axis=0)
                            return tot

                        new, acc = _mh_diag_units(
                            B_now[:, [k]], prec_q[:, None], rhs_q[:, None],
                            (prec_q + _logit_info(lm, axis=0))[:, None], col_ll_corr,
                            _step(f"b_mh_{g_}"), rng)
                        params["b"][g_] = new[:, 0]
                        B_now[:, k] = new[:, 0]
                        _adapt(f"b_mh_{g_}", acc)
                        _refresh_logit_cache(lm)

                # The likelihood cache of augmentable nodes is recomputed
                # once per stored sample (storage-time refresh), not here.

                # Update Sigma_b
                if b_corr_is_iw:
                    b_matrix = np.column_stack([params["b"][cg[k]] for k in range(G_cg)])
                    Psi0_b, nu0_b = b_corr_iw_params[cg_idx]
                    params["b.corr_matrix"][cg_idx] = _gibbs.gibbs_Sigma_invwishart(
                        b_matrix, mu_cg, Psi0_b, nu0_b, rng,
                    )
                elif b_corr_is_lkj:
                    # MH on scalar rho (G=2) or vine (G>2), on the centred b
                    b_matrix = (np.column_stack([params["b"][cg[k]] for k in range(G_cg)])
                                - mu_cg[None, :])
                    tau2_vec = np.array([params["b.variance"][g] for g in cg])
                    if G_cg == 2:
                        rho_curr = params["b.corr_matrix"][cg_idx][0, 1]
                        _b_lkj_prior = _b_rho_lp
                        key_b_rho = f"b_rho_{cg_idx}_0"
                        step_b = adapt_states[key_b_rho].step_size if key_b_rho in adapt_states else 0.1
                        rho_new, acc_b = _mh_rho_fisher_z(
                            b_matrix, tau2_vec.tolist(), rho_curr,
                            _b_lkj_prior, step_b, rng,
                        )
                        params["b.corr_matrix"][cg_idx] = np.array([[1.0, rho_new],
                                                                      [rho_new, 1.0]])
                        if key_b_rho in adapt_states and t < adapt_until:
                            adapt_states[key_b_rho] = adapt_step(adapt_states[key_b_rho], acc_b, t)
                    else:
                        step_list_b = [
                            adapt_states[f"b_rho_{cg_idx}_{idx}"].step_size
                            if f"b_rho_{cg_idx}_{idx}" in adapt_states else 0.1
                            for idx in range(G_cg * (G_cg - 1) // 2)
                        ]
                        C_new_b, acc_list_b = _mh_correlation_vine(
                            b_matrix, tau2_vec.tolist(),
                            params["b.corr_matrix"][cg_idx],
                            b_corr_lkj_eta, step_list_b, rng,
                        )
                        params["b.corr_matrix"][cg_idx] = C_new_b
                        for idx, acc in enumerate(acc_list_b):
                            key = f"b_rho_{cg_idx}_{idx}"
                            if key in adapt_states and t < adapt_until:
                                adapt_states[key] = adapt_step(adapt_states[key], acc, t)
                # FixedCorrelation: no update

        # === a.corr joint draws =============================================
        if any_a_corr:
            for cg_idx, cg in enumerate(a_corr_groups):
                G_cg = len(cg)
                rep_nodes = [a_reps[g] for g in cg]

                if a_corr_is_iw:
                    Sigma_a = params["a.corr_matrix"][cg_idx]
                else:
                    # For LKJ path: reconstruct Sigma_a from per-group variances
                    # a prior variance is used as the scale; resolve from representative node
                    var_vec = [_a_group_var(g_) for g_ in cg]
                    D_a = np.diag(np.sqrt(np.array(var_vec)))
                    Sigma_a = D_a @ params["a.corr_matrix"][cg_idx] @ D_a

                # Resolve mu_a from representative node priors
                mu_a_vec = np.array([
                    float(getattr(priors.resolve("a", rep_nodes[k]), "mean", 1.0))
                    for k in range(G_cg)
                ])

                # Componentwise exact Gibbs over the groups, vectorized
                # over items.  a_i ~ MVN(mu_a, Sigma_a) on the positive orthant,
                # so given the other groups a_ik has the conditional Normal prior
                # (precision Lam_kk, linear term Lam_kk mu_k - sum_h Lam_kh
                # (a_ih - mu_h)), truncated to a > 0.  A probit group adds its
                # Gaussian latent term (n, s) pooled over every member node on
                # the cells where Z was drawn, with the covariate offset in the
                # regressor; a group with a logit/GPCM member is updated by
                # per-item MH on log a with that prior.  Componentwise draws
                # need no rejection step, so the correlation is respected on
                # every iteration.
                Lam_a = np.linalg.inv(Sigma_a)
                A_now = np.column_stack([np.asarray(params["a"][g_], float) for g_ in cg])
                for k in range(G_cg):
                    g_ = cg[k]
                    members = _a_groups_all[g_]
                    lin = Lam_a[k, k] * mu_a_vec[k] - sum(
                        Lam_a[k, h] * (A_now[:, h] - mu_a_vec[h])
                        for h in range(G_cg) if h != k)
                    n_k, s_k = _a_gauss_stats(members)
                    lm = [r_ for r_ in members if not dispatch[r_].is_augmentable]
                    if not lm:
                        prec = Lam_a[k, k] + n_k
                        A_now[:, k] = _gibbs.truncnorm_draw(
                            (lin + s_k) / prec, 1.0 / np.sqrt(prec), 0.0, np.inf, rng)
                    else:
                        offs = {r_: _node_offset(r_, params, cov_index, data) for r_ in lm}
                        key_a = f"a_corr_mh_{g_}"
                        step = adapt_states[key_a].step_size if key_a in adapt_states else 0.3

                        def _tgt(av, lm=lm, offs=offs, lin=lin, n_k=n_k, s_k=s_k, k=k):
                            out = (lin + s_k) * av - 0.5 * (Lam_a[k, k] + n_k) * av * av \
                                  + np.log(av)                  # Jacobian of xi = log a
                            for r_ in lm:
                                lp = node_models[r_].log_prob(
                                    data.Y_star[r_], params["theta"][node_to_dim[r_]],
                                    params["b"][b_node_to_g[r_]], a_r=av,
                                    c_r=params["c"][c_node_to_g[r_]], offset=offs[r_])
                                out = out + np.where(data.T[r_], lp, 0.0).sum(axis=0)
                            return out

                        a_c = A_now[:, k]
                        a_p = np.exp(np.log(a_c) + step * rng.standard_normal(I))
                        acc = np.log(rng.uniform(size=I)) < (_tgt(a_p) - _tgt(a_c))
                        A_now[:, k] = np.where(acc, a_p, a_c)
                        if key_a in adapt_states and t < adapt_until:
                            adapt_states[key_a] = adapt_step(
                                adapt_states[key_a], float(np.mean(acc)), t)
                    params["a"][g_] = A_now[:, k].copy()
                    if lm:
                        _refresh_logit_cache(lm)

                # The likelihood cache of augmentable nodes is recomputed
                # once per stored sample (storage-time refresh), not here.

                # Update Sigma_a
                if a_corr_is_iw:
                    a_matrix = np.column_stack([params["a"][cg[k]] for k in range(G_cg)])
                    Psi0_a, nu0_a = a_corr_iw_params[cg_idx]
                    params["a.corr_matrix"][cg_idx] = _gibbs.gibbs_Sigma_invwishart(
                        a_matrix, mu_a_vec, Psi0_a, nu0_a, rng,
                    )
                # Estimated group variances: MH on log v_k with the
                # correlated MVN likelihood of the centred slopes,
                #   sum_i log N(a_i - mu; 0, D R D) + log p(v_k) + log v_k.
                # (Like the rho updates, this ignores the positive-orthant
                # normalising constant of the truncated MVN.)
                if not a_corr_is_iw:
                    A_c = (np.column_stack([params["a"][g_] for g_ in cg])
                           - mu_a_vec[None, :])
                    R_a = params["a.corr_matrix"][cg_idx]

                    def _mvn_ll(vv, A_c=A_c, R_a=R_a):
                        sd = np.sqrt(vv)
                        S_ = R_a * np.outer(sd, sd)
                        sign, logdet = np.linalg.slogdet(S_)
                        if sign <= 0:
                            return -np.inf
                        Q = np.linalg.solve(S_, A_c.T)
                        return -0.5 * A_c.shape[0] * logdet - 0.5 * float(np.sum(A_c.T * Q))

                    vv = np.array([_a_group_var(g_) for g_ in cg])
                    for k in range(G_cg):
                        g_ = cg[k]
                        if not params["a.variance.estimated"][g_]:
                            continue
                        key_v = f"a.variance_{g_}"
                        st_v = adapt_states[key_v].step_size if key_v in adapt_states else 0.3
                        vprior = getattr(priors.resolve("a", a_reps[g_]), "variance", None)
                        prop_v = vv.copy()
                        prop_v[k] = vv[k] * math.exp(st_v * rng.standard_normal())
                        la = ((_mvn_ll(prop_v) + _log_variance_hyperprior(prop_v[k], vprior)
                               + math.log(prop_v[k]))
                              - (_mvn_ll(vv) + _log_variance_hyperprior(vv[k], vprior)
                                 + math.log(vv[k])))
                        acc_v = math.log(max(rng.uniform(), 1e-300)) < la
                        if acc_v:
                            vv = prop_v
                            params["a.variance"][g_] = float(vv[k])
                        if key_v in adapt_states and t < adapt_until:
                            adapt_states[key_v] = adapt_step(adapt_states[key_v], acc_v, t)

                if a_corr_is_lkj:
                    # Centre at the prior mean: the rho updates evaluate
                    # MVN(0, D R D) (written for theta), and raw slopes near 1
                    # read as strong co-movement, pushing rho toward 1.
                    a_matrix = (np.column_stack([params["a"][cg[k]] for k in range(G_cg)])
                                - mu_a_vec[None, :])
                    var_vec_lkj = [_a_group_var(cg[k]) for k in range(G_cg)]
                    if G_cg == 2:
                        rho_curr_a = params["a.corr_matrix"][cg_idx][0, 1]
                        _a_lkj_prior = _a_rho_lp
                        key_a_rho = f"a_rho_{cg_idx}_0"
                        step_a = adapt_states[key_a_rho].step_size if key_a_rho in adapt_states else 0.1
                        rho_new_a, acc_a = _mh_rho_fisher_z(
                            a_matrix, var_vec_lkj, rho_curr_a,
                            _a_lkj_prior, step_a, rng,
                        )
                        params["a.corr_matrix"][cg_idx] = np.array([[1.0, rho_new_a],
                                                                      [rho_new_a, 1.0]])
                        if key_a_rho in adapt_states and t < adapt_until:
                            adapt_states[key_a_rho] = adapt_step(adapt_states[key_a_rho], acc_a, t)
                    else:
                        step_list_a = [
                            adapt_states[f"a_rho_{cg_idx}_{idx}"].step_size
                            if f"a_rho_{cg_idx}_{idx}" in adapt_states else 0.1
                            for idx in range(G_cg * (G_cg - 1) // 2)
                        ]
                        C_new_a, acc_list_a = _mh_correlation_vine(
                            a_matrix, var_vec_lkj,
                            params["a.corr_matrix"][cg_idx],
                            a_corr_lkj_eta, step_list_a, rng,
                        )
                        params["a.corr_matrix"][cg_idx] = C_new_a
                        for idx, acc in enumerate(acc_list_a):
                            key = f"a_rho_{cg_idx}_{idx}"
                            if key in adapt_states and t < adapt_until:
                                adapt_states[key] = adapt_step(adapt_states[key], acc, t)
                # FixedCorrelation: no update
        # ==========================================================================
        # Covariate coefficients (beta, kappa, gamma) and missing covariates
        # ==========================================================================
        # Each block is an exact Gibbs draw from a Normal linear regression on
        # the latent responses drawn in the per-node loop above.  At node r,
        # on the cells where Z_r was drawn (its latent mask),
        #
        #     Z_r = a_r * (core_r + X beta + Q kappa + W gamma) + eps,
        #
        # with core and mask set by the node type (see _latent_view and the
        # note above gibbs_samplers.gibbs_coef_pooled).  The residual for one
        # target removes every OTHER term:  E_r = Z_r - a_r (core_r + rest).
        #
        # Only augmented (probit) nodes enter.  Logit and GPCM nodes have no
        # latent response; their contribution is added by MH (items 3 and 4).
        # A coefficient group with no augmented member is left at its current
        # value here.
        #
        # Every block reads the CURRENT X/Q/W working copies (full
        # imputation), and the blocks run in sequence, each conditioning on
        # the latest draws of the others.
        # Rebuilt here: the a draws (per node, pooled, correlated) may have
        # run after the theta block that first built this list.
        _a_by_node = [
            (params["a"][a_node_to_g[r]] if a_node_to_g[r] is not None else None)
            for r in range(R)
        ]

        def _impute_missing(code, key, others, coef_full, prior):
            """Update the missing values of one covariate type in place.

            Target for unit u (person, item or cell), given everything else:

                log pi(v) = log N(v; m0, v0 I)                    (prior)
                          + Gaussian latent term of probit nodes  (exact given Z)
                          + sum_logit loglik_r(unit u's cells | v)

            The first two terms are (prec_u, rhs_u) from missing_terms.  Units
            with no relevant cell at a logit node get the exact Normal draw.
            The others get a random-walk proposal with covariance
            step^2 inv(prec_u + 1/4 logit information), all units proposed
            together and accepted one by one: the logit log-likelihood
            factorizes over units, so one pass over the matrix scores all.
            """
            units = _miss_units[code]
            m0, v0 = prior
            work = params[key]
            if code == "w":
                def get(arr):
                    return arr[units[:, 0], units[:, 1]]
                feat_obs = data.W_feature_observed[units[:, 0], units[:, 1]]
            else:
                def get(arr):
                    return arr[units]
                feat_obs = (data.X_feature_observed if code == "x"
                            else data.Q_feature_observed)[units]

            def put(arr, vals):
                if code == "w":
                    arr[units[:, 0], units[:, 1]] = vals
                else:
                    arr[units] = vals

            values = get(work).copy()
            resid = [_latent_resid(r, others) for r in range(R)]
            n_l, S_l = _gibbs.missing_unit_stats(code, units, resid, _mask_list, _a_by_node)
            groups = _gibbs.missing_terms(n_l, S_l, coef_full, values, feat_obs, m0, v0)

            logit_nodes = [r for r in range(R)
                           if not dispatch[r].is_augmentable and coef_full[r] is not None]
            if not logit_nodes:
                for sel, mi, prec, rhs in groups:
                    values[sel[:, None], mi[None, :]] = _gibbs.draw_from_terms(prec, rhs, rng)
                put(work, values)
                return

            # Logit information bound (p(1-p) <= 1/4) for the preconditioner,
            # from the same algebra with E = 0, mask = T and no prior.
            zr = [(np.zeros((P, I)) if r in logit_nodes else None) for r in range(R)]
            Tm = [(data.T[r] if r in logit_nodes else None) for r in range(R)]
            nL, SL = _gibbs.missing_unit_stats(code, units, zr, Tm, _a_by_node)
            lgroups = _gibbs.missing_terms(
                nL, SL, [(coef_full[r] if r in logit_nodes else None) for r in range(R)],
                values, feat_obs, 0.0, np.inf)
            touched = np.zeros(len(units), dtype=bool)
            for r in logit_nodes:
                touched |= nL[r] > 0

            def unit_ll(vals):
                """Per-unit logit log-likelihood with the units' values = vals."""
                arr = work.copy()
                put(arr, vals)
                p2 = dict(params)
                p2[key] = arr
                tot = np.zeros(len(units))
                for r in logit_nodes:
                    lp = node_models[r].log_prob(
                        data.Y_star[r], params["theta"][node_to_dim[r]],
                        params["b"][b_node_to_g[r]],
                        a_r=params["a"][a_node_to_g[r]],
                        c_r=params["c"][c_node_to_g[r]],
                        offset=_node_offset(r, p2, cov_index, data))
                    lpm = np.where(data.T[r], lp, 0.0)
                    if code == "x":
                        tot += lpm.sum(axis=1)[units]
                    elif code == "q":
                        tot += lpm.sum(axis=0)[units]
                    else:
                        tot += lpm[units[:, 0], units[:, 1]]
                return tot

            mh_key = f"{code.upper()}_missing_mh"
            step = adapt_states[mh_key].step_size if mh_key in adapt_states else 1.0
            values, acc = _mh_missing_units(
                values, groups, [lg[2] for lg in lgroups], touched, unit_ll, step, rng)
            if np.any(touched) and mh_key in adapt_states and t < adapt_until:
                adapt_states[mh_key] = adapt_step(
                    adapt_states[mh_key], float(np.mean(acc[touched])), t)
            put(work, values)
            for r in logit_nodes:
                cache.update(r, _log_node_lik_direct(
                    data.Y_star[r], data.T[r], params["theta"][node_to_dim[r]],
                    params["b"][b_node_to_g[r]], node_models[r],
                    params["a"][a_node_to_g[r]], params["c"][c_node_to_g[r]],
                    offset=_node_offset(r, params, cov_index, data)))

        def _coef_prior(kind, g, rep):
            """(tau2, hyperprior-or-None, prior mean) for a coefficient group."""
            pr = priors.resolve(kind, rep)
            v = getattr(pr, "variance", 1.0)
            mu0 = float(getattr(pr, "mean", 0.0))
            if isinstance(v, InvGammaPrior):
                return float(params[f"{kind}.variance"][g]), v, mu0
            return float(v), None, mu0

        if cov_index is not None:
            _mask_list = _masks()
            _Xc, _Qc, _Wc = _cov_matrices(params, data)
            _kind_spec = (
                ("beta",  "x", ("q", "w"), _Xc, cov_index.x_groups, cov_index.x_idx),
                ("kappa", "q", ("x", "w"), _Qc, cov_index.q_groups, cov_index.q_idx),
                ("gamma", "w", ("x", "q"), _Wc, cov_index.w_groups, cov_index.w_group_idx),
            )
            for kind, code, others, Mfull, groups, idx in _kind_spec:
                for g, group_nodes in enumerate(groups):
                    _design = (Mfull[:, :, idx[g]] if code == "w"
                               else Mfull[:, idx[g]])
                    K_c = _design.shape[-1]
                    _tau2, _hp, _mu0 = _coef_prior(kind, g, group_nodes[0])
                    _mu0v = np.full(K_c, _mu0)
                    _E = [(_latent_resid(r, others) if r in group_nodes else None)
                          for r in range(R)]

                    if not _coef_mh[kind][g]:
                        # Every member is augmented: exact Gibbs draw.
                        params[kind][g] = _gibbs.gibbs_coef_pooled(
                            code, group_nodes, _E, _mask_list, _a_by_node,
                            _design, _tau2, rng, prior_mean=_mu0v,
                        )
                    else:
                        # At least one member has no latent response: MH on
                        # the joint conditional.  Probit members enter through
                        # their Gaussian latent likelihood (exact given Z),
                        # folded with the prior into (prec, lin); logit/GPCM
                        # members through their observed-data likelihood.
                        G, h = _gibbs.coef_gaussian_terms(
                            code, group_nodes, _E, _mask_list, _a_by_node, _design)
                        prec = np.eye(K_c) / _tau2 + G
                        lin = _mu0v / _tau2 + h
                        _mh_nodes = [r for r in group_nodes
                                     if not dispatch[r].is_augmentable]
                        # Preconditioner: logit Fisher information is at most
                        # 1/4 a^2 d d' per cell (p(1-p) <= 1/4).
                        _zeros = [(np.zeros((P, I)) if r in _mh_nodes else None)
                                  for r in range(R)]
                        _Tm = [(data.T[r] if r in _mh_nodes else None)
                               for r in range(R)]
                        G_l, _ = _gibbs.coef_gaussian_terms(
                            code, _mh_nodes, _zeros, _Tm, _a_by_node, _design)
                        H = prec + 0.25 * G_l
                        try:
                            _Lp = np.linalg.cholesky(np.linalg.inv(H))
                        except np.linalg.LinAlgError:
                            _Lp = np.eye(K_c) * np.sqrt(_tau2)
                        _base = {r: _node_offset(r, params, cov_index, data,
                                                 include=others)
                                 for r in _mh_nodes}

                        def _ll_fn(c_vec, _mh_nodes=_mh_nodes, _base=_base,
                                   _design=_design, code=code):
                            if code == "x":
                                term = (_design @ c_vec)[:, None] * np.ones((1, I))
                            elif code == "q":
                                term = np.ones((P, 1)) * (_design @ c_vec)[None, :]
                            else:
                                term = _design @ c_vec
                            parts = []
                            for r_m in _mh_nodes:
                                off = term if _base[r_m] is None else _base[r_m] + term
                                parts.append(_log_node_lik_direct(
                                    data.Y_star[r_m], data.T[r_m],
                                    params["theta"][node_to_dim[r_m]],
                                    params["b"][b_node_to_g[r_m]],
                                    node_models[r_m],
                                    params["a"][a_node_to_g[r_m]],
                                    params["c"][c_node_to_g[r_m]],
                                    offset=off,
                                ))
                            return float(sum(parts)), parts

                        key = f"{kind}_mh_{g}"
                        step = (adapt_states[key].step_size if key in adapt_states
                                else 2.38 / np.sqrt(K_c))
                        c_new, acc, parts = _mh_coefficient(
                            np.asarray(params[kind][g], float), prec, lin,
                            _ll_fn, _Lp, step, rng,
                        )
                        params[kind][g] = c_new
                        for r_m, ll_m in zip(_mh_nodes, parts, strict=True):
                            cache.update(r_m, ll_m)
                        if key in adapt_states:
                            if t < adapt_until:
                                adapt_states[key] = adapt_step(adapt_states[key], acc, t)
                            else:
                                st = adapt_states[key]
                                adapt_states[key] = dataclasses.replace(
                                    st, n_accepted=st.n_accepted + int(acc),
                                    n_proposed=st.n_proposed + 1)

                    if _hp is not None:
                        params[f"{kind}.variance"][g] = float(_gibbs.gibbs_b_variance(
                            np.asarray(params[kind][g]) - _mu0, _hp.alpha, _hp.beta, rng))

            # --- Interweaving (ASIS; Yu & Meng 2011) -----------------------
            # The draws above use the non-centered form: b_i ~ N(mu, tau2)
            # with predictor theta - b_i + q_i'kappa, and theta_p ~ N(0, Sigma)
            # with predictor theta_pd + x_p'beta.  There kappa can only move
            # as far as b lets it (and beta as far as theta), so both crawl
            # along the ridge b - q'kappa = const.  In the centered form the
            # location eta_i = b_i - q_i'kappa (zeta_pd = theta_pd + x_p'beta)
            # carries the likelihood and kappa (beta) enters only its prior.
            # When every node sharing b (theta) uses this coefficient group,
            # the likelihood depends on (b, kappa) only through eta, so
            #     eta = b - Q kappa;  kappa ~ p(kappa | eta);  b = eta + Q kappa
            # is an exact Gibbs step in the centered parameterization, and it
            # leaves every likelihood term (Z, logit caches) unchanged.
            def _asis_draw(Y_c, D_c, prec_e, prior_var, prior_mu):
                """kappa | eta for Y_c[u, j] = D_c[u] . c + e, e ~ N(0, 1/prec_e[u])."""
                K_c = D_c.shape[1]
                w = prec_e * Y_c.shape[1]                       # [U]
                A = np.eye(K_c) / prior_var + D_c.T @ (w[:, None] * D_c)
                h = np.full(K_c, prior_mu / prior_var) + D_c.T @ (prec_e * Y_c.sum(axis=1))
                return _gibbs._mvn_from_precision(A, h, rng)

            for g, grp in enumerate(cov_index.q_groups):
                gb = b_node_to_g[grp[0]]
                members = _b_groups_all[gb]
                if set(members) != set(grp) or any(
                        cov_index.q_node_to_group[r] != g for r in members):
                    continue            # some node sharing b does not use kappa
                if b_node_in_b_corr[grp[0]] or isinstance(params["b"][gb], NodeBParams):
                    continue
                b_arr = np.asarray(params["b"][gb], dtype=np.float64)
                which = "b.poly" if b_arr.ndim == 2 else "b"
                mu_b, tau2_b = _b_prior(gb, which)
                Qd = _cov_matrices(params, data)[1][:, cov_index.q_idx[g]]
                kap = np.asarray(params["kappa"][g], dtype=np.float64)
                shift = Qd @ kap                                  # [I]
                eta = b_arr - (shift[:, None] if b_arr.ndim == 2 else shift)
                eta2 = eta if eta.ndim == 2 else eta[:, None]
                tau2_k, _hp_k, mu_k = _coef_prior("kappa", g, grp[0])
                # eta_ij ~ N(mu_b - q_i'kappa, tau2_b)  =>  mu_b - eta_ij = q_i'kappa + e
                kap_new = _asis_draw(mu_b - eta2, Qd, np.full(I, 1.0 / tau2_b),
                                     tau2_k, mu_k)
                new_shift = Qd @ kap_new
                params["kappa"][g] = kap_new
                params["b"][gb] = eta + (new_shift[:, None] if b_arr.ndim == 2 else new_shift)

            if cov_index.x_groups:
                _Sig = _theta_Sigma()
                _Lam = np.linalg.inv(np.atleast_2d(_Sig))
                Xc_all = _cov_matrices(params, data)[0]
                for g, grp in enumerate(cov_index.x_groups):
                    d_ = node_to_dim[grp[0]]
                    if set(dim_to_nodes[d_]) != set(grp) or any(
                            cov_index.x_node_to_group[r] != g for r in grp):
                        continue        # some node on this dimension does not use beta
                    Xd = Xc_all[:, cov_index.x_idx[g]]
                    bet = np.asarray(params["beta"][g], dtype=np.float64)
                    zeta = params["theta"][d_] + Xd @ bet
                    # theta_pd | theta_p,-d ~ N(m_p, 1/Lam_dd), m_p from Sigma^{-1}
                    m_p = np.zeros(P)
                    for e_ in range(D):
                        if e_ != d_ and _Lam[d_, e_] != 0:
                            m_p -= _Lam[d_, e_] * params["theta"][e_] / _Lam[d_, d_]
                    tau2_x, _hp_x, mu_x = _coef_prior("beta", g, grp[0])
                    # zeta_p - m_p = x_p'beta + e,  e ~ N(0, 1/Lam_dd)
                    bet_new = _asis_draw((zeta - m_p)[:, None], Xd,
                                         np.full(P, _Lam[d_, d_]), tau2_x, mu_x)
                    params["beta"][g] = bet_new
                    params["theta"][d_] = zeta - Xd @ bet_new

            # --- Missing covariate values (treated as parameters) ---
            # The imputation kernels index coefficients by GLOBAL feature
            # position, so each node's (group-keyed, selected-column)
            # coefficient is scattered back to full width with zeros on the
            # columns the node does not select: a node contributes nothing to
            # a feature it never uses, which is what the zero produces.
            def _full(kind, r):
                if kind == "x":
                    g, M_, idx, coef = (cov_index.x_node_to_group[r], data.X,
                                        cov_index.x_idx, params["beta"])
                    width = None if M_ is None else M_.shape[1]
                elif kind == "q":
                    g, M_, idx, coef = (cov_index.q_node_to_group[r], data.Q,
                                        cov_index.q_idx, params["kappa"])
                    width = None if M_ is None else M_.shape[1]
                else:
                    g, M_, idx, coef = (cov_index.w_node_to_group[r], data.W,
                                        cov_index.w_group_idx, params["gamma"])
                    width = None if M_ is None else M_.shape[2]
                if g is None or width is None:
                    return None
                out = np.zeros(width)
                out[idx[g]] = coef[g]
                return out

            def _miss_prior(name):
                pr = priors.resolve(name, None)
                return (float(getattr(pr, "mean", 0.0)),
                        float(getattr(pr, "variance", 1.0)))

            # Each missing unit (person row of X, item row of Q, cell of W)
            # has its own full conditional.  Units that no logit/GPCM node
            # touches are drawn exactly (Gibbs); the rest by a vectorized MH
            # step, accepted unit by unit (_impute_missing).
            for _code, _key, _others in (("x", "X_working", ("q", "w")),
                                         ("q", "Q_working", ("x", "w")),
                                         ("w", "W_working", ("x", "q"))):
                if params.get(_key) is None or _miss_units[_code] is None:
                    continue
                _impute_missing(_code, _key, _others,
                                [_full(_code, r) for r in range(R)],
                                _miss_prior(_miss_prior_name[_code]))
        # ==========================================================================

        # --- Evidence mode: detect when the LAST MH block has frozen ---
        # All adapt_step calls for iteration t are done by now.  The draw at the
        # freezing iteration itself used the pre-freeze step size, so the first
        # fully-fixed-kernel draw is t+1; keep_start is set accordingly.
        # Safety net: once the minimum warm-up has passed, a block that has
        # never proposed is not part of the sampler; ignore it rather than
        # wait forever for it to freeze.
        if _evidence and keep_start is None and adapt_states \
                and all(s_.frozen or (t >= floor_warmup and s_.n_proposed == 0)
                        for s_ in adapt_states.values()):
            keep_start = t + 1

        # --- Store sample ---
        if keep_start is not None and t >= keep_start \
                and (t - keep_start) % thin_by == 0:
            row = np.empty(len(param_names), dtype=np.float64)

            # theta and sigma2: indexed by latent dim
            for d in range(D):
                s, e = offsets[f"theta_{d}"]
                row[s:e] = params["theta"][d]
                s, e = offsets[f"sigma2_{d}"]
                row[s] = params["sigma2"][d]
            # b, a, c: indexed by group
            for g in range(len(b_groups)):
                s, e = offsets[f"b_{g}"]
                b_val = params["b"][g]
                if isinstance(b_val, NodeBParams):
                    row[s:e] = _flatten_mixed_b(b_val, I)
                else:
                    row[s:e] = b_val.ravel()
            for g in range(len(a_groups)):
                if f"a_{g}" in offsets:
                    s, e = offsets[f"a_{g}"]
                    row[s:e] = params["a"][g]
            for g in range(len(c_groups)):
                if f"c_{g}" in offsets:
                    s, e = offsets[f"c_{g}"]
                    row[s:e] = params["c"][g]

            # b.variance: one value per b-group
            for g in range(len(b_groups)):
                key = f"b_var_{g}"
                if key in offsets:
                    s, e = offsets[key]
                    row[s] = params["b.variance"][g]       # indexed by group

            # a.variance: one value per a-group with a forced hierarchical variance
            for g in range(len(a_groups)):
                key = f"a_var_{g}"
                if key in offsets:
                    s, e = offsets[key]
                    row[s] = params["a.variance"][g]

            # rho / Sigma_theta: theta correlation parameters
            if "rho" in offsets:
                s, e = offsets["rho"]
                if using_invwishart_theta:
                    # Store each block's lower triangle of Sigma
                    Sigma_stored = np.atleast_2d(params["corr_matrix"])
                    for k, (d1, d2) in enumerate(_th_tri):
                        row[s + k] = Sigma_stored[d1, d2]
                else:
                    for k, (d1, d2) in enumerate(_th_pairs):
                        row[s + k] = params["corr_matrix"][d1, d2]

            # b.corr_matrix: store per-group
            for cg_idx in range(len(b_corr_groups)):
                key_s = f"b_corr_Sigma_{cg_idx}"
                key_r = f"b_corr_rho_{cg_idx}"
                cg = b_corr_groups[cg_idx]
                G_cg = len(cg)
                if key_s in offsets:
                    s, e = offsets[key_s]
                    Sigma_b_stored = params["b.corr_matrix"][cg_idx]
                    k = 0
                    for i in range(G_cg):
                        for j in range(i, G_cg):
                            row[s + k] = float(Sigma_b_stored[i, j])
                            k += 1
                elif key_r in offsets:
                    s, e = offsets[key_r]
                    R_b_stored = params["b.corr_matrix"][cg_idx]
                    k = 0
                    for i in range(G_cg):
                        for j in range(i + 1, G_cg):
                            row[s + k] = float(R_b_stored[i, j])
                            k += 1

            # a.corr_matrix: store per-group
            for cg_idx in range(len(a_corr_groups)):
                key_s = f"a_corr_Sigma_{cg_idx}"
                key_r = f"a_corr_rho_{cg_idx}"
                cg = a_corr_groups[cg_idx]
                G_cg = len(cg)
                if key_s in offsets:
                    s, e = offsets[key_s]
                    Sigma_a_stored = params["a.corr_matrix"][cg_idx]
                    k = 0
                    for i in range(G_cg):
                        for j in range(i, G_cg):
                            row[s + k] = float(Sigma_a_stored[i, j])
                            k += 1
                elif key_r in offsets:
                    s, e = offsets[key_r]
                    R_a_stored = params["a.corr_matrix"][cg_idx]
                    k = 0
                    for i in range(G_cg):
                        for j in range(i + 1, G_cg):
                            row[s + k] = float(R_a_stored[i, j])
                            k += 1

            # beta, kappa: per-GROUP; gamma: per-node covariate coefficients
            if cov_index is not None:
                for g in range(len(cov_index.x_groups)):
                    if f"beta_{g}" in offsets:
                        s, e = offsets[f"beta_{g}"]
                        row[s:e] = params["beta"][g]
                for g in range(len(cov_index.q_groups)):
                    if f"kappa_{g}" in offsets:
                        s, e = offsets[f"kappa_{g}"]
                        row[s:e] = params["kappa"][g]
                for g in range(len(cov_index.w_groups)):
                    if f"gamma_{g}" in offsets:
                        s, e = offsets[f"gamma_{g}"]
                        row[s:e] = params["gamma"][g]
                for kind in ("beta", "kappa", "gamma"):
                    for g, v in enumerate(params.get(f"{kind}.variance") or []):
                        if f"{kind}_var_{g}" in offsets:
                            row[offsets[f"{kind}_var_{g}"][0]] = v

            # X_missing: only store missing feature values for missing persons
            if params["X_working"] is not None and data.n_missing_X > 0:
                for p in np.where(~data.X_observed)[0]:
                    key = f"X_missing_{p}"
                    if key in offsets:
                        s, e = offsets[key]
                        feat_miss = np.where(~data.X_feature_observed[p])[0]
                        row[s:e] = params["X_working"][p, feat_miss]

            # Q_missing: missing item covariate features
            if params.get("Q_working") is not None and data.n_missing_Q > 0:
                for i in np.where(~data.Q_observed)[0]:
                    key = f"Q_missing_{i}"
                    if key in offsets:
                        s, e = offsets[key]
                        feat_miss = np.where(~data.Q_feature_observed[i])[0]
                        row[s:e] = params["Q_working"][i, feat_miss]

            # W_missing: missing person×item covariate features
            if params.get("W_working") is not None and data.n_missing_W > 0:
                for cell in np.argwhere(~data.W_observed):
                    p, i = int(cell[0]), int(cell[1])
                    key = f"W_missing_{p}_{i}"
                    if key in offsets:
                        s, e = offsets[key]
                        feat_miss = np.where(~data.W_feature_observed[p, i])[0]
                        row[s:e] = params["W_working"][p, i, feat_miss]

            # Cache refresh for augmentable (Gibbs) nodes.  These nodes never
            # read the cache during an iteration -- only pure-MH node blocks do,
            # and each reads its own index, which it recomputes itself.  So the
            # Gibbs entries are recomputed here, once per STORED sample, rather
            # than on every parameter update.  With thin_by > 1 this is a strict
            # reduction; the log-likelihood trace also becomes exact rather than
            # lagging the final theta/b draws of the iteration.
            for _r in range(R):
                if dispatch[_r].is_augmentable:
                    recompute_node(
                        cache, _r, data,
                        params["theta"][node_to_dim[_r]],
                        params["b"][b_node_to_g[_r]],
                        node_models[_r],
                        a_r=params["a"][a_node_to_g[_r]],
                        c_r=params["c"][c_node_to_g[_r]],
                        offset=_node_offset(_r, params, cov_index, data),
                    )

            chain[stored_idx]          = row.astype(dtype)
            log_lik_trace[stored_idx] = cache.total
            stored_idx += 1

        # progress: emits every print_every iterations.  loglik is the
        # current data log-likelihood (cache.total, no priors); accept is the
        # running mean over MH blocks (None -> omitted for pure Gibbs).
        _reporter.step(
            t,
            loglik=float(cache.total),
            accept=mean_acceptance(adapt_states),
        )

    _reporter.finish()

    # --- Build accept_rates summary ---
    accept_rates = {}
    for key, st in adapt_states.items():
        if st.n_proposed > 0:
            accept_rates[key] = st.n_accepted / st.n_proposed

    # --- Warmup diagnostics ---
    frozen_at = {
        key: st.frozen_at_t
        for key, st in adapt_states.items()
        if st.frozen_at_t is not None
    }
    # effective_warmup: where kept-sample storage actually began.  In evidence
    # mode keep_start may still be None if nothing ever froze within n_iter
    # (degenerate); fall back to n_warmup_eff so the field is always meaningful.
    effective_warmup = keep_start if keep_start is not None else n_warmup_eff

    return MCMCResult(
        chain=chain[:stored_idx],
        param_names=param_names,
        log_lik_trace=log_lik_trace[:stored_idx],
        accept_rates=accept_rates,
        n_warmup=n_warmup_eff,
        n_iter=n_iter,
        final_params=params,
        rng_state=rng.bit_generator.state,
        warmup_mode=warmup_mode,
        effective_warmup=int(effective_warmup),
        frozen_at=frozen_at,
    )


# ===========================================================================
# Private helpers for pure-MH nodes
# ===========================================================================

def _mh_coefficient(
    c_curr:    np.ndarray,
    prec:      np.ndarray,
    lin:       np.ndarray,
    loglik_fn,
    prop_chol: np.ndarray,
    step:      float,
    rng:       np.random.Generator,
):
    """One preconditioned random-walk MH step for a covariate coefficient.

    Target (up to a constant), conditional on everything else:

        log pi(c) = lin'c - 1/2 c' prec c  +  loglik_fn(c)

    The quadratic part holds the Normal prior and the Gaussian latent
    likelihood of the probit nodes in the group (exact given Z); loglik_fn
    returns the observed-data log-likelihood of the non-augmented (logit /
    GPCM) nodes, as ``(total, per_node_list)``.

    Proposal: c' = c + step * L eps, eps ~ N(0, I), with L L' an approximate
    posterior covariance.  L does not depend on c, so the proposal is
    symmetric and the acceptance ratio is the target ratio.

    Returns (c_new, accepted, per_node_loglik_at_c_new).
    """
    K = len(c_curr)
    c_prop = c_curr + step * (prop_chol @ rng.standard_normal(K))
    ll_c, parts_c = loglik_fn(c_curr)
    ll_p, parts_p = loglik_fn(c_prop)

    def quad(c):
        return float(lin @ c - 0.5 * c @ prec @ c)

    log_alpha = (quad(c_prop) + ll_p) - (quad(c_curr) + ll_c)
    accepted = math.log(max(rng.uniform(), 1e-300)) < log_alpha
    if accepted:
        return c_prop, True, parts_p
    return c_curr, False, parts_c


def _mh_missing_units(values, groups, precL_list, touched, unit_ll, step, rng):
    """One update of a set of missing-covariate units.

    Unit u's target, given everything else:

        log pi(v) = rhs_u'v - 1/2 v' prec_u v  +  unit_ll(v)[u]

    with (prec_u, rhs_u) the prior plus probit Gaussian terms (``groups``, from
    gibbs_samplers.missing_terms) and unit_ll the per-unit logit
    log-likelihood.  Units with touched[u] = False have no logit term and are
    drawn exactly.  The others get v' = v + step * L_u eps with
    L_u L_u' = inv(prec_u + 1/4 logit information) (precL_list, aligned with
    groups); all are proposed together and accepted unit by unit, which is
    valid because the units are conditionally independent and the logit
    log-likelihood factorizes over them.

    Returns (new_values, accepted [U] bool).
    """
    values = values.copy()
    prop = values.copy()
    U = values.shape[0]
    quad_c = np.zeros(U)
    quad_p = np.zeros(U)
    for (sel, mi, prec, rhs), precL in zip(groups, precL_list, strict=True):
        t_sel = touched[sel]
        if np.any(~t_sel):
            g_sel = sel[~t_sel]
            draw = _gibbs.draw_from_terms(prec[~t_sel], rhs[~t_sel], rng)
            values[g_sel[:, None], mi[None, :]] = draw
            prop[g_sel[:, None], mi[None, :]] = draw
        if not np.any(t_sel):
            continue
        m_sel = sel[t_sel]
        Pm, Rm = prec[t_sel], rhs[t_sel]
        L = np.linalg.cholesky(np.linalg.inv(Pm + 0.25 * precL[t_sel]))
        u_c = values[m_sel[:, None], mi[None, :]]
        u_p = u_c + step * np.einsum('nab,nb->na', L, rng.standard_normal(u_c.shape))
        prop[m_sel[:, None], mi[None, :]] = u_p
        quad_c[m_sel] = (np.einsum('na,na->n', Rm, u_c)
                         - 0.5 * np.einsum('na,nab,nb->n', u_c, Pm, u_c))
        quad_p[m_sel] = (np.einsum('na,na->n', Rm, u_p)
                         - 0.5 * np.einsum('na,nab,nb->n', u_p, Pm, u_p))
    acc = np.zeros(U, dtype=bool)
    if np.any(touched):
        log_alpha = (quad_p + unit_ll(prop)) - (quad_c + unit_ll(values))
        u01 = rng.uniform(size=U)
        acc = touched & (np.log(np.maximum(u01, 1e-300)) < log_alpha)
        values[acc] = prop[acc]
    return values, acc


def _mh_diag_units(cur, prec_q, rhs_q, H, unit_ll, step, rng):
    """Vectorized MH over conditionally independent units.

    Unit u holds a k-vector v_u (a person's theta on one dimension, k = 1;
    an item's difficulty, k = 1; or an item's thresholds, k = M - 1).  Its
    target, given everything else, is

        log pi(v_u) = sum_k ( rhs_q[u,k] v_uk - 1/2 prec_q[u,k] v_uk^2 )
                      + unit_ll(v)[u],

    i.e. a diagonal Gaussian part (prior, conditional prior, probit latent
    terms) plus the logit log-likelihood of the unit's cells.  All units are
    proposed at once, v' = v + step * eps / sqrt(H), with H the Gaussian
    precision plus the logit information bound, and accepted one by one.
    That is valid because the units are conditionally independent and
    unit_ll factorizes over them (row sums for persons, column sums for
    items).  Returns (new [U, k], accepted [U] bool).
    """
    cur = np.asarray(cur, float)
    prop = cur + step * rng.standard_normal(cur.shape) / np.sqrt(H)

    def quad(v):
        return np.sum(rhs_q * v - 0.5 * prec_q * v * v, axis=1)

    log_alpha = (quad(prop) + unit_ll(prop)) - (quad(cur) + unit_ll(cur))
    acc = np.log(np.maximum(rng.uniform(size=cur.shape[0]), 1e-300)) < log_alpha
    out = cur.copy()
    out[acc] = prop[acc]
    return out, acc


def _threshold_proposal_scale(Y_list, T_list, K, cols=None):
    """Fixed per-item, per-coordinate proposal scales for _mh_ordered_thresholds.

    Computed once from the data alone (so the proposal stays symmetric):
    with cumulative category proportions c_k of item i over the pooled nodes
    and n relevant responses, a rough threshold location is
    t_k = sqrt(2) Phi^{-1}(c_k) (latent variance ~ 1 + var(theta) ~ 2) with
    delta-method sd s_k = sqrt(2) sqrt(c_k (1 - c_k) / n) / phi(Phi^{-1}(c_k)).
    In u = (tau_1, log gaps): scale_1 = s_1 and
    scale_k = sqrt(s_k^2 + s_{k-1}^2) / (t_k - t_{k-1}).
    Returns [n_items, K].
    """
    from scipy.stats import norm as _norm
    Ys = np.concatenate([np.where(T, Y, -1) for Y, T in zip(Y_list, T_list, strict=True)], axis=0)
    if cols is not None:
        Ys = Ys[:, cols]
    n_it = Ys.shape[1]
    out = np.ones((n_it, K))
    for i in range(n_it):
        y = Ys[:, i][Ys[:, i] >= 0].astype(int)
        n = max(len(y), 1)
        c = np.array([(np.sum(y <= k) + 0.5) / (n + 1.0) for k in range(K)])
        z = _norm.ppf(c)
        t_k = np.sqrt(2.0) * z
        s_k = np.sqrt(2.0) * np.sqrt(c * (1 - c) / n) / np.maximum(_norm.pdf(z), 1e-3)
        out[i, 0] = s_k[0]
        if K > 1:
            gaps = np.maximum(np.diff(t_k), 0.05)
            out[i, 1:] = np.sqrt(s_k[1:] ** 2 + s_k[:-1] ** 2) / gaps
    return out


def _mh_ordered_thresholds(T_cur, col_ll, mu0, var, step, rng, scale=None):
    """Cowles (1996) MH step for ordered thresholds, item by item.

    T_cur  : [n, K] current thresholds of n items (strictly increasing rows).
    col_ll : function T [n, K] -> [n], each item's MARGINAL log-likelihood
             (Z integrated out: the node models' own category probabilities,
             e.g. Phi(tau_m - mu) - Phi(tau_{m-1} - mu) for the GRM), summed
             over every node sharing the thresholds.  Item i's value may
             depend only on row i.
    Prior: K iid N(mu0, var) restricted to increasing order.

    Proposal: symmetric random walk on u = (tau_1, log(tau_2 - tau_1), ...),
    with fixed per-coordinate ``scale`` (see _threshold_proposal_scale),
    so every proposal is ordered; the target in u carries the Jacobian
    prod_k (tau_k - tau_{k-1}) = exp(sum_k u_k).  Items are conditionally
    independent, so all are proposed together and accepted one by one.
    Because Z is integrated out, the proposal is not confined to the narrow
    interval between neighbouring latent responses; the caller redraws Z
    given the new thresholds (a joint update of (tau_i, Z_.i)).

    Returns (T_new [n, K], accepted [n] bool).
    """
    T_cur = np.asarray(T_cur, dtype=np.float64)

    def to_u(T):
        u = T.copy()
        with np.errstate(divide="ignore"):
            u[:, 1:] = np.log(np.diff(T, axis=1))
        return u

    def to_T(u):
        T = u.copy()
        T[:, 1:] = u[:, :1] + np.cumsum(np.exp(u[:, 1:]), axis=1)
        return T

    def log_target(T, u):
        return (col_ll(T) - 0.5 * np.sum((T - mu0) ** 2, axis=1) / var
                + np.sum(u[:, 1:], axis=1))

    u = to_u(T_cur)
    sc = 1.0 if scale is None else scale
    u_p = u + step * sc * rng.standard_normal(u.shape)
    T_p = to_T(u_p)
    log_alpha = log_target(T_p, u_p) - log_target(T_cur, u)
    acc = np.log(np.maximum(rng.uniform(size=T_cur.shape[0]), 1e-300)) < log_alpha
    acc &= np.isfinite(log_alpha)
    return np.where(acc[:, None], T_p, T_cur), acc


def _log_node_lik_direct(
    y_star:     np.ndarray,
    T_r:        np.ndarray,
    theta_r:    np.ndarray,
    b_r:        np.ndarray,
    node_model: NodeModel,
    a_r,
    c_r,
    offset:     np.ndarray | None = None,
) -> float:
    """Compute log-likelihood for a node from scratch (used in MH steps).

    ``offset`` is the RAW [P, I] covariate linear predictor for this node
    (X@beta + Q@kappa + W@gamma).  It MUST be passed whenever the node has
    covariates, or the covariate contribution is dropped from the MH
    accept/reject and the coefficients never enter that node's likelihood.
    log_prob applies it on the correct scale for each model (e.g. before the
    a-scaling for 2PL/3PL; per-threshold for GRM/GPCM).
    """
    if not np.any(T_r):
        return 0.0
    lp = node_model.log_prob(y_star, theta_r, b_r, a_r=a_r, c_r=c_r,
                             offset=offset)
    # np.where, not T_r * lp: log_prob is NaN in irrelevant cells and
    # 0.0 * nan == nan.  See likelihood.log_node_lik for the same fix.
    return float(np.sum(np.where(T_r, lp, 0.0)))


def _node_item_loglik(
    y_star, T_r, theta_r, b_r, node_model, a_r, c_r, offset=None,
) -> np.ndarray:
    """Per-ITEM node log-likelihood, shape [I].

    Entry i is Σ_p T_{pi}·log Pr(y_{pi} | …), i.e. item i's contribution.
    The node log-likelihood factors over items (item i involves only a_i,
    b_i, c_i), so per-item accept/reject of a_i and c_i in the MH updates
    below is valid.  Irrelevant cells (T_r False, where log_prob may be NaN)
    are zeroed with np.where, as in ``_log_node_lik_direct``.
    """
    lp = node_model.log_prob(y_star, theta_r, b_r, a_r=a_r, c_r=c_r,
                             offset=offset)
    return np.where(T_r, lp, 0.0).sum(axis=0)


def _a_logprior_natural(a_vec: np.ndarray, prior) -> np.ndarray:
    """Natural-scale log-prior density for discrimination a > 0 (up to const).

    Returns a per-item [I] array.  Supports the two positive-support priors a
    logit node can carry:

      * ``LogNormalPrior(mean, var)`` — parameters on the log scale; the
        a-scale density is Normal(log a; mean, var)·(1/a), hence the −log a.
      * ``TruncNormalPrior`` / bounded ``NormalPrior`` — a truncated Normal on
        the natural scale; −∞ outside the support.

    Paired with a +（log a' − log a) Jacobian in the acceptance ratio (the
    log-scale random walk), this one function gives the correct target for
    BOTH prior families: for the LogNormal the −log a here cancels the +log a
    Jacobian (net Normal-on-log-a); for the TruncNormal the +log a Jacobian
    survives (truncated-Normal mapped to the log scale).  This is the same
    change-of-variables bookkeeping used throughout the MH blocks.
    """
    a_vec = np.asarray(a_vec, dtype=np.float64)
    mean = float(getattr(prior, "mean", 1.0))
    var  = getattr(prior, "variance", 0.5)
    var  = float(var) if isinstance(var, (int, float)) else 0.5
    if isinstance(prior, LogNormalPrior):
        return -0.5 * (np.log(a_vec) - mean) ** 2 / var - np.log(a_vec)
    lp = -0.5 * (a_vec - mean) ** 2 / var
    lo = getattr(prior, "lower", 0.0)
    hi = getattr(prior, "upper", None)
    if lo is not None:
        lp = np.where(a_vec < lo, -np.inf, lp)
    if hi is not None:
        lp = np.where(a_vec > hi, -np.inf, lp)
    return lp


def _mh_logit_discrimination(
    node_model, y_star, T_r, theta_r, b_r, a_curr, c_r, offset,
    prior, step_size, rng, common=False,
):
    """Log-scale random-walk MH for discrimination a on a NON-augmentable node.

    Targets the node's own (logit) likelihood directly — there is no augmented
    Z for a logit node, so this is a plain Metropolis-within-Gibbs block.

    Proposal is symmetric in ξ = log a (keeps a > 0).  The acceptance ratio is
    assembled in the natural scale plus the +（log a' − log a) Jacobian; see
    ``_a_logprior_natural`` for why that one formula serves both the
    TruncNormal and LogNormal priors.

    ``common`` True ⇒ one shared scalar a for all items (common-slope / 1PL
    node): propose a single ξ, apply to every item, one accept/reject on the
    total node log-likelihood.  Otherwise a_i are conditionally independent
    across items given θ, b, c, so each item accepts/rejects on its own.

    Returns (a_new [I], accept_signal) where accept_signal is the mean per-item
    acceptance (a float in [0, 1]) — the correct Robbins-Monro signal for the
    shared step size.
    """
    I = len(a_curr)
    log_a = np.log(a_curr)
    if common:
        log_a_prop = log_a + step_size * rng.standard_normal()   # one shift
    else:
        log_a_prop = log_a + step_size * rng.standard_normal(I)
    a_prop = np.exp(log_a_prop)

    ll_c = _node_item_loglik(y_star, T_r, theta_r, b_r, node_model, a_curr, c_r, offset)
    ll_p = _node_item_loglik(y_star, T_r, theta_r, b_r, node_model, a_prop, c_r, offset)
    lp_c = _a_logprior_natural(a_curr, prior)
    lp_p = _a_logprior_natural(a_prop, prior)
    # Target in the working coordinate ξ = log a:  ll(a) + logprior_natural(a)
    # + log a   (the +log a is the log-scale Jacobian; see _a_logprior_natural).

    if common:
        # One shared scalar a: single accept/reject on the TOTAL node
        # log-likelihood, with the prior and Jacobian evaluated once.
        la = (ll_p.sum() + lp_p[0] + log_a_prop[0]) \
             - (ll_c.sum() + lp_c[0] + log_a[0])
        if math.log(max(rng.uniform(), 1e-300)) < la:
            return a_prop, 1.0
        return a_curr, 0.0

    # Per-item accept/reject (a_i conditionally independent across items).
    la = (ll_p + lp_p + log_a_prop) - (ll_c + lp_c + log_a)
    accept = np.log(rng.uniform(size=I)) < la
    a_new = np.where(accept, a_prop, a_curr)
    return a_new, float(np.mean(accept))


def _mh_logit_guessing(
    node_model, y_star, T_r, theta_r, b_r, a_r, c_curr, offset,
    prior, step_size, rng,
):
    """Logit-scale random-walk MH for guessing c ∈ (0, 1) on a logit 3PL node.

    Proposal symmetric in ζ = logit c (keeps c in the open unit interval).
    Target is the node likelihood × Beta(α, β) prior, with the change-of-
    variables Jacobian |dc/dζ| = c(1 − c): the acceptance ratio gains
    +（log c'(1−c') − log c(1−c)).  Per item (c_i conditionally independent
    across items given θ, a, b).

    Returns (c_new [I], mean per-item acceptance).
    """
    I = len(c_curr)
    c_curr = np.clip(np.asarray(c_curr, dtype=np.float64), 1e-9, 1.0 - 1e-9)
    zeta   = np.log(c_curr / (1.0 - c_curr))
    zeta_p = zeta + step_size * rng.standard_normal(I)
    c_prop = 1.0 / (1.0 + np.exp(-zeta_p))
    c_prop = np.clip(c_prop, 1e-9, 1.0 - 1e-9)

    ll_c = _node_item_loglik(y_star, T_r, theta_r, b_r, node_model, a_r, c_curr, offset)
    ll_p = _node_item_loglik(y_star, T_r, theta_r, b_r, node_model, a_r, c_prop, offset)

    alpha = float(getattr(prior, "alpha", 5.0))
    beta  = float(getattr(prior, "beta", 17.0))
    lp_c = (alpha - 1.0) * np.log(c_curr) + (beta - 1.0) * np.log(1.0 - c_curr)
    lp_p = (alpha - 1.0) * np.log(c_prop) + (beta - 1.0) * np.log(1.0 - c_prop)
    jac_c = np.log(c_curr) + np.log(1.0 - c_curr)     # log|dc/dζ| at current
    jac_p = np.log(c_prop) + np.log(1.0 - c_prop)     # at proposed

    la = (ll_p + lp_p + jac_p) - (ll_c + lp_c + jac_c)
    accept = np.log(rng.uniform(size=I)) < la
    c_new = np.where(accept, c_prop, c_curr)
    return c_new, float(np.mean(accept))


def _log_trunc_normal_density(
    x:     np.ndarray,
    mean:  float,
    var:   float,
    lower: float | None,
    upper: float | None,
) -> float:
    """Total log-density of a truncated Normal over the vector x.

    This is the target for the a.variance MH block.  Written out in full
    because the normalising constant is the whole reason this parameter is
    not Gibbs-drawable:

        log p(x | mu, sigma^2)
            = sum_i [ -0.5 log(2 pi sigma^2) - (x_i - mu)^2 / (2 sigma^2) ]
              - n * log[ Phi((u - mu)/sigma) - Phi((l - mu)/sigma) ]

    The final term depends on sigma, so multiplying by an InvGamma prior does
    not close.  Dropping it -- which is what treating this as conjugate would
    amount to -- biases sigma^2 upward, because a wider sigma spills more mass
    outside the bounds and the missing renormalisation never penalises it.

    Returns -inf if any x_i lies outside the support, or if the support has
    negligible mass under (mean, var).
    """
    if var <= 0.0:
        return -np.inf
    sd = float(np.sqrt(var))
    x  = np.asarray(x, dtype=np.float64).ravel()
    n  = x.size
    if n == 0:
        return 0.0

    if lower is not None and np.any(x < lower):
        return -np.inf
    if upper is not None and np.any(x > upper):
        return -np.inf

    kernel = (-0.5 * n * math.log(2.0 * math.pi * var)
              - float(np.sum((x - mean) ** 2)) / (2.0 * var))

    p_lo = 0.0 if lower is None else float(_Phi((lower - mean) / sd))
    p_hi = 1.0 if upper is None else float(_Phi((upper - mean) / sd))
    mass = p_hi - p_lo
    if mass <= 1e-300:
        return -np.inf

    return kernel - n * math.log(mass)


def _log_variance_hyperprior(v: float, vprior) -> float:
    """Log prior density of a variance v under its hyperprior, in v.

    InvGammaPrior(alpha, beta): density of v directly.
    HalfNormalPrior(scale) on the SD s = sqrt(v): p(v) = p_s(sqrt v) / (2 sqrt v),
    i.e. log p(v) = -v / (2 scale^2) - 0.5 log v + const.
    """
    if v <= 0.0:
        return -np.inf
    if isinstance(vprior, HalfNormalPrior):
        return -v / (2.0 * float(vprior.scale) ** 2) - 0.5 * math.log(v)
    return _log_invgamma_density(v, float(getattr(vprior, "alpha", 1.0)),
                                 float(getattr(vprior, "beta", 1.0)))


def _log_invgamma_density(x: float, alpha: float, beta: float) -> float:
    """Log-density of InvGamma(alpha, beta) at x, up to an additive constant."""
    if x <= 0.0:
        return -np.inf
    return -(alpha + 1.0) * math.log(x) - beta / x


def _log_normal_prior(
    x:     np.ndarray,
    prior,
) -> float:
    """Log-density of a Normal prior on x (used in pure-MH nodes)."""
    mean = float(getattr(prior, "mean", 0.0))
    var  = getattr(prior, "variance", 1.0)
    if isinstance(var, (int, float)):
        var = float(var)
    else:
        var = 1.0
    return float(-0.5 * np.sum((x - mean) ** 2) / var)
