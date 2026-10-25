"""
irtree.likelihood
=================
Layer 4 — Node-level log-likelihood computation and the per-node cache.

FIRST PRINCIPLES
================
The IRTree log-likelihood factorises *completely* across nodes:

    log L(Θ) = Σ_{r=0}^{R-1}  ℓ_r(Θ_r)

where

    ℓ_r(Θ_r) = Σ_{(p,i): T_r[p,i]=1}  log Pr(Y*_r[p,i] | θ_p, params_r)

This factorisation is not an approximation — it follows directly from the
conditional-independence structure of the mapping matrix: once the
pseudo-responses Y*_r are fixed by the observed data Y, node r's
contribution depends only on the parameters node r uses
{θ_r, b_r, (a_r), (c_r), covariate offset}.  When nodes share a latent
dimension or an item-parameter group, one parameter enters several ℓ_r.

WHAT THE CACHE IS FOR
---------------------
A ``LikelihoodCache`` stores the R scalar node log-likelihoods; ``total`` is
their sum.  It is a bookkeeping store, not an input to any update:

  * The stored log-likelihood trace (MCMCResult.log_lik_trace) and the
    progress display read ``cache.total``.
  * Augmented (probit) nodes are recomputed once per STORED draw.
  * Logit/GPCM nodes are refreshed whenever an MH block changes one of
    their parameters (theta, b, a, c, a covariate coefficient or imputed
    value), so their entries are always current.

MH acceptance ratios do not read the cache: each block evaluates the
likelihood it needs directly, because a shared parameter changes several
node terms at once.  WAIC and LOO do not use it either; they rebuild
per-person log-likelihoods from the chain
(posterior._pointwise_log_lik_fast).

INNER SUM
---------
Within each node only the T_r[p, i] = 1 cells contribute.  The sum uses
``np.sum(np.where(T_r, log_prob_matrix, 0.0))``: a boolean select over the
full [P × I] matrix (sequential memory access, faster than fancy indexing
with row_idx / col_idx) that is also safe when ``log_prob`` returns NaN in
irrelevant cells, where a product ``T_r * log_prob_matrix`` would give
0 * NaN = NaN.

PUBLIC API
----------
``log_node_lik``     Compute the scalar log-likelihood for one node.
``init_cache``       Build a LikelihoodCache from scratch (all R nodes).
``recompute_node``   Recompute one node's cached log-likelihood in place.
``LikelihoodCache``  Dataclass: stores [R] node scalars, exposes .total.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from irtree.data_prep import PreparedData
from irtree.node_models import NodeModel

# ===========================================================================
# Core computation: log-likelihood for a single node
# ===========================================================================

def log_node_lik(
    y_star:     np.ndarray,
    T_r:        np.ndarray,
    theta_r:    np.ndarray,
    b_r:        np.ndarray,
    node_model: NodeModel,
    a_r:        np.ndarray | None = None,
    c_r:        np.ndarray | None = None,
    offset:     np.ndarray | None = None,
) -> float:
    """Compute the scalar log-likelihood contribution of one IRTree node.

    Evaluates

        ℓ_r = Σ_{(p,i) ∈ relevant}  log Pr(Y*_r[p,i] | θ_p, b_i, ...)

    with a boolean select that skips irrelevant cells.

    Parameters
    ----------
    y_star : ndarray [P × I]
        Pseudo-response matrix for node r.  Irrelevant cells (where
        ``T_r[p,i] = False``) contain 0 but are zeroed out by the mask
        and never contribute to the sum.
    T_r : ndarray [P × I], dtype bool
        Relevance mask.  ``T_r[p, i] = True`` iff node r is on the path
        for the observed category at (p, i).  Used for the inner sum via
        ``np.sum(np.where(T_r, log_prob_matrix, 0.0))`` (see module
        docstring).
    theta_r : ndarray [P]
        Current person trait parameters for node r.
    b_r : ndarray [I]
        Current item difficulty parameters for node r.
    node_model : NodeModel
        Node model instance whose ``log_prob`` method returns [P × I].
    a_r : ndarray [I] or None
        Discrimination parameters (2PL / 3PL nodes only).
    c_r : ndarray [I] or None
        Guessing parameters (3PL nodes only).
    offset : ndarray [P × I] or None
        Covariate linear predictor X·β + Q·κ + W·γ for this node (raw, not
        a-scaled; ``log_prob`` applies it on the model's own scale).

    Returns
    -------
    float
        Scalar log-likelihood contribution ℓ_r.
    """
    if not np.any(T_r):
        return 0.0

    # node_model.log_prob returns a [P × I] array of log-probabilities
    # for ALL (p,i) pairs, including irrelevant ones.
    log_prob_matrix = node_model.log_prob(
        y_star, theta_r, b_r, a_r=a_r, c_r=c_r, offset=offset,
    )

    # Boolean select over the full matrix (sequential memory access).
    # NOTE: do NOT use ``T_r * log_prob_matrix``.  node_model.log_prob returns
    # NaN in irrelevant cells (Y*_r carries a sentinel there), and
    # False * nan == 0.0 * nan == nan, which poisons the whole sum.  np.where
    # selects instead of multiplying, so irrelevant cells contribute exactly 0.
    return float(np.sum(np.where(T_r, log_prob_matrix, 0.0)))


# ===========================================================================
# LikelihoodCache — stores one scalar per node
# ===========================================================================

@dataclass
class LikelihoodCache:
    """Cache for node-level log-likelihood scalars.

    Stores an array of R floats, one per node, written with
    ``update(r, new_val)`` (see the module docstring for when each entry is
    refreshed).  The full log-likelihood is ``total = sum(node_logliks)``.

    Attributes
    ----------
    node_logliks : ndarray [R]
        One cached scalar per node.  Always float64.
    """
    node_logliks: np.ndarray

    def __post_init__(self):
        self.node_logliks = np.asarray(self.node_logliks, dtype=np.float64)
        if self.node_logliks.ndim != 1:
            raise ValueError(
                f"node_logliks must be 1-D; got shape {self.node_logliks.shape}."
            )

    def update(self, r: int, new_loglik: float) -> None:
        """Update the cached log-likelihood for node r."""
        self.node_logliks[r] = new_loglik

    @property
    def total(self) -> float:
        """Total log-likelihood = sum of all cached node contributions."""
        return float(np.sum(self.node_logliks))

    def copy(self) -> LikelihoodCache:
        """Return an independent copy (does not share the array)."""
        return LikelihoodCache(node_logliks=self.node_logliks.copy())


# ===========================================================================
# init_cache — build a fresh cache from data and current parameters
# ===========================================================================

def init_cache(
    data:        PreparedData,
    theta:       list[np.ndarray],
    b:           list[np.ndarray],
    node_models: list[NodeModel],
    a:           list[np.ndarray] | None = None,
    c:           list[np.ndarray] | None = None,
    offsets:     list[np.ndarray | None] | None = None,
) -> LikelihoodCache:
    """Build a LikelihoodCache from scratch, computing all R node log-liks.

    Parameters
    ----------
    data : PreparedData
    theta : list of R ndarrays [P]
    b : list of R ndarrays [I]
    node_models : list of R NodeModel instances
    a, c : optional lists of R ndarrays [I]
    offsets : optional list of R covariate offsets ([P × I] or None)

    Returns
    -------
    LikelihoodCache
        All R node log-likelihoods freshly computed.
    """
    R = data.n_nodes
    _check_length("theta", theta, R)
    _check_length("b", b, R)
    _check_length("node_models", node_models, R)

    logliks = np.empty(R, dtype=np.float64)
    for r in range(R):
        logliks[r] = log_node_lik(
            y_star=data.Y_star[r],
            T_r=data.T[r],
            theta_r=theta[r],
            b_r=b[r],
            node_model=node_models[r],
            a_r=a[r] if a is not None else None,
            c_r=c[r] if c is not None else None,
            offset=offsets[r] if offsets is not None else None,
        )
    return LikelihoodCache(node_logliks=logliks)


# ===========================================================================
# recompute_node — update one node in place
# ===========================================================================

def recompute_node(
    cache:      LikelihoodCache,
    r:          int,
    data:       PreparedData,
    theta_r:    np.ndarray,
    b_r:        np.ndarray,
    node_model: NodeModel,
    a_r:        np.ndarray | None = None,
    c_r:        np.ndarray | None = None,
    offset:     np.ndarray | None = None,
) -> float:
    """Recompute node r's log-likelihood and update the cache in place.

    The sampler calls it for every augmented node once per stored draw.
    It computes ℓ_r from the current parameters and stores the result in
    ``cache.node_logliks[r]``.  The R-1 other entries are untouched.

    Parameters
    ----------
    cache : LikelihoodCache
        Cache to update in place.
    r : int
        Node index to recompute.
    data : PreparedData
        Prepared data (uses ``data.Y_star[r]`` and ``data.T[r]``).
    theta_r : ndarray [P]
    b_r : ndarray [I]
    node_model : NodeModel
    a_r, c_r : optional
    offset : ndarray [P × I] or None — covariate linear predictor

    Returns
    -------
    float
        The newly computed log-likelihood for node r (also stored in cache).
    """
    new_ll = log_node_lik(
        y_star=data.Y_star[r],
        T_r=data.T[r],
        theta_r=theta_r,
        b_r=b_r,
        node_model=node_model,
        a_r=a_r,
        c_r=c_r,
        offset=offset,
    )
    cache.update(r, new_ll)
    return new_ll


# ===========================================================================
# Internal helpers
# ===========================================================================

def _check_length(name: str, lst: list, expected: int) -> None:
    if len(lst) != expected:
        raise ValueError(
            f"'{name}' must have length {expected} (one per node); "
            f"got {len(lst)}."
        )
