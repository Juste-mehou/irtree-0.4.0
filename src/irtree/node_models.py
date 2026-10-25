"""
irtree.node_models
==================
Layer 3 — Node-level probability models for IRTree decision nodes.

Each node model defines Pr(Y*_r = y | theta_r, params_r) and declares
whether it supports Albert-Chib data augmentation via the `augmentable`
property.

Available node models (and the easiest way to build them)
---------------------------------------------------------
Dichotomous: ``ProbitRaschNode`` / ``LogitRaschNode`` (no discrimination);
``ProbitTwoPLNode`` / ``LogitTwoPLNode`` (discrimination a); ``ProbitThreePLNode``
/ ``LogitThreePLNode`` (a and guessing c).  Polytomous / mixed-format:
``GRMNode`` (probit, cumulative), ``GPCMNode`` (logit, adjacent-category,
item discriminations) and ``PCMNode`` (GPCM with a_i = 1).  ``CustomNode``
is the base for a user-defined likelihood.  Probit nodes are augmentable
(Gibbs updates given the latent Z); logit, GPCM/PCM and custom nodes are
updated by Metropolis-Hastings against ``log_prob``.

Any node can be built directly (``LogitTwoPLNode(person_covariates=["female"])``)
or via the one-line factory ``make_node``::

    from irtree.node_models import make_node
    nodes = [
        make_node("logit-2pl", person_covariates=["female", "escs"]),  # disengagement
        make_node("grm", n_outcomes=3),                                # ability
    ]

``make_node`` keys are case-insensitive; the link-ambiguous ones ("rasch",
"2pl", "3pl") default to probit, with explicit "logit-2pl" etc. for logit.
See ``NODE_TYPES`` for the full list.

Mixed-format nodes
------------------
``GRMNode`` and ``GPCMNode`` support nodes where some items are binary
(2 pseudo-response outcomes) and others are polytomous (M > 2 outcomes).
This is specified via the ``pseudo_item_outcomes`` constructor argument,
which is an [I] integer array giving the number of distinct pseudo-response
levels for each item at this node.

    GRMNode(n_outcomes=3)                                   # homogeneous
    GRMNode(pseudo_item_outcomes=np.array([2,2,3,3,2,3]))   # mixed

When the node is mixed, ``b_r`` passed to ``log_prob`` must be a
``NodeBParams`` instance rather than a plain array.  Homogeneous nodes
take a plain array.

The name ``pseudo_item_outcomes`` is deliberate: these are the number of
distinct *pseudo-response* values item i can produce at node r, not the
number of observed categories on the full test.

Covariate selection
--------------------
Node models declare which covariate columns they use via three
per-instance attributes::

    person_covariates : None | True | Sequence[str | int]  — X [P, K_x]
    item_covariates    : None | True | Sequence[str | int]  — Q [I, K_q]
    pi_covariates      : None | True | Sequence[str | int]  — W [P, I, K_w]

``None`` (the default) means the node uses no covariates of that type.
``True`` means "every column". A list of column names (requires the matching ``X_names`` /
``Q_names`` / ``W_names`` to have been passed to ``expand_data``) or
integer indices selects exactly those columns, in the given order.

These are **specification, not data** — the actual covariate arrays live
in ``PreparedData`` (see ``data_prep.py``). Setting them never touches a
data array; ``data_prep.resolve_covariate_indices()`` is what turns a
node's selection into concrete column indices, once, before sampling.

Two equivalent ways to set them:

    node = ProbitRaschNode(person_covariates=["rt", "n_omitted"])   # per instance

    class MyRaschWithRT(ProbitRaschNode):                            # per subclass
        person_covariates = ["rt", "n_omitted"]

``has_person_covariates`` / ``has_item_covariates`` / ``has_pi_covariates``
are derived read-only properties, True iff the corresponding attribute is
truthy.

Pooling across shared parameters
---------------------------------
If two nodes share a latent dimension (``spec.node_to_latent``), their
``person_covariates`` must be declared identically — a covariate meant
to explain a shared trait cannot have two different coefficients for
that trait, since that would reopen the exact indeterminacy the sharing
was meant to close. The sampler then draws one pooled beta for the
group instead of one per node. The same applies to ``item_covariates``
across a shared b-group (``Constraints(b_shared=...)``).  Coefficients are
therefore keyed by group, not by node: ``beta_g``, ``kappa_g``, ``gamma_g``
with g the node's x-, q- or w-group (``Constraints(beta_shared=...,
kappa_shared=..., gamma_shared=...)`` pool them explicitly). This is
validated by ``data_prep.resolve_covariate_indices()``, not here — this
module only declares the selection, it doesn't know about other nodes.

The ``offset`` argument on ``log_prob``
----------------------------------------
When covariates are present, the sampler computes a [P, I] offset matrix::

    offset[p, i] = X_p @ beta_g  +  Q_i @ kappa_g  +  W_{pi} @ gamma_g

(g the node's coefficient group, using only the node's selected columns)
and passes it to ``log_prob``.  Every concrete class adds this offset to
the ability term before any discrimination scaling::

    eta = a_i * ((theta_p - b_i) + offset[p, i])      (a_i = 1 without a)

With ``offset=None`` the linear predictor is a_i (theta_p - b_i).

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.special import log_ndtr as _log_ndtr
from scipy.stats import norm as _norm

# ---------------------------------------------------------------------------
# Sentinel + helper for the covariate-selection attributes
# ---------------------------------------------------------------------------

_UNSET = object()   # distinguishes "argument not passed" from "explicitly None"


def _has_covariates(value) -> bool:
    """Safe truthiness check for a person/item/pi_covariates value.

    Deliberately avoids ``bool(value)`` directly: if a user passes a
    numpy array of column names/indices instead of a plain list, a bare
    ``bool()`` on a multi-element array raises (ambiguous truth value).
    """
    if value is None or value is False:
        return False
    if value is True:
        return True
    return len(value) > 0


def _normalize_cov(value):
    """Normalize a covariate selection to the canonical form.

    Accepts the ergonomic shorthands and returns what the rest of the
    package expects (None / False / True / a sequence of names-or-indices):

    - ``None`` / ``False`` / ``True`` pass through unchanged.
    - A bare **string** becomes a one-element list: ``"rt"`` → ``["rt"]``.
      This is the important case — without it a bare string would be treated
      as a *sequence of single characters* ("r", "t"), so
      ``person_covariates="rt"`` would silently select columns named "r" and
      "t".  Normalizing here makes the single-covariate shorthand safe.
    - Anything else (list, tuple, ndarray of names/indices) is returned as is.
    """
    if value is None or value is False or value is True:
        return value
    if isinstance(value, str):
        return [value]
    return value


# ---------------------------------------------------------------------------
# NodeBParams — b_r container for mixed-format nodes
# ---------------------------------------------------------------------------

@dataclass
class NodeBParams:
    """Threshold / difficulty parameters for a mixed-format node.

    Holds separate arrays for binary items (one scalar threshold each)
    and polytomous items (one threshold vector each), together with the
    index arrays that map each group back into the full [I] item space.

    Passed as ``b_r`` to ``log_prob``, ``gibbs_b``, and
    ``gibbs_thresholds_grm`` when ``node_model.is_mixed`` is True.

    Attributes
    ----------
    b_binary : np.ndarray, shape [I_bin]
        Scalar threshold τ_i for each binary item.  At a GRM node, drawn by
        ``gibbs_b`` (Normal full conditional, Albert-Chib); at a GPCM node,
        by item-wise MH.
    b_poly : np.ndarray, shape [I_poly, M_poly - 1]
        Ordered threshold vector for each polytomous item.  At a GRM node,
        updated by a Cowles step (Z integrated out) followed by
        ``gibbs_thresholds_grm`` (the prior truncated to the interval the
        category indicators allow); at a GPCM node, by item-wise MH.
        All polytomous items at a given node share the same M (see
        validation in ``GRMNode.__init__``).
    bin_idx : np.ndarray, shape [I_bin], dtype int
        Positions of binary items in the full [I] item array.
    poly_idx : np.ndarray, shape [I_poly], dtype int
        Positions of polytomous items in the full [I] item array.
    M_poly : int
        Number of pseudo-response outcomes for polytomous items (uniform
        across all polytomous items at this node).
    """
    b_binary:  np.ndarray
    b_poly:    np.ndarray
    bin_idx:   np.ndarray
    poly_idx:  np.ndarray
    M_poly:    int

    @property
    def has_binary(self) -> bool:
        """True when at least one binary item is present."""
        return len(self.bin_idx) > 0

    @property
    def has_poly(self) -> bool:
        """True when at least one polytomous item is present."""
        return len(self.poly_idx) > 0


# ---------------------------------------------------------------------------
# Helper: linear predictor
# ---------------------------------------------------------------------------

def _linear_predictor(theta_r: np.ndarray, b_r: np.ndarray) -> np.ndarray:
    """Compute eta[p, i] = theta_r[p] - b_r[i] via broadcasting."""
    return theta_r[:, None] - b_r[None, :]


def _log_sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable log-sigmoid: log(1 / (1 + exp(-x)))."""
    return np.where(x >= 0, -np.log1p(np.exp(-x)), x - np.log1p(np.exp(x)))


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class NodeModel(ABC):
    """Abstract base for all node models.

    Covariate selection
    -------------------
    Three attributes declare which covariate columns this node model
    uses.  They default to None (no covariates); set them either per
    subclass (class-level attribute) or per instance (constructor
    keyword) — see the module docstring for both styles.

        person_covariates : None | True | Sequence[str | int]  — X
        item_covariates    : None | True | Sequence[str | int]  — Q
        pi_covariates      : None | True | Sequence[str | int]  — W

    These carry no data — they're resolved against the actual X/Q/W
    matrices once, by ``data_prep.resolve_covariate_indices()``, before
    sampling starts.  ``has_person_covariates`` etc. remain available as
    derived boolean properties for any code that only needs presence/
    absence, not the selection itself.
    """

    link: str = "probit"
    has_discrimination: bool = False
    has_guessing: bool = False
    n_outcomes: int = 2

    #: Share one scalar discrimination across all of this node's items, rather
    #: than estimating one a_i per item: the "common-slope" node, e.g. the PISA
    #: disengagement node with a single a^(1).  Probit nodes draw it by exact
    #: Gibbs pooled over every cell (MH under a LogNormal prior); logit / GPCM
    #: nodes by one log-scale MH proposal.  Not allowed in a_shared / a_corr
    #: groups.  Ignored when has_discrimination is False.
    common_discrimination: bool = False

    # -- covariate selection (data-free; None = unused, True = every column) -
    person_covariates: bool | Sequence | None = None
    item_covariates:   bool | Sequence | None = None
    pi_covariates:     bool | Sequence | None = None

    def __init__(
        self,
        *,
        person_covariates=_UNSET,
        item_covariates=_UNSET,
        pi_covariates=_UNSET,
        common_discrimination=_UNSET,
    ):
        """Optionally set covariate selection per instance.

        Any argument left at its default (``_UNSET``) falls through to
        whatever is declared at the class level, so a subclass that sets
        ``person_covariates = True`` still works with a plain
        no-argument constructor call. Passing a value here (including
        ``None``) always overrides the class-level default for this
        instance.

        Whatever the source (constructor keyword or class-level attribute),
        the effective value is normalized once here and stored on the
        instance, so a single-covariate string shorthand
        (``person_covariates="rt"``) is turned into ``["rt"]`` and the raw
        attribute every downstream reader sees is already canonical.
        """
        # Resolve each selection to its effective value (explicit arg wins over
        # the class-level default), normalize the shorthand, and store it on the
        # instance so data_prep.resolve_covariate_indices() reads a canonical
        # value regardless of how (or at which level) it was set.
        eff_person = person_covariates if person_covariates is not _UNSET \
            else type(self).person_covariates
        eff_item = item_covariates if item_covariates is not _UNSET \
            else type(self).item_covariates
        eff_pi = pi_covariates if pi_covariates is not _UNSET \
            else type(self).pi_covariates
        self.person_covariates = _normalize_cov(eff_person)
        self.item_covariates   = _normalize_cov(eff_item)
        self.pi_covariates     = _normalize_cov(eff_pi)
        if common_discrimination is not _UNSET:
            self.common_discrimination = bool(common_discrimination)

    @property
    def has_person_covariates(self) -> bool:
        return _has_covariates(self.person_covariates)

    @property
    def has_item_covariates(self) -> bool:
        return _has_covariates(self.item_covariates)

    @property
    def has_pi_covariates(self) -> bool:
        return _has_covariates(self.pi_covariates)

    @property
    def augmentable(self) -> bool:
        """True if this node supports Albert-Chib Gibbs sampling.

        True for the probit nodes (Rasch, 2PL, 3PL, GRM); False for logit
        nodes (logit Rasch/2PL/3PL, GPCM/PCM) and for CustomNode.
        Subclasses may override, but the default links augmentability to
        the probit link so the rule is stated in one place.
        """
        return self.link == "probit"

    @abstractmethod
    def log_prob(
        self,
        y_star: np.ndarray,
        theta_r: np.ndarray,
        b_r,
        a_r: np.ndarray | None = None,
        c_r: np.ndarray | None = None,
        offset: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return [P × I] log Pr(Y*_r = y_star | theta_r, b_r, ...).

        Parameters
        ----------
        y_star  : ndarray [P × I]  observed pseudo-responses
        theta_r : ndarray [P]      person trait parameters
        b_r     : ndarray or NodeBParams  item difficulty / threshold parameters
        a_r     : ndarray [I] or None     discrimination (2PL/3PL only)
        c_r     : ndarray [I] or None     guessing (3PL only)
        offset  : ndarray [P × I] or None
            Covariate adjustment added to the linear predictor before the
            CDF is evaluated::

                eta_full[p, i] = (theta_p - b_i) + offset[p, i]

            When None (default) the code path is identical to the
            pre-covariate behaviour — no overhead, no branching cost.
        """


# ---------------------------------------------------------------------------
# Probit Rasch (default — full Gibbs via Albert-Chib)
# ---------------------------------------------------------------------------

class ProbitRaschNode(NodeModel):
    """Binary probit Rasch node.

    Pr(Y*=1 | theta, b) = Phi(theta - b)

    Supports full Albert-Chib augmentation: theta, b, sigma2 are all
    drawn via exact Gibbs.
    """

    link = "probit"
    has_discrimination = False
    has_guessing = False
    n_outcomes = 2

    def log_prob(self, y_star, theta_r, b_r, a_r=None, c_r=None,
                 offset=None):
        eta = _linear_predictor(theta_r, b_r)
        if offset is not None:
            eta = eta + offset
        sign = 2.0 * y_star - 1.0
        return _log_ndtr(sign * eta)


# ---------------------------------------------------------------------------
# Logit Rasch (MH for all parameters)
# ---------------------------------------------------------------------------

class LogitRaschNode(NodeModel):
    """Binary logit Rasch node.

    Pr(Y*=1 | theta, b) = sigmoid(theta - b)

    Does NOT support augmentation — all parameters require MH steps.
    """

    link = "logit"
    has_discrimination = False
    has_guessing = False
    n_outcomes = 2

    def log_prob(self, y_star, theta_r, b_r, a_r=None, c_r=None,
                 offset=None):
        eta = _linear_predictor(theta_r, b_r)
        if offset is not None:
            eta = eta + offset
        sign = 2.0 * y_star - 1.0
        return _log_sigmoid(sign * eta)


# ---------------------------------------------------------------------------
# Probit 2PL (Gibbs for theta/b, MH for a)
# ---------------------------------------------------------------------------

class ProbitTwoPLNode(NodeModel):
    """Binary probit 2PL node.

    Pr(Y*=1 | theta, a, b) = Phi(a * (theta - b))
    """

    link = "probit"
    has_discrimination = True
    has_guessing = False
    n_outcomes = 2

    def log_prob(self, y_star, theta_r, b_r, a_r=None, c_r=None,
                 offset=None):
        if a_r is None:
            raise ValueError("ProbitTwoPLNode requires a_r.")
        # offset carries the RAW covariate linear predictor (X@beta + Q@kappa +
        # W@gamma); under 2PL the whole ability term is scaled by a_i, so the
        # offset must be added BEFORE the a_i scaling.  This matches sample_Z
        # and the GRM/GPCM binary path.
        eta = _linear_predictor(theta_r, b_r)
        if offset is not None:
            eta = eta + offset
        eta = a_r[None, :] * eta
        sign = 2.0 * y_star - 1.0
        return _log_ndtr(sign * eta)


# ---------------------------------------------------------------------------
# Probit 3PL (Gibbs given Z and the guessing indicators U)
# ---------------------------------------------------------------------------

class ProbitThreePLNode(NodeModel):
    """Binary probit 3PL node.

    Pr(Y*=1 | theta, a, b, c) = c + (1-c) * Phi(a * (theta - b))
    """

    link = "probit"
    has_discrimination = True
    has_guessing = True
    n_outcomes = 2

    def log_prob(self, y_star, theta_r, b_r, a_r=None, c_r=None,
                 offset=None):
        if a_r is None or c_r is None:
            raise ValueError("ProbitThreePLNode requires a_r and c_r.")
        # See ProbitTwoPLNode: raw offset added before a_i scaling.
        eta = _linear_predictor(theta_r, b_r)
        if offset is not None:
            eta = eta + offset
        eta = a_r[None, :] * eta
        prob_1 = c_r[None, :] + (1.0 - c_r[None, :]) * _norm.cdf(eta)
        prob_1 = np.clip(prob_1, 1e-15, 1.0 - 1e-15)
        return np.where(y_star == 1, np.log(prob_1), np.log(1.0 - prob_1))


# ---------------------------------------------------------------------------
# Logit 2PL (MH for all parameters)
# ---------------------------------------------------------------------------

class LogitTwoPLNode(NodeModel):
    """Binary logit 2PL node.

    Pr(Y*=1 | theta, a, b) = sigmoid(a * (theta - b))

    The logit counterpart of :class:`ProbitTwoPLNode`.  Does NOT support
    Albert-Chib augmentation (logit link), so every parameter is updated by
    Metropolis-Hastings.

    Covariate offset ordering mirrors the probit 2PL: the raw covariate
    linear predictor is added to ``theta - b`` BEFORE the ``a`` scaling, so
    the covariate contribution is on the same (pre-discrimination) scale as
    the ability term::

        eta = a_i * ((theta_p - b_i) + offset[p, i])

    The discriminations are updated item by item by log-scale random-walk
    MH against the node log-likelihood and the ``a`` prior (jointly with
    the other members when ``a`` is shared or correlated).
    """

    link = "logit"
    has_discrimination = True
    has_guessing = False
    n_outcomes = 2

    def log_prob(self, y_star, theta_r, b_r, a_r=None, c_r=None,
                 offset=None):
        if a_r is None:
            raise ValueError("LogitTwoPLNode requires a_r.")
        eta = _linear_predictor(theta_r, b_r)
        if offset is not None:
            eta = eta + offset
        eta = a_r[None, :] * eta
        sign = 2.0 * y_star - 1.0
        return _log_sigmoid(sign * eta)


# ---------------------------------------------------------------------------
# Logit 3PL (MH for all parameters)
# ---------------------------------------------------------------------------

class LogitThreePLNode(NodeModel):
    """Binary logit 3PL node.

    Pr(Y*=1 | theta, a, b, c) = c + (1 - c) * sigmoid(a * (theta - b))

    The logit counterpart of :class:`ProbitThreePLNode` — the classic
    Birnbaum 3PL.  Does NOT support augmentation; all parameters via MH.

    Covariate offset is added before the ``a`` scaling, as in the 2PL.

    ``a`` and ``c`` are updated item by item by MH against the node
    log-likelihood (log scale for ``a``, logit scale for ``c``).  A guessing
    parameter shared with other nodes (``Constraints(c_shared=...)``) is not
    supported at a logit node: Constraints.validate raises
    NotImplementedError.
    """

    link = "logit"
    has_discrimination = True
    has_guessing = True
    n_outcomes = 2

    def log_prob(self, y_star, theta_r, b_r, a_r=None, c_r=None,
                 offset=None):
        if a_r is None or c_r is None:
            raise ValueError("LogitThreePLNode requires a_r and c_r.")
        eta = _linear_predictor(theta_r, b_r)
        if offset is not None:
            eta = eta + offset
        eta = a_r[None, :] * eta
        # sigmoid(eta) via the stable log-sigmoid, then the guessing mixture.
        s = np.exp(_log_sigmoid(eta))
        prob_1 = c_r[None, :] + (1.0 - c_r[None, :]) * s
        prob_1 = np.clip(prob_1, 1e-15, 1.0 - 1e-15)
        return np.where(y_star == 1, np.log(prob_1), np.log(1.0 - prob_1))


# ---------------------------------------------------------------------------
# GRM node (polytomous probit, cumulative — augmentable via Johnson-Albert)
# ---------------------------------------------------------------------------

def _validate_pseudo_item_outcomes(pio: np.ndarray, caller: str) -> None:
    """Shared validation for pseudo_item_outcomes arrays.

    Checks:
    1. All values >= 2 (minimum: binary).
    2. All polytomous items share the same M (no variable-M ragged storage).

    Parameters
    ----------
    pio : np.ndarray
        The pseudo_item_outcomes array to validate.
    caller : str
        Name of the calling class, used in error messages.
    """
    if np.any(pio < 2):
        bad = pio[pio < 2].tolist()
        raise ValueError(
            f"{caller}: all pseudo_item_outcomes values must be >= 2 "
            f"(minimum is binary with 2 pseudo-response levels). "
            f"Got values < 2: {bad}."
        )
    poly = pio[pio > 2]
    if len(poly) > 0 and not np.all(poly == poly[0]):
        raise ValueError(
            f"{caller}: all polytomous items at this node must have the "
            f"same number of pseudo-response outcomes (uniform M within "
            f"the polytomous group). Variable M across polytomous items "
            f"is not supported — it requires ragged threshold storage and "
            f"per-item conditionals in every downstream function. "
            f"Got M values: {np.unique(poly).tolist()}. "
            f"If your items genuinely differ in M, split them across "
            f"separate nodes or recode to a common M."
        )


class GRMNode(NodeModel):
    """Probit Graded Response Model node (cumulative, Samejima 1969).

    Supports three usage modes:

    1. **Homogeneous polytomous** — all items have the same M ≥ 2::

           GRMNode(n_outcomes=3)

       ``b_r`` passed to ``log_prob`` is a plain ``[I, M-1]`` array
       (or ``[I]`` when M=2, which is equivalent to ``ProbitRaschNode``).

    2. **Homogeneous binary** — all items binary, GRM formula::

           GRMNode(n_outcomes=2)

       Mathematically identical to ``ProbitRaschNode``.  Provided so
       that a pure-binary test can be upgraded to mixed without changing
       the class.  ``b_r`` is a plain ``[I]`` array.

    3. **Mixed** — some items binary, others polytomous (same M)::

           GRMNode(pseudo_item_outcomes=np.array([2, 2, 3, 3, 2, 3]))

       ``b_r`` passed to ``log_prob`` must be a ``NodeBParams`` instance.
       ``n_outcomes`` is set to ``max(pseudo_item_outcomes)`` automatically.

    Item parameters
    ---------------
    *Homogeneous polytomous / binary:*
        ``b_r``: ``[I, M-1]`` for M ≥ 3; ``[I]`` for M = 2.
        Thresholds must be ordered (τ_1 < τ_2 < ... < τ_{M-1}) per item
        for M ≥ 3.  Disordered thresholds produce ``-inf`` in ``log_prob``
        and are rejected by the MH sampler.

    *Mixed:*
        ``b_r``: ``NodeBParams`` with ``b_binary [I_bin]`` and
        ``b_poly [I_poly, M_poly-1]``.

    The ``offset`` argument
    -----------------------
    For homogeneous M=2 and mixed binary items, ``offset`` is added to
    the scalar linear predictor ``θ_p − τ_i`` before Φ is evaluated.
    For polytomous items (M ≥ 3), ``offset`` is broadcast across all
    M-1 threshold comparisons: the full linear predictor becomes
    ``θ_p − τ_{i,k} + offset[p, i]`` for each threshold k.

    Augmentation
    ------------
    Always augmentable (probit link).  For binary items, the Albert-Chib
    Normal full conditional (``gibbs_b``) is used.  For polytomous items,
    each iteration runs a Cowles (1996) step on the thresholds with Z
    integrated out, redraws Z, then draws the thresholds exactly given Z
    (``gibbs_thresholds_grm``: the prior truncated to the interval the
    category indicators allow).  Shared thresholds are pooled over every
    member node.  The sampler dispatches to the appropriate update for each
    group; ``GRMNode`` itself does not need to know which.

    Parameters
    ----------
    n_outcomes : int, optional
        Number of ordered pseudo-response outcomes, uniform across all items.
        Must be >= 2.  Required when ``pseudo_item_outcomes`` is not given.
    pseudo_item_outcomes : np.ndarray of int, optional
        Per-item outcome counts, shape [I].  Values must all be >= 2 and
        all values > 2 must be equal (uniform M in the polytomous group).
        Required when ``n_outcomes`` is not given.
    """

    link = "probit"
    has_discrimination = False
    has_guessing = False

    def __init__(
        self,
        n_outcomes: int | None = None,
        pseudo_item_outcomes: np.ndarray | None = None,
        *,
        person_covariates=_UNSET,
        item_covariates=_UNSET,
        pi_covariates=_UNSET,
        common_discrimination=_UNSET,
    ):
        super().__init__(
            person_covariates=person_covariates,
            item_covariates=item_covariates,
            pi_covariates=pi_covariates,
            common_discrimination=common_discrimination,
        )
        if n_outcomes is None and pseudo_item_outcomes is None:
            raise ValueError(
                "GRMNode requires either n_outcomes (homogeneous) or "
                "pseudo_item_outcomes (per-item).  Neither was provided."
            )

        if pseudo_item_outcomes is not None:
            pio = np.asarray(pseudo_item_outcomes, dtype=int)
            _validate_pseudo_item_outcomes(pio, "GRMNode")
            self.pseudo_item_outcomes = pio
            derived_n = int(np.max(pio))
            if n_outcomes is not None and n_outcomes != derived_n:
                raise ValueError(
                    f"GRMNode: n_outcomes={n_outcomes} conflicts with "
                    f"max(pseudo_item_outcomes)={derived_n}. "
                    f"Either omit n_outcomes or ensure it equals the maximum "
                    f"value in pseudo_item_outcomes."
                )
            self.n_outcomes = derived_n
        else:
            # Homogeneous path
            if n_outcomes < 2:
                raise ValueError(
                    f"GRMNode requires n_outcomes >= 2; got {n_outcomes}. "
                    f"(n_outcomes=2 is valid: GRM with one threshold per item, "
                    f"mathematically identical to ProbitRaschNode.)"
                )
            self.n_outcomes = n_outcomes
            self.pseudo_item_outcomes = None

    # -- item-group properties (None for homogeneous nodes) ------------------

    @property
    def is_mixed(self) -> bool:
        """True when the node has both binary and polytomous items."""
        return bool(
            self.pseudo_item_outcomes is not None
            and np.any(self.pseudo_item_outcomes == 2)
            and np.any(self.pseudo_item_outcomes  > 2)
        )

    @property
    def bin_idx(self) -> np.ndarray | None:
        """0-based indices of binary items, or None for homogeneous nodes."""
        if self.pseudo_item_outcomes is None:
            return None
        return np.where(self.pseudo_item_outcomes == 2)[0]

    @property
    def poly_idx(self) -> np.ndarray | None:
        """0-based indices of polytomous items, or None for homogeneous nodes."""
        if self.pseudo_item_outcomes is None:
            return None
        return np.where(self.pseudo_item_outcomes > 2)[0]

    @property
    def M_poly(self) -> int | None:
        """Uniform M for polytomous items, or None when no polytomous items."""
        pidx = self.poly_idx
        if pidx is None or len(pidx) == 0:
            return None
        return int(self.pseudo_item_outcomes[pidx[0]])

    # -- log-probability -----------------------------------------------------

    def log_prob(self, y_star, theta_r, b_r, a_r=None, c_r=None,
                 offset=None):
        """Log-probability under the probit GRM.

        Parameters
        ----------
        y_star : ndarray [P × I]
            Observed pseudo-responses, values in {0, ..., M-1}.
        theta_r : ndarray [P]
            Person trait parameters.
        b_r : ndarray [I × (M-1)] | ndarray [I] | NodeBParams
            Threshold parameters.

            - Plain ``[I, M-1]`` or ``[I]`` array for homogeneous nodes.
            - ``NodeBParams`` for mixed nodes (``self.is_mixed`` is True).
        a_r : ndarray [I] or None
            Item-level discrimination: Ψ_k(θ) = Φ(a_i(θ - τ_k)).
            Default None (discrimination = 1 for all items).
        c_r : ignored (no guessing in GRM).
        offset : ndarray [P × I] or None
            Covariate offset added to the linear predictor.  For M=2,
            added to the scalar eta.  For M≥3, broadcast across all
            threshold dimensions: the per-cell contribution is
            ``offset[p, i]`` and is the same across all M-1 thresholds
            within a given (p, i) cell.

        Returns
        -------
        ndarray [P × I]
            Log-probability of the observed pseudo-response at each cell.
        """
        if isinstance(b_r, NodeBParams):
            return self._log_prob_mixed(y_star, theta_r, b_r, a_r, offset)
        return self._log_prob_homogeneous(y_star, theta_r, b_r, a_r, offset)

    def _log_prob_homogeneous(self, y_star, theta_r, b_r, a_r=None,
                               offset=None):
        """GRM log-probability for a homogeneous item set (uniform M).

        Works for M=2 (binary case: one threshold per item, [I] array)
        as well as M≥3 ([I, M-1] array).

        The offset [P, I] is added to the linear predictor.  For M=2 this
        is a simple addition to eta.  For M≥3 the same per-cell offset is
        broadcast across all M-1 threshold comparisons within each cell,
        shifting θ_p relative to all thresholds uniformly — equivalent to
        treating the covariate-adjusted quantity θ_p + offset[p,i] as the
        effective person parameter at cell (p, i).
        """
        M = self.n_outcomes
        P = theta_r.shape[0]

        # Handle M=2 specially: b_r is [I], not [I, 1]
        if M == 2:
            b_1d = np.asarray(b_r).ravel()
            I = b_1d.shape[0]
            eta = theta_r[:, None] - b_1d[None, :]   # [P, I]
            if offset is not None:
                eta = eta + offset
            if a_r is not None:
                eta = a_r[None, :] * eta
            y_bin = np.asarray(np.nan_to_num(y_star, nan=0.0), dtype=int)
            return _log_ndtr((2 * y_bin - 1) * eta)

        # M >= 3 path
        b_2d = np.asarray(b_r)
        I = b_2d.shape[0]
        y_int = np.asarray(np.nan_to_num(y_star, nan=0.0), dtype=int)

        if b_2d.shape != (I, M - 1):
            raise ValueError(
                f"GRMNode with {M} outcomes expects b_r shape [{I}, {M-1}]; "
                f"got {np.asarray(b_r).shape}."
            )

        eta = theta_r[:, None, None] - b_2d[None, :, :]   # [P, I, M-1]
        if offset is not None:
            # offset is [P, I]; broadcast to [P, I, M-1]
            eta = eta + offset[:, :, np.newaxis]
        if a_r is not None:
            eta = a_r[None, :, None] * eta

        cum_prob = _norm.cdf(eta)   # [P, I, M-1]

        cat_prob = np.empty((P, I, M))
        cat_prob[:, :, 0] = 1.0 - cum_prob[:, :, 0]
        for m in range(1, M - 1):
            cat_prob[:, :, m] = cum_prob[:, :, m - 1] - cum_prob[:, :, m]
        cat_prob[:, :, M - 1] = cum_prob[:, :, M - 2]

        cat_prob = np.clip(cat_prob, 1e-300, None)

        p_idx = np.arange(P)[:, None]
        i_idx = np.arange(I)[None, :]
        result = np.log(cat_prob[p_idx, i_idx, y_int])

        return result

    def _log_prob_mixed(self, y_star, theta_r, b_r: NodeBParams, a_r=None,
                        offset=None):
        """GRM log-probability for a mixed binary/polytomous item set.

        Routes binary items through the M=2 probit formula (identical to
        ProbitRasch) and polytomous items through the standard GRM formula,
        then assembles both groups into a single [P, I] result array.

        When ``offset`` is not None, the relevant sub-matrix of the offset
        array is extracted for each group before being passed to the
        homogeneous helper.
        """
        P = theta_r.shape[0]
        I = len(b_r.bin_idx) + len(b_r.poly_idx)
        result = np.empty((P, I))

        if b_r.has_binary:
            # M=2: Φ(a_i(θ - τ)) — one scalar threshold per binary item
            eta = theta_r[:, None] - b_r.b_binary[None, :]   # [P, I_bin]
            if offset is not None:
                eta = eta + offset[:, b_r.bin_idx]
            if a_r is not None:
                eta = a_r[b_r.bin_idx][None, :] * eta
            y_bin = np.asarray(np.nan_to_num(y_star[:, b_r.bin_idx], nan=0.0), dtype=int)
            result[:, b_r.bin_idx] = _log_ndtr((2 * y_bin - 1) * eta)

        if b_r.has_poly:
            # Temporarily override n_outcomes for the polytomous subgroup
            orig_n = self.n_outcomes
            self.n_outcomes = b_r.M_poly
            a_poly = a_r[b_r.poly_idx] if a_r is not None else None
            off_poly = offset[:, b_r.poly_idx] if offset is not None else None
            result[:, b_r.poly_idx] = self._log_prob_homogeneous(
                y_star[:, b_r.poly_idx], theta_r, b_r.b_poly, a_poly,
                off_poly,
            )
            self.n_outcomes = orig_n

        return result


# ---------------------------------------------------------------------------
# GPCM node (polytomous logit, adjacent category — MH for all parameters)
# ---------------------------------------------------------------------------

class GPCMNode(NodeModel):
    """Logit Generalized Partial Credit Model node (adjacent category).

    Supports three usage modes (same pattern as ``GRMNode``):

    1. **Homogeneous polytomous**: ``GPCMNode(n_outcomes=3)``
    2. **Homogeneous binary**: ``GPCMNode(n_outcomes=2)`` — reduces to PCM/logit Rasch
    3. **Mixed**: ``GPCMNode(pseudo_item_outcomes=np.array([2,3,2,3]))``

    For M ordered outcomes Y* ∈ {0, 1, ..., M-1} and M-1 step difficulties
    b_{i1}, ..., b_{i,M-1} per item, the GPCM defines::

        Pr(Y* = 0) = 1 / D
        Pr(Y* = j) = exp(Σ_{k=1}^{j} a_i(θ - b_{ik})) / D    for j > 0
        D = 1 + Σ_{r=1}^{M-1} exp(Σ_{k=1}^{r} a_i(θ - b_{ik}))

    Unlike the GRM, step difficulties do NOT need to be ordered.
    With a_i = 1 this is the Partial Credit Model (PCM): pass
    ``discrimination=False`` or use ``PCMNode``.

    The ``offset`` argument shifts θ uniformly before all step comparisons:
    η_{pik} = a_i(θ_p + offset[p,i] − b_{ik}).

    **Augmentation:** never augmentable — logit link, all parameters via MH.
    The discriminations a_i are updated item by item by log-scale random-walk
    MH against the node likelihood and the ``a`` prior, like the logit 2PL.

    Parameters
    ----------
    n_outcomes : int, optional
        Uniform outcome count for all items.  Must be >= 2.
    pseudo_item_outcomes : np.ndarray of int, optional
        Per-item outcome counts, shape [I].
    discrimination : bool, default True
        Estimate one a_i per item (GPCM).  False fixes a_i = 1 (PCM).
        ``common_discrimination=True`` estimates one a shared by all items.
    """

    link = "logit"
    has_discrimination = True
    has_guessing = False

    def __init__(
        self,
        n_outcomes: int | None = None,
        pseudo_item_outcomes: np.ndarray | None = None,
        *,
        person_covariates=_UNSET,
        item_covariates=_UNSET,
        pi_covariates=_UNSET,
        common_discrimination=_UNSET,
        discrimination: bool | None = None,
    ):
        super().__init__(
            person_covariates=person_covariates,
            item_covariates=item_covariates,
            pi_covariates=pi_covariates,
            common_discrimination=common_discrimination,
        )
        if discrimination is not None:
            self.has_discrimination = bool(discrimination)
        if n_outcomes is None and pseudo_item_outcomes is None:
            raise ValueError(
                "GPCMNode requires either n_outcomes (homogeneous) or "
                "pseudo_item_outcomes (per-item).  Neither was provided."
            )

        if pseudo_item_outcomes is not None:
            pio = np.asarray(pseudo_item_outcomes, dtype=int)
            _validate_pseudo_item_outcomes(pio, "GPCMNode")
            self.pseudo_item_outcomes = pio
            derived_n = int(np.max(pio))
            if n_outcomes is not None and n_outcomes != derived_n:
                raise ValueError(
                    f"GPCMNode: n_outcomes={n_outcomes} conflicts with "
                    f"max(pseudo_item_outcomes)={derived_n}."
                )
            self.n_outcomes = derived_n
        else:
            if n_outcomes < 2:
                raise ValueError(
                    f"GPCMNode requires n_outcomes >= 2; got {n_outcomes}."
                )
            self.n_outcomes = n_outcomes
            self.pseudo_item_outcomes = None

    # -- item-group properties (mirror GRMNode) ------------------------------

    @property
    def is_mixed(self) -> bool:
        return bool(
            self.pseudo_item_outcomes is not None
            and np.any(self.pseudo_item_outcomes == 2)
            and np.any(self.pseudo_item_outcomes  > 2)
        )

    @property
    def bin_idx(self) -> np.ndarray | None:
        if self.pseudo_item_outcomes is None:
            return None
        return np.where(self.pseudo_item_outcomes == 2)[0]

    @property
    def poly_idx(self) -> np.ndarray | None:
        if self.pseudo_item_outcomes is None:
            return None
        return np.where(self.pseudo_item_outcomes > 2)[0]

    @property
    def M_poly(self) -> int | None:
        pidx = self.poly_idx
        if pidx is None or len(pidx) == 0:
            return None
        return int(self.pseudo_item_outcomes[pidx[0]])

    # -- log-probability -----------------------------------------------------

    def log_prob(self, y_star, theta_r, b_r, a_r=None, c_r=None,
                 offset=None):
        """Log-probability under the GPCM / PCM.

        Parameters
        ----------
        y_star : ndarray [P × I]
        theta_r : ndarray [P]
        b_r : ndarray [I × (M-1)] | ndarray [I] | NodeBParams
        a_r : ndarray [I] or None  (discrimination; None → PCM)
        c_r : ignored.
        offset : ndarray [P × I] or None
            Covariate adjustment.  Added to θ_p for all step comparisons
            within cell (p, i): η_{pik} = a_i(θ_p + offset[p,i] − b_{ik}).

        Returns
        -------
        ndarray [P × I]
        """
        if isinstance(b_r, NodeBParams):
            return self._log_prob_mixed(y_star, theta_r, b_r, a_r, offset)
        return self._log_prob_homogeneous(y_star, theta_r, b_r, a_r, offset)

    def _log_prob_homogeneous(self, y_star, theta_r, b_r, a_r=None,
                               offset=None):
        """GPCM log-probability for a homogeneous item set (uniform M).

        Works for M=2 (binary case: one step difficulty per item, [I] array,
        equivalent to logit Rasch) as well as M>=3 ([I, M-1] array).
        """
        M = self.n_outcomes
        P = theta_r.shape[0]

        # M=2 special case: b_r is [I], step difficulties are scalars
        if M == 2:
            b_1d = np.asarray(b_r).ravel()
            I = b_1d.shape[0]
            eta = theta_r[:, None] - b_1d[None, :]   # [P, I]
            if offset is not None:
                eta = eta + offset
            if a_r is not None:
                eta = a_r[None, :] * eta
            y_bin = np.asarray(np.nan_to_num(y_star, nan=0.0), dtype=int)
            return _log_sigmoid((2 * y_bin - 1) * eta)

        # M >= 3 path
        b_2d = np.asarray(b_r)
        I = b_2d.shape[0]
        y_int = np.asarray(np.nan_to_num(y_star, nan=0.0), dtype=int)

        if b_2d.shape != (I, M - 1):
            raise ValueError(
                f"GPCMNode with {M} outcomes expects b_r shape [{I}, {M-1}]; "
                f"got {np.asarray(b_r).shape}."
            )

        # theta_eff[p, i] = theta_r[p] + offset[p, i]  (offset=0 if None)
        if offset is not None:
            theta_eff = theta_r[:, None] + offset          # [P, I]
            eta_steps = theta_eff[:, :, None] - b_2d[None, :, :]  # [P, I, M-1]
        else:
            eta_steps = theta_r[:, None, None] - b_2d[None, :, :]  # [P, I, M-1]

        if a_r is not None:
            eta_steps = a_r[None, :, None] * eta_steps

        cumsum_eta = np.cumsum(eta_steps, axis=2)   # [P, I, M-1]

        log_numer = np.zeros((P, I, M))
        log_numer[:, :, 1:] = cumsum_eta

        max_ln   = np.max(log_numer, axis=2, keepdims=True)
        log_denom = max_ln + np.log(
            np.sum(np.exp(log_numer - max_ln), axis=2, keepdims=True)
        )

        log_cat_prob = log_numer - log_denom   # [P, I, M]

        p_idx = np.arange(P)[:, None]
        i_idx = np.arange(I)[None, :]
        return log_cat_prob[p_idx, i_idx, y_int]

    def _log_prob_mixed(self, y_star, theta_r, b_r: NodeBParams, a_r=None,
                        offset=None):
        """GPCM log-probability for a mixed binary/polytomous item set."""
        P = theta_r.shape[0]
        I = len(b_r.bin_idx) + len(b_r.poly_idx)
        result = np.empty((P, I))

        if b_r.has_binary:
            # M=2: 2PL logit formula (a_i=1 when a_r is None)
            eta = theta_r[:, None] - b_r.b_binary[None, :]   # [P, I_bin]
            if offset is not None:
                eta = eta + offset[:, b_r.bin_idx]
            if a_r is not None:
                eta = a_r[b_r.bin_idx][None, :] * eta
            y_bin = np.asarray(np.nan_to_num(y_star[:, b_r.bin_idx], nan=0.0), dtype=int)
            result[:, b_r.bin_idx] = _log_sigmoid((2 * y_bin - 1) * eta)

        if b_r.has_poly:
            orig_n = self.n_outcomes
            self.n_outcomes = b_r.M_poly
            a_poly = a_r[b_r.poly_idx] if a_r is not None else None
            off_poly = offset[:, b_r.poly_idx] if offset is not None else None
            result[:, b_r.poly_idx] = self._log_prob_homogeneous(
                y_star[:, b_r.poly_idx], theta_r, b_r.b_poly, a_poly,
                off_poly,
            )
            self.n_outcomes = orig_n

        return result


class PCMNode(GPCMNode):
    """Logit Partial Credit Model node: a GPCMNode with every a_i fixed at 1.

    Same arguments as ``GPCMNode`` (``discrimination`` is fixed to False).
    """

    has_discrimination = False

    def __init__(self, n_outcomes=None, pseudo_item_outcomes=None, **kwargs):
        kwargs.pop("discrimination", None)
        super().__init__(n_outcomes, pseudo_item_outcomes,
                         discrimination=False, **kwargs)


# ---------------------------------------------------------------------------
# Stub for user-defined custom nodes
# ---------------------------------------------------------------------------

class CustomNode(NodeModel):
    """Base class for user-defined node models.

    Subclass this and implement ``log_prob`` (the log-probability of each
    pseudo-response, a [P x I] array; it must accept and use ``offset`` if
    the node has covariates).  A custom node is NOT augmented: the sampler
    updates its theta (person by person) and b (item by item) by
    Metropolis-Hastings against your ``log_prob``, together with every node
    that shares them, and its covariate coefficients and missing covariates
    the same way.  Set ``has_discrimination = True`` to have per-item
    discriminations ``a_r`` estimated by MH and passed to ``log_prob``.
    ``n_outcomes`` > 2 makes ``b_r`` an [I, n_outcomes - 1] threshold array.

    Do not set ``augmentable = True``: augmented nodes are handled by the
    probit Gibbs paths, which use the probit model and never call your
    ``log_prob``.

    If your node requires covariates, set the relevant attributes::

        class MyNode(CustomNode):
            person_covariates = ["rt", "n_omitted"]   # or True for all columns
            item_covariates   = None
            pi_covariates     = None

    The sampler then computes the offset matrix and passes it to
    ``log_prob``.
    """

    link = "custom"
    has_discrimination = False
    has_guessing = False
    n_outcomes = 2

    @property
    def augmentable(self) -> bool:
        """Always False: a custom node is updated by MH on its log_prob."""
        return False

    def log_prob(self, y_star, theta_r, b_r, a_r=None, c_r=None,
                 offset=None):
        raise NotImplementedError(
            "CustomNode.log_prob must be implemented by the subclass."
        )


# ---------------------------------------------------------------------------
# One-line node construction: a small string → class registry + factory
# ---------------------------------------------------------------------------
# The classes above are the primary interface; this is sugar so a tree can be
# specified in one readable line per node, covariates included, without
# importing each class by name:
#
#     from irtree.node_models import make_node
#     nodes = [
#         make_node("logit-2pl", person_covariates=["female", "escs"]),  # disengagement
#         make_node("grm", n_outcomes=3),                                # ability
#     ]
#
# Every keyword is forwarded to the class constructor, so the covariate
# selection (and n_outcomes / pseudo_item_outcomes for GRM/GPCM) is passed
# exactly as it would be to the class directly.

#: Short, case-insensitive names → node classes.  "-" and "_" are
#: interchangeable in a key.  Link-ambiguous shorthands ("rasch", "2pl",
#: "3pl") default to PROBIT, the package default; the logit variants are
#: explicit ("logit-2pl", ...).
NODE_TYPES = {
    "rasch":         ProbitRaschNode,
    "1pl":           ProbitRaschNode,
    "probit-rasch":  ProbitRaschNode,
    "probit-1pl":    ProbitRaschNode,
    "logit-rasch":   LogitRaschNode,
    "logit-1pl":     LogitRaschNode,
    "2pl":           ProbitTwoPLNode,
    "probit-2pl":    ProbitTwoPLNode,
    "logit-2pl":     LogitTwoPLNode,
    "3pl":           ProbitThreePLNode,
    "probit-3pl":    ProbitThreePLNode,
    "logit-3pl":     LogitThreePLNode,
    "grm":           GRMNode,
    "gpcm":          GPCMNode,
    "pcm":           PCMNode,
    "custom":        CustomNode,
}


def make_node(kind: str, **kwargs) -> NodeModel:
    """Construct a node model from a short name, forwarding all keywords.

    Parameters
    ----------
    kind : str
        A key from ``NODE_TYPES`` (case-insensitive; "-" and "_"
        interchangeable).  Link-ambiguous names default to probit:
        "rasch"/"1pl", "2pl", "3pl" → probit; use "logit-2pl" etc. for logit.
    **kwargs
        Passed straight to the chosen class — e.g. ``n_outcomes`` /
        ``pseudo_item_outcomes`` for "grm"/"gpcm", and any of
        ``person_covariates`` / ``item_covariates`` / ``pi_covariates``
        (a bare string is accepted as a one-element selection).

    Returns
    -------
    NodeModel

    Raises
    ------
    ValueError
        If ``kind`` is not a known key (the message lists the valid keys).

    Examples
    --------
    >>> make_node("logit-2pl", person_covariates=["female", "escs"])  # doctest: +ELLIPSIS
    <...LogitTwoPLNode object at ...>
    >>> make_node("grm", n_outcomes=3)                                # doctest: +ELLIPSIS
    <...GRMNode object at ...>
    """
    key = str(kind).strip().lower().replace("_", "-")
    cls = NODE_TYPES.get(key)
    if cls is None:
        valid = ", ".join(sorted(NODE_TYPES))
        raise ValueError(
            f"Unknown node kind {kind!r}. Valid kinds are: {valid}."
        )
    return cls(**kwargs)


__all__ = [
    "NodeModel",
    "NodeBParams",
    "ProbitRaschNode",
    "LogitRaschNode",
    "ProbitTwoPLNode",
    "ProbitThreePLNode",
    "LogitTwoPLNode",
    "LogitThreePLNode",
    "GRMNode",
    "GPCMNode",
    "CustomNode",
    "NODE_TYPES",
    "make_node",
]
