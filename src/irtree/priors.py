"""
irtree.priors
=============
Layer 5 — Prior specification for all parameters in an IRTree model.

THREE-LEVEL SPECIFICATION SYSTEM
=================================

Level 1 — Package default.
Level 2 — Type-level:  priors.set("b", NormalPrior(mean=0, variance=2.0))
Level 3 — Node-level:  priors.set("b", NormalPrior(mean=0, variance=2.0), node=1)

Resolution order: node_level → type_level → package default.

PARAMETER KEYS
==============
IRT parameters (per node):
    "theta"     NormalPrior             person trait (one variance per latent
                                        dimension, resolved from the
                                        dimension's first node)
    "b"         NormalPrior             item difficulty / thresholds
    "a"         TruncNormalPrior        discrimination (nodes with
                (default) or LogNormal  has_discrimination: 2PL, 3PL, GPCM)
    "c"         BetaPrior               guessing (3PL nodes)
    "b.binary"  NormalPrior             binary-item thresholds (mixed nodes)
    "b.poly"    NormalPrior             polytomous-item thresholds (mixed nodes)

Covariate coefficients (per node, only when the node model selects columns):
    "beta"      NormalPrior             person covariate coefficients
    "kappa"     NormalPrior             item covariate coefficients
    "gamma"     NormalPrior             person×item covariate coefficients
    "X_missing" NormalPrior             missing person covariate values (X), global
    "Q_missing" NormalPrior             missing item covariate values (Q), global
    "W_missing" NormalPrior             missing person×item covariate values (W), global

Variance hyperparameters:
    The variance slot of a NormalPrior / TruncNormalPrior holds either a
    float (fixed) or a PriorSpec on (0, ∞), which makes the variance a
    parameter estimated each iteration.  One variance exists per latent
    dimension (theta) or per sharing group (b, a, beta, kappa, gamma); it is
    resolved from the dimension's or group's first node, so a node-level
    set() on that node sets it.  InvGammaPrior → exact Gibbs (conjugate);
    any other PriorSpec (e.g. HalfNormalPrior on the sd) → MH.

DEFAULTS
========
    theta   Normal(0, InvGamma(1, 1)); on a latent dimension where some node
            has free discriminations, Normal(0, 1) at node level (the 2PL
            identification convention: a_i and the theta scale trade off
            along a_i -> c a_i, theta -> theta / c).  These automatic entries
            are listed in summary() and removed by a type-level
            set("theta", ...).
    b, beta, kappa, gamma   Normal(0, InvGamma(1, 1))
    X/Q/W_missing           Normal(0, 1) (centred, standardised covariates)
    a       TruncNormal(1, 0.5) on (0, ∞); inside an a_corr group under an
            LKJ or fixed a.correlation, run_chain estimates each group's
            variance with Half-Normal(0, 0.5) on the sd (default a only)
    c       Beta(5, 17)
    correlations   FixedCorrelation(0) (independent); an InverseWishartPrior
            without a scale uses 0.01·I_G with df = G + 1

BOUNDED DISTRIBUTIONS
=====================
All prior classes accept optional lower / upper keyword arguments
(both default to None, meaning unbounded).

    NormalPrior(mean=0, variance=0.3, lower=-1, upper=1)

For parameters drawn by Gibbs (theta, b, beta, kappa, gamma at probit
nodes) a UserWarning is emitted at set() time because bounds switch the
draw to a truncated Normal.  Wherever a parameter is updated by MH (logit,
GPCM and custom nodes; "a" with LogNormalPrior; "c" at logit nodes;
correlations under LKJ or scalar priors) bounds enter as a -inf
log-density.  For "a" with TruncNormalPrior the bounds are structural: the
draw is an exact truncated-Normal Gibbs step at probit nodes and an MH step
with the truncated prior at logit/GPCM nodes.  At probit 3PL nodes "c" is
an exact Beta Gibbs draw given the guessing indicators.

CORRELATION SPECIFICATION
=========================
Three correlation keys are supported, one per parameter type:

    theta.correlation   — correlation across latent dimensions
    b.correlation       — correlation across b-groups (Constraints.b_corr)
    a.correlation       — correlation across a-groups (Constraints.a_corr)

All three accept the same set of prior_spec values and apply the same rules:

    FixedCorrelation(rho=scalar)        — any G, fixed (not estimated)
    FixedCorrelation(rho=matrix)        — any G, fixed matrix (validated)
    LKJPrior(eta)                       — any G, estimated via vine MH
    InverseWishartPrior(scale, df)      — any G, estimated via exact Gibbs
    NormalPrior(..., lower=-1, upper=1) — G==2 only, Fisher-z MH
    BetaPrior(..., lower=-1, upper=1)   — G==2 only, Fisher-z MH

G is the number of correlated dimensions or groups in one block:
theta.correlation applies to each block of Constraints(theta_corr=...)
(all D latent dimensions when none is given, otherwise block-diagonal);
b.correlation / a.correlation apply to each group of b_corr / a_corr.

For G > 2, bounded scalar priors raise NotImplementedError.
A UserWarning is emitted for G==2 estimated priors without (−1, 1) bounds.

Note: G is not always known at set() time, since it depends on the
Constraints passed to run_chain().  The G-dependent checks
(NotImplementedError for G>2, UserWarning for missing bounds) are therefore
applied at set() time where G is known and again at run_chain() time with
the real G.

CORRELATION CHAIN NAMES (when estimated)
LKJ / bounded scalar path (theta.correlation):
    one rho_i_j (1-based dimensions, i < j) per pair inside a block;
    one block of D dimensions gives D(D-1)/2 names
        D=2: ["rho_1_2"]
        D=3: ["rho_1_2", "rho_1_3", "rho_2_3"]

InverseWishartPrior path (theta.correlation):
    Sigma_theta_i_j for the lower triangle (diagonal included) of each
    block: G(G+1)/2 names per block

b.correlation / a.correlation — LKJ path:
    ["b.rho_{label1}_{label2}", ...]  one per off-diagonal pair per group
b.correlation / a.correlation — InverseWishartPrior path:
    ["Sigma_b_{tag}_{i}_{j}", ...]   lower triangle per group
    ["Sigma_a_{tag}_{i}_{j}", ...]
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

# ===========================================================================
# Prior distribution families
# ===========================================================================

class PriorSpec(ABC):
    """Abstract base for all prior distribution specifications."""

    @abstractmethod
    def free_slots(self) -> list[str]:
        """Names of variance slots that hold a PriorSpec (i.e. are estimated).

        InvGammaPrior → Gibbs. Any other PriorSpec in a variance slot → MH.
        A float means the variance is fixed.
        """

    @abstractmethod
    def display(self, param_name: str) -> str:
        """Human-readable string for priors.summary() TABLE 1."""

    @staticmethod
    def _bounds_str(lower: float | None, upper: float | None) -> str:
        """Return a bounds suffix string, or '' if unbounded."""
        if lower is None and upper is None:
            return ""
        lo = "-∞" if lower is None else lower
        hi = "+∞" if upper is None else upper
        return f" [{lo}, {hi}]"

    @staticmethod
    def _validate_bounds(lower: float | None, upper: float | None) -> None:
        if lower is not None and upper is not None and lower >= upper:
            raise ValueError(
                f"lower must be strictly less than upper; got lower={lower}, upper={upper}."
            )

    @property
    def is_bounded(self) -> bool:
        return (getattr(self, "lower", None) is not None or
                getattr(self, "upper", None) is not None)


@dataclass
class InvGammaPrior(PriorSpec):
    """Inverse-Gamma prior: parameter ~ InvGamma(alpha, beta)."""
    alpha: float = 1.0
    beta:  float = 1.0
    lower: float | None = None
    upper: float | None = None

    def __post_init__(self):
        self._validate_bounds(self.lower, self.upper)

    def free_slots(self) -> list[str]:
        return []

    def display(self, param_name: str) -> str:
        return f"InvGamma({self.alpha}, {self.beta}){self._bounds_str(self.lower, self.upper)}"

    def __repr__(self) -> str:
        return f"InvGammaPrior(alpha={self.alpha}, beta={self.beta})"





@dataclass
class NormalPrior(PriorSpec):
    """Normal prior: parameter ~ N(mean, variance).

    Parameters
    ----------
    lower, upper : Optional[float]
        Truncation bounds.  None means unbounded on that side.
        Bounds are used for correlations (two-dimensional blocks) and
        slopes; location parameters (theta, b, beta, kappa, gamma) reject
        them at set() time, because their Gibbs draws are unbounded.
    """
    mean:     float = 0.0
    variance: float | PriorSpec = 1.0
    lower:    float | None = None
    upper:    float | None = None

    def __post_init__(self):
        self._validate_bounds(self.lower, self.upper)

    def free_slots(self) -> list[str]:
        return ["variance"] if isinstance(self.variance, PriorSpec) else []

    def display(self, param_name: str) -> str:
        if isinstance(self.variance, PriorSpec):
            base = f"Normal({self.mean}, [estimated~{self.variance.__class__.__name__}] {param_name}.variance)"
        else:
            base = f"Normal({self.mean}, {self.variance})"
        return base + self._bounds_str(self.lower, self.upper)

    def __repr__(self) -> str:
        return f"NormalPrior(mean={self.mean}, variance={self.variance})"



@dataclass
class LogNormalPrior(PriorSpec):
    """LogNormal prior: log(parameter) ~ N(mean, variance)."""
    mean:     float = 0.0
    variance: float | PriorSpec = 0.5
    lower:    float | None = None
    upper:    float | None = None

    def __post_init__(self):
        self._validate_bounds(self.lower, self.upper)

    def free_slots(self) -> list[str]:
        return ["variance"] if isinstance(self.variance, PriorSpec) else []

    def display(self, param_name: str) -> str:
        if isinstance(self.variance, PriorSpec):
            base = f"LogNormal({self.mean}, [estimated~{self.variance.__class__.__name__}] {param_name}.variance)"
        else:
            base = f"LogNormal({self.mean}, {self.variance})"
        return base + self._bounds_str(self.lower, self.upper)

    def __repr__(self) -> str:
        return f"LogNormalPrior(mean={self.mean}, variance={self.variance})"


@dataclass
class BetaPrior(PriorSpec):
    """Beta prior: parameter ~ Beta(alpha, beta).

    Default Beta(5, 17): mean ≈ 0.23, near chance for 4-option MC.
    """
    alpha: float = 5.0
    beta:  float = 17.0
    lower: float | None = None
    upper: float | None = None

    def __post_init__(self):
        self._validate_bounds(self.lower, self.upper)

    def free_slots(self) -> list[str]:
        return []

    def display(self, param_name: str) -> str:
        return f"Beta({self.alpha}, {self.beta}){self._bounds_str(self.lower, self.upper)}"

    def __repr__(self) -> str:
        return f"BetaPrior(alpha={self.alpha}, beta={self.beta})"


@dataclass
class LKJPrior(PriorSpec):
    """LKJ prior on correlation matrices (estimated correlation).

    For R=2: p(ρ | η) ∝ (1 − ρ²)^(η−1); η=1 is uniform on (−1, 1).
    For R>2: prior on the full correlation matrix via vine parameterization.

    The rho parameter(s) are estimated and appear in the chain as
    rho_i_j for i < j.

    Parameters
    ----------
    eta : float > 0.  Concentration parameter.  Default 1.0 (uniform).
    lower, upper : Only meaningful for R=2 (scalar rho).  Ignored for R>2.
    """
    eta:   float = 1.0
    lower: float | None = None
    upper: float | None = None

    def __post_init__(self):
        self._validate_bounds(self.lower, self.upper)
        if self.eta <= 0:
            raise ValueError(f"LKJPrior eta must be > 0; got {self.eta}.")

    def free_slots(self) -> list[str]:
        return ["rho"]

    def display(self, param_name: str) -> str:
        return f"LKJ(eta={self.eta}){self._bounds_str(self.lower, self.upper)}"

    def __repr__(self) -> str:
        return f"LKJPrior(eta={self.eta})"


@dataclass
class FixedCorrelation(PriorSpec):
    """Fixed correlation — rho is not estimated.

    FixedCorrelation(rho=0.0) gives independent traits (the default for
    b.correlation and a.correlation; theta.correlation defaults to LKJ(1)).

    Parameters
    ----------
    rho : float or ndarray
        Scalar for R=2; [R, R] correlation matrix for R>2.

    Validation at construction (raises ValueError if violated):
        Scalar:  ρ ∈ (−1, 1).
        Matrix:  symmetric (tolerance 1e-10), ones on diagonal,
                 positive definite (all eigenvalues > 0).
    """
    rho: float | np.ndarray = 0.0

    def __post_init__(self):
        if isinstance(self.rho, (int, float)):
            rho_f = float(self.rho)
            if rho_f != 0.0 and not (-1.0 < rho_f < 1.0):
                raise ValueError(
                    f"FixedCorrelation scalar rho must be in (−1, 1); got {self.rho}.  "
                    f"Values at the boundary (±1) or outside make the correlation "
                    f"matrix singular or non-positive-definite."
                )
        elif isinstance(self.rho, np.ndarray):
            _validate_corr_matrix(self.rho)

    def free_slots(self) -> list[str]:
        return []

    def display(self, param_name: str) -> str:
        if isinstance(self.rho, (int, float)) and float(self.rho) == 0.0:
            return "Independent (no correlation)"
        if isinstance(self.rho, np.ndarray):
            R = self.rho.shape[0]
            # Show pairwise values compactly
            pairs = [f"ρ_{i+1}{j+1}={self.rho[i,j]:.3f}"
                     for i in range(R) for j in range(i+1, R)]
            return "Fixed  " + "   ".join(pairs)
        return f"Fixed(ρ={self.rho:.3f})"

    def __repr__(self) -> str:
        return f"FixedCorrelation(rho={self.rho})"


@dataclass
class HalfNormalPrior(PriorSpec):
    """HalfNormal prior on the standard deviation: sigma ~ HalfNormal(0, scale).

    Opt-in alternative to the default InvGamma prior on variance components.

    **Why opt-in only.** HalfNormal is more naturally interpretable on the
    standard-deviation scale ("sigma is unlikely to exceed 3") but breaks
    Normal-InvGamma conjugacy, forcing an MH update for that variance
    component instead of the default exact Gibbs draw.

    When used as the variance slot of NormalPrior (or LogNormalPrior), a
    UserWarning is emitted at priors.set() time explaining the MH cost.

    Parameters
    ----------
    scale : float > 0.  The HalfNormal scale parameter.
    lower, upper : Optional bounds (rarely needed for a half-normal; included
        for consistency with the rest of the prior API).

    Usage
    -----
    # Replace InvGamma (Gibbs) with HalfNormal (MH) for theta variance:
    priors.set("theta", NormalPrior(mean=0, variance=HalfNormalPrior(scale=1.0)))
    """
    scale: float = 1.0
    lower: float | None = None
    upper: float | None = None

    def __post_init__(self):
        self._validate_bounds(self.lower, self.upper)
        if self.scale <= 0:
            raise ValueError(f"HalfNormalPrior scale must be > 0; got {self.scale}.")

    def free_slots(self) -> list[str]:
        # HalfNormal: a prior on sigma — not an estimated variance; no free slots.
        # The variance component it governs IS estimated, but that is handled
        # by the sampler dispatching to MH, not by free_slots().
        return []

    def display(self, param_name: str) -> str:
        return f"HalfNormal(scale={self.scale}){self._bounds_str(self.lower, self.upper)}"

    def __repr__(self) -> str:
        return f"HalfNormalPrior(scale={self.scale})"


@dataclass
class TruncNormalPrior(PriorSpec):
    """Truncated Normal prior: parameter ~ TruncNormal(mean, variance, lower, upper).

    For item discrimination *a*: set ``lower=0`` (default when used as the
    prior for ``"a"``).  Unlike :class:`LogNormalPrior`, this prior keeps the
    parameter on its natural scale, which makes the full conditional of *a*
    exactly conjugate (a truncated Normal) given the Albert–Chib augmented
    data.  Specifying this prior unlocks exact Gibbs sampling for *a* — both
    uncorrelated (scalar per item) and cross-node correlated (truncated MVN).

    Can also be used for ``"theta"`` or ``"b"`` as an alternative to
    :class:`NormalPrior` with explicit bounds; the behaviour is identical.

    Parameters
    ----------
    mean : float
        Prior mean μ₀.  Default 1.0 for discrimination (near the centre of
        the positive half-line); 0.0 for all other parameters.
    variance : float or PriorSpec
        Prior variance τ².  May hold any PriorSpec on (0, ∞) for hierarchical
        estimation (InvGamma full conditional, same as for NormalPrior).
    lower : float or None
        Lower truncation bound.  ``None`` means unbounded below.
        Recommended: ``lower=0`` for discrimination parameters.
    upper : float or None
        Upper truncation bound.  ``None`` means unbounded above.

    Examples
    --------
    ::

        # Discrimination with positivity constraint (unlocks Gibbs for a):
        priors.set("a", TruncNormalPrior(mean=1.0, variance=0.5, lower=0))

        # With an estimated variance (InvGamma Gibbs):
        priors.set("a", TruncNormalPrior(mean=1.0,
                                         variance=InvGammaPrior(2, 1),
                                         lower=0))
    """
    mean:     float = 0.0
    variance: float | PriorSpec = 1.0
    lower:    float | None = None
    upper:    float | None = None

    def __post_init__(self):
        self._validate_bounds(self.lower, self.upper)

    def free_slots(self) -> list[str]:
        return ["variance"] if isinstance(self.variance, PriorSpec) else []

    def display(self, param_name: str) -> str:
        if isinstance(self.variance, PriorSpec):
            var_str = f"[estimated] {param_name}.variance"
        else:
            var_str = str(self.variance)
        base = f"TruncNormal({self.mean}, {var_str})"
        return base + self._bounds_str(self.lower, self.upper)

    def __repr__(self) -> str:
        return (f"TruncNormalPrior(mean={self.mean}, variance={self.variance}"
                f"{', lower=' + str(self.lower) if self.lower is not None else ''}"
                f"{', upper=' + str(self.upper) if self.upper is not None else ''})")


@dataclass
class InverseWishartPrior(PriorSpec):
    """Inverse-Wishart prior on a full covariance matrix Σ.

    Conjugate with the multivariate Normal likelihood for item or person
    parameter vectors.  When used for correlation priors
    (``"theta.correlation"``, ``"b.correlation"``, ``"a.correlation"``),
    enables an **exact Gibbs draw** for the full covariance matrix — no
    Metropolis–Hastings, no step-size tuning.

    The trade-off versus the LKJ + DRD separation strategy:

    * **Pro** — Conjugacy → exact Gibbs; often faster mixing.
    * **Con** — Variances and correlations are entangled in the prior.
      It is harder to express "I expect small correlations" without also
      implicitly constraining the variances.

    Full conditional (conjugate update)::

        Σ | data  ~  InvWishart(Ψ₀ + ScatterMatrix,  ν₀ + N)

    where N is the number of observations (persons or items) and
    ScatterMatrix = Σ_i (x_i − μ₀)(x_i − μ₀)ᵀ.

    Parameters
    ----------
    scale : ndarray [G, G] or None
        Scale matrix Ψ₀ (symmetric positive definite).  ``None`` uses the
        default ``0.01 * I_G`` (G is the correlation-group size, known at
        first use).  In the conjugate update Ψ₀ adds to the data's scatter
        matrix: with ν₀ = G + 1 and N rows, E[Σ | data] = (Ψ₀ + S) / N.  A
        small Ψ₀ therefore adds almost nothing, while Ψ₀ = I would add 1/N to
        every variance and pull correlations toward 0, which matters for
        groups with few items or small variances (e.g. discriminations):
        with N = 10 and true variances 0.1, Ψ₀ = I roughly doubled the
        variances and moved a 0.88 sample correlation to 0.43.  Pass a scale
        explicitly to encode real prior information about Σ.
    df : int or None
        Degrees of freedom ν₀.  Must be ≥ G + 1.  ``None`` uses the
        minimum valid value G + 1 (vague / diffuse prior).

    Class method
    ------------
    InverseWishartPrior.vague(G) — construct the vague (minimally
        informative) prior for dimension G.

    Examples
    --------
    ::

        from irtree.priors import InverseWishartPrior
        import numpy as np

        # Vague prior for a 2-group b-correlation:
        priors.set("b.correlation", InverseWishartPrior())

        # Informative prior for theta.correlation (2 latent dims):
        priors.set("theta.correlation",
                   InverseWishartPrior(scale=np.eye(2) * 0.5, df=4))
    """
    scale: np.ndarray | None = None
    df:    int | None        = None

    def __post_init__(self):
        if self.scale is not None:
            sc = np.asarray(self.scale)
            if sc.ndim != 2 or sc.shape[0] != sc.shape[1]:
                raise ValueError(
                    f"InverseWishartPrior scale must be a square 2-D array; "
                    f"got shape {sc.shape}."
                )
            eigvals = np.linalg.eigvalsh(sc)
            if np.any(eigvals <= 0):
                raise ValueError(
                    f"InverseWishartPrior scale matrix must be positive "
                    f"definite; non-positive eigenvalues: {eigvals[eigvals <= 0]}."
                )
            object.__setattr__(self, "scale", sc)
        if self.df is not None and self.df < 2:
            raise ValueError(
                f"InverseWishartPrior df must be >= G+1 where G >= 1, so "
                f"df must be at least 2; got {self.df}."
            )

    def free_slots(self) -> list[str]:
        # The covariance matrix is always estimated when InvWishart is used.
        return ["Sigma"]

    def display(self, param_name: str) -> str:
        # df: shown numerically when the user set it; symbolic "G+1" only in
        # the vague (df=None) case, because the concrete value depends on the
        # dimension G of the context this prior is applied to, which is not a
        # field of the prior (the same prior object may be attached to
        # contexts with different G, e.g. theta.correlation vs b.correlation).
        df_str = "G+1 (vague)" if self.df is None else str(self.df)
        # scale: the matrix, when user-supplied, is fully known here (no
        # context needed), so print its actual values rather than a generic
        # "user-supplied". Small matrices show entries; larger ones show shape
        # and diagonal to stay readable.
        if self.scale is None:
            scale_str = "0.01·I (vague)"
        else:
            sc = np.asarray(self.scale)
            if sc.shape[0] <= 3:
                scale_str = np.array2string(
                    sc, precision=3, suppress_small=True,
                    separator=", ",
                ).replace("\n", " ")
            else:
                diag = np.round(np.diag(sc), 3).tolist()
                scale_str = f"{sc.shape[0]}×{sc.shape[0]} (diag={diag})"
        return f"InvWishart(df={df_str}, scale={scale_str})"

    def __repr__(self) -> str:
        return f"InverseWishartPrior(df={self.df}, scale={'None' if self.scale is None else 'array'})"

    @classmethod
    def vague(cls, G: int) -> InverseWishartPrior:
        """Construct the minimally informative InvWishart prior for dimension G."""
        return cls(scale=np.eye(G) * 0.01, df=G + 1)

    def resolved_params(self, G: int):
        """Return (scale, df) with vague defaults filled in for dimension G."""
        scale = self.scale if self.scale is not None else np.eye(G) * 0.01
        df    = self.df    if self.df    is not None else G + 1
        if df < G + 1:
            raise ValueError(
                f"InverseWishartPrior df must be >= G+1={G+1}; got {df}."
            )
        return np.asarray(scale, dtype=np.float64), int(df)


def _validate_corr_matrix(C: np.ndarray) -> None:
    """Validate that C is a valid correlation matrix. Raises ValueError."""
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError(
            f"FixedCorrelation matrix must be square 2D; got shape {C.shape}."
        )
    # Check ones on diagonal
    diag_errs = np.abs(np.diag(C) - 1.0)
    if np.any(diag_errs > 1e-10):
        bad = np.where(diag_errs > 1e-10)[0].tolist()
        raise ValueError(
            f"FixedCorrelation matrix must have 1.0 on the diagonal.  "
            f"Diagonal entries at indices {bad} differ from 1: "
            f"{[C[i,i] for i in bad]}."
        )
    # Check symmetry
    asym = np.max(np.abs(C - C.T))
    if asym > 1e-10:
        raise ValueError(
            f"FixedCorrelation matrix must be symmetric.  "
            f"Maximum asymmetry: {asym:.2e}."
        )
    # Check positive definiteness
    eigvals = np.linalg.eigvalsh(C)
    if np.any(eigvals <= 0):
        bad_eig = eigvals[eigvals <= 0].tolist()
        raise ValueError(
            f"FixedCorrelation matrix must be positive definite (all eigenvalues > 0).  "
            f"Non-positive eigenvalues found: {bad_eig}.  "
            f"Check that all off-diagonal entries satisfy |ρ_ij| < 1 and that "
            f"the matrix is a valid correlation structure."
        )


# ===========================================================================
# Correlation chain name helper
# ===========================================================================

def _corr_chain_names(R: int) -> list[str]:
    """Return rho_i_j names for i < j, 1-based, for an R-node tree."""
    return [f"rho_{i}_{j}" for i in range(1, R + 1) for j in range(i + 1, R + 1)]


# ===========================================================================
# Package defaults
# ===========================================================================

def _make_defaults() -> dict[str, PriorSpec]:
    return {
        "theta":             NormalPrior(mean=0.0, variance=InvGammaPrior()),
        "b":                 NormalPrior(mean=0.0, variance=InvGammaPrior()),
        "a":                 TruncNormalPrior(mean=1.0, variance=0.5, lower=0.0),
        "c":                 BetaPrior(alpha=5.0, beta=17.0),
        # Traits are correlated by default: the correlations are scale-free,
        # so they stay identified when sigma2_d = 1 fixes the scale of a
        # dimension with free slopes.  LKJ (not InverseWishart) keeps that
        # constraint, since it samples R alone.  FixedCorrelation(rho=0.0)
        # gives independent traits.
        "theta.correlation": LKJPrior(eta=1.0),
        "b.correlation":     FixedCorrelation(rho=0.0),
        "a.correlation":     FixedCorrelation(rho=0.0),
        # Covariate coefficient priors — only active at nodes with the
        # corresponding covariate flag set (has_person_covariates, etc.).
        "beta":  NormalPrior(mean=0.0, variance=InvGammaPrior()),   # person covariates
        "kappa": NormalPrior(mean=0.0, variance=InvGammaPrior()),   # item covariates
        "gamma": NormalPrior(mean=0.0, variance=InvGammaPrior()),   # person×item covariates
        # Missing covariate values — sampled as parameters each iteration.
        # All three use Normal(0,1) as the default (assumes centered,
        # standardized covariates).  Users may override with InvGammaPrior()
        # (Gibbs) or HalfNormalPrior() (MH) to estimate the missing-value variance.
        "X_missing": NormalPrior(mean=0.0, variance=1.0),
        "Q_missing": NormalPrior(mean=0.0, variance=1.0),
        "W_missing": NormalPrior(mean=0.0, variance=1.0),
    }

# Gibbs parameters — bounds trigger truncated Normal and UserWarning
_GIBBS_PARAMS = {"theta", "b", "beta", "kappa", "gamma"}


# ===========================================================================
# Priors container
# ===========================================================================

@dataclass
class Priors:
    """Container for the full prior specification of an IRTree model.

    Stores priors at two levels (type-level and node-level) and resolves them
    in the order: node_level → type_level → package default.

    Construction
    ------------
    Always use the classmethod, not the dataclass constructor directly::

        priors = Priors.from_spec(spec, node_models)

    Modification
    ------------
    Use ``priors.set()`` to override any prior at the type or node level::

        priors.set("b", NormalPrior(mean=0, variance=2.0))          # all nodes
        priors.set("b", NormalPrior(mean=0, variance=4.0), node=2)  # node 2 only
        priors.set("a", TruncNormalPrior(mean=1.0, variance=0.5, lower=0))
        priors.set("theta.correlation", LKJPrior(eta=2.0))
        priors.set("b.correlation",     InverseWishartPrior())
        priors.set("a.correlation",     NormalPrior(mean=0, variance=0.3, lower=-1, upper=1))

    Estimated variance
    ------------------
    Pass any ``PriorSpec`` defined on (0, ∞) as a variance value to trigger
    hierarchical estimation of that variance::

        NormalPrior(mean=0, variance=InvGammaPrior())      # Gibbs (conjugate)
        NormalPrior(mean=0, variance=HalfNormalPrior(1.0)) # MH

    See module docstring for the full list of parameter keys, correlation
    specification options, and chain column naming conventions.
    """
    type_level:    dict[str, PriorSpec]             = field(default_factory=dict)
    node_level:    dict[tuple[str, int], PriorSpec] = field(default_factory=dict)
    _param_nodes:  dict[str, list[int]]             = field(default_factory=dict, repr=False)
    _n_nodes:      int                              = field(default=0, repr=False)
    _n_latent_dims: int                             = field(default=0, repr=False)
    _spec:         object                           = field(default=None, repr=False)
    _node_models:  list                             = field(default_factory=list, repr=False)
    _mixed_nodes:  list[int]                        = field(default_factory=list, repr=False)
    #: (param, node) keys of node-level priors set automatically by
    #: build_default_priors (theta variance fixed at 1 where slopes are free).
    #: Any user set() for the same parameter replaces them.
    _auto_node_level: set                           = field(default_factory=set, repr=False)

    # ------------------------------------------------------------------
    # Primary constructor
    # ------------------------------------------------------------------

    @classmethod
    def from_spec(cls, spec, node_models: list) -> Priors:
        """Build a Priors object by inspecting the tree and node models.

        Covariate nodes are detected via the ``has_person_covariates``,
        ``has_item_covariates``, and ``has_pi_covariates`` flags on each
        ``NodeModel``.  Nodes with these flags set are registered under the
        ``"beta"``, ``"kappa"``, and ``"gamma"`` parameter keys respectively.
        Default priors ``NormalPrior(0, InvGammaPrior())`` are applied to all
        three; the variance is estimated hierarchically by default.  The
        ``"X_missing"``, ``"Q_missing"``, ``"W_missing"`` priors (for missing covariate
        values treated as
        parameters) is set to ``NormalPrior(0, 1.0)`` by default and is
        applied globally (not per-node).
        """
        R = spec.n_nodes
        D = spec.n_latent_dims
        priors = cls(_n_nodes=R, _n_latent_dims=D, _spec=spec,
                     _node_models=list(node_models))
        param_nodes: dict[str, list[int]] = {
            "theta": list(range(R)),
            "b":     list(range(R)),
            "a":     [],
            "c":     [],
            # Covariate coefficient parameters — populated below
            "beta":  [],
            "kappa": [],
            "gamma": [],
        }
        mixed_nodes = []
        for r, nm in enumerate(node_models):
            if nm.has_discrimination:
                param_nodes["a"].append(r)
            if nm.has_guessing:
                param_nodes["c"].append(r)
            if getattr(nm, "is_mixed", False):
                mixed_nodes.append(r)
            # Register covariate nodes
            if getattr(nm, "has_person_covariates", False):
                param_nodes["beta"].append(r)
            if getattr(nm, "has_item_covariates", False):
                param_nodes["kappa"].append(r)
            if getattr(nm, "has_pi_covariates", False):
                param_nodes["gamma"].append(r)
        # b.binary and b.poly: sub-group priors for mixed-format nodes.
        # Mixed nodes have both binary items (Normal prior b.binary; Gibbs
        # given Z at GRM nodes) and polytomous items (Normal prior b.poly,
        # ordered; Cowles step + exact truncated-prior Gibbs at GRM nodes,
        # item-wise MH at GPCM nodes).  Both priors enter the updates.
        param_nodes["b.binary"] = list(mixed_nodes)
        param_nodes["b.poly"]   = list(mixed_nodes)
        priors._param_nodes = param_nodes
        priors._mixed_nodes  = mixed_nodes
        defaults = _make_defaults()
        for param in ("theta", "b", "a", "c", "theta.correlation",
                      "b.correlation", "a.correlation",
                      "beta", "kappa", "gamma",
                      "X_missing", "Q_missing", "W_missing"):
            priors.type_level[param] = defaults[param]

        # Identification.  On a dimension whose nodes all have free
        # discriminations, the likelihood depends on a_i theta_p only through
        # their product, so an estimated theta variance and the slopes trade
        # off along a_i -> c a_i, theta -> theta / c, identified only by the
        # a prior.  Fix the theta variance at 1 on those dimensions (the
        # usual 2PL convention).  A node without slopes (Rasch, GRM, PCM)
        # already fixes the scale through a = 1, so a dimension with one
        # keeps its variance estimated: fixing it as well would add a second,
        # substantive restriction.  Set at node level, flagged automatic; a
        # user priors.set("theta", ...) replaces it.
        dim_to_nodes = (spec.dim_to_nodes if getattr(spec, "dim_to_nodes", None) is not None
                        else [[r] for r in range(R)])
        for nodes_d in dim_to_nodes:
            if all(getattr(node_models[r], "has_discrimination", False) for r in nodes_d):
                for r in nodes_d:
                    priors.node_level[("theta", r)] = NormalPrior(mean=0.0, variance=1.0)
                    priors._auto_node_level.add(("theta", r))
        return priors

    # ------------------------------------------------------------------
    # set()
    # ------------------------------------------------------------------

    def set(
        self,
        param:      str,
        prior_spec: PriorSpec,
        node:       int | None = None,
        force:      bool = False,
    ) -> None:
        """Set a prior at type-level (node=None) or node-level (node=r).

        A user prior replaces any automatic default for the same parameter:
        a type-level set() removes the automatic node-level entries of that
        parameter (e.g. the identification default theta ~ N(0, 1) on
        dimensions with free discriminations), so it applies everywhere.
        ``node`` is 1-based.  See ``_set_impl`` for the validation rules.
        """
        self._set_impl(param, prior_spec, node=node, force=force)
        if node is None:
            for key in [k for k in self._auto_node_level if k[0] == param]:
                self.node_level.pop(key, None)
                self._auto_node_level.discard(key)
        else:
            self._auto_node_level.discard((param, node - 1))

    def _set_impl(
        self,
        param:      str,
        prior_spec: PriorSpec,
        node:       int | None = None,
        force:      bool = False,
    ) -> None:
        """Set a prior at type-level (node=None) or node-level (node=r).

        Raises
        ------
        TypeError
            If ``prior_spec`` is not a PriorSpec; if a bare ``InvGammaPrior``
            (or similar variance-slot distribution) is passed at top level; or
            if ``node`` is passed together with a ``.correlation`` key
            (correlation is a group-level prior with no node-level meaning).
            
        NotImplementedError
            non-LKJ/FixedCorrelation priors for theta.correlation when R > 2.
        ValueError
            node out of range; parameter does not exist at that node.
        NotImplementedError
            Bounds on a location prior; an estimated variance the sampler
            has no update for (see ``_check_sampler_support``).
            Non-LKJ prior on theta.correlation without (-1, 1) bounds (R=2).
        """
        # Universal precondition: whatever the parameter, the value must be a
        # PriorSpec. This is param-independent — "is this even a prior?" is not
        # specific to any key — so it is checked once, here, rather than in the
        # per-param validators; _validate_corr_prior therefore needs no
        # isinstance(prior_spec, PriorSpec) check of its own.
        if not isinstance(prior_spec, PriorSpec):
            raise TypeError(
                f"prior for '{param}' must be a PriorSpec instance "
                f"(e.g. NormalPrior, BetaPrior, LKJPrior, InverseWishartPrior, "
                f"FixedCorrelation, …); got {type(prior_spec).__name__}."
            )

        # Accept a bare PriorSpec in a variance slot as shorthand for
        # "estimate this variance using the given prior distribution".
        # Guard: catch distributions that belong in a variance slot, not at
        # the top level.  InvGammaPrior and HalfNormalPrior are defined on
        # (0, ∞) and have no meaningful interpretation as a prior for theta,
        # b, a, etc. directly.  The user almost certainly meant to put them
        # inside a NormalPrior / TruncNormalPrior variance slot.
        if isinstance(prior_spec, (InvGammaPrior, HalfNormalPrior)) \
                and not param.endswith(".correlation") \
                and _base_param(param) != "a" \
                and not force:
            _valid_keys = frozenset((
                "theta", "b", "b.binary", "b.poly", "a", "c",
                "beta", "kappa", "gamma",
                "X_missing", "Q_missing", "W_missing",
                "theta.correlation", "b.correlation", "a.correlation",
            ))
            if param not in _valid_keys:
                raise TypeError(
                    f"Unknown parameter key '{param}'.  "
                    f"Valid keys are: {', '.join(sorted(_valid_keys))}.  "
                    f"Check for typos (e.g. 'theta' not 'Theta', 'b' not 'B')."
                )
            raise TypeError(
                f"'{prior_spec.__class__.__name__}' is defined on (0, ∞) and is "
                f"unlikely to be a valid direct prior for '{param}'.  "
                f"It is typically used as a prior on a variance parameter, e.g.:\n"
                f"  priors.set({param!r}, "
                f"NormalPrior(mean=0.0, variance={prior_spec!r}))\n"
                f"If you intentionally want '{prior_spec.__class__.__name__}' as the "
                f"prior for '{param}', rerun with force=True:\n"
                f"  priors.set({param!r}, {prior_spec!r}, force=True)"
            )

        # Guard: catch misspelled or unrecognised parameter keys.
        _valid_keys = frozenset((
            "theta", "b", "b.binary", "b.poly", "a", "c",
            "beta", "kappa", "gamma",
            "X_missing", "Q_missing", "W_missing",
            "theta.correlation", "b.correlation", "a.correlation",
        ))
        if param not in _valid_keys:
            raise TypeError(
                f"Unknown parameter key '{param}'.  "
                f"Valid keys are: {', '.join(sorted(_valid_keys))}.  "
                f"Check for typos (e.g. 'theta' not 'Theta', 'b' not 'B')."
            )

        if param in ("theta.correlation", "b.correlation", "a.correlation"):
            # Correlation is a property of the whole correlated group (across
            # latent dimensions / item-parameter groups), not of any single
            # node — "the correlation at node r" is not a well-defined object.
            # A node= argument here is therefore a category error.  It is
            # rejected rather than ignored, which would leave the user
            # believing they set a node-specific prior.
            if node is not None:
                raise TypeError(
                    f"'{param}' is a group-level prior and does not accept a "
                    f"node= argument (got node={node}). Correlation is defined "
                    f"across the whole correlated set of dimensions/groups, not "
                    f"per node. Set it at type level:\n"
                    f"  priors.set({param!r}, {prior_spec.__class__.__name__}(...))"
                )
            G = self._n_latent_dims if param == "theta.correlation" else None
            self._validate_corr_prior(param, prior_spec, G=G)
            self.type_level[param] = prior_spec
            return

        # ---- support-domain check (all non-correlation keys) ------------
        # Reject priors whose support is incoherent with the parameter's
        # domain: a>0, c in [0,1], location params (theta/b/beta/kappa/gamma,
        # AND the b.binary/b.poly threshold sub-keys, which are difficulty-
        # scale locations on R) must not be one-sided. Placed BEFORE the
        # b.binary/b.poly branch so those keys are covered too — that branch
        # returns early, so a check after it would silently skip them.
        # An incoherent prior often runs without crashing and silently fails
        # to recover, so catch it here. force=True downgrades to a warning for
        # models that deliberately constrain a parameter.
        _check_support_domain(param, prior_spec, force=force)

        # A hierarchical variance is only Gibbs-drawable when the coefficient
        # prior is an unbounded Normal; bounds destroy the conjugacy. Checked
        # here, alongside the support check, so both fire before any
        # early-returning branch below.
        _check_variance_conjugacy(param, prior_spec, force=force)

        # Reject priors the sampler cannot honour (checked last, so the more
        # specific support / conjugacy messages above fire first).
        _check_sampler_support(param, prior_spec)

        # ---- b.binary / b.poly sub-group priors for mixed-format nodes ---
        # b.binary: Normal prior on scalar thresholds — enters the Gibbs full
        #   conditional exactly as the prior on "b" does for binary nodes.
        # b.poly: Normal prior on ordered polytomous thresholds — enters the
        #   gibbs_thresholds_grm full conditional (truncated Normal draw).
        #   When tau2 → ∞ this recovers the flat-prior uniform draw of
        #   Johnson-Albert (1999).
        if param in ("b.binary", "b.poly"):
            if node is not None:
                node_0 = node - 1
                if node_0 < 0 or node_0 >= self._n_nodes:
                    raise ValueError(
                        f"node={node} is out of range for a {self._n_nodes}-node "
                        f"tree (valid: 1–{self._n_nodes})."
                    )
                _warn_halfnormal_variance("b", prior_spec)
                self.node_level[(param, node_0)] = prior_spec
            else:
                _warn_halfnormal_variance("b", prior_spec)
                self.type_level[param] = prior_spec
            return

        # ---- standard parameter handling --------------------------------
        if node is not None:
            node_0 = node - 1
            if node_0 < 0 or node_0 >= self._n_nodes:
                raise ValueError(
                    f"node={node} is out of range for a {self._n_nodes}-node "
                    f"tree (valid: 1–{self._n_nodes})."
                )
            base = _base_param(param)
            if base in ("a", "c"):
                valid = self._param_nodes.get(base, [])
                if node_0 not in valid:
                    raise ValueError(
                        f"Parameter '{base}' does not exist at node {node}.  "
                        f"It exists at nodes {[r + 1 for r in valid]}."
                    )
            # HalfNormal variance warning (Gibbs parameters)
            if base in _GIBBS_PARAMS:
                _warn_halfnormal_variance(param, prior_spec)
            self.node_level[(param, node_0)] = prior_spec
        else:
            base = _base_param(param)
            if base in _GIBBS_PARAMS:
                _warn_halfnormal_variance(param, prior_spec)
            self.type_level[param] = prior_spec

    def _validate_corr_prior(
        self,
        param:      str,
        prior_spec: PriorSpec,
        G:          int | None = None,
    ) -> None:
        """Validate a correlation prior for any of the three correlation keys.

        Parameters
        ----------
        param      : full key — "theta.correlation", "b.correlation", or "a.correlation"
        prior_spec : the prior being validated
        G          : number of correlated dimensions / groups.
                     For theta.correlation: _n_latent_dims (known at set()
                     time; the theta correlation is over latent dimensions,
                     so Σ_theta is D×D, not R×R).
                     For b.correlation / a.correlation: the group size from
                     Constraints, known only at run_chain() time.  Pass None
                     at set() time; call again with the real G at run_chain() time.

        Rules (identical for all three keys):
            InverseWishartPrior  — valid for any G; scale matrix validated if given.
            LKJPrior             — valid for any G.
            FixedCorrelation     — valid for any G.
            Bounded scalar prior — valid only when G == 2 (Fisher-z MH).
                                   NotImplementedError when G > 2.
                                   UserWarning when bounds missing or outside (−1, 1).
            Anything else        — TypeError.
        """
        is_lkj   = isinstance(prior_spec, LKJPrior)
        is_fixed = isinstance(prior_spec, FixedCorrelation)
        is_iw    = isinstance(prior_spec, InverseWishartPrior)

        if is_iw:
            # Context-free checks (shape, PD, df >= 2) run in __post_init__.
            # Only the G-dependent df >= G+1 check needs context.
            if G is not None and prior_spec.df is not None and prior_spec.df < G + 1:
                raise ValueError(
                    f"InverseWishartPrior df must be >= G+1 = {G+1} for this "
                    f"correlation group of size G={G}; got df={prior_spec.df}."
                )
            return

        if is_lkj or is_fixed:
            return

        # Scalar-bounded prior (NormalPrior, BetaPrior, …) — Fisher-z MH path.
        # The "is it a PriorSpec at all?" check is not here: it is a universal
        # precondition enforced once in set() (and every internal caller passes
        # a resolved prior, which is always a PriorSpec). The assertion below is
        # a cheap internal guard against a future direct misuse, not a
        # user-facing validation.
        assert isinstance(prior_spec, PriorSpec), (
            f"_validate_corr_prior received a non-PriorSpec "
            f"({type(prior_spec).__name__}); callers must pass a resolved "
            f"PriorSpec (set() enforces this for user input)."
        )
        if G is not None and G > 2:
            raise NotImplementedError(
                f"Bounded scalar priors on '{param}' are only supported when "
                f"G == 2 (Fisher-z MH path).  For G > 2 use LKJPrior(eta) or "
                f"InverseWishartPrior.  Got: {type(prior_spec).__name__} with G={G}."
            )
        # G == 2 or G unknown — warn if bounds missing or outside (−1, 1)
        lo = getattr(prior_spec, "lower", None)
        hi = getattr(prior_spec, "upper", None)
        if lo is None or hi is None or lo < -1.0 or hi > 1.0:
            warnings.warn(
                f"The prior on '{param}' has support outside (−1, 1).  "
                f"Any draw outside this range makes the correlation matrix "
                f"non-positive-definite and will be rejected by the MH step.  "
                f"Consider adding lower=-1, upper=1 bounds.",
                UserWarning,
                stacklevel=3,
            )


    # ------------------------------------------------------------------
    # resolve()
    # ------------------------------------------------------------------

    def resolve(self, param: str, node: int | None) -> PriorSpec:
        """Return the active PriorSpec for (param, node).

        For per-node parameters (theta, b, a, c, beta, kappa, gamma,
        b.binary, b.poly) node is the 0-based node index and the
        resolution order is:
            node_level[(param, node)] →
            node_level[("b", node)]   (b.binary/b.poly only) →
            type_level[param]         →
            type_level["b"]           (b.binary/b.poly only) →
            package default

        For global parameters (theta.correlation, b.correlation,
        a.correlation, X_missing, Q_missing, W_missing) the node
        argument is ignored — pass None to make this explicit.
        These always resolve from type_level only.
        """
        # Mixed-node sub-group params use a two-key fallback chain
        if param in ("b.binary", "b.poly"):
            if (param, node) in self.node_level:
                return self.node_level[(param, node)]
            if ("b", node) in self.node_level:
                return self.node_level[("b", node)]
            if param in self.type_level:
                return self.type_level[param]
            if "b" in self.type_level:
                return self.type_level["b"]
            return _make_defaults()["b"]
        # Missing-value priors: global (not per-node), one per covariate type
        if param in ("X_missing", "Q_missing", "W_missing"):
            if param in self.type_level:
                return self.type_level[param]
            return _make_defaults()[param]
        # Standard three-level lookup (covers beta, kappa, gamma too)
        if (param, node) in self.node_level:
            return self.node_level[(param, node)]
        if param in self.type_level:
            return self.type_level[param]
        defaults = _make_defaults()
        if param in defaults:
            return defaults[param]
        raise KeyError(f"No prior found for '{param}' at node {node + 1}.")

    # ------------------------------------------------------------------
    # free_params()
    # ------------------------------------------------------------------

    def free_params(self, constraints=None) -> list[str]:
        """Chain column names of every estimated hyperparameter.

        The names are the ones the chain will carry (``sigma2_1``,
        ``b.variance_1&3``, ``a.variance_2``, ``rho_1_2``, ``b.rho_1_2``,
        ``beta.variance_1``, ...), so they can be looked up directly in a
        fit's results.  Sharing and correlation groups come from
        ``constraints`` (the same object later passed to ``irtree_fit``);
        without it every node is its own group.
        """
        from irtree.mcmc_sampler import hyperparameter_column_names
        return hyperparameter_column_names(self._spec, self._node_models, self, constraints)

    def summary(self, constraints=None) -> None:
        """Print TABLE 1, TABLE 2 (if any estimated variances), and Correlation section.

        ``constraints`` (optional) sets the sharing groups used for the chain
        names listed under TABLE 2.
        """
        R    = self._n_nodes
        col_w = max(32, 88 // max(R, 1))
        rule1 = "─" * (14 + (col_w + 2) * R)

        # ----------------------------------------------------------------
        # TABLE 1 — Main parameters
        # ----------------------------------------------------------------
        print()
        print("TABLE 1 — Main parameters")
        header = f"{'Parameter':<14}"
        for r in range(R):
            header += f"  {'Node ' + str(r + 1):<{col_w}}"
        print(rule1)
        print(header)
        print(rule1)

        for base_param in ("theta", "b", "a", "c"):
            nodes_for = self._param_nodes.get(base_param, [])
            row = f"{base_param:<14}"
            for r in range(R):
                if r not in nodes_for:
                    row += f"  {'—':<{col_w}}"
                else:
                    spec, level = self._resolve_with_level(base_param, r)
                    cell = spec.display(base_param) + f" [{level}]"
                    row += f"  {cell:<{col_w}}"
            print(row)
            print(rule1)

        print("Level key:  [1] package default  [2] type-level  [3] node-level")
        auto_theta = sorted(r + 1 for (p_, r) in self._auto_node_level if p_ == "theta")
        if auto_theta:
            print(f"Automatic: theta variance fixed at 1 at nodes {auto_theta} "
                  f"(their dimension has free discriminations; identification). "
                  f"priors.set('theta', ...) overrides it.")

        # ----------------------------------------------------------------
        # TABLE 2 — Estimated variance parameters (if any exist)
        # ----------------------------------------------------------------
        # Collect estimated-variance rows (exclude correlation)
        variance_free_exists = any(
            "variance" in name or name.startswith("sigma2_")
            for name in self.free_params(constraints)
        )

        if variance_free_exists:
            rule2 = "─" * (20 + (col_w + 2) * R + 14)
            print()
            print("TABLE 2 — Estimated variance parameters  (free parameters in chain)")
            header2 = f"{'Parameter':<20}"
            for r in range(R):
                header2 += f"  {'Node ' + str(r + 1):<{col_w}}"
            header2 += f"  {'Same prior':<10}"
            print(rule2)
            print(header2)
            print(rule2)

            mixed_set = set(getattr(self, "_mixed_nodes", []))

            for base_param in ("theta", "b", "a", "c"):
                raw_nodes_for = self._param_nodes.get(base_param, [])
                # For b, only show non-mixed nodes here; mixed nodes get
                # separate b.binary.variance / b.poly.variance rows below.
                if base_param == "b":
                    nodes_for = [r for r in raw_nodes_for if r not in mixed_set]
                else:
                    nodes_for = raw_nodes_for
                if not nodes_for:
                    continue
                cells = {}
                for r in nodes_for:
                    spec = self.resolve(base_param, r)
                    if "variance" in spec.free_slots():
                        hyper_dist = spec.variance
                        level = self._resolve_level(base_param, r)
                        cells[r] = hyper_dist.display(f"{base_param}.variance") + f" [{level}]"
                if not cells:
                    continue
                all_free = (set(cells.keys()) == set(nodes_for))
                no_overrides = not any((base_param, r) in self.node_level for r in cells)
                shared = all_free and no_overrides
                row = f"{base_param + '.variance':<20}"
                for r in range(R):
                    row += f"  {cells.get(r, '—'):<{col_w}}"
                row += f"  {'Yes' if shared else 'No':<10}"
                print(row)
                print(rule2)

            # Mixed-node b sub-group variance rows
            for subparam, display_name, note in (
                ("b.binary", "b.binary.variance",
                 ""),
                ("b.poly", "b.poly.variance  ",
                 "  [ordered thresholds; enters the Cowles and Gibbs steps]"),
            ):
                cells_mx = {}
                for r in sorted(mixed_set):
                    spec = self.resolve(subparam, r)
                    if "variance" in spec.free_slots():
                        hyper_dist = spec.variance
                        level = self._resolve_level(subparam, r)
                        cells_mx[r] = hyper_dist.display(f"{subparam}.variance") + f" [{level}]"
                if cells_mx:
                    row = f"{display_name.strip():<20}"
                    for r in range(R):
                        row += f"  {cells_mx.get(r, '—'):<{col_w}}"
                    row += f"  {'N/A':<10}"
                    print(row)
                    if note:
                        print(f"  Note:{note}")
                    print(rule2)

            # Chain columns of the estimated variances (groups follow the
            # constraints passed in; without them every node is its own group)
            var_names = [n for n in self.free_params(constraints)
                         if "variance" in n or n.startswith("sigma2_")]
            if var_names:
                print("Same prior: one type-level prior at every node. Whether the "
                      "variance itself is shared follows Constraints.")
                print(f"Chain columns: {', '.join(var_names)}")

        # ----------------------------------------------------------------
        # Covariate section — printed only when at least one covariate node exists
        # ----------------------------------------------------------------
        cov_nodes_exist = any(
            len(self._param_nodes.get(p, [])) > 0
            for p in ("beta", "kappa", "gamma")
        )
        if cov_nodes_exist:
            rule_cov = "─" * (14 + (col_w + 2) * R)
            print()
            print("Covariate parameters")
            header_cov = f"{'Parameter':<14}"
            for r in range(R):
                header_cov += f"  {'Node ' + str(r + 1):<{col_w}}"
            print(rule_cov)
            print(header_cov)
            print(rule_cov)

            for cov_param, display_name in (
                ("beta",  "beta  [X]"),
                ("kappa", "kappa [Q]"),
                ("gamma", "gamma [W]"),
            ):
                cov_nodes = self._param_nodes.get(cov_param, [])
                if not cov_nodes:
                    continue
                row = f"{display_name:<14}"
                for r in range(R):
                    if r not in cov_nodes:
                        row += f"  {'—':<{col_w}}"
                    else:
                        spec, level = self._resolve_with_level(cov_param, r)
                        cell = spec.display(cov_param) + f" [{level}]"
                        row += f"  {cell:<{col_w}}"
                print(row)
                print(rule_cov)

            print()
            print("Missing covariate prior")
            print(rule_cov)
            for _miss_key, _miss_label in [
                ("X_missing", "missing person covariate values (X)"),
                ("Q_missing", "missing item covariate values (Q)"),
                ("W_missing", "missing person×item covariate values (W)"),
            ]:
                _spec, _lvl = self._resolve_with_level(_miss_key, 0)
                print(f"  {_miss_key:<14}  {_spec.display(_miss_key)} [{_lvl}]  "
                      f"[{_miss_label} sampled as parameters]")
            print(rule_cov)
            print("Level key:  [1] package default  [2] type-level  [3] node-level")

        # ----------------------------------------------------------------
        # Correlation structure — always printed
        # ----------------------------------------------------------------
        rule_c = "─" * 70
        print()
        print("Correlation structure")
        print(rule_c)

        corr_spec = self.resolve("theta.correlation", None)
        # theta's correlation is D x D over latent DIMENSIONS, not R x R over
        # nodes (they differ when nodes share a dimension, e.g. ers_tree).
        D = self._n_latent_dims or R
        corr_names = _corr_chain_names(D)

        if isinstance(corr_spec, FixedCorrelation):
            print(f"  theta.correlation   {corr_spec.display('theta.correlation')}")

        elif isinstance(corr_spec, InverseWishartPrior):
            _, level = self._resolve_with_level("theta.correlation", 0)
            estimated_tag = f"[estimated, full Σ, D={D}]"
            print(f"  theta.correlation   {corr_spec.display('theta.correlation')} [{level}]   {estimated_tag}")
            sigma_names = [f"Sigma_theta_{i+1}_{j+1}" for i in range(D) for j in range(i, D)]
            print(f"  Chain names:        {', '.join(sigma_names)}")

        elif isinstance(corr_spec, LKJPrior):
            _, level = self._resolve_with_level("theta.correlation", 0)
            estimated_tag = f"[estimated, D={D}]"
            print(f"  theta.correlation   {corr_spec.display('theta.correlation')} [{level}]   {estimated_tag}")
            print(f"  Chain names:        {', '.join(corr_names)}")

        else:
            # Bounded scalar prior (R=2 only)
            _, level = self._resolve_with_level("theta.correlation", 0)
            print(f"  theta.correlation   {corr_spec.display('theta.correlation')} [{level}]   [estimated, D=2]")
            print(f"  Chain names:        {corr_names[0]}")

        # b.correlation and a.correlation
        _emitted_item_corr = False
        for item_corr_key, item_label in (
            ("b.correlation", "b.correlation"),
            ("a.correlation", "a.correlation"),
        ):
            item_spec = self.resolve(item_corr_key, None)
            if isinstance(item_spec, FixedCorrelation) and float(item_spec.rho) == 0.0:
                continue   # default — skip
            _, level = self._resolve_with_level(item_corr_key, None)
            if isinstance(item_spec, FixedCorrelation):
                estimated_tag = "[fixed]"
            elif isinstance(item_spec, InverseWishartPrior):
                estimated_tag = (
                    "[estimated — Gibbs; chain columns Sigma_[key]_gN_i_j "
                    "resolved at fit time]"
                )
            elif isinstance(item_spec, LKJPrior):
                estimated_tag = (
                    "[estimated — MH; chain columns [key].rho_gN_i_j "
                    "resolved at fit time]"
                )
            else:
                estimated_tag = "[estimated — Fisher-z MH (G=2); resolved at fit time]"
            print(
                f"  {item_label:<20}{item_spec.display(item_corr_key)} "
                f"[{level}]   {estimated_tag}"
            )
            _emitted_item_corr = True

        # Explain G once, in prose, rather than on every free-param name.
        if _emitted_item_corr:
            print(
                "  Note: G is the number of distinct units correlated across "
                "nodes — one per node, minus any nodes tied together by a b-/"
                "a-equality constraint (each tied set counts once). Correlation "
                "needs G ≥ 2, so if sharing collapses a group to a single unit "
                "that group has no correlation to estimate. The exact G, and "
                "the per-element chain columns, are fixed once Constraints are "
                "applied at fit time."
            )

        print()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_with_level(self, param: str, node: int) -> tuple[PriorSpec, int]:
        if param in ("b.binary", "b.poly"):
            if (param, node) in self.node_level:
                return self.node_level[(param, node)], 3
            if ("b", node) in self.node_level:
                return self.node_level[("b", node)], 3
            if param in self.type_level:
                return self.type_level[param], 2
            if "b" in self.type_level:
                return self.type_level["b"], 2
            return _make_defaults()["b"], 1
        if (param, node) in self.node_level:
            return self.node_level[(param, node)], 3
        if param in self.type_level:
            return self.type_level[param], 2
        defaults = _make_defaults()
        if param in defaults:
            return defaults[param], 1
        raise KeyError(f"No prior for '{param}' at node {node + 1}.")

    def _resolve_level(self, param: str, node: int) -> int:
        _, level = self._resolve_with_level(param, node)
        return level


# ===========================================================================
# Convenience constructor
# ===========================================================================

def build_default_priors(spec, node_models: list) -> Priors:
    """Return a Priors object with all package defaults."""
    return Priors.from_spec(spec, node_models)


# ===========================================================================
# Internal utilities
# ===========================================================================

def _base_param(param: str) -> str:
    """Extract base parameter name from dot-notation."""
    return param.split(".")[0]


_LOCATION_PARAMS = {"theta", "b", "b.binary", "b.poly", "beta", "kappa", "gamma"}
_MISSING_PARAMS = {"X_missing", "Q_missing", "W_missing"}


def _check_sampler_support(param: str, prior_spec: PriorSpec) -> None:
    """Raise when a prior is valid as a distribution but the sampler would not
    use it as written.

    A prior the sampler silently replaced or ignored would run without error
    and target a different posterior, so it is rejected here, at set() time,
    with the supported alternatives named:

    * location parameters (theta, b, b.binary, b.poly, beta, kappa, gamma):
      a Normal prior without bounds (the Gibbs draws are unbounded Normal);
      theta's mean must be 0 (it fixes the location of the latent scale);
      a variance slot holds a number or an InvGammaPrior, and a
      HalfNormalPrior only for theta (the only MH variance block);
    * missing covariate values: a Normal prior with a fixed mean and variance;
    * a: a bounded Normal / TruncNormal (Gibbs) or an unbounded LogNormal (MH);
    * c: a BetaPrior on (0, 1).
    """
    name = type(prior_spec).__name__
    lo = getattr(prior_spec, "lower", None)
    hi = getattr(prior_spec, "upper", None)

    if param in _LOCATION_PARAMS:
        if not isinstance(prior_spec, (NormalPrior, TruncNormalPrior)):
            raise TypeError(
                f"The prior on '{param}' must be a NormalPrior (got {name}): the "
                f"sampler draws '{param}' from a Normal full conditional.")
        if lo is not None or hi is not None:
            raise NotImplementedError(
                f"Bounds on the '{param}' prior are not supported: the Gibbs draws "
                f"of '{param}' are unbounded, so the bounds would be ignored. "
                f"Remove lower/upper.")
        if param == "theta" and float(prior_spec.mean) != 0.0:
            raise ValueError(
                f"The prior mean of theta must be 0 (got {prior_spec.mean}): it fixes "
                f"the location of the latent scale (shifting theta and b together "
                f"leaves the likelihood unchanged). Shift b instead, or use person "
                f"covariates.")
        var = prior_spec.variance
        if isinstance(var, PriorSpec):
            ok = (InvGammaPrior, HalfNormalPrior) if param == "theta" else (InvGammaPrior,)
            if not isinstance(var, ok):
                raise NotImplementedError(
                    f"An estimated variance for '{param}' needs an InvGammaPrior"
                    + (" or HalfNormalPrior" if param == "theta" else "")
                    + f" in the variance slot (got {type(var).__name__}).")
        return

    if param in _MISSING_PARAMS:
        if not isinstance(prior_spec, NormalPrior) or lo is not None or hi is not None:
            raise TypeError(
                f"The prior on '{param}' must be an unbounded NormalPrior (got {name}).")
        if isinstance(prior_spec.variance, PriorSpec):
            raise NotImplementedError(
                f"An estimated variance for '{param}' is not supported: missing "
                f"covariate values have a Normal prior with a fixed mean and variance, "
                f"e.g. NormalPrior(mean=0, variance=1).")
        return

    if param == "a":
        gibbs = isinstance(prior_spec, (NormalPrior, TruncNormalPrior)) and prior_spec.is_bounded
        mh = isinstance(prior_spec, LogNormalPrior) and not prior_spec.is_bounded
        if not (gibbs or mh):
            raise TypeError(
                f"Unsupported prior on 'a' ({prior_spec!r}). Use\n"
                f"  TruncNormalPrior(mean=1.0, variance=0.5, lower=0.0)   # exact Gibbs\n"
                f"  LogNormalPrior(mean=0.0, variance=0.5)                # MH, no bounds")
        return

    if param == "c":
        if not isinstance(prior_spec, BetaPrior) or (lo not in (None, 0, 0.0)) \
                or (hi not in (None, 1, 1.0)):
            raise TypeError(
                f"The prior on 'c' must be a BetaPrior on (0, 1) (got {prior_spec!r}): "
                f"the probit 3PL draw is the Beta full conditional of the "
                f"Beguin-Glas augmentation, and the logit 3PL step uses the Beta "
                f"density.")
        return


def _check_support_domain(param: str, prior_spec: PriorSpec, force: bool = False) -> None:
    """Reject priors whose support is incoherent with the parameter's domain.

    The purpose is fail-fast helpfulness: an incoherent prior often does NOT
    crash — the sampler runs, produces numbers, and silently cannot reach the
    truth (e.g. an unbounded Normal on the discrimination `a` puts half its
    mass on a <= 0 and disables the truncated-Normal Gibbs path). Catching it
    here saves the user a wasted run.

    Domains
    -------
    a       : (0, inf)   discrimination must be positive
    c       : (0, 1)     guessing is a probability
    theta,  : (-inf,inf) location parameters — must not be one-sided
    b, beta,             truncated to a half-line
    kappa, gamma

    A one-sided distribution (LogNormal, HalfNormal) is positive-support by
    construction. A NormalPrior / TruncNormalPrior is positive-support iff its
    lower bound is >= 0, and lives in [0,1] iff additionally upper <= 1.

    force
    -----
    When ``force=True`` the incoherence is downgraded from a TypeError to a
    UserWarning. This is the escape hatch for models that deliberately
    constrain a parameter (e.g. a sign-constrained covariate effect): the
    check protects against typos by default but never traps a user who means
    it. The library cannot know the model better than the user does.

    EXCEPTION: an unbounded prior on 'a' (discrimination) is rejected
    unconditionally, ignoring ``force``. There is no correct behavior to fall
    back to (the model is non-identified and the log-scale MH cannot represent
    a <= 0), so unlike the other cases there is nothing a user could mean by
    it. See the 'a' branch below.

    Raises
    ------
    TypeError
        When the support cannot match the parameter's domain and force=False.
    """
    base = _base_param(param)
    lo = getattr(prior_spec, "lower", None)
    hi = getattr(prior_spec, "upper", None)
    is_one_sided_positive = isinstance(prior_spec, (LogNormalPrior, HalfNormalPrior))

    msg: str | None = None

    if base == "a":
        # discrimination must be positive-support
        positive = is_one_sided_positive or (lo is not None and lo >= 0.0)
        if not positive:
            # NO force escape hatch for 'a'.  Unlike a deliberate one-sided
            # constraint on a location parameter, an unbounded prior on 'a' has
            # no correct behavior to fall back to: (1) it leaves the sign
            # reflection (theta, a, b) -> (-theta, -a, -b) unbroken, so the
            # model is not identified (bimodal posterior, label switching);
            # and (2) the discrimination MH sampler is a log-scale random walk
            # that cannot represent a <= 0 and would silently reinterpret the
            # Normal's (mean, variance) on the log scale.  So we always raise,
            # regardless of force.
            raise TypeError(
                f"Prior on 'a' (discrimination) must have positive support, "
                f"because discrimination is constrained to a > 0. Got "
                f"{prior_spec.__class__.__name__} with no lower>=0 bound.\n"
                f"This is rejected with no force override: an unbounded prior on "
                f"'a' makes the model non-identified (the sign reflection "
                f"(theta, a, b) -> (-theta, -a, -b) is unbroken) and cannot be "
                f"represented by the log-scale discrimination sampler. Use a "
                f"positive-support prior instead:\n"
                f"  TruncNormalPrior(mean=1.0, variance=0.5, lower=0.0)  "
                f"# exact Gibbs\n"
                f"  NormalPrior(mean=1.0, variance=0.5, lower=0.0)       "
                f"# exact Gibbs\n"
                f"  LogNormalPrior(mean=0.0, variance=0.5)               "
                f"# positive, MH"
            )
        return

    elif base == "c":
        # guessing is a probability in (0, 1)
        is_beta = isinstance(prior_spec, BetaPrior)
        in_unit = (lo is not None and hi is not None and lo >= 0.0 and hi <= 1.0)
        if not (is_beta or in_unit):
            msg = (
                f"Prior on 'c' (guessing) must have support within [0, 1], "
                f"because guessing is a probability. Got "
                f"{prior_spec.__class__.__name__}"
                f"{'' if (lo is None and hi is None) else f' with lower={lo}, upper={hi}'}"
                f". Use BetaPrior(alpha, beta) (default Beta(5,17)) or a "
                f"NormalPrior / TruncNormalPrior with lower>=0 and upper<=1."
            )

    elif base in ("theta", "b", "beta", "kappa", "gamma"):
        # location parameters live on the whole real line; a one-sided
        # positive prior forbids the negative half they must be able to take.
        if is_one_sided_positive:
            msg = (
                f"Prior on '{base}' must have support on the whole real line "
                f"(it is a location parameter that can be negative). Got "
                f"{prior_spec.__class__.__name__}, which is positive-support "
                f"only. Use NormalPrior (default) or a TruncNormalPrior only if "
                f"you deliberately intend a one-sided constraint."
            )
    # variance-slot distributions (InvGamma/HalfNormal at top level) are
    # handled by the dedicated guard earlier in set(); nothing to do here.

    if msg is None:
        return
    if force:
        warnings.warn(
            msg + "\n(Proceeding because force=True; make sure this "
            "constraint is intentional.)",
            UserWarning,
            stacklevel=4,
        )
    else:
        raise TypeError(
            msg + "\nIf this constraint is intentional, rerun with force=True."
        )


def _check_variance_conjugacy(
    param:      str,
    prior_spec: PriorSpec,
    force:      bool = False,
) -> None:
    """Reject a hierarchical variance whose likelihood is not conjugate to it.

    A hierarchical variance -- i.e. ``prior_spec.variance`` holding a
    ``PriorSpec`` rather than a float -- is estimated rather than fixed. That
    estimation is an exact Gibbs draw ONLY when the coefficient prior is an
    unbounded Normal, because then

        sigma^2 | x  ~  InvGamma(alpha + n/2,  beta + 0.5 * sum_i (x_i - mu)^2)

    is closed-form (Normal-InvGamma conjugacy).

    WHY BOUNDS BREAK IT
    -------------------
    A bounded/truncated Normal density carries a normalising constant

        1 / [ Phi((u - mu)/sigma) - Phi((l - mu)/sigma) ]

    which is itself a function of sigma. Multiplying n such factors by an
    InvGamma prior does not give an InvGamma posterior -- the conjugate form
    is destroyed, and no closed-form draw exists. The variance must then be
    updated by Metropolis-Hastings.

    This is exactly the situation for the default discrimination prior,
    ``TruncNormalPrior(mean=1.0, variance=0.5, lower=0.0)``: placing an
    InvGammaPrior in its ``variance`` slot looks conjugate but is not.

    force
    -----
    ``force=True`` downgrades the TypeError to a UserWarning and permits the
    hierarchical variance, which will then be estimated by an adapted MH block
    rather than a Gibbs draw. This costs a step size to tune and adds
    autocorrelation to an otherwise-exact sampler, but it is a legitimate
    model. As elsewhere in this module, the check protects against an easy
    mistake by default without trapping a user who means it.

    Raises
    ------
    TypeError
        When a hierarchical variance sits on a bounded prior and force=False.
    """
    var_slot = getattr(prior_spec, "variance", None)
    if not isinstance(var_slot, PriorSpec):
        return                      # fixed variance -- nothing to check

    lo = getattr(prior_spec, "lower", None)
    hi = getattr(prior_spec, "upper", None)
    if lo is None and hi is None:
        return                      # unbounded Normal -- conjugate, Gibbs is exact

    base   = _base_param(param)
    bounds = []
    if lo is not None:
        bounds.append(f"lower={lo}")
    if hi is not None:
        bounds.append(f"upper={hi}")
    bound_str = ", ".join(bounds)

    detail = (
        f"{type(var_slot).__name__} on the variance of '{base}' is NOT "
        f"conjugate, because '{base}' has a bounded prior ({bound_str}). The "
        f"truncation constant depends on sigma, so no closed-form InvGamma "
        f"full conditional exists and the variance cannot be drawn by Gibbs."
    )

    if not force:
        raise TypeError(
            detail + "\n\n"
            f"Either fix the variance to a float:\n"
            f"  priors.set({param!r}, {type(prior_spec).__name__}(..., variance=0.5))\n"
            f"or keep the hierarchical variance and accept an MH update:\n"
            f"  priors.set({param!r}, {prior_spec!r}, force=True)"
        )

    warnings.warn(
        detail + " Proceeding under force=True: this variance will be "
        "estimated with an adapted Metropolis-Hastings block instead of an "
        "exact Gibbs draw, which introduces a tuned step size and "
        "autocorrelation into an otherwise-exact sampler.",
        UserWarning,
        stacklevel=4,
    )


def _warn_halfnormal_variance(param: str, prior_spec: PriorSpec) -> None:
    """Emit UserWarning if a HalfNormalPrior is used as a variance prior."""
    is_halfnormal = isinstance(prior_spec, HalfNormalPrior)
    has_halfnormal_slot = (
        hasattr(prior_spec, "variance")
        and isinstance(prior_spec.variance, HalfNormalPrior)
    )
    if is_halfnormal or has_halfnormal_slot:
        base = _base_param(param)
        warnings.warn(
            f"HalfNormalPrior on '{base}' variance breaks conjugacy with the "
            f"Normal likelihood, forcing an MH update for that variance "
            f"component. This adds an adapted block to an otherwise-Gibbs "
            f"sampler. Default InvGamma is strongly preferred unless you have "
            f"a specific reason to prefer HalfNormal.",
            UserWarning,
            stacklevel=4,
        )
