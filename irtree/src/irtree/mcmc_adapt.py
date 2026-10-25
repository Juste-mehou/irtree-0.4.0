"""
irtree.mcmc_adapt
=================
Layer 7 — Robbins-Monro step-size adaptation for Metropolis-Hastings blocks.

SCOPE
=====
This module tunes every Metropolis-Hastings block in the sampler: probit
discriminations with a LogNormal prior, every parameter of logit/GPCM nodes
(theta person-wise, b item-wise, a, c), Cowles threshold steps, covariate
coefficients and missing covariates used by logit nodes, estimated a
variances, HalfNormal sigma2, and LKJ correlations.  Each block's target
acceptance depends on how many parameters it proposes jointly
(``target_accept``: 0.441 for one, falling to 0.234 for six or more); the
vectorized per-person and per-item blocks adapt one shared step on the mean
acceptance across their units.

A pure-Gibbs chain (binary probit nodes with conjugate priors; GRM nodes add a
Cowles MH step) has no MH block, so
this module is not called; such chains still discard a burn-in of n_warmup
iterations (at least 2*M*W in evidence mode; see mcmc_sampler.run_chain).

FIRST PRINCIPLES: WHY ROBBINS-MONRO ON log(ℓ)?
================================================
The Metropolis-Hastings algorithm proposes a new value from a Gaussian
centred at the current value:

    z* = z_t + ℓ · ε,     ε ~ Normal(0, I_d)

The single scalar ℓ controls the proposal spread.  If ℓ is too small the
chain moves in tiny steps (high acceptance, low mixing); if too large, the
chain rejects most proposals (low acceptance, wastes compute).

Roberts, Gelman & Gilks (1997) showed that for a d-dimensional RWM
on a target with iid components, the optimal acceptance rate is 0.234 as
d → ∞, achieved by setting ℓ_opt = 2.38 / √d.  This is the TARGET_RATE
constant below.

The **Robbins-Monro stochastic approximation** (Robbins & Monro 1951)
adapts log(ℓ) rather than ℓ directly.  Working on the log scale guarantees
ℓ = exp(log_ℓ) > 0 regardless of the update direction, and makes the
adaptation scale-free.

The update rule at iteration t is:

    log(ℓ_{t+1}) = log(ℓ_t) + γ_t · (accept_t − target)

where:
  - accept_t ∈ {0, 1} is the binary accept/reject outcome at step t
  - target   = 0.234 (the optimal acceptance rate)
  - γ_t > 0  is the diminishing learning rate (the "schedule")

When accept_t > target  (accepting too often → ℓ too small → increase)
When accept_t < target  (rejecting too often → ℓ too large → decrease)
At optimum: E[accept] = target, so E[Δ log ℓ] = 0 — the log step size
stops drifting.

ROBBINS-MONRO CONDITIONS
=========================
For convergence, the schedule γ_t = C / (t + t₀)^α must satisfy:

    (1)  Σ γ_t = ∞    [can reach any value — ensures exploration]
    (2)  Σ γ_t² < ∞   [noise averages out — ensures convergence]

These two conditions hold iff:

    α ∈ (0.5, 1]   and   C > 0   and   t₀ > 0

Default: α = 0.6, C = 1.0, t₀ = 10.
  - Condition (1): Σ 1/(t+10)^0.6 diverges  (since 0.6 ≤ 1) ✓
  - Condition (2): Σ 1/(t+10)^1.2 converges (since 1.2 > 1)  ✓

``make_schedule`` validates these conditions and raises ValueError if
the caller provides parameters outside the valid range.

FREEZE AFTER n_warmup
======================
After n_warmup adaptation steps the schedule is frozen: log(ℓ) is fixed
at its final adapted value and the Robbins-Monro update is no longer applied.
All iterations after warmup are drawn with the fixed, tuned step size.

In the sampler loop (mcmc_sampler.py):

    if t < n_warmup:
        adapt[r] = mcmc_adapt.adapt_step(adapt[r], accept, t)
    # else: adapt[r] is frozen — not passed to adapt_step

Equivalently, ``adapt_step`` itself silently no-ops when
``state.frozen is True`` (safe if the caller forgets the guard).

LOG_STEP_BOUNDS
===============
Numerical safeguard: log(ℓ) is clamped to [-23.03, 4.61], corresponding
to ℓ ∈ [10^{-10}, 100].  This prevents the step size from collapsing to
zero (where no proposals are ever accepted) or exploding to a value where
every proposal is rejected.

REFERENCES
==========
Robbins, H. & Monro, S. (1951). A stochastic approximation method.
    Annals of Mathematical Statistics, 22(3), 400–407.

Roberts, G.O., Gelman, A. & Gilks, W.R. (1997). Weak convergence and
    optimal scaling of random walk Metropolis algorithms.
    Annals of Applied Probability, 7(1), 110–120.

Haario, H., Saksman, E. & Tamminen, J. (2001). An adaptive Metropolis
    algorithm. Bernoulli, 7(2), 223–242.

Atchadé, Y.F. & Rosenthal, J.S. (2003). On adaptive Markov chain Monte
    Carlo algorithms. Bernoulli, 9(5), 815–828.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Optimal MH acceptance rate in the high-dimensional limit
#: (Roberts, Gelman & Gilks 1997). Correct for a random-walk block that proposes
#: many (~6+) roughly-independent components at once.
TARGET_RATE: float = 0.234

#: Finite-dimension optimal acceptance rates for a random-walk Metropolis block,
#: from the same diffusion-limit analysis evaluated at low dimension
#: (Roberts & Rosenthal 2001, "Optimal scaling for various Metropolis-Hastings
#: algorithms"). The optima descend from 0.441 (scalar) toward 0.234 as the block
#: dimension grows; beyond ~6 the difference is negligible and TARGET_RATE is used.
_OPTIMAL_ACCEPT = {1: 0.441, 2: 0.352, 3: 0.316, 4: 0.285, 5: 0.275}


def target_accept(dim: int) -> float:
    """Optimal RW-Metropolis acceptance rate for a block of the given dimension.

    ``dim`` is the number of parameters the block proposes and accepts/rejects
    **jointly** — not the number of blocks. A scalar block (one correlation, one
    variance, one item's difficulty) wants ~0.441; three GRM thresholds of
    one item ~0.316; a high-dimensional joint block wants 0.234. Per-unit
    vectorized blocks use the per-unit dimension. Values for dim >= 6 fall back to
    TARGET_RATE, where the finite-dimension correction is negligible.
    """
    return _OPTIMAL_ACCEPT.get(int(dim), TARGET_RATE)

#: log(ℓ) lower bound → ℓ_min = exp(-23.03) ≈ 10^{-10}
LOG_STEP_MIN: float = -23.025850929940457   # = log(1e-10)

#: log(ℓ) upper bound → ℓ_max = exp(4.605) ≈ 100
LOG_STEP_MAX: float = 4.605170185988092     # = log(100)


# ---------------------------------------------------------------------------
# Evidence-triggered warmup defaults
# ---------------------------------------------------------------------------
# The fixed-length warmup freezes every block at a user-chosen iteration count,
# a blind bet that the count is long enough for log(ℓ) to both reach and settle
# at the scale whose acceptance rate equals ``target``.  The evidence-triggered
# alternative freezes each block when its OWN acceptance rate has demonstrably
# stabilised: when a windowed acceptance rate stays within ``band`` of target
# for ``M`` consecutive windows (so the drift term E[Δ log ℓ] ≈ 0), or a cap is
# reached, whichever comes first.
#
# These defaults were chosen by simulating the package's actual Robbins-Monro
# schedule (C=1, t0=10, α=0.6) on a random-walk target started from badly-scaled
# steps and measuring both the warmup length and the TRUE post-freeze acceptance:
#   - realised post-freeze acceptance was target ± 0.03 and ESS/iter within 3%
#     of optimal across ALL tested (W, k, M), i.e. quality is insensitive to the
#     trigger — so the trigger is tuned for predictable, short warmup instead.
#   - freeze time ≈ M·W + one window; M=3 is predictable (no fat tail), M=5 had
#     a long tail (one out-of-band window resets the consecutive counter).
#   - k (band width in SE units) barely moved anything; k=2 ≈ "statistically
#     indistinguishable from target at this window size".

#: Non-overlapping window length over which the acceptance rate is measured.
#: np ≈ 44 at target 0.44 (Gaussian approximation to the rate is solid, ≫ 5).
EVIDENCE_WINDOW_W: int = 100

#: Band half-width in units of the rate's standard error.  band = k·SE where
#: SE = sqrt(target(1-target)/W).  k=2 ⇒ ~95% of windows land in-band at the
#: optimum, so the trigger fires reliably without being tripped by noise.
EVIDENCE_BAND_K: float = 2.0

#: Number of consecutive in-band windows required before freezing.  3 is enough
#: to rule out a still-drifting rate (which cannot hold a tight band 3× running)
#: without the long upper tail that larger M incurs.
EVIDENCE_NEED_M: int = 3

# ---------------------------------------------------------------------------
# Evidence-warmup CAP policy (used by the sampler, not by adapt_step itself)
# ---------------------------------------------------------------------------
# The cap is the forced-freeze backstop for a block that never satisfies the
# stability test.  It is intentionally NOT a fixed absolute number: a fixed low
# ceiling would force-freeze a genuinely-still-adapting complex block mis-tuned,
# which is the worst outcome.  Instead the cap is a FRACTION of the run, so it
# scales with n_iter (complex models, which are run longer, get proportionally
# more room to stabilise), bounded below by a guaranteed kept-sample budget so
# that even a freeze at the cap leaves enough draws for inference.
#
# A well-behaved block freezes at its own evidence point long before the cap, so
# raising the cap costs nothing for easy models and only helps hard ones.

#: Warmup may consume at most this fraction of n_iter before the forced freeze.
#: The other (1 - fraction) is the minimum kept-sample share.  1/2 balances
#: "give a hard block room to converge" against "keep enough to do inference".
EVIDENCE_WARMUP_FRACTION: float = 0.5

#: Default guaranteed number of KEPT (post-thinning) draws per chain.  Acts as a
#: floor that only binds on short runs (n_iter < 2·MIN_KEEP·thin_by); for longer
#: runs EVIDENCE_WARMUP_FRACTION governs and the kept phase is larger.  At the
#: ~0.22 ESS/iter of a well-tuned scalar block this is ESS ≈ 220/chain; raise it
#: for a high-dimensional joint block (logit/GPCM θ,b), whose ESS/iter is far
#: lower (≈0.04 at d≈10), via run_chain(min_keep=...).
EVIDENCE_MIN_KEEP: int = 1000


def evidence_band_delta(target: float, W: int, k: float = EVIDENCE_BAND_K) -> float:
    """Absolute band half-width for the windowed acceptance-rate test.

    The acceptance rate over W Bernoulli(target) trials has standard error
    sqrt(target·(1−target)/W); the band is ``k`` of those.  Computed per block
    from that block's own target, so a high-dimensional block (target 0.234)
    gets a different, correct band from a scalar block (target 0.44).

    Examples
    --------
    >>> round(evidence_band_delta(0.44, 100), 3)
    0.099
    >>> round(evidence_band_delta(0.234, 100), 3)
    0.085
    """
    if W < 1:
        raise ValueError(f"window W must be ≥ 1; got {W}.")
    if not (0.0 < target < 1.0):
        raise ValueError(f"target must be in (0, 1); got {target}.")
    return k * math.sqrt(target * (1.0 - target) / W)


# ---------------------------------------------------------------------------
# Schedule factory
# ---------------------------------------------------------------------------

def make_schedule(
    n_warmup:    int,
    C:           float = 1.0,
    t0:          float = 10.0,
    alpha:       float = 0.6,
) -> Callable[[int], float]:
    """Build the Robbins-Monro learning-rate schedule γ_t.

    Returns a callable ``gamma(t) -> float`` where t is the 0-based
    iteration index.  The schedule is:

        γ_t = C / (t + t₀)^α

    **Robbins-Monro conditions verified at construction**:

    The returned schedule is only valid for the adaptation phase
    (0 ≤ t < n_warmup).  The schedule object carries the validated
    parameters; the sampler must not call it for t ≥ n_warmup.

    Parameters
    ----------
    n_warmup : int
        Number of warmup (adaptation) iterations.  Must be ≥ 1.
        After this many calls to ``adapt_step``, the ``AdaptState``
        is frozen and γ_t is never called again.
    C : float > 0
        Scale constant.  Larger C means faster initial adaptation.
        Default 1.0.
    t0 : float > 0
        Offset that prevents γ_0 from being too large.  Default 10.0
        gives γ_0 = 1 / 10^0.6 ≈ 0.25, a moderate initial learning rate.
    alpha : float in (0.5, 1]
        Decay exponent.  Must satisfy 0.5 < alpha ≤ 1 to satisfy both
        Robbins-Monro conditions simultaneously.

    Returns
    -------
    Callable[[int], float]
        Function that maps iteration index t (int, 0-based) to γ_t (float).

    Raises
    ------
    ValueError
        If any parameter violates the Robbins-Monro conditions or basic
        positivity requirements.

    Examples
    --------
    >>> gamma = make_schedule(n_warmup=1000)
    >>> gamma(0)    # γ_0 = 1 / 10^0.6 ≈ 0.251
    0.25118864315095796
    >>> gamma(99)   # γ_99 = 1 / 109^0.6 ≈ 0.0419
    0.041897...
    """
    # --- Validation --------------------------------------------------------

    if n_warmup < 1:
        raise ValueError(
            f"n_warmup must be ≥ 1; got {n_warmup}."
        )
    if C <= 0:
        raise ValueError(
            f"C must be > 0 (scale of the schedule); got {C}."
        )
    if t0 <= 0:
        raise ValueError(
            f"t0 must be > 0 (offset preventing large initial step); got {t0}."
        )
    if alpha <= 0.5:
        raise ValueError(
            f"alpha must be > 0.5 to satisfy Robbins-Monro condition (2): "
            f"Σ γ_t² < ∞  requires  2·alpha > 1.  Got alpha={alpha}."
        )
    if alpha > 1.0:
        raise ValueError(
            f"alpha must be ≤ 1.0 to satisfy Robbins-Monro condition (1): "
            f"Σ γ_t = ∞  requires  alpha ≤ 1.  Got alpha={alpha}."
        )

    # --- Build callable ----------------------------------------------------

    def gamma(t: int) -> float:
        """Learning rate at iteration t.

        Parameters
        ----------
        t : int ≥ 0
            0-based iteration index within the warmup phase.

        Returns
        -------
        float
            γ_t = C / (t + t₀)^α  — strictly positive, decreasing in t.
        """
        return C / (t + t0) ** alpha

    # Attach metadata for inspection / repr
    gamma.n_warmup = n_warmup  # type: ignore[attr-defined]
    gamma.C        = C          # type: ignore[attr-defined]
    gamma.t0       = t0         # type: ignore[attr-defined]
    gamma.alpha    = alpha      # type: ignore[attr-defined]

    return gamma


# ---------------------------------------------------------------------------
# AdaptState
# ---------------------------------------------------------------------------

@dataclass
class AdaptState:
    """State of the Robbins-Monro step-size adaptation for one MH block.

    One ``AdaptState`` is created per MH block of the sampler (see the
    module docstring); pure-Gibbs parameters have none.  A vectorized
    per-unit block (persons or items) shares one state, adapted on the mean
    acceptance across its units.

    Attributes
    ----------
    log_step_size : float
        Current log of the MH proposal standard deviation ℓ.
        The step size itself is ``exp(log_step_size)``.
        Clamped to [LOG_STEP_MIN, LOG_STEP_MAX] after each update.
    gamma : Callable[[int], float]
        The Robbins-Monro learning-rate schedule from ``make_schedule``.
    target : float
        Target acceptance rate.  Default 0.234 (Roberts et al. 1997); the
        sampler sets ``target_accept(dim)`` for the block's dimension.
    n_accepted : int
        Cumulative number of accepted proposals (for rate reporting).
    n_proposed : int
        Cumulative number of proposals (for rate reporting).
    frozen : bool
        When True, ``adapt_step`` returns ``self`` unchanged.
        Set by ``adapt_step``: at t ≥ n_warmup (fixed mode), or on the
        stability test or at the cap (evidence mode).
    evidence : bool
        Evidence-triggered freezing (see adapt_step).  The fields below are
        inert when False.
    window_W : int
        Window length W of the acceptance-rate estimate (default 100).
    band_delta : float
        Half-width of the in-band test |window rate - target| <= band_delta;
        filled from evidence_band_delta(target, W) when <= 0.
    need_consecutive_M : int
        Consecutive in-band windows required to freeze (default 3), so the
        earliest freeze is at M·W iterations.
    window_accepted, window_proposed : float, int
        Running counts within the current window.
    consecutive_in_band : int
        Consecutive full windows in band so far.
    frozen_at_t : int or None
        Iteration at which the block froze (None while adapting).

    Examples
    --------
    Typical construction::

        state = AdaptState(
            log_step_size=0.0,
            gamma=make_schedule(n_warmup=1000),
        )

    Typical update inside sampler::

        if t < n_warmup:
            state = adapt_step(state, accepted=bool(accept), t=t)
    """

    log_step_size: float
    gamma:         Callable[[int], float]
    target:        float = TARGET_RATE
    n_accepted:    float = 0.0
    n_proposed:    int   = 0
    frozen:        bool  = False

    # -- evidence-triggered warmup (ignored when evidence is False) ----------
    #: When True, the block freezes on the windowed-acceptance stability test
    #: in ``adapt_step`` (and at gamma.n_warmup as a hard cap), rather than
    #: purely at gamma.n_warmup.  When False, behaviour is the original
    #: fixed-length warmup and all fields below are inert.
    evidence:            bool  = False
    #: Window length W for the acceptance-rate estimate.
    window_W:            int   = EVIDENCE_WINDOW_W
    #: Band half-width (absolute).  If <= 0 and evidence is True, it is filled
    #: in __post_init__ from evidence_band_delta(target, window_W).
    band_delta:          float = 0.0
    #: Consecutive in-band windows required to freeze.
    need_consecutive_M:  int   = EVIDENCE_NEED_M
    #: Running accept count within the current (not-yet-full) window.
    window_accepted:     float = 0.0
    #: Proposals seen within the current window (resets to 0 each full window).
    window_proposed:     int   = 0
    #: How many consecutive full windows have landed in-band so far.
    consecutive_in_band: int   = 0
    #: Iteration index at which this block froze (None while still adapting).
    #: Recorded so the sampler and MCMCResult can report per-block warmup length.
    frozen_at_t:         int | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.log_step_size):
            raise ValueError(
                f"log_step_size must be finite; got {self.log_step_size}."
            )
        if not callable(self.gamma):
            raise TypeError(
                f"gamma must be callable (use make_schedule()); "
                f"got {type(self.gamma).__name__}."
            )
        if not (0.0 < self.target < 1.0):
            raise ValueError(
                f"target must be in (0, 1); got {self.target}."
            )
        if self.n_accepted < 0 or self.n_proposed < 0:
            raise ValueError("n_accepted and n_proposed must be ≥ 0.")
        # Evidence-mode field validation + band_delta default.
        if self.evidence:
            if self.window_W < 1:
                raise ValueError(f"window_W must be ≥ 1; got {self.window_W}.")
            if self.need_consecutive_M < 1:
                raise ValueError(
                    f"need_consecutive_M must be ≥ 1; got {self.need_consecutive_M}."
                )
            if self.band_delta <= 0.0:
                self.band_delta = evidence_band_delta(self.target, self.window_W)
        # Clamp on construction so the object is always in valid bounds
        self.log_step_size = float(
            max(LOG_STEP_MIN, min(LOG_STEP_MAX, self.log_step_size))
        )

    # -- derived properties -------------------------------------------------

    @property
    def step_size(self) -> float:
        """MH proposal standard deviation ℓ = exp(log_step_size).

        Always in (0, ∞); clamped to [exp(LOG_STEP_MIN), exp(LOG_STEP_MAX)].
        """
        return math.exp(self.log_step_size)

    @property
    def acceptance_rate(self) -> float | None:
        """Empirical acceptance rate n_accepted / n_proposed.

        Returns None if no proposals have been made yet.
        """
        if self.n_proposed == 0:
            return None
        return self.n_accepted / self.n_proposed

    def __repr__(self) -> str:
        rate = self.acceptance_rate
        rate_str = f"{rate:.3f}" if rate is not None else "N/A"
        if self.frozen:
            where = f" @t={self.frozen_at_t}" if self.frozen_at_t is not None else ""
            status = f"frozen{where}"
        else:
            status = "adapting"
        mode = ""
        if self.evidence:
            mode = (f", evidence(W={self.window_W}, δ={self.band_delta:.3f}, "
                    f"M={self.need_consecutive_M}, in_band={self.consecutive_in_band})")
        return (
            f"AdaptState({status}, ℓ={self.step_size:.5f}, "
            f"log_ℓ={self.log_step_size:.4f}, "
            f"rate={rate_str} [{self.n_accepted:g}/{self.n_proposed}]{mode})"
        )


# ---------------------------------------------------------------------------
# Adaptation update
# ---------------------------------------------------------------------------

def adapt_step(
    state:    AdaptState,
    accepted,          # bool OR float acceptance fraction in [0,1]
    t:        int,
) -> AdaptState:
    """Apply one Robbins-Monro step to the adaptation state.

    Implements the update:

        log(ℓ_{t+1}) = clip(log(ℓ_t) + γ_t · (accept_t − target))

    where clip restricts log(ℓ) to [LOG_STEP_MIN, LOG_STEP_MAX].

    **Freeze logic**: the state becomes frozen when t ≥ n_warmup
    (as recorded in ``state.gamma.n_warmup``).  Once frozen, this
    function returns the state unchanged — a safe no-op if the sampler
    accidentally calls it after warmup.

    Parameters
    ----------
    state : AdaptState
        Current adaptation state.
    accepted : bool or float
        The acceptance signal at iteration t.  A bool for a single-proposal
        block (the usual case).  For a block that accepts/rejects many
        components in one step (e.g. per-item discrimination, which does I
        independent accept/rejects against a shared step size), pass the
        MEAN per-component acceptance in [0, 1]: Robbins-Monro then drives the
        shared step to where the mean per-component rate equals ``target``.
        Using the fraction rather than a single Bernoulli is lower-variance
        and is the correct signal for a shared-scale block.
    t : int ≥ 0
        0-based iteration index.  Used to look up γ_t and to check
        whether warmup is complete.

    Returns
    -------
    AdaptState
        Updated state.  Uses ``dataclasses.replace`` so the input is
        never mutated — the caller assigns the returned value.

    Notes
    -----
    Using ``dataclasses.replace`` (immutable-style update) rather than
    mutating ``state`` in place makes the adaptation history easy to
    reason about and test: each call produces a distinct object.

    Examples
    --------
    >>> gamma = make_schedule(n_warmup=500)
    >>> state = AdaptState(log_step_size=0.0, gamma=gamma)
    >>> state = adapt_step(state, accepted=True, t=0)
    >>> state.step_size  # slightly larger than 1.0 after one accept
    """
    # -- Already frozen: no-op (still tally for whole-run acceptance) --------
    if state.frozen:
        return replace(
            state,
            n_accepted = state.n_accepted + float(accepted),
            n_proposed = state.n_proposed + 1,
        )

    acc      = float(accepted)
    n_warmup = state.gamma.n_warmup  # type: ignore[attr-defined]

    # =====================================================================
    # EVIDENCE MODE — freeze when the windowed acceptance rate has stabilised
    # =====================================================================
    # Here n_warmup is the hard CAP (n_warmup_max), not a fixed length.  We
    # always apply the Robbins-Monro update, accumulate a non-overlapping
    # window of acceptances, and when a full window closes, test whether its
    # rate is within band_delta of target.  M consecutive in-band windows (or
    # hitting the cap) freezes the block.  The freeze is driven by evidence
    # that log(ℓ) has stopped drifting, not by a pre-chosen iteration count.
    if state.evidence:
        gamma_t   = state.gamma(t)
        new_log_l = state.log_step_size + gamma_t * (acc - state.target)
        new_log_l = max(LOG_STEP_MIN, min(LOG_STEP_MAX, new_log_l))

        # Accumulate the current (non-overlapping) window.
        win_acc = state.window_accepted + acc
        win_n   = state.window_proposed + 1
        consec  = state.consecutive_in_band

        if win_n >= state.window_W:
            rate    = win_acc / win_n
            in_band = abs(rate - state.target) <= state.band_delta
            consec  = consec + 1 if in_band else 0
            win_acc = 0.0           # reset the window
            win_n   = 0

        # Freeze on sustained stability OR on reaching the cap.
        do_freeze = (consec >= state.need_consecutive_M) or (t >= n_warmup)

        return replace(
            state,
            log_step_size       = new_log_l,
            n_accepted          = state.n_accepted + acc,
            n_proposed          = state.n_proposed + 1,
            window_accepted     = win_acc,
            window_proposed     = win_n,
            consecutive_in_band = consec,
            frozen              = do_freeze,
            frozen_at_t         = t if do_freeze else None,
        )

    # =====================================================================
    # FIXED MODE — freeze at t >= n_warmup
    # =====================================================================
    if t >= n_warmup:
        return replace(
            state,
            frozen      = True,
            frozen_at_t = t,
            n_accepted  = state.n_accepted + acc,
            n_proposed  = state.n_proposed + 1,
        )

    # -- Robbins-Monro update -----------------------------------------------
    gamma_t    = state.gamma(t)
    new_log_l  = state.log_step_size + gamma_t * (acc - state.target)

    # Clamp to numerical bounds
    new_log_l = max(LOG_STEP_MIN, min(LOG_STEP_MAX, new_log_l))

    return replace(
        state,
        log_step_size = new_log_l,
        n_accepted    = state.n_accepted + acc,
        n_proposed    = state.n_proposed + 1,
    )


# ---------------------------------------------------------------------------
# Convenience: build a ready-to-use AdaptState with defaults
# ---------------------------------------------------------------------------

def make_adapt_state(
    n_warmup:       int,
    log_step_size:  float = 0.0,
    target:         float = TARGET_RATE,
    C:              float = 1.0,
    t0:             float = 10.0,
    alpha:          float = 0.6,
    evidence:       bool  = False,
    window_W:       int   = EVIDENCE_WINDOW_W,
    band_k:         float = EVIDENCE_BAND_K,
    need_M:         int   = EVIDENCE_NEED_M,
) -> AdaptState:
    """Create an AdaptState with default parameters in one call.

    Combines ``make_schedule`` + ``AdaptState`` construction.

    Parameters
    ----------
    n_warmup : int
        Fixed mode: number of warmup iterations before freezing.
        Evidence mode: the hard CAP (n_warmup_max) — the block freezes on the
        stability test, or here if it never stabilises.  Either way this is
        what ``make_schedule`` records as ``gamma.n_warmup``.
    log_step_size : float
        Initial log step size.  Default 0.0 → ℓ = 1.0.
    target : float
        Target acceptance rate.  Default 0.234.
    C, t0, alpha : float
        Schedule parameters passed to ``make_schedule``.
    evidence : bool
        If True, build an evidence-triggered block (windowed-stability freeze).
    window_W, band_k, need_M :
        Evidence-mode knobs (window length, band in SE units, consecutive
        in-band windows to freeze).  ``band_delta`` is computed from
        ``evidence_band_delta(target, window_W, band_k)``.

    Returns
    -------
    AdaptState

    Examples
    --------
    >>> state = make_adapt_state(n_warmup=1000)
    >>> state.step_size
    1.0
    >>> state.frozen
    False
    >>> ev = make_adapt_state(n_warmup=2000, target=0.44, evidence=True)
    >>> ev.evidence and ev.band_delta > 0
    True
    """
    gamma = make_schedule(n_warmup=n_warmup, C=C, t0=t0, alpha=alpha)
    band_delta = evidence_band_delta(target, window_W, band_k) if evidence else 0.0
    return AdaptState(
        log_step_size=log_step_size,
        gamma=gamma,
        target=target,
        evidence=evidence,
        window_W=window_W,
        band_delta=band_delta,
        need_consecutive_M=need_M,
    )
