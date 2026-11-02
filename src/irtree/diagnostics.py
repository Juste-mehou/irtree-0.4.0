"""
irtree.diagnostics
==================
Layer 8 — MCMC convergence diagnostics.

DESIGN PHILOSOPHY
=================
This module serves two user profiles simultaneously:

  Profile A — MCMC-literate: wants raw numbers, calls functions directly.
  Profile B — IRT practitioner: wants a verdict, reads the console summary.

All raw statistical functions are importable and callable directly.  The
``convergence_report()`` and ``console_summary()`` functions assemble the
outputs into a two-layer interface: the console verdict, backed by the
full report object.

DIAGNOSTIC DISPATCH BY n_chains
================================
  n_chains = 1: Geweke (primary stationarity), H-W (step B), ESS, MCSE, ACF.
                R-hat raises ValueError.
  n_chains > 1: R-hat (primary), Geweke (supplementary), ESS, MCSE, ACF.


THRESHOLDS (module-level, user-overridable)
===========================================
  RHAT_WARN      = 1.01   warn when R-hat > this
  RHAT_FAIL      = 1.05   fail when R-hat > this
  ESS_WARN       = 400    warn when ESS < this  (Vehtari et al. 2021)
  GEWEKE_ALPHA   = 0.003  Geweke significance (|z| cutoff ~2.97); also the
                          default of convergence_report(geweke_alpha=...)
  HW_ALPHA       = 0.05   Heidelberger-Welch significance level
  HW_CRIT(_01, _10)       empirical critical values of the Bartlett-normalized
                          CvM statistic for alpha = 0.05 (0.01, 0.10)
  MCSE_SD_WARN   = 0.10   warn when MCSE/SD > this
  ACF_LAG50_WARN = 0.10   warn when |ACF(lag=50)| > this

Override at the module level:  ``import irtree.diagnostics as diag``
then ``diag.ESS_WARN = 200``.

IMPLEMENTATIONS
===============
rhat          — Vehtari et al. (2021) rank-normalized split-R-hat.
ess           — Vehtari et al. (2021) bulk ESS via rank transformation and the
                initial monotone sequence estimator of the ACF sum, plus a
                folded ESS (|z - median|) reported as the tail ESS.
mcse          — MCSE = SD / sqrt(ESS);  MCSE/SD = 1 / sqrt(ESS).
geweke        — First 10% vs last 50% z-test (Geweke 1992).
geweke_zcrit  — |z| cutoff for a two-sided significance level.
geweke_sweep  — Geweke z over successively later starting points.
heidelberger_welch — Cramér-von Mises stationarity test with progressive
                discard (Heidelberger & Welch 1983; no half-width test).
acf           — Sample ACF at specified lags via dot-product estimator.
convergence_report — Assembles DiagnosticReport from MCMCResult(s).
console_summary    — Prints the Layer 1 traffic-light summary.

REFERENCES
==========
Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Bürkner, P.-C. (2021).
    Rank-normalization, folding, and localization: An improved R̂ for assessing
    convergence of MCMC. Bayesian Analysis, 16, 667–718.

Geweke, J. (1992). Evaluating the accuracy of sampling-based approaches to
    calculating posterior moments. In J.M. Bernardo et al. (Eds.), Bayesian
    Statistics 4 (pp. 169–193). Oxford University Press.

Gelman, A. & Rubin, D.B. (1992). Inference from iterative simulation using
    multiple sequences. Statistical Science, 7, 457–472.

Heidelberger, P. & Welch, P.D. (1983). Simulation run length control in the
    presence of an initial transient. Operations Research, 31, 1109–1144.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy import stats as _stats
from scipy.fft import irfft as _irfft
from scipy.fft import next_fast_len as _next_fast_len
from scipy.fft import rfft as _rfft
from scipy.special import ndtri as _ndtri

# ===========================================================================
# Module-level thresholds (user-overridable)
# ===========================================================================

RHAT_WARN:      float = 1.01
RHAT_FAIL:      float = 1.05
ESS_WARN:       int   = 400
GEWEKE_ALPHA:   float = 0.003   # default Geweke significance (|z|>~2.97).
                                # Larger alpha -> smaller cutoff -> rejects more
                                # readily -> fewer parameters pass. User-overridable.
MCSE_SD_WARN:   float = 0.10
ACF_LAG50_WARN: float = 0.10
HW_ALPHA:       float = 0.05   # Heidelberger-Welch significance level
# Critical values for the Bartlett-kernel normalized CvM statistic.
# These are EMPIRICAL values calibrated for our Bartlett-S(0) normalization,
# not the batch-means normalization used in H-W (1983) Table 1.
# Computed via Monte Carlo (N=10,000 i.i.d. chains across lengths 200-5000).
HW_CRIT:        float = 0.45   # alpha=0.05  (rejects top 5% of stationary chains)
HW_CRIT_01:     float = 0.68   # alpha=0.01
HW_CRIT_10:     float = 0.35   # alpha=0.10


# ===========================================================================
# Internal helpers
# ===========================================================================

def _as_chains(chains) -> np.ndarray:
    """Coerce input to a float64 [S, C] array.

    Accepts:
      - ndarray [S, C] (multiple chains)
      - ndarray [S]    (single chain → shape [S, 1])
      - list of 1-D arrays of equal length
    """
    if isinstance(chains, np.ndarray):
        if chains.ndim == 1:
            return chains.astype(np.float64)[:, None]
        return chains.astype(np.float64)
    # list of arrays
    arrs = [np.asarray(c, dtype=np.float64).ravel() for c in chains]
    if len(set(len(a) for a in arrs)) != 1:
        raise ValueError(
            "All chains must have the same length; "
            f"got lengths {[len(a) for a in arrs]}."
        )
    return np.column_stack(arrs)


# ---------------------------------------------------------------------------
# Batched core: rank normalization, IMS ESS, R-hat and MCSE for many
# parameters at once.  The scalar functions (rhat, ess, mcse) call these with
# a single parameter, so the report and the scalar API give identical numbers.
# ---------------------------------------------------------------------------

_BATCH = 256          # parameters per FFT block (bounds memory)


def _rank_normalize_cols(flat: np.ndarray) -> np.ndarray:
    """Column-wise Blom rank-normal scores of an [N, d] array (average ranks)."""
    n = flat.shape[0]
    ranks = _stats.rankdata(flat, method="average", axis=0)
    return _ndtri((ranks - 3.0 / 8.0) / (n + 1.0 / 4.0))


def _ims_sums(z: np.ndarray) -> np.ndarray:
    """IMS autocorrelation sums for every chain and parameter of z [S, C, d].

    Same estimator as the scalar version: autocorrelations from the biased
    autocovariance; sum the pairs rho_t + rho_{t+1} for t = 1, 3, 5, ...
    (t < S//2 - 1), stopping before the first negative pair.  Autocovariances
    come from one zero-padded FFT along the draw axis (O(S log S)).
    """
    S = z.shape[0]
    zm = z - z.mean(axis=0, keepdims=True)
    var0 = np.einsum("scd,scd->cd", zm, zm) / S                    # [C, d]
    nfft = _next_fast_len(2 * S - 1)
    f = _rfft(zm, n=nfft, axis=0)
    acov = _irfft(f * np.conj(f), n=nfft, axis=0)[:S]              # [S, C, d], lag-sums
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = acov / acov[0]
    max_lag = min(S // 2, S - 1)
    t = np.arange(1, max_lag - 1, 2)
    if t.size == 0:
        out = np.zeros(var0.shape)
    else:
        pairs = rho[t] + rho[t + 1]                                 # [J, C, d]
        neg = pairs < 0
        keep = ~np.logical_or.accumulate(neg, axis=0)               # before first negative
        out = np.where(keep, pairs, 0.0).sum(axis=0)
    return np.where(var0 < 1e-14, 0.0, out)


def _ess_cols(z: np.ndarray) -> np.ndarray:
    """ESS = N / (1 + 2 mean_c IMS_c), clipped to [1, N], for z [S, C, d]."""
    S, C = z.shape[:2]
    N = S * C
    tau = _ims_sums(z).mean(axis=0)
    return np.clip(N / (1.0 + 2.0 * tau), 1.0, N)


def _batch_diagnostics(chains_3d: np.ndarray, want_rhat: bool = True, want_tail: bool = True):
    """R-hat, bulk ESS, folded ESS, MCSE and MCSE/SD for chains_3d [S, C, d].

    Returns a dict of length-d arrays ('rhat' is None when C < 2 or not
    requested; 'ess_tail' is None when want_tail is False).  Processes the
    parameters in blocks of _BATCH columns.
    """
    arr = np.asarray(chains_3d, dtype=np.float64)
    S, C, d = arr.shape
    out = {k: np.empty(d) for k in ("ess_bulk", "ess_tail", "mcse", "mcse_sd")}
    if not want_tail:
        out["ess_tail"] = None
    do_rhat = want_rhat and C >= 2
    out["rhat"] = np.empty(d) if do_rhat else None
    for lo in range(0, d, _BATCH):
        blk = arr[:, :, lo:lo + _BATCH]
        k = blk.shape[2]
        flat = blk.reshape(S * C, k)                     # row s*C + c, as ravel() of [S, C]
        z_flat = _rank_normalize_cols(flat)
        z = z_flat.reshape(S, C, k)
        bulk = _ess_cols(z)
        out["ess_bulk"][lo:lo + k] = bulk
        if want_tail:
            med = np.median(z_flat, axis=0)
            z_tail = _rank_normalize_cols(np.abs(z_flat - med)).reshape(S, C, k)
            out["ess_tail"][lo:lo + k] = _ess_cols(z_tail)
        sd = flat.std(axis=0)
        out["mcse"][lo:lo + k] = sd / np.sqrt(np.maximum(bulk, 1.0))
        out["mcse_sd"][lo:lo + k] = 1.0 / np.sqrt(np.maximum(bulk, 1.0))
        if do_rhat:
            S2 = S // 2
            zs = np.concatenate([z[:S2], z[S2:2 * S2]], axis=1)    # [S2, 2C, k]
            W = zs.var(axis=0, ddof=1).mean(axis=0)
            B_S = zs.mean(axis=0).var(axis=0, ddof=1)
            var_hat = (S2 - 1) / S2 * W + B_S
            with np.errstate(invalid="ignore", divide="ignore"):
                r = np.sqrt(var_hat / W)
            out["rhat"][lo:lo + k] = np.where(W < 1e-14, 1.0, r)
    return out


# ===========================================================================
# Public statistical functions
# ===========================================================================

def rhat(chains) -> float:
    """Rank-normalized split R-hat (Vehtari et al. 2021).

    Measures convergence across chains.  Values near 1.0 indicate
    convergence; values > 1.01 suggest poor mixing; > 1.05 indicate
    serious non-convergence.

    **Raises ValueError if n_chains == 1.**  R-hat is undefined for a
    single chain — the between-chain variance that forms the numerator
    does not exist.

    Algorithm (Vehtari 2021 §4.3):
      1. Collect all S × C values and rank-normalize: z_i = Φ^{-1}(...)
      2. Reshape z to [S, C] and split each chain in half → [S//2, 2C]
      3. Compute Gelman-Rubin var_hat and W on the split, rank-normalized chains
      4. R-hat = sqrt(var_hat / W)

    Parameters
    ----------
    chains : array-like [S, C] or list of C arrays of length S
        Posterior draws.  Must have C ≥ 2.

    Returns
    -------
    float
        Rank-normalized split R-hat.

    Raises
    ------
    ValueError
        If n_chains == 1 or chains has fewer than 4 samples (insufficient
        for the split).
    """
    arr = _as_chains(chains)
    S, C = arr.shape

    if C < 2:
        raise ValueError(
            "rhat() requires at least 2 chains (n_chains ≥ 2). "
            "R-hat is undefined for a single chain — there is no between-chain "
            "variance.  Run at least 2 chains to use R-hat.  "
            "For single-chain convergence assessment, use geweke() instead."
        )
    if S < 4:
        raise ValueError(
            f"rhat() requires at least 4 samples per chain; got S={S}."
        )

    return float(_batch_diagnostics(arr[:, :, None])["rhat"][0])


def ess(chains) -> tuple[float, float]:
    """Bulk ESS (Vehtari et al. 2021) and a folded ("tail") ESS.

    Returns two effective sample size estimates:
      - **Bulk ESS**: quantifies mixing in the center of the distribution.
        Computed on rank-normalized chains using the IMS ACF estimator, as
        in Vehtari et al. (2021).
      - **Tail (folded) ESS**: the ESS of the folded, rank-normalized draws
        |z_i - median(z)|, which measures how well the SPREAD mixes.  This
        is the folding Vehtari et al. use for their folded R-hat; it is not
        their tail-ESS, which is the minimum ESS of the indicators of the
        5% and 95% quantiles.

    For a single chain (C=1), between-chain variance is absent; ESS is
    computed from within-chain autocorrelation only.

    Parameters
    ----------
    chains : array-like [S, C] or [S] or list of arrays

    Returns
    -------
    (ess_bulk, ess_tail) : tuple[float, float]
    """
    arr = _as_chains(chains)
    out = _batch_diagnostics(arr[:, :, None], want_rhat=False)
    return float(out["ess_bulk"][0]), float(out["ess_tail"][0])


def mcse(chains) -> tuple[float, float]:
    """Monte Carlo Standard Error and MCSE/SD ratio.

    MCSE = SD / sqrt(ESS_bulk)
    MCSE/SD = 1 / sqrt(ESS_bulk)

    The MCSE/SD ratio is scale-free and directly answers: "what fraction
    of the posterior standard deviation is attributable to Monte Carlo
    noise?"  Values > 0.10 indicate that ≥ 10% of the reported SD is
    noise, which is the warning threshold (§10.4).

    Parameters
    ----------
    chains : array-like [S, C] or [S]

    Returns
    -------
    (mcse_val, mcse_sd_ratio) : tuple[float, float]
    """
    arr = _as_chains(chains)
    out = _batch_diagnostics(arr[:, :, None], want_rhat=False, want_tail=False)
    return float(out["mcse"][0]), float(out["mcse_sd"][0])


def geweke(chain, first: float = 0.1, last: float = 0.5) -> tuple[float, float]:
    """Geweke (1992) z-score for stationarity.

    Tests whether the mean of the first ``first`` fraction of the chain
    equals the mean of the last ``last`` fraction, using a Welch-like
    z-test that accounts for each segment's Monte Carlo variance.

    The variance of each segment mean is estimated as:
        var(mean) = S(0) / n_segment

    where S(0) is the spectral density at frequency zero, estimated
    as ``variance * (1 + 2 * IMS_ACF_sum)`` (the sum of autocorrelations
    truncated by the initial monotone sequence, as in the ESS).

    A large |z| (beyond geweke_zcrit(GEWEKE_ALPHA) ≈ 2.97, p < 0.003)
    suggests the chain is non-stationary — typically insufficient warmup
    or a drifting trajectory.

    Parameters
    ----------
    chain : array-like [S] — single chain (not multiple chains)
    first : float, default 0.1  — fraction from start
    last  : float, default 0.5  — fraction from end

    Returns
    -------
    (z, p_value) : tuple[float, float]
        z is the test statistic; p_value is two-sided.

    Raises
    ------
    ValueError
        If the chain is too short or the fractions are invalid.
    """
    x = np.asarray(chain, dtype=np.float64).ravel()
    S = len(x)

    if first + last > 1.0:
        raise ValueError(
            f"first ({first}) + last ({last}) must be ≤ 1.0."
        )
    if S < 20:
        raise ValueError(
            f"Chain must have at least 20 samples for Geweke; got {S}."
        )

    z, p_val = _geweke_cols(x[:, None], first, last)
    return float(z[0]), float(p_val[0])


def _geweke_cols(X: np.ndarray, first: float = 0.1, last: float = 0.5):
    """Geweke z and two-sided p for every column of X [S, d] (one chain each).

    Same statistic as geweke(): segment means compared with variances
    S(0)/n, S(0) = var * (1 + 2 * IMS sum); a segment with variance below
    1e-14 or fewer than 4 draws uses var / n; denominator below 1e-14 gives
    z = 0, p = 1.
    """
    S = X.shape[0]
    n1 = max(1, int(S * first))
    n2 = max(1, int(S * last))

    def _spectral_var(seg):
        n = seg.shape[0]
        var = seg.var(axis=0, ddof=1) if n > 1 else np.full(seg.shape[1], np.nan)
        if n < 4:
            return var / n
        tau = _ims_sums(seg[:, None, :])[0]
        return np.where(var < 1e-14, var / n, var * (1.0 + 2.0 * tau) / n)

    seg1, seg2 = X[:n1], X[S - n2:]
    denom = np.sqrt(_spectral_var(seg1) + _spectral_var(seg2))
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (seg1.mean(axis=0) - seg2.mean(axis=0)) / denom
    small = denom < 1e-14
    z = np.where(small, 0.0, z)
    p = np.where(small, 1.0, 2.0 * (1.0 - _stats.norm.cdf(np.abs(z))))
    return z, p


def _geweke_sweep_cols(X: np.ndarray, alpha: float = 0.003, first: float = 0.1,
                       last: float = 0.5, max_discard: float = 0.5, step: float = 0.1):
    """geweke_sweep() for every column of X [S, d]; returns a list of results.

    At each discard fraction only the columns that have not yet cleared the
    bar are tested, so the cost is one batched Geweke for all columns plus
    one per discard level for the flagged ones.
    """
    S, d = X.shape
    zc = geweke_zcrit(alpha)
    grid, k = [], 0.0
    while k <= max_discard + 1e-9:
        grid.append(round(k, 10))
        k += step
    trajs = [[] for _ in range(d)]
    done = [None] * d
    active = np.arange(d)
    for kk in grid:
        if active.size == 0:
            break
        start = int(kk * S)
        if S - start < 10:
            break
        z, _ = _geweke_cols(X[start:, active], first, last)
        still = []
        for j, zj in zip(active, z, strict=True):
            trajs[j].append((kk, float(zj)))
            if abs(zj) <= zc:
                done[j] = GewekeSweepResult("converged" if kk == 0.0 else "converged_after_burnin",
                                            kk, float(zj), trajs[j])
            else:
                still.append(j)
        active = np.array(still, dtype=int)
    for j in range(d):
        if done[j] is None:
            last_z = trajs[j][-1][1] if trajs[j] else float("nan")
            done[j] = GewekeSweepResult("not_converged", None, last_z, trajs[j])
    return done


# ---------------------------------------------------------------------------
# The meaning of the per-parameter status labels — one definition, reused by
# every rendering surface (console, HTML report, dataframe caption).
# ---------------------------------------------------------------------------

STATUS_LEGEND_LINES = (
    "PASS (converged): stationary with no burn-in needed, adequate mixing.",
    "WARN (usable, but check): at worst under-warmed (a recommended burn-in is "
    "given), OR the two stationarity tests (Geweke & Heidelberger-Welch) disagree (one passes, one flags), OR "
    "mixing/precision is marginal (low ESS, high MCSE, R-hat slightly > 1).",
    "FAIL (did not converge): fails BOTH stationarity tests (Geweke & "
    "Heidelberger-Welch), or R-hat exceeds its hard threshold. A parameter that "
    "passes EITHER stationarity test is a WARN, not a FAIL.",
)

STATUS_LEGEND = "\n".join(STATUS_LEGEND_LINES)


def geweke_zcrit(alpha: float) -> float:
    """Two-sided normal |z| cutoff for a Geweke test at significance ``alpha``.

    LARGER alpha  ->  SMALLER cutoff  ->  rejects stationarity more readily
    ->  fewer parameters pass.
        alpha=0.05  -> |z| > 1.96
        alpha=0.003 -> |z| > ~2.97   (the package default; strict, to limit
                       false flags when scanning hundreds of parameters)
    """
    return float(_stats.norm.ppf(1.0 - alpha / 2.0))


@dataclass
class GewekeSweepResult:
    """Result of a progressive-discard Geweke sweep for one parameter.

    status : str
        'converged'              — stationary with no discard (k=0 passes).
        'converged_after_burnin' — flagged at k=0 but passes after discarding a
                                   leading fraction; ``recommended_discard`` says
                                   how much.
        'not_converged'          — still flagged after discarding ``max_discard``.
    recommended_discard : float or None
        The first discard fraction k at which the parameter passes the SAME test
        that flagged it (alpha). None if it never passes.
    z_at_stop : float
        The Geweke z at the stopping point (the passing z, or the last z tried).
    trajectory : list[tuple[float, float]]
        (fraction_discarded, z) for every point actually computed. The sweep
        stops early at the first pass, so a converged parameter has one point.
    """
    status:               str
    recommended_discard:  float | None
    z_at_stop:            float
    trajectory:           list[tuple[float, float]]


def geweke_sweep(
    chain,
    alpha:       float = 0.003,
    first:       float = 0.1,
    last:        float = 0.5,
    max_discard: float = 0.5,
    step:        float = 0.1,
) -> GewekeSweepResult:
    """Progressive-discard Geweke diagnostic (the "sequential Geweke" idea).

    Recompute the Geweke z while dropping progressively more of the FRONT of the
    chain, keeping the two comparison windows at their same fractions of what
    remains. The trajectory distinguishes three cases the single fixed-window z
    cannot:

      - passes at k=0                         -> 'converged'
      - fails at k=0 but passes at some k>0   -> 'converged_after_burnin' (the
                                                 chain just needs more burn-in;
                                                 recommended_discard = that k)
      - fails through k=max_discard           -> 'not_converged'

    The stabilization bar is the SAME ``alpha`` used to flag (one knob, self-
    consistent): "discard front until the parameter clears the very test that
    flagged it".

    EARLY EXIT: the sweep stops at the first pass, so a parameter that is
    stationary at k=0 costs a single z; only genuinely-flagged parameters pay for
    the deeper discards.

    NOISE CAVEAT: discarding more shrinks the tested tail, so later points rest
    on less data and are noisier. The sweep therefore stops at ``max_discard``
    (default 0.5, matching heidelberger_welch's own cap) rather than going deeper.
    """
    x = np.asarray(chain, dtype=np.float64).ravel()
    return _geweke_sweep_cols(x[:, None], alpha, first, last, max_discard, step)[0]



def acf(chain, lags: Sequence[int] = (1, 5, 10, 50)) -> dict[int, float]:
    """Sample autocorrelation at specified lags.

    Uses the dot-product estimator (biased denominator n, not n-lag),
    consistent with the approach used in the ESS computation.

    ACF(lag) = Cov(x_t, x_{t+lag}) / Var(x)
             = (1/n) Σ_{t=1}^{n-lag} (x_t - x̄)(x_{t+lag} - x̄) / Var_biased(x)

    Parameters
    ----------
    chain : array-like [S] — single chain
    lags  : sequence of int — lags to compute (default: 1, 5, 10, 50)

    Returns
    -------
    dict[int, float]
        Maps each lag to its ACF value.  Returns NaN for lags ≥ chain length.
    """
    x  = np.asarray(chain, dtype=np.float64).ravel()
    n  = len(x)
    xm = x - x.mean()
    var = float(np.dot(xm, xm)) / n  # biased variance

    result: dict[int, float] = {}
    for lag in lags:
        if lag == 0:
            result[lag] = 1.0
        elif lag >= n:
            result[lag] = float('nan')
        elif var < 1e-14:
            result[lag] = 0.0
        else:
            result[lag] = float(np.dot(xm[lag:], xm[:-lag])) / (n * var)
    return result


# ===========================================================================
# DiagnosticReport dataclass
# ===========================================================================

def heidelberger_welch(
    chain,
    alpha:       float = 0.05,
    max_discard: float = 0.50,
    step:        float = 0.10,
) -> tuple[bool, int | None]:
    """Heidelberger & Welch (1983) stationarity test with progressive discard.

    Tests whether a single chain can be considered stationary using the
    Cramer-von Mises statistic on the partial-sum process, normalized by
    the spectral density at frequency zero.

    ALGORITHM (Heidelberger & Welch 1983)
    --------------------------------------
    1. Compute S(0): spectral density at frequency zero via the Bartlett
       kernel with bandwidth M = floor(sqrt(n)):

           S(0) = gamma(0) + 2 * Σ_{k=1}^{M} (1 - k/(M+1)) * gamma(k)

       where gamma(k) = (1/n) Σ_{t=1}^{n-k} (x_t - x̄)(x_{t+k} - x̄).

    2. Compute partial sums: S_i = Σ_{t=1}^{i} (x_t - x̄).

    3. Cramer-von Mises statistic:

           T = (1 / (n² × S(0))) × Σ_{i=1}^{n} S_i²

       Under stationarity T converges to ∫_0^1 B(t)^2 dt for a Brownian
       bridge B (the Cramér-von Mises limit; 95% quantile 0.461).  The test
       uses the module's empirical critical values for this normalization,
       HW_CRIT = 0.45 at alpha = 0.05 (HW_CRIT_01 = 0.68, HW_CRIT_10 = 0.35),
       so ``alpha`` must be 0.01, 0.05 or 0.10.

    4. Progressive discard: if the test fails, discard the first ``step``
       fraction of the chain and retest.  Continue up to ``max_discard``.
       Stop at the first pass; report how many observations were discarded.
       If the chain still fails after discarding ``max_discard``, return
       (False, None) — cannot achieve stationarity.

    WHY THIS COMPLEMENTS GEWEKE
    ---------------------------
    Geweke tests only the mean (first vs last segment).  A chain can have
    a stable mean while its variance or higher-order moments still drift —
    exactly the signature of slow mixing in strongly autocorrelated chains.
    The Cramér-von Mises statistic on the full partial-sum process detects
    this drift pattern regardless of which moment is non-stationary.

    Parameters
    ----------
    chain       : array-like [S], single chain
    alpha       : float, significance level: 0.01, 0.05 (default) or 0.10
    max_discard : float ∈ (0, 1), maximum fraction to discard (default 0.50)
    step        : float ∈ (0, 1), discard increment per retry (default 0.10)

    Returns
    -------
    (passed, n_discarded) : tuple[bool, int or None]
        passed        — True if the chain passes stationarity at ``alpha``.
        n_discarded   — Number of leading observations discarded before the
                        test passed.  0 if the chain passed immediately.
                        None if the chain never passed (FAIL).

    Raises
    ------
    ValueError
        If the chain has fewer than 10 observations.

    References
    ----------
    Heidelberger, P. & Welch, P.D. (1983). Simulation run length control in
        the presence of an initial transient. Operations Research, 31(6),
        1109–1144.
    """
    x = np.asarray(chain, dtype=np.float64).ravel()
    n_total = len(x)

    if n_total < 10:
        raise ValueError(
            f"heidelberger_welch() requires at least 10 observations; got {n_total}."
        )

    # Critical value lookup
    # Empirical critical values for our Bartlett-S(0) normalized CvM statistic.
    # See module-level HW_CRIT constants for derivation details.
    crit_table = {0.01: HW_CRIT_01, 0.05: HW_CRIT, 0.10: HW_CRIT_10}
    crit = crit_table.get(alpha)
    if crit is None:
        raise ValueError(
            f"alpha must be one of {list(crit_table.keys())}; got {alpha}."
        )

    # Progressive discard loop
    n_steps = int(round(max_discard / step))
    fractions = [0.0] + [step * k for k in range(1, n_steps + 1)]

    for frac in fractions:
        n_discard = int(n_total * frac)
        sub = x[n_discard:]
        if len(sub) < 10:
            break

        T = _hw_statistic(sub)
        if T <= crit:
            return True, n_discard

    return False, None


def _hw_statistic(x: np.ndarray) -> float:
    """Cramér-von Mises H-W statistic for a 1-D chain.

    T = (1 / (n² × S(0))) × Σ_{i=1}^{n} S_i²

    where S_i is the i-th partial sum of (x_t - x̄) and S(0) is the
    spectral density at frequency zero estimated via the Bartlett kernel.
    """
    n   = len(x)
    s0  = _bartlett_s0(x)
    xm  = x - x.mean()
    partial_sums = np.cumsum(xm)
    return float(np.sum(partial_sums ** 2) / (n ** 2 * s0))


def _bartlett_s0(x: np.ndarray) -> float:
    """Spectral density at frequency zero via the Bartlett kernel.

    S(0) = gamma(0) + 2 × Σ_{k=1}^{M} (1 - k/(M+1)) × gamma(k)

    Bandwidth: M = floor(sqrt(n)).  This is the Bartlett (1950) lag window
    with the Newey-West (1987) triangular taper.  The same spectral density
    estimate is shared with the ESS and Geweke computations via the IMS
    estimator, keeping the two estimates consistent.

    Returns the spectral density, clamped to a minimum of 1e-14 to avoid
    division by zero when the chain is (nearly) constant.
    """
    n = len(x)
    M = max(1, int(np.sqrt(n)))
    xm = x - x.mean()

    # Biased autocovariances gamma(0) ... gamma(M)
    gamma = np.empty(M + 1)
    gamma[0] = float(np.dot(xm, xm)) / n
    for k in range(1, M + 1):
        gamma[k] = float(np.dot(xm[k:], xm[:-k])) / n

    # Bartlett weights: w_k = 1 - k/(M+1)
    k_arr   = np.arange(1, M + 1, dtype=np.float64)
    weights = 1.0 - k_arr / (M + 1.0)
    s0      = gamma[0] + 2.0 * float(np.dot(weights, gamma[1:]))
    return float(max(s0, 1e-14))


def _is_notebook() -> bool:
    """Return True when running inside a Jupyter notebook kernel."""
    try:
        from IPython import get_ipython
        shell = get_ipython()
        if shell is None:
            return False
        return type(shell).__name__ == 'ZMQInteractiveShell'
    except ImportError:
        return False



# ===========================================================================
# Parameter names -> kind / node / index
# ===========================================================================

_NAME_RULES = [
    # (regex, kind, node group, index group, sub group)
    (re.compile(r"^theta_(?P<node>[^_]+)_(?P<index>\d+)$"), "theta"),
    (re.compile(r"^b_(?P<node>[^_]+)_(?P<index>\d+)_thresh_(?P<sub>\d+)$"), "threshold"),
    (re.compile(r"^b_(?P<node>[^_]+)_(?P<index>\d+)_tau(?P<sub>\d+)$"), "step"),
    (re.compile(r"^(?P<kind>[bac])_(?P<node>[^_]+)_(?P<index>\d+)$"), None),
    (re.compile(r"^sigma2_(?P<node>[^_]+)$"), "sigma2"),
    (re.compile(r"^(?P<kind>(?:b|a|beta|kappa|gamma)\.variance)_(?P<node>[^_]+)$"), None),
    (re.compile(r"^rho_(?P<node>\d+_\d+)$"), "rho"),
    (re.compile(r"^(?P<kind>[ab]\.rho)_(?P<node>[^_]+_[^_]+)$"), None),
    (re.compile(r"^Sigma_theta_(?P<node>\d+_\d+)$"), "Sigma_theta"),
    (re.compile(r"^Sigma_(?P<k>[ab])_(?P<node>[^_]+)_(?P<index>\d+_\d+)$"), None),
    (re.compile(r"^(?P<kind>[XQ])_missing_(?P<node>\d+)_(?P<index>\d+)$"), None),
    (re.compile(r"^W_missing_(?P<node>\d+_\d+)_(?P<index>\d+)$"), "W_missing"),
    (re.compile(r"^(?P<kind>beta|kappa|gamma)_(?P<node>[^_]+)_(?P<index>.+)$"), None),
]


def parse_param_name(name: str) -> dict:
    """Split a chain column name into kind, node, index and sub.

    kind  : parameter type (theta, b, a, c, threshold, step, sigma2, rho,
            b.variance, beta, kappa, gamma, X_missing, ...).
    node  : the node label, or for theta / sigma2 the dimension label (the
            '&'-joined nodes that share it), for correlations the pair.
    index : person or item number (int), or covariate name for coefficients.
    sub   : threshold or step number; for missing covariate values the
            feature number (index is then the person, item, or person_item).
    Unknown names get kind 'other'.
    """
    for rx, kind in _NAME_RULES:
        m = rx.match(name)
        if not m:
            continue
        g = m.groupdict()
        if kind is None:
            kind = g.get("kind") or (f"Sigma_{g['k']}" if g.get("k") else "other")
        if g.get("kind") in ("X", "Q"):
            kind = f"{g['kind']}_missing"
        if kind in ("X_missing", "Q_missing", "W_missing"):
            # node -> the unit (person, item, or person_item); index -> feature
            unit, feat = g.get("node"), g.get("index")
            return {"kind": kind, "node": None,
                    "index": int(unit) if unit.isdigit() else unit,
                    "sub": int(feat)}
        idx = g.get("index")
        if idx is not None and idx.isdigit():
            idx = int(idx)
        sub = g.get("sub")
        return {"kind": kind, "node": g.get("node"), "index": idx,
                "sub": int(sub) if sub is not None else None}
    return {"kind": "other", "node": None, "index": None, "sub": None}


def param_info(names):
    """DataFrame with columns kind, node, index, sub, indexed by parameter name."""
    import pandas as pd
    return pd.DataFrame([parse_param_name(n) for n in names], index=pd.Index(names, name="param"),
                        dtype=object)


@dataclass
class DiagnosticReport:
    """Full diagnostic report for one or more MCMC chains.

    Returned by ``convergence_report()``.  Provides both machine-readable
    data (DataFrames) and human-readable output (console summary).

    Attributes
    ----------
    summary_status : str
        Overall status: 'PASS', 'WARN', or 'FAIL'.
    n_chains : int
        Number of chains.
    n_iter : int
        Total iterations per chain.
    n_samples : int
        Stored samples per chain (after thinning).
    param_names : list[str]
        Parameter names corresponding to rows in diagnostic_df.
    chain_accept_rates : list[dict[str, float]]
        MH acceptance rate per block, one dict per chain.
    chain_frozen_at : list[dict[str, int]]
        Iteration at which each block's warm-up adaptation froze, per chain.

    Tables (pandas DataFrames, built on access)
    -------------------------------------------
    overview  : one row: status, chains, iterations, draws, counts, burn-in.
    params    : one row per parameter (traits included): kind, node, index,
                sub, rhat, ess_bulk, ess_tail, mcse, mcse_sd, geweke_z,
                geweke_p, geweke_status, geweke_burnin, hw_passed,
                hw_n_discarded, acf_1, acf_5, acf_10, acf_50, status.
    flagged   : the WARN / FAIL rows of ``params`` with a ``reason`` column.
    by_kind   : one row per parameter kind: counts and worst values.
    sampler   : one row per MH block and chain: acceptance rate, frozen_at.
    ``tables()`` returns them in a dict, ``to_excel(path)`` writes one sheet
    each, ``styled(name)`` returns a colour-coded Styler for display.
    Posterior summaries are ``fit.summary_df()``; the draws ``fit.draws()``.
    flagged_params : list[str]
        Parameters with Status = WARN or FAIL.
    rhat_values : dict[str, float]
        R-hat per parameter (empty if n_chains=1).
    ess_bulk : dict[str, float]
        Bulk ESS per parameter.
    ess_tail : dict[str, float]
        Folded ("tail") ESS per parameter (see ess()).
    mcse_values : dict[str, float]
        MCSE per parameter.
    mcse_sd_values : dict[str, float]
        MCSE/SD ratio per parameter.
    geweke_z : dict[str, float]
        Geweke z per parameter.
    geweke_p : dict[str, float]
        Geweke p-value per parameter.
    acf_values : dict[str, Dict[int, float]]
        ACF at lags [1,5,10,50] per parameter.
    status : dict[str, str]
        'PASS', 'WARN', or 'FAIL' per parameter.
    accept_rates : dict[str, float]
        MH acceptance rates from the sampler (empty for pure Gibbs).
    hw_passed : dict[str, bool]
        Whether each parameter passed H-W stationarity (after any discard).
    hw_n_discarded : dict[str, int or None]
        Number of leading observations discarded before the H-W test passed.
        0 = passed immediately; None = never passed (FAIL).
    geweke_status : dict[str, str]
        Progressive-discard Geweke result per parameter (geweke_sweep):
        'converged', 'converged_after_burnin' or 'not_converged'.
    geweke_burnin : dict[str, float or None]
        Fraction of the stored draws to discard before the parameter passes
        Geweke (0 if it passes as is; None if it never passes).
    n_pass, n_warn, n_fail : int
        Number of parameters with each status.
    n_needs_burnin : int
        Parameters whose Geweke flag clears after a burn-in.
    n_not_converged : int
        Parameters still flagged after the maximum discard.
    recommended_burnin : float
        Largest geweke_burnin among parameters that need one: discarding
        this fraction of the stored draws clears all of them.
    geweke_alpha : float
        Geweke significance level the report was built with; console_summary
        flags |z| > geweke_zcrit(geweke_alpha).
    """
    summary_status:  str
    n_chains:        int
    n_iter:          int
    n_samples:       int
    param_names:     list[str]
    flagged_params:  list[str]
    rhat_values:     dict[str, float]
    ess_bulk:        dict[str, float]
    ess_tail:        dict[str, float]
    mcse_values:     dict[str, float]
    mcse_sd_values:  dict[str, float]
    geweke_z:        dict[str, float]
    geweke_p:        dict[str, float]
    acf_values:      dict[str, dict[int, float]]
    status:          dict[str, str]
    accept_rates:    dict[str, float] = field(default_factory=dict)
    hw_passed:       dict[str, bool]  = field(default_factory=dict)
    hw_n_discarded:  dict[str, int | None] = field(default_factory=dict)
    # --- three-way Geweke stationarity (design changes #2, #3) ---
    geweke_status:   dict[str, str]   = field(default_factory=dict)
    geweke_burnin:   dict[str, float | None] = field(default_factory=dict)
    # --- status distribution instead of a collapsed verdict (design change #1) ---
    n_pass:          int = 0
    n_warn:          int = 0
    n_fail:          int = 0
    n_needs_burnin:  int = 0
    n_not_converged: int = 0
    recommended_burnin: float = 0.0
    geweke_alpha:    float = GEWEKE_ALPHA
    chain_accept_rates: list = field(default_factory=list, repr=False)
    chain_frozen_at:    list = field(default_factory=list, repr=False)

    # ------------------------------------------------------------------
    # Tables
    # ------------------------------------------------------------------
    @staticmethod
    def _pd():
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("pandas is required for the diagnostic tables. "
                              "Install with: pip install pandas") from exc
        return pd

    @property
    def overview(self):
        """One-row table: overall status, run size, status counts, burn-in."""
        pd = self._pd()
        return pd.DataFrame([{
            "status": self.summary_status, "n_chains": self.n_chains,
            "n_iter": self.n_iter, "n_draws_per_chain": self.n_samples,
            "n_params": len(self.param_names), "n_pass": self.n_pass,
            "n_warn": self.n_warn, "n_fail": self.n_fail,
            "n_needs_burnin": self.n_needs_burnin,
            "n_not_converged": self.n_not_converged,
            "recommended_burnin": self.recommended_burnin,
        }])

    @property
    def params(self):
        """One row per parameter (traits included), indexed by name."""
        pd = self._pd()
        nan = float("nan")
        names = self.param_names
        cols = {
            "rhat":           [self.rhat_values.get(n, nan) for n in names],
            "ess_bulk":       [self.ess_bulk.get(n, nan) for n in names],
            "ess_tail":       [self.ess_tail.get(n, nan) for n in names],
            "mcse":           [self.mcse_values.get(n, nan) for n in names],
            "mcse_sd":        [self.mcse_sd_values.get(n, nan) for n in names],
            "geweke_z":       [self.geweke_z.get(n, nan) for n in names],
            "geweke_p":       [self.geweke_p.get(n, nan) for n in names],
            "geweke_status":  [self.geweke_status.get(n) for n in names],
            "geweke_burnin":  [self.geweke_burnin.get(n) for n in names],
            "hw_passed":      [self.hw_passed.get(n, True) for n in names],
            "hw_n_discarded": [self.hw_n_discarded.get(n, 0) for n in names],
        }
        for lag in (1, 5, 10, 50):
            cols[f"acf_{lag}"] = [self.acf_values.get(n, {}).get(lag, nan) for n in names]
        cols["status"] = [self.status.get(n, "PASS") for n in names]
        df = pd.DataFrame(cols, index=pd.Index(names, name="param"))
        return param_info(names).join(df)

    def _reasons(self, row) -> str:
        out = []
        r = row["rhat"]
        if r == r and r > RHAT_FAIL:
            out.append(f"R-hat {r:.3f} > {RHAT_FAIL}")
        elif r == r and r > RHAT_WARN:
            out.append(f"R-hat {r:.3f} > {RHAT_WARN}")
        for col, label in (("ess_bulk", "bulk ESS"), ("ess_tail", "tail ESS")):
            v = row[col]
            if v == v and v < ESS_WARN:
                out.append(f"{label} {v:.0f} < {ESS_WARN}")
        v = row["mcse_sd"]
        if v == v and v > MCSE_SD_WARN:
            out.append(f"MCSE/SD {v:.3f} > {MCSE_SD_WARN}")
        a50 = row["acf_50"]
        if a50 == a50 and abs(a50) > ACF_LAG50_WARN:
            out.append(f"|ACF(50)| {abs(a50):.2f} > {ACF_LAG50_WARN}")
        gs = row["geweke_status"]
        if gs == "converged_after_burnin":
            out.append(f"Geweke: discard first {100 * (row['geweke_burnin'] or 0):.0f}%")
        elif gs == "not_converged":
            out.append("Geweke: not stationary")
        if self.n_chains == 1 and not row["hw_passed"]:
            out.append("Heidelberger-Welch failed")
        return "; ".join(out)

    @property
    def flagged(self):
        """WARN and FAIL rows of ``params``, worst first, with a reason."""
        df = self.params
        df = df[df["status"] != "PASS"].copy()
        df.insert(0, "reason", [self._reasons(r) for _, r in df.iterrows()])
        order = df["status"].map({"FAIL": 0, "WARN": 1})
        return df.assign(_o=order).sort_values(["_o", "kind"]).drop(columns="_o")

    @property
    def by_kind(self):
        """One row per parameter kind: counts and worst values."""
        df = self.params
        g = df.groupby("kind", sort=False)
        out = g.agg(n=("status", "size"),
                    n_pass=("status", lambda x: int((x == "PASS").sum())),
                    n_warn=("status", lambda x: int((x == "WARN").sum())),
                    n_fail=("status", lambda x: int((x == "FAIL").sum())),
                    max_rhat=("rhat", "max"),
                    min_ess_bulk=("ess_bulk", "min"),
                    min_ess_tail=("ess_tail", "min"),
                    max_mcse_sd=("mcse_sd", "max"))
        return out

    @property
    def sampler(self):
        """One row per Metropolis-Hastings block and chain."""
        pd = self._pd()
        rows = []
        for c, rates in enumerate(self.chain_accept_rates):
            frozen = self.chain_frozen_at[c] if c < len(self.chain_frozen_at) else {}
            for block, rate in rates.items():
                rows.append({"chain": c + 1, "block": block, "accept_rate": rate,
                             "frozen_at": frozen.get(block)})
        return pd.DataFrame(rows, columns=["chain", "block", "accept_rate", "frozen_at"])

    def tables(self) -> dict:
        """All tables in a dict: overview, by_kind, params, flagged, sampler."""
        return {"overview": self.overview, "by_kind": self.by_kind,
                "params": self.params, "flagged": self.flagged,
                "sampler": self.sampler}

    def to_excel(self, path: str) -> None:
        """Write every table to one sheet of an Excel file (needs openpyxl)."""
        pd = self._pd()
        with pd.ExcelWriter(path) as xw:
            for name, df in self.tables().items():
                df.to_excel(xw, sheet_name=name)

    def to_dataframe(self):
        """The ``params`` table (kept for compatibility)."""
        return self.params

    def to_csv(self, path: str) -> None:
        """Write the ``params`` table to a CSV file."""
        self.params.to_csv(path)

    def styled(self, name: str = "params"):
        """A colour-coded pandas Styler of one table, for display only."""
        df = self.tables()[name]

        def _status(v):
            return {"PASS": "background-color: #c8e6c9", "WARN": "background-color: #fff9c4",
                    "FAIL": "background-color: #ffcdd2"}.get(str(v), "")

        def _rhat(v):
            try:
                v = float(v)
            except (TypeError, ValueError):
                return ""
            return ("background-color: #ffcdd2" if v > RHAT_FAIL else
                    "background-color: #fff9c4" if v > RHAT_WARN else "")

        def _low(th):
            def f(v):
                try:
                    return "background-color: #fff9c4" if float(v) < th else ""
                except (TypeError, ValueError):
                    return ""
            return f

        def _high(th):
            def f(v):
                try:
                    return "background-color: #fff9c4" if float(v) > th else ""
                except (TypeError, ValueError):
                    return ""
            return f

        st = df.style
        for col, fn in (("status", _status), ("rhat", _rhat), ("max_rhat", _rhat),
                        ("ess_bulk", _low(ESS_WARN)), ("ess_tail", _low(ESS_WARN)),
                        ("min_ess_bulk", _low(ESS_WARN)), ("min_ess_tail", _low(ESS_WARN)),
                        ("mcse_sd", _high(MCSE_SD_WARN)), ("max_mcse_sd", _high(MCSE_SD_WARN))):
            if col in df.columns:
                st = st.map(fn, subset=[col])
        fmt = {c: "{:.3f}" for c in df.columns if df[c].dtype.kind == "f"}
        for c in ("ess_bulk", "ess_tail", "min_ess_bulk", "min_ess_tail"):
            if c in fmt:
                fmt[c] = "{:.0f}"
        return st.format(fmt, na_rep="")

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------
    def _headline(self) -> str:
        return (f"DiagnosticReport: {self.summary_status} | {len(self.param_names)} parameters, "
                f"{self.n_pass} PASS, {self.n_warn} WARN, {self.n_fail} FAIL | "
                f"{self.n_chains} chain(s) x {self.n_samples} draws"
                + (f" | recommended burn-in {100 * self.recommended_burnin:.0f}%"
                   if self.recommended_burnin else ""))

    _FOOTER = ("Tables: .overview  .by_kind  .params  .flagged  .sampler  "
               "(.tables() for all, .to_excel(path), .styled(name))")

    def __repr__(self) -> str:
        try:
            body = self.by_kind.round(3).to_string()
        except ImportError:
            body = ""
        return "\n".join([self._headline(), "", body, "", self._FOOTER])

    def _repr_html_(self) -> str:
        try:
            body = self.styled("by_kind").to_html()
        except Exception:
            body = ""
        return (f"<p><b>{self._headline()}</b></p>{body}"
                f"<p><code>{self._FOOTER}</code></p>")


# ===========================================================================
# Parameter status classification
# ===========================================================================

def _classify_param(
    name:          str,
    rhat_val:      float | None,
    ess_b:         float,
    ess_t:         float,
    mcse_sd:       float,
    acf_lag50:     float,
    geweke_status: str = "converged",
    hw_passed_v:   bool = True,
) -> str:
    """Return 'PASS', 'WARN', or 'FAIL' for one parameter.

    Stationarity uses the three-way Geweke sweep status:
      - 'converged'              -> no stationarity flag
      - 'converged_after_burnin' -> WARN: the chain is fine but was under-warmed;
                                    the report carries the recommended discard.
                                    (Deliberately NOT a FAIL — it is remediable.)
      - 'not_converged'          -> FAIL: never stabilises across the discard sweep.
    Combined with H-W (which tests the whole distribution, not just the mean):
    H-W failure is also a FAIL, and the two are reported separately so the user
    sees WHICH test flagged and can judge disagreement.
    """
    # --- Stationarity: hard FAIL only when a parameter passes NEITHER test. ---
    # Geweke (post-sweep) and H-W test different things (mean drift vs whole-
    # distribution drift), so one test vouching for stationarity is evidence the
    # chain is usable. A parameter is FAILed only if BOTH reject it; if EITHER
    # passes, disagreement is uncertainty -> WARN, not condemnation.
    geweke_ok = (geweke_status != "not_converged")   # converged or needs-burnin
    stationarity_fail = (not geweke_ok) and (not hw_passed_v)

    # FAIL conditions
    if rhat_val is not None and rhat_val > RHAT_FAIL:
        return "FAIL"
    if stationarity_fail:
        return "FAIL"

    # WARN conditions
    # any stationarity disagreement, or a remediable under-warmup, is a WARN
    if (not geweke_ok) or (not hw_passed_v) or (geweke_status == "converged_after_burnin"):
        return "WARN"
    if rhat_val is not None and rhat_val > RHAT_WARN:
        return "WARN"
    if min(ess_b, ess_t) < ESS_WARN:
        return "WARN"
    if mcse_sd > MCSE_SD_WARN:
        return "WARN"
    if not np.isnan(acf_lag50) and abs(acf_lag50) > ACF_LAG50_WARN:
        return "WARN"

    return "PASS"


# ===========================================================================
# Main assembly: convergence_report
# ===========================================================================

def convergence_report(
    results,
    param_names: list[str] | None = None,
    geweke_alpha: float = GEWEKE_ALPHA,
) -> DiagnosticReport:
    """Compute a full DiagnosticReport from one or more MCMCResult objects.

    Parameters
    ----------
    results : MCMCResult or list[MCMCResult]
        One result per chain.  Pass a list for multi-chain diagnostics.
        If a single MCMCResult is passed, its ``.chain`` is treated as
        one chain of shape [S, d].  If multiple MCMCResult objects are
        passed, their chains are stacked along a new chain axis.
    param_names : list[str] or None
        Override parameter names.  If None, taken from the first result.
    geweke_alpha : float
        Two-sided significance level of the Geweke test (default
        GEWEKE_ALPHA = 0.003, |z| cutoff ~2.97).  Larger values reject more
        readily.

    Returns
    -------
    DiagnosticReport
    """
    # Normalize input — accept a single result or a list
    if not isinstance(results, list):
        results = [results]

    if len(results) == 0:
        raise ValueError("results must contain at least one MCMCResult.")

    # Stack chains: each result.chain is [S, d]; stack → [S, C, d].  In
    # evidence-mode warm-up each chain freezes when its own adaptation has
    # settled, so chains can keep different numbers of draws.  Align them on
    # their last common number of draws, as posterior_summary and
    # IRTreeFit.draws_array do (the earliest retained draws are the ones
    # dropped, so every chain contributes its most-settled draws).
    S_vals = [r.chain.shape[0] for r in results]
    S_min = min(S_vals)
    chains_list = [r.chain[-S_min:].astype(np.float64) for r in results]

    S  = chains_list[0].shape[0]
    C  = len(results)
    d  = chains_list[0].shape[1]
    # chains_3d[s, c, param_idx]
    chains_3d = np.stack(chains_list, axis=1)  # [S, C, d]

    names = param_names or results[0].param_names
    if len(names) != d:
        raise ValueError(
            f"len(param_names)={len(names)} does not match chain width d={d}."
        )

    n_iter   = results[0].n_iter

    ACF_LAGS = (1, 5, 10, 50)

    rhat_vals         = {}
    ess_bulk_d        = {}
    ess_tail_d        = {}
    mcse_d            = {}
    mcse_sd_d         = {}
    geweke_z_d        = {}
    geweke_p_d        = {}
    hw_passed_d       = {}
    hw_n_discarded_d  = {}
    acf_d             = {}
    status_d          = {}
    geweke_status_d   = {}   # 'converged' | 'converged_after_burnin' | 'not_converged'
    geweke_burnin_d   = {}   # recommended discard fraction (None if converged/not)

    # R-hat, ESS and MCSE for every parameter at once (vectorized; the same
    # numbers as rhat(), ess() and mcse() parameter by parameter).
    _bd = _batch_diagnostics(chains_3d, want_rhat=(C >= 2))
    # Geweke (first chain; primary for one chain, supplementary otherwise),
    # the progressive-discard sweep, batched over parameters as well.
    _chain0 = np.asarray(chains_3d[:, 0, :], dtype=np.float64)
    if _chain0.shape[0] < 20:
        raise ValueError(f"Chain must have at least 20 samples for Geweke; got {_chain0.shape[0]}.")
    _gz, _gp = _geweke_cols(_chain0)
    _sweeps = _geweke_sweep_cols(_chain0, alpha=geweke_alpha)

    for idx, name in enumerate(names):
        param_chains = chains_3d[:, :, idx]  # [S, C]

        # --- R-hat (multi-chain only) ---
        if C >= 2:
            rhat_v = float(_bd["rhat"][idx])
            rhat_vals[name] = rhat_v
        else:
            rhat_v = None

        # --- ESS ---
        ess_b, ess_t = float(_bd["ess_bulk"][idx]), float(_bd["ess_tail"][idx])
        ess_bulk_d[name] = ess_b
        ess_tail_d[name] = ess_t

        # --- MCSE ---
        mcse_v, mcse_sd_v = float(_bd["mcse"][idx]), float(_bd["mcse_sd"][idx])
        mcse_d[name]    = mcse_v
        mcse_sd_d[name] = mcse_sd_v

        # --- Geweke (on concatenated chain for multi-chain, first chain for single) ---
        geweke_z_d[name] = float(_gz[idx])
        geweke_p_d[name] = float(_gp[idx])
        # progressive-discard sweep: distinguishes "needs burn-in" from "broken".
        _sweep = _sweeps[idx]
        geweke_status_d[name] = _sweep.status
        geweke_burnin_d[name] = _sweep.recommended_discard

        # --- ACF (first chain; representative) ---
        acf_d[name] = acf(param_chains[:, 0], lags=ACF_LAGS)

        # --- Heidelberger-Welch (first chain; primary for n_chains=1) ---
        try:
            hw_pass, hw_disc = heidelberger_welch(param_chains[:, 0])
        except Exception:
            hw_pass, hw_disc = True, 0   # don't let H-W crash the report
        hw_passed_d[name]      = hw_pass
        hw_n_discarded_d[name] = hw_disc

        # --- Status ---
        acf50 = acf_d[name].get(50, float('nan'))
        status_d[name] = _classify_param(
            name, rhat_v, ess_b, ess_t, mcse_sd_v, acf50,
            geweke_status=geweke_status_d[name],
            hw_passed_v=(hw_pass if C == 1 else True),
        )

    flagged = [n for n in names if status_d.get(n, "PASS") != "PASS"]

    # --- Counts, not a collapsed verdict (design change #1). ---
    # The old rule made the WHOLE run FAIL if ANY single parameter failed. With
    # hundreds/thousands of parameters tested at a per-parameter alpha, a few
    # flags are EXPECTED by chance even under perfect convergence, so an
    # any-fail-fails-all verdict is misleading. We instead report the
    # DISTRIBUTION of per-parameter statuses and let the user judge.
    statuses = list(status_d.values())
    n_pass  = statuses.count("PASS")
    n_warn  = statuses.count("WARN")
    n_fail  = statuses.count("FAIL")

    # Post-sweep breakdown of stationarity: how many flags are just under-warmup
    # (remediable) vs genuinely unresolved.
    n_needs_burnin = sum(1 for v in geweke_status_d.values()
                         if v == "converged_after_burnin")
    n_not_converged = sum(1 for v in geweke_status_d.values()
                          if v == "not_converged")
    # a recommended global burn-in: the largest per-parameter recommended discard
    # among parameters that needed one (so discarding this much clears them all).
    _burnins = [b for b in geweke_burnin_d.values() if b is not None and b > 0]
    recommended_burnin = max(_burnins) if _burnins else 0.0

    # `summary_status` is ADVISORY, not a gate.
    # A per-parameter FAIL already means "failed R-hat, OR failed BOTH stationarity
    # tests" (see _classify_param) — a parameter that passes EITHER Geweke or H-W
    # is at worst a WARN. So the run-level label just reads off the counts:
    #   FAIL if any parameter is a hard FAIL; else WARN if any WARN; else PASS.
    if n_fail > 0:
        overall = "FAIL"
    elif n_warn > 0:
        overall = "WARN"
    else:
        overall = "PASS"

    # Acceptance rates: per chain, and their mean over the chains that have
    # the block (the earlier running pairwise average weighted later chains more).
    chain_accept = [dict(getattr(r, "accept_rates", {}) or {}) for r in results]
    chain_frozen = [dict(getattr(r, "frozen_at", {}) or {}) for r in results]
    accept_rates: dict[str, float] = {}
    for k in dict.fromkeys(k for ca in chain_accept for k in ca):
        vals = [ca[k] for ca in chain_accept if k in ca]
        accept_rates[k] = float(np.mean(vals))

    return DiagnosticReport(
        summary_status=overall,
        n_chains=C,
        n_iter=n_iter,
        n_samples=S,
        param_names=names,
        flagged_params=flagged,
        rhat_values=rhat_vals,
        ess_bulk=ess_bulk_d,
        ess_tail=ess_tail_d,
        mcse_values=mcse_d,
        mcse_sd_values=mcse_sd_d,
        geweke_z=geweke_z_d,
        geweke_p=geweke_p_d,
        acf_values=acf_d,
        status=status_d,
        accept_rates=accept_rates,
        chain_accept_rates=chain_accept,
        chain_frozen_at=chain_frozen,
        hw_passed=hw_passed_d,
        hw_n_discarded=hw_n_discarded_d,
        geweke_status=geweke_status_d,
        geweke_burnin=geweke_burnin_d,
        n_pass=n_pass,
        n_warn=n_warn,
        n_fail=n_fail,
        n_needs_burnin=n_needs_burnin,
        n_not_converged=n_not_converged,
        geweke_alpha=geweke_alpha,
        recommended_burnin=recommended_burnin,
    )


# ===========================================================================
# Layer 1: console_summary
# ===========================================================================

def console_summary(report: DiagnosticReport) -> None:
    """Print the Layer 1 traffic-light console summary.

    Called
    automatically by ``fit()``; also callable directly on a
    DiagnosticReport.

    Parameters
    ----------
    report : DiagnosticReport
    """
    C = report.n_chains
    S = report.n_samples
    n_iter = report.n_iter
    total_draws = S * C

    # Header
    print(f"\nirtree fit complete  "
          f"({C} chain{'s' if C > 1 else ''} × {n_iter} iterations, "
          f"{total_draws} posterior draws)")

    # --- Status distribution, not a single verdict (design change #1). ---
    n_tot = report.n_pass + report.n_warn + report.n_fail
    print(f"   parameters: {report.n_pass} PASS · {report.n_warn} WARN · "
          f"{report.n_fail} FAIL   (of {n_tot})")
    if report.n_needs_burnin:
        pct = int(round(100 * report.recommended_burnin))
        print(f"   {report.n_needs_burnin} flagged only for under-warmup — "
              f"resolve by discarding the first ~{pct}% (recommended_burnin).")
    if report.n_not_converged:
        print(f"   {report.n_not_converged} parameter(s) did NOT converge even "
              f"after discarding — inspect these.")
    print("   (overall label is advisory; the per-parameter table is the "
          "decision. Call fit.diagnostics().)")
    print("   status key:")
    for _ln in STATUS_LEGEND_LINES:
        print(f"     {_ln}")

    if C == 1:
        print("⚠  Single chain — R-hat unavailable. "
              "Stationarity via Geweke + Heidelberger-Welch.")

        # Stationarity (Geweke, three-way sweep): separate "needs burn-in"
        # (remediable, WARN) from "never converges" (FAIL). Reported apart from
        # H-W so the user sees which test flagged and can judge disagreement.
        gew_burnin = [n for n in report.param_names
                      if report.geweke_status.get(n) == "converged_after_burnin"]
        gew_noconv = [n for n in report.param_names
                      if report.geweke_status.get(n) == "not_converged"]
        if gew_noconv:
            detail = (f"{len(gew_noconv)} never converged "
                      f"[{', '.join(gew_noconv[:5])}{'...' if len(gew_noconv) > 5 else ''}]"
                      f"; {len(gew_burnin)} more need burn-in")
        elif gew_burnin:
            detail = (f"{len(gew_burnin)} need burn-in "
                      f"[{', '.join(gew_burnin[:5])}{'...' if len(gew_burnin) > 5 else ''}]")
        else:
            detail = None
        _print_line("Stationarity", "Geweke", len(gew_noconv) == 0, detail)

        # Stationarity (Heidelberger-Welch — primary for single chain)
        hw_fail = [n for n in report.param_names
                   if not report.hw_passed.get(n, True)]
        _print_line(
            "Stationarity", "H-W   ",
            len(hw_fail) == 0,
            f"{len(hw_fail)} parameters failed  "
            f"[{', '.join(hw_fail[:5])}{'...' if len(hw_fail) > 5 else ''}]"
            if hw_fail else None,
        )

    else:
        # Convergence (R-hat)
        rhat_warn = [n for n in report.param_names
                     if report.rhat_values.get(n, 1.0) > RHAT_WARN]
        rhat_fail = [n for n in report.param_names
                     if report.rhat_values.get(n, 1.0) > RHAT_FAIL]

        if rhat_fail:
            mark = "✕"
            detail = (f"{len(rhat_fail)} parameters R-hat > {RHAT_FAIL}  "
                      f"[{', '.join(rhat_fail[:5])}{'...' if len(rhat_fail)>5 else ''}]"
                      f"  [call fit.diagnostics()]")
        elif rhat_warn:
            mark = "⚠"
            detail = (f"{len(rhat_warn)} parameters R-hat > {RHAT_WARN}  "
                      f"[{', '.join(rhat_warn[:5])}{'...' if len(rhat_warn)>5 else ''}]"
                      f"  [call fit.diagnostics()]")
        else:
            mark = "✓"
            detail = None
        _print_line("Convergence ", "R-hat  ", mark == "✓", detail, mark=mark)

        # Stationarity (Geweke — supplementary)
        _zc = geweke_zcrit(report.geweke_alpha)
        geweke_fail = [n for n in report.param_names
                       if abs(report.geweke_z.get(n, 0.0)) > _zc]
        _print_line(
            "Stationarity", "Geweke",
            len(geweke_fail) == 0,
            f"{len(geweke_fail)} parameters failed  "
            f"[{', '.join(geweke_fail[:5])}{'...' if len(geweke_fail) > 5 else ''}]"
            if geweke_fail else None,
        )

    # Mixing (ESS)
    ess_warn = [n for n in report.param_names
                if report.ess_bulk.get(n, float('inf')) < ESS_WARN]
    _print_line(
        "Mixing      ", "ESS    ",
        len(ess_warn) == 0,
        f"{len(ess_warn)} parameters ESS < {ESS_WARN}  [call fit.diagnostics()]"
        if ess_warn else None,
    )

    # Precision (MCSE/SD)
    mcse_warn = [n for n in report.param_names
                 if report.mcse_sd_values.get(n, 0.0) > MCSE_SD_WARN]
    _print_line(
        "Precision   ", "MCSE/SD",
        len(mcse_warn) == 0,
        f"{len(mcse_warn)} parameters MCSE/SD > {MCSE_SD_WARN}"
        if mcse_warn else None,
    )

    if C > 1:
        print("\nRun fit.diagnostics() for the full diagnostic table.")
        print("Run fit.report()      to save a self-contained HTML report.")


def _print_line(
    dimension: str,
    metric:    str,
    passed:    bool,
    detail:    str | None,
    mark:      str | None = None,
) -> None:
    """Print one traffic-light line of the console summary."""
    if mark is None:
        mark = "✓" if passed else "✕"
    label = f"{mark} {dimension} ({metric}): "
    if passed:
        print(f"{label}  All {metric.strip()} within thresholds")
    else:
        print(f"{label}  {detail}")
