"""
tests/test_module_outputs.py  (diagnostics + posterior)
=======================================================
Axis-1 output checks for `diagnostics` and `posterior`, against **constructed
input with a known answer** — no model fit. This is the deterministic half of
Axis 1 for these two modules (TESTING.md §4, "diagnostics / posterior" rows).

The design rule: separate "does it compute the statistic right" (here, on chains
we built with a known R-hat / ESS / mean) from "did a real chain converge"
(Axis 2). We never fit a model in this file.
"""

import numpy as np
import pytest

import irtree.diagnostics as dg
import irtree.mcmc_sampler as ms
import irtree.posterior as po

pytestmark = pytest.mark.fast


# ---------------------------------------------------------------------------
# constructed-chain builders (the "known answer" generators)
# ---------------------------------------------------------------------------

def _ar1(n, rho, seed):
    """A single AR(1) chain with lag-1 autocorrelation rho and unit marginal var.
    Its ESS/N ratio is analytically ~ (1-rho)/(1+rho)."""
    r = np.random.default_rng(seed)
    x = np.zeros(n)
    innov_sd = np.sqrt(1.0 - rho ** 2)
    for t in range(1, n):
        x[t] = rho * x[t - 1] + r.normal(0.0, innov_sd)
    return x


def _fake_result(chain_2d, names):
    """Minimal MCMCResult carrying a known [S, d] chain, for posterior_summary."""
    S = chain_2d.shape[0]
    return ms.MCMCResult(
        chain=chain_2d.astype(np.float32),
        param_names=list(names),
        log_lik_trace=np.zeros(S),
        accept_rates={},
        n_warmup=0, n_iter=S, final_params={}, rng_state={},
    )


# ===========================================================================
# diagnostics — R-hat against constructed chains
# ===========================================================================

def test_rhat_identical_chains_is_one():
    """Four identical chains (zero between-chain variance) → R-hat ≈ 1."""
    N = 3000
    one = np.random.default_rng(1).normal(size=N)
    chains = np.column_stack([one, one, one, one])   # [S, C]
    r = dg.rhat(chains)
    assert abs(r - 1.0) < 0.02, f"identical chains gave R-hat {r:.4f} (expected ≈1)"


def test_rhat_divergent_chains_is_large():
    """Chains with means 0,4,8,12 (huge between-chain variance) → R-hat ≫ 1."""
    N = 3000
    chains = np.column_stack([
        np.random.default_rng(s).normal(4.0 * s, 1.0, N) for s in range(4)
    ])
    r = dg.rhat(chains)
    assert r > 1.5, f"divergent chains gave R-hat {r:.3f} (expected ≫ 1)"


def test_rhat_wellmixed_near_one():
    """Four independent draws from the same target → R-hat ≈ 1 (below 1.05)."""
    N = 4000
    chains = np.column_stack([
        np.random.default_rng(s).normal(0.0, 1.0, N) for s in range(4)
    ])
    r = dg.rhat(chains)
    assert r < 1.05, f"well-mixed chains gave R-hat {r:.4f} (expected < 1.05)"


# ===========================================================================
# diagnostics — ESS against a known autocorrelation
# ===========================================================================

def test_ess_white_noise_near_n():
    """Independent draws → ESS ≈ N (ratio near 1)."""
    N = 4000
    chains = np.column_stack([
        np.random.default_rng(s).normal(size=N) for s in range(4)
    ])
    ess, _ = dg.ess(chains)
    ratio = ess / (N * 4)
    assert 0.8 < ratio < 1.15, f"white-noise ESS ratio {ratio:.3f} (expected ≈1)"


def test_ess_ar1_matches_theory():
    """AR(1) with rho=0.5 → ESS/N ≈ (1-rho)/(1+rho) = 1/3."""
    N, rho = 4000, 0.5
    chains = np.column_stack([_ar1(N, rho, s) for s in range(4)])
    ess, _ = dg.ess(chains)
    ratio = ess / (N * 4)
    theory = (1 - rho) / (1 + rho)          # 0.333
    assert abs(ratio - theory) < 0.06, (
        f"AR(1) ESS ratio {ratio:.3f} vs theory {theory:.3f}"
    )


def test_ess_autocorr_reduces_ess():
    """More autocorrelation → strictly fewer effective samples."""
    N = 4000
    lo = np.column_stack([_ar1(N, 0.2, s) for s in range(4)])
    hi = np.column_stack([_ar1(N, 0.8, s) for s in range(4)])
    ess_lo, _ = dg.ess(lo)
    ess_hi, _ = dg.ess(hi)
    assert ess_hi < ess_lo, (
        f"higher autocorrelation should lower ESS (got {ess_hi:.0f} ≥ {ess_lo:.0f})"
    )


# ===========================================================================
# diagnostics — mcse, geweke, acf, heidelberger_welch sanity on known input
# ===========================================================================

def test_mcse_shrinks_with_n():
    """MCSE of the mean ≈ sd/sqrt(ESS); doubling N should shrink it ~1/sqrt2."""
    small = np.column_stack([np.random.default_rng(s).normal(size=2000) for s in range(4)])
    large = np.column_stack([np.random.default_rng(s).normal(size=8000) for s in range(4)])
    m_small, _ = dg.mcse(small)
    m_large, _ = dg.mcse(large)
    assert m_large < m_small, "MCSE should shrink with more samples"


def test_acf_lag1_recovers_rho():
    """ACF at lag 1 of an AR(1) chain ≈ rho."""
    x = _ar1(20000, 0.6, 0)
    a = dg.acf(x, lags=[1])
    assert abs(a[1] - 0.6) < 0.05, f"lag-1 ACF {a[1]:.3f} vs rho 0.6"


def test_geweke_passes_on_stationary_chain():
    """A stationary chain should not be flagged by Geweke (|z| small)."""
    x = np.random.default_rng(0).normal(size=5000)
    z, p = dg.geweke(x)
    assert abs(z) < 3.0, f"Geweke z={z:.2f} flagged a stationary chain"


def test_heidelberger_welch_passes_on_stationary():
    """Heidelberger-Welch should pass a stationary chain (no discard needed)."""
    x = np.random.default_rng(0).normal(size=5000)
    passed, discard = dg.heidelberger_welch(x)
    assert passed, "H-W failed a stationary chain"


# ===========================================================================
# posterior — summaries against a chain with a known target
# ===========================================================================

def test_posterior_summary_recovers_known_moments():
    """A chain of N(2, 1.5^2) draws → mean≈2, sd≈1.5, 2.5%/97.5% ≈ 2 ∓ 1.96*1.5."""
    N = 60000
    x = np.random.default_rng(0).normal(2.0, 1.5, size=(N, 1))
    res = _fake_result(x, ["x"])
    s = po.posterior_summary(res)["x"]
    assert abs(s.mean - 2.0) < 0.03, f"mean {s.mean:.3f}"
    assert abs(s.sd - 1.5) < 0.03, f"sd {s.sd:.3f}"
    assert abs(s.q025 - (2.0 - 1.96 * 1.5)) < 0.06, f"q025 {s.q025:.3f}"
    assert abs(s.q975 - (2.0 + 1.96 * 1.5)) < 0.06, f"q975 {s.q975:.3f}"


def test_credible_interval_eti_width():
    """95% equal-tailed interval of N(0,1) ≈ (-1.96, 1.96)."""
    N = 60000
    x = np.random.default_rng(1).normal(0.0, 1.0, size=(N, 1))
    res = _fake_result(x, ["x"])
    lo, hi = po.credible_interval(res, level=0.95, method="eti")["x"]
    assert abs(lo + 1.96) < 0.05 and abs(hi - 1.96) < 0.05, f"ETI ({lo:.3f}, {hi:.3f})"


def test_hdi_narrower_than_eti_for_skewed():
    """For a skewed posterior the HDI is no wider than the ETI (it is the
    shortest interval of the given mass). Use a chi-square-like sample."""
    N = 60000
    x = np.random.default_rng(2).gamma(shape=2.0, scale=1.0, size=(N, 1))
    res = _fake_result(x, ["x"])
    lo_h, hi_h = po.credible_interval(res, level=0.95, method="hdi")["x"]
    lo_e, hi_e = po.credible_interval(res, level=0.95, method="eti")["x"]
    assert (hi_h - lo_h) <= (hi_e - lo_e) + 0.05, (
        f"HDI width {hi_h-lo_h:.3f} should be ≤ ETI width {hi_e-lo_e:.3f}"
    )


def test_posterior_summary_multiparam():
    """Two independent params summarized together keep their own moments."""
    N = 40000
    rng = np.random.default_rng(3)
    x = np.column_stack([rng.normal(-1.0, 0.5, N), rng.normal(3.0, 2.0, N)])
    res = _fake_result(x, ["a", "b"])
    s = po.posterior_summary(res)
    assert abs(s["a"].mean + 1.0) < 0.03 and abs(s["a"].sd - 0.5) < 0.03
    assert abs(s["b"].mean - 3.0) < 0.05 and abs(s["b"].sd - 2.0) < 0.05
