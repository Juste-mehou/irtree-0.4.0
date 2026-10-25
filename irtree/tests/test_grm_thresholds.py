"""Exact Albert-Chib update of GRM thresholds.

Given Z, tau enters the GRM only through the category indicators, so the
full conditional of tau_{i,k} is its prior truncated to
(max(tau_{k-1}, max Z in category k), min(tau_{k+1}, min Z in category k+1)).
"""
import numpy as np
import pytest
from scipy import stats
from scipy.stats import norm

from irtree import gibbs_samplers as gs


def test_truncnorm_draw_matches_scipy_including_far_tails():
    rng = np.random.default_rng(0)
    for lo, hi in [(-1.0, 0.5), (8.0, 9.0), (-40.0, -39.0), (2.0, np.inf),
                   (-np.inf, -3.0), (-np.inf, np.inf)]:
        x = gs.truncnorm_draw(np.zeros(40000), np.ones(40000), lo, hi, rng)
        ref = stats.truncnorm(lo, hi)
        assert np.all((x >= lo) & (x <= hi))
        assert abs(x.mean() - ref.mean()) < 0.02 * max(1.0, ref.std() * 5)
        assert abs(x.std() - ref.std()) < 0.03 * max(ref.std(), 0.05) + 0.005


@pytest.mark.slow
def test_thresholds_without_data_follow_ordered_prior():
    """No data: tau_i is K iid N(mu0, tau2) draws conditioned on order, i.e.
    their order statistics.  The old update never moved such thresholds."""
    rng = np.random.default_rng(1)
    I, M = 1, 4
    K = M - 1
    Z = np.zeros((1, I))
    Y = np.zeros((1, I))
    empty = np.array([], dtype=np.intp)
    tau = np.array([[-1.0, 0.0, 1.0]])
    draws = np.empty((40000, K))
    for t in range(len(draws)):
        tau = gs.gibbs_thresholds_grm(Z, Y, np.zeros((1, I), bool), empty, empty,
                                      M, rng, old_thresholds=tau, mu_0=0.5, tau2=2.0)
        draws[t] = tau[0]
    ref = np.sort(rng.normal(0.5, np.sqrt(2.0), (200000, K)), axis=1)
    np.testing.assert_allclose(draws[2000:].mean(0), ref.mean(0), atol=0.05)
    np.testing.assert_allclose(draws[2000:].std(0), ref.std(0), atol=0.05)


@pytest.mark.slow
def test_data_augmentation_recovers_exact_threshold_posterior():
    """Alternate Z | tau and tau | Z with theta fixed: the tau marginal must
    be p(tau | Y) on a grid.  The old pseudo-likelihood term biased it."""
    rng = np.random.default_rng(2)
    P, M = 12, 3
    theta = rng.normal(0, 1, P)
    Y = np.array([0, 1, 2, 1, 0, 2, 1, 1, 2, 0, 1, 2])[:, None].astype(float)
    T = np.ones((P, 1), bool)
    ri, ci = np.where(T)
    mu0, tau2 = 0.0, 1.5

    tau = np.array([[-0.5, 0.5]])
    draws = np.empty((60000, 2))
    for t in range(len(draws)):
        Z = gs.sample_Z_grm(Y, theta, tau, T, ri, ci, M, rng)
        tau = gs.gibbs_thresholds_grm(Z, Y, T, ri, ci, M, rng,
                                      old_thresholds=tau, mu_0=mu0, tau2=tau2)
        draws[t] = tau[0]
    draws = draws[2000:]

    g = np.linspace(-4, 4, 601)
    t0, t1 = np.meshgrid(g, g, indexing="ij")
    logp = norm.logpdf(t0, mu0, np.sqrt(tau2)) + norm.logpdf(t1, mu0, np.sqrt(tau2))
    for p in range(P):
        F0, F1 = norm.cdf(t0 - theta[p]), norm.cdf(t1 - theta[p])
        pr = {0: F0, 1: F1 - F0, 2: 1 - F1}[int(Y[p, 0])]
        logp = logp + np.log(np.clip(pr, 1e-300, None))
    logp = np.where(t0 < t1, logp, -np.inf)
    w = np.exp(logp - logp.max())
    w /= w.sum()
    m = np.array([np.sum(w * t0), np.sum(w * t1)])
    sd = np.sqrt([np.sum(w * (t0 - m[0]) ** 2), np.sum(w * (t1 - m[1]) ** 2)])
    np.testing.assert_allclose(draws.mean(0), m, atol=0.04)
    np.testing.assert_allclose(draws.std(0), sd, atol=0.04)
