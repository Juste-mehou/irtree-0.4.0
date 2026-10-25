"""Items 14, 15 (and 17): Cowles threshold step, location move, shared thresholds.

15: thresholds are proposed with Z integrated out (marginal GRM category
    probabilities), on u = (tau_1, log gaps) so they stay ordered; then Z is
    redrawn.  A location move shifts theta, the b-groups on the dimension and
    the GRM latent responses by a common c drawn from its exact conditional.
14: a GPCM node and a GRM node may share thresholds (one MH target).
17: GRM nodes sharing thresholds are updated once from all members.
"""
import warnings

import numpy as np
import pytest
from scipy.stats import norm

import irtree
from irtree import mcmc_sampler as ms
from irtree.constraints import Constraints
from irtree.data_prep import expand_data
from irtree.diagnostics import ess


@pytest.mark.slow
def test_cowles_kernel_targets_the_marginal_posterior():
    """theta fixed: the kernel alone must reproduce p(tau | Y) on a grid."""
    rng = np.random.default_rng(2)
    P, M = 12, 3
    theta = rng.normal(0, 1, P)
    Y = np.array([0, 1, 2, 1, 0, 2, 1, 1, 2, 0, 1, 2])[:, None].astype(float)
    nm = irtree.GRMNode(n_outcomes=M)
    mu0, tau2 = 0.0, 1.5

    def col_ll(Tm):
        return nm.log_prob(Y, theta, Tm).sum(axis=0)

    sc = ms._threshold_proposal_scale([Y], [np.ones_like(Y, bool)], M - 1)
    cur = np.array([[-0.5, 0.5]])
    draws = np.empty((60000, 2))
    for t in range(len(draws)):
        cur, _ = ms._mh_ordered_thresholds(cur, col_ll, mu0, tau2, 1.0, rng, scale=sc)
        draws[t] = cur[0]
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


def _fit(d, spec, nodes, n_iter=2000, **kw):
    warnings.filterwarnings("ignore")
    return irtree.run_chain(d, nodes, irtree.build_default_priors(spec, nodes),
                            n_iter=n_iter, seed=1, spec=spec, progress=False, **kw)


@pytest.mark.slow
def test_grm_thresholds_mix():
    """Item-6 Gibbs alone: median ESS ~ 10 per 1,500 draws here."""
    rng = np.random.default_rng(0)
    P, I, M = 800, 10, 4
    spec = irtree.IRTreeSpec(mapping=np.arange(M, dtype=float)[:, None])
    th = rng.normal(size=P)
    tau = np.sort(rng.normal(0, 1, (I, M - 1)), axis=1) + np.array([-0.6, 0, 0.6])
    Y = 1 + ((th[:, None] + rng.normal(size=(P, I)))[:, :, None] > tau[None]).sum(-1)
    d = expand_data(Y, spec, categories_start=1)
    res = _fit(d, spec, [irtree.GRMNode(n_outcomes=M)])
    bc = [i for i, n in enumerate(res.param_names) if n.startswith("b_1_")]
    ch = res.chain
    assert np.median([ess(ch[:, c])[0] for c in bc]) > 100
    est = ch[:, bc].mean(0).reshape(I, M - 1)
    assert np.sqrt(np.mean((est - tau) ** 2)) < 0.1
    assert 0.2 < res.accept_rates["tau_mh_0"] < 0.5


def _two_node_data(kind, rng, P=800, I=12):
    M = np.array([[0., np.nan], [1., np.nan], [2., 0.], [2., 1.], [2., 2.]])
    spec = irtree.IRTreeSpec(mapping=M)
    tau = np.sort(rng.normal(0, 0.8, (I, 2)), axis=1) + np.array([-0.4, 0.4])
    th1, th2 = rng.normal(size=P), rng.normal(size=P)

    def grm(th):
        return ((th[:, None] + rng.normal(size=(P, I)))[:, :, None] > tau[None]).sum(-1)

    def gpcm(th):
        eta = th[:, None, None] - tau[None]
        ln = np.concatenate([np.zeros((P, I, 1)), np.cumsum(eta, 2)], 2)
        pr = np.exp(ln - ln.max(2, keepdims=True))
        pr /= pr.sum(2, keepdims=True)
        return (rng.random((P, I, 1)) > np.cumsum(pr, 2)).sum(2)

    y1 = gpcm(th1) if kind == "gpcm" else grm(th1)
    Y = np.where(y1 < 2, y1, 2 + grm(th2)) + 1
    return spec, expand_data(Y, spec, categories_start=1), tau


@pytest.mark.slow
def test_thresholds_shared_by_two_grm_nodes():
    spec, d, tau = _two_node_data("grm", np.random.default_rng(0))
    nodes = [irtree.GRMNode(n_outcomes=3), irtree.GRMNode(n_outcomes=3)]
    res = _fit(d, spec, nodes, constraints=Constraints(b_shared=[[1, 2]]))
    bc = [i for i, n in enumerate(res.param_names) if n.startswith("b_1&2_")]
    est = res.chain[:, bc].mean(0).reshape(-1, 2)
    assert np.sqrt(np.mean((est - tau) ** 2)) < 0.07
    assert "tau_mh_0" in res.accept_rates


@pytest.mark.slow
def test_thresholds_shared_by_gpcm_and_grm_nodes():
    """Thresholds shared by a GPCM and a GRM node."""
    spec, d, tau = _two_node_data("gpcm", np.random.default_rng(1))
    nodes = [irtree.GPCMNode(n_outcomes=3, discrimination=False),
             irtree.GRMNode(n_outcomes=3)]
    res = _fit(d, spec, nodes, constraints=Constraints(b_shared=[[1, 2]]))
    bc = [i for i, n in enumerate(res.param_names) if n.startswith("b_1&2_")]
    est = res.chain[:, bc].mean(0).reshape(-1, 2)
    assert np.sqrt(np.mean((est - tau) ** 2)) < 0.1
    assert np.all(np.diff(res.chain[:, bc].reshape(len(res.chain), -1, 2), axis=2) > 0)
