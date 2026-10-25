"""GPCM discriminations estimated by MH.

GPCMNode now estimates one a_i per item (log-scale random-walk MH against
the node likelihood and the a prior).  PCMNode / discrimination=False keeps
a_i = 1.  A discrimination shared by several nodes, including logit/GPCM
ones, is updated once per iteration against every member's likelihood.
"""
import warnings

import numpy as np
import pytest

import irtree
from irtree.constraints import Constraints
from irtree.data_prep import expand_data
from irtree.node_models import LogitTwoPLNode
from irtree.priors import LKJPrior, NormalPrior


def _gpcm_data(rng, P, I, M, a, th):
    b = rng.normal(0, 1, (I, M - 1))
    eta = a[None, :, None] * (th[:, None, None] - b[None])
    ln = np.concatenate([np.zeros((P, I, 1)), np.cumsum(eta, axis=2)], axis=2)
    pr = np.exp(ln - ln.max(2, keepdims=True))
    pr /= pr.sum(2, keepdims=True)
    return 1 + (rng.random((P, I, 1)) > np.cumsum(pr, 2)).sum(2)


def _fit(d, spec, nodes, n_iter=1500, fix_theta_var=True, **kw):
    warnings.filterwarnings("ignore")
    pr = irtree.build_default_priors(spec, nodes)
    if fix_theta_var:     # identify the theta scale when slopes are free
        pr.set("theta", NormalPrior(mean=0.0, variance=1.0))
    return irtree.run_chain(d, nodes, pr, n_iter=n_iter, seed=1, spec=spec,
                            progress=False, warmup_mode="fixed", n_warmup=500, **kw)


@pytest.mark.slow
def test_gpcm_recovers_discriminations():
    rng = np.random.default_rng(0)
    P, I, M = 600, 10, 4
    a = rng.uniform(0.6, 2.0, I)
    Y = _gpcm_data(rng, P, I, M, a, rng.normal(size=P))
    spec = irtree.IRTreeSpec(mapping=np.arange(M, dtype=float)[:, None])
    d = expand_data(Y, spec, categories_start=1)
    res = _fit(d, spec, [irtree.GPCMNode(n_outcomes=M)])
    ac = [i for i, n in enumerate(res.param_names) if n.startswith("a_")]
    ah = res.chain[:, ac].mean(0)
    assert len(ac) == I
    assert np.corrcoef(ah, a)[0, 1] > 0.9
    assert abs(np.mean(ah - a)) < 0.12
    assert 0.25 < res.accept_rates["a_mh_0"] < 0.65


@pytest.mark.parametrize("node", [irtree.PCMNode(n_outcomes=3),
                                  irtree.GPCMNode(n_outcomes=3, discrimination=False)])
def test_pcm_has_no_discriminations(node):
    rng = np.random.default_rng(1)
    spec = irtree.IRTreeSpec(mapping=np.arange(3, dtype=float)[:, None])
    d = expand_data(rng.integers(1, 4, (40, 4)), spec, categories_start=1)
    res = _fit(d, spec, [node], n_iter=600)
    assert not any(n.startswith("a_") for n in res.param_names)


@pytest.mark.slow
def test_discrimination_shared_by_logit_2pl_and_gpcm_nodes():
    """One pooled update uses both members' likelihoods (per-node updates
    would each see only their own)."""
    rng = np.random.default_rng(2)
    P, I = 600, 10
    a = rng.uniform(0.6, 2.0, I)
    th1, th2 = rng.normal(size=P), rng.normal(size=P)
    b1, b2 = rng.normal(0, 0.7, I), rng.normal(0, 0.7, I)
    def sig(z):
        return 1 / (1 + np.exp(-z))
    y1 = rng.random((P, I)) < sig(a * (th1[:, None] - b1))
    y2 = rng.random((P, I)) < sig(a * (th2[:, None] - b2))
    Y = np.where(~y1, 1, np.where(~y2, 2, 3))
    spec = irtree.IRTreeSpec(mapping=np.array([[0., np.nan], [1., 0.], [1., 1.]]))
    d = expand_data(Y, spec, categories_start=1)
    res = _fit(d, spec, [LogitTwoPLNode(), irtree.GPCMNode(n_outcomes=2)],
               constraints=Constraints(a_shared=[[1, 2]]))
    ac = [i for i, n in enumerate(res.param_names) if n.startswith("a_")]
    assert len(ac) == I
    assert "a_shared_0" in res.accept_rates and "a_mh_0" not in res.accept_rates
    assert np.corrcoef(res.chain[:, ac].mean(0), a)[0, 1] > 0.85


@pytest.mark.slow
def test_correlated_discriminations_with_a_gpcm_node():
    """An a-correlation group with a GPCM node.  The GPCM slopes are updated by MH with the conditional prior from
    Sigma_a, the probit slopes by exact componentwise Gibbs."""
    from scipy.stats import norm
    rng = np.random.default_rng(3)
    P, I = 700, 20
    C = np.array([[1, .7], [.7, 1]])
    A = np.exp(np.log(1.2) + 0.3 * rng.normal(size=(I, 2)) @ np.linalg.cholesky(C).T)
    th1, th2 = rng.normal(size=P), rng.normal(size=P)
    b1, b2 = rng.normal(0, .8, I), rng.normal(0, .8, I)
    def sig(z):
        return 1 / (1 + np.exp(-z))
    y1 = rng.random((P, I)) < sig(A[:, 0] * (th1[:, None] - b1))
    y2 = rng.random((P, I)) < norm.cdf(A[:, 1] * (th2[:, None] - b2))
    Y = np.where(~y1, 1, np.where(~y2, 2, 3))
    spec = irtree.IRTreeSpec(mapping=np.array([[0., np.nan], [1., 0.], [1., 1.]]))
    d = expand_data(Y, spec, categories_start=1)
    nodes = [irtree.GPCMNode(n_outcomes=2), irtree.ProbitTwoPLNode()]
    warnings.filterwarnings("ignore")
    pr = irtree.build_default_priors(spec, nodes)
    pr.set("a.correlation", LKJPrior(eta=1.0))
    pr.set("theta", NormalPrior(mean=0.0, variance=1.0))
    res = irtree.run_chain(d, nodes, pr, n_iter=2000, seed=1, spec=spec, progress=False,
                           warmup_mode="fixed", n_warmup=500,
                           constraints=Constraints(a_corr=[[1, 2]]))
    for k, pref in enumerate(("a_1_", "a_2_")):
        cols = [i for i, n in enumerate(res.param_names) if n.startswith(pref)]
        est = res.chain[:, cols].mean(0)
        assert np.corrcoef(est, A[:, k])[0, 1] > 0.8
        assert abs(np.mean(est - A[:, k])) < 0.15
    assert 0.25 < res.accept_rates["a_corr_mh_0"] < 0.65
    assert "a_mh_0" not in res.accept_rates     # node block not registered
