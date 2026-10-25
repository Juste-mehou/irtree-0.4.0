"""One discrimination shared by all items of a probit node (common_discrimination).

The latent regression Z_pi = a d_pi + eps (d = theta - b + offset) has one
slope in every cell, so its full conditional pools every cell of the node:
precision 1/v + sum_pi d_pi^2, truncated to a > 0.
"""
import warnings

import numpy as np
import pytest
from scipy.stats import norm

import irtree
from irtree import gibbs_samplers as gs
from irtree.constraints import Constraints
from irtree.data_prep import expand_data
from irtree.priors import LogNormalPrior
from tests import _harness as h


def test_common_slope_draw_matches_pooled_truncated_normal():
    rng = np.random.default_rng(0)
    P, I = 60, 5
    theta, b = rng.normal(size=P), rng.normal(size=I)
    off = rng.normal(0, 0.3, (P, I))
    T = rng.random((P, I)) > 0.3
    ri, ci = np.where(T)
    Z = rng.normal(1.3 * (theta[:, None] - b + off), 1.0)
    m, v = 1.0, 0.5
    d = (theta[:, None] - b + off)[T]
    prec = 1 / v + np.sum(d ** 2)
    mu = (m / v + np.sum(d * Z[T])) / prec
    tm, tv = h.truncnorm_moments(mu, np.sqrt(1 / prec), lower=0.0)
    draws = h.draw_many(lambda r: gs.gibbs_a(Z, theta, b, ri, ci, m, v, 0.0, r,
                                             n_items=I, offset=off, common=True), 20000)
    assert np.all(draws == draws[:, :1])               # one value for every item
    h.assert_mean(draws[:, :1], [tm], label="common a mean")
    h.assert_var(draws[:, :1], [tv], label="common a var")


def _common_slope_data(seed, a=1.4, P=1500, I=20, guess=None):
    rng = np.random.default_rng(seed)
    th1, th2 = rng.normal(size=P), rng.normal(size=P)
    b1, b2 = rng.normal(0.5, 0.7, I), rng.normal(0, 0.8, I)
    p1 = norm.cdf(a * (th1[:, None] - b1))
    if guess is not None:
        p1 = guess + (1 - guess) * p1
    rg = rng.random((P, I)) < p1
    co = rng.random((P, I)) < norm.cdf(th2[:, None] - b2)
    spec = irtree.disengagement_tree(binary=True)
    return spec, expand_data(np.where(rg, 1, np.where(co, 3, 2)), spec, categories_start=1)


def _fit(spec, d, nodes, priors=None, n_iter=2000):
    warnings.filterwarnings("ignore")
    pr = priors or irtree.build_default_priors(spec, nodes)
    return irtree.run_chain(d, nodes, pr, n_iter=n_iter, seed=1, spec=spec, progress=False,
                            warmup_mode="fixed", n_warmup=500)


def _slope_cols(res):
    return [i for i, n in enumerate(res.param_names) if n.startswith("a_1_")]


@pytest.mark.slow
@pytest.mark.parametrize("prior", ["truncnormal_gibbs", "lognormal_mh"])
def test_probit_common_slope_is_recovered(prior):
    spec, d = _common_slope_data(0)
    nodes = [irtree.ProbitTwoPLNode(common_discrimination=True), irtree.ProbitRaschNode()]
    pr = irtree.build_default_priors(spec, nodes)
    if prior == "lognormal_mh":
        pr.set("a", LogNormalPrior(mean=0.0, variance=1.0), node=1)
    res = _fit(spec, d, nodes, pr)
    A = res.chain[:, _slope_cols(res)].astype(float)
    assert np.allclose(A, A[:, :1])                    # one slope, stored per item
    assert abs(A[:, 0].mean() - 1.4) < 0.12
    if prior == "lognormal_mh":
        assert 0.2 < res.accept_rates["a_0"] < 0.7


@pytest.mark.slow
def test_probit_3pl_common_slope_is_recovered():
    spec, d = _common_slope_data(1, a=1.2, guess=0.15)
    nodes = [irtree.ProbitThreePLNode(common_discrimination=True), irtree.ProbitRaschNode()]
    res = _fit(spec, d, nodes, n_iter=3000)
    A = res.chain[:, _slope_cols(res)].astype(float)
    assert np.allclose(A, A[:, :1])
    assert abs(A[:, 0].mean() - 1.2) < 0.2


def test_common_slope_rejected_in_sharing_and_correlation_groups():
    spec = irtree.IRTreeSpec(mapping=np.array([[0., np.nan], [1., 0.], [1., 1.]]))
    nodes = [irtree.ProbitTwoPLNode(common_discrimination=True), irtree.ProbitTwoPLNode()]
    with pytest.raises(NotImplementedError, match="common_discrimination"):
        Constraints(a_shared=[[1, 2]]).validate(spec, nodes)
    with pytest.raises(ValueError, match="common_discrimination"):
        Constraints(a_corr=[[1, 2]]).validate(spec, nodes)


def test_common_slope_rejects_an_estimated_slope_variance():
    from irtree.priors import InvGammaPrior, TruncNormalPrior
    spec, d = _common_slope_data(2, P=50, I=5)
    nodes = [irtree.ProbitTwoPLNode(common_discrimination=True), irtree.ProbitRaschNode()]
    pr = irtree.build_default_priors(spec, nodes)
    pr.set("a", TruncNormalPrior(mean=1.0, variance=InvGammaPrior(), lower=0.0), node=1, force=True)
    with pytest.raises(ValueError, match="one slope"):
        _fit(spec, d, nodes, pr, n_iter=600)
