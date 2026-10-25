"""Theta variance fixed at 1 where slopes are free (identification), and
estimated a-group variances with centred slopes under a correlated a.
"""
import warnings

import numpy as np
import pytest
from scipy.stats import norm

import irtree
from irtree.constraints import Constraints
from irtree.data_prep import expand_data
from irtree.priors import InvGammaPrior, LKJPrior, NormalPrior


def test_default_fixes_theta_variance_only_where_slopes_are_free():
    spec = irtree.disengagement_tree(binary=True)
    nodes = [irtree.ProbitRaschNode(), irtree.ProbitTwoPLNode()]
    pr = irtree.build_default_priors(spec, nodes)
    assert isinstance(pr.resolve("theta", 0).variance, InvGammaPrior)
    assert pr.resolve("theta", 1).variance == 1.0
    # a user's type-level theta prior replaces the automatic entry
    pr.set("theta", NormalPrior(mean=0.0, variance=InvGammaPrior()))
    assert isinstance(pr.resolve("theta", 1).variance, InvGammaPrior)


def test_shared_dimension_gets_the_default_on_every_node():
    spec = irtree.ers_tree()                       # nodes 2 and 3 share theta_2
    nodes = [irtree.ProbitRaschNode(), irtree.ProbitTwoPLNode(), irtree.LogitTwoPLNode()]
    pr = irtree.build_default_priors(spec, nodes)
    assert pr.resolve("theta", 1).variance == 1.0
    assert pr.resolve("theta", 2).variance == 1.0
    assert isinstance(pr.resolve("theta", 0).variance, InvGammaPrior)


def test_a_slope_free_node_keeps_the_variance_estimated():
    """A Rasch node fixes the scale (a = 1); fixing sigma2 = 1 as well would
    be a second, substantive restriction, so the variance stays estimated."""
    spec = irtree.ers_tree()                       # nodes 2 and 3 share theta_2
    nodes = [irtree.ProbitRaschNode(), irtree.ProbitTwoPLNode(), irtree.ProbitRaschNode()]
    pr = irtree.build_default_priors(spec, nodes)
    assert isinstance(pr.resolve("theta", 1).variance, InvGammaPrior)
    assert isinstance(pr.resolve("theta", 2).variance, InvGammaPrior)
    assert ("theta", 1) not in pr._auto_node_level


def test_theta_correlation_is_estimated_by_default():
    spec = irtree.disengagement_tree(binary=True)
    nodes = [irtree.ProbitTwoPLNode(), irtree.ProbitTwoPLNode()]
    pr = irtree.build_default_priors(spec, nodes)
    assert isinstance(pr.resolve("theta.correlation", None), irtree.LKJPrior)
    assert pr.resolve("theta.correlation", None).eta == 1.0
    assert "rho_1_2" in pr.free_params()


@pytest.mark.slow
def test_gpcm_slopes_unbiased_with_default_priors():
    """Before: mean slope bias +0.16 with the estimated theta variance."""
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)
    P, I, M = 600, 10, 4
    th, a = rng.normal(size=P), rng.uniform(0.6, 2.0, I)
    b = rng.normal(0, 1, (I, M - 1))
    eta = a[None, :, None] * (th[:, None, None] - b[None])
    ln = np.concatenate([np.zeros((P, I, 1)), np.cumsum(eta, axis=2)], axis=2)
    pr_ = np.exp(ln - ln.max(2, keepdims=True))
    pr_ /= pr_.sum(2, keepdims=True)
    Y = 1 + (rng.random((P, I, 1)) > np.cumsum(pr_, 2)).sum(2)
    spec = irtree.IRTreeSpec(mapping=np.arange(M, dtype=float)[:, None])
    d = expand_data(Y, spec, categories_start=1)
    nodes = [irtree.GPCMNode(n_outcomes=M)]
    res = irtree.run_chain(d, nodes, irtree.build_default_priors(spec, nodes),
                           n_iter=1500, seed=1, spec=spec, progress=False,
                           warmup_mode="fixed", n_warmup=500)
    ac = [i for i, n in enumerate(res.param_names) if n.startswith("a_")]
    assert abs(np.mean(res.chain[:, ac].mean(0) - a)) < 0.08


@pytest.mark.slow
def test_correlated_slopes_estimate_group_variances():
    """Before: variances fixed at 0.5 and uncentred slopes gave rho ~ 0.92
    for a sample correlation of 0.61."""
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)
    P, I = 1000, 30
    C = np.array([[1, .7], [.7, 1]])
    A = np.exp(np.log(1.2) + 0.3 * rng.normal(size=(I, 2)) @ np.linalg.cholesky(C).T)
    th1, th2 = rng.normal(size=P), rng.normal(size=P)
    b1, b2 = rng.normal(0, .8, I), rng.normal(0, .8, I)
    y1 = rng.random((P, I)) < norm.cdf(A[:, 0] * (th1[:, None] - b1))
    y2 = rng.random((P, I)) < norm.cdf(A[:, 1] * (th2[:, None] - b2))
    Y = np.where(~y1, 1, np.where(~y2, 2, 3))
    spec = irtree.IRTreeSpec(mapping=np.array([[0., np.nan], [1., 0.], [1., 1.]]))
    d = expand_data(Y, spec, categories_start=1)
    nodes = [irtree.ProbitTwoPLNode(), irtree.ProbitTwoPLNode()]
    pr = irtree.build_default_priors(spec, nodes)
    pr.set("a.correlation", LKJPrior(eta=1.0))
    res = irtree.run_chain(d, nodes, pr, n_iter=2500, seed=1, spec=spec, progress=False,
                           warmup_mode="fixed", n_warmup=800,
                           constraints=Constraints(a_corr=[[1, 2]]))
    nm, ch = res.param_names, res.chain
    var_cols = [i for i, n in enumerate(nm) if n.startswith("a.variance")]
    assert len(var_cols) == 2
    assert np.all(ch[:, var_cols].mean(0) < 0.35)          # truth ~0.11-0.16
    rho = ch[:, [i for i, n in enumerate(nm) if n.startswith("a.rho")]].mean()
    assert abs(rho - np.corrcoef(A.T)[0, 1]) < 0.2
    assert pr.resolve("a", 0).variance == 0.5             # user's object untouched
