"""MH updates for covariate coefficients at logit nodes.

A coefficient group with at least one non-augmented (logit / GPCM) node is
updated by preconditioned random-walk MH on

    log pi(c) = lin'c - c' prec c / 2  +  sum_logit loglik_r(c),

where the quadratic part is the prior plus the probit members' Gaussian
latent likelihood.  The tests check the MH kernel against known targets
(closed form and 1-D quadrature) and recovery in full chains.
"""
import warnings

import numpy as np
import pytest

import irtree
from irtree import mcmc_sampler as ms
from irtree.constraints import Constraints
from irtree.data_prep import expand_data


def _run_kernel(prec, lin, ll_fn, n, seed=0, H=None):
    """Run the kernel with the run_chain preconditioner inv(H) (default prec)."""
    rng = np.random.default_rng(seed)
    K = len(lin)
    L = np.linalg.cholesky(np.linalg.inv(prec if H is None else H))
    step = 2.38 / np.sqrt(K)
    c = np.zeros(K)
    out = np.empty((n, K))
    for t in range(n):
        c, _, _ = ms._mh_coefficient(c, prec, lin, ll_fn, L, step, rng)
        out[t] = c
    return out


def test_mh_kernel_gaussian_target():
    prec = np.array([[2.0, 0.6], [0.6, 1.0]])
    lin = np.array([1.0, -0.5])
    draws = _run_kernel(prec, lin, lambda c: (0.0, []), 40000)[2000:]
    V = np.linalg.inv(prec)
    np.testing.assert_allclose(draws.mean(0), V @ lin, atol=0.05)
    np.testing.assert_allclose(np.cov(draws.T), V, atol=0.06)


def test_mh_kernel_logit_target_matches_quadrature():
    rng = np.random.default_rng(1)
    n = 60
    x = rng.normal(size=n)
    o = rng.normal(0, 0.5, n)
    y = rng.random(n) < 1 / (1 + np.exp(-(0.9 * x + o)))
    tau2 = 2.0

    def ll(c):
        eta = x * c[0] + o
        return float(np.sum(y * eta - np.logaddexp(0, eta))), []

    grid = np.linspace(-4, 6, 20001)
    logp = np.array([ll([g])[0] for g in grid]) - grid ** 2 / (2 * tau2)
    w = np.exp(logp - logp.max())
    w /= w.sum()
    m = float(grid @ w)
    sd = float(np.sqrt(((grid - m) ** 2) @ w))

    H = np.array([[1 / tau2 + 0.25 * np.sum(x ** 2)]])
    draws = _run_kernel(np.array([[1 / tau2]]), np.zeros(1), ll, 40000, H=H)
    draws = draws[2000:, 0]
    assert abs(draws.mean() - m) < 0.03
    assert abs(draws.std() - sd) < 0.03


def _sig(z):
    return 1 / (1 + np.exp(-z))


@pytest.mark.slow
def test_logit_node_person_covariate_is_estimated():
    """A beta group used only by a logit node moves away from its start."""
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)
    P, I = 500, 20
    spec = irtree.IRTreeSpec(mapping=np.array([[0.], [1.]]))
    x = rng.normal(size=P)
    th, b = rng.normal(0, 0.6, P), rng.normal(0, 0.7, I)
    Y = (rng.random((P, I)) < _sig(th[:, None] + x[:, None] - b)).astype(int) + 1
    d = expand_data(Y, spec, categories_start=1, X=x[:, None])
    nodes = [irtree.LogitRaschNode(person_covariates=True)]
    res = irtree.run_chain(d, nodes, irtree.build_default_priors(spec, nodes),
                           n_iter=1500, seed=1, spec=spec, progress=False,
                           warmup_mode="fixed", n_warmup=500)
    col = [i for i, n in enumerate(res.param_names) if n.startswith("beta_")][0]
    assert abs(res.chain[:, col].mean() - 1.0) < 0.25
    assert 0.25 < res.accept_rates["beta_mh_0"] < 0.65


@pytest.mark.slow
def test_gamma_shared_by_logit_and_probit_nodes():
    """A pooled group spanning a logit and a probit node uses both likelihoods."""
    from scipy.stats import norm
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)
    P, I = 500, 20
    spec = irtree.disengagement_tree(binary=True)
    w = rng.normal(size=(P, I, 1))
    th1, th2 = rng.normal(size=P), rng.normal(size=P)
    b1, b2 = rng.normal(0.5, 0.5, I), rng.normal(0, 0.7, I)
    e1 = rng.random((P, I)) < _sig(th1[:, None] - b1 + 0.8 * w[..., 0])
    e2 = rng.random((P, I)) < norm.cdf(th2[:, None] - b2 + 0.8 * w[..., 0])
    Y = np.where(e1, 1, np.where(e2, 3, 2))
    d = expand_data(Y, spec, categories_start=1, W=w)
    nodes = [irtree.LogitRaschNode(pi_covariates=True),
             irtree.ProbitRaschNode(pi_covariates=True)]
    res = irtree.run_chain(d, nodes, irtree.build_default_priors(spec, nodes),
                           n_iter=1500, seed=2, spec=spec, progress=False,
                           warmup_mode="fixed", n_warmup=500,
                           constraints=Constraints(gamma_shared=[[1, 2]]))
    col = [i for i, n in enumerate(res.param_names) if n.startswith("gamma_")]
    assert len(col) == 1
    assert abs(res.chain[:, col[0]].mean() - 0.8) < 0.15
