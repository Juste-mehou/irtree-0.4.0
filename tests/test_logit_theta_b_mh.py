"""Person-wise and item-wise MH for theta and b at logit/GPCM nodes.

theta_pd (dimension with a logit node) has target
    conditional prior given the person's other dimensions (Sigma^{-1})
  + Gaussian latent term of every probit node on d (exact given Z)
  + every logit node's row log-likelihood;
b_i (b-group with a logit member) has target
    N(mu, tau2_g) with the current hierarchical variance
  + probit members' Gaussian term + logit members' column log-likelihood.
Probit-only dimensions are drawn by Gibbs GIVEN the MH dimensions.
"""
import warnings

import numpy as np
import pytest
from scipy.stats import norm

import irtree
from irtree import gibbs_samplers as gs
from irtree import mcmc_sampler as ms
from irtree.data_prep import expand_data
from irtree.diagnostics import ess
from irtree.priors import LKJPrior


def _sig(z):
    return 1 / (1 + np.exp(-z))


def test_mh_diag_units_matches_targets():
    rng = np.random.default_rng(0)
    n = 40
    x = rng.uniform(0.5, 1.5, n)
    o = rng.normal(0, 0.5, n)
    y = rng.random(n) < _sig(0.7 * x + o)

    def ll1(v):
        eta = x[None, :] * np.atleast_1d(v)[:, None] + o[None, :]
        return np.sum(y * eta - np.logaddexp(0, eta), axis=1)

    def unit_ll(vals):                 # unit 0: no logit cells
        return np.array([0.0, ll1(vals[1, 0])[0]])

    prec = np.array([[2.0], [1.2]])
    rhs = np.array([[1.0], [0.3]])
    H = prec + np.array([[0.0], [0.25 * np.sum(x ** 2)]])
    v = np.zeros((2, 1))
    draws = np.empty((30000, 2))
    for t in range(len(draws)):
        v, _ = ms._mh_diag_units(v, prec, rhs, H, unit_ll, 2.38, rng)
        draws[t] = v[:, 0]
    draws = draws[1000:]
    assert abs(draws[:, 0].mean() - 0.5) < 0.03
    assert abs(draws[:, 0].var() - 0.5) < 0.04

    grid = np.linspace(-4, 5, 20001)
    logp = ll1(grid) + 0.3 * grid - 0.6 * grid ** 2
    w = np.exp(logp - logp.max())
    w /= w.sum()
    m = grid @ w
    sd = np.sqrt(((grid - m) ** 2) @ w)
    assert abs(draws[:, 1].mean() - m) < 0.03
    assert abs(draws[:, 1].std() - sd) < 0.03


class _ZeroRNG:
    def standard_normal(self, shape):
        return np.zeros(shape)


def test_gibbs_theta_given_is_the_conditional_of_the_joint():
    rng = np.random.default_rng(1)
    D, P = 3, 5
    A = rng.normal(size=(D, D))
    Sigma = A @ A.T + D * np.eye(D)
    S, n = rng.normal(size=(P, D)), rng.uniform(1, 5, (P, D))
    theta = rng.normal(size=(P, D))
    owned = [0, 2]
    got = gs.gibbs_theta_given(S, n, Sigma, theta, owned, _ZeroRNG())
    Lam = np.linalg.inv(Sigma)
    for p in range(P):
        # joint precision with likelihood on owned dims only
        J = Lam.copy()
        h = np.zeros(D)
        for d in owned:
            J[d, d] += n[p, d]
            h[d] = S[p, d]
        G, M = np.array(owned), np.array([1])
        mu = np.linalg.solve(J[np.ix_(G, G)], h[G] - J[np.ix_(G, M)] @ theta[p, M])
        np.testing.assert_allclose(got[p], mu, rtol=1e-10)


def _fit(d, spec, nodes, priors=None, n_iter=2000, seed=1):
    warnings.filterwarnings("ignore")
    pr = priors or irtree.build_default_priors(spec, nodes)
    return irtree.run_chain(d, nodes, pr, n_iter=n_iter, seed=seed, spec=spec,
                            progress=False, warmup_mode="fixed", n_warmup=500)


@pytest.mark.slow
def test_logit_rasch_theta_mixes():
    """Theta mixes (a joint (theta, b) random walk gives ESS ~ 6/1500)."""
    rng = np.random.default_rng(0)
    P, I = 500, 20
    spec = irtree.IRTreeSpec(mapping=np.array([[0.], [1.]]))
    th, b = rng.normal(size=P), rng.normal(0, 1, I)
    Y = (rng.random((P, I)) < _sig(th[:, None] - b)).astype(int) + 1
    d = expand_data(Y, spec, categories_start=1)
    res = _fit(d, spec, [irtree.LogitRaschNode()])
    nm, ch = res.param_names, res.chain
    tc = [i for i, n in enumerate(nm) if n.startswith("theta_1_")]
    bc = [i for i, n in enumerate(nm) if n.startswith("b_1_")]
    assert np.corrcoef(ch[:, tc].mean(0), th)[0, 1] > 0.83
    assert np.corrcoef(ch[:, bc].mean(0), b)[0, 1] > 0.98
    assert np.median([ess(ch[:, c])[0] for c in tc[:40]]) > 100
    assert 0.3 < res.accept_rates["theta_mh_0"] < 0.6


@pytest.mark.slow
def test_theta_shared_by_logit_and_probit_nodes():
    """ERS tree: theta_2 is shared by a logit (node 2) and a probit (node 3) node."""
    rng = np.random.default_rng(0)
    P, I = 500, 20
    spec = irtree.ers_tree()
    t1, t2 = rng.normal(size=P), rng.normal(size=P)
    b1, b2, b3 = rng.normal(0, .7, I), rng.normal(.5, .7, I), rng.normal(.5, .7, I)
    pos = rng.random((P, I)) < norm.cdf(t1[:, None] - b1)
    ext_neg = rng.random((P, I)) < _sig(t2[:, None] - b2)
    ext_pos = rng.random((P, I)) < norm.cdf(t2[:, None] - b3)
    Y = np.where(pos, np.where(ext_pos, 4, 3), np.where(ext_neg, 1, 2))
    d = expand_data(Y, spec, categories_start=1)
    res = _fit(d, spec, [irtree.ProbitRaschNode(), irtree.LogitRaschNode(),
                         irtree.ProbitRaschNode()])
    nm, ch = res.param_names, res.chain
    tc = [i for i, n in enumerate(nm) if n.startswith("theta_2&3_")]
    assert np.corrcoef(ch[:, tc].mean(0), t2)[0, 1] > 0.85


@pytest.mark.slow
def test_correlation_between_logit_and_probit_dimensions():
    """rho between a logit and a probit dimension is recovered (truth 0.7)."""
    rng = np.random.default_rng(0)
    P, I = 600, 20
    C = np.array([[1, .7], [.7, 1]])
    th = rng.normal(size=(P, 2)) @ np.linalg.cholesky(C).T
    b1, b2 = rng.normal(.5, .5, I), rng.normal(0, .7, I)
    e1 = rng.random((P, I)) < _sig(th[:, [0]] - b1)
    e2 = rng.random((P, I)) < norm.cdf(th[:, [1]] - b2)
    Y = np.where(e1, 1, np.where(e2, 3, 2))
    spec = irtree.disengagement_tree(binary=True)
    d = expand_data(Y, spec, categories_start=1)
    nodes = [irtree.LogitRaschNode(), irtree.ProbitRaschNode()]
    pr = irtree.build_default_priors(spec, nodes)
    pr.set("theta.correlation", LKJPrior(eta=1.0))
    res = _fit(d, spec, nodes, priors=pr, n_iter=2500)
    rho = res.chain[:, res.param_names.index("rho_1_2")].mean()
    assert abs(rho - 0.7) < 0.15
