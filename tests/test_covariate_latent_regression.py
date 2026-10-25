"""Covariate coefficients and missing-covariate imputation.

At an augmented node the latent response is

    Z_pi = a_i (core_pi + x_p'beta + q_i'kappa + w_pi'gamma) + eps

on the node's latent mask (all relevant cells, or the U = 0 cells at a 3PL
node).  With E = Z - a (core + other terms), every update is a Normal linear
regression with design rows a_i d_pi.  The unit tests below build that design
matrix explicitly, cell by cell, and compare the closed-form posterior with
what each kernel draws.  A recording RNG returns the mean it is asked to draw
around, so the comparison is exact.
"""
import types

import numpy as np
import pytest

import irtree
from irtree import gibbs_samplers as gs
from irtree import mcmc_sampler as ms
from irtree.data_prep import expand_data


class MeanRNG:
    """Stands in for numpy's Generator: every draw returns its mean."""

    def __init__(self):
        self.covs = []

    def normal(self, loc, scale, size=None):
        self.covs.append(np.atleast_2d(np.square(scale)))
        return loc if size is None else np.full(size, loc)

    def multivariate_normal(self, mean, cov):
        self.covs.append(np.asarray(cov))
        return np.asarray(mean, float).copy()

    def standard_normal(self, shape):
        return np.zeros(shape)


def _posterior(rows, y, prior_mean, prior_var):
    """Closed form for y = rows @ u + N(0,1), u ~ N(prior_mean, prior_var I)."""
    rows = np.asarray(rows, float).reshape(len(y), -1)
    K = rows.shape[1]
    prec = np.eye(K) / prior_var + rows.T @ rows
    rhs = np.full(K, prior_mean) / prior_var + rows.T @ np.asarray(y, float)
    V = np.linalg.inv(prec)
    return V @ rhs, V


def _setup(seed=0, P=7, I=5, R=3):
    rng = np.random.default_rng(seed)
    E = [rng.normal(size=(P, I)) for _ in range(R)]
    M = [rng.random((P, I)) < 0.7 for _ in range(R)]
    # heterogeneous discriminations: Rasch, 2PL, 2PL with different a
    A = [None, rng.uniform(0.5, 2.0, I), rng.uniform(0.5, 2.0, I)][:R]
    return rng, E, M, A


# ---------------------------------------------------------------------------
# Coefficients
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["x", "q", "w"])
def test_coef_pooled_matches_explicit_regression(kind):
    rng, E, M, A = _setup()
    P, I = E[0].shape
    K = 2
    design = {"x": rng.normal(size=(P, K)),
              "q": rng.normal(size=(I, K)),
              "w": rng.normal(size=(P, I, K))}[kind]
    group = [0, 1, 2]
    rows, y = [], []
    for r in group:
        for p, i in zip(*np.where(M[r]), strict=True):
            a = 1.0 if A[r] is None else A[r][i]
            d = (design[p] if kind == "x" else
                 design[i] if kind == "q" else design[p, i])
            rows.append(a * d)
            y.append(E[r][p, i])
    mu, V = _posterior(rows, y, 0.0, 2.5)

    rec = MeanRNG()
    got = gs.gibbs_coef_pooled(kind, group, E, M, A, design, 2.5, rec)
    np.testing.assert_allclose(got, mu, rtol=1e-10)
    np.testing.assert_allclose(rec.covs[-1], V, rtol=1e-10)


def test_coef_pooled_ignores_nodes_without_latent_response():
    rng, E, M, A = _setup()
    design = rng.normal(size=(E[0].shape[0], 1))
    a = gs.gibbs_coef_pooled("x", [0, 1], E, M, A, design, 1.0, MeanRNG())
    b = gs.gibbs_coef_pooled("x", [0, 1, 2], [E[0], E[1], None], M, A,
                             design, 1.0, MeanRNG())
    np.testing.assert_allclose(a, b)


# ---------------------------------------------------------------------------
# Imputation
# ---------------------------------------------------------------------------

def test_X_missing_matches_explicit_regression():
    rng, E, M, A = _setup(seed=1)
    P, I = E[0].shape
    K = 3
    X = rng.normal(size=(P, K))
    fobs = np.ones((P, K), bool)
    fobs[2, [0, 2]] = False          # partially missing
    fobs[4, :] = False               # fully missing
    beta = [rng.normal(size=K), None, rng.normal(size=K)]   # node 1 has no X
    m0, v0 = 0.3, 1.7

    expected = {}
    for p in (2, 4):
        mis, obs = np.where(~fobs[p])[0], np.where(fobs[p])[0]
        rows, y = [], []
        for r in (0, 2):
            for i in np.where(M[r][p])[0]:
                a = 1.0 if A[r] is None else A[r][i]
                rows.append(a * beta[r][mis])
                y.append(E[r][p, i] - a * X[p, obs] @ beta[r][obs])
        expected[p] = _posterior(rows, y, m0, v0)[0]

    Xw = gs.gibbs_X_missing(np.array([2, 4]), E, M, A, beta, X.copy(), fobs,
                            m0, v0, MeanRNG())
    np.testing.assert_allclose(Xw[2, [0, 2]], expected[2], rtol=1e-10)
    np.testing.assert_allclose(Xw[4], expected[4], rtol=1e-10)
    np.testing.assert_array_equal(Xw[2, 1], X[2, 1])     # observed kept


def test_Q_missing_matches_explicit_regression():
    rng, E, M, A = _setup(seed=2)
    P, I = E[0].shape
    K = 2
    Q = rng.normal(size=(I, K))
    fobs = np.ones((I, K), bool)
    fobs[1, 0] = False
    fobs[3, :] = False
    kappa = [rng.normal(size=K), rng.normal(size=K), None]
    m0, v0 = 0.0, 1.0

    expected = {}
    for i in (1, 3):
        mis, obs = np.where(~fobs[i])[0], np.where(fobs[i])[0]
        rows, y = [], []
        for r in (0, 1):
            a = 1.0 if A[r] is None else A[r][i]
            for p in np.where(M[r][:, i])[0]:
                rows.append(a * kappa[r][mis])
                y.append(E[r][p, i] - a * Q[i, obs] @ kappa[r][obs])
        expected[i] = _posterior(rows, y, m0, v0)[0]

    Qw = gs.gibbs_Q_missing(np.array([1, 3]), E, M, A, kappa, Q.copy(), fobs,
                            m0, v0, MeanRNG())
    np.testing.assert_allclose(Qw[1, 0], expected[1][0], rtol=1e-10)
    np.testing.assert_allclose(Qw[3], expected[3], rtol=1e-10)
    assert Qw[1, 1] == Q[1, 1]


def test_W_missing_matches_explicit_regression():
    rng, E, M, A = _setup(seed=3)
    P, I = E[0].shape
    K = 3
    W = rng.normal(size=(P, I, K))
    fobs = np.ones((P, I, K), bool)
    cells = [(0, 1), (2, 3), (5, 0), (6, 4)]
    fobs[0, 1, 2] = False
    fobs[2, 3, :] = False
    fobs[5, 0, [0, 1]] = False
    fobs[6, 4, 1] = False
    M[0][6, 4] = M[1][6, 4] = M[2][6, 4] = False      # used by no node: prior
    gamma = [rng.normal(size=K), rng.normal(size=K), rng.normal(size=K)]
    m0, v0 = 0.1, 0.8

    Ww = gs.gibbs_W_missing(np.array(cells), E, M, A, gamma, W.copy(), fobs,
                            m0, v0, MeanRNG())
    for p, i in cells:
        mis, obs = np.where(~fobs[p, i])[0], np.where(fobs[p, i])[0]
        rows, y = [], []
        for r in range(3):
            if not M[r][p, i]:
                continue
            a = 1.0 if A[r] is None else A[r][i]
            rows.append(a * gamma[r][mis])
            y.append(E[r][p, i] - a * W[p, i, obs] @ gamma[r][obs])
        mu = (_posterior(rows, y, m0, v0)[0] if rows
              else np.full(len(mis), m0))
        np.testing.assert_allclose(Ww[p, i, mis], mu, rtol=1e-10)
        np.testing.assert_array_equal(Ww[p, i, obs], W[p, i, obs])


# ---------------------------------------------------------------------------
# Imputed values enter the likelihood
# ---------------------------------------------------------------------------

def test_node_offset_uses_working_copies_and_include():
    rng = np.random.default_rng(4)
    P, I = 4, 3
    X, Q, W = rng.normal(size=(P, 2)), rng.normal(size=(I, 2)), rng.normal(size=(P, I, 2))
    Qw, Ww = Q + 1.0, W - 1.0
    data = types.SimpleNamespace(n_persons=P, n_items=I, Q=Q, W=W,
                                 W_observed=np.zeros((P, I), bool))
    cov = types.SimpleNamespace(
        x_node_to_group=[0], q_node_to_group=[0], w_node_to_group=[0],
        x_idx=[np.array([0, 1])], q_idx=[np.array([1])], w_group_idx=[np.array([0])])
    beta, kappa, gamma = np.array([0.5, -1.0]), np.array([2.0]), np.array([0.7])
    params = dict(beta=[beta], kappa=[kappa], gamma=[gamma],
                  X_working=X, Q_working=Qw, W_working=Ww)

    full = ms._node_offset(0, params, cov, data)
    expect = ((X @ beta)[:, None] + (Qw[:, [1]] @ kappa)[None, :]
              + Ww[:, :, [0]] @ gamma)
    np.testing.assert_allclose(full, expect)

    no_beta = ms._node_offset(0, params, cov, data, include=("q", "w"))
    np.testing.assert_allclose(no_beta, expect - (X @ beta)[:, None])
    assert ms._node_offset(0, params, cov, data, include=()) is None


# ---------------------------------------------------------------------------
# End to end: 2PL node, missing person covariate plus an item covariate
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_2pl_missing_X_with_item_covariate_recovers_values():
    """Guards against divergence (kappa ~ 1e17) when the imputation ignores
    a and Q.kappa, or the kappa draw ignores the imputed values."""
    from scipy.stats import norm
    rng = np.random.default_rng(0)
    P, I = 600, 25
    spec = irtree.IRTreeSpec(mapping=np.array([[0.], [1.]]))
    x, q = rng.normal(size=P), rng.normal(size=I)
    th, b = rng.normal(0, 0.5, P), rng.normal(0, 0.4, I)
    a = rng.uniform(1.6, 2.0, I)
    eta = a * (th[:, None] + 1.5 * x[:, None] - b + 0.8 * q)
    Y = (rng.random((P, I)) < norm.cdf(eta)).astype(int) + 1
    X = x[:, None].copy()
    miss = rng.choice(P, 80, replace=False)
    X[miss] = np.nan
    d = expand_data(Y, spec, categories_start=1, X=X, Q=q[:, None])
    nodes = [irtree.ProbitTwoPLNode(person_covariates=True, item_covariates=True)]
    res = irtree.run_chain(d, nodes, irtree.build_default_priors(spec, nodes),
                           n_iter=800, seed=1, spec=spec, progress=False,
                           warmup_mode="fixed", n_warmup=300)
    ch = res.chain
    cols = [res.param_names.index(f"X_missing_{p + 1}_1") for p in miss]
    imp = ch[:, cols].mean(0)
    assert np.all(np.isfinite(res.chain))
    assert np.corrcoef(imp, x[miss])[0, 1] > 0.8
    assert 0.7 < np.polyfit(x[miss], imp, 1)[0] < 1.25
