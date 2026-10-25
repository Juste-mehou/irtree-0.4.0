"""MH imputation of missing covariates used by logit nodes.

Each missing unit (person row of X, item row of Q, cell of W) has the target

    log pi(v) = rhs'v - v' prec v / 2  +  logit log-likelihood of its cells,

with (prec, rhs) the prior plus the probit nodes' Gaussian latent terms.
Units with no logit cell get the exact Normal draw; the others an MH step.
"""
import warnings

import numpy as np
import pytest

import irtree
from irtree import mcmc_sampler as ms
from irtree.data_prep import expand_data


def _sig(z):
    return 1 / (1 + np.exp(-z))


@pytest.mark.slow
def test_mh_missing_units_matches_targets():
    rng = np.random.default_rng(0)
    # unit 0: Gaussian only (no logit cells) -> N(0.5, 0.5)
    # unit 1: prior/probit part N-form (prec 1.2, rhs 0.3) times a logit lik
    n = 40
    x = rng.uniform(0.5, 1.5, n)          # per-cell slopes a_i * beta
    o = rng.normal(0, 0.5, n)
    y = rng.random(n) < _sig(0.7 * x + o)

    def ll_unit1(v):
        eta = x * v + o
        return float(np.sum(y * eta - np.logaddexp(0, eta)))

    def unit_ll(vals):
        return np.array([0.0, ll_unit1(vals[1, 0])])

    prec = np.array([[[2.0]], [[1.2]]])
    rhs = np.array([[1.0], [0.3]])
    groups = [(np.array([0, 1]), np.array([0]), prec, rhs)]
    precL = [np.array([[[0.0]], [[0.25 * np.sum(x ** 2)]]])]
    touched = np.array([False, True])

    vals = np.zeros((2, 1))
    draws = np.empty((30000, 2))
    for t in range(len(draws)):
        vals, _ = ms._mh_missing_units(vals, groups, precL, touched, unit_ll,
                                       2.38, rng)
        draws[t] = vals[:, 0]
    draws = draws[1000:]

    assert abs(draws[:, 0].mean() - 0.5) < 0.02
    assert abs(draws[:, 0].var() - 0.5) < 0.03

    grid = np.linspace(-4, 5, 20001)
    logp = np.array([ll_unit1(g) for g in grid]) + 0.3 * grid - 0.6 * grid ** 2
    w = np.exp(logp - logp.max())
    w /= w.sum()
    m = grid @ w
    sd = np.sqrt(((grid - m) ** 2) @ w)
    assert abs(draws[:, 1].mean() - m) < 0.03
    assert abs(draws[:, 1].std() - sd) < 0.03


@pytest.mark.slow
def test_logit_node_missing_person_covariate_is_imputed():
    """A missing X used only by a logit node is imputed from the data, not the prior."""
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)
    P, I = 500, 25
    spec = irtree.IRTreeSpec(mapping=np.array([[0.], [1.]]))
    x = rng.normal(size=P)
    th, b = rng.normal(0, 0.5, P), rng.normal(0, 0.7, I)
    Y = (rng.random((P, I)) < _sig(th[:, None] + 1.5 * x[:, None] - b)).astype(int) + 1
    X = x[:, None].copy()
    miss = rng.choice(P, 80, replace=False)
    X[miss] = np.nan
    d = expand_data(Y, spec, categories_start=1, X=X)
    nodes = [irtree.LogitRaschNode(person_covariates=True)]
    res = irtree.run_chain(d, nodes, irtree.build_default_priors(spec, nodes),
                           n_iter=1500, seed=1, spec=spec, progress=False,
                           warmup_mode="fixed", n_warmup=500)
    cols = [res.param_names.index(f"X_missing_{p + 1}_1") for p in miss]
    imp = res.chain[:, cols].mean(0)
    assert np.corrcoef(imp, x[miss])[0, 1] > 0.8
    assert 0.6 < np.polyfit(x[miss], imp, 1)[0] < 1.3
    assert 0.2 < res.accept_rates["X_missing_mh"] < 0.7
