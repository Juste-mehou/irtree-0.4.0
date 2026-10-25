"""WAIC / LOO rebuild the linear predictor the chain was drawn under.

posterior._pointwise_log_lik_fast finds covariate columns by group label and
covariate name, as the chain names them, so named, pooled and
column-selected covariates all enter; mixed nodes' b columns are used, and
a node whose columns are missing raises.  The test compares, at the last stored draw, the rebuilt per-person
log-likelihood with the likelihood of the sampler's own final state.
"""
import warnings

import numpy as np
import pytest

import irtree
from irtree import posterior
from irtree.constraints import Constraints
from irtree.data_prep import expand_data, resolve_covariate_indices
from irtree.mcmc_sampler import _node_offset


def _direct_loglik(res, data, nodes, spec, constraints):
    """Per-person log-likelihood of res.final_params, from the sampler's state."""
    fp = res.final_params
    cov = resolve_covariate_indices(nodes, data, spec=spec, constraints=constraints)
    con = constraints or irtree.constraints.NO_CONSTRAINTS
    R = data.n_nodes
    b_g, a_g, c_g = (con.b_node_to_group_for(R), con.a_node_to_group_for(R),
                     con.c_node_to_group_for(R))
    n2d = list(spec.node_to_latent) if spec.node_to_latent is not None else list(range(R))
    out = np.zeros(data.n_persons)
    for r, nm in enumerate(nodes):
        lp = nm.log_prob(data.Y_star[r], fp["theta"][n2d[r]], fp["b"][b_g[r]],
                         a_r=fp["a"][a_g[r]], c_r=fp["c"][c_g[r]],
                         offset=_node_offset(r, fp, cov, data))
        out += np.where(data.T[r], lp, 0.0).sum(axis=1)
    return out


def _check(res, data, nodes, spec, constraints=None):
    got = posterior._pointwise_log_lik_fast(
        res.chain[-1:].astype(np.float64), res.param_names, data, nodes,
        spec=spec, constraints=constraints)[0]
    np.testing.assert_allclose(got, _direct_loglik(res, data, nodes, spec, constraints),
                               rtol=1e-8, atol=1e-8)


def test_named_pooled_and_missing_covariates():
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)
    P, I = 80, 6
    spec = irtree.disengagement_tree(binary=True)
    Y = rng.integers(1, 4, size=(P, I))
    X = rng.normal(size=(P, 2))
    X[:4, 0] = np.nan
    Q = rng.normal(size=(I, 1))
    W = rng.normal(size=(P, I, 1))
    W[rng.random((P, I)) < 0.05, 0] = np.nan
    d = expand_data(Y, spec, categories_start=1, X=X, Q=Q, W=W,
                    X_names=["ses", "age"], Q_names=["len"], W_names=["rt"])
    kw = dict(person_covariates=["ses"], item_covariates=True, pi_covariates=True)
    nodes = [irtree.ProbitTwoPLNode(**kw), irtree.ProbitRaschNode(**kw)]
    con = Constraints(b_shared=[[1, 2]], gamma_shared=[[1, 2]])
    res = irtree.run_chain(d, nodes, irtree.build_default_priors(spec, nodes),
                           n_iter=40, seed=1, spec=spec, progress=False,
                           warmup_mode="fixed", n_warmup=10, constraints=con,
                           chain_dtype="float64")
    assert any(n.startswith("kappa_1&2_len") for n in res.param_names)
    _check(res, d, nodes, spec, con)


def test_mixed_gpcm_node_with_covariates():
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(1)
    P, I = 80, 6
    spec = irtree.IRTreeSpec(mapping=np.array([[0.], [1.], [2.]]))
    Y = rng.integers(1, 4, size=(P, I))
    Y[:, 0::2] = rng.integers(1, 3, size=(P, 3))
    d = expand_data(Y, spec, categories_start=1, X=rng.normal(size=(P, 1)))
    nodes = [irtree.GPCMNode(pseudo_item_outcomes=np.array([2, 3] * 3),
                             person_covariates=True)]
    res = irtree.run_chain(d, nodes, irtree.build_default_priors(spec, nodes),
                           n_iter=40, seed=1, spec=spec, progress=False,
                           warmup_mode="fixed", n_warmup=10, chain_dtype="float64")
    _check(res, d, nodes, spec)


def test_missing_columns_raise_instead_of_dropping_the_node():
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(2)
    spec = irtree.disengagement_tree(binary=True)
    d = expand_data(rng.integers(1, 4, size=(30, 4)), spec, categories_start=1)
    nodes = [irtree.ProbitRaschNode(), irtree.ProbitRaschNode()]
    res = irtree.run_chain(d, nodes, irtree.build_default_priors(spec, nodes),
                           n_iter=20, seed=1, spec=spec, progress=False,
                           warmup_mode="fixed", n_warmup=5)
    names = [n.replace("b_2_", "zz_") for n in res.param_names]
    with pytest.raises(ValueError, match="Node 2"):
        posterior._pointwise_log_lik_fast(res.chain, names, d, nodes, spec=spec)
