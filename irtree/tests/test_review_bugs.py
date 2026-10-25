"""Bugs found in the documentation review (2026-10-23).

1. LogitThreePLNode went through the probit 3PL augmentation path.
2. The 3PL guessing indicators ignored the covariate offset.
3. Constraints(theta_corr=...) was validated nowhere and never used.
4. CustomNode was augmentable by default (its log_prob was ignored).
5. HalfNormal sigma2 adaptation blocks were keyed by node, not dimension.
Also: Constraints.validate() is now called by run_chain.
"""
import warnings

import numpy as np
import pytest
from scipy.stats import norm

import irtree
from irtree.constraints import Constraints
from irtree.data_prep import expand_data
from irtree.node_models import CustomNode, LogitThreePLNode, _log_sigmoid
from irtree.priors import HalfNormalPrior, InverseWishartPrior, LKJPrior, NormalPrior

SPEC1 = irtree.IRTreeSpec(mapping=np.array([[0.], [1.]]))


def _run(d, spec, nodes, priors=None, n_iter=1500, **kw):
    warnings.filterwarnings("ignore")
    pr = priors or irtree.build_default_priors(spec, nodes)
    return irtree.run_chain(d, nodes, pr, n_iter=n_iter, seed=1, spec=spec,
                            progress=False, warmup_mode="fixed",
                            n_warmup=min(500, n_iter // 3), **kw)


def _cols(res, prefix):
    return [i for i, n in enumerate(res.param_names) if n.startswith(prefix)]


@pytest.mark.slow
def test_logit_3pl_is_fitted_by_its_own_mh_blocks():
    rng = np.random.default_rng(0)
    P, I = 1200, 15
    th, a, b = rng.normal(size=P), rng.uniform(0.8, 2.0, I), rng.normal(0, 1, I)
    Y = (rng.random((P, I)) < 0.2 + 0.8 / (1 + np.exp(-a * (th[:, None] - b)))).astype(int) + 1
    res = _run(expand_data(Y, SPEC1, categories_start=1), SPEC1, [LogitThreePLNode()])
    assert {"a_mh_0", "c_mh_0", "theta_mh_0", "b_mh_0"} <= set(res.accept_rates)
    assert abs(np.mean(res.chain[:, _cols(res, "a_1_")].mean(0) - a)) < 0.15
    assert abs(res.chain[:, _cols(res, "c_1_")].mean() - 0.2) < 0.05


@pytest.mark.slow
def test_probit_3pl_guessing_uses_the_covariate_offset():
    """Before: c 0.06 and a bias -0.37 with a person covariate."""
    rng = np.random.default_rng(1)
    P, I = 1200, 20
    x, th = rng.normal(size=P), rng.normal(0, 0.8, P)
    a, b = rng.uniform(0.8, 1.6, I), rng.normal(0, 1, I)
    Y = (rng.random((P, I)) < 0.2 + 0.8 * norm.cdf(a * (th[:, None] + x[:, None] - b))).astype(int) + 1
    d = expand_data(Y, SPEC1, categories_start=1, X=x[:, None])
    nodes = [irtree.ProbitThreePLNode(person_covariates=True)]
    pr = irtree.build_default_priors(SPEC1, nodes)
    pr.set("theta", NormalPrior(mean=0.0, variance=0.64))
    res = _run(d, SPEC1, nodes, pr)
    assert abs(res.chain[:, _cols(res, "c_1_")].mean() - 0.2) < 0.05
    assert abs(res.chain[:, _cols(res, "beta_")].mean() - 1.0) < 0.15


def _three_dim_data(rng, P=800, I=20):
    C = np.array([[1, .7, 0], [.7, 1, 0], [0, 0, 1]])
    th = rng.normal(size=(P, 3)) @ np.linalg.cholesky(C).T
    bs = rng.normal(0, .7, (3, I))
    ys = [(rng.random((P, I)) < norm.cdf(th[:, [k]] - bs[k])).astype(int) for k in range(3)]
    Y = 1 + 4 * ys[0] + 2 * ys[1] + ys[2]
    spec = irtree.IRTreeSpec(mapping=np.array(
        [[(k >> 2) & 1, (k >> 1) & 1, k & 1] for k in range(8)], float))
    return spec, expand_data(Y, spec, categories_start=1), th


@pytest.mark.slow
@pytest.mark.parametrize("prior", [LKJPrior(eta=1.0), InverseWishartPrior()])
def test_theta_correlation_blocks(prior):
    spec, d, th = _three_dim_data(np.random.default_rng(0))
    nodes = [irtree.ProbitRaschNode()] * 3
    pr = irtree.build_default_priors(spec, nodes)
    pr.set("theta.correlation", prior)
    res = _run(d, spec, nodes, pr, constraints=Constraints(theta_corr=[[1, 2]]))
    nm, ch = res.param_names, res.chain
    if isinstance(prior, LKJPrior):
        assert [n for n in nm if n.startswith("rho_")] == ["rho_1_2"]
        rho = ch[:, nm.index("rho_1_2")].mean()
    else:
        assert [n for n in nm if n.startswith("Sigma_theta")] == [
            "Sigma_theta_1_1", "Sigma_theta_1_2", "Sigma_theta_2_2"]
        S = [ch[:, nm.index(f"Sigma_theta_{k}")].mean() for k in ("1_1", "1_2", "2_2")]
        rho = S[1] / np.sqrt(S[0] * S[2])
    assert abs(rho - np.corrcoef(th.T)[0, 1]) < 0.1


def test_single_block_reproduces_the_joint_layout():
    spec, d, _ = _three_dim_data(np.random.default_rng(1), P=100, I=5)
    nodes = [irtree.ProbitRaschNode()] * 3
    names = []
    for con in (None, Constraints(theta_corr=[[1, 2, 3]])):
        pr = irtree.build_default_priors(spec, nodes)
        pr.set("theta.correlation", LKJPrior(eta=1.0))
        res = _run(d, spec, nodes, pr, n_iter=40, constraints=con)
        names.append([n for n in res.param_names if n.startswith("rho_")])
    assert names[0] == names[1] == ["rho_1_2", "rho_1_3", "rho_2_3"]


def test_invalid_constraints_are_rejected():
    spec, d, _ = _three_dim_data(np.random.default_rng(2), P=50, I=4)
    nodes = [irtree.ProbitRaschNode()] * 3
    with pytest.raises(ValueError, match="theta_corr"):
        _run(d, spec, nodes, n_iter=20, constraints=Constraints(theta_corr=[[1, 4]]))


@pytest.mark.slow
def test_custom_node_is_fitted_through_its_log_prob():
    class MyLogitRasch(CustomNode):
        def log_prob(self, y_star, theta_r, b_r, a_r=None, c_r=None, offset=None):
            eta = theta_r[:, None] - np.asarray(b_r)[None, :]
            if offset is not None:
                eta = eta + offset
            return _log_sigmoid((2 * np.nan_to_num(y_star, nan=0.0) - 1) * eta)

    assert not MyLogitRasch().augmentable
    rng = np.random.default_rng(2)
    th, b = rng.normal(size=400), rng.normal(0, 1, 12)
    Y = (rng.random((400, 12)) < 1 / (1 + np.exp(-(th[:, None] - b)))).astype(int) + 1
    d = expand_data(Y, SPEC1, categories_start=1)
    r1 = _run(d, SPEC1, [MyLogitRasch()])
    r2 = _run(d, SPEC1, [irtree.LogitRaschNode()])
    np.testing.assert_allclose(r1.chain[:, _cols(r1, "b_1_")].mean(0),
                               r2.chain[:, _cols(r2, "b_1_")].mean(0), atol=1e-6)


def test_halfnormal_sigma2_blocks_follow_dimensions():
    rng = np.random.default_rng(3)
    spec = irtree.ers_tree()                       # 3 nodes, 2 dimensions
    d = expand_data(rng.integers(1, 5, size=(200, 6)), spec, categories_start=1)
    nodes = [irtree.ProbitRaschNode()] * 3
    pr = irtree.build_default_priors(spec, nodes)
    pr.set("theta", NormalPrior(mean=0.0, variance=HalfNormalPrior(scale=1.0)))
    res = _run(d, spec, nodes, pr, n_iter=300)
    assert sorted(k for k in res.accept_rates if k.startswith("log_sigma")) == [
        "log_sigma_0", "log_sigma_1"]


def test_shared_guessing_with_a_logit_node_is_rejected_at_validation():
    spec = irtree.IRTreeSpec(mapping=np.array([[0., np.nan], [1., 0.], [1., 1.]]))
    nodes = [LogitThreePLNode(), irtree.ProbitThreePLNode()]
    with pytest.raises(NotImplementedError, match="probit 3PL nodes only"):
        Constraints(c_shared=[[1, 2]]).validate(spec, nodes)
    Constraints(c_shared=[[1, 2]]).validate(
        spec, [irtree.ProbitThreePLNode(), irtree.ProbitThreePLNode()])
