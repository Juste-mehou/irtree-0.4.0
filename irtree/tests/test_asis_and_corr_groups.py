"""Items 12 and 13.

12: interweaving (ASIS) steps for kappa/b and beta/theta.  Each is an exact
    Gibbs step in the centered parameterization (eta = b - Q kappa,
    zeta = theta + X beta) and leaves the likelihood unchanged.
13: b- and a-correlation groups containing logit/GPCM nodes.
Also: a fixed theta variance is now the variance used (it stayed at 1).
"""
import warnings

import numpy as np
import pytest
from scipy.stats import norm

import irtree
from irtree.constraints import Constraints
from irtree.data_prep import expand_data
from irtree.diagnostics import ess
from irtree.priors import InverseWishartPrior, NormalPrior


def _sig(z):
    return 1 / (1 + np.exp(-z))


@pytest.mark.slow
def test_interweaving_mixes_kappa_and_beta():
    """Without ASIS: ESS(kappa) ~ 9, ESS(beta) ~ 170 of 3400 here."""
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)
    P, I = 600, 25
    spec = irtree.IRTreeSpec(mapping=np.array([[0.], [1.]]))
    q, x = rng.normal(size=I), rng.normal(size=P)
    th, b = rng.normal(0, 0.5, P), rng.normal(0, 0.4, I)
    Y = (rng.random((P, I)) < norm.cdf(th[:, None] + x[:, None] - b + 0.8 * q)).astype(int) + 1
    d = expand_data(Y, spec, categories_start=1, X=x[:, None], Q=q[:, None])
    nodes = [irtree.ProbitRaschNode(person_covariates=True, item_covariates=True)]
    res = irtree.run_chain(d, nodes, irtree.build_default_priors(spec, nodes),
                           n_iter=4000, seed=1, spec=spec, progress=False)
    nm, ch = res.param_names, res.chain
    k = ch[:, [i for i, n in enumerate(nm) if n.startswith("kappa_")][0]]
    be = ch[:, [i for i, n in enumerate(nm) if n.startswith("beta_")][0]]
    assert ess(k)[0] > 1000 and ess(be)[0] > 300
    assert abs(be.mean() - 1.0) < 0.15 and abs(k.mean() - 0.8) < 0.3


def test_fixed_theta_variance_is_used():
    warnings.filterwarnings("ignore")
    spec = irtree.IRTreeSpec(mapping=np.array([[0.], [1.]]))
    d = expand_data(np.random.default_rng(0).integers(1, 3, (100, 6)), spec,
                    categories_start=1)
    nodes = [irtree.ProbitRaschNode()]
    pr = irtree.build_default_priors(spec, nodes)
    pr.set("theta", NormalPrior(mean=0.0, variance=0.25))
    res = irtree.run_chain(d, nodes, pr, n_iter=30, seed=1, spec=spec,
                           progress=False, warmup_mode="fixed", n_warmup=5)
    assert res.final_params["sigma2"][0] == 0.25
    col = res.param_names.index("sigma2_1")
    assert np.all(res.chain[:, col] == 0.25)


@pytest.mark.slow
def test_correlated_b_with_a_logit_node():
    """A b-correlation group whose member is a logit node."""
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)
    P, I = 800, 30
    C = np.array([[1, .7], [.7, 1]])
    B = rng.normal(size=(I, 2)) @ np.linalg.cholesky(C).T
    th1, th2 = rng.normal(size=P), rng.normal(size=P)
    y1 = rng.random((P, I)) < _sig(th1[:, None] - B[:, 0])
    y2 = rng.random((P, I)) < norm.cdf(th2[:, None] - B[:, 1])
    Y = np.where(~y1, 1, np.where(~y2, 2, 3))
    spec = irtree.IRTreeSpec(mapping=np.array([[0., np.nan], [1., 0.], [1., 1.]]))
    d = expand_data(Y, spec, categories_start=1)
    nodes = [irtree.LogitRaschNode(), irtree.ProbitRaschNode()]
    pr = irtree.build_default_priors(spec, nodes)
    pr.set("b.correlation", InverseWishartPrior())
    res = irtree.run_chain(d, nodes, pr, n_iter=2000, seed=1, spec=spec, progress=False,
                           warmup_mode="fixed", n_warmup=500,
                           constraints=Constraints(b_corr=[[1, 2]]))
    nm, ch = res.param_names, res.chain
    S = ch[:, [i for i, n in enumerate(nm) if n.startswith("Sigma_b")]].mean(0)
    assert abs(S[1] / np.sqrt(S[0] * S[2]) - np.corrcoef(B.T)[0, 1]) < 0.15
    for k, pref in enumerate(("b_1_", "b_2_")):
        est = ch[:, [i for i, n in enumerate(nm) if n.startswith(pref)]].mean(0)
        assert np.corrcoef(est, B[:, k])[0, 1] > 0.97
    assert 0.3 < res.accept_rates["b_mh_0"] < 0.6
