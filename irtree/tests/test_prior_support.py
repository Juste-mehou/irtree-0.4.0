"""
tests/test_prior_support.py
===========================
Every prior that ``Priors.set`` accepts must be honoured by the sampler, and
every prior the sampler cannot honour must be rejected at ``set()`` time.

Known answers used here:

* An item nobody answered does not enter the likelihood, so its difficulty's
  marginal posterior is its prior.  With b ~ N(2, 1) the chain must average
  about 2 with SD about 1, at a node-level Gibbs draw, a pooled (shared) draw
  and a correlated draw.
* With a very informative prior on a correlation and little data, the
  posterior of rho sits at the prior.
"""

import warnings

import numpy as np
import pytest

import irtree
from irtree import (
    BetaPrior,
    Constraints,
    FixedCorrelation,
    HalfNormalPrior,
    InvGammaPrior,
    LogNormalPrior,
    NormalPrior,
    TruncNormalPrior,
)

pytestmark = pytest.mark.fast


def _data(P=150, I=6, seed=0, unanswered=True):
    rng = np.random.default_rng(seed)
    spec = irtree.disengagement_tree(binary=True)
    Y = rng.integers(1, 4, size=(P, I))
    if unanswered:
        Y[:, -1] = 99                       # last item: nobody answered
    return spec, irtree.expand_data(Y, spec, categories_start=1)


def _fit(data, spec, nodes, priors, constraints=None, n_iter=3000):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return irtree.irtree_fit(data, spec, nodes, priors=priors, constraints=constraints,
                                 n_iter=n_iter, n_warmup=500, warmup_mode="fixed",
                                 n_chains=1, seed=3, progress=False)


def _draws(fit, name):
    res = fit.results[0]
    return res.chain[:, res.param_names.index(name)].astype(float)


# ---------------------------------------------------------------------------
# 1. the b prior mean reaches every Gibbs path
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("second", ["rasch", "2pl"])
def test_b_prior_mean_node_level(second):
    spec, data = _data()
    nodes = [irtree.ProbitRaschNode(),
             irtree.ProbitRaschNode() if second == "rasch" else irtree.ProbitTwoPLNode()]
    pr = irtree.build_default_priors(spec, nodes)
    pr.set("b", NormalPrior(mean=2.0, variance=1.0))
    fit = _fit(data, spec, nodes, pr)
    for node in (1, 2):
        x = _draws(fit, f"b_{node}_6")
        assert abs(x.mean() - 2.0) < 0.15, (node, x.mean())
        assert 0.85 < x.std() < 1.15, (node, x.std())


def test_b_prior_mean_pooled_shared_b():
    spec, data = _data()
    nodes = [irtree.ProbitRaschNode(), irtree.ProbitRaschNode()]
    pr = irtree.build_default_priors(spec, nodes)
    pr.set("b", NormalPrior(mean=2.0, variance=1.0))
    fit = _fit(data, spec, nodes, pr, constraints=Constraints(b_shared=[[1, 2]]))
    x = _draws(fit, "b_1&2_6")
    assert abs(x.mean() - 2.0) < 0.15 and 0.85 < x.std() < 1.15


def test_b_prior_mean_correlated_b():
    spec, data = _data()
    nodes = [irtree.ProbitRaschNode(), irtree.ProbitRaschNode()]
    pr = irtree.build_default_priors(spec, nodes)
    pr.set("b", NormalPrior(mean=2.0, variance=1.0), node=1)
    pr.set("b", NormalPrior(mean=-1.0, variance=1.0), node=2)
    pr.set("b.correlation", FixedCorrelation(rho=0.5))
    fit = _fit(data, spec, nodes, pr, constraints=Constraints(b_corr=[[1, 2]]))
    x1, x2 = _draws(fit, "b_1_6"), _draws(fit, "b_2_6")
    assert abs(x1.mean() - 2.0) < 0.15 and abs(x2.mean() + 1.0) < 0.15
    assert abs(np.corrcoef(x1, x2)[0, 1] - 0.5) < 0.1


# ---------------------------------------------------------------------------
# 2. bounded scalar correlation priors run and are honoured
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("prior, target", [
    (NormalPrior(mean=0.6, variance=0.0025, lower=-1, upper=1), 0.6),
    (BetaPrior(alpha=80.0, beta=20.0, lower=-1, upper=1), 0.6),   # mean -1 + 2(0.8)
])
def test_bounded_theta_correlation_prior(prior, target):
    spec, data = _data(P=40, unanswered=False)
    nodes = [irtree.ProbitRaschNode(), irtree.ProbitRaschNode()]
    pr = irtree.build_default_priors(spec, nodes)
    pr.set("theta.correlation", prior)
    fit = _fit(data, spec, nodes, pr, n_iter=2500)
    rho = _draws(fit, "rho_1_2")
    assert abs(rho.mean() - target) < 0.1, rho.mean()


def test_bounded_b_correlation_prior_is_updated():
    spec, data = _data(unanswered=False)
    nodes = [irtree.ProbitRaschNode(), irtree.ProbitRaschNode()]
    pr = irtree.build_default_priors(spec, nodes)
    pr.set("b.correlation", NormalPrior(mean=0.0, variance=0.3, lower=-1, upper=1))
    fit = _fit(data, spec, nodes, pr, constraints=Constraints(b_corr=[[1, 2]]), n_iter=800)
    rho = _draws(fit, "b.rho_1_2")
    assert np.std(rho) > 0.01            # it moves (it was never updated before)


# ---------------------------------------------------------------------------
# 3. priors the sampler cannot honour are rejected at set() time
# ---------------------------------------------------------------------------

def _priors(second="2pl"):
    spec = irtree.disengagement_tree(binary=True)
    nodes = [irtree.ProbitRaschNode(person_covariates=True),
             {"2pl": irtree.ProbitTwoPLNode, "3pl": irtree.ProbitThreePLNode}[second]()]
    return irtree.build_default_priors(spec, nodes)


@pytest.mark.parametrize("param, prior, exc", [
    ("theta", NormalPrior(mean=0.5, variance=1.0), ValueError),
    ("theta", NormalPrior(mean=0.0, variance=1.0, lower=-4, upper=4), NotImplementedError),
    ("b", NormalPrior(mean=0.0, variance=1.0, lower=-3, upper=3), NotImplementedError),
    ("b", NormalPrior(mean=0.0, variance=HalfNormalPrior(scale=1.0)), NotImplementedError),
    ("beta", NormalPrior(mean=0.0, variance=1.0, lower=-2, upper=2), NotImplementedError),
    ("beta", NormalPrior(mean=0.0, variance=HalfNormalPrior(scale=1.0)), NotImplementedError),
    ("b", LogNormalPrior(mean=0.0, variance=1.0), TypeError),
    ("X_missing", NormalPrior(mean=0.0, variance=InvGammaPrior()), NotImplementedError),
    ("a", HalfNormalPrior(scale=1.0), TypeError),
    ("a", LogNormalPrior(mean=0.0, variance=0.5, lower=0.2), TypeError),
])
def test_unsupported_priors_rejected(param, prior, exc):
    with pytest.raises(exc):
        _priors().set(param, prior)


def test_c_requires_beta():
    pr = _priors("3pl")
    with pytest.raises(TypeError):
        pr.set("c", TruncNormalPrior(mean=0.2, variance=0.01, lower=0.0, upper=1.0))
    pr.set("c", BetaPrior(alpha=4.0, beta=16.0))        # accepted


@pytest.mark.parametrize("param, prior", [
    ("theta", NormalPrior(mean=0.0, variance=HalfNormalPrior(scale=1.0))),
    ("b", NormalPrior(mean=1.5, variance=InvGammaPrior(alpha=2.0, beta=1.0))),
    ("a", TruncNormalPrior(mean=1.0, variance=0.5, lower=0.0)),
    ("a", LogNormalPrior(mean=0.0, variance=0.5)),
    ("X_missing", NormalPrior(mean=0.0, variance=2.0)),
])
def test_supported_priors_accepted(param, prior):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _priors().set(param, prior)
