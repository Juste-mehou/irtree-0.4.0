"""
tests/test_api_consistency.py
=============================
One name and one number per quantity across the API:

* ``Priors.free_params(constraints)`` lists exactly the estimated
  hyperparameter columns of the chain layout;
* each b-group's variance column holds that group's variance (it held another
  group's when a shared group was not first);
* estimated covariate-group variances are stored in the chain;
* the summary's n_eff is the diagnostics' bulk ESS across chains;
* ``irtree_fit`` uses the same ``categories_start`` default as
  ``expand_data`` and ``load_data``.
"""

import inspect
import warnings

import numpy as np
import pytest

import irtree
from irtree import Constraints, InverseWishartPrior, LKJPrior, NormalPrior
from irtree.data_prep import resolve_covariate_indices
from irtree.diagnostics import _batch_diagnostics
from irtree.mcmc_sampler import _build_chain_layout, _effective_priors

pytestmark = pytest.mark.fast

NAN = float("nan")
_HYPER = ("b.variance_", "a.variance_", "rho_", "Sigma_", "b.rho_", "a.rho_",
          "beta.variance_", "kappa.variance_", "gamma.variance_")


def _four_node_spec():
    M = [[0, 0, NAN, NAN], [0, 1, NAN, NAN], [1, NAN, 0, NAN],
         [1, NAN, 1, 0], [1, NAN, 1, 1]]
    return irtree.IRTreeSpec(mapping=M)


def _layout_hyper(spec, nodes, priors, constraints, data):
    cov = resolve_covariate_indices(nodes, data, spec=spec, constraints=constraints)
    names, _ = _build_chain_layout(data, nodes, _effective_priors(priors, constraints, len(nodes)),
                                   spec=spec, constraints=constraints, cov_index=cov)
    return [n for n in names if n.startswith(_HYPER)], names


def _case(name):
    rng = np.random.default_rng(0)
    if name == "covariates":
        spec = irtree.disengagement_tree(binary=True)
        nodes = [irtree.ProbitRaschNode(person_covariates=True), irtree.ProbitTwoPLNode()]
        Y = rng.integers(1, 4, size=(40, 6))
        data = irtree.expand_data(Y, spec, categories_start=1, X=rng.normal(size=(40, 2)))
        pr = irtree.build_default_priors(spec, nodes)
        pr.set("theta.correlation", LKJPrior(eta=2.0))
        return spec, nodes, pr, None, data
    if name == "ers_shared":
        spec = irtree.ers_tree()
        nodes = [irtree.ProbitRaschNode() for _ in range(3)]
        Y = rng.integers(1, 5, size=(40, 6))
        data = irtree.expand_data(Y, spec, categories_start=1)
        pr = irtree.build_default_priors(spec, nodes)
        pr.set("theta.correlation", InverseWishartPrior())
        return spec, nodes, pr, Constraints(b_shared=[[2, 3]]), data
    if name == "corr_groups":
        spec = _four_node_spec()
        nodes = [irtree.ProbitTwoPLNode() for _ in range(4)]
        Y = rng.integers(0, 5, size=(40, 6))
        data = irtree.expand_data(Y, spec, categories_start=0)
        pr = irtree.build_default_priors(spec, nodes)
        pr.set("b.correlation", InverseWishartPrior())
        pr.set("a.correlation", LKJPrior(eta=1.0))
        return spec, nodes, pr, Constraints(b_corr=[[1, 2]], a_corr=[[3, 4]]), data
    raise KeyError(name)


@pytest.mark.parametrize("name", ["covariates", "ers_shared", "corr_groups"])
def test_free_params_are_the_chain_hyperparameter_columns(name):
    spec, nodes, pr, con, data = _case(name)
    free = pr.free_params(con)
    hyper, all_names = _layout_hyper(spec, nodes, pr, con, data)
    assert [n for n in free if not n.startswith("sigma2_")] == hyper
    assert all(n in all_names for n in free)


def _quick_fit(data, spec, nodes, pr, con, n_chains=1, n_iter=300):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return irtree.irtree_fit(data, spec, nodes, priors=pr, constraints=con,
                                 n_iter=n_iter, n_warmup=100, warmup_mode="fixed",
                                 n_chains=n_chains, seed=1, progress=False)


def test_b_variance_columns_follow_their_group():
    rng = np.random.default_rng(0)
    spec = _four_node_spec()
    data = irtree.expand_data(rng.integers(0, 5, size=(150, 6)), spec, categories_start=0)
    nodes = [irtree.ProbitRaschNode() for _ in range(4)]
    pr = irtree.build_default_priors(spec, nodes)
    pr.set("b", NormalPrior(mean=0.0, variance=1.0), node=4)     # node 4: fixed variance
    con = Constraints(b_shared=[[2, 3]])
    fit = _quick_fit(data, spec, nodes, pr, con)
    res = fit.results[0]
    last = res.chain[-1].astype(float)
    groups = con.b_groups_for(4)
    for g, grp in enumerate(groups):
        label = "&".join(str(r + 1) for r in grp)
        col = f"b.variance_{label}"
        if col in res.param_names:
            assert last[res.param_names.index(col)] == pytest.approx(
                res.final_params["b.variance"][g], rel=1e-5)


def test_coefficient_variance_in_chain():
    spec, nodes, pr, con, data = _case("covariates")
    fit = _quick_fit(data, spec, nodes, pr, con)
    res = fit.results[0]
    x = res.chain[:, res.param_names.index("beta.variance_1")]
    assert np.all(x > 0) and np.std(x) > 0


def test_summary_n_eff_is_the_bulk_ess_across_chains():
    spec, nodes, pr, con, data = _case("covariates")
    fit = _quick_fit(data, spec, nodes, pr, con, n_chains=2)
    chains = np.stack([r.chain.astype(float) for r in fit.results], axis=1)
    bulk = _batch_diagnostics(chains, want_rhat=False, want_tail=False)["ess_bulk"]
    n_eff = np.array([s.n_eff for s in fit._get_summary().values()])
    np.testing.assert_allclose(n_eff, bulk, rtol=1e-10)
    rep = fit.diagnostics()
    np.testing.assert_allclose(n_eff, [rep.ess_bulk[n] for n in rep.param_names], rtol=1e-6)


def test_categories_start_defaults_agree():
    d = {f: inspect.signature(f).parameters["categories_start"].default
         for f in (irtree.irtree_fit, irtree.expand_data, irtree.load_data)}
    assert len(set(d.values())) == 1, d
