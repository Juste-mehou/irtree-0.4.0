"""
tests/test_logit_conversion.py
==============================
Axis-1 output check for the probit -> logistic-equivalent rescaling
(``posterior.logit_scale_factors`` / ``IRTreeFit.to_logit``), TESTING.md §9.3.

Known answer, from the premises Phi(x) ~ L(1.702 x) and
eta = a (theta - b + offset):

* a dimension whose nodes are all probit without slopes stretches its metric:
  theta, b, thresholds, beta x 1.702, sigma2 x 1.702**2;
* otherwise the metric is kept and probit slopes x 1.702;
* correlations, guessing and logit nodes are unchanged;
* covariance entries take the product of the two factors.

The layout-level checks fit no model; one tiny fit checks that to_logit()
is the summary times the factors.
"""

import dataclasses

import numpy as np
import pytest

import irtree
from irtree.data_prep import resolve_covariate_indices
from irtree.mcmc_sampler import _build_chain_layout, _effective_priors
from irtree.posterior import LOGIT_D, logit_scale_factors

pytestmark = pytest.mark.fast

D = LOGIT_D


def _factors(spec, nodes, priors=None, constraints=None, X=None, Y=None):
    rng = np.random.default_rng(0)
    P, I = 30, 6
    if Y is None:
        Y = rng.integers(1, spec.mapping.shape[0] + 1, size=(P, I))
    kw = {}
    if X is not None:
        kw = dict(X=X, X_names=[f"x{j + 1}" for j in range(X.shape[1])])
    data = irtree.expand_data(Y, spec, categories_start=1, **kw)
    priors = priors or irtree.build_default_priors(spec, nodes)
    cov = resolve_covariate_indices(nodes, data, spec=spec, constraints=constraints)
    names, _ = _build_chain_layout(data, nodes, _effective_priors(priors, constraints, data.n_nodes),
                                   spec=spec, constraints=constraints, cov_index=cov)
    f, notes = logit_scale_factors(names, data, spec, nodes, priors, constraints)
    return dict(zip(names, f, strict=True)), notes


def _by_prefix(fac, prefix):
    vals = {v for k, v in fac.items() if k.startswith(prefix)}
    assert vals, f"no columns start with {prefix!r}"
    return vals


def test_rasch_probit_stretches_metric_and_2pl_scales_slopes():
    spec = irtree.disengagement_tree(binary=True)
    nodes = [irtree.ProbitRaschNode(person_covariates=["x1"]), irtree.ProbitTwoPLNode()]
    X = np.random.default_rng(1).normal(size=(30, 1))
    fac, notes = _factors(spec, nodes, X=X)
    # node 1: Rasch probit, its own dimension -> metric x 1.702
    assert _by_prefix(fac, "theta_1_") == {D}
    assert _by_prefix(fac, "b_1_") == {D}
    assert fac["sigma2_1"] == pytest.approx(D ** 2)
    assert _by_prefix(fac, "beta_1") == {D}
    # node 2: 2PL probit -> metric kept, slopes x 1.702
    assert _by_prefix(fac, "theta_2_") == {1.0}
    assert _by_prefix(fac, "b_2_") == {1.0}
    assert fac["sigma2_2"] == 1.0
    assert _by_prefix(fac, "a_2_") == {D}
    assert notes == []


def test_logit_nodes_unchanged():
    spec = irtree.disengagement_tree(binary=True)
    nodes = [irtree.LogitRaschNode(), irtree.LogitTwoPLNode()]
    fac, _ = _factors(spec, nodes)
    assert set(fac.values()) == {1.0}


def test_3pl_guessing_unchanged():
    spec = irtree.disengagement_tree(binary=True)
    nodes = [irtree.ProbitRaschNode(), irtree.ProbitThreePLNode()]
    fac, _ = _factors(spec, nodes)
    assert _by_prefix(fac, "c_2_") == {1.0}
    assert _by_prefix(fac, "a_2_") == {D}


def test_grm_thresholds_follow_the_metric():
    M = 4
    spec = irtree.IRTreeSpec(mapping=np.arange(M, dtype=float)[:, None])
    fac, _ = _factors(spec, [irtree.GRMNode(n_outcomes=M)])
    assert _by_prefix(fac, "b_1_") == {D}          # all thresholds
    assert fac["sigma2_1"] == pytest.approx(D ** 2)


def test_shared_dimension_rasch_with_2pl_keeps_metric_and_notes_implied_slope():
    spec = dataclasses.replace(irtree.disengagement_tree(binary=True), node_to_latent=(1, 1))
    nodes = [irtree.ProbitRaschNode(), irtree.ProbitTwoPLNode()]
    fac, notes = _factors(spec, nodes)
    assert _by_prefix(fac, "theta_1&2_") == {1.0}
    assert _by_prefix(fac, "b_1_") == {1.0}         # Rasch node, but metric kept
    assert _by_prefix(fac, "a_2_") == {D}
    assert any("node 1" in n and "common value" in n for n in notes)


def test_correlations_unchanged_covariances_take_products():
    spec = irtree.disengagement_tree(binary=True)
    nodes = [irtree.ProbitRaschNode(), irtree.ProbitTwoPLNode()]
    pr = irtree.build_default_priors(spec, nodes)
    pr.set("theta.correlation", irtree.LKJPrior(eta=2.0))
    fac, _ = _factors(spec, nodes, priors=pr)
    assert fac["rho_1_2"] == 1.0

    pr = irtree.build_default_priors(spec, nodes)
    pr.set("theta.correlation", irtree.InverseWishartPrior(df=4, scale=np.eye(2)))
    fac, _ = _factors(spec, nodes, priors=pr)
    assert fac["Sigma_theta_1_1"] == pytest.approx(D * D)
    assert fac["Sigma_theta_1_2"] == pytest.approx(D)
    assert fac["Sigma_theta_2_2"] == pytest.approx(1.0)


def test_every_column_gets_a_finite_factor_in_a_standard_model():
    spec = irtree.disengagement_tree(binary=False)
    nodes = [irtree.ProbitRaschNode(), irtree.ProbitTwoPLNode(), irtree.ProbitRaschNode()]
    fac, _ = _factors(spec, nodes)
    assert np.isfinite(list(fac.values())).all()


def test_to_logit_is_summary_times_factor():
    rng = np.random.default_rng(3)
    spec = irtree.disengagement_tree(binary=True)
    Y = rng.integers(1, 4, size=(60, 8))
    data = irtree.expand_data(Y, spec, categories_start=1)
    nodes = [irtree.ProbitRaschNode(), irtree.ProbitTwoPLNode()]
    fit = irtree.irtree_fit(data, spec, nodes, n_iter=300, n_warmup=100,
                            warmup_mode="fixed", n_chains=1, seed=1, progress=False)
    base, conv = fit.summary_df(), fit.to_logit()
    f = conv["factor"].to_numpy()
    for col in ("mean", "sd", "q2.5", "q50", "q97.5", "hdi_lo", "hdi_hi"):
        np.testing.assert_allclose(conv[col].to_numpy(), base[col].to_numpy() * f)
    np.testing.assert_array_equal(conv["n_eff"].to_numpy(), base["n_eff"].to_numpy())
    assert "1.702" in conv.attrs["scale"]
    # round trip: dividing by the factor recovers the probit summary
    np.testing.assert_allclose(conv["mean"].to_numpy() / f, base["mean"].to_numpy())


def test_b_shared_across_metrics_keeps_the_fitted_metric():
    # b pooled between a Rasch probit node and a 2PL node: one b array needs one
    # metric, so the Rasch dimension keeps its fitted metric (factor 1) and the
    # Rasch node gets the implied common slope 1.702 instead of a NaN.
    spec = irtree.disengagement_tree(binary=True)
    nodes = [irtree.ProbitRaschNode(), irtree.ProbitTwoPLNode()]
    fac, notes = _factors(spec, nodes, constraints=irtree.Constraints(b_shared=[[1, 2]]))
    assert _by_prefix(fac, "b_1&2_") == {1.0}
    assert _by_prefix(fac, "theta_1_") == {1.0}
    assert _by_prefix(fac, "a_2_") == {D}
    assert any("node 1" in n and "common value" in n for n in notes)
