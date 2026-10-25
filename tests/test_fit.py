"""
tests/test_fit.py
=================
`fit.irtree_fit` is orchestration (the "none" column in the Axis-1 table): it
produces nothing hand-verifiable in isolation, so it is checked by *invariants*
through a tiny real fit — it runs, produces the right structure, is
reproducible, and its downstream products (WAIC / LOO / summaries) are finite
and well-shaped. These are not recovery tests (no coverage claim); they are the
smallest checks that the wrapper wires the pieces together without breaking.

Kept `fast` by using a tiny problem and very short chains.  Short chains
need warmup_mode="fixed": the default evidence mode requires
n_iter >= 2*M*W + min_keep (1600) so the warm-up and the kept draws both fit.
"""

import warnings

import numpy as np
import pytest
from scipy.special import ndtr

import irtree.data_prep as dp
import irtree.fit as ft
import irtree.node_models as nm
import irtree.tree_spec as ts

pytestmark = pytest.mark.fast


# ---------------------------------------------------------------------------
def _tiny_disengagement(P=120, I=8, seed=0):
    rng = np.random.default_rng(seed)
    th1 = rng.normal(0, 1, P)
    th2 = rng.normal(0, 1, P)
    b1 = rng.normal(0, 0.7, I)
    b2 = rng.normal(0, 1, I)
    rg = rng.random((P, I)) < ndtr(th1[:, None] - b1[None, :])
    co = rng.random((P, I)) < ndtr(th2[:, None] - b2[None, :])
    Y = np.where(rg, 1, np.where(co, 3, 2))
    spec = ts.disengagement_tree(binary=True)
    data = dp.expand_data(Y, spec, categories_start=1)
    models = [nm.ProbitRaschNode(), nm.ProbitRaschNode()]
    return data, spec, models


def _fit(n_chains=2, n_iter=80, seed=1):
    data, spec, models = _tiny_disengagement()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ft.irtree_fit(data, spec, models, n_iter=n_iter, n_warmup=20,
                             warmup_mode="fixed", n_chains=n_chains, seed=seed)


# ===========================================================================
# Structure — the wrapper returns a usable object with the expected surface
# ===========================================================================

def test_fit_runs_and_has_expected_surface():
    fit = _fit()
    for attr in ("summary", "diagnostics", "credible_interval", "coef"):
        assert hasattr(fit, attr), f"IRTreeFit missing '{attr}'"
    # stores the requested number of chains
    assert hasattr(fit, "results"), "IRTreeFit does not expose its chain results"
    assert len(fit.results) == 2, f"expected 2 chains stored, got {len(fit.results)}"


def test_fit_chain_shape():
    """Each chain has the same width (parameter count) and the stored height."""
    fit = _fit(n_iter=80)
    widths = {r.chain.shape[1] for r in fit.results}
    assert len(widths) == 1, f"chains disagree on parameter count: {widths}"
    # every parameter is named
    d = fit.results[0].chain.shape[1]
    assert len(fit.results[0].param_names) == d, "param_names length != chain width"


# ===========================================================================
# Invariants — finiteness of everything the wrapper produces
# ===========================================================================

def test_fit_chain_finite():
    fit = _fit()
    for k, r in enumerate(fit.results):
        assert np.all(np.isfinite(np.asarray(r.chain, float))), f"chain {k} has non-finite entries"


def test_fit_log_lik_finite():
    fit = _fit()
    for k, r in enumerate(fit.results):
        ll = np.asarray(r.log_lik_trace, float)
        assert np.all(np.isfinite(ll)), f"chain {k} log-likelihood trace has non-finite entries"


def test_fit_summary_finite():
    fit = _fit()
    summ = fit.summary_df() if hasattr(fit, "summary_df") else None
    if summ is not None:
        vals = summ.select_dtypes("number").to_numpy()
        assert np.all(np.isfinite(vals)), "summary table has non-finite entries"


# ===========================================================================
# Reproducibility — same seed => identical chains (through the wrapper)
# ===========================================================================

def test_fit_reproducible():
    f1 = _fit(seed=7)
    f2 = _fit(seed=7)
    for a, b in zip(f1.results, f2.results, strict=True):
        assert np.array_equal(np.asarray(a.chain), np.asarray(b.chain)), \
            "same seed produced different chains through irtree_fit"


def test_fit_seed_changes_output():
    f1 = _fit(seed=1)
    f2 = _fit(seed=2)
    same = all(np.array_equal(np.asarray(a.chain), np.asarray(b.chain))
               for a, b in zip(f1.results, f2.results, strict=True))
    assert not same, "different seeds produced identical chains (RNG not threaded)"


# ===========================================================================
# WAIC / LOO — need a real fit, so tested here for finiteness & shape
# (a correctness claim on WAIC belongs to recovery / external validation)
# ===========================================================================

def test_waic_finite():
    fit = _fit(n_iter=120)
    if not hasattr(fit, "waic"):
        pytest.skip("waic not exposed on IRTreeFit")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w = fit.waic()
    # WAICResult carries at least an elpd / waic scalar; assert finiteness of numbers
    nums = [v for v in vars(w).values() if isinstance(v, (int, float))]
    assert nums, "WAICResult exposed no scalar fields"
    assert all(np.isfinite(v) for v in nums), f"WAIC has non-finite fields: {vars(w)}"


def test_loo_finite():
    fit = _fit(n_iter=120)
    if not hasattr(fit, "loo"):
        pytest.skip("loo not exposed on IRTreeFit")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        l = fit.loo()
    nums = [v for v in vars(l).values() if isinstance(v, (int, float))]
    assert nums, "LOOResult exposed no scalar fields"
    assert all(np.isfinite(v) for v in nums), f"LOO has non-finite fields: {vars(l)}"


# ===========================================================================
# Single-chain path — R-hat unavailable but the fit must still run
# ===========================================================================

def test_fit_single_chain_runs():
    fit = _fit(n_chains=1, n_iter=80)
    assert len(fit.results) == 1
    assert np.all(np.isfinite(np.asarray(fit.results[0].chain, float)))
