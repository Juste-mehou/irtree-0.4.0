"""Pure-Gibbs chains burn in like MH chains warm up.

With no MH block there is no freeze test, so in evidence mode (the default)
the burn-in is n_warmup clamped to [2*M*W, n_iter - min_keep*thin_by], or
2*M*W when n_warmup = 0.  Keeping draws from iteration 0 would put the
transient from the starting values into the posterior sample.
"""
import numpy as np
import pytest

import irtree
from irtree import mcmc_adapt
from irtree.data_prep import expand_data

FLOOR = 2 * mcmc_adapt.EVIDENCE_NEED_M * mcmc_adapt.EVIDENCE_WINDOW_W


def _setup():
    rng = np.random.default_rng(0)
    spec = irtree.disengagement_tree(binary=True)
    Y = rng.integers(1, 4, size=(60, 6))
    d = expand_data(Y, spec, categories_start=1)
    nodes = [irtree.ProbitRaschNode(), irtree.ProbitRaschNode()]
    return d, spec, nodes, irtree.build_default_priors(spec, nodes)


def _run(n_iter, **kw):
    d, spec, nodes, pr = _setup()
    return irtree.run_chain(d, nodes, pr, n_iter=n_iter, seed=1, spec=spec,
                            progress=False, **kw)


def test_default_burnin_is_the_floor():
    res = _run(1700, min_keep=100)
    assert res.effective_warmup == FLOOR
    assert res.chain.shape[0] == 1700 - FLOOR


@pytest.mark.slow
def test_user_burnin_is_used_and_clamped():
    assert _run(1700, min_keep=100, n_warmup=900).effective_warmup == 900
    assert _run(1700, min_keep=100, n_warmup=50).effective_warmup == FLOOR
    with pytest.warns(UserWarning):
        res = _run(1700, min_keep=1000, n_warmup=1500)
    assert res.effective_warmup == 700          # n_iter - min_keep
    assert res.chain.shape[0] == 1000


def test_short_pure_gibbs_run_raises_in_evidence_mode():
    with pytest.raises(ValueError, match="needs n_iter"):
        _run(500)


def test_fixed_mode_keeps_exact_burnin():
    res = _run(100, warmup_mode="fixed", n_warmup=20)
    assert res.chain.shape[0] == 80
