"""Observed values in partially-missing covariate rows must survive.

Only the missing cells are seeded (in data_prep and at sampler start-up);
zeroing the whole row while the per-cell mask still marks the other values
as observed would make the imputation treat those zeros as data.
"""
import numpy as np

import irtree
from irtree.data_prep import expand_data


def _data_with_partial_missing():
    rng = np.random.default_rng(0)
    P, I = 40, 6
    spec = irtree.disengagement_tree(binary=True)
    Y = rng.integers(1, 4, size=(P, I))
    X = rng.normal(size=(P, 3))
    X[0, 1] = np.nan            # person 0: only feature 1 missing
    X[1, :] = np.nan            # person 1: fully missing
    Q = rng.normal(size=(I, 2))
    Q[2, 0] = np.nan            # item 2: only feature 0 missing
    W = rng.normal(size=(P, I, 2))
    W[3, 4, 1] = np.nan         # cell (3, 4): only feature 1 missing
    return spec, Y, X, Q, W


def test_expand_data_keeps_observed_cells_of_partial_rows():
    spec, Y, X, Q, W = _data_with_partial_missing()
    d = expand_data(Y, spec, categories_start=1, X=X, Q=Q, W=W)

    # partially-missing rows keep their observed values
    assert d.X[0, 0] == X[0, 0] and d.X[0, 2] == X[0, 2]
    assert d.Q[2, 1] == Q[2, 1]
    assert d.W[3, 4, 0] == W[3, 4, 0]
    # missing cells are seeded with 0
    assert d.X[0, 1] == 0.0 and np.all(d.X[1] == 0.0)
    assert d.Q[2, 0] == 0.0 and d.W[3, 4, 1] == 0.0
    # row-level and cell-level masks are unchanged in meaning
    assert not d.X_observed[0] and not d.X_observed[1]
    assert d.X_feature_observed[0].tolist() == [True, False, True]
    assert d.n_missing_X == 2


def test_sampler_working_copy_keeps_observed_cells_of_partial_rows():
    spec, Y, X, Q, W = _data_with_partial_missing()
    d = expand_data(Y, spec, categories_start=1, X=X)
    nodes = [irtree.ProbitRaschNode(person_covariates=True),
             irtree.ProbitRaschNode(person_covariates=True)]
    res = irtree.run_chain(d, nodes, irtree.build_default_priors(spec, nodes),
                           n_iter=30, seed=1, spec=spec, progress=False,
                           warmup_mode="fixed", n_warmup=5)
    Xw = res.final_params["X_working"]
    # observed features of person 0 are untouched data
    assert Xw[0, 0] == X[0, 0] and Xw[0, 2] == X[0, 2]
    # the missing feature was imputed (moved off its 0.0 seed)
    assert Xw[0, 1] != 0.0
    # fully observed persons are untouched
    np.testing.assert_array_equal(Xw[2:], X[2:])
    # only the missing cells appear as X_missing chain columns
    miss_cols = [n for n in res.param_names if n.startswith("X_missing_1_")]
    assert miss_cols == ["X_missing_1_2"]
