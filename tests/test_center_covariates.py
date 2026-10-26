"""center_covariates and load_data(center=...)."""
import numpy as np
import pandas as pd
import pytest

import irtree

pytestmark = pytest.mark.fast


def test_centers_observed_values_and_keeps_missing_and_constants():
    X = np.array([[1, 0, 1.0], [1, 1, 3], [1, np.nan, 5], [1, 1, 99]])
    r = irtree.center_covariates(X, names=["one", "male", "age"], verbose=False)
    assert list(r.centered) == [False, True, True]
    np.testing.assert_allclose(r.values[:, 0], 1.0)                 # intercept untouched
    assert np.isnan(r.values[2, 1]) and np.isnan(r.values[3, 2])     # NaN and code 99 missing
    np.testing.assert_allclose(np.nanmean(r.values[:, 1:], axis=0), 0.0, atol=1e-12)


def test_scale_and_column_selection_and_w_layout():
    X = np.array([[1.0, 10], [2, 20], [3, 30]])
    r = irtree.center_covariates(X, columns=[1], scale=True, verbose=False)
    np.testing.assert_allclose(r.values[:, 0], X[:, 0])
    np.testing.assert_allclose(r.values[:, 1], [-1, 0, 1])
    W = np.random.default_rng(0).normal(3, 1, (4, 5, 2))
    np.testing.assert_allclose(irtree.center_covariates(W, verbose=False).values.mean(axis=(0, 1)), 0, atol=1e-12)


def test_load_data_center_option():
    spec = irtree.disengagement_tree(binary=True)
    Y = pd.DataFrame({"person_id": list("abcd"), "i1": [0, 1, 2, 1], "i2": [2, 2, 0, 1]})
    xc = pd.DataFrame({"person_id": list("abcd"), "anger": [10, 20, 30, 40], "male": [1, 0, 1, 1]})
    qc = pd.DataFrame({"item_id": ["i1", "i2"], "intercept": [1, 1], "load": [2.0, 5.0]})
    d = irtree.load_data(Y, spec=spec, person_covariates=xc, item_covariates=qc,
                         center={"X": ["anger"], "Q": True}, verbose=False)
    np.testing.assert_allclose(d.X[:, 0], [-15, -5, 5, 15])
    np.testing.assert_allclose(d.X[:, 1], [1, 0, 1, 1])              # not selected
    np.testing.assert_allclose(d.Q, [[1, -1.5], [1, 1.5]])           # intercept kept
    raw = irtree.load_data(Y, spec=spec, person_covariates=xc, verbose=False)
    np.testing.assert_allclose(raw.X[:, 0], [10, 20, 30, 40])        # default: unchanged
