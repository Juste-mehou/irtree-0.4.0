"""The inverse-Wishart default scale is 0.01*I.

Under IW(Psi0, G+1) the posterior mean of Sigma is (Psi0 + S) / N, so Psi0
acts as additive pseudo-scatter.  Psi0 = I was considered and rejected: it
biases small variances upward and correlations toward 0 when N is small.
"""
import numpy as np
import pytest

from irtree import gibbs_samplers as gs
from irtree.priors import InverseWishartPrior


def test_default_scale_and_df():
    scale, df = InverseWishartPrior().resolved_params(3)
    np.testing.assert_array_equal(scale, 0.01 * np.eye(3))
    assert df == 4
    v = InverseWishartPrior.vague(2)
    np.testing.assert_array_equal(v.scale, 0.01 * np.eye(2))


@pytest.mark.slow
def test_default_barely_moves_small_variance_groups():
    rng = np.random.default_rng(0)
    N, G = 10, 2
    C = np.array([[1, .5], [.5, 1]])
    B = np.sqrt(0.1) * rng.normal(size=(N, G)) @ np.linalg.cholesky(C).T
    S = B.T @ B
    scale, df = InverseWishartPrior().resolved_params(G)
    draws = np.array([gs.gibbs_Sigma_invwishart(B, np.zeros(G), scale, df, rng)
                      for _ in range(20000)])
    expected = (scale + S) / (df + N - G - 1)       # IW posterior mean
    np.testing.assert_allclose(draws.mean(0), expected, rtol=0.05)
    # Psi0 adds about 1-2% to each scatter diagonal (S_kk ~ N * 0.1 = 1)
    assert np.all(np.diag(scale) / np.diag(S) < 0.02)
