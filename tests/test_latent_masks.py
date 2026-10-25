"""Theta and b statistics use only the cells where Z was drawn.

At a 3PL node Z is drawn only on the ability cells (U = 0); the guess
cells hold Z = 0, so using every relevant cell in the theta update, the
pooled b draw or the correlated b draw would enter those zeros as data.
Likewise theta's statistics must keep the binary items' difficulties at a
mixed GRM node.  All of these come from each node's latent view (mask,
discrimination, core).
"""
import warnings

import numpy as np
import pytest
from scipy.stats import norm

import irtree
from irtree.data_prep import expand_data
from irtree.priors import NormalPrior


def _run(d, spec, nodes, priors=None, n_iter=1500):
    warnings.filterwarnings("ignore")
    pr = priors or irtree.build_default_priors(spec, nodes)
    return irtree.run_chain(d, nodes, pr, n_iter=n_iter, seed=1, spec=spec,
                            progress=False, warmup_mode="fixed", n_warmup=600)


def _cols(res, prefix):
    return [i for i, n in enumerate(res.param_names) if n.startswith(prefix)]


@pytest.mark.slow
def test_probit_3pl_recovery():
    """Before: b bias -3.6, a bias +2.6, c 0.12 for truth 0.2."""
    rng = np.random.default_rng(0)
    P, I = 1000, 20
    th = rng.normal(size=P)
    a, b, c = rng.uniform(0.8, 1.8, I), rng.normal(0, 1, I), np.full(I, 0.2)
    Y = (rng.random((P, I)) < c + (1 - c) * norm.cdf(a * (th[:, None] - b))).astype(int) + 1
    spec = irtree.IRTreeSpec(mapping=np.array([[0.], [1.]]))
    d = expand_data(Y, spec, categories_start=1)
    nodes = [irtree.ProbitThreePLNode()]
    pr = irtree.build_default_priors(spec, nodes)
    pr.set("theta", NormalPrior(mean=0.0, variance=1.0))
    res = _run(d, spec, nodes, pr)
    ch = res.chain
    assert abs(np.mean(ch[:, _cols(res, "b_1_")].mean(0) - b)) < 0.2
    assert abs(np.mean(ch[:, _cols(res, "a_1_")].mean(0) - a)) < 0.3
    assert abs(ch[:, _cols(res, "c_1_")].mean() - 0.2) < 0.05
    assert np.polyfit(th, ch[:, _cols(res, "theta_1_")].mean(0), 1)[0] > 0.7


@pytest.mark.slow
def test_mixed_grm_theta_uses_binary_difficulties():
    """Before: theta mean -3.7 (truth ~0), correlation 0.46."""
    rng = np.random.default_rng(0)
    P, I = 800, 12
    pio = np.array([2, 3] * 6)
    bi, pi_ = np.where(pio == 2)[0], np.where(pio == 3)[0]
    th = rng.normal(size=P)
    b_bin = rng.normal(1.5, 0.5, len(bi))
    tau = np.sort(rng.normal(0, 0.8, (len(pi_), 2)), axis=1) + np.array([-0.5, 0.5])
    Z = th[:, None] + rng.normal(size=(P, I))
    Y = np.empty((P, I), int)
    Y[:, bi] = 1 + (Z[:, bi] > b_bin)
    Y[:, pi_] = 1 + (Z[:, pi_][:, :, None] > tau[None]).sum(-1)
    spec = irtree.IRTreeSpec(mapping=np.array([[0.], [1.], [2.]]))
    d = expand_data(Y, spec, categories_start=1)
    res = _run(d, spec, [irtree.GRMNode(pseudo_item_outcomes=pio)])
    est = res.chain[:, _cols(res, "theta_1_")].mean(0)
    assert abs(est.mean() - th.mean()) < 0.2
    assert np.corrcoef(est, th)[0, 1] > 0.85
