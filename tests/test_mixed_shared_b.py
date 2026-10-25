"""Mixed-format shared b.

Nodes with the same per-item formats (binary and polytomous items) share one
NodeBParams.  Binary difficulties: exact Gibbs from the GRM members' latent
statistics, or per-item MH adding the GPCM members' likelihood.  Polytomous
thresholds: pooled Cowles step, GRM members' Z redrawn, exact pooled Gibbs
when every member is GRM.
"""
import warnings

import numpy as np
import pytest

import irtree
from irtree.constraints import Constraints
from irtree.data_prep import expand_data


def _data(kind, seed=0, P=700, I=12):
    rng = np.random.default_rng(seed)
    pio = np.array([2, 3] * (I // 2))
    bi, pi_ = np.where(pio == 2)[0], np.where(pio == 3)[0]
    spec = irtree.IRTreeSpec(mapping=np.array([[k // 3, k % 3] for k in range(9)], float))
    b_bin = rng.normal(0, 0.7, len(bi))
    tau = np.sort(rng.normal(0, 0.7, (len(pi_), 2)), axis=1) + np.array([-0.4, 0.4])

    def grm(th):
        y = np.empty((P, I), int)
        Z = th[:, None] + rng.normal(size=(P, I))
        y[:, bi] = Z[:, bi] > b_bin
        y[:, pi_] = (Z[:, pi_][:, :, None] > tau[None]).sum(-1)
        return y

    def gpcm(th):
        y = np.empty((P, I), int)
        y[:, bi] = rng.random((P, len(bi))) < 1 / (1 + np.exp(-(th[:, None] - b_bin)))
        eta = th[:, None, None] - tau[None]
        ln = np.concatenate([np.zeros((P, len(pi_), 1)), np.cumsum(eta, 2)], 2)
        pr = np.exp(ln - ln.max(2, keepdims=True))
        pr /= pr.sum(2, keepdims=True)
        y[:, pi_] = (rng.random((P, len(pi_), 1)) > np.cumsum(pr, 2)).sum(2)
        return y

    y1 = gpcm(rng.normal(size=P)) if kind == "gpcm" else grm(rng.normal(size=P))
    Y = 1 + 3 * y1 + grm(rng.normal(size=P))
    return spec, expand_data(Y, spec, categories_start=1), pio, bi, pi_, b_bin, tau


@pytest.mark.slow
@pytest.mark.parametrize("kind", ["grm", "gpcm"])
def test_mixed_format_shared_b_is_recovered(kind):
    warnings.filterwarnings("ignore")
    spec, d, pio, bi, pi_, b_bin, tau = _data(kind)
    n1 = (irtree.GPCMNode(pseudo_item_outcomes=pio, discrimination=False)
          if kind == "gpcm" else irtree.GRMNode(pseudo_item_outcomes=pio))
    nodes = [n1, irtree.GRMNode(pseudo_item_outcomes=pio)]
    res = irtree.run_chain(d, nodes, irtree.build_default_priors(spec, nodes),
                           n_iter=1800, seed=1, spec=spec, progress=False,
                           warmup_mode="fixed", n_warmup=600,
                           constraints=Constraints(b_shared=[[1, 2]]))
    nm, ch = res.param_names, res.chain
    eb = ch[:, [nm.index(f"b_1&2_{i + 1}") for i in bi]].mean(0)
    et = ch[:, [nm.index(f"b_1&2_{i + 1}_tau{k}") for i in pi_ for k in (1, 2)]]
    assert np.sqrt(np.mean((eb - b_bin) ** 2)) < 0.1
    assert np.sqrt(np.mean((et.mean(0).reshape(-1, 2) - tau) ** 2)) < 0.1
    assert np.all(np.diff(et.reshape(len(ch), -1, 2), axis=2) > 0)   # ordered
    if kind == "gpcm":
        assert 0.25 < res.accept_rates["bbin_mh_0"] < 0.65
