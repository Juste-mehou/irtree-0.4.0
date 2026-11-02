"""DiagnosticReport tables, parameter-name parsing, and fit.draws()."""
import contextlib
import io
import warnings

import numpy as np
import pytest

import irtree
from irtree.diagnostics import parse_param_name

pytestmark = pytest.mark.fast


@pytest.mark.parametrize("name, kind, node, index, sub", [
    ("theta_1&2_7", "theta", "1&2", 7, None), ("b_2_3_thresh_1", "threshold", "2", 3, 1),
    ("b_2_3_tau2", "step", "2", 3, 2), ("a_1_4", "a", "1", 4, None),
    ("sigma2_1", "sigma2", "1", None, None), ("rho_1_2", "rho", "1_2", None, None),
    ("b.variance_2&3", "b.variance", "2&3", None, None), ("beta_1_escs_z", "beta", "1", "escs_z", None),
    ("X_missing_5_2", "X_missing", None, 5, 2), ("Sigma_theta_1_2", "Sigma_theta", "1_2", None, None),
])
def test_parse_param_name(name, kind, node, index, sub):
    assert parse_param_name(name) == {"kind": kind, "node": node, "index": index, "sub": sub}


def test_tables_and_draws():
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)
    spec = irtree.disengagement_tree(binary=True)
    data = irtree.expand_data(rng.integers(0, 3, size=(60, 5)), spec, categories_start=0)
    nodes = [irtree.LogitRaschNode(), irtree.ProbitTwoPLNode()]
    with contextlib.redirect_stdout(io.StringIO()):
        fit = irtree.irtree_fit(data, spec, nodes, n_iter=300, n_warmup=100, warmup_mode="fixed",
                                n_chains=2, seed=1, progress=False)
    rep = fit.diagnostics()
    t = rep.tables()
    assert set(t) == {"overview", "by_kind", "params", "flagged", "sampler"}
    assert len(t["params"]) == len(rep.param_names) and "theta" in set(t["params"]["kind"])
    assert set(t["sampler"]["chain"]) == {1, 2}
    assert (t["flagged"]["status"] != "PASS").all()
    assert "DiagnosticReport" in repr(rep)
    wide = fit.draws(kind="rho", log_lik=True)
    assert list(wide.columns) == ["chain", "draw", "rho_1_2", "log_lik"]
    arr, names = fit.draws_array(params=["rho_1_2", "a_2_1"])
    assert arr.shape[0] == 2 and arr.shape[2] == 2
    long = fit.draws(params=["rho_1_2", "a_2_1"], long=True)
    assert len(long) == arr.shape[0] * arr.shape[1] * 2
    np.testing.assert_allclose(long.loc[long.param == "a_2_1", "value"], arr[:, :, 1].ravel())


def test_report_aligns_chains_with_different_lengths():
    """Evidence-mode warm-up can leave chains with different numbers of draws."""
    import copy
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(1)
    spec = irtree.disengagement_tree(binary=True)
    data = irtree.expand_data(rng.integers(0, 3, size=(60, 5)), spec, categories_start=0)
    nodes = [irtree.ProbitRaschNode(), irtree.ProbitRaschNode()]
    with contextlib.redirect_stdout(io.StringIO()):
        fit = irtree.irtree_fit(data, spec, nodes, n_iter=300, n_warmup=100, warmup_mode="fixed",
                                n_chains=2, seed=1, progress=False)
    short = copy.copy(fit.results[1])
    short.chain = short.chain[25:]
    short.log_lik_trace = short.log_lik_trace[25:]
    rep = irtree.convergence_report([fit.results[0], short])
    assert rep.n_samples == short.chain.shape[0]
    assert len(rep.params) == len(rep.param_names)
