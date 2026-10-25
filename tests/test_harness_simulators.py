"""The shared simulator draws GRM and GPCM categories with the models' own probabilities.

Recovery tests judge the sampler against data from _harness.simulate_irtree,
so the simulator must match the node model: a probit GRM has
P(Y = m) = Phi(theta - tau_m) - Phi(theta - tau_{m+1}).
"""
import numpy as np

import irtree
from tests import _harness as h


def test_grm_simulator_matches_grm_node_probabilities():
    rng = np.random.default_rng(0)
    P = 200_000
    tau = np.array([[-0.5, 0.4], [0.0, 1.0]])
    Y = h._draw_grm(np.full(P, 0.3), tau, None, None, rng)
    emp = np.array([[np.mean(Y[:, i] == m) for m in range(3)] for i in range(2)])
    node = irtree.GRMNode(n_outcomes=3)
    exact = np.array([[np.exp(node.log_prob(np.array([[m, m]]), np.array([0.3]), tau))[0, i]
                       for m in range(3)] for i in range(2)])
    se = np.sqrt(exact * (1 - exact) / P)
    assert np.all(np.abs(emp - exact) < 4 * se), (emp, exact)


def test_gpcm_simulator_matches_gpcm_node_probabilities():
    rng = np.random.default_rng(1)
    P = 200_000
    steps = np.array([[-0.5, 0.4], [0.3, -0.2]])     # GPCM steps need not be ordered
    a = np.array([1.3, 0.7])
    Y = h._draw_gpcm(np.full(P, 0.2), steps, a, None, rng)
    emp = np.array([[np.mean(Y[:, i] == m) for m in range(3)] for i in range(2)])
    node = irtree.GPCMNode(n_outcomes=3)
    exact = np.array([[np.exp(node.log_prob(np.array([[m, m]]), np.array([0.2]), steps, a_r=a))[0, i]
                       for m in range(3)] for i in range(2)])
    se = np.sqrt(exact * (1 - exact) / P)
    assert np.all(np.abs(emp - exact) < 4 * se), (emp, exact)


def test_simulate_irtree_accepts_gpcm_nodes():
    rng = np.random.default_rng(2)
    spec = irtree.disengagement_tree(binary=False)
    P, I = 50, 4
    truth = [dict(theta=rng.normal(size=P), b=rng.normal(size=I)),
             dict(theta=rng.normal(size=P), b=np.sort(rng.normal(size=(I, 2)), axis=1))]
    Y = h.simulate_irtree(spec, truth, P, I, rng, node_types=['binary', 'gpcm'],
                          node_links=['logit', 'logit'])
    assert set(np.unique(Y)) <= {1, 2, 3, 4}


def test_analytic_kappa_marginal_matches_brute_force_joint():
    """Integrating b out by hand must equal the (kappa, b) joint Gaussian posterior."""
    rng = np.random.default_rng(3)
    P, I, K = 40, 6, 2
    theta, Q = rng.normal(size=P), rng.normal(size=(I, K))
    T = rng.random((P, I)) > 0.3
    Z = rng.normal(theta[:, None] + (Q @ [0.5, -0.4])[None, :] - rng.normal(0, 1, I)[None, :], 1)
    tb, tk = 0.7, 2.0
    rows, y = [], []                                # unknowns u = (kappa, b_1..b_I)
    for p, i in zip(*np.where(T), strict=True):
        r = np.zeros(K + I)
        r[:K] = Q[i]
        r[K + i] = -1.0
        rows.append(r)
        y.append(Z[p, i] - theta[p])
    D = np.array(rows)
    prior_prec = np.diag([1 / tk] * K + [1 / tb] * I)
    Vj = np.linalg.inv(prior_prec + D.T @ D)
    mj = Vj @ (D.T @ np.array(y))
    mu, V = h.analytic_kappa_marginal(Z, theta, T, Q, tb, tk)
    np.testing.assert_allclose(mu, mj[:K], rtol=1e-10)
    np.testing.assert_allclose(V, Vj[:K, :K], rtol=1e-10)
