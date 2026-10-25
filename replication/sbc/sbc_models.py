"""Simulation-based calibration (SBC) for irtree: models and one replication.

SBC (Talts, Betancourt, Simpson, Vehtari & Gelman, 2018) checks that a sampler
draws from the right posterior.  For each replication:

1. draw every parameter from the prior,
2. simulate a data set from those parameters,
3. fit the model to the simulated data with the *same* prior,
4. for each tracked parameter, record the rank of its true value among L
   thinned posterior draws (an integer in 0..L).

If the sampler targets the correct posterior, each rank is uniform on 0..L
across replications.  A sampler error shows as a non-uniform rank histogram:
a U shape (posterior too narrow), a hump (too wide), or a slope (biased).

The prior used to simulate (``draw_prior``) must be the prior used to fit
(``make_priors``).  The SBC priors below are proper and moderately
informative (e.g. variances ~ InvGamma(4, 3), mean 1), so simulated data sets
look like real tests; SBC is valid for any proper prior.

This file is imported by ``71_sbc.ipynb``.  The functions live in a module
(not in the notebook) so that the replications can run in parallel worker
processes on every platform, Windows included.
"""
import contextlib
import io
import time
import warnings

import numpy as np
from scipy.stats import norm, truncnorm

import irtree

# ---------------------------------------------------------------------------
# SBC prior hyperparameters (shared by draw_prior and make_priors)
# ---------------------------------------------------------------------------
IG_ALPHA, IG_BETA = 4.0, 3.0      # InvGamma for estimated variances (mean 1)
A_MEAN, A_VAR = 1.0, 0.5          # TruncNormal(1, 0.5) on (0, inf): package default
C_ALPHA, C_BETA = 5.0, 17.0       # Beta(5, 17): package default
COEF_VAR = 0.5                    # Normal(0, 0.5) for covariate coefficients
LKJ_ETA = 1.0                     # LKJ(1): package default (uniform rho for D = 2)

MODELS = {
    "S1": "binary tree: probit Rasch node with a person covariate + probit 2PL node",
    "S2": "polytomous tree: probit Rasch node + graded response node with an item covariate",
    "S3": "binary tree: logit 2PL node + probit 3PL node",
}


def _inv_gamma(rng, size=None):
    return IG_BETA / rng.gamma(IG_ALPHA, 1.0, size=size)


def _trunc_normal_a(rng, size):
    sd = np.sqrt(A_VAR)
    return truncnorm.rvs((0 - A_MEAN) / sd, np.inf, loc=A_MEAN, scale=sd,
                         size=size, random_state=rng)


def _lkj2(rng):
    """LKJ(eta) for D = 2: (rho + 1) / 2 ~ Beta(eta, eta)."""
    return 2.0 * rng.beta(LKJ_ETA, LKJ_ETA) - 1.0


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------
def spec_for(model):
    return irtree.disengagement_tree(binary=(model != "S2"))


def nodes_for(model):
    if model == "S1":
        return [irtree.ProbitRaschNode(person_covariates=True), irtree.ProbitTwoPLNode()]
    if model == "S2":
        return [irtree.ProbitRaschNode(), irtree.GRMNode(n_outcomes=3, item_covariates=True)]
    if model == "S3":
        return [irtree.LogitTwoPLNode(), irtree.ProbitThreePLNode()]
    raise KeyError(model)


def make_priors(model, spec, nodes):
    """The fitting prior, matching draw_prior exactly."""
    pr = irtree.build_default_priors(spec, nodes)
    ig = irtree.InvGammaPrior(alpha=IG_ALPHA, beta=IG_BETA)
    pr.set("b", irtree.NormalPrior(mean=0.0, variance=ig))
    # theta variance: estimated (InvGamma) on slope-free dimensions only; on a
    # dimension whose nodes all have slopes it stays at the automatic sigma2 = 1.
    for r, nm in enumerate(nodes, start=1):
        if not nm.has_discrimination:
            pr.set("theta", irtree.NormalPrior(mean=0.0, variance=ig), node=r)
    pr.set("beta", irtree.NormalPrior(mean=0.0, variance=COEF_VAR))
    pr.set("kappa", irtree.NormalPrior(mean=0.0, variance=COEF_VAR))
    pr.set("theta.correlation", irtree.LKJPrior(eta=LKJ_ETA))
    return pr


def draw_prior(model, P, I, rng):
    """Draw all parameters from the prior.  Returns (params, truth), where
    truth maps chain column names to true values for the tracked parameters."""
    rho = _lkj2(rng)
    s2 = np.array([_inv_gamma(rng), 1.0])           # node 1 dim, node 2 dim
    if model == "S2":
        s2[1] = _inv_gamma(rng)                      # GRM: slope-free, estimated
    if model == "S3":
        s2[0] = 1.0                                  # logit 2PL: sigma2 = 1
    sd = np.sqrt(s2)
    cov = np.array([[s2[0], rho * sd[0] * sd[1]], [rho * sd[0] * sd[1], s2[1]]])
    theta = rng.multivariate_normal(np.zeros(2), cov, size=P)
    tau2 = _inv_gamma(rng, size=2)                   # b-group variances
    prm = dict(theta=theta, rho=rho, s2=s2, tau2=tau2)

    prm["b1"] = rng.normal(0, np.sqrt(tau2[0]), I)
    if model == "S2":
        prm["thr"] = np.sort(rng.normal(0, np.sqrt(tau2[1]), (I, 2)), axis=1)
        prm["kappa"] = rng.normal(0, np.sqrt(COEF_VAR))
        q = rng.normal(size=I)
        prm["Q"] = (q - q.mean())[:, None]
    else:
        prm["b2"] = rng.normal(0, np.sqrt(tau2[1]), I)
    if model == "S1":
        prm["a2"] = _trunc_normal_a(rng, I)
        prm["beta"] = rng.normal(0, np.sqrt(COEF_VAR))
        x = rng.normal(size=P)
        prm["X"] = (x - x.mean())[:, None]
    if model == "S3":
        prm["a1"] = _trunc_normal_a(rng, I)
        prm["a2"] = _trunc_normal_a(rng, I)
        prm["c2"] = rng.beta(C_ALPHA, C_BETA, I)

    truth = {"rho_1_2": rho, "theta_1_1": theta[0, 0], "theta_2_1": theta[0, 1],
             "b.variance_1": tau2[0], "b.variance_2": tau2[1], "b_1_1": prm["b1"][0]}
    if model != "S3":
        truth["sigma2_1"] = s2[0]
    if model == "S1":
        truth.update({"b_2_1": prm["b2"][0], "a_2_1": prm["a2"][0], "beta_1_x1": prm["beta"]})
    if model == "S2":
        truth.update({"sigma2_2": s2[1], "b_2_1_thresh_1": prm["thr"][0, 0],
                      "b_2_1_thresh_2": prm["thr"][0, 1], "kappa_2_q1": prm["kappa"]})
    if model == "S3":
        truth.update({"b_2_1": prm["b2"][0], "a_1_1": prm["a1"][0],
                      "a_2_1": prm["a2"][0], "c_2_1": prm["c2"][0]})
    return prm, truth


def simulate(model, prm, rng):
    """Simulate the observed categories (0-based) from the drawn parameters.

    Disengagement tree: node 1 = 1 -> category 0 (rapid guess); node 1 = 0 ->
    node 2 decides among the remaining categories."""
    th = prm["theta"]
    P, I = th.shape[0], len(prm["b1"])
    eta1 = th[:, [0]] - prm["b1"][None, :]
    if model == "S1":
        eta1 = eta1 + prm["X"] @ np.array([[prm["beta"]]])
    if model == "S3":
        p1 = 1.0 / (1.0 + np.exp(-prm["a1"][None, :] * eta1))   # logit 2PL
    else:
        p1 = norm.cdf(eta1)
    y1 = rng.random((P, I)) < p1

    if model == "S2":                                # graded response, probit
        z = th[:, [1]] + (prm["Q"] @ np.array([[prm["kappa"]]])).T + rng.normal(size=(P, I))
        y2 = (z > prm["thr"][None, :, 0]).astype(int) + (z > prm["thr"][None, :, 1])
    else:
        eta2 = prm["a2"][None, :] * (th[:, [1]] - prm["b2"][None, :])
        p2 = norm.cdf(eta2)
        if model == "S3":
            p2 = prm["c2"][None, :] + (1 - prm["c2"][None, :]) * p2
        y2 = (rng.random((P, I)) < p2).astype(int)
    return np.where(y1, 0, 1 + y2)


# ---------------------------------------------------------------------------
# One replication
# ---------------------------------------------------------------------------
def one_replication(model, rep, P, I, n_iter, L, seed0):
    """Run replication ``rep`` of ``model``.  Returns a list of dict rows, one
    per tracked parameter: rank (0..L), true value, posterior mean, bulk ESS
    of the retained chain, and wall time."""
    rng = np.random.default_rng([seed0, int(model[1]), rep])
    prm, truth = draw_prior(model, P, I, rng)
    Y = simulate(model, prm, rng)
    spec, nodes = spec_for(model), nodes_for(model)
    kw = {}
    if "X" in prm:
        kw.update(X=prm["X"], X_names=["x1"])
    if "Q" in prm:
        kw.update(Q=prm["Q"], Q_names=["q1"])
    data = irtree.expand_data(Y, spec, categories_start=0, **kw)
    priors = make_priors(model, spec, nodes)

    t0 = time.time()
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        res = irtree.run_chain(data, nodes, priors, n_iter=n_iter, spec=spec,
                               seed=int(rng.integers(2**31)), progress=False)
    seconds = time.time() - t0

    chain = res.chain.astype(float)
    S = chain.shape[0]
    keep = np.linspace(0, S - 1, L).round().astype(int)    # L thinned draws
    rows = []
    for name, true in truth.items():
        if name not in res.param_names:
            raise KeyError(f"{model}: tracked parameter {name!r} is not a chain column")
        x = chain[:, res.param_names.index(name)]
        rows.append(dict(model=model, rep=rep, param=name, true=float(true),
                         rank=int(np.sum(x[keep] < true)), post_mean=float(x.mean()),
                         ess=float(irtree.ess(x)[0]), n_draws=S, seconds=seconds))
    return rows
