"""
irtree
======
Bayesian IRTree estimation via hybrid Gibbs/MH MCMC.

An IRTree models an observed response category as the endpoint of a sequence of
latent decisions (nodes), each governed by its own trait. This package fits such
models with an Albert-Chib data-augmented Gibbs sampler (probit nodes, with the
Beguin-Glas augmentation for 3PL guessing) and Metropolis-Hastings steps where
no closed-form conditional exists: every parameter of logit/GPCM nodes, probit
discriminations under a LogNormal prior, the Cowles step for GRM thresholds,
non-conjugate variances and LKJ correlations.

Quick start
-----------
    import irtree

    spec  = irtree.disengagement_tree(binary=True)
    data  = irtree.expand_data(Y, spec, categories_start=1)
    nodes = [irtree.ProbitRaschNode(), irtree.ProbitRaschNode()]
    fit   = irtree.irtree_fit(data, spec, nodes, n_iter=2000, n_chains=4, seed=42)

    fit.summary()          # posterior summaries
    fit.diagnostics()      # R-hat, ESS, Geweke, Heidelberger-Welch

The public API is re-exported here for convenience; every name is also available
from its defining submodule (e.g. ``irtree.node_models.ProbitRaschNode``).

Imports are direct (not guarded): a broken submodule raises at import time with
a full traceback, rather than silently dropping a name from the namespace.  The
one exception is ``load_data`` (id-keyed ingestion of data frames), which needs
pandas: it is loaded on first access, so ``import irtree`` needs only numpy and
scipy.
"""

from __future__ import annotations

__version__ = "0.4.0"

# ---------------------------------------------------------------------------
# Tree specification
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Constraints (sharing / correlation across nodes)
# ---------------------------------------------------------------------------
from irtree.constraints import (
    Constraints,
    share_nodes,
)

# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
from irtree.data_prep import (
    CenteredCovariates,
    PreparedData,
    center_covariates,
    expand_data,
    scale_covariates,
    summarize_prepared,
)

# ---------------------------------------------------------------------------
# Convergence diagnostics
# ---------------------------------------------------------------------------
from irtree.diagnostics import (
    STATUS_LEGEND,
    GewekeSweepResult,
    acf,
    console_summary,
    convergence_report,
    ess,
    geweke,
    geweke_sweep,
    geweke_zcrit,
    heidelberger_welch,
    mcse,
    rhat,
)

# ---------------------------------------------------------------------------
# High-level fit wrapper (parallel chains, warm start, unified result object)
# ---------------------------------------------------------------------------
from irtree.fit import (
    IRTreeFit,
    irtree_fit,
)

# ---------------------------------------------------------------------------
# Sampler result type (returned by the low-level run_chain; also what
# posterior/diagnostics consume)
# ---------------------------------------------------------------------------
from irtree.mcmc_sampler import (
    MCMCResult,
    run_chain,
)

# ---------------------------------------------------------------------------
# Node models
# ---------------------------------------------------------------------------
from irtree.node_models import (
    NODE_TYPES,
    CustomNode,
    GPCMNode,
    GRMNode,
    LogitRaschNode,
    LogitThreePLNode,
    LogitTwoPLNode,
    NodeBParams,
    NodeModel,
    PCMNode,
    ProbitRaschNode,
    ProbitThreePLNode,
    ProbitTwoPLNode,
    make_node,
)

# ---------------------------------------------------------------------------
# Posterior summaries and information criteria
# ---------------------------------------------------------------------------
from irtree.posterior import (
    LOOResult,
    PosteriorSummary,
    WAICResult,
    compare_loo,
    compare_waic,
    credible_interval,
    loo_cv,
    posterior_summary,
    waic,
)

# ---------------------------------------------------------------------------
# Priors
# ---------------------------------------------------------------------------
from irtree.priors import (
    BetaPrior,
    FixedCorrelation,
    HalfNormalPrior,
    InverseWishartPrior,
    InvGammaPrior,
    LKJPrior,
    LogNormalPrior,
    NormalPrior,
    Priors,
    TruncNormalPrior,
    build_default_priors,
)

# ---------------------------------------------------------------------------
# Progress monitoring (dependency-free)
# ---------------------------------------------------------------------------
from irtree.progress import (
    ProgressReporter,
    mean_acceptance,
)
from irtree.tree_spec import (
    IRTreeSpec,
    disengagement_tree,
    distractor_tree,
    ers_tree,
)

__all__ = [
    "__version__",
    # tree_spec
    "IRTreeSpec", "disengagement_tree", "distractor_tree", "ers_tree",
    # data_prep
    "expand_data", "PreparedData", "summarize_prepared",
    "center_covariates", "scale_covariates", "CenteredCovariates",
    # node_models
    "NodeModel", "ProbitRaschNode", "ProbitTwoPLNode",
    "ProbitThreePLNode", "LogitRaschNode", "LogitTwoPLNode", "LogitThreePLNode",
    "GRMNode", "GPCMNode", "PCMNode", "NodeBParams", "CustomNode",
    "NODE_TYPES", "make_node",
    # ingest (loaded on first access; needs pandas)
    "load_data",
    # priors
    "Priors", "NormalPrior", "LogNormalPrior", "TruncNormalPrior",
    "BetaPrior", "InvGammaPrior", "HalfNormalPrior",
    "FixedCorrelation", "LKJPrior", "InverseWishartPrior",
    "build_default_priors",
    # constraints
    "Constraints", "share_nodes",
    # sampler
    "MCMCResult", "run_chain",
    # diagnostics
    "rhat", "ess", "mcse", "acf", "geweke", "geweke_sweep", "geweke_zcrit",
    "GewekeSweepResult", "STATUS_LEGEND", "heidelberger_welch", "convergence_report", "console_summary",
    # posterior
    "posterior_summary", "credible_interval", "waic", "loo_cv",
    "compare_waic", "compare_loo",
    "PosteriorSummary", "WAICResult", "LOOResult",
    # fit
    "irtree_fit", "IRTreeFit",
    # progress
    "ProgressReporter", "mean_acceptance",
]


def __getattr__(name):
    """Load ``load_data`` on first access (PEP 562), so pandas is imported
    only by users of the data-frame ingestion path."""
    if name == "load_data":
        from irtree.ingest import load_data
        globals()["load_data"] = load_data
        return load_data
    raise AttributeError(f"module 'irtree' has no attribute {name!r}")
