"""
irtree.fit
==========
Layer 12 — User-facing entry point: ``irtree_fit()`` and ``IRTreeFit``.

USAGE
-----
::

    from irtree import irtree_fit, disengagement_tree, ProbitRaschNode

    spec = disengagement_tree()
    fit  = irtree_fit(
        data         = Y,
        spec         = spec,
        node_models  = [ProbitRaschNode(), ProbitRaschNode()],
        n_iter       = 2000,
        n_chains     = 4,
        seed         = 42,
    )

    # Console summary prints automatically.
    fit.summary()                  # posterior table
    fit.diagnostics()              # DiagnosticReport
    fit.report('output.html')      # self-contained HTML
    fig = fit.plot_trace()         # matplotlib Figure (no plt.show())

PARALLEL CHAINS
---------------
Multiple chains run in parallel via ``concurrent.futures.ProcessPoolExecutor``.
Each chain receives a unique seed derived from the master seed via
``np.random.SeedSequence.spawn()``, ensuring reproducibility without
correlation between streams.

Because ``ProcessPoolExecutor`` pickles its arguments and the worker
function, all objects passed to the worker must be picklable.
``_chain_worker`` is a module-level function (not a closure or lambda)
so it pickles correctly.

PLOT POLICY
-----------
No method ever calls ``plt.show()``.  All plot methods return
``matplotlib.figure.Figure`` objects.  The caller decides whether to
display, save, or embed them.  This ensures identical behavior in
scripts, Jupyter notebooks, and CI pipelines.

HTML REPORT
-----------
``fit.report(path)`` writes a self-contained single-file HTML document
with six sections (model summary, convergence, diagnostic table, flagged
parameters, posterior summary, trace and autocorrelation), all plots
embedded as base64 PNG, and no external dependencies.  The document can be emailed, posted, or archived.
"""

from __future__ import annotations

import base64
import io
import math
import os
import sys
import textwrap
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# Module-level worker — must be picklable (top-level function, not a closure)
# ---------------------------------------------------------------------------

def _chain_worker(args: dict):
    """Top-level worker function for ProcessPoolExecutor.

    Accepts a single dict of kwargs so the function signature stays stable
    across Python versions and pickle works without issue.
    """
    from irtree.mcmc_sampler import run_chain
    return run_chain(**args)


# ---------------------------------------------------------------------------
# IRTreeFit
# ---------------------------------------------------------------------------

@dataclass
class IRTreeFit:
    """The fitted IRTree model.  Returned by :func:`irtree_fit`.

    Attributes
    ----------
    results : list[MCMCResult]
        One MCMCResult per chain.
    data : PreparedData
    spec : IRTreeSpec
    node_models : list[NodeModel]
    priors : Priors
    n_chains : int
    n_iter : int
    n_warmup : int
        Iterations discarded per chain.  In evidence mode this is the
        effective warm-up (the latest point at which every block of every
        chain had frozen, or the pure-Gibbs burn-in), not the argument.
    constraints : Constraints or None
        Sharing and correlation constraints used for the fit (needed again
        by waic() / loo()).
    """
    results:     list
    data:        object   # PreparedData
    spec:        object   # IRTreeSpec
    node_models: list
    priors:      object   # Priors
    n_chains:    int
    n_iter:      int
    n_warmup:    int
    constraints: object   = None   # Constraints or None

    # Private cache
    _diagnostic_report: object = field(default=None, repr=False)
    _posterior_summ:    object = field(default=None, repr=False)
    _waic_result:       object = field(default=None, repr=False)
    _loo_result:        object = field(default=None, repr=False)

    # ----------------------------------------------------------------
    # Diagnostics
    # ----------------------------------------------------------------

    def diagnostics(self):
        """Return the DiagnosticReport (cached after first call).

        For n_chains > 1, R-hat is the primary convergence metric.
        For n_chains = 1, Geweke and Heidelberger-Welch are primary.

        Returns
        -------
        DiagnosticReport
        """
        if self._diagnostic_report is None:
            from irtree.diagnostics import convergence_report
            self._diagnostic_report = convergence_report(self.results)
        return self._diagnostic_report

    # ----------------------------------------------------------------
    # Posterior summaries
    # ----------------------------------------------------------------

    def _get_summary(self):
        if self._posterior_summ is None:
            from irtree.posterior import posterior_summary
            self._posterior_summ = posterior_summary(self.results)
        return self._posterior_summ

    def summary(self, max_params: int = 50) -> None:
        """Print a posterior summary table (mean, SD, 95% HDI, ESS).

        Parameters
        ----------
        max_params : int
            Maximum number of parameters to show.  Set to None to show all.
        """
        summ = self._get_summary()
        names = list(summ.keys())
        if max_params is not None and len(names) > max_params:
            shown = names[:max_params]
            print(f"(Showing first {max_params} of {len(names)} parameters. "
                  f"Call fit.summary(max_params=None) for all.)")
        else:
            shown = names

        S = self.results[0].chain.shape[0]
        print(f"\nPosterior Summary  "
              f"({self.n_chains} chain(s) × {self.n_iter} iters, "
              f"{S * self.n_chains} draws)")
        print(f"{'Parameter':<28} {'Mean':>8} {'SD':>8} "
              f"{'2.5%':>8} {'50%':>8} {'97.5%':>8} "
              f"{'HDI lo':>8} {'HDI hi':>8} {'ESS':>7}")
        print("─" * 100)
        for name in shown:
            s = summ[name]
            print(f"{name:<28} {s.mean:>8.3f} {s.sd:>8.3f} "
                  f"{s.q025:>8.3f} {s.q50:>8.3f} {s.q975:>8.3f} "
                  f"{s.hdi_lo:>8.3f} {s.hdi_hi:>8.3f} {s.n_eff:>7.0f}")

    def summary_df(self):
        """Return the posterior summary as a pandas DataFrame."""
        from irtree.posterior import summary_to_dataframe
        return summary_to_dataframe(self._get_summary())

    def to_logit(self):
        """Posterior summary on the logistic-equivalent scale (D = 1.702).

        A probit node's probability Phi(eta) is within 0.0095 of
        L(1.702 eta), so a probit fit can be read as a logit fit after
        rescaling.  The factor goes on whichever quantity carries the latent
        scale (see :func:`irtree.posterior.logit_scale_factors`):

        * dimensions whose nodes are all probit without slopes (Rasch, GRM
          without a): theta, b, thresholds and covariate effects × 1.702,
          variances × 1.702², slopes stay 1;
        * otherwise the theta metric is kept and probit slopes × 1.702.

        Correlations and guessing parameters are unchanged; logit nodes are
        already on the logit scale.  Because the rescaling is linear with a
        positive factor, means, SDs, quantiles and HDIs convert exactly and
        n_eff is unchanged.  The values are *logistic-equivalent*, not logit
        estimates: a logit refit differs by about 1-3% (the links differ in
        the tails).

        Returns
        -------
        pandas.DataFrame
            The ``summary_df()`` columns, rescaled, plus ``factor`` (the
            multiplier applied; NaN where no single factor exists).
            ``df.attrs["notes"]`` lists caveats (implied common slopes,
            custom nodes, unconverted parameters).
        """
        import warnings

        from irtree.posterior import LOGIT_D, logit_scale_factors, summary_to_dataframe
        df = summary_to_dataframe(self._get_summary())
        factors, notes = logit_scale_factors(
            list(df.index), self.data, self.spec, self.node_models, self.priors,
            self.constraints)
        for col in ("mean", "sd", "q2.5", "q50", "q97.5", "hdi_lo", "hdi_hi"):
            df[col] = df[col].to_numpy() * factors
        df["factor"] = factors
        df.attrs["scale"] = f"logistic-equivalent (D = {LOGIT_D} rescaling)"
        df.attrs["notes"] = notes
        if np.isnan(factors).any():
            warnings.warn(notes[-1], stacklevel=2)
        return df

    def coef(self) -> dict[str, float]:
        """Return posterior means keyed by parameter name."""
        return {name: s.mean for name, s in self._get_summary().items()}

    def credible_interval(self, level: float = 0.95, method: str = "hdi"):
        """Return credible intervals for all parameters."""
        from irtree.posterior import credible_interval
        return credible_interval(self.results, level=level, method=method)

    # ----------------------------------------------------------------
    # Model comparison
    # ----------------------------------------------------------------

    def waic(self):
        """Return WAICResult (cached)."""
        if self._waic_result is None:
            from irtree.posterior import waic as _waic
            self._waic_result = _waic(
                self.results, self.data, self.node_models,
                spec=self.spec,
                constraints=self.constraints,
            )
        return self._waic_result

    def loo(self):
        """Return LOOResult (cached)."""
        if self._loo_result is None:
            from irtree.posterior import loo_cv
            self._loo_result = loo_cv(
                self.results, self.data, self.node_models,
                spec=self.spec,
                constraints=self.constraints,
            )
        return self._loo_result

    # ----------------------------------------------------------------
    # Plotting (no plt.show() anywhere)
    # ----------------------------------------------------------------

    def plot_trace(
        self,
        params: Sequence[str] | None = None,
        max_params: int = 12,
    ):
        """Trace plots for the requested parameters.

        Parameters
        ----------
        params    : list[str] or None — parameter names to plot.
                    If None, selects up to ``max_params`` b and theta parameters.
        max_params : int — cap when params is None.

        Returns
        -------
        matplotlib.figure.Figure  (no plt.show() is called)
        """
        import matplotlib.pyplot as plt

        names = self._select_params(params, max_params)
        n     = len(names)
        if n == 0:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No parameters to plot",
                    ha="center", va="center", transform=ax.transAxes)
            return fig

        fig, axes = plt.subplots(n, self.n_chains,
                                 figsize=(4 * self.n_chains, 1.8 * n),
                                 squeeze=False)

        for row, name in enumerate(names):
            for col, result in enumerate(self.results):
                ax = axes[row, col]
                if name in result.param_names:
                    idx = result.param_names.index(name)
                    ax.plot(result.chain[:, idx].astype(float),
                            lw=0.6, alpha=0.8)
                    if col == 0:
                        ax.set_ylabel(name, fontsize=7, rotation=0,
                                      ha="right", labelpad=50)
                    if row == 0:
                        ax.set_title(f"Chain {col + 1}", fontsize=8)
                ax.tick_params(labelsize=6)

        fig.suptitle("Trace plots", fontsize=10, y=1.01)
        fig.tight_layout()
        return fig

    def plot_acf(
        self,
        params: Sequence[str] | None = None,
        lags: Sequence[int] = (1, 5, 10, 20, 50),
        max_params: int = 12,
    ):
        """Autocorrelation plots for the requested parameters.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        from irtree.diagnostics import acf as _acf

        names = self._select_params(params, max_params)
        n     = len(names)
        if n == 0:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No parameters to plot",
                    ha="center", va="center", transform=ax.transAxes)
            return fig

        lags = list(lags)
        fig, axes = plt.subplots(n, self.n_chains,
                                 figsize=(4 * self.n_chains, 1.8 * n),
                                 squeeze=False)

        for row, name in enumerate(names):
            for col, result in enumerate(self.results):
                ax = axes[row, col]
                if name in result.param_names:
                    idx    = result.param_names.index(name)
                    chain  = result.chain[:, idx].astype(float)
                    acf_v  = _acf(chain, lags=lags)
                    vals   = [acf_v[lag] for lag in lags]
                    ax.bar(range(len(lags)), vals,
                           tick_label=[str(l) for l in lags],
                           width=0.6, color="steelblue", alpha=0.8)
                    ax.axhline(0, color="black", lw=0.5)
                    ax.axhline(0.1, color="orange", lw=0.8, ls="--",
                               alpha=0.6)
                    ax.axhline(-0.1, color="orange", lw=0.8, ls="--",
                               alpha=0.6)
                    ax.set_ylim(-0.3, 1.1)
                    if col == 0:
                        ax.set_ylabel(name, fontsize=7, rotation=0,
                                      ha="right", labelpad=50)
                    if row == 0:
                        ax.set_title(f"Chain {col + 1}", fontsize=8)
                ax.tick_params(labelsize=6)

        fig.suptitle("Autocorrelation", fontsize=10, y=1.01)
        fig.tight_layout()
        return fig

    def plot_posterior(
        self,
        params: Sequence[str] | None = None,
        max_params: int = 16,
    ):
        """Posterior density plots with 95% HDI shaded.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        from irtree.posterior import _hdi_1d, _stack_chains

        names  = self._select_params(params, max_params)
        n      = len(names)
        if n == 0:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No parameters to plot",
                    ha="center", va="center", transform=ax.transAxes)
            return fig

        ncols = min(4, n)
        nrows = math.ceil(n / ncols)
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(3.5 * ncols, 2.5 * nrows),
                                 squeeze=False)

        chain_all = _stack_chains(self.results)
        name_to_col = {nm: i for i, nm in
                       enumerate(self.results[0].param_names)}

        for k, name in enumerate(names):
            row, col = divmod(k, ncols)
            ax = axes[row][col]
            if name not in name_to_col:
                ax.set_visible(False)
                continue

            samples = chain_all[:, name_to_col[name]]
            lo, hi  = _hdi_1d(samples, 0.95)

            ax.hist(samples, bins=40, density=True,
                    color="steelblue", alpha=0.6, edgecolor="none")

            # Shade 95% HDI
            hist_vals, bin_edges = np.histogram(samples, bins=40, density=True)
            max_h = hist_vals.max()
            ax.fill_betweenx([0, max_h * 1.05], lo, hi,
                              alpha=0.25, color="orange",
                              label="95% HDI")
            ax.axvline(lo, color="orange", lw=1.0)
            ax.axvline(hi, color="orange", lw=1.0)
            ax.axvline(float(np.mean(samples)), color="red",
                       lw=1.2, ls="--", label="Mean")
            ax.set_title(name, fontsize=8)
            ax.tick_params(labelsize=6)
            if k == 0:
                ax.legend(fontsize=6)

        # Hide unused axes
        for k in range(n, nrows * ncols):
            row, col = divmod(k, ncols)
            axes[row][col].set_visible(False)

        fig.suptitle("Posterior distributions (95% HDI shaded)",
                     fontsize=10, y=1.01)
        fig.tight_layout()
        return fig

    # ----------------------------------------------------------------
    # HTML report
    # ----------------------------------------------------------------

    def report(self, path: str = "irtree_report.html", plot_params="default") -> None:
        """Write a self-contained HTML diagnostic report to ``path``.

        Section order:
          1. Model summary
          2. Convergence          (status counts; number of flagged parameters)
          3. Diagnostic table      (with the PASS/WARN/FAIL colour legend above it)
          4. Flagged parameters    (per-parameter: all five indicators with
                                    interpretation, plus that parameter's trace + ACF)
          5. Posterior summary
          6. Trace & autocorrelation (per parameter, general run)

        Parameters
        ----------
        path : str, default 'irtree_report.html'
        plot_params : {'default', 'all', int}, default 'default'
            Which parameters get a trace + ACF pair in section 6. 'default' shows
            the first 50; 'all' shows every parameter (large models -> large file);
            an int shows the first N. Flagged parameters always get their own
            plots in section 4 regardless of this setting.
        """
        # resolve the plot cap
        _n_all = len(self.results[0].param_names)
        if plot_params == "all":
            _plot_cap = _n_all
            if _n_all > 100:
                import warnings as _w
                _w.warn(
                    f"plot_params='all' will render {_n_all} trace+ACF figure "
                    f"pairs: this is slow and produces a large HTML file. "
                    f"Consider plot_params=50 or a specific count.",
                    stacklevel=2,
                )
        elif plot_params == "default":
            _plot_cap = min(50, _n_all)
        else:
            _plot_cap = int(plot_params)
        rep   = self.diagnostics()
        summ  = self._get_summary()

        from irtree.diagnostics import STATUS_LEGEND_LINES

        sections = []

        # --- 1. Model summary ---
        sections.append(_html_section("1. Model Summary", _model_summary_html(self)))

        # --- 2. Convergence (counts + flagged count; no collapsed "verdict") ---
        sections.append(_html_section(
            "2. Convergence",
            _convergence_html(rep),
        ))

        # --- 3. Diagnostic table, with the colour legend right before it ---
        _legend = ("<p><b>Status key</b></p><ul>"
                   + "".join(f"<li>{ln}</li>" for ln in STATUS_LEGEND_LINES)
                   + "</ul>")
        sections.append(_html_section(
            "3. Diagnostic Table",
            _legend + _diagnostic_table_html(rep),
        ))

        # --- 4. Flagged parameters (enriched: all indicators + own plots) ---
        sections.append(_html_section(
            "4. Flagged Parameters (WARN / FAIL)",
            _flagged_section_html(self, rep),
        ))

        # --- 5. Posterior summary ---
        sections.append(_html_section(
            "5. Posterior Summary",
            _posterior_table_html(summ),
        ))

        # --- 6. Trace & autocorrelation, per parameter ---
        _note = (f"<p>Showing the first {_plot_cap} of {_n_all} parameters. "
                 f"Pass <code>plot_params='all'</code> or <code>plot_params=N</code> "
                 f"to report()  to change this. (Flagged parameters also have their "
                 f"trace &amp; ACF in section 4.)</p>")
        sections.append(_html_section(
            "6. Trace &amp; Autocorrelation",
            _note + _trace_acf_grid_html(self, _plot_cap),
        ))

        html = _wrap_html(
            title="IRTree Fit Report",
            body="\n".join(sections),
        )

        path = os.path.abspath(path)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(html)

        print(f"Report written to: {path}")

    # ----------------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------------

    def _select_params(
        self,
        params: Sequence[str] | None,
        max_params: int,
    ) -> list[str]:
        """Return the parameter names to plot."""
        all_names = self.results[0].param_names
        if params is not None:
            return [p for p in params if p in all_names]

        # Default: pick b and theta parameters, up to max_params
        preferred = (
            [n for n in all_names if n.startswith("b_")]
            + [n for n in all_names if n.startswith("theta_")]
            + [n for n in all_names if n.startswith("sigma2_")]
        )
        return preferred[:max_params]

    def __repr__(self) -> str:
        S = self.results[0].chain.shape[0]
        return (f"IRTreeFit("
                f"n_chains={self.n_chains}, "
                f"n_iter={self.n_iter}, "
                f"n_samples_per_chain={S}, "
                f"n_params={len(self.results[0].param_names)})")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def irtree_fit(
    data,
    spec,
    node_models:     list,
    priors           = None,
    n_iter:          int   = 2000,
    n_warmup:        int   = 0,
    warmup_mode:     str   = "evidence",
    min_keep:        int | None = None,
    n_chains:        int   = 4,
    seed:            int | None = 42,
    warm_start:      bool  = True,
    chain_dtype:     str   = "float32",
    thin_by:         int   = 1,
    categories_start: int  = 0,
    max_workers:     int | None = None,
    constraints      = None,
    X                = None,
    Q                = None,
    W                = None,
    X_missing_codes  = (99,),
    Q_missing_codes  = (99,),
    W_missing_codes  = (99,),
    progress:        bool = True,
    print_every:     int | None = None,
) -> IRTreeFit:
    """Fit an IRTree model via MCMC and return an :class:`IRTreeFit`.

    Parameters
    ----------
    data : ndarray [P × I] or PreparedData
        Raw response matrix (integer categories) **or** an already-expanded
        ``PreparedData`` from :func:`~irtree.data_prep.expand_data`.
        If a raw matrix is passed, ``spec`` and ``categories_start`` are
        used to expand it automatically.
    spec : IRTreeSpec
        Tree specification from :func:`~irtree.tree_spec.IRTreeSpec` or a
        convenience constructor such as :func:`~irtree.tree_spec.disengagement_tree`.
    node_models : list[NodeModel]
        One model per IRTree node, e.g. ``[ProbitRaschNode(), ProbitRaschNode()]``.
    priors : Priors or None
        Prior specification.  If ``None``, defaults are built via
        :func:`~irtree.priors.build_default_priors` (which fixes the theta
        variance at 1 on dimensions with free discriminations).  Trait covariance is
        specified through the ``Priors`` object via
        ``priors.set("theta.correlation", FixedCorrelation(rho=0.3))``
        or ``priors.set("theta.correlation", LKJPrior(eta=1.0))``.
    n_iter : int
        Total iterations per chain (warmup + sampling).
    n_warmup : int
        Iterations discarded before draws are kept.  In warmup_mode="evidence"
        it is the CAP of the MH warm-up (0 => auto) for models with MH blocks,
        and the burn-in length for pure-Gibbs models (binary probit Rasch,
        2PL and 3PL nodes with conjugate priors and no estimated correlation;
        GRM nodes add a Cowles MH step): the value clamped to
        [2·M·W, n_iter - min_keep·thin_by], or 2·M·W = 600 when 0.
    warmup_mode : str
        "evidence" (default): each MH block freezes when its acceptance rate
        has stabilised (or at the cap); draws are kept only once every block
        has frozen.  Removes the need to guess n_warmup, and tunes step sizes
        even when n_warmup is left at 0.  A block freezes after M = 3
        consecutive stable windows of W = 100 iterations, so no earlier than
        iteration 300.  Every model needs n_iter >= ~1600
        (2·M·W + min_keep·thin_by): room for a warm-up of up to 600
        iterations (the cap's floor; also the default pure-Gibbs burn-in)
        plus min_keep kept draws.
        "fixed": adapt for exactly n_warmup iterations, then freeze — use it
        with a small n_warmup for short diagnostic runs.
        Per-block freeze times are in each chain's ``result.frozen_at``.
    min_keep : int or None
        Evidence mode only.  Guaranteed number of kept (post-thinning) draws per
        chain; the warmup cap is bounded so a worst-case freeze still leaves this
        many samples.  None uses the package default (1000).  Raise it when
        the slowest-mixing parameters need more draws for their ESS.
    n_chains : int
        Number of parallel chains.
    seed : int or None
        Master seed.  Each chain receives a unique child seed derived via
        ``np.random.SeedSequence.spawn()``.
    warm_start : bool
        If True (default), initialise item difficulties and thresholds from
        the empirical response proportions (probit quantiles), instead of
        zeros / evenly spaced values, to shorten the initial transient.
    chain_dtype : str
        ``'float32'`` (default, 2× memory savings) or ``'float64'``.
    thin_by : int
        Store every ``thin_by``-th post-warmup sample.
    categories_start : int
        ``0`` if response categories are 0-based (default, as in
        ``expand_data`` and ``load_data``), ``1`` if 1-based.  Only used when
        ``data`` is a raw ndarray.
    max_workers : int or None
        Maximum worker processes for ``ProcessPoolExecutor``.
        ``None`` defaults to ``min(n_chains, os.cpu_count())``.
    constraints : Constraints or None
        Parameter sharing and correlation: ``b_shared``, ``a_shared``,
        ``c_shared``, ``gamma_shared``, ``b_corr``, ``a_corr`` (1-based node
        indices) and ``theta_corr`` blocks (1-based latent-dimension
        indices); see :class:`~irtree.constraints.Constraints`.  None:
        nothing shared.
    X : array-like [P, K_x] or None
        Person-level covariate matrix.  Required when any node model has
        ``has_person_covariates=True``.  Can also be passed pre-processed
        inside a ``PreparedData`` object.
    Q : array-like [I, K_q] or None
        Item-level covariate matrix.  Required when any node model has
        ``has_item_covariates=True``.
    W : array-like [P, I, K_w] or None
        Person-by-item covariate tensor.  Required when any node model
        has ``has_pi_covariates=True``.
    X_missing_codes, Q_missing_codes, W_missing_codes : tuple
        Sentinel values treated as missing in the respective covariate
        matrices.  Default ``(99,)``.  Pass empty tuples to detect NaN
        only.
    progress : bool
        If True (default), print a live progress line (single sequential chain
        only; parallel chains report on completion).
    print_every : int or None
        How often, in iterations, the progress line is printed.  ``None``
        (default) caps the run at ~10 updates by printing every
        ``ceil(n_iter / 10)`` iterations — e.g. ``n_iter=10000`` prints every
        1000.  Set an explicit interval for finer or coarser reporting.

    Returns
    -------
    IRTreeFit
        The fitted object.  Console diagnostics print automatically.
    """
    from irtree.data_prep import PreparedData, expand_data
    from irtree.priors import build_default_priors

    # --- Expand data if a raw array was passed ---
    if isinstance(data, PreparedData):
        prepared = data
    else:
        prepared = expand_data(
            np.asarray(data), spec,
            categories_start=categories_start,
            X=X, Q=Q, W=W,
            X_missing_codes=X_missing_codes,
            Q_missing_codes=Q_missing_codes,
            W_missing_codes=W_missing_codes,
        )

    # --- Build default priors if none supplied ---
    if priors is None:
        priors = build_default_priors(spec, node_models)

    # --- Spawn per-chain seeds ---
    ss    = np.random.SeedSequence(seed)
    seeds = [int(child.generate_state(1)[0])
             for child in ss.spawn(n_chains)]

    # --- Build worker argument dicts (must be picklable) ---
    worker_args = [
        dict(
            data        = prepared,
            node_models = node_models,
            priors      = priors,
            n_iter      = n_iter,
            n_warmup    = n_warmup,
            warmup_mode = warmup_mode,
            min_keep    = min_keep,
            seed        = seeds[c],
            warm_start  = warm_start,
            chain_dtype = chain_dtype,
            thin_by     = thin_by,
            spec        = spec,
            constraints = constraints,
            # live per-iteration line only when a single sequential chain;
            # parallel workers stay silent to avoid interleaved output.
            progress    = progress and (n_chains == 1),
            progress_label = '' if n_chains == 1 else f'chain {c+1}/{n_chains}',
            print_every = print_every,
        )
        for c in range(n_chains)
    ]

    # --- Run chains (parallel or sequential) ---
    if n_chains == 1:
        # Sequential: avoid subprocess overhead for a single chain
        results = [_chain_worker(worker_args[0])]
    else:
        n_workers = max_workers or min(n_chains, os.cpu_count() or n_chains)
        results   = [None] * n_chains
        try:
            _t_par = time.perf_counter()
            _n_done = 0
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                futures = {pool.submit(_chain_worker, args): idx
                           for idx, args in enumerate(worker_args)}
                for future in as_completed(futures):
                    idx          = futures[future]
                    results[idx] = future.result()
                    _n_done += 1
                    if progress:
                        _el = time.perf_counter() - _t_par
                        _m, _s = divmod(int(_el), 60)
                        sys.stderr.write(
                            f'chain {_n_done}/{n_chains} complete · elapsed {_m}:{_s:02d}\n')
                        sys.stderr.flush()
        except Exception:
            # Fallback to sequential if multiprocessing fails
            # (common in interactive sessions and on some platforms)
            results = [_chain_worker(args) for args in worker_args]

    # --- Assemble IRTreeFit ---
    # In evidence mode the warmup length is discovered at runtime, not set by
    # the user, so report the longest effective warmup across chains (the point
    # by which every block of every chain had frozen) rather than the raw arg.
    _reported_warmup = n_warmup
    if warmup_mode == "evidence" and results:
        _reported_warmup = max(
            int(getattr(r, "effective_warmup", n_warmup)) for r in results
        )
    fit = IRTreeFit(
        results     = results,
        data        = prepared,
        spec        = spec,
        node_models = node_models,
        priors      = priors,
        n_chains    = n_chains,
        n_iter      = n_iter,
        n_warmup    = _reported_warmup,
        constraints = constraints,
    )

    # --- Console summary fires automatically ---
    try:
        from irtree.diagnostics import console_summary, convergence_report
        rep = convergence_report(results)
        fit._diagnostic_report = rep   # cache it
        console_summary(rep)
    except Exception:
        pass   # never let the summary crash the fit

    return fit


# ---------------------------------------------------------------------------
# HTML report helpers
# ---------------------------------------------------------------------------

def _html_section(title: str, content: str) -> str:
    return textwrap.dedent(f"""\
        <section>
          <h2>{title}</h2>
          {content}
        </section>
        <hr>
    """)


def _wrap_html(title: str, body: str) -> str:
    return textwrap.dedent(f"""\
        <!DOCTYPE html>
        <html lang="en">
        <head>
          <meta charset="UTF-8">
          <meta name="viewport" content="width=device-width, initial-scale=1">
          <title>{title}</title>
          <style>
            body {{ font-family: system-ui, sans-serif; max-width: 1100px;
                    margin: 0 auto; padding: 1.5em; color: #1a1a1a; }}
            h1   {{ border-bottom: 2px solid #333; padding-bottom: .3em; }}
            h2   {{ color: #2c5282; margin-top: 1.5em; }}
            table {{ border-collapse: collapse; width: 100%;
                     font-size: .85em; margin: .8em 0; }}
            th, td {{ border: 1px solid #ccc; padding: 4px 8px; text-align: right; }}
            th   {{ background: #e8edf2; text-align: center; }}
            td:first-child {{ text-align: left; font-family: monospace; }}
            .pass {{ background: #c8e6c9; }}
            .warn {{ background: #fff9c4; }}
            .fail {{ background: #ffcdd2; }}
            .verdict-pass {{ color: #1b5e20; font-weight: bold; }}
            .verdict-warn {{ color: #e65100; font-weight: bold; }}
            .verdict-fail {{ color: #b71c1c; font-weight: bold; }}
            pre  {{ background: #f5f5f5; padding: .8em; border-radius: 4px;
                    overflow-x: auto; font-size: .8em; }}
            hr   {{ border: none; border-top: 1px solid #ddd; margin: 2em 0; }}
            img  {{ border: 1px solid #ddd; border-radius: 4px; display: block;
                    margin: .5em 0; }}
          </style>
        </head>
        <body>
          <h1>{title}</h1>
          {body}
        </body>
        </html>
    """)


def _model_summary_html(fit: IRTreeFit) -> str:
    S_per_chain = fit.results[0].chain.shape[0]
    total_draws = S_per_chain * fit.n_chains
    n_params    = len(fit.results[0].param_names)

    rows = [
        ("Tree spec",         type(fit.spec).__name__),
        ("Nodes (R)",         str(fit.spec.n_nodes)),
        ("Node models",       ", ".join(type(nm).__name__ for nm in fit.node_models)),
        ("Persons (P)",       str(fit.data.n_persons)),
        ("Items (I)",         str(fit.data.n_items)),
        ("Chains",            str(fit.n_chains)),
        ("Iterations",        str(fit.n_iter)),
        ("Warmup",            str(fit.n_warmup)),
        ("Stored per chain",  str(S_per_chain)),
        ("Total draws",       str(total_draws)),
        ("Parameters",        str(n_params)),
    ]

    html = "<table><tr><th>Setting</th><th>Value</th></tr>\n"
    for k, v in rows:
        html += f"<tr><td>{k}</td><td>{v}</td></tr>\n"
    html += "</table>"
    return html


def _convergence_html(rep) -> str:
    color_class = {"PASS": "verdict-pass",
                   "WARN": "verdict-warn",
                   "FAIL": "verdict-fail"}.get(rep.summary_status, "")

    n_tot = rep.n_pass + rep.n_warn + rep.n_fail
    # counts distribution, not a collapsed verdict; overall label is advisory
    lines = [
        f'<p class="{color_class}">Overall (advisory): {rep.summary_status}</p>',
        f"<p><b>{rep.n_pass} PASS</b> · <b>{rep.n_warn} WARN</b> · "
        f"<b>{rep.n_fail} FAIL</b> (of {n_tot} parameters)</p>",
        f"<p>{rep.n_chains} chain(s) × {rep.n_iter} iterations, "
        f"{rep.n_samples} stored samples per chain.</p>",
    ]

    if rep.n_needs_burnin:
        pct = int(round(100 * rep.recommended_burnin))
        lines.append(
            f"<p>{rep.n_needs_burnin} parameter(s) flagged <em>only</em> for "
            f"under-warmup: resolve by discarding the first ~{pct}% "
            f"(recommended burn-in).</p>")
    if rep.n_not_converged:
        lines.append(
            f"<p>{rep.n_not_converged} parameter(s) did not converge even after "
            f"discarding: inspect these.</p>")

    if rep.n_chains > 1:
        rhat_max = (max(rep.rhat_values.values()) if rep.rhat_values else float("nan"))
        lines.append(f"<p>Max R-hat: {rhat_max:.4f}</p>")
    ess_min = (min(rep.ess_bulk.values()) if rep.ess_bulk else float("nan"))
    lines.append(f"<p>Min ESS (bulk): {ess_min:.0f}</p>")

    # flagged: just the count (per-parameter detail is section 4)
    n_flagged = len(rep.flagged_params)
    if n_flagged:
        lines.append(f"<p>{n_flagged} parameter(s) flagged (WARN or FAIL), "
                     f"see section 4 for the per-parameter detail.</p>")
    else:
        lines.append("<p>No parameters flagged. All diagnostics within thresholds.</p>")
    lines.append("<p>The overall label is advisory; the per-parameter diagnostic "
                 "table (section 3) is the decision.</p>")
    return "\n".join(lines)


def _posterior_table_html(summ: dict) -> str:
    html = ("<table>"
            "<tr><th>Parameter</th><th>Mean</th><th>SD</th>"
            "<th>2.5%</th><th>50%</th><th>97.5%</th>"
            "<th>HDI lo</th><th>HDI hi</th><th>ESS</th></tr>\n")
    for name, s in summ.items():
        html += (f"<tr><td>{name}</td>"
                 f"<td>{s.mean:.3f}</td><td>{s.sd:.3f}</td>"
                 f"<td>{s.q025:.3f}</td><td>{s.q50:.3f}</td>"
                 f"<td>{s.q975:.3f}</td>"
                 f"<td>{s.hdi_lo:.3f}</td><td>{s.hdi_hi:.3f}</td>"
                 f"<td>{s.n_eff:.0f}</td></tr>\n")
    html += "</table>"
    return html


def _diagnostic_table_html(rep) -> str:
    html = ("<table>"
            "<tr><th>Parameter</th>"
            + ("<th>R-hat</th>" if rep.n_chains > 1 else "")
            + "<th>ESS bulk</th><th>ESS tail</th>"
            "<th>MCSE/SD</th><th>Geweke z</th>"
            "<th>HW pass</th><th>ACF lag50</th>"
            "<th>Status</th></tr>\n")

    for name in rep.param_names:
        status = rep.status.get(name, "PASS")
        css    = {"PASS": "pass", "WARN": "warn", "FAIL": "fail"}.get(status, "")
        rhat_s = ""
        if rep.n_chains > 1:
            rv = rep.rhat_values.get(name, float("nan"))
            rhat_s = f"<td>{rv:.4f}</td>"

        ess_b  = rep.ess_bulk.get(name, float("nan"))
        ess_t  = rep.ess_tail.get(name, float("nan"))
        mcse   = rep.mcse_sd_values.get(name, float("nan"))
        gew_z  = rep.geweke_z.get(name, float("nan"))
        hw_p   = rep.hw_passed.get(name, True)
        acf50  = rep.acf_values.get(name, {}).get(50, float("nan"))

        hw_s = "✓" if hw_p else "✗"
        html += (f"<tr class=\"{css}\"><td>{name}</td>"
                 + rhat_s
                 + f"<td>{ess_b:.0f}</td><td>{ess_t:.0f}</td>"
                 f"<td>{mcse:.3f}</td><td>{gew_z:.2f}</td>"
                 f"<td>{hw_s}</td>"
                 f"<td>{acf50:.3f}</td>"
                 f"<td class=\"{css}\">{status}</td></tr>\n")
    html += "</table>"
    return html


class _TraceAcfFigure:
    """One reusable trace + ACF figure for the HTML report.

    The report draws a trace and an ACF bar chart for many parameters.  Building
    a pyplot figure for each, with tight_layout and bbox_inches="tight", renders
    every figure two or three times and dominates the report's run time.  This
    class builds one matplotlib Figure with its own Agg canvas (no pyplot, so the
    user's backend is never changed), fixes the layout once, and for each
    parameter only updates the line data, the bar heights and the titles before
    writing one PNG.
    """

    LAGS = (1, 5, 10, 20, 50)

    def __init__(self, n_chains: int, height: float):
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        self.fig = Figure(figsize=(9, height), dpi=100)
        FigureCanvasAgg(self.fig)
        self.ax1, self.ax2 = self.fig.subplots(1, 2)
        self.fig.subplots_adjust(left=0.06, right=0.99, bottom=0.14, top=0.86, wspace=0.18)
        self.lines = [self.ax1.plot([], [], lw=0.6, alpha=0.7)[0] for _ in range(n_chains)]
        self.bars = self.ax2.bar(range(len(self.LAGS)), [0.0] * len(self.LAGS),
                                 tick_label=[str(lag) for lag in self.LAGS],
                                 color="steelblue", alpha=0.8)
        self.ax2.axhline(0.1, color="orange", ls="--", lw=0.8)
        from matplotlib.ticker import MaxNLocator
        for ax in (self.ax1, self.ax2):
            ax.tick_params(labelsize=6)
            ax.yaxis.set_major_locator(MaxNLocator(4))
        self.ax1.xaxis.set_major_locator(MaxNLocator(5, integer=True))

    # The trace axis is about 400 pixels wide, so a chain of 10,000 draws puts
    # 25 points in each pixel column.  Drawing each block's minimum and maximum
    # gives the same picture (every excursion stays visible) with ~800 points.
    MAX_TRACE_POINTS = 800

    @classmethod
    def _envelope(cls, y):
        n = len(y)
        if n <= cls.MAX_TRACE_POINTS:
            return np.arange(n), y
        k = -(-2 * n // cls.MAX_TRACE_POINTS)          # block length, ceil(2n / max)
        m = n // k * k
        blocks = y[:m].reshape(-1, k)
        x0 = np.arange(0, m, k)
        xs = np.column_stack([x0, x0 + k // 2]).ravel()
        ys = np.column_stack([blocks.min(axis=1), blocks.max(axis=1)]).ravel()
        if m < n:                                       # leftover tail draws
            xs = np.concatenate([xs, np.arange(m, n)])
            ys = np.concatenate([ys, y[m:]])
        return xs, ys

    def render(self, traces, acf_vals, title1: str, title2: str) -> str:
        lo, hi, n = np.inf, -np.inf, 1
        for line, y in zip(self.lines, traces, strict=True):
            xs, ys = self._envelope(y)
            line.set_data(xs, ys)
            if len(y):
                lo, hi, n = min(lo, np.nanmin(ys)), max(hi, np.nanmax(ys)), max(n, len(y))
        if np.isfinite(lo) and np.isfinite(hi):
            pad = 0.05 * (hi - lo) if hi > lo else 0.5
            self.ax1.set_ylim(lo - pad, hi + pad)
        self.ax1.set_xlim(0, n)
        vals = [0.0 if not np.isfinite(v) else float(v) for v in acf_vals]
        for bar, v in zip(self.bars, vals, strict=True):
            bar.set_height(v)
        lo, hi = min(0.0, min(vals)), max(0.1, max(vals))
        pad = 0.05 * (hi - lo)
        self.ax2.set_ylim(lo - pad, hi + pad)
        self.ax1.set_title(title1, fontsize=8)
        self.ax2.set_title(title2, fontsize=8)
        buf = io.BytesIO()
        self.fig.savefig(buf, format="png")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f'<img src="data:image/png;base64,{b64}" style="max-width:100%;">'


def _param_traces_and_acf(fit: IRTreeFit, idx: int):
    """Each chain's trace of parameter idx, and the first chain's ACF at LAGS."""
    from irtree.diagnostics import acf as _acf
    traces = [r.chain[:, idx].astype(float) for r in fit.results]
    av = _acf(traces[0], lags=_TraceAcfFigure.LAGS)
    return traces, [av[lag] for lag in _TraceAcfFigure.LAGS]


def _trace_acf_grid_html(fit: IRTreeFit, cap: int) -> str:
    """Per-parameter trace + ACF pairs for the first ``cap`` parameters."""
    names = fit.results[0].param_names[:cap]
    if not names:
        return ""
    plot = _TraceAcfFigure(len(fit.results), 2.3)
    parts = []
    for idx, name in enumerate(names):
        try:
            traces, acf_vals = _param_traces_and_acf(fit, idx)
            parts.append(plot.render(traces, acf_vals, f"{name}: trace", f"{name}: ACF"))
        except Exception:
            parts.append(f"<p>(could not plot {name})</p>")
    return "\n".join(parts)


def _flagged_section_html(fit: IRTreeFit, rep) -> str:
    if not rep.flagged_params:
        return "<p>No parameters flagged. All diagnostics within thresholds.</p>"

    plot = _TraceAcfFigure(len(fit.results), 2.5)
    index = {n: j for j, n in enumerate(fit.results[0].param_names)}
    parts = []
    for name in rep.flagged_params:
        status  = rep.status.get(name, "WARN")
        css     = {"PASS": "pass", "WARN": "warn", "FAIL": "fail"}.get(status, "")
        reasons = _build_reason_text(name, rep)

        parts.append(f'<h3 class="{css}">{name}: {status}</h3>')
        parts.append(reasons)   # already an HTML <ul> of per-indicator lines

        if name not in index:
            continue
        try:
            traces, acf_vals = _param_traces_and_acf(fit, index[name])
            parts.append(plot.render(traces, acf_vals, "Trace", "ACF"))
        except Exception as exc:
            parts.append(f"<p><em>(Plot unavailable: {exc})</em></p>")

    return "\n".join(parts)


def _build_reason_text(name: str, rep) -> str:
    """Build a full per-indicator readout for a flagged parameter.

    Reports ALL five indicators (Geweke sweep, Heidelberger-Welch, R-hat, ESS,
    MCSE/SD), each with its value AND a plain interpretation — not only the
    failing one. Seeing the passing indicators is the point: a parameter flagged
    by H-W alone, whose Geweke passes and precision is fine, is a very different
    situation from one where everything is bad, and the reader needs both.
    Returns an HTML <ul> of <li> lines (one per indicator).
    """
    from irtree.diagnostics import (
        ESS_WARN,
        MCSE_SD_WARN,
        RHAT_FAIL,
        RHAT_WARN,
    )
    lines = []

    # --- Geweke (three-way sweep) ---
    gstat = rep.geweke_status.get(name, "converged")
    gz = rep.geweke_z.get(name, 0.0)
    gk = rep.geweke_burnin.get(name)
    if gstat == "converged":
        lines.append(f"<b>Geweke</b>: PASS | stationary (z = {gz:.2f}), no burn-in needed.")
    elif gstat == "converged_after_burnin":
        pct = int(round(100 * gk)) if gk is not None else 0
        lines.append(
            f"<b>Geweke</b>: WARN | flagged at full length (z = {gz:.2f}) but "
            f"stationary after discarding the first ~{pct}%. The chain is fine; "
            f"it was under-warmed.")
    else:  # not_converged
        lines.append(
            f"<b>Geweke</b>: FAIL | never stationary across the discard sweep "
            f"(z = {gz:.2f}). The mean keeps drifting.")

    # --- Heidelberger-Welch ---
    hw = rep.hw_passed.get(name, True)
    if hw:
        lines.append("<b>Heidelberger-Welch</b>: PASS | the partial-sum process is "
                     "consistent with a stationary distribution.")
    else:
        n_disc = rep.hw_n_discarded.get(name)
        disc_s = ("even after discarding up to 50%"
                  if n_disc is None else f"after discarding {n_disc} samples")
        lines.append(f"<b>Heidelberger-Welch</b>: FAIL | non-stationary {disc_s} "
                     "(the whole distribution, not just the mean, is still drifting).")

    # --- the both-tests rule, stated inline for this parameter ---
    geweke_ok = (gstat != "not_converged")
    if (not geweke_ok) and (not hw):
        lines.append("<em>Both stationarity tests reject → this is a hard FAIL.</em>")
    elif (not geweke_ok) or (not hw):
        lines.append("<em>One stationarity test passes, the other flags → WARN "
                     "(the tests detect different kinds of drift).</em>")

    # --- R-hat (multi-chain only) ---
    rv = rep.rhat_values.get(name)
    if rv is not None:
        if rv > RHAT_FAIL:
            lines.append(f"<b>R-hat</b>: FAIL | {rv:.3f} > {RHAT_FAIL}. Chains have "
                         "not converged to the same distribution (multimodality or "
                         "too few iterations).")
        elif rv > RHAT_WARN:
            lines.append(f"<b>R-hat</b>: WARN | {rv:.3f} > {RHAT_WARN}. Chains mix "
                         "slowly; consider more iterations.")
        else:
            lines.append(f"<b>R-hat</b>: PASS | {rv:.3f} (chains agree).")
    else:
        lines.append("<b>R-hat</b>: n/a (single chain).")

    # --- ESS (bulk) ---
    ess_b = rep.ess_bulk.get(name, float("inf"))
    if ess_b < ESS_WARN:
        lines.append(f"<b>ESS (bulk)</b>: WARN | {ess_b:.0f} < {ESS_WARN}. High "
                     "autocorrelation; few effective draws. Run longer or thin.")
    else:
        lines.append(f"<b>ESS (bulk)</b>: PASS | {ess_b:.0f} effective draws.")

    # --- MCSE / SD (precision) ---
    mcse = rep.mcse_sd_values.get(name, 0.0)
    if mcse > MCSE_SD_WARN:
        lines.append(f"<b>MCSE/SD</b>: WARN | {mcse:.3f} > {MCSE_SD_WARN}. Monte "
                     f"Carlo noise is {mcse:.0%} of the posterior SD; estimates are "
                     "imprecise. More samples needed.")
    else:
        lines.append(f"<b>MCSE/SD</b>: PASS | {mcse:.3f} ({mcse:.0%} of the posterior "
                     "SD; precise).")

    return "<ul>" + "".join(f"<li>{ln}</li>" for ln in lines) + "</ul>"
