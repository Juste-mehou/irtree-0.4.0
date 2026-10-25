"""
irtree.progress
===============
Dependency-free progress monitoring for the MCMC sampler.

Design:
  - **on by default**  — samplers report unless progress=False.
  - **log-likelihood + acceptance** on every line — the two nearly-free health
    signals.  The value is the data log-likelihood at the current state (the
    sampler's cached sum over nodes), without prior terms (keyword and label
    ``loglik``).
  - **stderr, with an optional callback** — the default callback prints a single
    self-updating line to stderr; a user may pass their own callback to log,
    feed a widget, etc.
  - **iteration-based cadence** — the reporter emits once every `print_every`
    iterations. Left unset, `print_every` defaults so the whole run prints at
    most ~10 times (every ceil(n_iter / 10) iterations), so a 200-iteration smoke
    run and a 200,000-iteration production run each yield about ten lines, with no
    per-iteration print cost. The user may set `print_every` to any interval.

No third-party dependencies: in-place redraw uses a bare carriage return when
stderr is a TTY, and falls back to plain scrolling lines otherwise (notebooks,
log files, pipes). Nothing here touches the sampling itself, so results are
unaffected.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable


def _fmt_hms(seconds: float) -> str:
    """Compact H:MM:SS / M:SS formatting for elapsed / ETA."""
    if seconds != seconds or seconds < 0:          # NaN or negative
        return "--:--"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


# ---------------------------------------------------------------------------
# The default sink: a single self-updating stderr line.
# ---------------------------------------------------------------------------

class _StderrLine:
    """Writes one status line to stderr, redrawing in place on a TTY.

    Stateless across calls except for remembering whether it is on a TTY (so it
    knows whether to use '\\r' or '\\n') and whether it has printed yet (to close
    the line with a newline at the end).
    """

    def __init__(self, stream=None):
        self.stream = stream if stream is not None else sys.stderr
        self.is_tty = bool(getattr(self.stream, "isatty", lambda: False)())
        self._opened = False

    def update(self, text: str) -> None:
        end = "\r" if self.is_tty else "\n"
        self.stream.write(text + end)
        self.stream.flush()
        self._opened = True

    def close(self, text: str | None = None) -> None:
        if text is not None:
            # final line always ends with a newline so subsequent output is clean
            self.stream.write(text + "\n")
        elif self._opened and self.is_tty:
            self.stream.write("\n")
        self.stream.flush()


# The callback signature the sampler calls. A user-supplied callback receives the
# same keyword arguments and may do anything (print, log, update a GUI).
ProgressCallback = Callable[..., None]


class ProgressReporter:
    """Adaptive-cadence progress for one chain.

    The sampler calls ``.step(t, ...)`` every iteration; the reporter emits
    only every ``print_every`` iterations (and at the last one), so the
    per-iteration cost is a single modulus check.

    Parameters
    ----------
    n_iter : int
        Total iterations, for percent / ETA.
    n_warmup : int
        Warmup count, shown so the user can see when sampling proper begins.
    enabled : bool
        Master switch (progress=... from the sampler).
    callback : callable or None
        If None, use the built-in stderr line. Otherwise this is called with
        keyword args (t, n_iter, n_warmup, elapsed, rate, eta, loglik,
        accept, phase, label); the built-in line is not used.  ``loglik``
        is the data log-likelihood at the current state (no prior terms);
        ``accept`` is the mean MH acceptance so far (None for pure Gibbs).
    print_every : int or None
        Emit once every ``print_every`` iterations (plus the final iteration).
        ``None`` (default) caps the whole run at ~10 updates by using
        ``max(1, ceil(n_iter / 10))`` — e.g. n_iter=10000 prints every 1000.
    min_interval : float
        Accepted and stored but not used: emission is iteration-based
        (``print_every``).
    label : str
        Optional prefix (used by irtree_fit to tag a chain, e.g. "chain 1/4").
    stream :
        Output stream for the default sink (defaults to sys.stderr).
    """

    def __init__(self, n_iter, n_warmup=0, enabled=True, callback=None,
                 min_interval=0.5, label="", stream=None, print_every=None):
        self.n_iter = int(n_iter)
        self.n_warmup = int(n_warmup)
        self.enabled = bool(enabled)
        self.callback = callback
        self.min_interval = float(min_interval)   # accepted, not used for gating
        self.label = label
        # Iteration-based cadence. Default: at most ~10 updates over the run.
        if print_every is None:
            self.print_every = max(1, (self.n_iter + 9) // 10)   # ceil(n_iter / 10)
        else:
            self.print_every = max(1, int(print_every))
        self._sink = _StderrLine(stream) if (enabled and callback is None) else None
        self._t0 = None
        self._last_emit = 0.0

    # -- lifecycle -------------------------------------------------------
    def start(self) -> None:
        self._t0 = time.perf_counter()
        self._last_emit = 0.0

    def step(self, t: int, loglik: float | None = None,
             accept: float | None = None) -> None:
        """Called every iteration; emits once every ``print_every`` iterations."""
        if not self.enabled:
            return
        if self._t0 is None:
            self.start()
        is_last = (t == self.n_iter - 1)
        due = ((t + 1) % self.print_every == 0)
        if not (due or is_last):
            return
        elapsed = time.perf_counter() - self._t0
        self._last_emit = elapsed
        self._emit(t + 1, elapsed, loglik, accept)

    def finish(self, elapsed: float | None = None) -> None:
        """Close the line (default sink only)."""
        if not self.enabled:
            return
        if self._sink is not None:
            if elapsed is None and self._t0 is not None:
                elapsed = time.perf_counter() - self._t0
            self._sink.close(self._final_text(elapsed))

    # -- emission --------------------------------------------------------
    def _phase(self, done: int) -> str:
        return "warmup" if done <= self.n_warmup else "sample"

    def _emit(self, done, elapsed, loglik, accept) -> None:
        rate = done / elapsed if elapsed > 0 else float("nan")
        eta = (self.n_iter - done) / rate if rate and rate == rate else float("nan")
        if self.callback is not None:
            self.callback(
                t=done, n_iter=self.n_iter, n_warmup=self.n_warmup,
                elapsed=elapsed, rate=rate, eta=eta,
                loglik=loglik, accept=accept, phase=self._phase(done),
                label=self.label,
            )
        else:
            self._sink.update(self._line(done, elapsed, rate, eta, loglik, accept))

    def _line(self, done, elapsed, rate, eta, loglik, accept) -> str:
        pct = 100.0 * done / self.n_iter
        parts = []
        if self.label:
            parts.append(self.label)
        parts.append(f"{self._phase(done)} {done}/{self.n_iter} ({pct:.0f}%)")
        parts.append(f"{rate:.1f} it/s")
        parts.append(f"elapsed {_fmt_hms(elapsed)}")
        parts.append(f"eta {_fmt_hms(eta)}")
        if loglik is not None and loglik == loglik:
            parts.append(f"loglik {loglik:.1f}")
        if accept is not None and accept == accept:
            parts.append(f"accept {accept:.2f}")
        return " · ".join(parts)

    def _final_text(self, elapsed) -> str:
        parts = []
        if self.label:
            parts.append(self.label)
        parts.append(f"done {self.n_iter}/{self.n_iter} (100%)")
        parts.append(f"elapsed {_fmt_hms(elapsed)}")
        return " · ".join(parts)


def mean_acceptance(adapt_states) -> float | None:
    """Mean acceptance across all MH blocks so far (None if no MH blocks).

    Cheap: reads the running n_accepted / n_proposed already tracked in the
    AdaptState objects. Returns None for a pure-Gibbs chain (no MH blocks), so
    the reporter simply omits the accept field.
    """
    rates = []
    for st in adapt_states.values():
        n_prop = getattr(st, "n_proposed", 0)
        if n_prop > 0:
            rates.append(getattr(st, "n_accepted", 0) / n_prop)
    if not rates:
        return None
    return sum(rates) / len(rates)
