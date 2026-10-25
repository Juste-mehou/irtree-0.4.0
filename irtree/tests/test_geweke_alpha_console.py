"""console_summary's multi-chain Geweke line uses the report's geweke_alpha."""
from irtree.diagnostics import DiagnosticReport, console_summary


def _report(alpha):
    names = ["p1", "p2"]
    one = {n: 1.0 for n in names}
    return DiagnosticReport(
        summary_status="PASS", n_chains=2, n_iter=100, n_samples=100,
        param_names=names, flagged_params=[], rhat_values=one,
        ess_bulk={n: 1000.0 for n in names}, ess_tail={n: 1000.0 for n in names},
        mcse_values=one, mcse_sd_values={n: 0.01 for n in names},
        geweke_z={"p1": 2.5, "p2": 0.1}, geweke_p={n: 0.5 for n in names},
        acf_values={n: {} for n in names}, status={n: "PASS" for n in names},
        geweke_alpha=alpha)


def _geweke_line(out):
    return next(l for l in out.splitlines() if "Stationarity" in l and "Geweke" in l)


def test_geweke_line_follows_alpha(capsys):
    console_summary(_report(0.003))          # |z| cutoff ~2.97: p1 passes
    assert "failed" not in _geweke_line(capsys.readouterr().out)
    console_summary(_report(0.05))           # cutoff 1.96: p1 flagged
    line = _geweke_line(capsys.readouterr().out)
    assert "1 parameters failed" in line and "p1" in line
