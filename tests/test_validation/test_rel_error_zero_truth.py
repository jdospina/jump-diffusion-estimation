"""Regression tests: ValidationExperiment with a zero true parameter
(audit finding B4) must not produce infinite relative errors."""

import contextlib
import io

import numpy as np

from jump_diffusion.distributions import SkewNormalJump
from jump_diffusion.validation import ValidationExperiment

TRUE_PARAMS = {
    "mu": 0.05,
    "sigma": 0.2,
    "jump_prob": 0.2,
    "jump_scale": 0.1,
    "jump_skew": 0.0,  # true value exactly zero
}


def _run_experiment():
    experiment = ValidationExperiment(TRUE_PARAMS, jump_distribution=SkewNormalJump())
    with contextlib.redirect_stdout(io.StringIO()):
        results = experiment.run_experiment(n_simulations=3, n_steps=300, seed_base=1)
    return experiment, results


def test_zero_truth_yields_nan_not_inf():
    _, results = _run_experiment()
    assert len(results) > 0

    rel = results["jump_skew_rel_error"].to_numpy()
    assert not np.any(np.isinf(rel))
    assert np.all(np.isnan(rel))

    # Non-zero parameters keep finite relative errors.
    assert np.all(np.isfinite(results["sigma_rel_error"]))


def test_analysis_is_inf_free_and_reports_accuracy_rate():
    experiment, _ = _run_experiment()
    with contextlib.redirect_stdout(io.StringIO()):
        analysis = experiment.analyze_results()

    skew_stats = analysis["jump_skew"]
    # Relative measures are undefined for a zero truth: nan, never inf.
    assert np.isnan(skew_stats["mean_rel_error"])
    assert np.isnan(skew_stats["within_5pct_rate"])
    # Absolute measures remain finite.
    assert np.isfinite(skew_stats["bias"])
    assert np.isfinite(skew_stats["rmse"])

    # The honestly-named accuracy rate replaces the old "coverage_95".
    assert "coverage_95" not in skew_stats
    sigma_stats = analysis["sigma"]
    assert 0.0 <= sigma_stats["within_5pct_rate"] <= 1.0
