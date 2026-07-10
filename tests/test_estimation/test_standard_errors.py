"""Unit tests for the likelihood profiling standard errors calculation."""

import numpy as np
import pytest
from jump_diffusion.distributions import NormalJump
from jump_diffusion.estimation import JumpDiffusionEstimator
from jump_diffusion.simulation import JumpDiffusionSimulator


def test_standard_errors_calculation_runs_and_estimates():
    # Simulate a small dataset to keep tests fast
    true_params = {
        "mu": 0.05,
        "sigma": 0.2,
        "jump_prob": 0.1,
        "jump_scale": 0.15,
    }
    sim = JumpDiffusionSimulator(jump_distribution=NormalJump(), **true_params)
    times, path, _ = sim.simulate_path(T=0.5, n_steps=50, x0=100.0, seed=123)
    increments = np.diff(path)
    dt = times[1] - times[0]

    # Fit estimator
    estimator = JumpDiffusionEstimator(increments, dt, jump_distribution=NormalJump())
    results = estimator.estimate()
    assert results["convergence"]

    # Calculate standard errors
    se_results = estimator.estimate_standard_errors(n_points=5, confidence_level=0.90)

    # Check results dict structure
    assert "standard_errors" in se_results
    assert "confidence_intervals" in se_results

    # Check that standard errors are computed for all parameters
    for param in estimator._param_names:
        se = se_results["standard_errors"][param]
        # It should be a positive float or nan if fitting fails (should be float here)
        assert np.isfinite(se)
        assert se > 0

        ci_low, ci_high = se_results["confidence_intervals"][param]
        assert np.isfinite(ci_low)
        assert np.isfinite(ci_high)
        assert ci_low < results["parameters"][param] < ci_high

    # Run diagnostics to verify it prints standard errors table without crashing
    import io
    import sys
    
    captured_output = io.StringIO()
    sys.stdout = captured_output
    try:
        estimator.diagnostics()
    finally:
        sys.stdout = sys.__stdout__
        
    output_str = captured_output.getvalue()
    assert "Parameter" in output_str
    assert "Estimate" in output_str
    assert "Std Error" in output_str
    assert "95% Conf. Interval" in output_str
    assert "mu" in output_str
    assert "sigma" in output_str
