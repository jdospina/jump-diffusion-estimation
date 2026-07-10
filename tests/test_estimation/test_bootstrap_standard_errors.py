"""Unit tests for the parametric bootstrap standard errors."""

import numpy as np
import pytest
from jump_diffusion.distributions import NormalJump
from jump_diffusion.estimation import JumpDiffusionEstimator
from jump_diffusion.simulation import JumpDiffusionSimulator


def _fit_normal_jump(seed: int = 123):
    """Simulate a moderate Normal-jump dataset and fit an estimator to it."""
    true_params = {
        "mu": 0.05,
        "sigma": 0.2,
        "jump_prob": 0.1,
        "jump_scale": 0.15,
    }
    sim = JumpDiffusionSimulator(jump_distribution=NormalJump(), **true_params)
    times, path, _ = sim.simulate_path(T=1.0, n_steps=300, x0=100.0, seed=seed)
    increments = np.diff(path)
    dt = times[1] - times[0]

    estimator = JumpDiffusionEstimator(increments, dt, jump_distribution=NormalJump())
    results = estimator.estimate()
    return estimator, results


def test_bootstrap_standard_errors_structure_and_values():
    estimator, results = _fit_normal_jump()

    boot = estimator.estimate_bootstrap_standard_errors(
        n_bootstrap=20, confidence_level=0.95, seed=0
    )

    assert "standard_errors" in boot
    assert "confidence_intervals" in boot
    assert boot["n_successful"] > 0

    for param in ("mu", "sigma", "jump_prob"):
        se = boot["standard_errors"][param]
        assert np.isfinite(se)
        assert se > 0

        ci_low, ci_high = boot["confidence_intervals"][param]
        assert np.isfinite(ci_low) and np.isfinite(ci_high)
        assert ci_low < ci_high

    # The point estimate should sit inside its percentile interval, since
    # replicates are simulated from the fitted parameters themselves.
    ci_low, ci_high = boot["confidence_intervals"]["sigma"]
    assert ci_low <= results["parameters"]["sigma"] <= ci_high


def test_bootstrap_results_are_stored_on_estimator():
    estimator, _ = _fit_normal_jump()
    estimator.estimate_bootstrap_standard_errors(n_bootstrap=10, seed=0)

    assert "bootstrap_standard_errors" in estimator.results
    assert "bootstrap_confidence_intervals" in estimator.results
    assert "bootstrap_estimates" in estimator.results

    estimates = np.asarray(estimator.results["bootstrap_estimates"])
    # One row per successful replicate, one column per parameter.
    assert estimates.shape[1] == len(estimator._param_names)


def test_bootstrap_is_reproducible_with_seed():
    estimator, _ = _fit_normal_jump()

    first = estimator.estimate_bootstrap_standard_errors(n_bootstrap=10, seed=42)
    second = estimator.estimate_bootstrap_standard_errors(n_bootstrap=10, seed=42)

    for param in estimator._param_names:
        assert first["standard_errors"][param] == pytest.approx(
            second["standard_errors"][param]
        )


def test_bootstrap_requires_fitted_model():
    increments = np.random.default_rng(0).normal(size=100) * 0.1
    estimator = JumpDiffusionEstimator(increments, dt=1.0 / 252)
    with pytest.raises(ValueError):
        estimator.estimate_bootstrap_standard_errors(n_bootstrap=5)
