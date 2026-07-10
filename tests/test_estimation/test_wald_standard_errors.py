"""Unit tests for the Wald (observed Fisher information) standard errors."""

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


def test_wald_standard_errors_structure_and_values():
    estimator, results = _fit_normal_jump()
    assert results["convergence"]

    wald = estimator.estimate_wald_standard_errors(confidence_level=0.95)

    assert "standard_errors" in wald
    assert "confidence_intervals" in wald

    # The diffusion parameters are well identified: expect finite, positive
    # standard errors whose Wald interval brackets the point estimate.
    for param in ("mu", "sigma", "jump_prob"):
        se = wald["standard_errors"][param]
        assert np.isfinite(se)
        assert se > 0

        ci_low, ci_high = wald["confidence_intervals"][param]
        assert np.isfinite(ci_low) and np.isfinite(ci_high)
        assert ci_low < results["parameters"][param] < ci_high

    # Every declared parameter must appear in both dicts.
    for param in estimator._param_names:
        assert param in wald["standard_errors"]
        assert param in wald["confidence_intervals"]


def test_wald_results_are_stored_on_estimator():
    estimator, _ = _fit_normal_jump()
    estimator.estimate_wald_standard_errors()

    assert "wald_standard_errors" in estimator.results
    assert "wald_confidence_intervals" in estimator.results
    assert "observed_information" in estimator.results


def test_observed_information_is_symmetric():
    estimator, _ = _fit_normal_jump()
    estimator.estimate_wald_standard_errors()

    info = np.asarray(estimator.results["observed_information"])
    k = len(estimator._param_names)
    assert info.shape == (k, k)
    # The Hessian is symmetric by construction (mixed partials mirrored).
    finite = np.isfinite(info)
    assert np.allclose(info[finite], info.T[finite])


def test_wald_and_profile_standard_errors_are_comparable_for_sigma():
    # For a well-identified parameter the two very different machineries
    # (Hessian inversion vs. likelihood profiling) should broadly agree.
    estimator, _ = _fit_normal_jump()

    profile = estimator.estimate_standard_errors(n_points=7)
    wald = estimator.estimate_wald_standard_errors()

    se_profile = profile["standard_errors"]["sigma"]
    se_wald = wald["standard_errors"]["sigma"]
    assert np.isfinite(se_profile) and np.isfinite(se_wald)
    ratio = se_wald / se_profile
    assert 0.2 < ratio < 5.0


def test_wald_requires_fitted_model():
    increments = np.random.default_rng(0).normal(size=100) * 0.1
    estimator = JumpDiffusionEstimator(increments, dt=1.0 / 252)
    with pytest.raises(ValueError):
        estimator.estimate_wald_standard_errors()
