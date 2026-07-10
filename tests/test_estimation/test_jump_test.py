"""Unit tests for the parametric bootstrap test for the presence of jumps."""

import numpy as np
import pytest
from jump_diffusion.distributions import NormalJump
from jump_diffusion.estimation import JumpDiffusionEstimator
from jump_diffusion.simulation import JumpDiffusionSimulator


def _fit_with_jumps(seed: int = 123):
    """Fit an estimator on data that genuinely contains jumps."""
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
    estimator.estimate()
    return estimator


def _fit_pure_diffusion(seed: int = 7):
    """Fit an estimator on pure-diffusion (no-jump) data."""
    rng = np.random.default_rng(seed)
    dt = 1.0 / 252
    increments = rng.normal(0.05 * dt, 0.2 * np.sqrt(dt), size=300)
    estimator = JumpDiffusionEstimator(increments, dt, jump_distribution=NormalJump())
    estimator.estimate()
    return estimator


def test_jump_test_detects_jumps():
    estimator = _fit_with_jumps()
    result = estimator.test_for_jumps(n_bootstrap=30, seed=0)

    assert 0.0 <= result["p_value"] <= 1.0
    assert result["lr_statistic"] > 0.0
    assert result["n_successful"] > 0
    # The jumps here dwarf the diffusion, so the test should reject H0.
    assert result["p_value"] < 0.2


def test_jump_test_structure_under_null():
    estimator = _fit_pure_diffusion()
    result = estimator.test_for_jumps(n_bootstrap=30, seed=0)

    assert 0.0 <= result["p_value"] <= 1.0
    assert result["lr_statistic"] >= 0.0
    assert result["log_likelihood_full"] >= result["log_likelihood_null"] - 1e-6
    assert len(result["bootstrap_statistics"]) == result["n_successful"]


def test_jump_test_is_stored_and_reproducible():
    estimator = _fit_with_jumps()

    first = estimator.test_for_jumps(n_bootstrap=15, seed=42)
    assert "jump_test" in estimator.results

    second = estimator.test_for_jumps(n_bootstrap=15, seed=42)
    assert first["p_value"] == pytest.approx(second["p_value"])
    assert first["lr_statistic"] == pytest.approx(second["lr_statistic"])


def test_jump_test_requires_fitted_model():
    increments = np.random.default_rng(0).normal(size=100) * 0.1
    estimator = JumpDiffusionEstimator(increments, dt=1.0 / 252)
    with pytest.raises(ValueError):
        estimator.test_for_jumps(n_bootstrap=5)
