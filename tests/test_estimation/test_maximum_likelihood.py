"""
Unit tests for maximum likelihood estimation.
"""

import numpy as np
from jump_diffusion.estimation import JumpDiffusionEstimator
from jump_diffusion.simulation import JumpDiffusionSimulator


class TestJumpDiffusionEstimator:
    """Test suite for JumpDiffusionEstimator class."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.increments = np.random.normal(0.001, 0.02, 252)
        self.dt = 1 / 252
        self.estimator = JumpDiffusionEstimator(self.increments, self.dt)

    def test_initialization(self):
        """Test estimator initialization."""
        assert len(self.estimator.increments) == 252
        assert self.estimator.dt == 1 / 252
        assert not self.estimator.fitted
        assert self.estimator.n_obs == 252

    def test_log_likelihood(self):
        """Test log-likelihood calculation."""
        params = np.array(
            [
                0.05,
                0.2,
                0.1,
                0.15,
                1.0,
            ]
        )  # [mu, sigma, jump_prob, jump_scale, jump_skew]
        log_lik = self.estimator.log_likelihood(params)

        assert np.isfinite(log_lik)
        assert isinstance(log_lik, float)

    def test_parameter_estimation(self):
        """Test parameter estimation process."""
        results = self.estimator.estimate()

        assert self.estimator.fitted
        assert "parameters" in results
        assert "log_likelihood" in results
        assert "convergence" in results
        assert len(results["parameters"]) == 5

    def test_diagnostics(self):
        """Test diagnostic output."""
        self.estimator.estimate()
        # This should not raise an exception
        self.estimator.diagnostics()

    def test_estimation_on_simulated_path(self):
        """Estimate parameters from a simulated jump-diffusion path."""
        sim = JumpDiffusionSimulator(
            mu=0.02, sigma=0.1, jump_prob=0.05, jump_scale=0.2, jump_skew=0.0
        )
        _, path, _ = sim.simulate_path(T=1.0, n_steps=int(10000), x0=0.0, seed=int(42))
        increments = np.diff(path)
        estimator = JumpDiffusionEstimator(increments, 1 / 10000)
        results = estimator.estimate()

        params = results["parameters"]
        assert np.isclose(params["mu"], 0.02, atol=0.02)
        assert np.isclose(params["sigma"], 0.1, atol=0.02)
