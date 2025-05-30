"""
Unit tests for maximum likelihood estimation.
"""

import numpy as np
import pytest
from jump_diffusion.estimation import JumpDiffusionEstimator

class TestJumpDiffusionEstimator:
    """Test suite for JumpDiffusionEstimator class."""
    
    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.increments = np.random.normal(0.001, 0.02, 252)
        self.dt = 1/252
        self.estimator = JumpDiffusionEstimator(self.increments, self.dt)
    
    def test_initialization(self):
        """Test estimator initialization."""
        assert len(self.estimator.increments) == 252
        assert self.estimator.dt == 1/252
        assert not self.estimator.fitted
        assert self.estimator.n_obs == 252
    
    def test_log_likelihood(self):
        """Test log-likelihood calculation."""
        params = [0.05, 0.2, 0.1, 0.15, 1.0]  # [mu, sigma, jump_prob, jump_scale, jump_skew]
        log_lik = self.estimator.log_likelihood(params)
        
        assert np.isfinite(log_lik)
        assert isinstance(log_lik, float)
    
    def test_parameter_estimation(self):
        """Test parameter estimation process."""
        results = self.estimator.estimate()
        
        assert self.estimator.fitted
        assert 'parameters' in results
        assert 'log_likelihood' in results
        assert 'convergence' in results
        assert len(results['parameters']) == 5
    
    def test_diagnostics(self):
        """Test diagnostic output."""
        self.estimator.estimate()
        # This should not raise an exception
        self.estimator.diagnostics()
