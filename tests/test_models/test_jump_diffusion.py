"""
Unit tests for jump-diffusion models.
"""

import numpy as np
import pytest
from jump_diffusion.models import JumpDiffusionModel

class TestJumpDiffusionModel:
    """Test suite for JumpDiffusionModel class."""
    
    def test_initialization(self):
        """Test model initialization with default parameters."""
        model = JumpDiffusionModel()
        params = model.get_parameters()
        
        assert 'mu' in params
        assert 'sigma' in params
        assert 'jump_prob' in params
        assert 'jump_scale' in params
        assert 'jump_skew' in params
        assert not model.fitted
    
    def test_parameter_update(self):
        """Test parameter updating functionality."""
        model = JumpDiffusionModel(mu=0.05)
        assert model.get_parameters()['mu'] == 0.05
        
        model.update_parameters(mu=0.08)
        assert model.get_parameters()['mu'] == 0.08
    
    def test_simulation(self):
        """Test simulation functionality."""
        model = JumpDiffusionModel(mu=0.05, sigma=0.2, jump_prob=0.1)
        times, path, jumps = model.simulate(T=1.0, n_steps=252, x0=100.0, seed=42)
        
        assert len(times) == 253  # n_steps + 1
        assert len(path) == 253
        assert len(jumps) == 252
        assert path[0] == 100.0
        assert np.all(np.isfinite(path))
    
    def test_log_likelihood(self):
        """Test log-likelihood calculation."""
        np.random.seed(42)
        increments = np.random.normal(0, 0.1, 252)
        model = JumpDiffusionModel()
        
        log_lik = model.log_likelihood(increments, dt=1/252)
        assert np.isfinite(log_lik)
        assert isinstance(log_lik, float)
    
    def test_parameter_bounds(self):
        """Test parameter bounds."""
        model = JumpDiffusionModel()
        bounds = model.get_parameter_bounds()
        
        assert len(bounds) == 5  # Five parameters
        assert all(isinstance(bound, tuple) for bound in bounds)
