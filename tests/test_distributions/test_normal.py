"""
Unit tests for the normal (Merton) jump distribution.
"""

import numpy as np
import pytest
from scipy.stats import norm

from jump_diffusion.distributions import NormalJump


class TestNormalJump:
    """Test suite for NormalJump."""

    def test_pdf_matches_scipy(self):
        nj = NormalJump()
        x = np.linspace(-1, 1, 11)
        params = {"jump_scale": 0.2}
        assert np.allclose(nj.pdf(x, params), norm.pdf(x, loc=0, scale=0.2))

    def test_diffusion_convolved_pdf_is_normal_normal_sum(self):
        """Normal + Normal must be Normal with summed variances."""
        nj = NormalJump()
        params = {"jump_scale": 0.15}
        diffusion_mean, diffusion_std = 0.05 / 252, 0.2 * np.sqrt(1 / 252)
        x = np.linspace(-0.05, 0.05, 21)

        closed = nj.diffusion_convolved_pdf(x, params, diffusion_mean, diffusion_std)
        expected_std = np.sqrt(diffusion_std**2 + params["jump_scale"] ** 2)
        expected = norm.pdf(x, loc=diffusion_mean, scale=expected_std)

        assert np.allclose(closed, expected)

    def test_diffusion_convolved_pdf_matches_fft_fallback(self):
        nj = NormalJump()
        params = {"jump_scale": 0.15}
        diffusion_mean, diffusion_std = 0.05 / 252, 0.2 * np.sqrt(1 / 252)
        x = np.linspace(-0.05, 0.05, 21)

        closed = nj.diffusion_convolved_pdf(x, params, diffusion_mean, diffusion_std)
        fft = nj.fft_convolved_pdf(x, params, diffusion_mean, diffusion_std)

        assert np.max(np.abs((closed - fft) / closed)) < 0.01

    def test_rvs_recovers_moments(self):
        nj = NormalJump()
        params = {"jump_scale": 0.2}
        rng = np.random.default_rng(0)
        samples = nj.rvs(params, size=200_000, random_state=rng)

        assert np.mean(samples) == pytest.approx(0.0, abs=0.01)
        assert np.std(samples) == pytest.approx(0.2, abs=0.01)
