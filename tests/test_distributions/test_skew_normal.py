"""
Unit tests for the skew-normal jump distribution.
"""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm, skewnorm

from jump_diffusion.distributions import SkewNormalJump


class TestSkewNormalJump:
    """Test suite for SkewNormalJump."""

    def test_pdf_matches_scipy(self):
        """The pdf should match scipy's skewnorm at symmetric params."""
        sn = SkewNormalJump()
        x = np.linspace(-1, 1, 11)
        params = {"jump_scale": 0.2, "jump_skew": 0.0}
        density = sn.pdf(x, params)
        assert np.allclose(density, norm.pdf(x, loc=0, scale=0.2))

    def test_diffusion_convolved_pdf_matches_numerical_convolution(self):
        """
        The closed-form density of (diffusion + jump) must match direct
        numerical integration of the convolution integral. This is the
        real correctness check for the closed-form shortcut.
        """
        sn = SkewNormalJump()
        params = {"jump_scale": 0.15, "jump_skew": 2.0}
        diffusion_mean, diffusion_std = 0.05 / 252, 0.2 * np.sqrt(1 / 252)
        x0 = 0.02

        closed = sn.diffusion_convolved_pdf(
            np.array([x0]), params, diffusion_mean, diffusion_std
        )[0]

        def integrand(j):
            return norm.pdf(
                x0 - j, loc=diffusion_mean, scale=diffusion_std
            ) * sn.pdf(np.array([j]), params)[0]

        true_value, _ = quad(integrand, -2, 2, limit=500)

        assert closed == pytest.approx(true_value, rel=1e-6)

    def test_diffusion_convolved_pdf_matches_fft_fallback(self):
        """
        The closed form and the generic FFT-convolution fallback should
        agree within the FFT's numerical approximation error.
        """
        sn = SkewNormalJump()
        params = {"jump_scale": 0.15, "jump_skew": 2.0}
        diffusion_mean, diffusion_std = 0.05 / 252, 0.2 * np.sqrt(1 / 252)
        x = np.linspace(-0.05, 0.05, 21)

        closed = sn.diffusion_convolved_pdf(x, params, diffusion_mean, diffusion_std)
        fft = sn.fft_convolved_pdf(x, params, diffusion_mean, diffusion_std)

        assert np.max(np.abs((closed - fft) / closed)) < 0.01

    def test_rvs_recovers_moments(self):
        """Simulated jump sizes should match the theoretical mean/std."""
        sn = SkewNormalJump()
        params = {"jump_scale": 0.2, "jump_skew": 3.0}
        rng = np.random.default_rng(0)
        samples = sn.rvs(params, size=200_000, random_state=rng)

        expected_mean = skewnorm.mean(a=params["jump_skew"], scale=params["jump_scale"])
        expected_std = skewnorm.std(a=params["jump_skew"], scale=params["jump_scale"])

        assert np.isfinite(samples).all()
        assert np.mean(samples) == pytest.approx(expected_mean, abs=0.01)
        assert np.std(samples) == pytest.approx(expected_std, abs=0.01)
