"""
Unit tests for the Kou double-exponential jump distribution.
"""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import expon, norm

from jump_diffusion.distributions import KouJump


class TestKouJump:
    """Test suite for KouJump."""

    def test_pdf_matches_scipy_expon_pieces(self):
        """The pdf should split into scaled one-sided exponentials."""
        kj = KouJump()
        params = {"jump_prob_up": 0.3, "jump_scale_up": 0.1, "jump_scale_down": 0.2}

        x_up = np.linspace(0.001, 1, 11)
        x_down = np.linspace(-1, -0.001, 11)

        assert np.allclose(kj.pdf(x_up, params), 0.3 * expon.pdf(x_up, scale=0.1))
        assert np.allclose(kj.pdf(x_down, params), 0.7 * expon.pdf(-x_down, scale=0.2))

    def test_pdf_integrates_to_one(self):
        kj = KouJump()
        params = {"jump_prob_up": 0.35, "jump_scale_up": 0.12, "jump_scale_down": 0.09}
        total, _ = quad(lambda j: kj.pdf(np.array([j]), params)[0], -5, 5, limit=1000)
        assert total == pytest.approx(1.0, rel=1e-6)

    def test_diffusion_convolved_pdf_matches_numerical_convolution(self):
        """
        The closed-form density of (diffusion + jump) must match direct
        numerical integration of the convolution integral. This is the
        real correctness check for the closed-form shortcut.
        """
        kj = KouJump()
        params = {"jump_prob_up": 0.35, "jump_scale_up": 0.12, "jump_scale_down": 0.09}
        diffusion_mean, diffusion_std = 0.05 / 252, 0.2 * np.sqrt(1 / 252)
        x0 = 0.02

        closed = kj.diffusion_convolved_pdf(
            np.array([x0]), params, diffusion_mean, diffusion_std
        )[0]

        def integrand(j):
            return (
                norm.pdf(x0 - j, loc=diffusion_mean, scale=diffusion_std)
                * kj.pdf(np.array([j]), params)[0]
            )

        true_value, _ = quad(integrand, -3, 3, limit=1000)

        assert closed == pytest.approx(true_value, rel=1e-6)

    def test_diffusion_convolved_pdf_matches_fft_fallback(self):
        """
        The closed form and the generic FFT-convolution fallback should
        agree within the FFT's numerical approximation error.
        """
        kj = KouJump()
        params = {"jump_prob_up": 0.35, "jump_scale_up": 0.12, "jump_scale_down": 0.09}
        diffusion_mean, diffusion_std = 0.05 / 252, 0.2 * np.sqrt(1 / 252)
        x = np.linspace(-0.05, 0.05, 21)

        closed = kj.diffusion_convolved_pdf(x, params, diffusion_mean, diffusion_std)
        fft = kj.fft_convolved_pdf(x, params, diffusion_mean, diffusion_std)

        assert np.max(np.abs((closed - fft) / closed)) < 0.01

    def test_rvs_recovers_moments(self):
        """Simulated jump sizes should match the theoretical mean/std."""
        kj = KouJump()
        params = {"jump_prob_up": 0.3, "jump_scale_up": 0.2, "jump_scale_down": 0.1}
        rng = np.random.default_rng(0)
        samples = kj.rvs(params, size=200_000, random_state=rng)

        p, su, sd = (
            params["jump_prob_up"],
            params["jump_scale_up"],
            params["jump_scale_down"],
        )
        expected_mean = p * su - (1 - p) * sd
        expected_var = p * 2 * su**2 + (1 - p) * 2 * sd**2 - expected_mean**2
        expected_std = np.sqrt(expected_var)

        assert np.isfinite(samples).all()
        assert np.mean(samples) == pytest.approx(expected_mean, abs=0.01)
        assert np.std(samples) == pytest.approx(expected_std, abs=0.01)

    def test_rvs_all_positive_when_prob_up_is_one(self):
        kj = KouJump()
        params = {
            "jump_prob_up": 1 - 1e-9,
            "jump_scale_up": 0.1,
            "jump_scale_down": 0.1,
        }
        rng = np.random.default_rng(1)
        samples = kj.rvs(params, size=1000, random_state=rng)
        assert np.all(samples > 0)
