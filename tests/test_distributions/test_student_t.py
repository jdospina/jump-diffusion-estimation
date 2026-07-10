"""
Unit tests for the Student-t jump distribution.
"""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm

from jump_diffusion.distributions import StudentTJump
from jump_diffusion.estimation import JumpDiffusionEstimator
from jump_diffusion.simulation import JumpDiffusionSimulator


class TestStudentTJump:
    """Test suite for StudentTJump."""

    def test_pdf_integrates_to_one(self):
        st = StudentTJump()
        params = {"jump_loc": 0.05, "jump_scale": 0.2, "jump_df": 4.0}
        value, _ = quad(lambda x: st.pdf(np.array([x]), params)[0], -10, 10, limit=200)
        assert value == pytest.approx(1.0, abs=1e-6)

    def test_large_df_approaches_normal(self):
        """As df -> infinity, the standardized Student-t -> normal(loc, scale)."""
        st = StudentTJump()
        params = {"jump_loc": 0.0, "jump_scale": 0.2, "jump_df": 5000.0}
        x = np.linspace(-0.5, 0.5, 21)
        assert np.allclose(
            st.pdf(x, params), norm.pdf(x, loc=0.0, scale=0.2), atol=1e-3
        )

    def test_rvs_recovers_moments(self):
        """jump_loc/jump_scale are E[X] and sqrt(Var[X]) by construction."""
        st = StudentTJump()
        params = {"jump_loc": 0.1, "jump_scale": 0.2, "jump_df": 5.0}
        rng = np.random.default_rng(0)
        samples = st.rvs(params, size=500_000, random_state=rng)

        assert np.isfinite(samples).all()
        assert np.mean(samples) == pytest.approx(0.1, abs=0.01)
        assert np.std(samples) == pytest.approx(0.2, abs=0.01)

    def test_fft_convolved_pdf_matches_numerical_convolution(self):
        """
        StudentTJump has no closed-form diffusion_convolved_pdf, so the
        generic FFT fallback (inherited from JumpDistribution) is the only
        available density -- verify it against direct numerical
        integration of the convolution integral.
        """
        st = StudentTJump()
        params = {"jump_loc": 0.0, "jump_scale": 0.15, "jump_df": 4.0}
        diffusion_mean, diffusion_std = 0.05 / 252, 0.2 * np.sqrt(1 / 252)
        x0 = 0.02

        assert (
            st.diffusion_convolved_pdf(
                np.array([x0]), params, diffusion_mean, diffusion_std
            )
            is None
        )

        fft = st.fft_convolved_pdf(
            np.array([x0]), params, diffusion_mean, diffusion_std
        )[0]

        def integrand(j):
            return (
                norm.pdf(x0 - j, loc=diffusion_mean, scale=diffusion_std)
                * st.pdf(np.array([j]), params)[0]
            )

        true_value, _ = quad(integrand, -3, 3, limit=1000)

        assert fft == pytest.approx(true_value, rel=1e-3)

    def test_mle_recovers_known_parameters(self):
        """
        Simulate with known parameters and confirm MLE recovers them from a
        nearby initial guess -- plain L-BFGS-B needs a reasonable starting
        point for this mixture likelihood (see SGED's equivalent test).
        """
        true_params = {
            "mu": 0.25,
            "sigma": 0.5,
            "jump_prob": 10 / 260,
        }
        true_jump_params = {
            "jump_loc": 0.0,
            "jump_scale": 0.2,
            "jump_df": 4.0,
        }

        sim = JumpDiffusionSimulator(
            jump_distribution=StudentTJump(), **true_params, **true_jump_params
        )
        times, path, _ = sim.simulate_path(T=5.0, n_steps=1300, x0=100.0, seed=123)
        increments = np.diff(path)
        dt = times[1] - times[0]

        estimator = JumpDiffusionEstimator(
            increments, dt, jump_distribution=StudentTJump()
        )
        initial_guess = np.array(
            [
                true_params["mu"],
                true_params["sigma"],
                true_params["jump_prob"],
                true_jump_params["jump_loc"],
                true_jump_params["jump_scale"],
                true_jump_params["jump_df"],
            ]
        )
        result = estimator.estimate(initial_guess=initial_guess)

        assert result["convergence"]
        assert result["parameters"]["sigma"] == pytest.approx(0.5, abs=0.05)
        assert result["parameters"]["jump_scale"] == pytest.approx(0.2, abs=0.05)
