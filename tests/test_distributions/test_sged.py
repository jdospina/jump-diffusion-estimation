"""
Unit tests for the Skewed Generalized Error Distribution (SGED) jump
distribution, ported from Ospina Arango (2009), "Estimacion de un modelo
de difusion con saltos con distribucion de error generalizada asimetrica
usando algoritmos evolutivos" (Universidad Nacional de Colombia).
"""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import laplace, norm

from jump_diffusion.distributions import SGEDJump
from jump_diffusion.estimation import JumpDiffusionEstimator
from jump_diffusion.simulation import JumpDiffusionSimulator


class TestSGEDJump:
    """Test suite for SGEDJump."""

    def test_pdf_integrates_to_one(self):
        sged = SGEDJump()
        params = {
            "jump_loc": 0.0,
            "jump_scale": 0.2,
            "jump_nu": 1.5,
            "jump_xi": 2.0,
        }
        value, _ = quad(
            lambda x: sged.pdf(np.array([x]), params)[0], -10, 10, limit=200
        )
        assert value == pytest.approx(1.0, abs=1e-6)

    def test_nu2_xi1_matches_standard_normal(self):
        """SGED(0,1,2,1) is the standard normal (per Theodossiou 2000)."""
        sged = SGEDJump()
        params = {"jump_loc": 0.0, "jump_scale": 1.0, "jump_nu": 2.0, "jump_xi": 1.0}
        x = np.linspace(-4, 4, 17)
        assert np.allclose(sged.pdf(x, params), norm.pdf(x), atol=1e-10)

    def test_nu1_xi1_matches_standard_laplace(self):
        """SGED(0,1,1,1) is the unit-variance (standardized) Laplace."""
        sged = SGEDJump()
        params = {"jump_loc": 0.0, "jump_scale": 1.0, "jump_nu": 1.0, "jump_xi": 1.0}
        x = np.linspace(-4, 4, 17)
        assert np.allclose(
            sged.pdf(x, params), laplace.pdf(x, scale=1 / np.sqrt(2)), atol=1e-10
        )

    def test_rvs_recovers_moments(self):
        """jump_loc/jump_scale are E[X] and sqrt(Var[X]) by construction."""
        sged = SGEDJump()
        params = {
            "jump_loc": 0.1,
            "jump_scale": 0.2,
            "jump_nu": 1.5,
            "jump_xi": 2.0,
        }
        rng = np.random.default_rng(0)
        samples = sged.rvs(params, size=200_000, random_state=rng)

        assert np.isfinite(samples).all()
        assert np.mean(samples) == pytest.approx(0.1, abs=0.01)
        assert np.std(samples) == pytest.approx(0.2, abs=0.01)

    def test_mle_recovers_known_parameters(self):
        """
        Replicates the applied experiment from the thesis: simulate with
        known parameters and confirm MLE recovers them from a nearby
        initial guess. Plain L-BFGS-B needs a reasonable starting point
        for this mixture likelihood (the thesis's own finding -- this is
        exactly why it also explores Differential Evolution, out of scope
        here).
        """
        true_params = {
            "mu": 0.25,
            "sigma": 0.5,
            "jump_prob": 10 / 260,
        }
        true_jump_params = {
            "jump_loc": 0.0,
            "jump_scale": 0.2,
            "jump_nu": 1.5,
            "jump_xi": 2.0,
        }

        sim = JumpDiffusionSimulator(
            jump_distribution=SGEDJump(), **true_params, **true_jump_params
        )
        times, path, _ = sim.simulate_path(T=5.0, n_steps=1300, x0=100.0, seed=123)
        increments = np.diff(path)
        dt = times[1] - times[0]

        estimator = JumpDiffusionEstimator(
            increments, dt, jump_distribution=SGEDJump()
        )
        initial_guess = np.array(
            [
                true_params["mu"],
                true_params["sigma"],
                true_params["jump_prob"],
                true_jump_params["jump_loc"],
                true_jump_params["jump_scale"],
                true_jump_params["jump_nu"],
                true_jump_params["jump_xi"],
            ]
        )
        result = estimator.estimate(initial_guess=initial_guess)

        assert result["convergence"]
        assert result["parameters"]["sigma"] == pytest.approx(0.5, abs=0.05)
        assert result["parameters"]["jump_scale"] == pytest.approx(0.2, abs=0.05)
