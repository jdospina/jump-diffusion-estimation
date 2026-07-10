"""
Unit tests for the differential-evolution estimation method.

The headline test replicates the applied finding of Ospina Arango (2009):
on the SGED mixture likelihood, L-BFGS-B started from the default
moment-based guess gets stuck (poor local solution, no convergence),
while differential evolution -- needing no initial guess at all, only a
wide data-driven search box -- reaches a far better optimum and recovers
the well-identified parameters.
"""

import numpy as np
import pytest

from jump_diffusion.distributions import SGEDJump
from jump_diffusion.estimation import JumpDiffusionEstimator
from jump_diffusion.simulation import JumpDiffusionSimulator


def _simulated_sged_increments():
    true_params = {"mu": 0.25, "sigma": 0.5, "jump_prob": 10 / 260}
    true_jump = {
        "jump_loc": 0.0,
        "jump_scale": 0.2,
        "jump_nu": 1.5,
        "jump_xi": 2.0,
    }
    sim = JumpDiffusionSimulator(
        jump_distribution=SGEDJump(), **true_params, **true_jump
    )
    times, path, _ = sim.simulate_path(T=2.5, n_steps=650, x0=100.0, seed=123)
    return np.diff(path), times[1] - times[0]


class TestDifferentialEvolution:
    """Test suite for estimate(method="differential_evolution")."""

    def test_de_recovers_sged_parameters_without_initial_guess(self):
        """
        The thesis's key applied result: differential evolution needs no
        initial guess on the SGED mixture likelihood -- it must converge,
        recover the well-identified parameters, and never end up worse
        than L-BFGS-B started from the default moment-based guess.

        Whether L-BFGS-B itself fails here is platform-dependent (its line
        search takes different trajectories under different BLAS builds:
        it fails outright on macOS but can converge on Linux CI), so this
        test only asserts the portable claims. The L-BFGS-B failure mode
        that motivates DE is documented on real S&P 500 data in
        notebooks/sp500_jump_diffusion_example.ipynb.
        """
        increments, dt = _simulated_sged_increments()

        lbfgsb = JumpDiffusionEstimator(
            increments, dt, jump_distribution=SGEDJump()
        ).estimate()

        de = JumpDiffusionEstimator(
            increments, dt, jump_distribution=SGEDJump()
        ).estimate(method="differential_evolution", seed=42)

        assert de["convergence"]
        assert de["log_likelihood"] >= lbfgsb["log_likelihood"] - 1e-3
        assert de["parameters"]["sigma"] == pytest.approx(0.5, abs=0.05)
        assert de["parameters"]["jump_scale"] == pytest.approx(0.2, abs=0.05)

    def test_de_method_aliases_and_explicit_bounds(self):
        """method="DE" works, and an explicit bounds= box is honored."""
        rng = np.random.default_rng(0)
        increments = rng.normal(0.001, 0.02, 300)
        dt = 1 / 252

        bounds = [
            (-1.0, 1.0),  # mu
            (0.01, 2.0),  # sigma
            (1e-6, 0.5),  # jump_prob
            (0.001, 0.5),  # jump_scale
            (-5.0, 5.0),  # jump_skew
        ]
        result = JumpDiffusionEstimator(increments, dt).estimate(
            method="DE", bounds=bounds, seed=1, maxiter=30
        )

        params = result["parameters"]
        for (low, high), name in zip(bounds, params):
            assert low <= params[name] <= high
        assert np.isfinite(result["log_likelihood"])

    def test_finite_bounds_are_all_finite_and_respect_existing(self):
        """
        _finite_bounds must close every open end (DE needs a finite box)
        while leaving already-finite bounds untouched.
        """
        increments, dt = _simulated_sged_increments()
        estimator = JumpDiffusionEstimator(increments, dt, jump_distribution=SGEDJump())

        finite = estimator._finite_bounds()
        model_bounds = estimator._model.get_parameter_bounds()

        assert len(finite) == len(model_bounds)
        for (low, high), (orig_low, orig_high) in zip(finite, model_bounds):
            assert np.isfinite(low) and np.isfinite(high)
            assert low < high
            if orig_low is not None:
                assert low == orig_low
            if orig_high is not None:
                assert high == orig_high

    def test_initial_guess_seeds_de_population(self):
        """Passing initial_guess to DE must be accepted (seeds one member)."""
        rng = np.random.default_rng(3)
        increments = rng.normal(0.001, 0.02, 300)
        dt = 1 / 252

        estimator = JumpDiffusionEstimator(increments, dt)
        guess = estimator._get_initial_guess()
        result = estimator.estimate(
            method="differential_evolution", initial_guess=guess, seed=1, maxiter=20
        )
        assert np.isfinite(result["log_likelihood"])

    def test_log_likelihood_penalizes_non_finite_values(self, monkeypatch):
        """
        Candidate parameters that make the likelihood non-numeric must map
        to +inf so population-based optimizers discard them (the thesis's
        large-constant penalty device).
        """
        rng = np.random.default_rng(0)
        increments = rng.normal(0.001, 0.02, 300)
        estimator = JumpDiffusionEstimator(increments, 1 / 252)

        monkeypatch.setattr(
            type(estimator._model),
            "log_likelihood",
            lambda self, data, dt: float("nan"),
        )
        params = np.array([0.05, 0.2, 0.1, 0.15, 1.0])
        assert estimator.log_likelihood(params) == np.inf
