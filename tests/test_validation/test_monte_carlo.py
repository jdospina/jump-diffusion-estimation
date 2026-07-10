"""Unit tests for the ValidationExperiment class."""

from jump_diffusion.distributions import NormalJump, KouJump
from jump_diffusion.validation import ValidationExperiment


def test_validation_experiment_runs_with_default_skew_normal():
    # Use very few steps and simulations to keep the test fast
    true_params = {
        "mu": 0.05,
        "sigma": 0.2,
        "jump_prob": 0.1,
        "jump_scale": 0.15,
        "jump_skew": 1.0,
    }
    # SkewNormalJump is default
    experiment = ValidationExperiment(true_params)
    results = experiment.run_experiment(
        n_simulations=2, T=0.1, n_steps=20, seed_base=42
    )

    assert len(results) > 0
    assert "mu_est" in results.columns
    assert "jump_skew_est" in results.columns

    analysis = experiment.analyze_results()
    assert "mu" in analysis
    assert "jump_skew" in analysis


def test_validation_experiment_runs_with_normal():
    true_params = {
        "mu": 0.05,
        "sigma": 0.2,
        "jump_prob": 0.1,
        "jump_scale": 0.15,
    }
    experiment = ValidationExperiment(true_params, jump_distribution=NormalJump())
    results = experiment.run_experiment(
        n_simulations=2, T=0.1, n_steps=20, seed_base=42
    )

    assert len(results) > 0
    assert "mu_est" in results.columns
    assert "jump_scale_est" in results.columns
    assert "jump_skew_est" not in results.columns

    analysis = experiment.analyze_results()
    assert "mu" in analysis
    assert "jump_scale" in analysis
    assert "jump_skew" not in analysis


def test_validation_experiment_runs_with_kou():
    true_params = {
        "mu": 0.05,
        "sigma": 0.2,
        "jump_prob": 0.1,
        "jump_prob_up": 0.5,
        "jump_scale_up": 0.1,
        "jump_scale_down": 0.1,
    }
    experiment = ValidationExperiment(true_params, jump_distribution=KouJump())
    results = experiment.run_experiment(
        n_simulations=2, T=0.1, n_steps=20, seed_base=42
    )

    assert len(results) > 0
    assert "mu_est" in results.columns
    assert "jump_scale_up_est" in results.columns
    assert "jump_scale_down_est" in results.columns

    analysis = experiment.analyze_results()
    assert "mu" in analysis
    assert "jump_scale_up" in analysis
    assert "jump_scale_down" in analysis
