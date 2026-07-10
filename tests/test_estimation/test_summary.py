"""Unit tests for the public param_names property and summary() table."""

import numpy as np
import pandas as pd
import pytest
from jump_diffusion.distributions import NormalJump
from jump_diffusion.estimation import JumpDiffusionEstimator
from jump_diffusion.simulation import JumpDiffusionSimulator


def _fit_normal_jump(seed: int = 123):
    true_params = {
        "mu": 0.05,
        "sigma": 0.2,
        "jump_prob": 0.1,
        "jump_scale": 0.15,
    }
    sim = JumpDiffusionSimulator(jump_distribution=NormalJump(), **true_params)
    times, path, _ = sim.simulate_path(T=1.0, n_steps=300, x0=100.0, seed=seed)
    increments = np.diff(path)
    dt = times[1] - times[0]

    estimator = JumpDiffusionEstimator(increments, dt, jump_distribution=NormalJump())
    estimator.estimate()
    return estimator


def test_param_names_is_public_and_ordered():
    estimator = _fit_normal_jump()
    assert estimator.param_names == estimator._param_names
    assert estimator.param_names[:3] == ("mu", "sigma", "jump_prob")


def test_summary_before_any_inference_has_only_estimates():
    estimator = _fit_normal_jump()
    table = estimator.summary()

    assert isinstance(table, pd.DataFrame)
    assert list(table["parameter"]) == list(estimator.param_names)
    assert "estimate" in table.columns
    # No inference route has been run, so no SE columns should be present.
    assert not any(col.endswith("_se") for col in table.columns)


def test_summary_collects_all_three_inference_routes():
    estimator = _fit_normal_jump()
    estimator.estimate_standard_errors(n_points=7)
    estimator.estimate_wald_standard_errors()
    estimator.estimate_bootstrap_standard_errors(n_bootstrap=15, seed=0)

    table = estimator.summary()

    for label in ("profile", "wald", "bootstrap"):
        for suffix in ("_se", "_ci_low", "_ci_high"):
            assert f"{label}{suffix}" in table.columns

    # sigma is well identified: finite, positive SE from every route.
    sigma_row = table.set_index("parameter").loc["sigma"]
    for label in ("profile", "wald", "bootstrap"):
        assert np.isfinite(sigma_row[f"{label}_se"])
        assert sigma_row[f"{label}_se"] > 0


def test_summary_requires_fitted_model():
    increments = np.random.default_rng(0).normal(size=100) * 0.1
    estimator = JumpDiffusionEstimator(increments, dt=1.0 / 252)
    with pytest.raises(ValueError):
        estimator.summary()
