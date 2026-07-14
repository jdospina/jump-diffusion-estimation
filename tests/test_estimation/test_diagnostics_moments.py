"""Regression test: diagnostics() must include the jump contribution in
the theoretical mean (audit finding B3)."""

import contextlib
import io

import numpy as np
from jump_diffusion.distributions import SkewNormalJump
from jump_diffusion.estimation import JumpDiffusionEstimator
from jump_diffusion.simulation import JumpDiffusionSimulator


def _parse_theoretical(output: str, section: str) -> float:
    lines = output.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith(section))
    theo_line = next(ln for ln in lines[start:] if ln.strip().startswith("Theoretical"))
    return float(theo_line.split(":")[1])


def test_theoretical_mean_includes_jump_contribution():
    # Strongly skewed jumps: E[J] > 0, so mu*dt alone is badly off.
    sim = JumpDiffusionSimulator(
        mu=0.05,
        sigma=0.2,
        jump_prob=0.1,
        jump_distribution=SkewNormalJump(),
        jump_scale=0.15,
        jump_skew=3.0,
    )
    times, path, _ = sim.simulate_path(T=2000 / 252, n_steps=2000, x0=0.0, seed=3)
    increments = np.diff(path)
    dt = times[1] - times[0]

    estimator = JumpDiffusionEstimator(
        increments, dt, jump_distribution=SkewNormalJump()
    )
    result = estimator.estimate()

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        estimator.diagnostics()
    printed = _parse_theoretical(buffer.getvalue(), "Mean increment:")

    params = result["parameters"]
    jump_params = {
        name: params[name] for name in estimator._model.jump_distribution.param_names
    }
    expected = params["mu"] * dt + params["jump_prob"] * SkewNormalJump().mean(
        jump_params
    )

    # Self-consistency with the model's exact mixture mean (print precision).
    assert abs(printed - expected) < 1e-6
    # And the jump term must actually matter: mu*dt alone is far away.
    assert abs(printed - params["mu"] * dt) > 5 * abs(params["mu"] * dt)
    # Sanity: the fitted model's mean tracks the empirical one.
    assert abs(printed - np.mean(increments)) < 0.005
