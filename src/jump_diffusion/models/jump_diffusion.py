"""
Jump-Diffusion Model Implementation

This module implements the jump-diffusion model with a pluggable jump-size
distribution.
"""

import numpy as np
from scipy.stats import norm
from typing import Dict, Optional, Tuple
from .base_model import BaseStochasticModel
from ..distributions import JumpDistribution, SkewNormalJump


class JumpDiffusionModel(BaseStochasticModel):
    """
    Jump-diffusion model with a pluggable jump-size distribution.

    The model follows the SDE:
    dX_t = μ dt + σ dW_t + J_t dN_t

    Where J_t follows the distribution given by ``jump_distribution``
    (skew-normal by default) and N_t is a Poisson process approximated by
    Bernoulli trials.
    """

    def __init__(
        self,
        mu: float = 0.05,
        sigma: float = 0.2,
        jump_prob: float = 0.1,
        jump_distribution: Optional[JumpDistribution] = None,
        **jump_params: float,
    ):
        r"""
        Initialize the jump-diffusion model.

        Parameters:
        -----------
        mu : float
            Drift parameter
        sigma : float
            Diffusion volatility
        jump_prob : float
            Probability of jump per time step
        jump_distribution : JumpDistribution, optional
            Distribution used for the jump sizes. Defaults to
            :class:`SkewNormalJump`.
        \*\*jump_params : float
            Values for ``jump_distribution.param_names``. Any name not
            supplied falls back to ``jump_distribution.default_params()``.
        """
        self.jump_distribution = jump_distribution or SkewNormalJump()

        resolved_jump_params = self.jump_distribution.default_params()
        resolved_jump_params.update(jump_params)

        super().__init__(
            mu=mu,
            sigma=sigma,
            jump_prob=jump_prob,
            **resolved_jump_params,
        )

    def _jump_params(self) -> Dict[str, float]:
        """Extract this model's jump-distribution parameter values."""
        return {
            name: self.parameters[name] for name in self.jump_distribution.param_names
        }

    def simulate(
        self,
        T: float,
        n_steps: int,
        x0: float = 1.0,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate a jump-diffusion path.

        Returns:
        --------
        tuple
            (times, path, jumps) where jumps contains the jump component
        """
        if seed is not None:
            np.random.seed(seed)

        dt = T / n_steps
        times = np.linspace(0, T, n_steps + 1)
        path = np.zeros(n_steps + 1)
        path[0] = x0

        # Generate innovations
        brownian_innovations = np.random.normal(0, 1, n_steps)
        jump_indicators = np.random.binomial(
            1,
            self.parameters["jump_prob"],
            n_steps,
        )
        jump_sizes = self.jump_distribution.rvs(self._jump_params(), size=n_steps)

        # Construct path
        for i in range(n_steps):
            drift = self.parameters["mu"] * dt
            sigma_term = self.parameters["sigma"] * np.sqrt(dt)
            diffusion = sigma_term * brownian_innovations[i]
            jump = jump_indicators[i] * jump_sizes[i]

            path[i + 1] = path[i] + drift + diffusion + jump

        jump_components = jump_indicators * jump_sizes

        return times, path, jump_components

    def log_likelihood(self, data: np.ndarray, dt: float) -> float:
        """
        Calculate log-likelihood of observed increments.
        """
        increments = data if data.ndim == 1 else np.diff(data)

        # Calculate mixture densities
        densities = self._mixture_density(increments, dt)
        densities = np.maximum(densities, 1e-300)  # Numerical stability

        return np.sum(np.log(densities))

    def _mixture_density(self, x: np.ndarray, dt: float) -> np.ndarray:
        """
        Calculate mixture density for each increment.
        """
        mu = self.parameters["mu"]
        sigma = self.parameters["sigma"]
        p = self.parameters["jump_prob"]

        # Pure diffusion component
        diffusion_mean = mu * dt
        diffusion_std = sigma * np.sqrt(dt)
        diffusion_density = norm.pdf(
            x,
            loc=diffusion_mean,
            scale=diffusion_std,
        )

        # Jump + diffusion component: use a closed form if the jump
        # distribution has one, otherwise fall back to FFT convolution.
        jump_params = self._jump_params()
        jump_diffusion_density = self.jump_distribution.diffusion_convolved_pdf(
            x, jump_params, diffusion_mean, diffusion_std
        )
        if jump_diffusion_density is None:
            jump_diffusion_density = self.jump_distribution.fft_convolved_pdf(
                x, jump_params, diffusion_mean, diffusion_std
            )

        # Mixture
        return (1 - p) * diffusion_density + p * jump_diffusion_density

    def get_parameter_bounds(self) -> list:
        """Get parameter bounds for optimization."""
        bounds = [
            (None, None),  # mu
            (1e-6, None),  # sigma > 0
            (1e-6, 1 - 1e-6),  # 0 < jump_prob < 1
        ]
        jump_bounds = self.jump_distribution.param_bounds()
        bounds += [jump_bounds[name] for name in self.jump_distribution.param_names]
        return bounds
