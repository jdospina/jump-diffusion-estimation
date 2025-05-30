"""
Jump-Diffusion Model Implementation

This module implements the jump-diffusion model with asymmetric jump distributions.
"""

import numpy as np
from scipy.stats import norm, skewnorm
from typing import Optional, Tuple, Dict, Any
from .base_model import BaseStochasticModel

class JumpDiffusionModel(BaseStochasticModel):
    """
    Jump-diffusion model with asymmetric jump distributions.

    The model follows the SDE:
    dX_t = μ dt + σ dW_t + J_t dN_t

    Where J_t follows a skew-normal distribution and N_t is a Poisson process
    approximated by Bernoulli trials.
    """

    def __init__(self, mu: float = 0.05, sigma: float = 0.2,
                 jump_prob: float = 0.1, jump_scale: float = 0.15,
                 jump_skew: float = 0.0):
        """
        Initialize the jump-diffusion model.

        Parameters:
        -----------
        mu : float
            Drift parameter
        sigma : float
            Diffusion volatility
        jump_prob : float
            Probability of jump per time step
        jump_scale : float
            Scale parameter for jump sizes
        jump_skew : float
            Skewness parameter for jump distribution
        """
        super().__init__(
            mu=mu, sigma=sigma, jump_prob=jump_prob,
            jump_scale=jump_scale, jump_skew=jump_skew
        )

    def simulate(self, T: float, n_steps: int, x0: float = 1.0,
                 seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        jump_indicators = np.random.binomial(1, self.parameters['jump_prob'], n_steps)
        jump_sizes = skewnorm.rvs(
            a=self.parameters['jump_skew'],
            loc=0,
            scale=self.parameters['jump_scale'],
            size=n_steps
        )

        # Construct path
        for i in range(n_steps):
            drift = self.parameters['mu'] * dt
            diffusion = self.parameters['sigma'] * np.sqrt(dt) * brownian_innovations[i]
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
        mu = self.parameters['mu']
        sigma = self.parameters['sigma']
        p = self.parameters['jump_prob']
        omega = self.parameters['jump_scale']
        alpha = self.parameters['jump_skew']

        # Pure diffusion component
        diffusion_mean = mu * dt
        diffusion_std = sigma * np.sqrt(dt)
        diffusion_density = norm.pdf(x, loc=diffusion_mean, scale=diffusion_std)

        # Jump + diffusion component
        combined_mean = mu * dt
        combined_var = sigma**2 * dt + omega**2
        combined_std = np.sqrt(combined_var)
        adjusted_skew = alpha * omega / combined_std

        jump_diffusion_density = skewnorm.pdf(
            x, a=adjusted_skew, loc=combined_mean, scale=combined_std
        )

        # Mixture
        return (1 - p) * diffusion_density + p * jump_diffusion_density

    def get_parameter_bounds(self) -> list:
        """Get parameter bounds for optimization."""
        return [
            (None, None),      # mu
            (1e-6, None),      # sigma > 0
            (1e-6, 1-1e-6),    # 0 < jump_prob < 1
            (1e-6, None),      # jump_scale > 0
            (-10, 10)          # jump_skew bounded for stability
        ]
