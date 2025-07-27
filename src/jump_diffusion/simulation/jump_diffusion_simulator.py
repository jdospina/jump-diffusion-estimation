"""
Jump-Diffusion Simulator Implementation

This module contains our original JumpDiffusionSimulator class,
refactored to fit the new architecture.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skewnorm
from typing import Optional, Tuple
from .base_simulator import BaseSimulator


class JumpDiffusionSimulator(BaseSimulator):
    """
    Simulator for jump-diffusion processes with asymmetric jumps.

    This is our original simulator, refactored to inherit from BaseSimulator
    and integrate with the new modular architecture.
    """

    def __init__(
        self,
        mu: float = 0.05,
        sigma: float = 0.2,
        jump_prob: float = 0.1,
        jump_scale: float = 0.3,
        jump_skew: float = 2.0,
    ):
        """
        Initialize the jump-diffusion simulator.

        Parameters:
        -----------
        mu : float
            Drift parameter
        sigma : float
            Diffusion volatility
        jump_prob : float
            Probability of jump per period
        jump_scale : float
            Scale parameter for jump sizes
        jump_skew : float
            Skewness parameter for jumps
        """
        super().__init__(
            mu=mu,
            sigma=sigma,
            jump_prob=jump_prob,
            jump_scale=jump_scale,
            jump_skew=jump_skew,
        )

        # Store last simulation results
        self.last_path: Optional[np.ndarray] = None
        self.last_jumps: Optional[np.ndarray] = None
        self.last_jump_times: Optional[np.ndarray] = None

    def generate_jump_component(
        self, n_steps: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate jump components for simulation.

        Returns:
        --------
        tuple
            (actual_jumps, jump_indicators, jump_times)
        """
        # Bernoulli process for jump timing
        jump_indicators = np.random.binomial(
            1,
            self.parameters["jump_prob"],
            n_steps,
        )

        # Jump magnitudes from skew-normal distribution
        all_jump_sizes = skewnorm.rvs(
            a=self.parameters["jump_skew"],
            loc=0,
            scale=self.parameters["jump_scale"],
            size=n_steps,
        )

        # Actual jumps (zero when no jump occurs)
        actual_jumps = jump_indicators * all_jump_sizes
        jump_times = np.where(jump_indicators == 1)[0]

        return actual_jumps, jump_indicators, jump_times

    def simulate_path(
        self,
        T: float = 1.0,
        n_steps: int = 252,
        x0: float = 1.0,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate a complete jump-diffusion path.

        Parameters:
        -----------
        T : float
            Total simulation time
        n_steps : int
            Number of time steps
        x0 : float
            Initial value
        seed : int, optional
            Random seed for reproducibility

        Returns:
        --------
        tuple
            (times, path, jumps)
        """
        if seed is not None:
            np.random.seed(seed)

        dt = T / n_steps
        times = np.linspace(0, T, n_steps + 1)
        path = np.zeros(n_steps + 1)
        path[0] = x0

        # Generate all random components
        diffusion_innovations = np.random.normal(0, 1, n_steps)
        (
            jump_components,
            jump_indicators,
            jump_times,
        ) = self.generate_jump_component(n_steps)

        # Build path step by step
        for i in range(n_steps):
            drift_term = self.parameters["mu"] * dt
            sigma_dt = self.parameters["sigma"] * np.sqrt(dt)
            diffusion_term = sigma_dt * diffusion_innovations[i]
            jump_term = jump_components[i]

            increment = drift_term + diffusion_term + jump_term
            path[i + 1] = path[i] + increment

        # Store for plotting
        self.last_path = path
        self.last_jumps = jump_components
        self.last_jump_times = jump_times

        return times, path, jump_components

    def plot_simulation(
        self,
        times=None,
        path=None,
        jumps=None,
        figsize=(12, 8),
    ):
        """
        Plot simulation results with comprehensive diagnostics.
        """
        if times is None or path is None:
            if self.last_path is None:
                raise ValueError(
                    "No simulation data available. Run simulate_path() first."
                )
            times = np.linspace(0, 1, len(self.last_path))
            path = self.last_path
            jumps = self.last_jumps

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # Main trajectory plot
        axes[0, 0].plot(times, path, "b-", linewidth=1.5, alpha=0.8)
        if len(self.last_jump_times) > 0:
            axes[0, 0].scatter(
                times[1:][self.last_jump_times],
                path[1:][self.last_jump_times],
                color="red",
                s=30,
                alpha=0.7,
                zorder=5,
            )
        axes[0, 0].set_title("Jump-Diffusion Path")
        axes[0, 0].set_xlabel("Time")
        axes[0, 0].set_ylabel("X(t)")
        axes[0, 0].grid(True, alpha=0.3)

        # Jump magnitudes
        if len(self.last_jump_times) > 0:
            jump_magnitudes = jumps[self.last_jump_times]
            axes[0, 1].stem(self.last_jump_times, jump_magnitudes, basefmt=" ")
            axes[0, 1].set_title(
                f"Detected Jumps (Total: {len(self.last_jump_times)})",
            )
        else:
            axes[0, 1].text(
                0.5,
                0.5,
                "No jumps detected",
                ha="center",
                va="center",
                transform=axes[0, 1].transAxes,
            )
            axes[0, 1].set_title("Detected Jumps (Total: 0)")
        axes[0, 1].set_xlabel("Time Index")
        axes[0, 1].set_ylabel("Jump Magnitude")
        axes[0, 1].grid(True, alpha=0.3)

        # Increment distribution
        increments = np.diff(path)
        axes[1, 0].hist(
            increments,
            bins=50,
            density=True,
            alpha=0.7,
            color="skyblue",
            edgecolor="black",
        )
        axes[1, 0].set_title("Distribution of Increments")
        axes[1, 0].set_xlabel("ΔX")
        axes[1, 0].set_ylabel("Density")
        axes[1, 0].grid(True, alpha=0.3)

        # Jump size distribution
        if len(self.last_jump_times) > 0:
            actual_jumps = jumps[jumps != 0]
            axes[1, 1].hist(
                actual_jumps,
                bins=20,
                density=True,
                alpha=0.7,
                color="orange",
                edgecolor="black",
            )
            axes[1, 1].set_title("Jump Size Distribution")
        else:
            axes[1, 1].text(
                0.5,
                0.5,
                "No jumps to analyze",
                ha="center",
                va="center",
                transform=axes[1, 1].transAxes,
            )
        axes[1, 1].set_xlabel("Jump Size")
        axes[1, 1].set_ylabel("Density")
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
