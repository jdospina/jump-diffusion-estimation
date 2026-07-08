"""
Jump-Diffusion Simulator Implementation

This module contains JumpDiffusionSimulator, which builds on the path
and likelihood math implemented in :class:`JumpDiffusionModel` and adds
state-tracking and visualisation on top of it.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple
from .base_simulator import BaseSimulator
from ..models.jump_diffusion import JumpDiffusionModel


class JumpDiffusionSimulator(BaseSimulator, JumpDiffusionModel):
    """
    Simulator for jump-diffusion processes with asymmetric jumps.

    Inherits path generation from :class:`JumpDiffusionModel` and adds
    state-tracking (for repeated plotting) and visualisation on top.
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
        JumpDiffusionModel.__init__(
            self,
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
        times, path, jump_components = self.simulate(
            T=T, n_steps=n_steps, x0=x0, seed=seed
        )
        jump_times = np.where(jump_components != 0)[0]

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

        Parameters
        ----------
        times : array-like, optional
            Time grid for the plotted path. When ``None``, the time points from
            the most recent call to :meth:`simulate_path` (uniformly spaced
            between 0 and 1) are used.
        path : array-like, optional
            Simulated trajectory to visualise. Defaults to the last simulated
            path stored by the simulator.
        jumps : array-like, optional
            Jump magnitudes for each step. Required to display jump-related
            diagnostics. If ``None``, the jump component from the last
            simulation is used.
        figsize : tuple, optional
            Figure size passed to :func:`matplotlib.pyplot.subplots`.

        Notes
        -----
        The method creates a 2×2 grid of plots showing the trajectory with jump
        markers, a stem plot of jump magnitudes, the distribution of increments,
        and the distribution of jump sizes. If no arrays are provided and a
        simulation has been run previously, these plots display the stored
        results. Supplying ``times``, ``path`` and ``jumps`` overrides the stored
        data and visualises the provided arrays instead.
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
