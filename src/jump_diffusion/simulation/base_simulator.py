"""
Base simulator class for stochastic processes.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple
import numpy as np

class BaseSimulator(ABC):
    """
    Abstract base class for all simulators.
    """

    def __init__(self, **kwargs):
        """Initialize simulator with model parameters."""
        self.parameters = kwargs

    @abstractmethod
    def simulate_path(self, T: float, n_steps: int, x0: float = 1.0,
                      seed: Optional[int] = None) -> Tuple[np.ndarray, ...]:
        """
        Simulate a single path.

        Parameters:
        -----------
        T : float
            Time horizon
        n_steps : int
            Number of time steps
        x0 : float
            Initial value
        seed : int, optional
            Random seed

        Returns:
        --------
        tuple
            Simulation results
        """
        pass

    @abstractmethod
    def plot_simulation(self, *args, **kwargs):
        """Plot simulation results."""
        pass
