"""
Base classes for stochastic process models.

This module provides abstract base classes that define the interface
for all stochastic process models in the library.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np


class BaseStochasticModel(ABC):
    """
    Abstract base class for all stochastic process models.

    This class defines the common interface that all models must implement,
    ensuring consistency and extensibility across different model types.
    """

    def __init__(self, **kwargs):
        """Initialize the model with parameters."""
        self.parameters = kwargs
        self.fitted = False

    @abstractmethod
    def simulate(
        self,
        T: float,
        n_steps: int,
        x0: float = 1.0,
        seed: Optional[int] = None,
    ) -> tuple:
        """
        Simulate a path from the stochastic process.

        Parameters:
        -----------
        T : float
            Total time horizon
        n_steps : int
            Number of time steps
        x0 : float
            Initial value
        seed : int, optional
            Random seed for reproducibility

        Returns:
        --------
        tuple
            (times, path) or (times, path, additional_info)
        """
        pass

    @abstractmethod
    def log_likelihood(self, data: np.ndarray, dt: float) -> float:
        """
        Calculate the log-likelihood of observed data.

        Parameters:
        -----------
        data : np.ndarray
            Observed increments
        dt : float
            Time step size

        Returns:
        --------
        float
            Log-likelihood value
        """
        pass

    @abstractmethod
    def get_parameter_bounds(self) -> list:
        """
        Get parameter bounds for optimization.

        Returns:
        --------
        list
            List of (lower, upper) tuples for each parameter
        """
        pass

    def update_parameters(self, **kwargs):
        """Update model parameters."""
        self.parameters.update(kwargs)
        self.fitted = False

    def get_parameters(self) -> Dict[str, Any]:
        """Get current model parameters."""
        return self.parameters.copy()
