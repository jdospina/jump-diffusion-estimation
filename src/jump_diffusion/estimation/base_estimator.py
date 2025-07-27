"""
Base estimator class for parameter estimation.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np


class BaseEstimator(ABC):
    """
    Abstract base class for all parameter estimators.
    """

    def __init__(self, data: np.ndarray, dt: float):
        """
        Initialize estimator with data.

        Parameters:
        -----------
        data : np.ndarray
            One-dimensional array of observed increments
        dt : float
            Time step size
        """
        self.data = data
        self.dt = dt
        self.fitted = False
        self.results: Optional[Dict[str, Any]] = None

    @abstractmethod
    def estimate(self, **kwargs) -> Dict[str, Any]:
        """
        Estimate model parameters.

        Returns:
        --------
        dict
            Estimation results including parameters and diagnostics
        """
        pass

    @abstractmethod
    def log_likelihood(self, params: np.ndarray) -> float:
        """
        Calculate log-likelihood for given parameters.

        Parameters:
        -----------
        params : np.ndarray
            Parameter values

        Returns:
        --------
        float
            Log-likelihood value
        """
        pass

    def get_results(self) -> Optional[Dict[str, Any]]:
        """Get estimation results if available."""
        return self.results
