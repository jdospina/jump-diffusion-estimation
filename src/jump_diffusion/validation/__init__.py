"""
Validation and Testing Tools

This module provides tools for validating estimation methods
through Monte Carlo experiments and diagnostic tests.
"""

from .monte_carlo import ValidationExperiment

__all__ = [
    "ValidationExperiment",
]
