"""
Jump-Diffusion Parameter Estimation Library

A comprehensive Python library for simulating and estimating parameters
of jump-diffusion processes with asymmetric jump distributions.
"""

__version__ = "0.1.0"
__author__ = "Juan David OSPINA ARANGO"
__email__ = "jdospina@gmail.com"

# Core imports for easy access
from .simulation import JumpDiffusionSimulator
from .estimation import JumpDiffusionEstimator
from .models import JumpDiffusionModel
from .validation import ValidationExperiment

# Make key classes available at package level
__all__ = [
    "JumpDiffusionSimulator",
    "JumpDiffusionEstimator",
    "JumpDiffusionModel",
    "ValidationExperiment",
]


# Version information
def get_version():
    """Return the current version of the library."""
    return __version__


def get_info():
    """Return basic information about the library."""
    return {
        "name": "jump-diffusion-estimation",
        "version": __version__,
        "author": __author__,
        "description": "Jump-Diffusion Parameter Estimation Library",
    }
