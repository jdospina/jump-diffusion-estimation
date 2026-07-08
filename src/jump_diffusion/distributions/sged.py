"""
Skewed Generalized Error Distribution (SGED) jump distribution.

Ported from Ospina Arango (2009), "Estimacion de un modelo de difusion con
saltos con distribucion de error generalizada asimetrica usando algoritmos
evolutivos" (Universidad Nacional de Colombia, Escuela de Estadistica). This
is the Fernandez & Steel (1998) skewed extension of the Generalized Error
Distribution (GED), in the parameterization used by the ``fGarch`` R
package: X ~ SGED(loc, scale, nu, xi) has E[X] = loc and Var[X] = scale**2.

Special cases (verified against the closed forms in tests):
    nu=2, xi=1 -> standard normal
    nu=1, xi=1 -> standard Laplace

No closed form is known for the convolution of SGED with an independent
normal, so this distribution relies on the generic FFT-convolution and
inverse-CDF sampling fallbacks defined in ``JumpDistribution``.
"""

from typing import Dict, Optional, Tuple

import numpy as np
from scipy.special import gamma

from .base import JumpDistribution


def _ged_lambda(nu: float) -> float:
    return float(np.sqrt(2 ** (-2 / nu) * gamma(1 / nu) / gamma(3 / nu)))


def _ged_pdf(z: np.ndarray, nu: float) -> np.ndarray:
    """Standardized (mean 0, var 1) Generalized Error Distribution pdf."""
    lam = _ged_lambda(nu)
    return (
        nu
        * np.exp(-0.5 * np.abs(z / lam) ** nu)
        / (lam * 2 ** (1 + 1 / nu) * gamma(1 / nu))
    )


def _fernandez_steel_pdf(z: np.ndarray, nu: float, xi: float) -> np.ndarray:
    """Fernandez & Steel skewed GED, not yet re-centered/re-scaled."""
    sign = np.sign(z)
    return (2.0 / (xi + 1.0 / xi)) * _ged_pdf(np.power(xi, -sign) * z, nu)


def _skew_moments(nu: float, xi: float) -> Tuple[float, float]:
    """Mean and std of the (not yet standardized) skewed GED."""
    lam = _ged_lambda(nu)
    m1 = gamma(2 / nu) * lam * 2 ** (1 / nu) / gamma(1 / nu)
    m2 = gamma(3 / nu) * lam**2 * 2 ** (2 / nu) / gamma(1 / nu)
    mu_xi = (xi - 1 / xi) * m1
    sigma_xi2 = (xi**2 + 1 / xi**2) * (m2 - m1**2) + 2 * m1**2 - m2
    return float(mu_xi), float(np.sqrt(sigma_xi2))


def _standardized_sged_pdf(z: np.ndarray, nu: float, xi: float) -> np.ndarray:
    """Mean-0, var-1 SGED density (eq. densidadsged in the thesis)."""
    mu_xi, sigma_xi = _skew_moments(nu, xi)
    z_xi = sigma_xi * z + mu_xi
    return sigma_xi * _fernandez_steel_pdf(z_xi, nu, xi)


class SGEDJump(JumpDistribution):
    """Jump sizes distributed as a Skewed Generalized Error Distribution."""

    param_names: Tuple[str, ...] = ("jump_loc", "jump_scale", "jump_nu", "jump_xi")

    def default_params(self) -> Dict[str, float]:
        return {
            "jump_loc": 0.0,
            "jump_scale": 0.15,
            "jump_nu": 2.0,
            "jump_xi": 1.0,
        }

    def pdf(self, x: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        loc = params["jump_loc"]
        scale = params["jump_scale"]
        nu = params["jump_nu"]
        xi = params["jump_xi"]
        z = (np.asarray(x, dtype=float) - loc) / scale
        return _standardized_sged_pdf(z, nu, xi) / scale

    def param_bounds(self) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
        return {
            "jump_loc": (None, None),
            "jump_scale": (1e-6, None),
            "jump_nu": (0.1, 30.0),
            "jump_xi": (0.05, 20.0),
        }

    def initial_guess(
        self,
        mean_increment: float,
        std_increment: float,
        skewness: float,
    ) -> Dict[str, float]:
        return {
            "jump_loc": 0.0,
            "jump_scale": std_increment * 0.5,
            "jump_nu": 2.0,
            "jump_xi": 1.0 + float(np.sign(skewness)) * 0.3,
        }
