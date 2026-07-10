"""
Student-t jump distribution.

Location-scale Student-t, standardized so that ``jump_scale`` is the actual
standard deviation of the jump (matching the convention used elsewhere in
this package -- e.g. ``SGEDJump`` -- rather than scipy's raw ``t`` scale
parameter, whose relationship to the standard deviation depends on the
degrees of freedom). This is the same "standardized Student-t"
parameterization used for Student-t innovations in GARCH-t models
(Bollerslev, 1987): ``jump_df`` alone controls tail fatness, independent of
overall spread.

No closed form is known for the convolution of Student-t with an
independent normal, so this distribution relies on the generic
FFT-convolution and inverse-CDF fallbacks defined in ``JumpDistribution``.
"""

from typing import Dict, Optional, Tuple, cast

import numpy as np
from scipy.stats import t

from .base import JumpDistribution


def _scipy_scale(scale: float, df: float) -> float:
    """Convert the standardized (std-dev) scale to scipy's raw ``t`` scale."""
    return scale * np.sqrt((df - 2) / df)


class StudentTJump(JumpDistribution):
    """Jump sizes distributed as a (standardized) Student-t."""

    param_names: Tuple[str, ...] = ("jump_loc", "jump_scale", "jump_df")

    def default_params(self) -> Dict[str, float]:
        return {"jump_loc": 0.0, "jump_scale": 0.15, "jump_df": 5.0}

    def pdf(self, x: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        df = params["jump_df"]
        return cast(
            np.ndarray,
            t.pdf(
                x,
                df=df,
                loc=params["jump_loc"],
                scale=_scipy_scale(params["jump_scale"], df),
            ),
        )

    def param_bounds(self) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
        return {
            "jump_loc": (None, None),
            "jump_scale": (1e-6, None),
            "jump_df": (2.05, 100.0),
        }

    def initial_guess(
        self,
        mean_increment: float,
        std_increment: float,
        skewness: float,
    ) -> Dict[str, float]:
        return {
            "jump_loc": float(np.sign(skewness)) * std_increment * 0.1,
            "jump_scale": std_increment * 0.5,
            "jump_df": 5.0,
        }

    def rvs(
        self,
        params: Dict[str, float],
        size: int,
        random_state: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        df = params["jump_df"]
        return cast(
            np.ndarray,
            t.rvs(
                df=df,
                loc=params["jump_loc"],
                scale=_scipy_scale(params["jump_scale"], df),
                size=size,
                random_state=random_state,
            ),
        )
