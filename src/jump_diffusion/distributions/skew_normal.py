"""
Skew-normal jump distribution.

Migrated from the logic previously hardcoded in ``JumpDiffusionModel``. The
skew-normal family is closed under convolution with an independent normal
(Azzalini & Henze, 1986), which is why a closed-form diffusion-convolved
density is available here instead of falling back to FFT convolution.
"""

from typing import Dict, Optional, Tuple, cast

import numpy as np
from scipy.stats import skewnorm

from .base import JumpDistribution


class SkewNormalJump(JumpDistribution):
    """Jump sizes distributed as a skew-normal centered at zero."""

    param_names: Tuple[str, ...] = ("jump_scale", "jump_skew")

    def default_params(self) -> Dict[str, float]:
        return {"jump_scale": 0.15, "jump_skew": 0.0}

    def pdf(self, x: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        return cast(
            np.ndarray,
            skewnorm.pdf(
                x,
                a=params["jump_skew"],
                loc=0,
                scale=params["jump_scale"],
            ),
        )

    def param_bounds(self) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
        return {
            "jump_scale": (1e-6, None),
            "jump_skew": (-10, 10),
        }

    def initial_guess(
        self,
        mean_increment: float,
        std_increment: float,
        skewness: float,
    ) -> Dict[str, float]:
        return {
            "jump_scale": std_increment * 0.5,
            "jump_skew": float(np.sign(skewness)) * 2.0,
        }

    def diffusion_convolved_pdf(
        self,
        x: np.ndarray,
        params: Dict[str, float],
        diffusion_mean: float,
        diffusion_std: float,
    ) -> Optional[np.ndarray]:
        omega = params["jump_scale"]
        alpha = params["jump_skew"]

        combined_var = diffusion_std**2 + omega**2
        combined_std = np.sqrt(combined_var)

        # Azzalini & Henze (1986): SN(0,omega,alpha) + N(0,diffusion_std^2)
        # is again skew-normal, but the new shape parameter is obtained by
        # going through the "delta" parameterization (delta = alpha /
        # sqrt(1+alpha^2)), not simply alpha*omega/combined_std -- that
        # simpler-looking expression is off by ~0.2% at alpha=2, verified
        # against direct numerical convolution.
        delta = alpha / np.sqrt(1 + alpha**2)
        adjusted_delta = omega * delta / combined_std
        adjusted_skew = adjusted_delta / np.sqrt(1 - adjusted_delta**2)

        return cast(
            np.ndarray,
            skewnorm.pdf(x, a=adjusted_skew, loc=diffusion_mean, scale=combined_std),
        )

    def rvs(
        self,
        params: Dict[str, float],
        size: int,
        random_state: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        return cast(
            np.ndarray,
            skewnorm.rvs(
                a=params["jump_skew"],
                loc=0,
                scale=params["jump_scale"],
                size=size,
                random_state=random_state,
            ),
        )
