"""
Normal jump distribution (Merton, 1976).

The classic symmetric jump-diffusion model. Nested inside
:class:`~jump_diffusion.distributions.skew_normal.SkewNormalJump` at
``jump_skew=0``, which makes it a natural null model for a
likelihood-ratio test against the skew-normal or SGED jump models.
"""

from typing import Dict, Optional, Tuple, cast

import numpy as np
from scipy.stats import norm

from .base import JumpDistribution


class NormalJump(JumpDistribution):
    """
    Jump sizes distributed as a normal centered at zero.
    
    References:
    - Merton, R. C. (1976). Option pricing when underlying stock returns 
      are discontinuous. Journal of financial economics, 3(1-2), 125-144.
    """

    param_names: Tuple[str, ...] = ("jump_scale",)

    def default_params(self) -> Dict[str, float]:
        return {"jump_scale": 0.15}

    def pdf(self, x: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        return cast(np.ndarray, norm.pdf(x, loc=0, scale=params["jump_scale"]))

    def param_bounds(self) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
        return {"jump_scale": (1e-6, None)}

    def initial_guess(
        self,
        mean_increment: float,
        std_increment: float,
        skewness: float,
    ) -> Dict[str, float]:
        return {"jump_scale": std_increment * 0.5}

    def diffusion_convolved_pdf(
        self,
        x: np.ndarray,
        params: Dict[str, float],
        diffusion_mean: float,
        diffusion_std: float,
    ) -> Optional[np.ndarray]:
        combined_std = np.sqrt(diffusion_std**2 + params["jump_scale"] ** 2)
        return cast(
            np.ndarray,
            norm.pdf(x, loc=diffusion_mean, scale=combined_std),
        )

    def rvs(
        self,
        params: Dict[str, float],
        size: int,
        random_state: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        return cast(
            np.ndarray,
            norm.rvs(
                loc=0,
                scale=params["jump_scale"],
                size=size,
                random_state=random_state,
            ),
        )
