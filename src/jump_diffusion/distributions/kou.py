"""
Kou (2002) asymmetric double-exponential jump distribution.

Jump sizes are a mixture of one-sided exponentials: with probability
``jump_prob_up`` the jump is positive with mean ``jump_scale_up``, otherwise
it is negative with mean magnitude ``jump_scale_down``. Unlike the
symmetric-around-zero ``NormalJump``/``SkewNormalJump`` parameterizations,
this distribution has no separate loc/skew parameter -- asymmetry comes
directly from ``jump_prob_up`` and the two scales differing.

The convolution of a one-sided exponential with an independent normal is the
well-known exponentially-modified-Gaussian ("ex-Gaussian") density, already
implemented (numerically stably, via ``erfcx``) as ``scipy.stats.exponnorm``.
Kou's double-exponential is just a ``jump_prob_up``-weighted mixture of two
such terms (one direct, one reflected for the downward branch), so the
closed-form diffusion-convolved density below reuses ``exponnorm`` rather
than re-deriving/re-implementing the stability tricks by hand. Verified
against direct numerical convolution in tests.
"""

from typing import Dict, Optional, Tuple, cast

import numpy as np
from scipy.stats import expon, exponnorm

from .base import JumpDistribution


class KouJump(JumpDistribution):
    """Jump sizes distributed as Kou's asymmetric double exponential."""

    param_names: Tuple[str, ...] = ("jump_prob_up", "jump_scale_up", "jump_scale_down")

    def default_params(self) -> Dict[str, float]:
        return {"jump_prob_up": 0.5, "jump_scale_up": 0.1, "jump_scale_down": 0.1}

    def pdf(self, x: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        p = params["jump_prob_up"]
        x = np.asarray(x, dtype=float)
        up = p * expon.pdf(x, scale=params["jump_scale_up"])
        down = (1 - p) * expon.pdf(-x, scale=params["jump_scale_down"])
        return cast(np.ndarray, up + down)

    def param_bounds(self) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
        return {
            "jump_prob_up": (1e-6, 1 - 1e-6),
            "jump_scale_up": (1e-6, None),
            "jump_scale_down": (1e-6, None),
        }

    def characteristic_scale(self, params: Dict[str, float]) -> float:
        return max(params["jump_scale_up"], params["jump_scale_down"])

    def initial_guess(
        self,
        mean_increment: float,
        std_increment: float,
        skewness: float,
    ) -> Dict[str, float]:
        return {
            "jump_prob_up": float(np.clip(0.5 + 0.1 * np.sign(skewness), 0.1, 0.9)),
            "jump_scale_up": std_increment * 0.5,
            "jump_scale_down": std_increment * 0.5,
        }

    def diffusion_convolved_pdf(
        self,
        x: np.ndarray,
        params: Dict[str, float],
        diffusion_mean: float,
        diffusion_std: float,
    ) -> Optional[np.ndarray]:
        p = params["jump_prob_up"]
        x = np.asarray(x, dtype=float)

        up = p * exponnorm.pdf(
            x,
            K=params["jump_scale_up"] / diffusion_std,
            loc=diffusion_mean,
            scale=diffusion_std,
        )
        down = (1 - p) * exponnorm.pdf(
            -x,
            K=params["jump_scale_down"] / diffusion_std,
            loc=-diffusion_mean,
            scale=diffusion_std,
        )
        return cast(np.ndarray, up + down)

    def rvs(
        self,
        params: Dict[str, float],
        size: int,
        random_state: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        rand = random_state if random_state is not None else np.random
        is_up = rand.uniform(0.0, 1.0, size=size) < params["jump_prob_up"]
        up = expon.rvs(
            scale=params["jump_scale_up"], size=size, random_state=random_state
        )
        down = expon.rvs(
            scale=params["jump_scale_down"], size=size, random_state=random_state
        )
        return cast(np.ndarray, np.where(is_up, up, -down))
