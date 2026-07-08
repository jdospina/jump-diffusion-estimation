"""
Base interface for pluggable jump-size distributions.

This module defines the ``JumpDistribution`` abstract interface used by
:class:`~jump_diffusion.models.jump_diffusion.JumpDiffusionModel` to plug in
different jump-magnitude distributions. Concrete distributions only need to
implement the probability density (``pdf``) plus a few descriptive methods;
the diffusion-convolved mixture density and random sampling have generic
numerical fallbacks here, so any new distribution works out of the box.
Closed-form overrides remain available for speed where they exist (see
``SkewNormalJump`` and ``NormalJump``).
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.stats import norm


class JumpDistribution(ABC):
    """Abstract interface for a jump-size distribution."""

    param_names: Tuple[str, ...] = ()

    @abstractmethod
    def default_params(self) -> Dict[str, float]:
        """Return sensible default parameter values."""

    @abstractmethod
    def pdf(self, x: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        """Probability density of the jump size at ``x``."""

    @abstractmethod
    def param_bounds(self) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
        """Optimization bounds for each parameter in ``param_names``."""

    @abstractmethod
    def initial_guess(
        self,
        mean_increment: float,
        std_increment: float,
        skewness: float,
    ) -> Dict[str, float]:
        """Heuristic initial parameter guess derived from data moments."""

    def diffusion_convolved_pdf(
        self,
        x: np.ndarray,
        params: Dict[str, float],
        diffusion_mean: float,
        diffusion_std: float,
    ) -> Optional[np.ndarray]:
        """
        Closed-form density of (diffusion + jump), if one is known.

        Returns ``None`` when no closed form is available, signalling
        callers to fall back to :meth:`fft_convolved_pdf`.
        """
        return None

    def fft_convolved_pdf(
        self,
        x: np.ndarray,
        params: Dict[str, float],
        diffusion_mean: float,
        diffusion_std: float,
        j: int = 15,
        h: Optional[float] = None,
    ) -> np.ndarray:
        """
        Numerically approximate the density of (diffusion + jump) via FFT
        convolution.

        Implements the discretization scheme from Ospina Arango (2009),
        "Estimacion de un modelo de difusion con saltos con distribucion de
        error generalizada asimetrica usando algoritmos evolutivos"
        (Universidad Nacional de Colombia): both densities are discretized
        on a symmetric grid of ``2 * 2**j`` half-integer-offset points
        around zero, convolved via FFT, and linearly interpolated to
        evaluate at the requested points.
        """
        x = np.asarray(x, dtype=float)
        jump_std = params.get("jump_scale")
        if jump_std is None or jump_std <= 0:
            return np.zeros_like(x)

        if h is None:
            h = min(diffusion_std, jump_std) / 200.0

        m = 2**j
        k = np.arange(-m + 0.5, m, 1.0)  # length 2*m, half-integer offsets
        x_grid = k * h

        f_diffusion = norm.pdf(x_grid, loc=diffusion_mean, scale=diffusion_std)
        f_jump = self.pdf(x_grid, params)
        if not (np.all(np.isfinite(f_diffusion)) and np.all(np.isfinite(f_jump))):
            return np.zeros_like(x)

        # numpy's ifft already divides by n, unlike R's fft(..., inverse=TRUE)
        conv = np.real(np.fft.ifft(np.fft.fft(f_diffusion) * np.fft.fft(f_jump)))
        conv *= h
        conv = np.concatenate([conv[m:], conv[:m]])  # re-center around x_grid
        conv = np.maximum(conv, 0.0)

        if x.min() < x_grid[0] or x.max() > x_grid[-1]:
            return np.zeros_like(x)

        return np.interp(x, x_grid, conv)

    def rvs(
        self,
        params: Dict[str, float],
        size: int,
        random_state: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Draw random jump sizes via numerical inverse-CDF sampling.

        Generic fallback for distributions without a native fast sampler:
        discretizes ``pdf`` on a grid (in the same spirit as
        :meth:`fft_convolved_pdf`) and inverts its cumulative sum.

        When ``random_state`` is ``None`` (the default), draws from
        NumPy's global legacy random state -- the same one seeded by
        ``JumpDiffusionModel.simulate(seed=...)`` via ``np.random.seed``
        -- so that simulations stay reproducible. Pass an explicit
        ``np.random.Generator`` for an isolated, independent stream.
        """
        jump_std = params.get("jump_scale", 1.0)
        h = jump_std / 200.0
        m = 2**12  # a coarser grid than fft_convolved_pdf suffices here
        k = np.arange(-m + 0.5, m, 1.0)
        x_grid = k * h
        density = np.maximum(self.pdf(x_grid, params), 0.0)
        cdf = np.cumsum(density) * h
        cdf /= cdf[-1]
        rand = random_state if random_state is not None else np.random
        u = rand.uniform(0.0, 1.0, size=size)
        return np.interp(u, cdf, x_grid)
