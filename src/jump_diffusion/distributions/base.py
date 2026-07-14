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

    def characteristic_scale(self, params: Dict[str, float]) -> float:
        """
        Rough scale of the jump size, used to size the FFT/inverse-CDF grids
        in :meth:`fft_convolved_pdf` and :meth:`rvs`.

        Default reads ``jump_scale`` directly, the convention used by
        ``NormalJump``/``SkewNormalJump``/``SGEDJump``. Override this for
        distributions with a different parameterization (e.g. ``KouJump``,
        which has separate up/down scales instead of a single one).
        """
        return params.get("jump_scale", 1.0)

    def characteristic_location(self, params: Dict[str, float]) -> float:
        """
        Rough center of the jump-size distribution, used to place the
        FFT/inverse-CDF grids in :meth:`fft_convolved_pdf` and :meth:`rvs`
        so that a shifted distribution (e.g. ``StudentTJump``/``SGEDJump``
        with a large ``jump_loc``) does not fall outside a grid centered at
        the origin.

        Default reads ``jump_loc`` (0 when absent). Distributions whose
        center is not a ``jump_loc`` parameter but stays within a few
        characteristic scales of zero (e.g. ``KouJump``) are already covered
        by the grids' tail margins and need no override.
        """
        return params.get("jump_loc", 0.0)

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

        Two guards keep the discretization honest across the whole
        parameter space an optimizer may probe:

        * When ``h`` is not given, the step is chosen for resolution
          (1/200th of the narrowest density) but then *coarsened* if the
          resulting span ``2**j * h`` could not contain the densities'
          mass -- e.g. a jump scale much larger than the diffusion scale,
          or a jump location far from the origin. A slightly coarser grid
          that preserves total mass beats a fine grid that silently
          truncates it.
        * Points of ``x`` outside the grid evaluate to 0 (the density is
          genuinely negligible there once the span covers the mass); they
          no longer zero out the *whole* result, which previously created
          an artificial likelihood cliff whenever a single extreme
          observation fell outside the grid of a small-scale candidate.
        """
        x = np.asarray(x, dtype=float)
        jump_std = self.characteristic_scale(params)
        if jump_std is None or jump_std <= 0:
            return np.zeros_like(x)
        jump_loc = self.characteristic_location(params)

        m = 2**j
        if h is not None:
            step = h
        else:
            # Step chosen for resolution, then coarsened if the span could
            # not contain the mass of both densities (plus tail margins).
            step = min(diffusion_std, jump_std) / 200.0
            half_span_needed = (
                abs(diffusion_mean) + abs(jump_loc) + 10.0 * (diffusion_std + jump_std)
            )
            if m * step < half_span_needed:
                step = half_span_needed / m

        k = np.arange(-m + 0.5, m, 1.0)  # length 2*m, half-integer offsets
        x_grid = k * step

        f_diffusion = norm.pdf(x_grid, loc=diffusion_mean, scale=diffusion_std)
        f_jump = self.pdf(x_grid, params)
        if not (np.all(np.isfinite(f_diffusion)) and np.all(np.isfinite(f_jump))):
            return np.zeros_like(x)

        # numpy's ifft already divides by n, unlike R's fft(..., inverse=TRUE)
        conv = np.real(np.fft.ifft(np.fft.fft(f_diffusion) * np.fft.fft(f_jump)))
        conv *= step
        conv = np.concatenate([conv[m:], conv[:m]])  # re-center around x_grid
        conv = np.maximum(conv, 0.0)

        return np.interp(x, x_grid, conv, left=0.0, right=0.0)

    def _moment_grid(
        self, params: Dict[str, float]
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Shared discretization of ``pdf`` used by :meth:`rvs`, :meth:`mean`
        and :meth:`variance`: ``2**13`` points spanning ~20 characteristic
        scales around the characteristic location.

        Returns ``(x_grid, density, step)``.
        """
        jump_std = self.characteristic_scale(params)
        h = jump_std / 200.0
        m = 2**12  # a coarser grid than fft_convolved_pdf suffices here
        k = np.arange(-m + 0.5, m, 1.0)
        # Centered on the distribution's location so that a shifted
        # distribution (large jump_loc) does not fall off the grid.
        x_grid = self.characteristic_location(params) + k * h
        density = np.maximum(self.pdf(x_grid, params), 0.0)
        return x_grid, density, h

    def mean(self, params: Dict[str, float]) -> float:
        """
        Mean jump size ``E[J]``, evaluated numerically on the sampling grid
        (see :meth:`_moment_grid`). Accurate to the grid's truncation
        (~20 characteristic scales), which is ample for reporting purposes;
        heavy-tailed distributions lose a small tail contribution.

        Returns ``nan`` if the density has no mass on the grid.
        """
        x_grid, density, h = self._moment_grid(params)
        total = float(np.sum(density) * h)
        if total <= 0:
            return float("nan")
        return float(np.sum(x_grid * density) * h / total)

    def variance(self, params: Dict[str, float]) -> float:
        """
        Jump-size variance ``Var[J]``, evaluated numerically on the sampling
        grid (see :meth:`_moment_grid`, same truncation caveat as
        :meth:`mean`).

        Returns ``nan`` if the density has no mass on the grid.
        """
        x_grid, density, h = self._moment_grid(params)
        total = float(np.sum(density) * h)
        if total <= 0:
            return float("nan")
        mean = np.sum(x_grid * density) * h / total
        return float(np.sum((x_grid - mean) ** 2 * density) * h / total)

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
        x_grid, density, h = self._moment_grid(params)
        cdf = np.cumsum(density) * h
        cdf /= cdf[-1]
        rand = random_state if random_state is not None else np.random
        u = rand.uniform(0.0, 1.0, size=size)
        return np.interp(u, cdf, x_grid)
