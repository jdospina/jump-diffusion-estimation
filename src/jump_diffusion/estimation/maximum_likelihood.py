"""
Maximum Likelihood Estimator for Jump-Diffusion Models

This module estimates JumpDiffusionModel parameters by maximizing the
mixture likelihood implemented on the model itself.
"""

import numpy as np
import warnings
from scipy import stats
from scipy.optimize import Bounds, differential_evolution, minimize
from typing import Any, Dict, List, Optional, Tuple
from .base_estimator import BaseEstimator
from ..models.jump_diffusion import JumpDiffusionModel
from ..distributions import JumpDistribution


class JumpDiffusionEstimator(BaseEstimator):
    """
    Maximum likelihood estimator for jump-diffusion models.

    This estimator handles jump-diffusion processes with a pluggable
    jump-size distribution (skew-normal by default) using mixture
    likelihood functions. The input ``data`` must be a one-dimensional
    array of increments.
    """

    def __init__(
        self,
        data: np.ndarray,
        dt: float,
        jump_distribution: Optional[JumpDistribution] = None,
    ):
        """
        Initialize the estimator.

        Parameters:
        -----------
        data : np.ndarray
            One-dimensional array of observed increments. If you have
            a path, compute ``np.diff(path)`` first.
        dt : float
            Time step size between consecutive increments.
        jump_distribution : JumpDistribution, optional
            Distribution assumed for the jump sizes. Defaults to
            :class:`SkewNormalJump`.
        """
        # Accept only 1D arrays of increments
        if data.ndim == 1:
            self.increments = data
        else:
            msg = "data must be a one-dimensional array of increments"
            raise ValueError(msg)

        super().__init__(self.increments, dt)

        # Model used to evaluate the mixture likelihood at trial parameters
        self._model = JumpDiffusionModel(jump_distribution=jump_distribution)
        self._param_names = (
            "mu",
            "sigma",
            "jump_prob",
        ) + self._model.jump_distribution.param_names

        # Calculate basic statistics
        self.n_obs = len(self.increments)
        self.mean_increment = float(np.mean(self.increments))
        self.std_increment = float(np.std(self.increments))
        self.skewness = float(stats.skew(self.increments))
        self.kurtosis = float(stats.kurtosis(self.increments))

    def log_likelihood(self, params: np.ndarray) -> float:
        """
        Calculate negative log-likelihood for optimization.

        Parameters:
        -----------
        params : np.ndarray
            Parameter vector, ordered as ``self._param_names``:
            ``[mu, sigma, jump_prob, *jump_distribution.param_names]``.

        Returns:
        --------
        float
            Negative log-likelihood value
        """
        sigma, jump_prob = params[1], params[2]

        # Parameter constraints shared by every jump distribution
        if sigma <= 0:
            return np.inf
        if jump_prob < 0 or jump_prob > 1:
            return np.inf

        self._model.update_parameters(**dict(zip(self._param_names, params)))
        value = self._model.log_likelihood(self.increments, self.dt)
        # Penalize non-numeric likelihood values so that population-based
        # optimizers (differential evolution) simply discard the offending
        # candidates instead of corrupting the ranking -- same device as in
        # Ospina Arango (2009), where NaN evaluations are mapped to a large
        # positive constant.
        if not np.isfinite(value):
            return np.inf
        return -value

    def _get_initial_guess(self) -> np.ndarray:
        """Generate intelligent initial parameter guess."""
        initial_mu = self.mean_increment / self.dt
        initial_sigma = self.std_increment / np.sqrt(self.dt)
        initial_jump_prob = 0.1

        jump_guess = self._model.jump_distribution.initial_guess(
            self.mean_increment, self.std_increment, self.skewness
        )

        values = [initial_mu, initial_sigma, initial_jump_prob]
        values += [
            jump_guess[name] for name in self._model.jump_distribution.param_names
        ]
        return np.array(values)

    def _finite_bounds(self) -> List[Tuple[float, float]]:
        """
        Data-driven finite search box for differential evolution.

        Differential evolution samples its initial population uniformly
        over a bounded region, so the open-ended bounds used by L-BFGS-B
        (e.g. ``mu`` unbounded, ``sigma < inf``) must be replaced by finite
        ones. This mirrors the :math:`\\theta_L`/:math:`\\theta_U` limits
        in Ospina Arango (2009), whose applied result is that even *very*
        generous boxes (little prior knowledge of the solution) still lead
        differential evolution to the optimum where gradient methods fail.
        Bounds that are already finite are kept as-is; only ``None`` ends
        are replaced with wide data-driven limits.
        """
        initial_mu = self.mean_increment / self.dt
        initial_sigma = self.std_increment / np.sqrt(self.dt)
        mu_halfwidth = max(1.0, 10 * abs(initial_mu), 10 * initial_sigma)
        # Jump parameters live on the scale of individual increments; a
        # single jump 20x the typical increment size is a generous cap
        # (and std_increment is itself inflated by any jumps in the data).
        jump_cap = max(20 * self.std_increment, 1e-3)

        finite = [
            (initial_mu - mu_halfwidth, initial_mu + mu_halfwidth),  # mu
            (1e-6, 10 * initial_sigma),  # sigma
            (1e-6, 1 - 1e-6),  # jump_prob
        ]
        jump_bounds = self._model.jump_distribution.param_bounds()
        for name in self._model.jump_distribution.param_names:
            low, high = jump_bounds[name]
            finite.append(
                (
                    -jump_cap if low is None else low,
                    jump_cap if high is None else high,
                )
            )
        return finite

    def estimate(
        self,
        initial_guess: Optional[np.ndarray] = None,
        method: str = "L-BFGS-B",
        bounds: Optional[List[Tuple[float, float]]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        r"""
        Estimate parameters using maximum likelihood.

        Parameters:
        -----------
        initial_guess : np.array, optional
            Initial parameter values. Optional for every method: gradient
            methods fall back to a moment-based heuristic guess, while
            ``"differential_evolution"`` does not need one at all (its
            initial population is sampled over ``bounds``; when a guess
            *is* supplied it merely seeds one population member).
        method : str
            Optimization method. Any ``scipy.optimize.minimize`` method
            name (default ``"L-BFGS-B"``), or ``"differential_evolution"``
            for the global, population-based optimizer from
            ``scipy.optimize.differential_evolution``. Differential
            evolution is markedly more robust to poor prior knowledge on
            this mixture likelihood -- the applied finding of Ospina
            Arango (2009), where L-BFGS-B stalled at box boundaries and
            simulated annealing diverged, while DE (rand/1, population 70,
            400 generations in R's ``DEoptim``) recovered the true
            parameters even under very wide bounds. The trade-off is
            computational cost (thousands of likelihood evaluations).
        bounds : list of (low, high) tuples, optional
            Optimization box, one pair per parameter in the order
            ``[mu, sigma, jump_prob, *jump_distribution.param_names]``.
            Defaults to the model's own bounds for gradient methods, or a
            wide data-driven finite box (see ``_finite_bounds``) for
            differential evolution, which requires every bound finite.
        \*\*kwargs : dict
            Additional arguments for the optimizer: merged into
            ``options`` for ``scipy.optimize.minimize``, passed directly
            to ``scipy.optimize.differential_evolution`` (e.g. ``seed=42``
            for reproducibility, ``maxiter``, ``popsize``, ``workers``).

        Returns:
        --------
        dict
            Estimation results
        """
        if method.lower() in ("differential_evolution", "de"):
            result = self._optimize_differential_evolution(
                initial_guess, bounds, **kwargs
            )
        else:
            result = self._optimize_minimize(initial_guess, method, bounds, **kwargs)

        # Process results
        parameters = dict(zip(self._param_names, result.x))

        results: Dict[str, Any] = {
            "parameters": parameters,
            "log_likelihood": -result.fun,
            "aic": 2 * len(result.x) + 2 * result.fun,
            "bic": len(result.x) * np.log(self.n_obs) + 2 * result.fun,
            "optimization_result": result,
            "convergence": result.success,
            "data_stats": {
                "n_obs": self.n_obs,
                "mean_increment": self.mean_increment,
                "std_increment": self.std_increment,
                "skewness": self.skewness,
                "kurtosis": self.kurtosis,
            },
        }

        self.results = results
        self.fitted = True
        return results

    def _optimize_minimize(
        self,
        initial_guess: Optional[np.ndarray],
        method: str,
        bounds: Optional[List[Tuple[float, float]]],
        **kwargs,
    ) -> Any:
        """Local optimization via ``scipy.optimize.minimize``."""
        if initial_guess is None:
            initial_guess = self._get_initial_guess()
        initial_guess = np.asarray(initial_guess, dtype=float)

        if bounds is None:
            bounds = self._model.get_parameter_bounds()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return minimize(
                fun=self.log_likelihood,  # type: ignore
                x0=initial_guess,
                method=method,
                bounds=bounds,
                options={"maxiter": 1000, "ftol": 1e-9, **kwargs},
            )

    def _optimize_differential_evolution(
        self,
        initial_guess: Optional[np.ndarray],
        bounds: Optional[List[Tuple[float, float]]],
        **kwargs,
    ) -> Any:
        """
        Global optimization via ``scipy.optimize.differential_evolution``.

        Defaults port the configuration from Ospina Arango (2009), which
        used R's ``DEoptim``: strategy rand/1 with binomial crossover
        (scipy's ``"rand1bin"``) and 400 generations. ``DEoptim`` used a
        fixed population of 70 individuals for the 7-parameter SGED model;
        scipy sizes the population as ``popsize * n_params``, so
        ``popsize=10`` reproduces exactly that for the SGED case and
        scales proportionally for other jump distributions. The thesis
        also observes convergence well before the generation cap and
        suggests a convergence-based stopping criterion -- scipy's ``tol``
        (default 0.01) provides exactly that, so runs typically stop
        early. Any of these can be overridden via ``**kwargs``.
        """
        if bounds is None:
            bounds = self._finite_bounds()
        lows, highs = zip(*bounds)
        de_bounds = Bounds(np.asarray(lows, float), np.asarray(highs, float))

        de_kwargs: Dict[str, Any] = {
            "strategy": "rand1bin",
            "maxiter": 400,
            "popsize": 10,
        }
        de_kwargs.update(kwargs)
        if initial_guess is not None:
            de_kwargs["x0"] = np.asarray(initial_guess, dtype=float)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return differential_evolution(
                func=self.log_likelihood,
                bounds=de_bounds,
                **de_kwargs,
            )

    def diagnostics(self) -> None:
        """Print diagnostic information about the estimation.

        The method writes a summary of the fitted model to ``stdout`` and
        returns ``None``. The following metrics are reported:

        * Estimated parameters (μ, σ, jump probability, and the jump
          distribution's own parameters)
        * Log-likelihood value
        * Information criteria (AIC and BIC)
        * Optimizer convergence flag
        * Comparison of empirical vs. theoretical mean, standard deviation
          and the expected number of jumps
        """
        if not self.fitted or self.results is None:
            print("Model not fitted. Run estimate() first.")
            return

        params = self.results["parameters"]

        print("=" * 60)
        print("JUMP-DIFFUSION ESTIMATION RESULTS")
        print("=" * 60)
        print(f"Drift (μ):              {params['mu']:.6f}")
        print(f"Volatility (σ):         {params['sigma']:.6f}")
        print(f"Jump probability (p):   {params['jump_prob']:.6f}")
        for name in self._model.jump_distribution.param_names:
            print(f"{name:<24}{params[name]:.6f}")
        print(
            f"\nLog-likelihood:         {self.results['log_likelihood']:.2f}",
        )
        print(f"AIC:                    {self.results['aic']:.2f}")
        print(f"BIC:                    {self.results['bic']:.2f}")
        print(f"Convergence:            {self.results['convergence']}")

        # Model comparison with data
        print("\nDATA vs MODEL COMPARISON")
        print("-" * 30)
        theoretical_mean = params["mu"] * self.dt
        # Approximation: treats the distribution's characteristic scale as
        # the jump size's standard deviation, which holds exactly for the
        # Normal jump distribution and approximately for the others.
        jump_scale = self._model.jump_distribution.characteristic_scale(
            {k: params[k] for k in self._model.jump_distribution.param_names}
        )
        theoretical_var = (
            params["sigma"] ** 2 * self.dt + params["jump_prob"] * jump_scale**2
        )

        print("Mean increment:")
        print(f"  Empirical:  {self.mean_increment:.6f}")
        print(f"  Theoretical: {theoretical_mean:.6f}")
        print("Std deviation:")
        print(f"  Empirical:  {self.std_increment:.6f}")
        print(f"  Theoretical: {np.sqrt(theoretical_var):.6f}")
        print(f"Expected jumps: {params['jump_prob'] * self.n_obs:.1f}")
