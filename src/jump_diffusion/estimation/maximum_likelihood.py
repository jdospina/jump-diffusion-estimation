"""
Maximum Likelihood Estimator for Jump-Diffusion Models

This module contains our original JumpDiffusionLikelihood class,
refactored as JumpDiffusionEstimator to fit the new architecture.
"""

import numpy as np
import warnings
from scipy import stats
from scipy.stats import norm, skewnorm
from scipy.optimize import minimize
from typing import Dict, Any, Optional, List
from .base_estimator import BaseEstimator


class JumpDiffusionEstimator(BaseEstimator):
    """
    Maximum likelihood estimator for jump-diffusion models.

    This estimator handles jump-diffusion processes with asymmetric
    jump distributions using mixture likelihood functions. The input
    ``data`` must be a one-dimensional array of increments.
    """

    def __init__(self, data: np.ndarray, dt: float):
        """
        Initialize the estimator.

        Parameters:
        -----------
        data : np.ndarray
            One-dimensional array of observed increments. If you have
            a path, compute ``np.diff(path)`` first.
        dt : float
            Time step size between consecutive increments.
        """
        # Accept only 1D arrays of increments
        if data.ndim == 1:
            self.increments = data
        else:
            msg = "data must be a one-dimensional array of increments"
            raise ValueError(msg)

        super().__init__(self.increments, dt)

        # Calculate basic statistics
        self.n_obs = len(self.increments)
        self.mean_increment = np.mean(self.increments)
        self.std_increment = np.std(self.increments)
        self.skewness = stats.skew(self.increments)
        self.kurtosis = stats.kurtosis(self.increments)

    def _diffusion_density(
        self,
        x: np.ndarray,
        mu: float,
        sigma: float,
    ) -> np.ndarray:
        """Calculate pure diffusion density."""
        mean = mu * self.dt
        std = sigma * np.sqrt(self.dt)
        return norm.pdf(x, loc=mean, scale=std)

    def _jump_diffusion_density(
        self,
        x: np.ndarray,
        mu: float,
        sigma: float,
        jump_scale: float,
        jump_skew: float,
    ) -> np.ndarray:
        """Calculate jump + diffusion density."""
        combined_mean = mu * self.dt
        combined_var = sigma**2 * self.dt + jump_scale**2
        combined_std = np.sqrt(combined_var)
        adjusted_skew = jump_skew * jump_scale / combined_std

        return skewnorm.pdf(
            x,
            a=adjusted_skew,
            loc=combined_mean,
            scale=combined_std,
        )

    def _mixture_density(
        self,
        x: np.ndarray,
        mu: float,
        sigma: float,
        jump_prob: float,
        jump_scale: float,
        jump_skew: float,
    ) -> np.ndarray:
        """Calculate mixture density."""
        diffusion_component = (1 - jump_prob) * self._diffusion_density(
            x,
            mu,
            sigma,
        )
        jump_component = jump_prob * self._jump_diffusion_density(
            x, mu, sigma, jump_scale, jump_skew
        )
        return diffusion_component + jump_component

    def log_likelihood(self, params: np.ndarray) -> float:
        """
        Calculate negative log-likelihood for optimization.

        Parameters:
        -----------
        params : np.ndarray
            Parameter vector [mu, sigma, jump_prob, jump_scale, jump_skew]

        Returns:
        --------
        float
            Negative log-likelihood value
        """
        mu, sigma, jump_prob, jump_scale, jump_skew = params

        # Parameter constraints
        if sigma <= 0 or jump_scale <= 0:
            return np.inf
        if jump_prob < 0 or jump_prob > 1:
            return np.inf

        # Calculate densities
        densities = self._mixture_density(
            self.increments, mu, sigma, jump_prob, jump_scale, jump_skew
        )

        # Numerical stability
        densities = np.maximum(densities, 1e-300)

        # Return negative log-likelihood
        return -np.sum(np.log(densities))

    def _get_initial_guess(self) -> List[float]:
        """Generate intelligent initial parameter guess."""
        initial_mu = self.mean_increment / self.dt
        initial_sigma = self.std_increment / np.sqrt(self.dt)
        initial_jump_prob = 0.1
        initial_jump_scale = self.std_increment * 0.5
        initial_jump_skew = float(np.sign(self.skewness)) * 2.0

        return [
            initial_mu,
            initial_sigma,
            initial_jump_prob,
            initial_jump_scale,
            initial_jump_skew,
        ]

    def _get_parameter_bounds(self) -> List[tuple[Optional[float], Optional[float]]]:
        """Get parameter bounds for optimization."""
        return [
            (None, None),  # mu
            (1e-6, None),  # sigma > 0
            (1e-6, 1 - 1e-6),  # 0 < jump_prob < 1
            (1e-6, None),  # jump_scale > 0
            (-10, 10),  # jump_skew
        ]

    def estimate(
        self,
        initial_guess: Optional[List[float]] = None,
        method: str = "L-BFGS-B",
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Estimate parameters using maximum likelihood.

        Parameters:
        -----------
        initial_guess : list, optional
            Initial parameter values
        method : str
            Optimization method
        **kwargs : dict
            Additional arguments for optimizer

        Returns:
        --------
        dict
            Estimation results
        """
        if initial_guess is None:
            initial_guess = self._get_initial_guess()

        bounds = self._get_parameter_bounds()

        # Optimization
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = minimize(
                self.log_likelihood,
                initial_guess,
                method=method,
                bounds=bounds,
                options={"maxiter": 1000, "ftol": 1e-9, **kwargs},
            )

        # Process results
        mu_hat, sigma_hat, p_hat, omega_hat, alpha_hat = result.x

        results: Dict[str, Any] = {
            "parameters": {
                "mu": mu_hat,
                "sigma": sigma_hat,
                "jump_prob": p_hat,
                "jump_scale": omega_hat,
                "jump_skew": alpha_hat,
            },
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

    def diagnostics(self) -> None:
        """Print diagnostic information about the estimation."""
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
        print(f"Jump scale (ω):         {params['jump_scale']:.6f}")
        print(f"Jump skewness (α):      {params['jump_skew']:.6f}")
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
        theoretical_var = (
            params["sigma"] ** 2 * self.dt
            + params["jump_prob"] * params["jump_scale"] ** 2
        )

        print("Mean increment:")
        print(f"  Empirical:  {self.mean_increment:.6f}")
        print(f"  Theoretical: {theoretical_mean:.6f}")
        print("Std deviation:")
        print(f"  Empirical:  {self.std_increment:.6f}")
        print(f"  Theoretical: {np.sqrt(theoretical_var):.6f}")
        print(f"Expected jumps: {params['jump_prob'] * self.n_obs:.1f}")
