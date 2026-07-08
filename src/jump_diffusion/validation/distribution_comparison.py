"""
Jump-distribution goodness-of-fit comparison.

This module fits JumpDiffusionEstimator under several candidate jump-size
distributions on the same data and compares how well each one fits, via
information criteria (AIC/BIC) and a simulation-based Kolmogorov-Smirnov
test. The KS test only relies on ``JumpDistribution.rvs``, so it works the
same way for any current or future jump distribution.
"""

from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from ..distributions import JumpDistribution
from ..estimation import JumpDiffusionEstimator
from ..models.jump_diffusion import JumpDiffusionModel


class JumpDistributionComparison:
    """
    Fits several jump-size distributions to the same data and compares
    their goodness of fit.
    """

    def __init__(self, data: np.ndarray, dt: float):
        """
        Parameters:
        -----------
        data : np.ndarray
            One-dimensional array of observed increments.
        dt : float
            Time step size between consecutive increments.
        """
        self.data = data
        self.dt = dt
        self.results: Dict[str, Dict[str, Any]] = {}

    def fit(
        self,
        name: str,
        jump_distribution: JumpDistribution,
        seed: Optional[int] = None,
        **estimate_kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Fit ``jump_distribution`` to the data and record its goodness of fit.

        Parameters:
        -----------
        name : str
            Label used to identify this fit in ``results``/``compare()``.
        jump_distribution : JumpDistribution
            Jump-size distribution to fit.
        seed : int, optional
            Random seed for the simulation-based KS test.
        **estimate_kwargs : dict
            Additional arguments forwarded to
            ``JumpDiffusionEstimator.estimate``.

        Returns:
        --------
        dict
            The estimation results, extended with ``ks_statistic``,
            ``ks_pvalue`` and ``simulated_increments``.
        """
        estimator = JumpDiffusionEstimator(
            self.data, self.dt, jump_distribution=jump_distribution
        )
        result = dict(estimator.estimate(**estimate_kwargs))

        fitted_model = JumpDiffusionModel(
            jump_distribution=jump_distribution, **result["parameters"]
        )
        n_obs = len(self.data)
        _, simulated_path, _ = fitted_model.simulate(
            T=n_obs * self.dt, n_steps=n_obs, x0=0.0, seed=seed
        )
        simulated_increments = np.diff(simulated_path)

        ks_statistic, ks_pvalue = ks_2samp(self.data, simulated_increments)
        result["ks_statistic"] = ks_statistic
        result["ks_pvalue"] = ks_pvalue
        result["simulated_increments"] = simulated_increments

        self.results[name] = result
        return result

    def compare(self) -> pd.DataFrame:
        """
        Summarize all fitted distributions, ranked by AIC (best first).
        """
        if not self.results:
            print("No fits yet. Call fit() first.")
            return pd.DataFrame()

        rows = [
            {
                "distribution": name,
                "log_likelihood": result["log_likelihood"],
                "aic": result["aic"],
                "bic": result["bic"],
                "ks_statistic": result["ks_statistic"],
                "ks_pvalue": result["ks_pvalue"],
                "convergence": result["convergence"],
            }
            for name, result in self.results.items()
        ]
        return pd.DataFrame(rows).sort_values("aic").reset_index(drop=True)

    def plot_comparison(
        self,
        bins: int = 50,
        figsize: Tuple[float, float] = (12, 6),
    ) -> None:
        """
        Overlay a histogram of the real data with each fitted
        distribution's simulated sample.
        """
        if not self.results:
            print("No fits yet. Call fit() first.")
            return

        plt.figure(figsize=figsize)
        plt.hist(
            self.data,
            bins=bins,
            density=True,
            alpha=0.35,
            color="black",
            label="Datos reales",
        )
        for name, result in self.results.items():
            plt.hist(
                result["simulated_increments"],
                bins=bins,
                density=True,
                alpha=0.35,
                histtype="step",
                linewidth=2,
                label=f"Simulado ({name})",
            )

        plt.title("Comparación de distribuciones de salto")
        plt.xlabel("Incremento")
        plt.ylabel("Densidad")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
