"""
Jump-distribution goodness-of-fit comparison.

This module fits JumpDiffusionEstimator under several candidate jump-size
distributions on the same data and compares how well each one fits, via
information criteria (AIC/BIC) and a Kolmogorov-Smirnov goodness-of-fit
test whose p-value is obtained by parametric bootstrap, so it accounts for
the fact that the parameters were estimated from the same data. The KS
machinery only relies on ``JumpDistribution.rvs`` (through model
simulation), so it works the same way for any current or future jump
distribution.
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

    def _simulate_increments(
        self,
        model: JumpDiffusionModel,
        size: int,
        seed: Optional[int],
    ) -> np.ndarray:
        """Simulate ``size`` increments from ``model`` at this object's dt."""
        _, path, _ = model.simulate(T=size * self.dt, n_steps=size, x0=0.0, seed=seed)
        return np.diff(path)

    def _parametric_bootstrap_ks_pvalue(
        self,
        jump_distribution: JumpDistribution,
        fitted_model: JumpDiffusionModel,
        ks_observed: float,
        n_obs: int,
        reference_size: int,
        n_bootstrap: int,
        seed: Optional[int],
        estimate_kwargs: Dict[str, Any],
    ) -> float:
        """
        Monte Carlo p-value for the KS goodness-of-fit statistic.

        A plain two-sample KS p-value is invalid here because the reference
        distribution was estimated from the very data being tested, which
        makes the naive p-value strongly anti-conservative. The parametric
        bootstrap restores validity: under the fitted (null) model we
        simulate ``n_bootstrap`` datasets of the observed size, **re-fit** the
        model on each, and recompute the KS distance of each dataset to its
        own re-fitted model. The fraction of these bootstrap distances that
        equal or exceed the observed one estimates the p-value, with the
        usual ``(1 + count) / (n_successful + 1)`` small-sample correction.
        Re-fitting inside the loop is what makes the test account for
        estimation error.

        Returns ``nan`` if ``n_bootstrap <= 0`` or no replicate converged.
        """
        if n_bootstrap <= 0:
            return float("nan")

        exceed = 0
        n_successful = 0
        for b in range(n_bootstrap):
            data_seed = None if seed is None else seed + 1000 + b
            boot_data = self._simulate_increments(fitted_model, n_obs, data_seed)

            estimator = JumpDiffusionEstimator(
                boot_data, self.dt, jump_distribution=jump_distribution
            )
            rep_result = estimator.estimate(**estimate_kwargs)
            if not rep_result["convergence"]:
                continue

            model_b = JumpDiffusionModel(
                jump_distribution=jump_distribution, **rep_result["parameters"]
            )
            ref_seed = None if seed is None else seed + 100_000 + b
            reference_b = self._simulate_increments(model_b, reference_size, ref_seed)
            ks_b = float(ks_2samp(boot_data, reference_b).statistic)
            if ks_b >= ks_observed:
                exceed += 1
            n_successful += 1

        if n_successful == 0:
            return float("nan")
        return (1.0 + exceed) / (n_successful + 1.0)

    def fit(
        self,
        name: str,
        jump_distribution: JumpDistribution,
        seed: Optional[int] = None,
        n_bootstrap: int = 199,
        ks_reference_size: Optional[int] = None,
        **estimate_kwargs: Any,
    ) -> Dict[str, Any]:
        r"""
        Fit ``jump_distribution`` to the data and record its goodness of fit.

        The KS statistic is the distance between the data and the fitted
        model's distribution, the latter approximated by a large simulated
        reference sample so that simulation noise in the statistic itself is
        negligible. Its p-value is obtained by parametric bootstrap (see
        :meth:`_parametric_bootstrap_ks_pvalue`), which is the dominant cost
        of this method.

        Parameters:
        -----------
        name : str
            Label used to identify this fit in ``results``/``compare()``.
        jump_distribution : JumpDistribution
            Jump-size distribution to fit.
        seed : int, optional
            Base random seed. When given, the reference sample, the plotting
            sample and every bootstrap replicate use distinct derived seeds,
            so the whole ``fit`` is reproducible.
        n_bootstrap : int
            Number of parametric bootstrap replicates for the KS p-value.
            Each re-fits the model, so cost scales linearly; set to ``0`` to
            skip the p-value (it is then reported as ``nan``).
        ks_reference_size : int, optional
            Size of the reference sample used to evaluate the KS statistic.
            Defaults to ``max(20 * n_obs, 20000)``. Larger values give a
            more stable statistic at higher simulation cost.
        \*\*estimate_kwargs : dict
            Additional arguments forwarded to
            ``JumpDiffusionEstimator.estimate`` (also used for every
            bootstrap re-fit).

        Returns:
        --------
        dict
            The estimation results, extended with ``ks_statistic``,
            ``ks_pvalue`` (parametric bootstrap), ``ks_n_bootstrap`` and
            ``simulated_increments`` (an observation-sized sample kept for
            :meth:`plot_comparison`).
        """
        estimator = JumpDiffusionEstimator(
            self.data, self.dt, jump_distribution=jump_distribution
        )
        result = dict(estimator.estimate(**estimate_kwargs))

        fitted_model = JumpDiffusionModel(
            jump_distribution=jump_distribution, **result["parameters"]
        )
        n_obs = len(self.data)
        reference_size = (
            ks_reference_size
            if ks_reference_size is not None
            else max(20 * n_obs, 20000)
        )

        # Stable KS distance to the fitted model, via a large reference
        # sample so the statistic is not dominated by simulation noise.
        ref_seed = None if seed is None else seed
        reference_increments = self._simulate_increments(
            fitted_model, reference_size, ref_seed
        )
        ks_statistic = float(ks_2samp(self.data, reference_increments).statistic)

        # A same-size sample kept only for the histogram overlay in
        # plot_comparison (not used by the statistic).
        plot_seed = None if seed is None else seed + 1
        simulated_increments = self._simulate_increments(fitted_model, n_obs, plot_seed)

        ks_pvalue = self._parametric_bootstrap_ks_pvalue(
            jump_distribution,
            fitted_model,
            ks_statistic,
            n_obs,
            reference_size,
            n_bootstrap,
            seed,
            estimate_kwargs,
        )

        result["ks_statistic"] = ks_statistic
        result["ks_pvalue"] = ks_pvalue
        result["ks_n_bootstrap"] = n_bootstrap
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
