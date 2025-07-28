"""
Model diagnostics and validation tools.

This module provides diagnostic tools for validating jump-diffusion models
and their parameter estimates.
"""

import numpy as np
import matplotlib.pyplot as plt

# from typing import Dict, Any, Optional
from typing import Dict, List, Union
from scipy import stats


class ModelDiagnostics:
    """
    Diagnostic tools for jump-diffusion models.

    This class provides various diagnostic tests and visualizations
    to assess model fit and parameter estimation quality.
    """

    def __init__(
        self,
        data: np.ndarray,
        estimated_params: Dict[str, float],
        dt: float,
    ):
        """
        Initialize diagnostics with data and estimated parameters.

        Parameters:
        -----------
        data : np.ndarray
            Observed increments or path
        estimated_params : dict
            Estimated model parameters
        dt : float
            Time step size
        """
        self.data = data if data.ndim == 1 else np.diff(data)
        self.params = estimated_params
        self.dt = dt
        self.n_obs = len(self.data)

    def ljung_box_test(self, lags: int = 10) -> Dict[str, Union[float, int, bool]]:
        """
        Ljung-Box test for autocorrelation in residuals.

        Parameters:
        -----------
        lags : int
            Number of lags to test

        Returns:
        --------
        dict
            Test statistic and p-value
        """
        # Simple implementation of Ljung-Box test
        residuals = self.data - np.mean(self.data)
        n = len(residuals)

        # Calculate autocorrelations
        autocorrs: List[float] = []
        for lag in range(1, lags + 1):
            if lag < n:
                autocorr = np.corrcoef(residuals[:-lag], residuals[lag:])[0, 1]
                autocorrs.append(autocorr if not np.isnan(autocorr) else 0)
            else:
                autocorrs.append(0)

        # Ljung-Box statistic
        autocorr_array = np.array(autocorrs)
        denom = np.arange(n - 1, n - lags - 1, -1)
        lb_stat = n * (n + 2) * np.sum(autocorr_array**2 / denom)

        # Approximate p-value (chi-square distribution)
        p_value = 1 - stats.chi2.cdf(lb_stat, lags)

        return {
            "statistic": lb_stat,
            "p_value": p_value,
            "lags": lags,
            "reject_independence": p_value < 0.05,
        }

    def normality_test(self) -> Dict[str, Union[float, bool]]:
        """
        Test normality of standardized residuals.

        Returns:
        --------
        dict
            Shapiro-Wilk test results
        """
        # Standardize residuals
        std_residuals = (self.data - np.mean(self.data)) / np.std(self.data)

        # Shapiro-Wilk test
        statistic, p_value = stats.shapiro(std_residuals)

        return {
            "statistic": statistic,
            "p_value": p_value,
            "reject_normality": p_value < 0.05,
        }

    def plot_diagnostics(self, figsize: tuple = (12, 8)) -> None:
        """
        Create diagnostic plots.
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # Time series plot
        axes[0, 0].plot(self.data)
        axes[0, 0].set_title("Time Series of Increments")
        axes[0, 0].set_xlabel("Time")
        axes[0, 0].set_ylabel("Increment")
        axes[0, 0].grid(True, alpha=0.3)

        # Histogram
        axes[0, 1].hist(self.data, bins=30, density=True, alpha=0.7)
        axes[0, 1].set_title("Distribution of Increments")
        axes[0, 1].set_xlabel("Increment")
        axes[0, 1].set_ylabel("Density")
        axes[0, 1].grid(True, alpha=0.3)

        # Q-Q plot
        stats.probplot(self.data, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title("Q-Q Plot vs Normal")
        axes[1, 0].grid(True, alpha=0.3)

        # Autocorrelation
        from statsmodels.tsa.stattools import acf

        try:
            autocorr = acf(self.data, nlags=20, fft=True)
            axes[1, 1].plot(autocorr)
            axes[1, 1].axhline(y=0, color="k", linestyle="-", alpha=0.5)
            axes[1, 1].set_title("Autocorrelation Function")
            axes[1, 1].set_xlabel("Lag")
            axes[1, 1].set_ylabel("Autocorrelation")
            axes[1, 1].grid(True, alpha=0.3)
        except ImportError:
            # Fallback if statsmodels not available
            axes[1, 1].text(
                0.5,
                0.5,
                "Install statsmodels for ACF plot",
                ha="center",
                va="center",
                transform=axes[1, 1].transAxes,
            )

        plt.tight_layout()
        plt.show()

    def summary_report(self) -> None:
        """
        Print a comprehensive diagnostic report.
        """
        print("=" * 60)
        print("MODEL DIAGNOSTIC REPORT")
        print("=" * 60)

        # Basic statistics
        print(f"Sample size: {self.n_obs}")
        print(f"Mean increment: {np.mean(self.data):.6f}")
        print(f"Std increment: {np.std(self.data):.6f}")
        print(f"Skewness: {stats.skew(self.data):.4f}")
        print(f"Kurtosis: {stats.kurtosis(self.data):.4f}")

        # Ljung-Box test
        lb_result = self.ljung_box_test()
        print("\nLjung-Box Test (Independence):")
        print(f"  Statistic: {lb_result['statistic']:.4f}")
        print(f"  P-value: {lb_result['p_value']:.4f}")
        print(f"  Reject independence: {lb_result['reject_independence']}")

        # Normality test
        norm_result = self.normality_test()
        print("\nShapiro-Wilk Test (Normality):")
        print(f"  Statistic: {norm_result['statistic']:.4f}")
        print(f"  P-value: {norm_result['p_value']:.4f}")
        print(f"  Reject normality: {norm_result['reject_normality']}")
