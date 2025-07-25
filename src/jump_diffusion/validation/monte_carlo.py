"""
Monte Carlo validation experiments for jump-diffusion estimators.

This module provides tools to validate estimation methods through
controlled experiments with known parameters.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any
from ..simulation import JumpDiffusionSimulator
from ..estimation import JumpDiffusionEstimator


class ValidationExperiment:
    """
    Monte Carlo validation experiment for jump-diffusion estimators.

    This class orchestrates validation experiments where we simulate
    data with known parameters and test how well our estimators
    can recover those parameters.
    """

    def __init__(self, true_params: Dict[str, float]):
        """
        Initialize validation experiment.

        Parameters:
        -----------
        true_params : dict
            True parameter values to use for simulation
        """
        self.true_params = true_params
        self.results = []
        self.completed_experiments = 0

    def run_experiment(
        self,
        n_simulations: int = 10,
        T: float = 1.0,
        n_steps: int = 252,
        x0: float = 100.0,
        seed_base: int = 42,
    ) -> pd.DataFrame:
        """
        Run Monte Carlo validation experiment.

        Parameters:
        -----------
        n_simulations : int
            Number of simulation runs
        T : float
            Time horizon for each simulation
        n_steps : int
            Number of time steps per simulation
        x0 : float
            Initial value
        seed_base : int
            Base seed for reproducibility

        Returns:
        --------
        pd.DataFrame
            Results of all experiments
        """
        print(f"Running {n_simulations} validation experiments...")
        print(f"True parameters: {self.true_params}")

        # Create simulator
        simulator = JumpDiffusionSimulator(**self.true_params)

        results = []
        successful_runs = 0

        for i in range(n_simulations):
            try:
                # Simulate data
                times, path, jumps = simulator.simulate_path(
                    T=T, n_steps=n_steps, x0=x0, seed=seed_base + i
                )

                # Estimate parameters
                increments = np.diff(path)
                dt = times[1] - times[0]

                estimator = JumpDiffusionEstimator(increments, dt)
                est_results = estimator.estimate()

                if est_results["convergence"]:
                    # Store results
                    row = {
                        "simulation_id": i + 1,
                        "seed": seed_base + i,
                        "convergence": True,
                        "log_likelihood": est_results["log_likelihood"],
                        "aic": est_results["aic"],
                        "bic": est_results["bic"],
                    }

                    # Add estimated parameters
                    for param, value in est_results["parameters"].items():
                        row[f"{param}_est"] = value
                        row[f"{param}_true"] = self.true_params[param]
                        row[f"{param}_error"] = value - self.true_params[param]
                        row[f"{param}_rel_error"] = (
                            (value - self.true_params[param])
                            / self.true_params[param]
                            * 100
                        )

                    results.append(row)
                    successful_runs += 1
                    print(
                        f"✓ Experiment {i+1}/{n_simulations} completed "
                        f"successfully",
                    )
                else:
                    message = "✗ Experiment {}/{} failed to converge".format(
                        i + 1,
                        n_simulations,
                    )
                    print(message)

            except Exception as e:
                print(
                    f"✗ Experiment {i+1}/{n_simulations} failed with error: "
                    f"{str(e)}",
                )

        print(
            f"\nCompleted: {successful_runs}/{n_simulations} successful "
            f"experiments",
        )

        if successful_runs == 0:
            print("No successful experiments. Cannot proceed with analysis.")
            return pd.DataFrame()

        self.results = pd.DataFrame(results)
        self.completed_experiments = successful_runs

        return self.results

    def analyze_results(self) -> Dict[str, Any]:
        """
        Analyze validation results and compute statistics.

        Returns:
        --------
        dict
            Analysis results including bias, RMSE, etc.
        """
        if len(self.results) == 0:
            print("No results to analyze. Run experiment first.")
            return {}

        param_names = ["mu", "sigma", "jump_prob", "jump_scale", "jump_skew"]
        analysis = {}

        print("\n" + "=" * 60)
        print("VALIDATION ANALYSIS RESULTS")
        print("=" * 60)

        for param in param_names:
            true_val = self.true_params[param]
            estimated_vals = self.results[f"{param}_est"]
            errors = self.results[f"{param}_error"]
            rel_errors = self.results[f"{param}_rel_error"]

            stats = {
                "true_value": true_val,
                "mean_estimate": np.mean(estimated_vals),
                "bias": np.mean(errors),
                "rmse": np.sqrt(np.mean(errors**2)),
                "mae": np.mean(np.abs(errors)),
                "mean_rel_error": np.mean(rel_errors),
                "std_estimate": np.std(estimated_vals),
                "coverage_95": np.mean(
                    np.abs(rel_errors) <= 5
                ),  # Within 5% of true value
            }

            analysis[param] = stats

            print(f"\n{param.upper()} (true: {true_val:.4f})")
            print(f"  Mean estimate:    {stats['mean_estimate']:.6f}")
            print(f"  Bias:            {stats['bias']:.6f}")
            print(f"  RMSE:            {stats['rmse']:.6f}")
            print(f"  Mean rel. error: {stats['mean_rel_error']:.2f}%")
            print(f"  Std deviation:   {stats['std_estimate']:.6f}")
            print(f"  95% accuracy:    {stats['coverage_95']:.1%}")

        return analysis

    def plot_results(self, figsize: tuple = (15, 10)):
        """
        Create comprehensive plots of validation results.
        """
        if len(self.results) == 0:
            print("No results to plot. Run experiment first.")
            return

        param_names = ["mu", "sigma", "jump_prob", "jump_scale", "jump_skew"]
        param_labels = [
            "Drift (μ)",
            "Volatility (σ)",
            "Jump Prob (p)",
            "Jump Scale (ω)",
            "Skewness (α)",
        ]

        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()

        # Parameter accuracy plots
        for i, (param, label) in enumerate(zip(param_names, param_labels)):
            ax = axes[i]

            true_val = self.true_params[param]
            estimated_vals = self.results[f"{param}_est"]

            # Scatter plot: estimated vs true
            ax.scatter(
                np.full_like(estimated_vals, true_val),
                estimated_vals,
                alpha=0.6,
                s=50,
            )

            # Perfect estimation line
            ax.axline(
                (true_val, true_val),
                slope=1,
                color="red",
                linestyle="--",
                alpha=0.8,
                label="Perfect Estimation",
            )

            # Confidence bands (±10% and ±20%)
            ax.axhspan(
                true_val * 0.9,
                true_val * 1.1,
                alpha=0.2,
                color="green",
                label="±10% band",
            )
            ax.axhspan(
                true_val * 0.8,
                true_val * 1.2,
                alpha=0.1,
                color="yellow",
                label="±20% band",
            )

            ax.set_xlabel("True Value")
            ax.set_ylabel("Estimated Value")
            ax.set_title(f"{label}")
            ax.grid(True, alpha=0.3)

            if i == 0:  # Add legend to first plot
                ax.legend()

        # Summary statistics plot
        ax = axes[5]
        param_biases = []
        for param in param_names:
            param_biases.append(np.mean(self.results[f"{param}_rel_error"]))
        param_rmses = [
            np.sqrt(np.mean(self.results[f"{param}_error"] ** 2))
            / self.true_params[param]
            * 100
            for param in param_names
        ]

        x = np.arange(len(param_names))
        width = 0.35

        ax.bar(
            x - width / 2,
            np.abs(param_biases),
            width,
            label="|Bias| (%)",
            alpha=0.7,
        )
        ax.bar(
            x + width / 2,
            param_rmses,
            width,
            label="Rel. RMSE (%)",
            alpha=0.7,
        )

        ax.set_xlabel("Parameters")
        ax.set_ylabel("Error (%)")
        ax.set_title("Estimation Accuracy Summary")
        ax.set_xticks(x)
        ax.set_xticklabels(param_labels, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
