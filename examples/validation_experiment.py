"""
Complete validation experiment example.

This script demonstrates how to use the ValidationExperiment class
to test the accuracy of our jump-diffusion parameter estimators.
"""

from jump_diffusion import ValidationExperiment
import matplotlib.pyplot as plt

def main():
    """Run a comprehensive validation experiment."""
    print("Jump-Diffusion Validation Experiment")
    print("="*40)

    # Define true parameters for the experiment
    true_parameters = {
        'mu': 0.06,         # 6% annual drift
        'sigma': 0.22,      # 22% annual volatility
        'jump_prob': 0.08,  # 8% jump probability
        'jump_scale': 0.14, # moderate jump scale
        'jump_skew': 2.2    # positive skew
    }

    print("True parameters for validation:")
    for param, value in true_parameters.items():
        print(f"  {param}: {value}")

    # Create and run validation experiment
    experiment = ValidationExperiment(true_parameters)

    print(f"\nRunning Monte Carlo experiment...")
    results_df = experiment.run_experiment(
        n_simulations=15,  # More simulations for better statistics
        T=1.5,            # 1.5 years of data
        n_steps=378,      # Daily data
        x0=100.0,
        seed_base=123
    )

    if len(results_df) > 0:
        # Analyze results
        analysis = experiment.analyze_results()

        # Create plots
        experiment.plot_results()

        # Additional analysis
        print(f"\n" + "="*60)
        print("EXPERIMENT SUMMARY")
        print("="*60)
        print(f"Successful runs: {len(results_df)}")
        print(f"Average log-likelihood: {results_df['log_likelihood'].mean():.2f}")

        # Parameter recovery quality
        param_names = ['mu', 'sigma', 'jump_prob', 'jump_scale', 'jump_skew']
        best_recovered = min(param_names, key=lambda p: abs(results_df[f'{p}_rel_error'].mean()))
        worst_recovered = max(param_names, key=lambda p: abs(results_df[f'{p}_rel_error'].mean()))

        print(f"Best recovered parameter: {best_recovered}")
        print(f"Worst recovered parameter: {worst_recovered}")

        return results_df, analysis
    else:
        print("Experiment failed. No valid results obtained.")
        return None, None

if __name__ == "__main__":
    results_df, analysis = main()
