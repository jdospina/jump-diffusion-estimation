"""
Basic Usage Example for Jump-Diffusion Estimation

This example demonstrates the core functionality of the library
with a simple simulation and estimation workflow.
"""

import numpy as np
import matplotlib.pyplot as plt
from jump_diffusion import JumpDiffusionSimulator, JumpDiffusionEstimator

def main():
    """Run basic usage example."""
    print("Jump-Diffusion Parameter Estimation - Basic Example")
    print("="*55)
    
    # Step 1: Define true parameters
    true_params = {
        'mu': 0.08,         # 8% annual drift
        'sigma': 0.25,      # 25% annual volatility
        'jump_prob': 0.06,  # 6% jump probability per period
        'jump_scale': 0.12, # moderate jump sizes
        'jump_skew': 1.5    # positive skew
    }
    
    print("True parameters:")
    for param, value in true_params.items():
        print(f"  {param}: {value}")
    
    # Step 2: Simulate data
    print("\nSimulating jump-diffusion path...")
    simulator = JumpDiffusionSimulator(**true_params)
    times, path, jumps = simulator.simulate_path(
        T=2.0,       # 2 years
        n_steps=504, # daily observations
        x0=100.0,    # initial price
        seed=42      # reproducibility
    )
    
    print(f"Simulated {len(path)} observations over {times[-1]:.1f} years")
    print(f"Final value: {path[-1]:.2f} (return: {(path[-1]/path[0]-1)*100:.1f}%)")
    
    # Step 3: Estimate parameters
    print("\nEstimating parameters...")
    increments = np.diff(path)
    dt = times[1] - times[0]
    
    estimator = JumpDiffusionEstimator(increments, dt)
    results = estimator.estimate()
    
    # Step 4: Display results
    estimator.diagnostics()
    
    # Step 5: Visual comparison
    print("\nCreating comparison plots...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Original path
    axes[0,0].plot(times, path, 'b-', linewidth=1.5, alpha=0.8)
    axes[0,0].set_title('Simulated Jump-Diffusion Path')
    axes[0,0].set_xlabel('Time (years)')
    axes[0,0].set_ylabel('Price')
    axes[0,0].grid(True, alpha=0.3)
    
    # Parameter comparison
    param_names = ['mu', 'sigma', 'jump_prob', 'jump_scale', 'jump_skew']
    param_labels = ['Drift (μ)', 'Volatility (σ)', 'Jump Prob (p)', 'Jump Scale (ω)', 'Skewness (α)']
    
    x_pos = np.arange(len(param_names))
    true_vals = [true_params[p] for p in param_names]
    estimated_vals = [results['parameters'][p] for p in param_names]
    
    axes[0,1].bar(x_pos - 0.2, true_vals, 0.4, label='True', alpha=0.7)
    axes[0,1].bar(x_pos + 0.2, estimated_vals, 0.4, label='Estimated', alpha=0.7)
    axes[0,1].set_xlabel('Parameters')
    axes[0,1].set_ylabel('Value')
    axes[0,1].set_title('True vs Estimated Parameters')
    axes[0,1].set_xticks(x_pos)
    axes[0,1].set_xticklabels(param_labels, rotation=45)
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Increment distribution
    axes[1,0].hist(increments, bins=50, density=True, alpha=0.7, 
                   color='skyblue', edgecolor='black')
    axes[1,0].set_title('Distribution of Price Increments')
    axes[1,0].set_xlabel('Increment')
    axes[1,0].set_ylabel('Density')
    axes[1,0].grid(True, alpha=0.3)
    
    # Error analysis
    errors = [(estimated_vals[i] - true_vals[i])/true_vals[i] * 100 
              for i in range(len(param_names))]
    
    bars = axes[1,1].bar(param_labels, errors, alpha=0.7)
    axes[1,1].set_title('Relative Estimation Errors (%)')
    axes[1,1].set_ylabel('Error (%)')
    axes[1,1].tick_params(axis='x', rotation=45)
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Color bars by error magnitude
    for i, bar in enumerate(bars):
        if abs(errors[i]) > 10:
            bar.set_color('red')
        elif abs(errors[i]) > 5:
            bar.set_color('orange')
        else:
            bar.set_color('green')
    
    plt.tight_layout()
    plt.show()
    
    print("\nExample completed successfully!")
    print("Check the plots to see how well the estimation worked.")

if __name__ == "__main__":
    main()
