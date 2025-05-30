# ==========================================
# scripts/run_validation.py
# ==========================================

#!/usr/bin/env python3
"""
Script to run validation experiments from command line.

Usage:
    python scripts/run_validation.py --n_sims 20 --T 2.0 --output validation_results.json
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from jump_diffusion import ValidationExperiment

def main():
    parser = argparse.ArgumentParser(description='Run jump-diffusion validation experiment')
    parser.add_argument('--n_sims', type=int, default=10, help='Number of simulations')
    parser.add_argument('--T', type=float, default=1.0, help='Time horizon')
    parser.add_argument('--n_steps', type=int, default=252, help='Number of time steps')
    parser.add_argument('--output', type=str, help='Output file for results')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Default parameters
    true_params = {
        'mu': 0.05,
        'sigma': 0.2,
        'jump_prob': 0.1,
        'jump_scale': 0.15,
        'jump_skew': 2.0
    }
    
    print(f"Running validation with {args.n_sims} simulations...")
    
    experiment = ValidationExperiment(true_params)
    results_df = experiment.run_experiment(
        n_simulations=args.n_sims,
        T=args.T,
        n_steps=args.n_steps,
        seed_base=args.seed
    )
    
    if len(results_df) > 0:
        analysis = experiment.analyze_results()
        
        if args.output:
            # Save results
            output_data = {
                'true_parameters': true_params,
                'experiment_settings': {
                    'n_simulations': args.n_sims,
                    'T': args.T,
                    'n_steps': args.n_steps,
                    'seed': args.seed
                },
                'results': results_df.to_dict('records'),
                'analysis': analysis
            }
            
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2, default=str)
            
            print(f"Results saved to {args.output}")
        
        experiment.plot_results()
    else:
        print("Validation experiment failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()
