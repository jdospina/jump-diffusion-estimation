# Jump-Diffusion Parameter Estimation

A comprehensive Python library for simulating and estimating parameters of jump-diffusion processes with asymmetric jump distributions.

## 🚀 Features

- **Flexible Simulation**: Generate jump-diffusion paths with customizable parameters
- **Maximum Likelihood Estimation**: Robust parameter estimation using mixture distributions
- **Pluggable Jump Distributions**: Skew-normal, Normal (Merton), and the Skewed Generalized Error Distribution (SGED) built in, with a simple interface (`jump_distribution=`) to add more
- **Goodness-of-Fit Comparison**: Rank candidate jump distributions on the same data via AIC/BIC and a simulation-based Kolmogorov-Smirnov test
- **Validation Tools**: Monte Carlo experiments for method validation
- **Extensible Architecture**: Easy to add new models, jump distributions, and estimation methods
- **Educational Focus**: Comprehensive documentation and tutorials

## 📊 Model

Our implementation focuses on jump-diffusion processes of the form:

```
dX_t = μ dt + σ dW_t + J_t dN_t
```

Where:
- `μ`: drift parameter
- `σ`: diffusion volatility
- `W_t`: Brownian motion
- `J_t`: jump sizes (asymmetrically distributed)
- `N_t`: jump arrival times (Bernoulli approximation)

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/jdospina/jump-diffusion-estimation.git
cd jump-diffusion-estimation

# Install the package
pip install -e .

# Or install from PyPI (when available)
pip install jump-diffusion-estimation
```

## 🎯 Quick Start

```python
import numpy as np
from jump_diffusion import JumpDiffusionSimulator, JumpDiffusionEstimator

# Create simulator
simulator = JumpDiffusionSimulator(
    mu=0.05,           # 5% annual drift
    sigma=0.2,         # 20% annual volatility
    jump_prob=0.1,     # 10% jump probability per period
    jump_scale=0.15,   # jump magnitude scale
    jump_skew=2.0      # positive skewness
)

# Simulate a path
times, path, jumps = simulator.simulate_path(T=1.0, n_steps=252)

# Estimate parameters
increments = np.diff(path)
dt = times[1] - times[0]
estimator = JumpDiffusionEstimator(increments, dt)
results = estimator.estimate()

print(f"Estimated drift: {results['parameters']['mu']:.4f}")
print(f"Estimated volatility: {results['parameters']['sigma']:.4f}")
```

### Using a different jump distribution

Jumps follow a skew-normal distribution by default. Other distributions can be plugged in via `jump_distribution`, both when simulating and when estimating:

```python
from jump_diffusion.distributions import SGEDJump

simulator = JumpDiffusionSimulator(
    mu=0.05, sigma=0.2, jump_prob=0.1,
    jump_distribution=SGEDJump(),
    jump_loc=0.0, jump_scale=0.15, jump_nu=1.5, jump_xi=2.0,
)
times, path, jumps = simulator.simulate_path(T=1.0, n_steps=252)

increments = np.diff(path)
dt = times[1] - times[0]
estimator = JumpDiffusionEstimator(increments, dt, jump_distribution=SGEDJump())
results = estimator.estimate()
```

Distributions without a known closed-form likelihood (like SGED) fall back to a generic FFT-based convolution to approximate the mixture density, so adding a new distribution only requires implementing its `pdf`.

### Robust estimation with differential evolution

The default `L-BFGS-B` optimizer needs a reasonable initial guess and can stall on harder mixture likelihoods (SGED in particular). Differential evolution searches globally instead, needing no initial guess — the applied finding of the thesis this library is based on:

```python
results = estimator.estimate(method="differential_evolution", seed=42)
```

It costs thousands of likelihood evaluations (seconds instead of milliseconds), with defaults ported from the thesis (rand/1 strategy, `DEoptim`-style population sizing, early stopping on convergence).

### Comparing jump distributions

`JumpDistributionComparison` fits several candidate jump distributions to the same data and ranks them by AIC/BIC plus a simulation-based Kolmogorov-Smirnov test:

```python
from jump_diffusion.distributions import NormalJump, SGEDJump, SkewNormalJump
from jump_diffusion.validation import JumpDistributionComparison

comparison = JumpDistributionComparison(increments, dt)
comparison.fit("Normal", NormalJump())
comparison.fit("SkewNormal", SkewNormalJump())
comparison.fit("SGED", SGEDJump())

print(comparison.compare())  # ranked by AIC, includes KS statistic/p-value
comparison.plot_comparison()
```

## 📚 Examples

Ready-to-run scripts are available in the `examples/` directory:

- [basic_usage.py](examples/basic_usage.py) – demonstrates basic library usage
- [validation_experiment.py](examples/validation_experiment.py) – runs Monte Carlo validation experiments
- [jump_diffusion_playground.ipynb](notebooks/jump_diffusion_playground.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jdospina/jump-diffusion-estimation/blob/main/notebooks/jump_diffusion_playground.ipynb) – interactive playground: pick a jump distribution (Normal, Skew-Normal, SGED, Kou, Student-t), simulate, and try the "guess the parameters" game
- [sp500_jump_diffusion_example.ipynb](notebooks/sp500_jump_diffusion_example.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jdospina/jump-diffusion-estimation/blob/main/notebooks/sp500_jump_diffusion_example.ipynb) – applies the model to real S&P 500 data: parameter estimation, simulated-vs-real comparison, and ranking all five jump distributions by AIC/BIC/KS

### Notebook setup

Install optional dependencies and launch Jupyter to explore the notebook:

```bash
pip install notebook ipywidgets matplotlib
jupyter notebook
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by classical jump-diffusion literature
- Built with love for the quantitative finance community
- Special thanks to contributors and users
