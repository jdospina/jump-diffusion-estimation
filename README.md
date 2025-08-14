# Jump-Diffusion Parameter Estimation

A comprehensive Python library for simulating and estimating parameters of jump-diffusion processes with asymmetric jump distributions.

## 🚀 Features

- **Flexible Simulation**: Generate jump-diffusion paths with customizable parameters
- **Maximum Likelihood Estimation**: Robust parameter estimation using mixture distributions
- **Asymmetric Jump Distributions**: Support for skew-normal and other asymmetric distributions
- **Validation Tools**: Monte Carlo experiments for method validation
- **Extensible Architecture**: Easy to add new models and estimation methods
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

## 📚 Examples

Ready-to-run scripts are available in the `examples/` directory:

- [basic_usage.py](examples/basic_usage.py) – demonstrates basic library usage
- [validation_experiment.py](examples/validation_experiment.py) – runs Monte Carlo validation experiments

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by classical jump-diffusion literature
- Built with love for the quantitative finance community
- Special thanks to contributors and users
