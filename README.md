# Jump-Diffusion Parameter Estimation

[![PyPI version](https://img.shields.io/pypi/v/jump-diffusion-estimation.svg)](https://pypi.org/project/jump-diffusion-estimation/)
[![Python versions](https://img.shields.io/pypi/pyversions/jump-diffusion-estimation.svg)](https://pypi.org/project/jump-diffusion-estimation/)
[![Documentation Status](https://readthedocs.org/projects/jump-diffusion-estimation/badge/?version=latest)](https://jump-diffusion-estimation.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21305522.svg)](https://doi.org/10.5281/zenodo.21305522)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

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
# Install from PyPI
pip install jump-diffusion-estimation
```

Or install the latest development version from source:

```bash
git clone https://github.com/jdospina/jump-diffusion-estimation.git
cd jump-diffusion-estimation
pip install -e .
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

### Standard errors via Likelihood Profiling

In complex jump-diffusion mixture models, the numerical Hessian is often unstable or ill-conditioned. Standard errors and 95% confidence intervals can be robustly calculated using Profile Likelihood. After estimating the parameters (preferably with global optimization), you can run:

```python
# Compute standard errors and confidence intervals using a Wilks' theorem threshold
se_results = estimator.estimate_standard_errors(n_points=5, confidence_level=0.95)

# The results table now includes standard errors and CI bounds
estimator.diagnostics()

# Visualize the profile log-likelihood curves
estimator.plot_profiles()
```

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

Ready-to-run scripts live in the `examples/` directory:

- [basic_usage.py](examples/basic_usage.py) – demonstrates basic library usage
- [validation_experiment.py](examples/validation_experiment.py) – runs Monte Carlo validation experiments

Interactive notebooks live in the `notebooks/` directory. Every notebook is available in English and in Spanish (suffix `_spanish`):

- **[getting_started.ipynb](notebooks/getting_started.ipynb)** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jdospina/jump-diffusion-estimation/blob/main/notebooks/getting_started.ipynb) – **The canonical end-to-end tutorial: simulate, estimate (L-BFGS-B and Differential Evolution), quantify uncertainty via the three inference routes (profile / Wald / bootstrap), test for jumps, and compare jump distributions. Start here.**
  - 🇪🇸 [getting_started_spanish.ipynb](notebooks/getting_started_spanish.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jdospina/jump-diffusion-estimation/blob/main/notebooks/getting_started_spanish.ipynb) – Tutorial completo de principio a fin: simulación, estimación (L-BFGS-B y Evolución Diferencial), inferencia por tres rutas (perfil / Wald / bootstrap), contraste de saltos y comparación de distribuciones.
- [jump_diffusion_playground.ipynb](notebooks/jump_diffusion_playground.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jdospina/jump-diffusion-estimation/blob/main/notebooks/jump_diffusion_playground.ipynb) – Interactive playground: pick a jump distribution (Normal, Skew-Normal, SGED, Kou, Student-t), wiggle the sliders, simulate, and play the "guess the parameters" game.
  - 🇪🇸 [jump_diffusion_playground_spanish.ipynb](notebooks/jump_diffusion_playground_spanish.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jdospina/jump-diffusion-estimation/blob/main/notebooks/jump_diffusion_playground_spanish.ipynb) – Laboratorio interactivo: elige una distribución de saltos, mueve los controles, simula y juega a adivinar los parámetros.
- [differential_evolution_showcase.ipynb](notebooks/differential_evolution_showcase.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jdospina/jump-diffusion-estimation/blob/main/notebooks/differential_evolution_showcase.ipynb) – Differential Evolution vs. L-BFGS-B on the multimodal mixture likelihood of the SGED jump-diffusion model.
  - 🇪🇸 [differential_evolution_showcase_spanish.ipynb](notebooks/differential_evolution_showcase_spanish.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jdospina/jump-diffusion-estimation/blob/main/notebooks/differential_evolution_showcase_spanish.ipynb) – Evolución Diferencial frente a L-BFGS-B sobre la verosimilitud multimodal del modelo de difusión con saltos SGED.
- [sp500_case_study.ipynb](notebooks/sp500_case_study.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jdospina/jump-diffusion-estimation/blob/main/notebooks/sp500_case_study.ipynb) – Real S&P 500 data: parameter estimation, simulated-vs-real comparison, and a ranking of all five jump distributions by AIC/BIC and bootstrap KS.
  - 🇪🇸 [sp500_case_study_spanish.ipynb](notebooks/sp500_case_study_spanish.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jdospina/jump-diffusion-estimation/blob/main/notebooks/sp500_case_study_spanish.ipynb) – Datos reales del S&P 500: estimación de parámetros, comparación simulado-vs-real y ranking de las cinco distribuciones de salto por AIC/BIC y KS bootstrap.

### Notebook setup

Install optional dependencies and launch Jupyter to explore the notebooks:

```bash
pip install notebook ipywidgets matplotlib
jupyter notebook
```

## 📖 Academic References

The numerical methods and statistical models implemented in this library build on the following literature:

1. **Calibration with Differential Evolution and the SGED:**
   - Ospina Arango, J. D. (2009). *MSc thesis*. Universidad Nacional de Colombia. (Foundations of applying the SGED and Differential Evolution to jump-diffusion processes.)
   - Ardia, D., Ospina, J. D., & Giraldo, N. D. (2011). *Jump-diffusion calibration using differential evolution.* Wilmott, 2011(55), 76-79.
   - Storn, R., & Price, K. (1997). *Differential evolution–a simple and efficient heuristic for global optimization over continuous spaces.* Journal of global optimization, 11(4), 341-359.

2. **Jump-diffusion models and distributions:**
   - Merton, R. C. (1976). *Option pricing when underlying stock returns are discontinuous.* Journal of financial economics, 3(1-2), 125-144.
   - Theodossiou, P. (2015). *Skewed Generalized Error Distribution of Financial Assets and Option Pricing.* Multinational Finance Journal, 19(4), 223-266.

3. **Statistical inference (likelihood profiling):**
   - Wilks, S. S. (1938). *The large-sample distribution of the likelihood ratio for testing composite hypotheses.* The Annals of Mathematical Statistics, 9(1), 60-62.

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by classical jump-diffusion literature
- Built with love for the quantitative finance community
- Special thanks to contributors and users
