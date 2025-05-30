# ==========================================
# CONTRIBUTING.md
# ==========================================

# Contributing to Jump-Diffusion Estimation

We welcome contributions to the Jump-Diffusion Parameter Estimation library! This document provides guidelines for contributing to the project.

## 🚀 How to Contribute

### Reporting Issues
- Use the GitHub issue tracker to report bugs
- Include a clear description of the problem
- Provide a minimal reproducible example
- Specify your Python version and library dependencies

### Suggesting Enhancements
- Use GitHub issues to suggest new features
- Explain the use case and expected behavior
- Consider backward compatibility

### Pull Requests
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass (`pytest`)
6. Update documentation if needed
7. Commit your changes (`git commit -m 'Add amazing feature'`)
8. Push to the branch (`git push origin feature/amazing-feature`)
9. Open a Pull Request

## 🧪 Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/jump-diffusion-estimation.git
cd jump-diffusion-estimation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .[dev]

# Run tests
pytest

# Run linting
flake8 src/ tests/
black src/ tests/
```

## 📋 Code Standards

### Style Guidelines
- Follow PEP 8 style guide
- Use Black for code formatting
- Maximum line length: 88 characters
- Use meaningful variable and function names

### Documentation
- All public functions must have docstrings
- Use Google-style docstrings
- Include examples in docstrings when helpful
- Update README.md for significant changes

### Testing
- Write tests for all new functionality
- Aim for >90% test coverage
- Use pytest for testing
- Test edge cases and error conditions

## 🏗️ Architecture Guidelines

### Adding New Models
1. Inherit from `BaseStochasticModel`
2. Implement all abstract methods
3. Add to `models/__init__.py`
4. Write comprehensive tests
5. Add documentation and examples

### Adding New Estimators
1. Inherit from `BaseEstimator`
2. Implement all abstract methods
3. Add to `estimation/__init__.py`
4. Include parameter bounds and validation
5. Write tests and documentation

### Adding New Distributions
1. Create in `distributions/` directory
2. Follow the established interface pattern
3. Include analytical properties when possible
4. Add tests for distribution properties

## 🔬 Research Extensions

We particularly welcome contributions in these areas:

### New Jump Distributions
- Asymmetric Laplace distributions
- Generalized hyperbolic distributions
- Tempered stable distributions
- Custom mixture distributions

### Advanced Models
- Multiple jump types per period
- Stochastic volatility with jumps
- Regime-switching jump-diffusion
- Fractional jump-diffusion

### Estimation Methods
- Bayesian estimation (MCMC)
- Method of moments with characteristic functions
- Particle filter methods
- Machine learning approaches

### Computational Improvements
- GPU acceleration
- Parallel processing for Monte Carlo
- Optimized numerical methods
- Memory-efficient implementations

## 📊 Benchmarking

When adding new estimators or models:
- Include benchmark comparisons
- Test on various parameter regimes
- Compare computational efficiency
- Provide accuracy metrics

## 🤝 Community Guidelines

- Be respectful and inclusive
- Provide constructive feedback
- Help others learn and improve
- Share knowledge and experiences

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

## 🙏 Recognition

Contributors will be acknowledged in:
- README.md contributors section
- Release notes for significant contributions
- Academic papers arising from this work (when appropriate)

Thank you for contributing to the Jump-Diffusion Estimation library!
