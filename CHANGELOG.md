# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-07-10

### Added
- **Likelihood-based inference**, three complementary routes on
  `JumpDiffusionEstimator`:
  - `estimate_standard_errors()` — profile-likelihood intervals (Wilks).
  - `estimate_wald_standard_errors()` — Wald intervals from the observed
    Fisher information (numerical Hessian).
  - `estimate_bootstrap_standard_errors()` — parametric bootstrap standard
    errors and percentile intervals.
- `test_for_jumps()` — a parametric-bootstrap likelihood-ratio test of
  `H0: jump_prob = 0`, correctly handling the boundary and
  unidentified-nuisance-parameter pathologies.
- `summary()` — a tidy table collecting the estimates and every inference
  route computed, and a public `param_names` property.
- Canonical end-to-end tutorial notebook, with English and Spanish versions
  of all notebooks.

### Changed
- The goodness-of-fit Kolmogorov-Smirnov check in `JumpDistributionComparison`
  now uses a valid parametric-bootstrap p-value (a large reference sample for
  a stable statistic, plus re-fitting under the fitted null), replacing the
  previous single-sample comparison.

### Fixed
- Restored a green CI baseline by resolving accumulated linting/formatting
  debt across the package.

## [0.1.0]

### Added
- Initial release: jump-diffusion simulation and maximum-likelihood
  estimation with pluggable jump-size distributions (skew-normal, Merton
  normal, SGED, Kou, Student-t), L-BFGS-B and Differential Evolution
  optimizers, Monte Carlo validation, and goodness-of-fit distribution
  comparison.
