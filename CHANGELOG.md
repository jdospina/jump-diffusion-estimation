# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.2] - 2026-07-14

Fixes from a systematic audit of the library (line-by-line code review
plus a numerical battery verifying mathematical invariants).

### Fixed
- FFT-convolution density (`fft_convolved_pdf`), used by the SGED and
  Student-t families:
  - A single evaluation point outside the grid no longer zeroes the
    *whole* returned density (previously an artificial likelihood cliff
    for small-scale candidates whenever the data contained an extreme
    observation). Out-of-grid points now evaluate to 0 individually.
  - The grid step is coarsened when the default span cannot contain the
    densities' mass (dominant jump scale, or a jump location far from
    the origin), preserving total mass instead of silently truncating
    it. The default regime is unchanged.
- `diagnostics()` theoretical moments now include the jump contribution
  (`E[dX] = mu*dt + p*E[J]`; exact mixture variance). Previously the
  mean ignored jumps entirely, badly misreporting models with
  asymmetric jumps.
- `ValidationExperiment` no longer produces infinite relative errors
  when a true parameter is zero (now `NaN`), and its plots/summaries
  are NaN-safe.

### Added
- `JumpDistribution.mean()` / `.variance()`: generic numerical jump-size
  moments, verified against closed forms.
- `JumpDistribution.characteristic_location()`: grid placement for
  shifted jump distributions; the generic sampler now centers on it.

### Changed
- `ValidationExperiment.analyze_results()`: the misnamed `coverage_95`
  statistic is now `within_5pct_rate` (it is an accuracy rate, not
  confidence-interval coverage).
- English is now the repository's primary language: README fully in
  English, and notebooks renamed — English names unsuffixed
  (`getting_started`, `jump_diffusion_playground`,
  `differential_evolution_showcase`, `sp500_case_study`) with Spanish
  versions carrying the `_spanish` suffix. Links to the old notebook
  filenames no longer work.

## [0.2.1] - 2026-07-13

### Fixed
- `plot_simulation()` no longer renders twice in Jupyter notebooks: the
  figure is now deregistered from pyplot before being returned, so it
  displays exactly once via the returned object. In non-cell contexts
  (e.g. `ipywidgets.Output`, callbacks), display the returned figure
  explicitly with `display(fig)`.

### Changed
- Packaging modernized to `pyproject.toml` (PEP 621) with automated PyPI
  publishing via trusted publishing; the package is now on PyPI and
  documentation is hosted on Read the Docs.
- All notebooks (Spanish and English) install the released package from
  PyPI instead of GitHub, and the S&P 500 notebooks cap the KS
  parametric-bootstrap cost with an explicit `n_bootstrap=49`.
- Citation metadata: Zenodo DOI and author ORCID added; JOSS paper draft
  included under `paper/`.

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
