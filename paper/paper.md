---
title: 'jump-diffusion-estimation: Likelihood-based inference for jump-diffusion processes with asymmetric, heavy-tailed jump distributions in Python'
tags:
  - Python
  - jump-diffusion
  - stochastic processes
  - maximum likelihood
  - profile likelihood
  - parametric bootstrap
  - differential evolution
  - quantitative finance
authors:
  - name: Juan David Ospina Arango
    orcid: 0000-0002-3547-5247
    affiliation: 1
affiliations:
  - name: Departamento de Ciencias de la Computación y de la Decisión, Facultad de Minas, Universidad Nacional de Colombia, Medellín 050035, Colombia
    index: 1
date: 12 July 2026
bibliography: paper.bib
---

# Summary

Jump-diffusion processes extend the classical diffusion model of asset prices
with a compound jump component, capturing the sudden large movements that
Brownian motion alone cannot reproduce [@Merton1976]. `jump-diffusion-estimation`
is a Python library for simulating these processes and estimating their
parameters by maximum likelihood from discretely sampled data, with a focus on
*inference* — not just point estimates — under flexible jump-size distributions.

The library models increments of

$$dX_t = \mu\,dt + \sigma\,dW_t + J_t\,dN_t,$$

where jump sizes $J_t$ follow a pluggable distribution. Five families are
built in: normal [@Merton1976], skew-normal [@Azzalini1985], the skewed
generalized error distribution (SGED) [@Theodossiou2015], Kou's double
exponential [@Kou2002], and Student-t. When the Gaussian–jump convolution in
the transition density has no closed form, the library falls back to a generic
FFT-based convolution, so adding a new jump distribution only requires
implementing its `pdf`.

On top of estimation, the library provides a likelihood-based inference
toolkit: profile-likelihood confidence intervals with Wilks' threshold
[@Wilks1938], Wald standard errors from the observed Fisher information,
parametric-bootstrap standard errors and intervals, a parametric-bootstrap
likelihood-ratio test for the presence of jumps ($H_0\!: p = 0$), and
goodness-of-fit comparison of candidate jump distributions via AIC/BIC and a
simulation-based Kolmogorov–Smirnov test. Likelihoods can be maximized with
L-BFGS-B or with Differential Evolution [@StornPrice1997], a global optimizer
whose effectiveness for jump-diffusion calibration was documented in
@Ardia2011, building on @Ospina2009.

# Statement of need

Maximum-likelihood calibration of jump-diffusions is standard practice in
quantitative finance and econometrics, yet the software landscape leaves
practitioners assembling the pipeline by hand. General-purpose stochastic
simulation packages in Python (e.g., `sdepy`, `stochastic`) generate
jump-diffusion paths but offer no estimation. In R, the `yuima` framework
[@Brouste2014] supports quasi-likelihood estimation for a broad class of SDEs
with jumps, but it does not target the discrete-time mixture likelihood with
interchangeable asymmetric jump laws, nor does it bundle the inference layer —
profile-likelihood intervals, bootstrap intervals, and a jump test — that
applied work needs. Published empirical studies typically rely on one-off
research code.

Reliable inference is the harder part of this problem, and the part most often
skipped. The mixture likelihood is multimodal, so local optimizers can silently
converge to spurious optima; the numerical Hessian is frequently
ill-conditioned, making Wald intervals fragile; and testing for the presence
of jumps is doubly non-standard, since the null hypothesis places the jump
probability on the boundary of the parameter space [@SelfLiang1987] and leaves
the jump-law parameters unidentified [@Davies1987], so the classical
likelihood-ratio test does not have its textbook null distribution.
`jump-diffusion-estimation` addresses each pathology directly: global
optimization via Differential Evolution, profile-likelihood and
parametric-bootstrap intervals as robust alternatives to Wald, and a
parametric-bootstrap likelihood-ratio test whose null distribution is
simulated rather than assumed.

The library is aimed at researchers and practitioners in quantitative finance,
econometrics, and applied statistics, as well as instructors: the
documentation includes bilingual (English/Spanish) tutorial notebooks, runnable
on Google Colab, covering simulation, estimation with both optimizers, all
three uncertainty-quantification routes, the jump test, and model comparison
on real S&P 500 data. It also serves as the computational engine for an
ongoing Monte Carlo study of finite-sample coverage of profile-likelihood
intervals and the size and power of the bootstrap jump test.

# Functionality

- **Simulation** of jump-diffusion paths with any of the built-in or
  user-supplied jump distributions.
- **Estimation** by maximum likelihood on a Bernoulli (at-most-one-jump)
  discrete-time mixture, using L-BFGS-B with moment-based starting values or
  Differential Evolution (rand/1 strategy, `DEoptim`-style population sizing).
- **Uncertainty quantification** via profile likelihood, Wald (observed Fisher
  information), and parametric bootstrap, with diagnostic tables and profile
  plots.
- **Jump testing**: a parametric-bootstrap likelihood-ratio test of pure
  diffusion against the jump-diffusion alternative.
- **Model comparison**: fitting several jump laws to the same data and ranking
  them by AIC, BIC, and a simulation-based Kolmogorov–Smirnov statistic.
- **Validation tools**: reproducible Monte Carlo experiments for bias, RMSE,
  and coverage of the estimators.

The implementation builds on NumPy [@Harris2020], SciPy [@Virtanen2020],
pandas [@McKinney2010], and Matplotlib [@Hunter2007]. The package is
distributed on PyPI, documented on Read the Docs, tested with continuous
integration, and archived on Zenodo.

# Acknowledgements

The estimation methodology originates in the author's MSc thesis at
Universidad Nacional de Colombia [@Ospina2009]; the Differential Evolution
configuration follows the subsequent collaboration in @Ardia2011.

# References
