Quick Start
===========

Basic usage
-----------

.. code-block:: python

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

Using a different jump distribution
------------------------------------

Jumps follow a skew-normal distribution by default. Other distributions can
be plugged in via ``jump_distribution``, both when simulating and when
estimating:

.. code-block:: python

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

Distributions without a known closed-form likelihood (like SGED) fall back
to a generic FFT-based convolution to approximate the mixture density, so
adding a new distribution only requires implementing its ``pdf`` -- see
:doc:`api/index` and the "Adding New Distributions" section of
`CONTRIBUTING.md <https://github.com/jdospina/jump-diffusion-estimation/blob/main/CONTRIBUTING.md>`_.

Comparing jump distributions
------------------------------

:class:`~jump_diffusion.validation.JumpDistributionComparison` fits several
candidate jump distributions to the same data and ranks them by AIC/BIC plus
a simulation-based Kolmogorov-Smirnov test:

.. code-block:: python

   from jump_diffusion.distributions import KouJump, NormalJump, SGEDJump, SkewNormalJump
   from jump_diffusion.validation import JumpDistributionComparison

   comparison = JumpDistributionComparison(increments, dt)
   comparison.fit("Normal", NormalJump())
   comparison.fit("SkewNormal", SkewNormalJump())
   comparison.fit("SGED", SGEDJump())
   comparison.fit("Kou", KouJump())

   print(comparison.compare())  # ranked by AIC, includes KS statistic/p-value
   comparison.plot_comparison()

More examples
-------------

Ready-to-run scripts and notebooks live in the ``examples/`` and
``notebooks/`` directories of the repository -- see the README for an
annotated list.
