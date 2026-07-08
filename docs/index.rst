Jump-Diffusion Parameter Estimation
====================================

A Python library for simulating and estimating parameters of jump-diffusion
processes, with pluggable jump-size distributions and goodness-of-fit
comparison tools.

The model follows the stochastic differential equation:

.. math::

   dX_t = \mu\, dt + \sigma\, dW_t + J_t\, dN_t

where :math:`\mu` is the drift, :math:`\sigma` is the diffusion volatility,
:math:`W_t` is a standard Brownian motion, :math:`J_t` is the jump size
(drawn from a pluggable :class:`~jump_diffusion.distributions.JumpDistribution`),
and :math:`N_t` is a jump-arrival process approximated by Bernoulli trials.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   api/index

Indices and tables
===================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
