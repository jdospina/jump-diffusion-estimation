"""
Maximum Likelihood Estimator for Jump-Diffusion Models

This module estimates JumpDiffusionModel parameters by maximizing the
mixture likelihood implemented on the model itself.
"""

import numpy as np
import warnings
from scipy import stats
from scipy.optimize import Bounds, differential_evolution, minimize
from typing import Any, Dict, List, Optional, Tuple
from .base_estimator import BaseEstimator
from ..models.jump_diffusion import JumpDiffusionModel
from ..distributions import JumpDistribution


class JumpDiffusionEstimator(BaseEstimator):
    """
    Maximum likelihood estimator for jump-diffusion models.

    This estimator handles jump-diffusion processes with a pluggable
    jump-size distribution (skew-normal by default) using mixture
    likelihood functions. The input ``data`` must be a one-dimensional
    array of increments.

    References:
    - Ardia, D., Ospina, J. D., & Giraldo, N. D. (2011). Jump-diffusion
      calibration using differential evolution. Wilmott, 2011(55), 76-79.
    """

    def __init__(
        self,
        data: np.ndarray,
        dt: float,
        jump_distribution: Optional[JumpDistribution] = None,
    ):
        """
        Initialize the estimator.

        Parameters:
        -----------
        data : np.ndarray
            One-dimensional array of observed increments. If you have
            a path, compute ``np.diff(path)`` first.
        dt : float
            Time step size between consecutive increments.
        jump_distribution : JumpDistribution, optional
            Distribution assumed for the jump sizes. Defaults to
            :class:`SkewNormalJump`.
        """
        # Accept only 1D arrays of increments
        if data.ndim == 1:
            self.increments = data
        else:
            msg = "data must be a one-dimensional array of increments"
            raise ValueError(msg)

        super().__init__(self.increments, dt)

        # Model used to evaluate the mixture likelihood at trial parameters
        self._model = JumpDiffusionModel(jump_distribution=jump_distribution)
        self._param_names = (
            "mu",
            "sigma",
            "jump_prob",
        ) + self._model.jump_distribution.param_names

        # Calculate basic statistics
        self.n_obs = len(self.increments)
        self.mean_increment = float(np.mean(self.increments))
        self.std_increment = float(np.std(self.increments))
        self.skewness = float(stats.skew(self.increments))
        self.kurtosis = float(stats.kurtosis(self.increments))

    def log_likelihood(self, params: np.ndarray) -> float:
        """
        Calculate negative log-likelihood for optimization.

        Parameters:
        -----------
        params : np.ndarray
            Parameter vector, ordered as ``self._param_names``:
            ``[mu, sigma, jump_prob, *jump_distribution.param_names]``.

        Returns:
        --------
        float
            Negative log-likelihood value
        """
        sigma, jump_prob = params[1], params[2]

        # Parameter constraints shared by every jump distribution
        if sigma <= 0:
            return np.inf
        if jump_prob < 0 or jump_prob > 1:
            return np.inf

        self._model.update_parameters(**dict(zip(self._param_names, params)))
        value = self._model.log_likelihood(self.increments, self.dt)
        # Penalize non-numeric likelihood values so that population-based
        # optimizers (differential evolution) simply discard the offending
        # candidates instead of corrupting the ranking -- same device as in
        # Ospina Arango (2009), where NaN evaluations are mapped to a large
        # positive constant.
        if not np.isfinite(value):
            return np.inf
        return -value

    def _get_initial_guess(self) -> np.ndarray:
        """Generate intelligent initial parameter guess."""
        initial_mu = self.mean_increment / self.dt
        initial_sigma = self.std_increment / np.sqrt(self.dt)
        initial_jump_prob = 0.1

        jump_guess = self._model.jump_distribution.initial_guess(
            self.mean_increment, self.std_increment, self.skewness
        )

        values = [initial_mu, initial_sigma, initial_jump_prob]
        values += [
            jump_guess[name] for name in self._model.jump_distribution.param_names
        ]
        return np.array(values)

    def _finite_bounds(self) -> List[Tuple[float, float]]:
        """
        Data-driven finite search box for differential evolution.

        Differential evolution samples its initial population uniformly
        over a bounded region, so the open-ended bounds used by L-BFGS-B
        (e.g. ``mu`` unbounded, ``sigma < inf``) must be replaced by finite
        ones. This mirrors the :math:`\\theta_L`/:math:`\\theta_U` limits
        in Ospina Arango (2009), whose applied result is that even *very*
        generous boxes (little prior knowledge of the solution) still lead
        differential evolution to the optimum where gradient methods fail.
        Bounds that are already finite are kept as-is; only ``None`` ends
        are replaced with wide data-driven limits.
        """
        initial_mu = self.mean_increment / self.dt
        initial_sigma = self.std_increment / np.sqrt(self.dt)
        mu_halfwidth = max(1.0, 10 * abs(initial_mu), 10 * initial_sigma)
        # Jump parameters live on the scale of individual increments; a
        # single jump 20x the typical increment size is a generous cap
        # (and std_increment is itself inflated by any jumps in the data).
        jump_cap = max(20 * self.std_increment, 1e-3)

        finite = [
            (initial_mu - mu_halfwidth, initial_mu + mu_halfwidth),  # mu
            (1e-6, 10 * initial_sigma),  # sigma
            (1e-6, 1 - 1e-6),  # jump_prob
        ]
        jump_bounds = self._model.jump_distribution.param_bounds()
        for name in self._model.jump_distribution.param_names:
            low, high = jump_bounds[name]
            finite.append(
                (
                    -jump_cap if low is None else low,
                    jump_cap if high is None else high,
                )
            )
        return finite

    def estimate(
        self,
        initial_guess: Optional[np.ndarray] = None,
        method: str = "L-BFGS-B",
        bounds: Optional[List[Tuple[float, float]]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        r"""
        Estimate parameters using maximum likelihood.

        Parameters:
        -----------
        initial_guess : np.array, optional
            Initial parameter values. Optional for every method: gradient
            methods fall back to a moment-based heuristic guess, while
            ``"differential_evolution"`` does not need one at all (its
            initial population is sampled over ``bounds``; when a guess
            *is* supplied it merely seeds one population member).
        method : str
            Optimization method. Any ``scipy.optimize.minimize`` method
            name (default ``"L-BFGS-B"``), or ``"differential_evolution"``
            for the global, population-based optimizer from
            ``scipy.optimize.differential_evolution``. Differential
            evolution is markedly more robust to poor prior knowledge on
            this mixture likelihood -- the applied finding of Ospina
            Arango (2009), where L-BFGS-B stalled at box boundaries and
            simulated annealing diverged, while DE (rand/1, population 70,
            400 generations in R's ``DEoptim``) recovered the true
            parameters even under very wide bounds. The trade-off is
            computational cost (thousands of likelihood evaluations).
            For the differential evolution algorithm, see Storn & Price (1997).
        bounds : list of (low, high) tuples, optional
            Optimization box, one pair per parameter in the order
            ``[mu, sigma, jump_prob, *jump_distribution.param_names]``.
            Defaults to the model's own bounds for gradient methods, or a
            wide data-driven finite box (see ``_finite_bounds``) for
            differential evolution, which requires every bound finite.
        \*\*kwargs : dict
            Additional arguments for the optimizer: merged into
            ``options`` for ``scipy.optimize.minimize``, passed directly
            to ``scipy.optimize.differential_evolution`` (e.g. ``seed=42``
            for reproducibility, ``maxiter``, ``popsize``, ``workers``).

        Returns:
        --------
        dict
            Estimation results
        """
        if method.lower() in ("differential_evolution", "de"):
            result = self._optimize_differential_evolution(
                initial_guess, bounds, **kwargs
            )
        else:
            result = self._optimize_minimize(initial_guess, method, bounds, **kwargs)

        # Process results
        parameters = dict(zip(self._param_names, result.x))

        results: Dict[str, Any] = {
            "parameters": parameters,
            "log_likelihood": -result.fun,
            "aic": 2 * len(result.x) + 2 * result.fun,
            "bic": len(result.x) * np.log(self.n_obs) + 2 * result.fun,
            "optimization_result": result,
            "convergence": result.success,
            "data_stats": {
                "n_obs": self.n_obs,
                "mean_increment": self.mean_increment,
                "std_increment": self.std_increment,
                "skewness": self.skewness,
                "kurtosis": self.kurtosis,
            },
        }

        self.results = results
        self.fitted = True
        return results

    def _optimize_minimize(
        self,
        initial_guess: Optional[np.ndarray],
        method: str,
        bounds: Optional[List[Tuple[float, float]]],
        **kwargs,
    ) -> Any:
        """Local optimization via ``scipy.optimize.minimize``."""
        if initial_guess is None:
            initial_guess = self._get_initial_guess()
        initial_guess = np.asarray(initial_guess, dtype=float)

        if bounds is None:
            bounds = self._model.get_parameter_bounds()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return minimize(
                fun=self.log_likelihood,  # type: ignore
                x0=initial_guess,
                method=method,
                bounds=bounds,
                options={"maxiter": 1000, "ftol": 1e-9, **kwargs},
            )

    def _optimize_differential_evolution(
        self,
        initial_guess: Optional[np.ndarray],
        bounds: Optional[List[Tuple[float, float]]],
        **kwargs,
    ) -> Any:
        """
        Global optimization via ``scipy.optimize.differential_evolution``.

        Defaults port the configuration from Ospina Arango (2009), which
        used R's ``DEoptim``: strategy rand/1 with binomial crossover
        (scipy's ``"rand1bin"``) and 400 generations. ``DEoptim`` used a
        fixed population of 70 individuals for the 7-parameter SGED model;
        scipy sizes the population as ``popsize * n_params``, so
        ``popsize=10`` reproduces exactly that for the SGED case and
        scales proportionally for other jump distributions. The thesis
        also observes convergence well before the generation cap and
        suggests a convergence-based stopping criterion -- scipy's ``tol``
        (default 0.01) provides exactly that, so runs typically stop
        early. Any of these can be overridden via ``**kwargs``.
        """
        if bounds is None:
            bounds = self._finite_bounds()
        lows, highs = zip(*bounds)
        de_bounds = Bounds(np.asarray(lows, float), np.asarray(highs, float))

        de_kwargs: Dict[str, Any] = {
            "strategy": "rand1bin",
            "maxiter": 400,
            "popsize": 10,
        }
        de_kwargs.update(kwargs)
        if initial_guess is not None:
            de_kwargs["x0"] = np.asarray(initial_guess, dtype=float)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return differential_evolution(
                func=self.log_likelihood,
                bounds=de_bounds,
                **de_kwargs,
            )

    def estimate_standard_errors(
        self,
        n_points: int = 7,
        confidence_level: float = 0.95,
        grid_width_factor: float = 3.0,
    ) -> Dict[str, Any]:
        """
        Estimate parameter standard errors and confidence intervals using
        likelihood profiling based on Wilks' theorem (Wilks, 1938).

        Parameters:
        -----------
        n_points : int
            Number of points to evaluate in the grid for profiling.
        confidence_level : float
            Confidence level for the intervals (e.g. 0.95 for 95% confidence).
        grid_width_factor : float
            Factor determining the width of the grid around the MLE.

        Returns:
        --------
        dict
            Dictionary containing standard errors and confidence intervals.
        """
        if not self.fitted or self.results is None:
            raise ValueError("Model must be fitted first.")

        opt_params = self.results["parameters"]
        opt_param_vals = np.array([opt_params[name] for name in self._param_names])
        opt_log_lik = self.results["log_likelihood"]

        standard_errors = {}
        confidence_intervals = {}
        profile_data = {}

        # Critical chi-squared value for the profile likelihood threshold
        # (Wilks' theorem)
        critical_val = stats.chi2.ppf(confidence_level, df=1)
        threshold = opt_log_lik - 0.5 * critical_val

        # Get optimization bounds
        bounds = self._model.get_parameter_bounds()

        for idx, param_name in enumerate(self._param_names):
            mle_val = opt_param_vals[idx]

            # Initialize profile dictionary with MLE
            profile_dict = {mle_val: opt_log_lik}
            low_bound, high_bound = bounds[idx]

            # Optimization closure
            def optimize_profile(val):
                def neg_log_lik_profile(free_params):
                    full_params = np.zeros(len(self._param_names))
                    full_params[idx] = val
                    free_idx = 0
                    for j in range(len(self._param_names)):
                        if j != idx:
                            full_params[j] = free_params[free_idx]
                            free_idx += 1
                    return self.log_likelihood(full_params)

                free_init = [
                    opt_param_vals[j] for j in range(len(self._param_names)) if j != idx
                ]
                free_bounds = [
                    bounds[j] for j in range(len(self._param_names)) if j != idx
                ]

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = minimize(
                        fun=neg_log_lik_profile,
                        x0=free_init,
                        method="L-BFGS-B",
                        bounds=free_bounds,
                        options={"maxiter": 100, "ftol": 1e-6},
                    )

                if res.success and np.isfinite(res.fun):
                    return -res.fun
                return -np.inf

            # Adaptive search parameters
            initial_step = max(0.01 * abs(mle_val), 1e-4)
            max_step_factor = 2.0
            max_search_steps = 15

            # Outward search right
            curr_val = mle_val
            step = initial_step
            for _ in range(max_search_steps):
                next_val = curr_val + step
                if high_bound is not None and next_val >= high_bound:
                    next_val = high_bound - 1e-5
                    val_lik = optimize_profile(next_val)
                    profile_dict[next_val] = val_lik
                    break
                val_lik = optimize_profile(next_val)
                profile_dict[next_val] = val_lik

                if val_lik < threshold:
                    break

                curr_val = next_val
                step *= max_step_factor

            # Outward search left
            curr_val = mle_val
            step = initial_step
            for _ in range(max_search_steps):
                next_val = curr_val - step
                if low_bound is not None and next_val <= low_bound:
                    next_val = low_bound + 1e-5
                    val_lik = optimize_profile(next_val)
                    profile_dict[next_val] = val_lik
                    break
                val_lik = optimize_profile(next_val)
                profile_dict[next_val] = val_lik

                if val_lik < threshold:
                    break

                curr_val = next_val
                step *= max_step_factor

            # Sort evaluated points
            sorted_vals = sorted(list(profile_dict.keys()))

            # Infill points if we have fewer than n_points
            while len(sorted_vals) < n_points:
                # Find the gap with the largest absolute distance
                max_gap_idx = -1
                max_gap_dist = -1
                for i in range(len(sorted_vals) - 1):
                    v1, v2 = sorted_vals[i], sorted_vals[i + 1]
                    dist = v2 - v1
                    if dist > max_gap_dist:
                        max_gap_dist = dist
                        max_gap_idx = i

                if max_gap_idx == -1:
                    break

                v1 = sorted_vals[max_gap_idx]
                v2 = sorted_vals[max_gap_idx + 1]
                mid_val = (v1 + v2) / 2.0
                val_lik = optimize_profile(mid_val)
                profile_dict[mid_val] = val_lik
                sorted_vals = sorted(list(profile_dict.keys()))

            grid = np.array(sorted_vals)
            profile_vals = np.array([profile_dict[v] for v in grid])

            valid_mask = np.isfinite(profile_vals)
            grid_valid = grid[valid_mask]
            profile_valid = profile_vals[valid_mask]

            if len(grid_valid) > 0:
                min_val = grid_valid.min()
                max_val = grid_valid.max()
            else:
                min_val = low_bound if low_bound is not None else mle_val - 1.0
                max_val = high_bound if high_bound is not None else mle_val + 1.0

            se_val = np.nan
            ci_low, ci_high = np.nan, np.nan

            if len(profile_valid) >= 3:
                # Fit parabola: y = a*x^2 + b*x + c
                # Use up to 5 points with the highest log-likelihood to ensure
                # we are near the peak and avoid arbitrary absolute
                # log-likelihood thresholds.
                n_fit_points = min(5, len(profile_valid))
                top_indices = np.argsort(profile_valid)[-n_fit_points:]
                p_coefs = np.polyfit(
                    grid_valid[top_indices], profile_valid[top_indices], 2
                )
                a = p_coefs[0]
                if a < 0:
                    se_val = np.sqrt(-1.0 / (2.0 * a))

                # Find roots where L_p(x) == threshold using linear interpolation
                left_mask = grid_valid <= mle_val
                grid_left = grid_valid[left_mask]
                prof_left = profile_valid[left_mask]
                if len(prof_left) >= 2 and prof_left[0] < threshold:
                    ci_low = float(np.interp(threshold, prof_left, grid_left))
                else:
                    ci_low = float(min_val)

                right_mask = grid_valid >= mle_val
                grid_right = grid_valid[right_mask]
                prof_right = profile_valid[right_mask]
                if len(prof_right) >= 2 and prof_right[-1] < threshold:
                    ci_high = float(
                        np.interp(threshold, prof_right[::-1], grid_right[::-1])
                    )
                else:
                    ci_high = float(max_val)

                # Fallback: If parabolic fit failed (a >= 0, causing se_val=nan),
                # estimate SE from the profile likelihood confidence interval width.
                if not np.isfinite(se_val):
                    if (
                        np.isfinite(ci_low)
                        and np.isfinite(ci_high)
                        and ci_low < ci_high
                    ):
                        # Ensure the bounds actually crossed the threshold
                        # (are inside the grid)
                        if ci_low > min_val + 1e-6 and ci_high < max_val - 1e-6:
                            z_val = np.sqrt(critical_val)
                            dist = (ci_high - ci_low) / 2.0
                            se_val = float(dist / z_val)

            standard_errors[param_name] = se_val
            confidence_intervals[param_name] = (ci_low, ci_high)
            profile_data[param_name] = {
                "grid": grid.tolist(),
                "values": profile_vals.tolist(),
                "threshold": threshold,
            }

        self.results["standard_errors"] = standard_errors
        self.results["confidence_intervals"] = confidence_intervals
        self.results["profile_data"] = profile_data

        return {
            "standard_errors": standard_errors,
            "confidence_intervals": confidence_intervals,
        }

    def _observed_information_matrix(
        self,
        param_vals: np.ndarray,
        rel_step: float = 1e-4,
    ) -> np.ndarray:
        """
        Numerically approximate the observed Fisher information matrix.

        The observed information is the Hessian of the *negative*
        log-likelihood evaluated at the estimate. Since
        :meth:`log_likelihood` already returns the negative log-likelihood,
        its Hessian is exactly the observed information. It is built here by
        central finite differences: second partial derivatives on the
        diagonal and mixed second partials off the diagonal.

        Following Efron & Hinkley (1978), the *observed* information (the
        Hessian at the estimate) is preferred over the *expected* Fisher
        information for constructing Wald standard errors, being both easier
        to compute for this mixture likelihood and, in their analysis, a
        more relevant conditioning quantity.

        Parameters:
        -----------
        param_vals : np.ndarray
            Parameter vector at which to evaluate the Hessian, ordered as
            ``self._param_names``. Normally the MLE.
        rel_step : float
            Relative step size for the finite differences. The per-parameter
            absolute step is ``rel_step * max(|value|, 1e-2)``, further
            shrunk so that every perturbed point stays strictly inside the
            parameter bounds (``log_likelihood`` returns ``+inf`` outside
            them, which would corrupt the Hessian).

        Returns:
        --------
        np.ndarray
            The ``(k, k)`` symmetric observed information matrix, where
            ``k = len(self._param_names)``. Entries that could not be
            evaluated (e.g. a parameter pinned to a bound) are ``nan``.
        """
        n = len(param_vals)
        bounds = self._model.get_parameter_bounds()

        # Per-parameter finite-difference steps, kept inside the bounds.
        steps = np.empty(n)
        for i in range(n):
            scale = max(abs(param_vals[i]), 1e-2)
            h = rel_step * scale
            low, high = bounds[i]
            if low is not None:
                h = min(h, 0.25 * (param_vals[i] - low))
            if high is not None:
                h = min(h, 0.25 * (high - param_vals[i]))
            steps[i] = h if h > 0 else np.nan

        f0 = self.log_likelihood(param_vals)
        hessian = np.full((n, n), np.nan)

        def f(shift: np.ndarray) -> float:
            return self.log_likelihood(param_vals + shift)

        # Diagonal: second partial derivatives
        for i in range(n):
            hi = steps[i]
            if not np.isfinite(hi):
                continue
            ei = np.zeros(n)
            ei[i] = hi
            f_plus, f_minus = f(ei), f(-ei)
            if np.isfinite(f_plus) and np.isfinite(f_minus) and np.isfinite(f0):
                hessian[i, i] = (f_plus - 2.0 * f0 + f_minus) / (hi * hi)

        # Off-diagonal: mixed second partial derivatives
        for i in range(n):
            hi = steps[i]
            if not np.isfinite(hi):
                continue
            for j in range(i + 1, n):
                hj = steps[j]
                if not np.isfinite(hj):
                    continue
                ei = np.zeros(n)
                ei[i] = hi
                ej = np.zeros(n)
                ej[j] = hj
                f_pp, f_pm = f(ei + ej), f(ei - ej)
                f_mp, f_mm = f(-ei + ej), f(-ei - ej)
                if all(np.isfinite(v) for v in (f_pp, f_pm, f_mp, f_mm)):
                    val = (f_pp - f_pm - f_mp + f_mm) / (4.0 * hi * hj)
                    hessian[i, j] = val
                    hessian[j, i] = val

        return hessian

    def estimate_wald_standard_errors(
        self,
        confidence_level: float = 0.95,
    ) -> Dict[str, Any]:
        """
        Estimate standard errors and confidence intervals from the observed
        Fisher information (the Wald method).

        This is the classical large-sample alternative to the profile
        likelihood intervals of :meth:`estimate_standard_errors`. It inverts
        the observed information matrix (see
        :meth:`_observed_information_matrix`) to obtain the asymptotic
        covariance of the maximum likelihood estimator, takes the square
        root of its diagonal as the standard errors, and forms symmetric
        intervals ``estimate +/- z * SE`` with ``z`` the standard normal
        quantile.

        Wald intervals are cheap (a single Hessian, no re-optimization) but
        are symmetric by construction and are known to under-cover in finite
        samples for parameters near a boundary -- notably the jump
        probability as it approaches zero. Providing both Wald and profile
        intervals is deliberate: comparing their coverage across jump
        distributions is one of the intended research uses of this library.

        Parameters:
        -----------
        confidence_level : float
            Confidence level for the intervals (e.g. 0.95 for 95%).

        Returns:
        --------
        dict
            Dictionary with ``standard_errors`` and
            ``confidence_intervals``, each keyed by parameter name. A
            parameter's standard error is ``nan`` when the observed
            information could not be evaluated or inverted to a positive
            variance (e.g. an estimate pinned to a bound, or a singular
            Hessian); its interval is then ``(nan, nan)``. The results are
            also stored on ``self.results`` under ``wald_standard_errors``,
            ``wald_confidence_intervals`` and ``observed_information``.

        Raises:
        -------
        ValueError
            If the model has not been fitted yet.
        """
        if not self.fitted or self.results is None:
            raise ValueError("Model must be fitted first.")
        results = self.results

        opt_params = results["parameters"]
        param_vals = np.array([opt_params[name] for name in self._param_names])

        information = self._observed_information_matrix(param_vals)

        alpha = 1.0 - confidence_level
        z = float(stats.norm.ppf(1.0 - alpha / 2.0))

        # Invert the observed information to get the asymptotic covariance
        # matrix, guarding against a non-finite or singular Hessian.
        covariance: Optional[np.ndarray]
        if np.all(np.isfinite(information)):
            try:
                covariance = np.linalg.inv(information)
            except np.linalg.LinAlgError:
                covariance = None
        else:
            covariance = None

        standard_errors: Dict[str, float] = {}
        confidence_intervals: Dict[str, Tuple[float, float]] = {}

        for idx, name in enumerate(self._param_names):
            se_val = np.nan
            if covariance is not None:
                var = covariance[idx, idx]
                if np.isfinite(var) and var > 0:
                    se_val = float(np.sqrt(var))

            if np.isfinite(se_val):
                est = float(param_vals[idx])
                ci_low, ci_high = est - z * se_val, est + z * se_val
            else:
                ci_low, ci_high = np.nan, np.nan

            standard_errors[name] = se_val
            confidence_intervals[name] = (ci_low, ci_high)

        results["wald_standard_errors"] = standard_errors
        results["wald_confidence_intervals"] = confidence_intervals
        results["observed_information"] = information

        return {
            "standard_errors": standard_errors,
            "confidence_intervals": confidence_intervals,
        }

    def estimate_bootstrap_standard_errors(
        self,
        n_bootstrap: int = 200,
        confidence_level: float = 0.95,
        seed: Optional[int] = None,
        **estimate_kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Estimate standard errors and confidence intervals by parametric
        bootstrap.

        This is the third inference route in the library, alongside the
        profile likelihood intervals of :meth:`estimate_standard_errors` and
        the Wald intervals of :meth:`estimate_wald_standard_errors`. It makes
        no asymptotic-normality or regularity assumption, so it is the most
        trustworthy of the three near a boundary (e.g. the jump probability
        approaching zero) -- at the cost of re-fitting the model
        ``n_bootstrap`` times.

        The procedure is the textbook parametric bootstrap: treat the fitted
        parameters as if they were the truth, simulate ``n_bootstrap`` fresh
        datasets of the same length from that model, re-estimate the
        parameters on each, and read the sampling variability directly off
        the resulting replicate estimates. Standard errors are their
        (ddof=1) standard deviation and confidence intervals are percentile
        intervals.

        Parameters:
        -----------
        n_bootstrap : int
            Number of bootstrap replicates (model re-fits). More replicates
            reduce Monte Carlo error in the standard errors and intervals
            but scale the cost linearly.
        confidence_level : float
            Confidence level for the percentile intervals (e.g. 0.95).
        seed : int, optional
            Base seed for reproducibility. When given, replicate ``b`` is
            simulated with seed ``seed + b``; when ``None`` the replicates
            draw from NumPy's global random state and are not reproducible.
        \\*\\*estimate_kwargs : dict
            Forwarded to :meth:`estimate` for each replicate re-fit (e.g.
            ``method="differential_evolution"``). Defaults to the same
            gradient-based fit used elsewhere.

        Returns:
        --------
        dict
            Dictionary with ``standard_errors`` and
            ``confidence_intervals`` (each keyed by parameter name), plus
            ``n_successful`` -- the number of replicates whose re-fit
            converged and contributed to the estimates. Non-converged
            replicates are discarded. If fewer than two replicates succeed,
            all standard errors and interval endpoints are ``nan``. The
            results are also stored on ``self.results`` under
            ``bootstrap_standard_errors``, ``bootstrap_confidence_intervals``
            and ``bootstrap_estimates`` (the raw replicate matrix).

        Raises:
        -------
        ValueError
            If the model has not been fitted yet.
        """
        if not self.fitted or self.results is None:
            raise ValueError("Model must be fitted first.")
        results = self.results

        fitted_params = results["parameters"]
        jump_dist = self._model.jump_distribution

        # Data-generating model: the fitted parameters treated as truth.
        model = JumpDiffusionModel(jump_distribution=jump_dist, **fitted_params)
        n_steps = self.n_obs
        horizon = n_steps * self.dt

        replicates: List[List[float]] = []
        for b in range(n_bootstrap):
            rep_seed = None if seed is None else seed + b
            _, path, _ = model.simulate(
                T=horizon, n_steps=n_steps, x0=0.0, seed=rep_seed
            )
            increments = np.diff(path)

            estimator = JumpDiffusionEstimator(
                increments, self.dt, jump_distribution=jump_dist
            )
            rep_result = estimator.estimate(**estimate_kwargs)
            if rep_result["convergence"]:
                rep_params = rep_result["parameters"]
                replicates.append([rep_params[name] for name in self._param_names])

        n_successful = len(replicates)

        standard_errors: Dict[str, float] = {}
        confidence_intervals: Dict[str, Tuple[float, float]] = {}

        alpha = 1.0 - confidence_level
        lower_pct, upper_pct = 100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)

        if n_successful >= 2:
            estimates = np.asarray(replicates, dtype=float)
            ses = np.std(estimates, axis=0, ddof=1)
            ci_lows = np.percentile(estimates, lower_pct, axis=0)
            ci_highs = np.percentile(estimates, upper_pct, axis=0)
            for idx, name in enumerate(self._param_names):
                standard_errors[name] = float(ses[idx])
                confidence_intervals[name] = (
                    float(ci_lows[idx]),
                    float(ci_highs[idx]),
                )
        else:
            estimates = np.asarray(replicates, dtype=float)
            for name in self._param_names:
                standard_errors[name] = np.nan
                confidence_intervals[name] = (np.nan, np.nan)

        results["bootstrap_standard_errors"] = standard_errors
        results["bootstrap_confidence_intervals"] = confidence_intervals
        results["bootstrap_estimates"] = estimates

        return {
            "standard_errors": standard_errors,
            "confidence_intervals": confidence_intervals,
            "n_successful": n_successful,
        }

    def _pure_diffusion_log_likelihood(
        self, increments: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        Maximized log-likelihood under the no-jump null (pure diffusion).

        Under ``H0: jump_prob = 0`` the increments are i.i.d. Gaussian, so
        the maximum likelihood fit is available in closed form: the mean and
        (population, MLE) standard deviation of the increments. This returns
        that maximized log-likelihood together with the two fitted moments,
        which also parameterize the data-generating process for the
        bootstrap in :meth:`test_for_jumps`.

        Parameters:
        -----------
        increments : np.ndarray
            One-dimensional array of increments.

        Returns:
        --------
        tuple
            ``(log_likelihood, mean, std)``. ``log_likelihood`` is
            ``-inf`` when the increments are constant (``std == 0``).
        """
        mean = float(np.mean(increments))
        std = float(np.std(increments))  # ddof=0 -> the Gaussian MLE
        if std <= 0:
            return -np.inf, mean, std
        log_lik = float(np.sum(stats.norm.logpdf(increments, loc=mean, scale=std)))
        return log_lik, mean, std

    def test_for_jumps(
        self,
        n_bootstrap: int = 200,
        seed: Optional[int] = None,
        **estimate_kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Test for the presence of jumps via a parametric bootstrap likelihood
        ratio test of ``H0: jump_prob = 0`` (pure diffusion) against
        ``H1: jump_prob > 0``.

        The statistic is the usual likelihood ratio
        ``LR = 2 * (loglik_full - loglik_null)``, where ``loglik_full`` is
        the maximized jump-diffusion log-likelihood (already available from
        the fit) and ``loglik_null`` is the closed-form maximized Gaussian
        log-likelihood (see :meth:`_pure_diffusion_log_likelihood`).

        Its asymptotic null distribution is **non-standard**: under ``H0``
        the jump probability sits on the boundary of the parameter space
        (Chernoff, 1954; Self & Liang, 1987) *and* the jump-size parameters
        become unidentified (Davies, 1977, 1987) -- so the usual chi-squared
        calibration does not apply. We therefore obtain the p-value by
        parametric bootstrap (a Monte Carlo test): simulate ``n_bootstrap``
        pure-diffusion datasets from the fitted null model, recompute the LR
        statistic on each, and compare the observed statistic against that
        simulated null distribution. This sidesteps both pathologies by
        construction.

        Parameters:
        -----------
        n_bootstrap : int
            Number of bootstrap datasets drawn under the null. Each requires
            re-fitting the full jump-diffusion model, so cost scales
            linearly.
        seed : int, optional
            Seed for the bootstrap data generation, for reproducibility.
        \\*\\*estimate_kwargs : dict
            Forwarded to :meth:`estimate` for each bootstrap re-fit.

        Returns:
        --------
        dict
            Dictionary with:

            * ``lr_statistic`` -- observed likelihood ratio statistic.
            * ``p_value`` -- bootstrap p-value
              ``(1 + #{LR* >= LR_obs}) / (n_successful + 1)``; ``nan`` if no
              bootstrap re-fit converged.
            * ``log_likelihood_full`` / ``log_likelihood_null`` -- the two
              maximized log-likelihoods on the observed data.
            * ``n_bootstrap`` / ``n_successful`` -- requested and converged
              replicate counts.
            * ``bootstrap_statistics`` -- the simulated null LR values.

            The result is also stored on ``self.results`` under
            ``jump_test``.

        Raises:
        -------
        ValueError
            If the model has not been fitted yet.
        """
        if not self.fitted or self.results is None:
            raise ValueError("Model must be fitted first.")
        results = self.results

        log_lik_full = results["log_likelihood"]
        log_lik_null, mean0, std0 = self._pure_diffusion_log_likelihood(self.increments)
        # The full model nests the null, so LR is non-negative in theory;
        # clamp to guard against the optimizer stopping just short of it.
        lr_observed = max(0.0, 2.0 * (log_lik_full - log_lik_null))

        jump_dist = self._model.jump_distribution
        rng = np.random.default_rng(seed)

        bootstrap_stats: List[float] = []
        for _ in range(n_bootstrap):
            increments_b = rng.normal(mean0, std0, size=self.n_obs)
            estimator = JumpDiffusionEstimator(
                increments_b, self.dt, jump_distribution=jump_dist
            )
            rep_result = estimator.estimate(**estimate_kwargs)
            if not rep_result["convergence"]:
                continue
            log_lik_full_b = rep_result["log_likelihood"]
            log_lik_null_b, _, _ = self._pure_diffusion_log_likelihood(increments_b)
            bootstrap_stats.append(max(0.0, 2.0 * (log_lik_full_b - log_lik_null_b)))

        n_successful = len(bootstrap_stats)
        stats_array = np.asarray(bootstrap_stats, dtype=float)
        if n_successful >= 1:
            exceed = int(np.sum(stats_array >= lr_observed))
            p_value = (1.0 + exceed) / (n_successful + 1.0)
        else:
            p_value = np.nan

        jump_test = {
            "lr_statistic": lr_observed,
            "p_value": p_value,
            "log_likelihood_full": log_lik_full,
            "log_likelihood_null": log_lik_null,
            "n_bootstrap": n_bootstrap,
            "n_successful": n_successful,
            "bootstrap_statistics": stats_array,
        }
        results["jump_test"] = jump_test
        return jump_test

    def plot_profiles(self, figsize: Tuple[float, float] = (15, 10)) -> None:
        """
        Plot the profile log-likelihood curves for each parameter.
        """
        if (
            not self.fitted
            or self.results is None
            or "profile_data" not in self.results
        ):
            print("Profile data not available. Run estimate_standard_errors() first.")
            return

        import matplotlib.pyplot as plt

        profile_data = self.results["profile_data"]
        standard_errors = self.results["standard_errors"]
        confidence_intervals = self.results["confidence_intervals"]
        opt_params = self.results["parameters"]

        label_map = {
            "mu": "Drift (μ)",
            "sigma": "Volatility (σ)",
            "jump_prob": "Jump Prob (p)",
            "jump_scale": "Jump Scale",
            "jump_skew": "Skewness (α)",
            "jump_loc": "Jump Location",
            "jump_nu": "Kurtosis (ν)",
            "jump_xi": "Asymmetry (ξ)",
            "jump_df": "Degrees of Freedom (df)",
            "jump_prob_up": "Jump Prob Up (p_up)",
            "jump_scale_up": "Jump Scale Up (η_up)",
            "jump_scale_down": "Jump Scale Down (η_down)",
        }

        n_plots = len(self._param_names)
        n_cols = min(3, n_plots)
        n_rows = int(np.ceil(n_plots / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_plots == 1:
            axes = np.array([axes])
        else:
            axes = axes.flatten()

        for idx, name in enumerate(self._param_names):
            ax = axes[idx]
            data = profile_data[name]
            grid = np.array(data["grid"])
            values = np.array(data["values"])
            threshold = data["threshold"]
            mle_val = opt_params[name]
            se = standard_errors[name]
            ci_low, ci_high = confidence_intervals[name]

            # Filter finite values for plotting
            valid = np.isfinite(values)
            ax.plot(grid[valid], values[valid], "b-o", label="Profile Log-Lik")
            ax.axvline(
                x=mle_val, color="red", linestyle="--", label=f"MLE: {mle_val:.4f}"
            )
            ax.axhline(
                y=threshold, color="green", linestyle=":", label="95% CI Threshold"
            )

            if np.isfinite(ci_low) and ci_low > grid.min():
                ax.axvline(x=ci_low, color="green", linestyle="--", alpha=0.7)
            if np.isfinite(ci_high) and ci_high < grid.max():
                ax.axvline(x=ci_high, color="green", linestyle="--", alpha=0.7)

            label = label_map.get(name, name.replace("_", " ").title())
            se_str = f", SE: {se:.4f}" if np.isfinite(se) else ""
            ax.set_title(f"{label}{se_str}")
            ax.set_xlabel("Value")
            ax.set_ylabel("Log-Likelihood")
            ax.grid(True, alpha=0.3)
            if idx == 0:
                ax.legend()

        # Delete unused subplots
        for j in range(n_plots, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()

    def diagnostics(self) -> None:
        """Print diagnostic information about the estimation.

        The method writes a summary of the fitted model to ``stdout`` and
        returns ``None``. The following metrics are reported:

        * Estimated parameters (μ, σ, jump probability, and the jump
          distribution's own parameters)
        * Log-likelihood value
        * Information criteria (AIC and BIC)
        * Optimizer convergence flag
        * Comparison of empirical vs. theoretical mean, standard deviation
          and the expected number of jumps
        """
        if not self.fitted or self.results is None:
            print("Model not fitted. Run estimate() first.")
            return

        params = self.results["parameters"]
        se_dict = self.results.get("standard_errors", {})
        ci_dict = self.results.get("confidence_intervals", {})

        print("=" * 60)
        print("JUMP-DIFFUSION ESTIMATION RESULTS")
        print("=" * 60)

        if se_dict:
            print(
                f"{'Parameter':<18} | {'Estimate':<10} | "
                f"{'Std Error':<10} | {'95% Conf. Interval':<22}"
            )
            print("-" * 68)
            for name in self._param_names:
                val = params[name]
                se = se_dict.get(name, np.nan)
                se_str = f"{se:.6f}" if np.isfinite(se) else "N/A"
                ci = ci_dict.get(name, (np.nan, np.nan))
                ci_str = (
                    f"[{ci[0]:.4f}, {ci[1]:.4f}]" if np.all(np.isfinite(ci)) else "N/A"
                )
                print(f"{name:<18} | {val:<10.6f} | {se_str:<10} | {ci_str:<22}")
        else:
            print(f"Drift (μ):              {params['mu']:.6f}")
            print(f"Volatility (σ):         {params['sigma']:.6f}")
            print(f"Jump probability (p):   {params['jump_prob']:.6f}")
            for name in self._model.jump_distribution.param_names:
                print(f"{name:<24}{params[name]:.6f}")

        print(
            f"\nLog-likelihood:         {self.results['log_likelihood']:.2f}",
        )
        print(f"AIC:                    {self.results['aic']:.2f}")
        print(f"BIC:                    {self.results['bic']:.2f}")
        print(f"Convergence:            {self.results['convergence']}")

        # Model comparison with data
        print("\nDATA vs MODEL COMPARISON")
        print("-" * 30)
        theoretical_mean = params["mu"] * self.dt
        # Approximation: treats the distribution's characteristic scale as
        # the jump size's standard deviation, which holds exactly for the
        # Normal jump distribution and approximately for the others.
        jump_scale = self._model.jump_distribution.characteristic_scale(
            {k: params[k] for k in self._model.jump_distribution.param_names}
        )
        theoretical_var = (
            params["sigma"] ** 2 * self.dt + params["jump_prob"] * jump_scale**2
        )

        print("Mean increment:")
        print(f"  Empirical:  {self.mean_increment:.6f}")
        print(f"  Theoretical: {theoretical_mean:.6f}")
        print("Std deviation:")
        print(f"  Empirical:  {self.std_increment:.6f}")
        print(f"  Theoretical: {np.sqrt(theoretical_var):.6f}")
        print(f"Expected jumps: {params['jump_prob'] * self.n_obs:.1f}")
