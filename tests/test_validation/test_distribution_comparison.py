"""
Unit tests for JumpDistributionComparison.
"""

import numpy as np

from jump_diffusion.distributions import NormalJump, SkewNormalJump
from jump_diffusion.simulation import JumpDiffusionSimulator
from jump_diffusion.validation import JumpDistributionComparison


class TestJumpDistributionComparison:
    """Test suite for JumpDistributionComparison."""

    def setup_method(self):
        sim = JumpDiffusionSimulator(
            mu=0.05,
            sigma=0.2,
            jump_prob=0.1,
            jump_distribution=SkewNormalJump(),
            jump_scale=0.15,
            jump_skew=2.0,
        )
        times, path, _ = sim.simulate_path(T=1.0, n_steps=500, x0=100.0, seed=7)
        self.increments = np.diff(path)
        self.dt = times[1] - times[0]

    def test_fit_returns_ks_fields(self):
        comparison = JumpDistributionComparison(self.increments, self.dt)
        result = comparison.fit(
            "Normal", NormalJump(), seed=1, n_bootstrap=15, ks_reference_size=2000
        )

        assert "ks_statistic" in result
        assert "ks_pvalue" in result
        assert 0.0 <= result["ks_pvalue"] <= 1.0
        assert result["ks_n_bootstrap"] == 15
        assert np.isfinite(result["aic"])
        assert np.isfinite(result["bic"])

    def test_bootstrap_pvalue_can_be_skipped(self):
        comparison = JumpDistributionComparison(self.increments, self.dt)
        result = comparison.fit(
            "Normal", NormalJump(), seed=1, n_bootstrap=0, ks_reference_size=2000
        )

        # Opting out of the bootstrap still yields a (stable) statistic but
        # no p-value.
        assert np.isfinite(result["ks_statistic"])
        assert np.isnan(result["ks_pvalue"])

    def test_bootstrap_pvalue_is_reproducible_with_seed(self):
        first = JumpDistributionComparison(self.increments, self.dt).fit(
            "Normal", NormalJump(), seed=3, n_bootstrap=12, ks_reference_size=2000
        )
        second = JumpDistributionComparison(self.increments, self.dt).fit(
            "Normal", NormalJump(), seed=3, n_bootstrap=12, ks_reference_size=2000
        )
        assert first["ks_pvalue"] == second["ks_pvalue"]

    def test_compare_ranks_by_aic(self):
        comparison = JumpDistributionComparison(self.increments, self.dt)
        comparison.fit("Normal", NormalJump(), seed=1, n_bootstrap=10)
        comparison.fit("SkewNormal", SkewNormalJump(), seed=1, n_bootstrap=10)

        table = comparison.compare()

        assert table["aic"].is_monotonic_increasing
        assert set(table.columns) >= {
            "distribution",
            "log_likelihood",
            "aic",
            "bic",
            "ks_statistic",
            "ks_pvalue",
            "convergence",
        }

    def test_compare_before_fit_returns_empty(self):
        comparison = JumpDistributionComparison(self.increments, self.dt)
        table = comparison.compare()
        assert len(table) == 0
