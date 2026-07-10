"""
Unit tests for JumpDiffusionSimulator's plotting helper.
"""

import matplotlib

matplotlib.use("Agg")

from jump_diffusion.simulation import JumpDiffusionSimulator  # noqa: E402


class TestPlotSimulation:
    """Test suite for JumpDiffusionSimulator.plot_simulation."""

    def test_show_theoretical_overlays_pdf_on_jump_histogram(self):
        """
        With show_theoretical=True, the jump-size subplot (axes[1, 1]) should
        gain a line plot of the theoretical pdf on top of the histogram.
        """
        sim = JumpDiffusionSimulator(
            mu=0.1, sigma=0.2, jump_prob=0.5, jump_scale=0.15, jump_skew=0.0
        )
        sim.simulate_path(T=1.0, n_steps=200, x0=0.0, seed=1)

        fig = sim.plot_simulation(show_theoretical=True)
        jump_axis = fig.axes[3]
        # One Line2D for the theoretical pdf overlay, plus a legend.
        assert len(jump_axis.get_lines()) == 1
        assert jump_axis.get_legend() is not None

    def test_show_theoretical_false_omits_overlay(self):
        sim = JumpDiffusionSimulator(
            mu=0.1, sigma=0.2, jump_prob=0.5, jump_scale=0.15, jump_skew=0.0
        )
        sim.simulate_path(T=1.0, n_steps=200, x0=0.0, seed=1)

        fig = sim.plot_simulation(show_theoretical=False)
        jump_axis = fig.axes[3]
        assert len(jump_axis.get_lines()) == 0
        assert jump_axis.get_legend() is None

    def test_plot_simulation_with_no_jumps_does_not_crash(self):
        sim = JumpDiffusionSimulator(
            mu=0.1, sigma=0.2, jump_prob=0.0, jump_scale=0.15, jump_skew=0.0
        )
        sim.simulate_path(T=1.0, n_steps=50, x0=0.0, seed=1)

        fig = sim.plot_simulation(show_theoretical=True)
        assert fig is not None
