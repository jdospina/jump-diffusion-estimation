"""Tests for the generic numerical jump-size moments (audit finding B3)."""

import numpy as np

from jump_diffusion.distributions import (
    KouJump,
    NormalJump,
    SkewNormalJump,
    StudentTJump,
)


class TestMomentsAgainstClosedForms:
    def test_normal_moments(self):
        d, p = NormalJump(), {"jump_scale": 0.15}
        assert abs(d.mean(p)) < 1e-6
        assert abs(d.variance(p) - 0.15**2) < 1e-6

    def test_skew_normal_moments(self):
        omega, alpha = 0.15, 3.0
        d = SkewNormalJump()
        p = {"jump_scale": omega, "jump_skew": alpha}
        delta = alpha / np.sqrt(1 + alpha**2)
        mean_cf = omega * delta * np.sqrt(2 / np.pi)
        var_cf = omega**2 * (1 - 2 * delta**2 / np.pi)
        assert abs(d.mean(p) - mean_cf) < 1e-5
        assert abs(d.variance(p) - var_cf) < 1e-6

    def test_kou_moments(self):
        pu, su, sd = 0.4, 0.08, 0.12
        d = KouJump()
        p = {"jump_prob_up": pu, "jump_scale_up": su, "jump_scale_down": sd}
        mean_cf = pu * su - (1 - pu) * sd
        second_cf = pu * 2 * su**2 + (1 - pu) * 2 * sd**2
        assert abs(d.mean(p) - mean_cf) < 1e-4
        assert abs(d.variance(p) - (second_cf - mean_cf**2)) < 1e-4

    def test_student_t_moments_with_location(self):
        d = StudentTJump()
        p = {"jump_loc": 0.3, "jump_scale": 0.1, "jump_df": 8.0}
        # Standardized parameterization: mean = loc, std = jump_scale
        # (up to the grid's ~20-scale tail truncation).
        assert abs(d.mean(p) - 0.3) < 1e-3
        assert abs(d.variance(p) - 0.1**2) < 5e-4
