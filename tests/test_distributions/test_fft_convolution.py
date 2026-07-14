"""Regression tests for the FFT-convolution grid (audit findings B1/B2).

The generic ``fft_convolved_pdf`` previously (a) returned an all-zero
density whenever ANY requested point fell outside its grid -- creating an
artificial likelihood cliff for small-scale candidates on data with
extreme observations -- and (b) sized/centered its grid ignoring both a
dominant jump scale and a shifted jump location, silently losing the
distribution's mass.
"""

import numpy as np
from scipy.integrate import simpson

from jump_diffusion.distributions import NormalJump, SGEDJump, StudentTJump

DM, DS = 0.0002, 0.0126  # typical daily drift/vol of the diffusion part


class TestGridCoverage:
    def test_outlier_does_not_zero_whole_density(self):
        # Small-scale candidate + one extreme observation beyond the grid:
        # in-grid points must keep their density; only the outlier is ~0.
        dist = StudentTJump()
        params = {"jump_loc": 0.0, "jump_scale": 0.001, "jump_df": 5.0}
        xs = np.array([0.0, 0.01, 0.3])  # 0.3 lies beyond the small grid

        dens = dist.fft_convolved_pdf(xs, params, DM, DS)

        assert dens[0] > 1.0  # density near the mode is order 1/DS
        assert dens[1] > 0.0
        assert dens[2] == 0.0  # genuinely negligible there

    def test_mass_preserved_when_jump_scale_dominates(self):
        # jump_scale / diffusion_std ~ 160: the old grid (sized by the
        # *smaller* scale) could not contain the jump's mass at all.
        dist = StudentTJump()
        params = {"jump_loc": 0.0, "jump_scale": 0.5, "jump_df": 5.0}
        grid = np.linspace(-8, 8, 400001)

        mass = simpson(dist.fft_convolved_pdf(grid, params, 0.0, 0.00315), x=grid)

        assert abs(mass - 1.0) < 1e-2

    def test_mass_preserved_with_far_location(self):
        # jump_loc far from the origin: the grid must still cover the mass.
        dist = StudentTJump()
        params = {"jump_loc": 3.0, "jump_scale": 0.02, "jump_df": 5.0}
        grid = np.linspace(2.0, 4.0, 200001)

        mass = simpson(dist.fft_convolved_pdf(grid, params, DM, DS), x=grid)

        assert abs(mass - 1.0) < 1e-2

    def test_fft_matches_closed_form_in_normal_regime(self):
        # Accuracy guard: in the well-covered regime the (unchanged) grid
        # must keep matching the closed form.
        dist = NormalJump()
        params = {"jump_scale": 0.1}
        xs = np.linspace(-0.4, 0.4, 41)

        fft_dens = dist.fft_convolved_pdf(xs, params, DM, DS)
        closed = dist.diffusion_convolved_pdf(xs, params, DM, DS)

        np.testing.assert_allclose(fft_dens, closed, rtol=5e-3, atol=1e-6)


class TestGenericRvsLocation:
    def test_generic_rvs_respects_location(self):
        # SGED uses the generic inverse-CDF sampler; a shifted location
        # must shift the sample, not fall off a zero-centered grid.
        dist = SGEDJump()
        params = {
            "jump_loc": 0.5,
            "jump_scale": 0.05,
            "jump_nu": 1.5,
            "jump_xi": 1.2,
        }
        sample = dist.rvs(params, size=20000, random_state=np.random.default_rng(0))

        assert abs(np.mean(sample) - 0.5) < 0.01
        assert abs(np.std(sample) - 0.05) < 0.01
