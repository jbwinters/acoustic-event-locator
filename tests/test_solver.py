import math

import numpy as np
import pytest

import locate_event as le
from helpers import C, arrivals, l_xyz, linear_xyz, nearest_solution_error, pos_error, square_xyz


class TestPerfectData:
    def test_square_array_exact(self):
        XYZ, src = square_xyz(20.0), np.array([4.0, 7.0])
        sol = le.solve_tdoa(arrivals(src, XYZ, t0=0.7), XYZ, C)
        assert pos_error(sol, src) < 1e-6
        assert abs(sol.t0 - 0.7) < 1e-9
        assert sol.dof == 1 and sol.converged and not sol.ambiguous and not sol.degenerate
        assert np.max(np.abs(sol.residuals_s)) < 1e-9
        assert np.allclose(sol.delta, 0.0) and sol.rejected == []

    def test_three_recordings_exactly_determined(self):
        XYZ, src = square_xyz(20.0)[:3], np.array([4.0, 7.0])
        sol = le.solve_tdoa(arrivals(src, XYZ), XYZ, C)
        assert pos_error(sol, src) < 1e-6 and sol.dof == 0

    def test_source_outside_array(self):
        XYZ, src = square_xyz(20.0), np.array([-60.0, 45.0])
        sol = le.solve_tdoa(arrivals(src, XYZ), XYZ, C)
        assert pos_error(sol, src) < 1e-4

    def test_far_source_uses_large_search_radius(self):
        XYZ, src = square_xyz(10.0), np.array([120.0, 90.0])
        sol = le.solve_tdoa(arrivals(src, XYZ), XYZ, C, sigma_t=np.full(4, 0.1e-3))
        assert pos_error(sol, src) < 0.05

    def test_heights_matter(self):
        XYZ = square_xyz(20.0)
        XYZ[:, 2] = 6.0
        src = np.array([4.0, 7.0])
        t = arrivals(src, XYZ, source_z=0.0)
        exact = le.solve_tdoa(t, XYZ, C, source_z=0.0)
        flat = XYZ.copy()
        flat[:, 2] = 0.0
        ignored = le.solve_tdoa(t, flat, C)
        assert pos_error(exact, src) < 1e-6
        assert pos_error(ignored, src) > 0.3

    def test_deterministic(self):
        XYZ, src = l_xyz(), np.array([12.0, 30.0])
        t = arrivals(src, XYZ, noise=0.3e-3)
        a, b = le.solve_tdoa(t, XYZ, C), le.solve_tdoa(t, XYZ, C)
        assert np.array_equal(a.s_xy, b.s_xy) and np.array_equal(a.cov, b.cov)


class TestNoiseAndUncertainty:
    def test_monte_carlo_accuracy_and_coverage(self):
        XYZ, src, sigma = l_xyz(), np.array([12.0, 30.0]), 0.2e-3
        errs, inside = [], []
        for seed in range(150):
            t = arrivals(src, XYZ, noise=sigma, rng=np.random.default_rng(seed))
            sol = le.solve_tdoa(t, XYZ, C, sigma_t=np.full(len(XYZ), sigma))
            errs.append(pos_error(sol, src))
            inside.append(le.mahalanobis_xy(sol, src) <= math.sqrt(5.991))
        assert np.mean(errs) < 0.15 and np.max(errs) < 0.6
        assert 0.90 <= np.mean(inside) <= 1.0

    def test_ellipse_grows_with_timing_noise(self):
        XYZ, src = square_xyz(20.0), np.array([4.0, 7.0])
        t = arrivals(src, XYZ)
        a1 = le.ellipse_from_cov2(le.solve_tdoa(t, XYZ, C, sigma_t=np.full(4, 0.1e-3)).cov_xy)[0]
        a2 = le.ellipse_from_cov2(le.solve_tdoa(t, XYZ, C, sigma_t=np.full(4, 1.0e-3)).cov_xy)[0]
        assert 8 < a2 / a1 < 12

    def test_uncertainty_inflated_when_residuals_exceed_assumed_noise(self):
        XYZ, src = l_xyz(), np.array([12.0, 30.0])
        t = arrivals(src, XYZ, noise=1.0e-3, rng=np.random.default_rng(3))
        sol = le.solve_tdoa(t, XYZ, C, sigma_t=np.full(len(XYZ), 0.1e-3))
        assert sol.scale > 5.0
        assert le.mahalanobis_xy(sol, src) < 4.0


class TestRobustness:
    def test_outlier_rejected_with_redundancy(self):
        XYZ, src = l_xyz()[:5], np.array([12.0, 30.0])
        t = arrivals(src, XYZ)
        t[2] += 0.008
        sol = le.solve_tdoa(t, XYZ, C)
        assert sol.rejected == [2] and sol.weights[2] == 0.0
        assert pos_error(sol, src) < 1e-4
        assert abs(sol.residuals_s[2] - 0.008) < 1e-6

    def test_outlier_with_four_recordings_cannot_be_identified_but_is_flagged(self):
        # with 4 recordings (1 degree of freedom) a bad arrival is absorbed by moving the source;
        # the misfit must then show up as an inflated uncertainty that still covers the truth
        XYZ, src = square_xyz(20.0), np.array([4.0, 7.0])
        t = arrivals(src, XYZ)
        t[1] += 0.006
        sol = le.solve_tdoa(t, XYZ, C)
        assert sol.rejected == []  # cannot drop below 4 recordings
        assert sol.scale > 4.0
        assert le.mahalanobis_xy(sol, src) < 4.0

    def test_rejection_can_be_disabled(self):
        XYZ, src = l_xyz()[:5], np.array([12.0, 30.0])
        t = arrivals(src, XYZ)
        t[2] += 0.008
        sol = le.solve_tdoa(t, XYZ, C, reject_k=0.0)
        assert sol.rejected == [] and sol.scale > 4.0
        assert pos_error(sol, src) > pos_error(le.solve_tdoa(t, XYZ, C), src)

    def test_leverage_outlier_on_l_array(self):
        # the classic failure of plain M-estimators: one bad arrival, everything else perfect
        XYZ, src = l_xyz(), np.array([12.0, 30.0])
        for bad in range(6):
            t = arrivals(src, XYZ)
            t[bad] += 0.006
            sol = le.solve_tdoa(t, XYZ, C)
            assert sol.rejected == [bad], (bad, sol.rejected, sol.residuals_s)
            assert pos_error(sol, src) < 1e-4

    def test_no_false_rejection_on_noisy_data(self):
        XYZ, src = l_xyz(), np.array([12.0, 30.0])
        n_rej = 0
        for seed in range(30):
            t = arrivals(src, XYZ, noise=0.3e-3, rng=np.random.default_rng(seed))
            sol = le.solve_tdoa(t, XYZ, C, sigma_t=np.full(6, 0.3e-3))
            n_rej += len(sol.rejected)
        assert n_rej <= 1


class TestClockPrior:
    def test_prior_mode_recovers_position(self):
        XYZ, src = l_xyz(), np.array([12.0, 30.0])
        offs = np.array([0.0, 0.002, -0.001, 0.0015, -0.0005, 0.001])
        t = arrivals(src, XYZ, offsets=offs)
        sol = le.solve_tdoa(t, XYZ, C, sigma_t=np.full(6, 0.2e-3), clock_sigma=0.002)
        assert sol.estimate_offsets and len(sol.delta) == 6
        assert pos_error(sol, src) < 0.6
        assert le.mahalanobis_xy(sol, src) < 3.0
        assert np.corrcoef(sol.delta, offs)[0, 1] > 0.8  # shrunk toward zero but ordered correctly
        assert sol.dof == 3

    def test_synced_model_on_offset_data_inflates_uncertainty(self):
        XYZ, src = l_xyz(), np.array([12.0, 30.0])
        offs = np.array([0.0, 0.002, -0.001, 0.0015, -0.0005, 0.001])
        t = arrivals(src, XYZ, offsets=offs)
        sol = le.solve_tdoa(t, XYZ, C, sigma_t=np.full(6, 0.2e-3))
        assert sol.scale > 4.0 and le.mahalanobis_xy(sol, src) < 3.5

    def test_prior_zero_means_synchronised(self):
        XYZ, src = square_xyz(20.0), np.array([4.0, 7.0])
        sol = le.solve_tdoa(arrivals(src, XYZ), XYZ, C, clock_sigma=0.0)
        assert not sol.estimate_offsets and sol.cov.shape == (3, 3)


class TestGeometry:
    def test_linear_array_reports_mirror_ambiguity(self):
        XYZ, src = linear_xyz(5, 20.0), np.array([8.3, 55.3])
        t = arrivals(src, XYZ, noise=0.1e-3, rng=np.random.default_rng(1))
        sol = le.solve_tdoa(t, XYZ, C, sigma_t=np.full(5, 0.1e-3))
        assert sol.ambiguous and len(sol.alternatives) >= 1
        assert nearest_solution_error(sol, src) < 0.2
        mirror = min(sol.alternatives + [{"x": sol.s_xy[0], "y": sol.s_xy[1]}], key=lambda a: abs(a["x"] + 8.3))
        assert abs(mirror["x"] + 8.3) < 0.3 and abs(mirror["y"] - 55.3) < 0.3

    def test_square_array_not_ambiguous(self):
        XYZ, src = square_xyz(20.0), np.array([4.0, 7.0])
        sol = le.solve_tdoa(arrivals(src, XYZ, noise=0.1e-3), XYZ, C, sigma_t=np.full(4, 0.1e-3))
        assert not sol.ambiguous

    def test_grid_search_init_finds_basin(self):
        XYZ, src = square_xyz(20.0), np.array([4.0, 7.0])
        t = arrivals(src, XYZ)
        picks = le.grid_search_init(t, XYZ, C, np.ones(4), 0.0, (-50, 70, -50, 70), 1.0, n_best=4)
        assert len(picks) >= 1
        assert np.linalg.norm(picks[0][0] - src) <= 1.0
        assert picks[0][1] <= picks[-1][1]


class TestValidation:
    def test_errors(self):
        XYZ = square_xyz(20.0)
        with pytest.raises(le.LocatorError):
            le.solve_tdoa([1.0, 1.0], XYZ[:2], C)
        with pytest.raises(le.LocatorError):
            le.solve_tdoa([1.0, 1.0, 1.0], XYZ, C)
        with pytest.raises(le.LocatorError):
            le.solve_tdoa([1.0, np.nan, 1.0, 1.0], XYZ, C)
        with pytest.raises(le.LocatorError):
            le.solve_tdoa([1.0, 1.0, 1.0, 1.0], XYZ, C, sigma_t=[1e-3, -1e-3, 1e-3, 1e-3])
        with pytest.raises(le.LocatorError):
            le.solve_tdoa([1.0, 1.0, 1.0, 1.0], XYZ[:, :2], C)


class TestEllipse:
    def test_isotropic(self):
        from scipy.stats import chi2
        a, b, _ = le.ellipse_from_cov2(np.eye(2) * 4.0)
        assert abs(a - 2 * math.sqrt(chi2.ppf(0.95, 2))) < 1e-9 and abs(b - a) < 1e-9

    def test_orientation(self):
        R = np.array([[math.cos(0.5), -math.sin(0.5)], [math.sin(0.5), math.cos(0.5)]])
        cov = R @ np.diag([9.0, 1.0]) @ R.T
        a, b, ang = le.ellipse_from_cov2(cov)
        assert abs(a / b - 3.0) < 1e-9
        assert min(abs(ang - math.degrees(0.5)), abs(ang - math.degrees(0.5) + 180), abs(ang - math.degrees(0.5) - 180)) < 1e-6

    def test_mahalanobis_zero_at_estimate(self):
        XYZ, src = square_xyz(20.0), np.array([4.0, 7.0])
        sol = le.solve_tdoa(arrivals(src, XYZ), XYZ, C)
        assert le.mahalanobis_xy(sol, sol.s_xy) == 0.0
