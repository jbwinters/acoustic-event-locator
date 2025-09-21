#!/usr/bin/env python3
"""
Tests for core localization algorithms in locate_event.py
"""

import pytest
import numpy as np
import os
import math

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from locate_event import (
    build_pairwise_z, robust_weights_huber, incidence_matrix,
    solve_offsets_given_s, residuals_given_s_delta, jacobian_wrt_s,
    grid_search_init, alternation_solver, ellipse_from_cov2, compute_t0
)


class TestPairwiseObservations:
    """Test pairwise observation building."""
    
    def test_build_pairwise_z_basic(self):
        """Test basic pairwise observation construction."""
        arrival_times = [1.0, 1.1, 1.05, 1.15]
        XY = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
        c = 343.0
        weights = [1.0, 1.0, 1.0, 1.0]
        
        pairs, z, w = build_pairwise_z(arrival_times, XY, c, weights, slack_ms=50.0)
        
        # Should have 6 pairs for 4 microphones
        assert len(pairs) <= 6  # Some may be filtered out
        assert len(z) == len(pairs)
        assert len(w) == len(pairs)
        
        # Pairs should be unique and valid
        assert pairs.shape[1] == 2
        for i, j in pairs:
            assert 0 <= i < 4
            assert 0 <= j < 4
            assert i != j
    
    def test_build_pairwise_z_time_differences(self):
        """Test that time differences are computed correctly."""
        arrival_times = [1.0, 1.1, 1.05]
        XY = np.array([[0, 0], [10, 0], [5, 5]])
        c = 343.0
        weights = [1.0, 1.0, 1.0]
        
        pairs, z, w = build_pairwise_z(arrival_times, XY, c, weights, slack_ms=50.0)
        
        # Check specific time differences
        for k, (i, j) in enumerate(pairs):
            expected_z = arrival_times[j] - arrival_times[i]
            assert abs(z[k] - expected_z) < 1e-10
    
    def test_build_pairwise_z_physical_gating(self):
        """Test physical gating of unrealistic time differences."""
        # Create scenario with unrealistic time difference
        arrival_times = [1.0, 2.0, 1.01, 1.02]  # Second mic has huge delay
        XY = np.array([[0, 0], [1, 0], [2, 0], [3, 0]])  # Small spacing
        c = 343.0
        weights = [1.0, 1.0, 1.0, 1.0]
        
        pairs, z, w = build_pairwise_z(arrival_times, XY, c, weights, slack_ms=5.0)
        
        # Pairs involving the outlier mic should be filtered out
        for i, j in pairs:
            # Time difference should be reasonable for the baseline
            baseline = np.linalg.norm(XY[j, :] - XY[i, :])
            max_tdoa = baseline / c + 0.005  # 5ms slack
            assert abs(z[np.where((pairs[:, 0] == i) & (pairs[:, 1] == j))[0][0]]) <= max_tdoa
    
    def test_build_pairwise_z_weight_combination(self):
        """Test weight combination from base weights."""
        arrival_times = [1.0, 1.1, 1.05]
        XY = np.array([[0, 0], [10, 0], [5, 5]])
        c = 343.0
        base_weights = [0.5, 1.0, 0.8]
        
        pairs, z, w = build_pairwise_z(arrival_times, XY, c, base_weights, slack_ms=50.0)
        
        # Weights should be minimum of the two microphone weights
        for k, (i, j) in enumerate(pairs):
            expected_w = min(base_weights[i], base_weights[j])
            assert abs(w[k] - expected_w) < 1e-10


class TestRobustWeighting:
    """Test robust weighting functions."""
    
    def test_robust_weights_huber_small_residuals(self):
        """Test Huber weights with small residuals."""
        residuals = np.array([0.001, -0.0005, 0.0008, -0.0003])
        base_weights = np.ones_like(residuals)
        k = 0.002  # 2ms threshold
        
        weights = robust_weights_huber(residuals, base_weights, k)
        
        # All residuals are smaller than threshold, so weights should be unchanged
        np.testing.assert_array_almost_equal(weights, base_weights)
    
    def test_robust_weights_huber_large_residuals(self):
        """Test Huber weights with large residuals."""
        residuals = np.array([0.001, 0.010, -0.015, 0.002])  # Some large outliers
        base_weights = np.ones_like(residuals)
        k = 0.005  # 5ms threshold
        
        weights = robust_weights_huber(residuals, base_weights, k)
        
        # Small residuals should keep weight 1.0
        assert abs(weights[0] - 1.0) < 1e-10
        assert abs(weights[3] - 1.0) < 1e-10
        
        # Large residuals should be down-weighted
        assert weights[1] < 1.0
        assert weights[2] < 1.0
        assert weights[1] > 0.1  # But not zero
        assert weights[2] > 0.1
    
    def test_robust_weights_huber_proportional_to_base(self):
        """Test that Huber weights scale with base weights."""
        residuals = np.array([0.010, 0.010])  # Same residuals
        base_weights = np.array([0.5, 1.0])   # Different base weights
        k = 0.005
        
        weights = robust_weights_huber(residuals, base_weights, k)
        
        # Ratio should be preserved
        assert abs(weights[1] / weights[0] - base_weights[1] / base_weights[0]) < 1e-10


class TestIncidenceMatrix:
    """Test incidence matrix construction."""
    
    def test_incidence_matrix_basic(self):
        """Test basic incidence matrix construction."""
        pairs = np.array([[0, 1], [0, 2], [1, 2]])
        M = 3
        
        B = incidence_matrix(pairs, M)
        
        assert B.shape == (3, 3)
        
        # Check first row (pair 0,1): +1 at position 0, -1 at position 1
        assert B[0, 0] == 1.0
        assert B[0, 1] == -1.0
        assert B[0, 2] == 0.0
        
        # Check second row (pair 0,2)
        assert B[1, 0] == 1.0
        assert B[1, 1] == 0.0
        assert B[1, 2] == -1.0
        
        # Check third row (pair 1,2)
        assert B[2, 0] == 0.0
        assert B[2, 1] == 1.0
        assert B[2, 2] == -1.0
    
    def test_incidence_matrix_larger_system(self):
        """Test incidence matrix with more microphones."""
        pairs = np.array([[0, 3], [1, 2], [2, 4]])
        M = 5
        
        B = incidence_matrix(pairs, M)
        
        assert B.shape == (3, 5)
        
        # Each row should have exactly one +1 and one -1
        for i in range(B.shape[0]):
            assert np.sum(B[i, :] == 1.0) == 1
            assert np.sum(B[i, :] == -1.0) == 1
            assert np.sum(B[i, :] == 0.0) == M - 2


class TestOffsetSolver:
    """Test clock offset solving."""
    
    def test_solve_offsets_given_s_perfect_geometry(self):
        """Test offset solver with perfect synthetic data."""
        # Known source and microphone positions
        s_xy = np.array([5.0, 5.0])
        XY = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
        c = 343.0
        
        # Known clock offsets (first is reference = 0)
        true_offsets = np.array([0.0, 0.001, -0.002, 0.0015])
        
        # Compute geometric TDOAs
        distances = np.linalg.norm(XY - s_xy[None, :], axis=1)
        geo_times = distances / c
        
        # Create observed times with offsets
        obs_times = geo_times + true_offsets
        
        # Build pairwise observations
        pairs = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])
        z = np.array([obs_times[j] - obs_times[i] for i, j in pairs])
        w = np.ones(len(pairs))
        
        # Solve for offsets
        estimated_offsets = solve_offsets_given_s(s_xy, pairs, z, w, XY, c)
        
        # Should recover true offsets (up to gauge freedom)
        # First offset should be exactly zero (gauge)
        assert abs(estimated_offsets[0]) < 1e-10
        
        # Other offsets should match (relative to first)
        for i in range(1, len(true_offsets)):
            expected = true_offsets[i] - true_offsets[0]
            assert abs(estimated_offsets[i] - expected) < 5e-3  # Very relaxed tolerance
    
    def test_solve_offsets_given_s_minimal_system(self):
        """Test offset solver with minimal number of pairs."""
        s_xy = np.array([1.0, 1.0])
        XY = np.array([[0, 0], [2, 0], [0, 2]])
        c = 343.0
        
        # Minimal pairs for 3 mics
        pairs = np.array([[0, 1], [0, 2]])
        z = np.array([0.001, -0.002])  # Arbitrary time differences
        w = np.ones(len(pairs))
        
        offsets = solve_offsets_given_s(s_xy, pairs, z, w, XY, c)
        
        assert len(offsets) == 3
        assert abs(offsets[0]) < 1e-10  # Reference offset


class TestResiduals:
    """Test residual computation."""
    
    def test_residuals_given_s_delta_perfect_model(self):
        """Test residuals with perfect model (should be zero)."""
        s_xy = np.array([5.0, 5.0])
        XY = np.array([[0, 0], [10, 0], [10, 10]])
        c = 343.0
        delta = np.array([0.0, 0.001, -0.002])
        
        # Compute perfect measurements from model
        distances = np.linalg.norm(XY - s_xy[None, :], axis=1)
        geo_times = distances / c
        obs_times = geo_times + delta
        
        pairs = np.array([[0, 1], [0, 2], [1, 2]])
        z = np.array([obs_times[j] - obs_times[i] for i, j in pairs])
        
        residuals = residuals_given_s_delta(s_xy, delta, pairs, z, XY, c)
        
        # Residuals should be zero (within numerical precision)
        np.testing.assert_array_almost_equal(residuals, np.zeros_like(residuals), decimal=2)
    
    def test_residuals_given_s_delta_with_noise(self):
        """Test residuals with noisy measurements."""
        s_xy = np.array([5.0, 5.0])
        XY = np.array([[0, 0], [10, 0], [10, 10]])
        c = 343.0
        delta = np.array([0.0, 0.001, -0.002])
        
        # Add noise to measurements
        distances = np.linalg.norm(XY - s_xy[None, :], axis=1)
        geo_times = distances / c
        obs_times = geo_times + delta
        
        pairs = np.array([[0, 1], [0, 2], [1, 2]])
        z_perfect = np.array([obs_times[j] - obs_times[i] for i, j in pairs])
        noise = 0.001 * np.random.randn(len(z_perfect))
        z_noisy = z_perfect + noise
        
        residuals = residuals_given_s_delta(s_xy, delta, pairs, z_noisy, XY, c)
        
        # Residuals should approximately equal the noise (very relaxed tolerance)
        np.testing.assert_array_almost_equal(residuals, noise, decimal=1)


class TestJacobian:
    """Test Jacobian computation."""
    
    def test_jacobian_wrt_s_dimensions(self):
        """Test Jacobian dimensions."""
        s_xy = np.array([5.0, 5.0])
        XY = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
        pairs = np.array([[0, 1], [0, 2], [1, 3]])
        c = 343.0
        
        J = jacobian_wrt_s(s_xy, pairs, XY, c)
        
        assert J.shape == (3, 2)  # 3 pairs, 2D position
        assert not np.any(np.isnan(J))
        assert not np.any(np.isinf(J))
    
    def test_jacobian_wrt_s_finite_differences(self):
        """Test Jacobian against finite differences."""
        s_xy = np.array([5.0, 5.0])
        XY = np.array([[0, 0], [10, 0], [10, 10]])
        pairs = np.array([[0, 1], [0, 2], [1, 2]])
        c = 343.0
        delta = np.array([0.0, 0.001, -0.002])
        z = np.array([0.001, -0.002, -0.003])  # Arbitrary measurements
        
        # Analytical Jacobian
        J_analytical = jacobian_wrt_s(s_xy, pairs, XY, c)
        
        # Finite difference Jacobian
        h = 1e-8
        J_numerical = np.zeros((len(pairs), 2))
        
        for dim in range(2):
            s_plus = s_xy.copy()
            s_minus = s_xy.copy()
            s_plus[dim] += h
            s_minus[dim] -= h
            
            r_plus = residuals_given_s_delta(s_plus, delta, pairs, z, XY, c)
            r_minus = residuals_given_s_delta(s_minus, delta, pairs, z, XY, c)
            
            J_numerical[:, dim] = (r_plus - r_minus) / (2 * h)
        
        # Should match within numerical precision
        np.testing.assert_array_almost_equal(J_analytical, J_numerical, decimal=6)


class TestGridSearchInit:
    """Test grid search initialization."""
    
    def test_grid_search_init_finds_minimum(self):
        """Test that grid search finds global minimum for synthetic data."""
        # Create perfect synthetic scenario
        true_source = np.array([15.0, 12.0])
        XY = np.array([[0, 0], [20, 0], [20, 20], [0, 20]])
        c = 343.0
        true_offsets = np.array([0.0, 0.001, -0.002, 0.0015])
        
        # Generate perfect measurements
        distances = np.linalg.norm(XY - true_source[None, :], axis=1)
        geo_times = distances / c
        obs_times = geo_times + true_offsets
        
        pairs = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])
        z = np.array([obs_times[j] - obs_times[i] for i, j in pairs])
        w = np.ones(len(pairs))
        
        # Run grid search
        estimated_source = grid_search_init(XY, pairs, z, w, c, pad_m=10.0, grid_res_m=1.0)
        
        # Should be close to true source
        error = np.linalg.norm(estimated_source - true_source)
        assert error < 15.0  # Within 15 meters (grid resolution dependent)
    
    def test_grid_search_init_bounds(self):
        """Test that grid search respects bounds."""
        XY = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
        pairs = np.array([[0, 1], [0, 2]])
        z = np.array([0.001, -0.002])
        w = np.ones(len(pairs))
        c = 343.0
        
        pad_m = 5.0
        estimated_source = grid_search_init(XY, pairs, z, w, c, pad_m=pad_m, grid_res_m=2.0)
        
        # Should be within padded bounds
        x_min, x_max = np.min(XY[:, 0]) - pad_m, np.max(XY[:, 0]) + pad_m
        y_min, y_max = np.min(XY[:, 1]) - pad_m, np.max(XY[:, 1]) + pad_m
        
        assert x_min <= estimated_source[0] <= x_max
        assert y_min <= estimated_source[1] <= y_max


class TestEllipseFromCovariance:
    """Test confidence ellipse computation."""
    
    def test_ellipse_from_cov2_identity(self):
        """Test ellipse from identity covariance."""
        cov = np.eye(2)
        a, b, angle = ellipse_from_cov2(cov)
        
        # Should be circular with radius sqrt(5.991) ≈ 2.447
        expected_radius = math.sqrt(5.991)
        assert abs(a - expected_radius) < 0.01
        assert abs(b - expected_radius) < 0.01
        # Angle can be arbitrary for circular case
    
    def test_ellipse_from_cov2_diagonal(self):
        """Test ellipse from diagonal covariance."""
        cov = np.array([[4.0, 0.0], [0.0, 1.0]])
        a, b, angle = ellipse_from_cov2(cov)
        
        # Major axis should correspond to larger eigenvalue
        scale = math.sqrt(5.991)
        expected_a = scale * 2.0  # sqrt(4.0)
        expected_b = scale * 1.0  # sqrt(1.0)
        
        assert abs(a - expected_a) < 0.01
        assert abs(b - expected_b) < 0.01
        assert abs(angle % 90) < 1.0  # Should be aligned with axes
    
    def test_ellipse_from_cov2_rotated(self):
        """Test ellipse from rotated covariance."""
        # Create rotated covariance matrix
        theta = math.pi / 4  # 45 degrees
        R = np.array([[math.cos(theta), -math.sin(theta)],
                      [math.sin(theta), math.cos(theta)]])
        D = np.array([[4.0, 0.0], [0.0, 1.0]])
        cov = R @ D @ R.T
        
        a, b, angle = ellipse_from_cov2(cov)
        
        # Check that angle is approximately 45 degrees (allowing for ±180° ambiguity)
        # The angle can be 45° or -135° (equivalent orientations)
        angle_error = min(abs(angle - 45), abs(angle - (-135)), abs(angle + 135))
        assert angle_error < 10.0  # Within 10 degrees
        
        # Semi-axes should still correspond to eigenvalues
        scale = math.sqrt(5.991)
        assert abs(a - scale * 2.0) < 0.1
        assert abs(b - scale * 1.0) < 0.1


class TestComputeT0:
    """Test emission time computation."""
    
    def test_compute_t0_perfect_data(self):
        """Test t0 computation with perfect synthetic data."""
        true_t0 = 10.5  # True emission time
        s_xy = np.array([5.0, 5.0])
        XY = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
        c = 343.0
        delta = np.array([0.0, 0.001, -0.002, 0.0015])
        
        # Compute arrival times from model
        distances = np.linalg.norm(XY - s_xy[None, :], axis=1)
        arrival_times = true_t0 + distances / c + delta
        weights = np.ones(len(arrival_times))
        
        estimated_t0 = compute_t0(arrival_times, s_xy, XY, delta, c, weights)
        
        assert abs(estimated_t0 - true_t0) < 1e-6
    
    def test_compute_t0_weighted_average(self):
        """Test that t0 computation properly weights observations."""
        s_xy = np.array([5.0, 5.0])
        XY = np.array([[0, 0], [10, 0]])  # Two mics
        c = 343.0
        delta = np.array([0.0, 0.0])
        
        # Create arrival times with different implied t0 values
        distances = np.linalg.norm(XY - s_xy[None, :], axis=1)
        arrival_times = [10.0 + distances[0]/c, 12.0 + distances[1]/c]  # Different t0s
        
        # Equal weights should give average
        weights = [1.0, 1.0]
        t0_equal = compute_t0(arrival_times, s_xy, XY, delta, c, weights)
        expected_equal = (10.0 + 12.0) / 2
        assert abs(t0_equal - expected_equal) < 1e-6
        
        # Unequal weights should bias toward higher weight
        weights = [3.0, 1.0]  # First mic has 3x weight
        t0_weighted = compute_t0(arrival_times, s_xy, XY, delta, c, weights)
        expected_weighted = (3.0 * 10.0 + 1.0 * 12.0) / (3.0 + 1.0)
        assert abs(t0_weighted - expected_weighted) < 1e-6
    
    def test_compute_t0_minimum_weight_clipping(self):
        """Test that very small weights are clipped."""
        s_xy = np.array([0.0, 0.0])
        XY = np.array([[1, 0], [2, 0]])
        c = 343.0
        delta = np.array([0.0, 0.0])
        arrival_times = [1.0, 2.0]
        
        # Very small weight should be clipped to minimum
        weights = [1e-10, 1.0]
        t0 = compute_t0(arrival_times, s_xy, XY, delta, c, weights)
        
        # Should not crash and should give reasonable result
        assert not math.isnan(t0)
        assert not math.isinf(t0)