#!/usr/bin/env python3
"""
Test suite for time-dependent warp profiles
"""

import pytest
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from stress_energy.exotic_matter_profile import ExoticMatterProfiler


class TestTimeDependentProfiles:
    """Test cases for time-dependent warp bubble profiles."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.profiler = ExoticMatterProfiler(r_min=0.1, r_max=3.0, n_points=50)
    
    def test_time_dependent_profile_basic(self):
        """Test basic time-dependent profile computation."""
        # Simple linear trajectory R(t) = R0 + v*t
        R0, v = 1.0, 0.1
        R_func = lambda t: R0 + v * t
        sigma = 0.5
        
        # Test at different times
        times = [0.0, 1.0, 2.0]
        
        for t in times:
            profile = self.profiler.alcubierre_profile_time_dep(
                self.profiler.r_array, t, R_func, sigma
            )
            
            # Profile should be finite and real
            assert np.all(np.isfinite(profile))
            assert np.all(np.isreal(profile))
            
            # Profile should be bounded (roughly between 0 and 1 for well-behaved cases)
            assert np.all(profile >= -2.0)  # Allow some negative excursion
            assert np.all(profile <= 2.0)   # Allow some positive excursion
    
    def test_time_dependent_profile_trajectory_consistency(self):
        """Test consistency across time trajectory."""
        R0, v = 1.5, 0.05
        R_func = lambda t: R0 + v * t
        sigma = 0.4
        
        times = np.linspace(0, 5, 6)
        profiles = []
        
        for t in times:
            profile = self.profiler.alcubierre_profile_time_dep(
                self.profiler.r_array, t, R_func, sigma
            )
            profiles.append(profile)
        
        profiles = np.array(profiles)
        
        # All profiles should be finite
        assert np.all(np.isfinite(profiles))
        
        # Profiles should evolve smoothly (no sudden jumps)
        for i in range(1, len(times)):
            diff = np.abs(profiles[i] - profiles[i-1])
            max_diff = np.max(diff)
            assert max_diff < 1.0, f"Profile changed too rapidly between t={times[i-1]} and t={times[i]}"
    
    def test_time_dependent_T00_computation(self):
        """Test time-dependent T^{00} computation."""
        R0, v = 1.0, 0.1
        R_func = lambda t: R0 + v * t
        sigma = 0.6
        
        times = np.array([0.0, 0.5, 1.0])
        
        r_array, T00_rt = self.profiler.compute_T00_profile_time_dep(
            R_func, sigma, times
        )
        
        # Check array shapes
        assert len(r_array) == self.profiler.n_points
        assert T00_rt.shape == (len(times), self.profiler.n_points)
        
        # T00 should be finite (may have some extreme values near singularities)
        finite_mask = np.isfinite(T00_rt)
        assert np.sum(finite_mask) > 0.5 * T00_rt.size, "Most T00 values should be finite"
    
    def test_stationary_limit(self):
        """Test that stationary trajectory gives consistent results."""
        # Stationary bubble: R(t) = constant
        R_const = 1.5
        R_func = lambda t: R_const
        sigma = 0.5
        
        times = np.array([0.0, 1.0, 2.0])
        
        profiles = []
        for t in times:
            profile = self.profiler.alcubierre_profile_time_dep(
                self.profiler.r_array, t, R_func, sigma
            )
            profiles.append(profile)
        
        profiles = np.array(profiles)
        
        # All profiles should be essentially identical for stationary case
        for i in range(1, len(times)):
            diff = np.abs(profiles[i] - profiles[0])
            max_diff = np.max(diff)
            assert max_diff < 1e-10, f"Stationary profile should not change with time"
    
    def test_edge_cases(self):
        """Test edge cases for time-dependent profiles."""
        sigma = 0.5
        
        # Test R(t) = 0 case
        R_zero = lambda t: 1e-12  # Very small radius
        profile_zero = self.profiler.alcubierre_profile_time_dep(
            self.profiler.r_array, 0.0, R_zero, sigma
        )
        # Should return flat profile (approximately 1)
        assert np.allclose(profile_zero, 1.0, atol=0.1)
        
        # Test very large R(t)
        R_large = lambda t: 100.0
        profile_large = self.profiler.alcubierre_profile_time_dep(
            self.profiler.r_array, 0.0, R_large, sigma
        )
        assert np.all(np.isfinite(profile_large))
        
        # Test negative time (should still work)
        R_func = lambda t: 1.0 + 0.1 * t
        profile_neg = self.profiler.alcubierre_profile_time_dep(
            self.profiler.r_array, -1.0, R_func, sigma
        )
        assert np.all(np.isfinite(profile_neg))
    
    def test_acceleration_trajectory(self):
        """Test accelerating bubble trajectory."""
        # Quadratic trajectory: R(t) = R0 + v0*t + 0.5*a*t^2
        R0, v0, a = 1.0, 0.1, 0.05
        R_func = lambda t: R0 + v0 * t + 0.5 * a * t**2
        sigma = 0.4
        
        times = np.linspace(0, 2, 5)
        
        profiles = []
        for t in times:
            profile = self.profiler.alcubierre_profile_time_dep(
                self.profiler.r_array, t, R_func, sigma
            )
            profiles.append(profile)
        
        profiles = np.array(profiles)
        
        # Should produce valid profiles
        assert np.all(np.isfinite(profiles))
        
        # Check that bubble radius is increasing as expected
        R_values = [R_func(t) for t in times]
        assert np.all(np.diff(R_values) > 0), "Radius should be increasing for accelerating trajectory"


class TestTimeDependentIntegration:
    """Test integration of time-dependent profiles with other components."""
    
    def setup_method(self):
        """Setup integration test fixtures."""
        self.profiler = ExoticMatterProfiler(r_min=0.1, r_max=2.0, n_points=30)
    
    def test_time_dependent_with_coil_optimization(self):
        """Test that time-dependent profiles can be used for coil optimization."""
        # This is more of an integration test
        R_func = lambda t: 1.0 + 0.1 * t
        sigma = 0.5
        
        # Generate time-dependent profile at t=1.0
        r_array, T00_rt = self.profiler.compute_T00_profile_time_dep(
            R_func, sigma, np.array([1.0])
        )
        
        T00_t1 = T00_rt[0, :]  # Extract profile at t=1.0
        
        # Check that we get reasonable stress-energy values
        finite_mask = np.isfinite(T00_t1)
        assert np.sum(finite_mask) > len(T00_t1) // 2, "Should have reasonable number of finite T00 values"
        
        if np.any(finite_mask):
            T00_finite = T00_t1[finite_mask]
            assert len(T00_finite) > 0, "Should have some finite T00 values"
    
    def test_multiple_time_snapshots(self):
        """Test computation of multiple time snapshots."""
        R_func = lambda t: 1.2 + 0.05 * t
        sigma = 0.3
        
        times = np.linspace(0, 3, 4)
        
        r_array, T00_rt = self.profiler.compute_T00_profile_time_dep(
            R_func, sigma, times
        )
        
        # Should get array with correct dimensions
        assert T00_rt.shape[0] == len(times)
        assert T00_rt.shape[1] == len(r_array)
        
        # Each time slice should have some finite values
        for i, t in enumerate(times):
            T00_slice = T00_rt[i, :]
            finite_count = np.sum(np.isfinite(T00_slice))
            assert finite_count > 0, f"Time slice {i} (t={t}) should have finite T00 values"


def test_time_dependent_profile_module_import():
    """Test that time-dependent functionality is properly accessible."""
    profiler = ExoticMatterProfiler(r_min=0.1, r_max=1.0, n_points=10)
    
    # Should have time-dependent methods
    assert hasattr(profiler, 'alcubierre_profile_time_dep')
    assert hasattr(profiler, 'compute_T00_profile_time_dep')
    assert callable(profiler.alcubierre_profile_time_dep)
    assert callable(profiler.compute_T00_profile_time_dep)


def test_time_dependent_numerical_stability():
    """Test numerical stability of time-dependent computations."""
    profiler = ExoticMatterProfiler(r_min=0.1, r_max=1.0, n_points=20)
    
    # Test with various parameter combinations that might cause instability
    test_cases = [
        (lambda t: 0.5, 0.1),      # Small sigma
        (lambda t: 0.5, 2.0),      # Large sigma  
        (lambda t: 0.1, 0.5),      # Small R
        (lambda t: 2.0, 0.5),      # Large R
        (lambda t: 0.5 + 0.01*t, 0.5),  # Slow evolution
    ]
    
    for R_func, sigma in test_cases:
        try:
            profile = profiler.alcubierre_profile_time_dep(
                profiler.r_array, 1.0, R_func, sigma
            )
            
            # Should not contain NaN or inf
            nan_count = np.sum(np.isnan(profile))
            inf_count = np.sum(np.isinf(profile))
            
            assert nan_count == 0, f"Profile should not contain NaN values"
            assert inf_count == 0, f"Profile should not contain Inf values"
            
        except Exception as e:
            pytest.fail(f"Time-dependent profile computation failed for R_func={R_func}, sigma={sigma}: {e}")
