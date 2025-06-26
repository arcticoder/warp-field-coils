#!/usr/bin/env python3
"""
Test suite for exotic matter profile computation
"""

import pytest
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from stress_energy.exotic_matter_profile import ExoticMatterProfiler, alcubierre_profile, gaussian_warp_profile

class TestExoticMatterProfiler:
    """Test cases for ExoticMatterProfiler class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.profiler = ExoticMatterProfiler(r_min=0.1, r_max=5.0, n_points=100)
    
    def test_initialization(self):
        """Test profiler initialization."""
        assert self.profiler.r_min == 0.1
        assert self.profiler.r_max == 5.0
        assert self.profiler.n_points == 100
        assert len(self.profiler.r_array) == 100
        assert np.isclose(self.profiler.r_array[0], 0.1)
        assert np.isclose(self.profiler.r_array[-1], 5.0)
    
    def test_metric_setup(self):
        """Test metric tensor setup."""
        assert self.profiler.g.shape == (4, 4)
        assert self.profiler.g_inv.shape == (4, 4)
        
        # Check metric signature (-,+,+,+)
        assert self.profiler.g[0, 0] == -1  # -dt²
        assert self.profiler.g[2, 2] == self.profiler.r**2  # r²dθ²
    
    def test_alcubierre_profile_function(self):
        """Test Alcubierre warp profile function."""
        r_test = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
        R = 1.0
        sigma = 0.1
        
        profile_vals = [alcubierre_profile(r, R, sigma) for r in r_test]
        
        # Check boundary conditions
        assert profile_vals[0] == 1.0  # r < R-σ should give 1
        assert profile_vals[-1] == 0.0  # r > R+σ should give 0
        
        # Check continuity (values should be between 0 and 1)
        for val in profile_vals:
            assert 0.0 <= val <= 1.0
    
    def test_gaussian_profile_function(self):
        """Test Gaussian warp profile function."""
        r_test = np.array([0.0, 0.5, 1.0, 2.0, 5.0])
        A = 1.0
        sigma = 1.0
        
        profile_vals = [gaussian_warp_profile(r, A, sigma) for r in r_test]
        
        # Check peak at r=0
        assert np.isclose(profile_vals[0], A)
        
        # Check decay with distance
        assert profile_vals[1] > profile_vals[2] > profile_vals[3] > profile_vals[4]
        
        # Check all values are positive
        for val in profile_vals:
            assert val >= 0.0
    
    def test_compute_T00_profile(self):
        """Test T^{00} profile computation."""
        # Use simple Gaussian profile
        profile_func = lambda r: gaussian_warp_profile(r, A=0.5, sigma=1.0)
        
        r_array, T00_profile = self.profiler.compute_T00_profile(profile_func)
        
        # Check output shapes
        assert len(r_array) == self.profiler.n_points
        assert len(T00_profile) == self.profiler.n_points
        assert np.array_equal(r_array, self.profiler.r_array)
        
        # Check that profile is finite
        assert np.all(np.isfinite(T00_profile))
    
    def test_identify_exotic_regions(self):
        """Test exotic matter region identification."""
        # Create test profile with negative regions
        T00_test = np.array([0.1, -0.2, -0.3, 0.1, 0.2])
        
        exotic_info = self.profiler.identify_exotic_regions(T00_test)
        
        assert exotic_info['has_exotic'] == True
        assert len(exotic_info['exotic_r']) == 2  # Two negative values
        assert len(exotic_info['exotic_T00']) == 2
        assert exotic_info['total_exotic_energy'] > 0  # Should be positive (magnitude)
        
        # Test case with no exotic matter
        T00_positive = np.array([0.1, 0.2, 0.3, 0.1, 0.2])
        exotic_info_pos = self.profiler.identify_exotic_regions(T00_positive)
        
        assert exotic_info_pos['has_exotic'] == False
        assert len(exotic_info_pos['exotic_r']) == 0
    
    def test_numerical_derivatives(self):
        """Test numerical derivative calculations."""
        # Test function: f(x) = x²
        test_func = lambda x: x**2
        x_test = 2.0
        
        # Test first derivative (should be ≈ 2x = 4)
        df_dx = self.profiler._numerical_derivative(test_func, x_test)
        assert np.isclose(df_dx, 4.0, rtol=1e-4)
        
        # Test second derivative (should be ≈ 2)
        d2f_dx2 = self.profiler._numerical_second_derivative(test_func, x_test)
        assert np.isclose(d2f_dx2, 2.0, rtol=1e-3)

class TestWarpProfiles:
    """Test cases for warp profile functions."""
    
    def test_alcubierre_profile_properties(self):
        """Test mathematical properties of Alcubierre profile."""
        R = 2.0
        sigma = 0.3
        
        # Test values inside, on boundary, and outside
        r_inside = R - 2*sigma
        r_boundary = R
        r_outside = R + 2*sigma
        
        f_inside = alcubierre_profile(r_inside, R, sigma)
        f_boundary = alcubierre_profile(r_boundary, R, sigma)
        f_outside = alcubierre_profile(r_outside, R, sigma)
        
        assert f_inside == 1.0
        assert 0.0 < f_boundary < 1.0  # Smooth transition
        assert f_outside == 0.0
    
    def test_gaussian_profile_scaling(self):
        """Test Gaussian profile amplitude and width scaling."""
        r = 1.0
        
        # Test amplitude scaling
        A1, A2 = 1.0, 2.0
        sigma = 1.0
        
        f1 = gaussian_warp_profile(r, A1, sigma)
        f2 = gaussian_warp_profile(r, A2, sigma)
        
        assert np.isclose(f2, 2.0 * f1)
        
        # Test width scaling
        A = 1.0
        sigma1, sigma2 = 0.5, 1.0
        
        g1 = gaussian_warp_profile(r, A, sigma1)
        g2 = gaussian_warp_profile(r, A, sigma2)
        
        assert g1 < g2  # Narrower profile should have smaller value at r=1

@pytest.fixture
def sample_profiler():
    """Fixture providing a configured profiler."""
    return ExoticMatterProfiler(r_min=0.5, r_max=3.0, n_points=50)

def test_end_to_end_alcubierre(sample_profiler):
    """End-to-end test with Alcubierre profile."""
    # Define Alcubierre profile
    profile_func = lambda r: alcubierre_profile(r, R=1.5, sigma=0.2)
    
    # Compute T^{00} profile
    r_array, T00_profile = sample_profiler.compute_T00_profile(profile_func)
    
    # Should have some finite values
    assert np.any(np.isfinite(T00_profile))
    
    # Identify exotic regions
    exotic_info = sample_profiler.identify_exotic_regions(T00_profile)
    
    # For Alcubierre profile, we expect exotic matter regions
    # (though this depends on the specific metric implementation)
    assert isinstance(exotic_info['has_exotic'], bool)
    assert isinstance(exotic_info['total_exotic_energy'], float)

def test_end_to_end_gaussian(sample_profiler):
    """End-to-end test with Gaussian profile."""
    # Define Gaussian profile
    profile_func = lambda r: gaussian_warp_profile(r, A=0.8, sigma=0.5)
    
    # Compute T^{00} profile
    r_array, T00_profile = sample_profiler.compute_T00_profile(profile_func)
    
    # Should have some finite values
    assert np.any(np.isfinite(T00_profile))
    assert len(r_array) == len(T00_profile)
    
    # Check that computation doesn't crash
    exotic_info = sample_profiler.identify_exotic_regions(T00_profile)
    assert 'has_exotic' in exotic_info

if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__, "-v"])
