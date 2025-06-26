#!/usr/bin/env python3
"""
Test suite for advanced coil optimizer
"""

import pytest
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    import jax.numpy as jnp
    from coil_optimizer.advanced_coil_optimizer import AdvancedCoilOptimizer, CoilGeometryParams
    HAS_JAX = True
except ImportError:
    print("JAX not available - skipping JAX-dependent tests")
    HAS_JAX = False
    jnp = np

class TestAdvancedCoilOptimizer:
    """Test cases for AdvancedCoilOptimizer class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.optimizer = AdvancedCoilOptimizer(r_min=0.1, r_max=5.0, n_points=100)
    
    def test_initialization(self):
        """Test optimizer initialization."""
        assert self.optimizer.r_min == 0.1
        assert self.optimizer.r_max == 5.0
        assert self.optimizer.n_points == 100
        assert len(self.optimizer.rs) == 100
        
        # Check physical constants
        assert self.optimizer.c > 0
        assert self.optimizer.G > 0
        assert self.optimizer.mu0 > 0
        assert self.optimizer.eps0 > 0
    
    def test_set_target_profile(self):
        """Test target profile setting."""
        r_array = np.linspace(0.1, 5.0, 50)
        T00_profile = -0.1 * np.exp(-((r_array - 2.0)/0.5)**2)
        
        self.optimizer.set_target_profile(r_array, T00_profile)
        
        assert self.optimizer.target_T00 is not None
        assert len(self.optimizer.target_T00) == self.optimizer.n_points
        assert self.optimizer.target_r_array is not None
    
    def test_gaussian_ansatz(self):
        """Test multi-Gaussian ansatz function."""
        r = self.optimizer.rs
        
        # Single Gaussian: [A, r_center, sigma]
        theta_single = jnp.array([1.0, 2.0, 0.5])
        f_single = self.optimizer.gaussian_ansatz(r, theta_single)
        
        assert len(f_single) == len(r)
        assert jnp.max(f_single) > 0  # Should have positive peak
        
        # Multi-Gaussian: [A1, r1, sigma1, A2, r2, sigma2]
        theta_multi = jnp.array([0.5, 1.0, 0.3, 0.8, 3.0, 0.4])
        f_multi = self.optimizer.gaussian_ansatz(r, theta_multi)
        
        assert len(f_multi) == len(r)
        # Should have contributions from both Gaussians
        assert jnp.max(f_multi) > jnp.max(f_single) * 0.5
    
    def test_polynomial_ansatz(self):
        """Test polynomial ansatz with exponential envelope."""
        r = self.optimizer.rs
        
        # Quadratic polynomial with exponential decay: [a2, a1, a0, tau]
        theta_poly = jnp.array([0.1, 0.5, 1.0, 2.0])
        f_poly = self.optimizer.polynomial_ansatz(r, theta_poly)
        
        assert len(f_poly) == len(r)
        # Should decay to zero at large r due to exponential envelope
        assert f_poly[0] > f_poly[-1]
    
    def test_current_distribution(self):
        """Test current distribution computation."""
        r = self.optimizer.rs
        params = jnp.array([1.0, 2.0, 0.5])
        
        # Test Gaussian ansatz
        J_gaussian = self.optimizer.current_distribution(r, params, "gaussian")
        assert len(J_gaussian) == len(r)
        assert jnp.all(jnp.isfinite(J_gaussian))
        
        # Test polynomial ansatz
        params_poly = jnp.array([0.1, 0.5, 1.0, 2.0])
        J_poly = self.optimizer.current_distribution(r, params_poly, "polynomial")
        assert len(J_poly) == len(r)
        assert jnp.all(jnp.isfinite(J_poly))
        
        # Test default case
        J_default = self.optimizer.current_distribution(r, params, "unknown")
        assert len(J_default) == len(r)
    
    def test_magnetic_field_coil(self):
        """Test magnetic field computation from current distribution."""
        r = self.optimizer.rs
        J = jnp.exp(-((r - 2.0)/0.5)**2)  # Gaussian current distribution
        
        B = self.optimizer.magnetic_field_coil(r, J)
        
        assert len(B) == len(r)
        assert jnp.all(B >= 0)  # Magnetic field magnitude should be positive
        assert jnp.all(jnp.isfinite(B))
        
        # Should scale with current
        B_scaled = self.optimizer.magnetic_field_coil(r, 2*J)
        assert jnp.allclose(B_scaled, 2*B)
    
    def test_stress_energy_tensor_00_coil(self):
        """Test electromagnetic stress-energy tensor computation."""
        r = self.optimizer.rs
        params = jnp.array([0.5, 2.0, 0.5])  # Modest amplitude to avoid numerical issues
        
        T00_coil = self.optimizer.stress_energy_tensor_00_coil(r, params, "gaussian")
        
        assert len(T00_coil) == len(r)
        assert jnp.all(jnp.isfinite(T00_coil))
        
        # Should be positive (electromagnetic energy density)
        assert jnp.all(T00_coil >= 0)
    
    def test_objective_function(self):
        """Test objective function computation."""
        # Set up target profile
        r_array = np.array(self.optimizer.rs)
        T00_target = -0.05 * np.exp(-((r_array - 2.0)/0.5)**2)
        self.optimizer.set_target_profile(r_array, T00_target)
        
        params = jnp.array([0.1, 2.0, 0.5])
        objective_val = self.optimizer.objective_function(params, "gaussian")
        
        assert jnp.isfinite(objective_val)
        assert objective_val >= 0  # L2 norm should be non-negative
        
        # Test that objective function raises error without target
        optimizer_no_target = AdvancedCoilOptimizer()
        with pytest.raises(ValueError):
            optimizer_no_target.objective_function(params)
    
    def test_gradient_objective(self):
        """Test objective function gradient computation."""
        # Set up target profile
        r_array = np.array(self.optimizer.rs)
        T00_target = -0.05 * np.exp(-((r_array - 2.0)/0.5)**2)
        self.optimizer.set_target_profile(r_array, T00_target)
        
        params = jnp.array([0.1, 2.0, 0.5])
        gradient = self.optimizer.gradient_objective(params, "gaussian")
        
        assert len(gradient) == len(params)
        assert jnp.all(jnp.isfinite(gradient))
        
        # Test numerical gradient vs analytical (rough check)
        h = 1e-6
        grad_numerical = []
        for i in range(len(params)):
            params_plus = params.at[i].add(h)
            params_minus = params.at[i].add(-h)
            
            obj_plus = self.optimizer.objective_function(params_plus)
            obj_minus = self.optimizer.objective_function(params_minus)
            
            grad_num = (obj_plus - obj_minus) / (2 * h)
            grad_numerical.append(grad_num)
        
        grad_numerical = jnp.array(grad_numerical)
        
        # Check that analytical and numerical gradients are close
        assert jnp.allclose(gradient, grad_numerical, rtol=1e-3, atol=1e-6)

class TestCoilGeometryParams:
    """Test cases for CoilGeometryParams dataclass."""
    
    def test_coil_geometry_creation(self):
        """Test CoilGeometryParams creation and access."""
        params = CoilGeometryParams(
            inner_radius=0.5,
            outer_radius=1.5,
            height=2.0,
            turn_density=1000.0,
            current=5000.0,
            n_layers=5,
            wire_gauge=1e-6
        )
        
        assert params.inner_radius == 0.5
        assert params.outer_radius == 1.5
        assert params.height == 2.0
        assert params.turn_density == 1000.0
        assert params.current == 5000.0
        assert params.n_layers == 5
        assert params.wire_gauge == 1e-6

class TestOptimizationMethods:
    """Test cases for optimization methods."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.optimizer = AdvancedCoilOptimizer(r_min=0.5, r_max=3.0, n_points=50)
        
        # Set simple target profile
        r_array = np.array(self.optimizer.rs)
        T00_target = -0.01 * np.exp(-((r_array - 1.5)/0.3)**2)  # Small target to avoid numerical issues
        self.optimizer.set_target_profile(r_array, T00_target)
    
    def test_extract_coil_geometry_gaussian(self):
        """Test coil geometry extraction from Gaussian parameters."""
        # Three Gaussians: [A1, r1, s1, A2, r2, s2, A3, r3, s3]
        optimal_params = np.array([0.1, 1.0, 0.2, -0.2, 1.5, 0.3, 0.05, 2.0, 0.1])
        
        geometry = self.optimizer.extract_coil_geometry(optimal_params, "gaussian")
        
        assert isinstance(geometry, CoilGeometryParams)
        assert geometry.inner_radius > 0
        assert geometry.outer_radius > geometry.inner_radius
        assert geometry.height > 0
        assert geometry.turn_density > 0
        assert geometry.current > 0
    
    def test_extract_coil_geometry_polynomial(self):
        """Test coil geometry extraction from polynomial parameters."""
        optimal_params = np.array([0.1, 0.2, 0.5, 1.0])
        
        geometry = self.optimizer.extract_coil_geometry(optimal_params, "polynomial")
        
        assert isinstance(geometry, CoilGeometryParams)
        assert geometry.inner_radius > 0
        assert geometry.outer_radius > geometry.inner_radius

@pytest.fixture
def simple_optimizer():
    """Fixture providing a simple optimizer setup."""
    optimizer = AdvancedCoilOptimizer(r_min=0.5, r_max=2.5, n_points=30)
    
    # Set very simple target
    r_array = np.array(optimizer.rs)
    T00_target = -0.001 * np.exp(-((r_array - 1.5)/0.5)**2)
    optimizer.set_target_profile(r_array, T00_target)
    
    return optimizer

def test_optimization_convergence(simple_optimizer):
    """Test that optimization converges for simple case."""
    initial_params = np.array([0.01, 1.5, 0.5])  # Small amplitude
    
    # Test L-BFGS optimization with few iterations
    result = simple_optimizer.optimize_lbfgs(initial_params, maxiter=10)
    
    assert 'optimal_params' in result
    assert 'optimal_objective' in result
    assert len(result['optimal_params']) == len(initial_params)
    assert result['optimal_objective'] >= 0
    
    # Final objective should be finite
    assert np.isfinite(result['optimal_objective'])

def test_optimization_bounds_respected(simple_optimizer):
    """Test that optimization respects parameter bounds."""
    initial_params = np.array([0.01, 1.5, 0.5])
    
    result = simple_optimizer.optimize_lbfgs(initial_params, maxiter=5)
    
    if result['success']:
        # All parameters should be within bounds [0.01, 10.0]
        assert np.all(result['optimal_params'] >= 0.01)
        assert np.all(result['optimal_params'] <= 10.0)

if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__, "-v"])
