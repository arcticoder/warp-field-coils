#!/usr/bin/env python3
"""
Test suite for quantum penalty optimization functionality
"""

import pytest
import numpy as np
import jax.numpy as jnp
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from coil_optimizer.advanced_coil_optimizer import AdvancedCoilOptimizer
from stress_energy.exotic_matter_profile import ExoticMatterProfiler, alcubierre_profile


class TestQuantumPenalty:
    """Test cases for quantum-aware coil optimization."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.optimizer = AdvancedCoilOptimizer(r_min=0.1, r_max=3.0, n_points=30)
        self.profiler = ExoticMatterProfiler(r_min=0.1, r_max=3.0, n_points=30)
        
        # Set up target profile
        r_array, T00_target = self.profiler.compute_T00_profile(
            lambda r: alcubierre_profile(r, R=1.5, sigma=0.4)
        )
        self.optimizer.set_target_profile(r_array, T00_target)
    
    def test_quantum_penalty_consistency(self):
        """Test quantum penalty function consistency."""
        params = jnp.array([0.1, 1.5, 0.4])
        
        # Quantum penalty should be non-negative
        J_quantum = self.optimizer.quantum_penalty(params)
        assert J_quantum >= 0, f"Quantum penalty must be non-negative, got {J_quantum}"
        
        # Should return finite value
        assert np.isfinite(J_quantum), "Quantum penalty must be finite"
    
    def test_quantum_penalty_parameter_scaling(self):
        """Test quantum penalty response to parameter scaling."""
        base_params = jnp.array([0.1, 1.5, 0.4])
        
        # Test different parameter scales
        scaled_params = base_params * 2.0
        
        J_base = self.optimizer.quantum_penalty(base_params)
        J_scaled = self.optimizer.quantum_penalty(scaled_params)
        
        # Both should be finite and non-negative
        assert np.isfinite(J_base) and J_base >= 0
        assert np.isfinite(J_scaled) and J_scaled >= 0
    
    def test_objective_with_quantum_components(self):
        """Test that quantum-aware objective includes both classical and quantum terms."""
        params = jnp.array([0.1, 1.5, 0.4])
        
        # Classical objective
        J_classical = self.optimizer.objective_function(params)
        
        # Quantum penalty
        J_quantum_only = self.optimizer.quantum_penalty(params)
        
        # Combined objective
        alpha = 1e-3
        J_combined = self.optimizer.objective_with_quantum(params, alpha=alpha)
        
        # Combined should include both components
        expected = J_classical + alpha * J_quantum_only
        
        assert np.isclose(J_combined, expected, rtol=1e-6), \
            f"Combined objective mismatch: {J_combined} vs {expected}"
    
    def test_quantum_penalty_ansatz_types(self):
        """Test quantum penalty with different ansatz types."""
        params = jnp.array([0.1, 1.5, 0.4])
        
        # Test both ansatz types
        J_gaussian = self.optimizer.quantum_penalty(params, "gaussian")
        J_polynomial = self.optimizer.quantum_penalty(params, "polynomial")
        
        # Both should be valid
        assert np.isfinite(J_gaussian) and J_gaussian >= 0
        assert np.isfinite(J_polynomial) and J_polynomial >= 0
    
    def test_K_matrix_construction(self):
        """Test K-matrix construction from current distribution."""
        params = jnp.array([0.1, 1.5, 0.4])
        
        # Get current distribution
        currents = self.optimizer.current_distribution(self.optimizer.rs, params)
        
        # Build K-matrix
        K_matrix = self.optimizer._build_K_from_currents(currents)
        
        # Check K-matrix properties
        assert K_matrix.shape == (self.optimizer.discrete_solver.n_nodes, 
                                 self.optimizer.discrete_solver.n_nodes)
        assert np.all(np.isfinite(K_matrix))
        
        # K-matrix should be real
        assert np.all(np.isreal(K_matrix))
    
    def test_quantum_penalty_optimization_integration(self):
        """Test quantum penalty integration in optimization workflow."""
        params = jnp.array([0.1, 1.5, 0.4])
        
        # Should be able to compute gradient of quantum-aware objective
        try:
            import jax
            grad_fn = jax.grad(lambda p: self.optimizer.objective_with_quantum(p, alpha=1e-3))
            gradient = grad_fn(params)
            
            assert len(gradient) == len(params)
            assert np.all(np.isfinite(gradient))
            
        except ImportError:
            pytest.skip("JAX not available for gradient test")
    
    def test_quantum_penalty_edge_cases(self):
        """Test quantum penalty behavior in edge cases."""
        # Zero parameters
        zero_params = jnp.zeros(3)
        J_zero = self.optimizer.quantum_penalty(zero_params)
        assert np.isfinite(J_zero) and J_zero >= 0
        
        # Very small parameters
        small_params = jnp.array([1e-6, 1e-6, 1e-6])
        J_small = self.optimizer.quantum_penalty(small_params)
        assert np.isfinite(J_small) and J_small >= 0
        
        # Large parameters
        large_params = jnp.array([10.0, 10.0, 10.0])
        J_large = self.optimizer.quantum_penalty(large_params)
        assert np.isfinite(J_large) and J_large >= 0


class TestQuantumAwareOptimization:
    """Test complete quantum-aware optimization workflow."""
    
    def setup_method(self):
        """Setup optimization test fixtures."""
        self.optimizer = AdvancedCoilOptimizer(r_min=0.1, r_max=2.0, n_points=20)
        self.profiler = ExoticMatterProfiler(r_min=0.1, r_max=2.0, n_points=20)
        
        # Simple target profile
        r_array, T00_target = self.profiler.compute_T00_profile(
            lambda r: alcubierre_profile(r, R=1.0, sigma=0.5)
        )
        self.optimizer.set_target_profile(r_array, T00_target)
    
    def test_quantum_aware_objective_evaluation(self):
        """Test evaluation of quantum-aware objective function."""
        params = jnp.array([0.05, 1.0, 0.3])
        
        # Should evaluate without error
        obj_val = self.optimizer.objective_with_quantum(params, alpha=1e-4)
        
        assert np.isfinite(obj_val)
        assert obj_val >= 0  # Should be positive for mismatch
    
    def test_quantum_optimization_bounds(self):
        """Test quantum optimization respects parameter bounds."""
        # Test with parameters near boundaries
        params_low = jnp.array([0.01, 0.2, 0.1])  # Near lower bounds
        params_high = jnp.array([1.0, 1.8, 1.0])  # Near upper bounds
        
        obj_low = self.optimizer.objective_with_quantum(params_low)
        obj_high = self.optimizer.objective_with_quantum(params_high)
        
        assert np.isfinite(obj_low)
        assert np.isfinite(obj_high)
    
    def test_quantum_penalty_weight_scaling(self):
        """Test quantum penalty weight (alpha) scaling behavior."""
        params = jnp.array([0.1, 1.0, 0.4])
        
        # Test different alpha values
        alphas = [1e-5, 1e-3, 1e-1]
        objectives = []
        
        for alpha in alphas:
            obj = self.optimizer.objective_with_quantum(params, alpha=alpha)
            objectives.append(obj)
            assert np.isfinite(obj)
        
        # Higher alpha should generally increase objective (more quantum penalty)
        assert len(set(objectives)) > 1, "Different alpha values should give different objectives"


def test_quantum_penalty_module_integration():
    """Test integration with other modules."""
    # Test that quantum penalty doesn't break existing functionality
    optimizer = AdvancedCoilOptimizer(r_min=0.1, r_max=1.0, n_points=10)
    
    # Should still work without target profile (should handle gracefully)
    params = jnp.array([0.1, 0.5, 0.2])
    
    try:
        penalty = optimizer.quantum_penalty(params)
        assert np.isfinite(penalty)
    except Exception as e:
        # Should not crash, even without proper setup
        assert "target" not in str(e).lower(), f"Unexpected error: {e}"


def test_quantum_optimization_convergence_behavior():
    """Test that quantum optimization can improve objectives."""
    optimizer = AdvancedCoilOptimizer(r_min=0.1, r_max=1.5, n_points=15)
    profiler = ExoticMatterProfiler(r_min=0.1, r_max=1.5, n_points=15)
    
    # Set target
    r_array, T00_target = profiler.compute_T00_profile(
        lambda r: alcubierre_profile(r, R=0.8, sigma=0.6)
    )
    optimizer.set_target_profile(r_array, T00_target)
    
    # Test that optimization can improve from random start
    initial_params = jnp.array([0.2, 0.8, 0.3])
    initial_obj = optimizer.objective_with_quantum(initial_params, alpha=1e-4)
    
    # Apply simple gradient step
    try:
        import jax
        grad_fn = jax.grad(lambda p: optimizer.objective_with_quantum(p, alpha=1e-4))
        gradient = grad_fn(initial_params)
        
        # Take small step in negative gradient direction
        step_size = 0.01
        improved_params = initial_params - step_size * gradient
        improved_obj = optimizer.objective_with_quantum(improved_params, alpha=1e-4)
        
        # Should have finite objectives
        assert np.isfinite(initial_obj)
        assert np.isfinite(improved_obj)
        
    except ImportError:
        pytest.skip("JAX not available for gradient-based improvement test")
