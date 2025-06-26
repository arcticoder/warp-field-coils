#!/usr/bin/env python3
"""
Advanced Coil Geometry Optimizer
Implements Step 2 of the roadmap: optimize coil geometry to match exotic matter profile
Based on warp-bubble-optimizer's advanced_shape_optimizer.py
"""

import numpy as np
import jax.numpy as jnp
from jax import jit, grad, hessian
import scipy.optimize
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Callable, Optional, List
from dataclasses import dataclass

@dataclass
class CoilGeometryParams:
    """Parameters defining coil geometry."""
    inner_radius: float  # Inner radius (m)
    outer_radius: float  # Outer radius (m)
    height: float       # Coil height (m)
    turn_density: float # Turns per unit length (m^-1)
    current: float      # Current per turn (A)
    n_layers: int       # Number of coil layers
    wire_gauge: float   # Wire cross-sectional area (m²)

class AdvancedCoilOptimizer:
    """
    JAX-accelerated coil geometry optimizer to match target exotic matter profiles.
    Implements Step 2 of the roadmap.
    """
    
    def __init__(self, r_min: float = 0.1, r_max: float = 10.0, n_points: int = 500):
        """
        Initialize the coil optimizer.
        
        Args:
            r_min: Minimum radial coordinate
            r_max: Maximum radial coordinate
            n_points: Number of grid points
        """
        self.r_min = r_min
        self.r_max = r_max
        self.n_points = n_points
        self.dr = (r_max - r_min) / (n_points - 1)
        self.rs = jnp.linspace(r_min, r_max, n_points)
        
        # Physical constants
        self.c = 299792458.0  # m/s
        self.G = 6.67430e-11  # m³/(kg⋅s²)
        self.mu0 = 4 * jnp.pi * 1e-7  # H/m
        self.eps0 = 8.854187817e-12  # F/m
        
        # Target profile storage
        self.target_T00 = None
        self.target_r_array = None
    
    def set_target_profile(self, r_array: np.ndarray, T00_profile: np.ndarray):
        """
        Set the target exotic matter profile to match.
        
        Args:
            r_array: Radial coordinate array
            T00_profile: Target T^{00}(r) profile
        """
        # Interpolate target profile onto our grid
        from scipy.interpolate import interp1d
        interp_func = interp1d(r_array, T00_profile, kind='cubic', 
                             bounds_error=False, fill_value=0.0)
        self.target_T00 = jnp.array(interp_func(self.rs))
        self.target_r_array = self.rs
    
    def gaussian_ansatz(self, r: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
        """Multi-Gaussian ansatz for coil current distribution."""
        n_gaussians = len(theta) // 3
        f = jnp.zeros_like(r)
        for i in range(n_gaussians):
            A = theta[3*i]      # Amplitude
            r_center = theta[3*i + 1]  # Center position
            sigma = theta[3*i + 2]     # Width
            f += A * jnp.exp(-((r - r_center)/sigma)**2)
        return f
    
    def polynomial_ansatz(self, r: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
        """Polynomial ansatz with exponential envelope."""
        poly = jnp.polyval(theta[:-1], r)
        envelope = jnp.exp(-r / theta[-1])
        return poly * envelope
    
    def current_distribution(self, r: jnp.ndarray, params: jnp.ndarray, 
                           ansatz_type: str = "gaussian") -> jnp.ndarray:
        """
        Compute current distribution J(r) from parameters.
        
        Args:
            r: Radial coordinate array
            params: Parameter vector defining current distribution
            ansatz_type: Type of ansatz ("gaussian" or "polynomial")
            
        Returns:
            Current density J(r)
        """
        if ansatz_type == "gaussian":
            return self.gaussian_ansatz(r, params)
        elif ansatz_type == "polynomial":
            return self.polynomial_ansatz(r, params)
        else:
            # Default: simple Gaussian
            A, sigma = params[:2]
            return A * jnp.exp(-(r/sigma)**2)
    
    def magnetic_field_coil(self, r: jnp.ndarray, J: jnp.ndarray) -> jnp.ndarray:
        """
        Compute magnetic field B(r) from current distribution J(r).
        Using simplified toroidal coil model.
        
        Args:
            r: Radial coordinate array
            J: Current density distribution
            
        Returns:
            Magnetic field B(r)
        """
        # Simplified model: B ∝ μ₀ * J * r for toroidal geometry
        # More accurate models would use Biot-Savart integration
        B = self.mu0 * J * r / (2 * jnp.pi)
        return B
    
    def stress_energy_tensor_00_coil(self, r: jnp.ndarray, params: jnp.ndarray,
                                   ansatz_type: str = "gaussian") -> jnp.ndarray:
        """
        Compute T^{00} from coil electromagnetic fields.
        
        Args:
            r: Radial coordinate array
            params: Coil parameter vector
            ansatz_type: Current distribution ansatz type
            
        Returns:
            Electromagnetic stress-energy T^{00}_coil(r)
        """
        # Get current distribution
        J = self.current_distribution(r, params, ansatz_type)
        
        # Compute magnetic field
        B = self.magnetic_field_coil(r, J)
        
        # Electric field from time-varying current (simplified)
        # For static case, this would be zero, but we include dynamic effects
        dJ_dt = jnp.gradient(J, self.dr)  # Simplified time derivative
        E = -dJ_dt / (self.eps0 * self.c)  # From Faraday's law approximation
        
        # Electromagnetic stress-energy tensor T^{00} component
        # T^{00}_EM = (1/2)(ε₀E² + B²/μ₀)
        T00_coil = 0.5 * (self.eps0 * E**2 + B**2 / self.mu0)
        
        # Convert to geometric units (factor of c⁴/8πG for comparison with spacetime)
        geometric_factor = self.c**4 / (8 * jnp.pi * self.G)
        T00_coil_geometric = T00_coil / geometric_factor
        
        return T00_coil_geometric
    
    def objective_function(self, params: jnp.ndarray, ansatz_type: str = "gaussian") -> float:
        """
        Objective function J(p) = ∫[T^{00}_coil(r;p) - T^{00}_target(r)]² dr
        
        Args:
            params: Coil parameter vector
            ansatz_type: Current distribution ansatz type
            
        Returns:
            Objective function value
        """
        if self.target_T00 is None:
            raise ValueError("Target profile not set. Call set_target_profile() first.")
        
        # Compute coil stress-energy profile
        T00_coil = self.stress_energy_tensor_00_coil(self.rs, params, ansatz_type)
        
        # Compute L2 difference
        diff = T00_coil - self.target_T00
        objective = jnp.trapezoid(diff**2, self.rs)
        
        return objective
    
    def gradient_objective(self, params: jnp.ndarray, ansatz_type: str = "gaussian") -> jnp.ndarray:
        """Compute gradient of objective function using JAX autodiff."""
        return grad(lambda p: self.objective_function(p, ansatz_type))(params)
    
    def hessian_objective(self, params: jnp.ndarray, ansatz_type: str = "gaussian") -> jnp.ndarray:
        """Compute Hessian of objective function using JAX autodiff."""
        return hessian(lambda p: self.objective_function(p, ansatz_type))(params)
    
    def optimize_lbfgs(self, initial_params: np.ndarray, ansatz_type: str = "gaussian",
                      maxiter: int = 1000) -> Dict:
        """
        Optimize coil geometry using L-BFGS-B algorithm.
        
        Args:
            initial_params: Initial parameter guess
            ansatz_type: Current distribution ansatz type
            maxiter: Maximum number of iterations
            
        Returns:
            Optimization result dictionary
        """
        # Convert to JAX arrays
        initial_params_jax = jnp.array(initial_params)
        
        # Define objective and gradient functions for scipy
        def obj_func(p):
            return float(self.objective_function(jnp.array(p), ansatz_type))
        
        def grad_func(p):
            return np.array(self.gradient_objective(jnp.array(p), ansatz_type))
        
        # Set bounds (all parameters should be positive for physical meaning)
        bounds = [(0.01, 10.0) for _ in range(len(initial_params))]
        
        # Run L-BFGS-B optimization
        result = scipy.optimize.minimize(
            obj_func, initial_params, method='L-BFGS-B',
            jac=grad_func, bounds=bounds,
            options={'maxiter': maxiter, 'disp': True}
        )
        
        return {
            'optimal_params': result.x,
            'optimal_objective': result.fun,
            'success': result.success,
            'message': result.message,
            'n_iterations': result.nit,
            'scipy_result': result
        }
    
    def optimize_cmaes(self, initial_params: np.ndarray, ansatz_type: str = "gaussian",
                      sigma: float = 0.5, maxiter: int = 100) -> Dict:
        """
        Optimize coil geometry using CMA-ES (Evolution Strategy).
        
        Args:
            initial_params: Initial parameter guess
            ansatz_type: Current distribution ansatz type
            sigma: Initial step size
            maxiter: Maximum number of generations
            
        Returns:
            Optimization result dictionary
        """
        def obj_func(p):
            return float(self.objective_function(jnp.array(p), ansatz_type))
        
        # Set bounds
        bounds = [(0.01, 10.0) for _ in range(len(initial_params))]
        
        # Run differential evolution (scipy's version of evolutionary algorithm)
        result = differential_evolution(
            obj_func, bounds, maxiter=maxiter,
            seed=42, disp=True
        )
        
        return {
            'optimal_params': result.x,
            'optimal_objective': result.fun,
            'success': result.success,
            'message': result.message,
            'n_iterations': result.nit,
            'scipy_result': result
        }
    
    def optimize_hybrid(self, initial_params: np.ndarray, ansatz_type: str = "gaussian") -> Dict:
        """
        Hybrid optimization: CMA-ES followed by L-BFGS-B refinement.
        
        Args:
            initial_params: Initial parameter guess
            ansatz_type: Current distribution ansatz type
            
        Returns:
            Optimization result dictionary
        """
        print("Stage 1: CMA-ES global optimization...")
        cmaes_result = self.optimize_cmaes(initial_params, ansatz_type, maxiter=50)
        
        print("Stage 2: L-BFGS-B local refinement...")
        lbfgs_result = self.optimize_lbfgs(cmaes_result['optimal_params'], 
                                         ansatz_type, maxiter=500)
        
        return {
            'optimal_params': lbfgs_result['optimal_params'],
            'optimal_objective': lbfgs_result['optimal_objective'],
            'success': lbfgs_result['success'],
            'cmaes_stage': cmaes_result,
            'lbfgs_stage': lbfgs_result
        }
    
    def plot_optimization_result(self, optimal_params: np.ndarray, 
                               ansatz_type: str = "gaussian",
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot optimization result: target vs. optimized profiles.
        
        Args:
            optimal_params: Optimized parameters
            ansatz_type: Current distribution ansatz type
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure object
        """
        # Compute optimized profile
        T00_optimized = np.array(self.stress_energy_tensor_00_coil(
            self.rs, jnp.array(optimal_params), ansatz_type
        ))
        
        # Create plot
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
        
        # Plot stress-energy profiles
        ax1.plot(self.rs, self.target_T00, 'r-', linewidth=2, label='Target $T^{00}(r)$')
        ax1.plot(self.rs, T00_optimized, 'b--', linewidth=2, label='Optimized Coil $T^{00}(r)$')
        ax1.set_xlabel('Radial Distance $r$')
        ax1.set_ylabel('$T^{00}$')
        ax1.set_title('Stress-Energy Profile Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot current distribution
        J_optimized = np.array(self.current_distribution(
            self.rs, jnp.array(optimal_params), ansatz_type
        ))
        ax2.plot(self.rs, J_optimized, 'g-', linewidth=2, label='Optimized Current $J(r)$')
        ax2.set_xlabel('Radial Distance $r$')
        ax2.set_ylabel('Current Density $J(r)$')
        ax2.set_title('Optimized Current Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot difference
        diff = T00_optimized - np.array(self.target_T00)
        ax3.plot(self.rs, diff, 'm-', linewidth=2, label='Difference')
        ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Radial Distance $r$')
        ax3.set_ylabel(r'$T^{00}_{\mathrm{coil}} - T^{00}_{\mathrm{target}}$')
        ax3.set_title('Optimization Error')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def extract_coil_geometry(self, optimal_params: np.ndarray, 
                            ansatz_type: str = "gaussian") -> CoilGeometryParams:
        """
        Extract physical coil geometry parameters from optimization result.
        
        Args:
            optimal_params: Optimized parameter vector
            ansatz_type: Current distribution ansatz type
            
        Returns:
            CoilGeometryParams object with physical parameters
        """
        if ansatz_type == "gaussian":
            # For multi-Gaussian, extract dominant Gaussian parameters
            n_gaussians = len(optimal_params) // 3
            # Find the Gaussian with maximum amplitude
            amplitudes = optimal_params[::3]
            max_idx = np.argmax(np.abs(amplitudes))
            
            A = optimal_params[3*max_idx]
            r_center = optimal_params[3*max_idx + 1]
            sigma = optimal_params[3*max_idx + 2]
            
            # Convert to physical coil parameters
            inner_radius = max(0.1, r_center - 2*sigma)
            outer_radius = r_center + 2*sigma
            height = 4*sigma  # Approximate height
            turn_density = abs(A) * 1000  # Scaling factor
            current = 1000.0  # A, typical superconducting coil current
            
        else:  # polynomial
            # Extract characteristic scales from polynomial coefficients
            inner_radius = 0.5
            outer_radius = 2.0
            height = 1.0
            turn_density = 1000.0
            current = 1000.0
        
        return CoilGeometryParams(
            inner_radius=inner_radius,
            outer_radius=outer_radius,
            height=height,
            turn_density=turn_density,
            current=current,
            n_layers=10,  # Default
            wire_gauge=1e-6  # m²
        )

if __name__ == "__main__":
    # Example usage
    optimizer = AdvancedCoilOptimizer()
    
    # Create a sample target profile (negative energy shell)
    r_target = optimizer.rs
    T00_target = -0.1 * np.exp(-((r_target - 2.0)/0.5)**2)  # Negative Gaussian
    
    optimizer.set_target_profile(r_target, T00_target)
    
    # Initial guess: single Gaussian [A, r_center, sigma]
    initial_params = np.array([1.0, 2.0, 0.5])
    
    # Run hybrid optimization
    result = optimizer.optimize_hybrid(initial_params, ansatz_type="gaussian")
    
    print(f"Optimization success: {result['success']}")
    print(f"Final objective: {result['optimal_objective']:.6e}")
    print(f"Optimal parameters: {result['optimal_params']}")
    
    # Plot results
    fig = optimizer.plot_optimization_result(result['optimal_params'])
    plt.show()
    
    # Extract coil geometry
    coil_params = optimizer.extract_coil_geometry(result['optimal_params'])
    print(f"Extracted coil geometry:")
    print(f"  Inner radius: {coil_params.inner_radius:.3f} m")
    print(f"  Outer radius: {coil_params.outer_radius:.3f} m")
    print(f"  Height: {coil_params.height:.3f} m")
    print(f"  Turn density: {coil_params.turn_density:.1f} turns/m")
