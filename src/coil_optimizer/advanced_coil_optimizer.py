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
import time

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
    
    def __init__(self, r_min: float = 0.1, r_max: float = 5.0, n_points: int = 100):
        """Initialize the advanced coil optimizer with quantum geometry integration."""
        self.r_min = r_min
        self.r_max = r_max
        self.n_points = n_points
        self.rs = jnp.linspace(r_min, r_max, n_points)
        self.dr = (r_max - r_min) / (n_points - 1)  # Grid spacing
        
        # Physical constants
        self.c = 299792458.0  # Speed of light (m/s)
        self.G = 6.67430e-11  # Gravitational constant (m³/kg·s²)
        self.mu0 = 4*np.pi*1e-7  # Permeability of free space (H/m)
        self.eps0 = 8.854187817e-12  # Permittivity of free space (F/m)
        
        # Target profile storage
        self.target_T00 = None
        self.target_r_array = None
        
        # Quantum geometry integration
        from src.quantum_geometry.discrete_stress_energy import DiscreteQuantumGeometry
        self.discrete_solver = DiscreteQuantumGeometry(n_nodes=20)  # Smaller for efficiency
        self.su2_calculator = self.discrete_solver.su2_calculator
    
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
    
    def quantum_penalty(self, params: jnp.ndarray, ansatz_type: str = "gaussian") -> float:
        """
        Compute quantum geometry penalty from SU(2) generating functional.
        
        Penalizes deviations from G(K) = 1 (perfect quantum consistency).
        
        Args:
            params: Current distribution parameters
            ansatz_type: Type of current ansatz
            
        Returns:
            Quantum penalty term (1/G - 1)²
        """
        try:
            # Get current distribution
            J = self.current_distribution(self.rs, params, ansatz_type)
            
            # Build K-matrix from current distribution
            # Map currents to quantum geometry via adjacency structure
            K_matrix = self._build_K_from_currents(J)
            
            # Compute generating functional
            G = self.su2_calculator.compute_generating_functional(K_matrix)
            
            # Penalty for deviation from unity
            penalty = (1.0/G - 1.0)**2
            
            return float(penalty)
            
        except Exception as e:
            # Return large penalty if computation fails
            return 1e6
    
    def _build_K_from_currents(self, currents: jnp.ndarray) -> jnp.ndarray:
        """
        Build quantum geometry K-matrix from current distribution.
        
        Maps electromagnetic currents to SU(2) node interactions.
        """
        n_nodes = self.discrete_solver.n_nodes
        
        # Initialize K-matrix
        K = jnp.zeros((n_nodes, n_nodes))
        
        # Map current density to quantum nodes
        # Scale currents to appropriate range for SU(2) calculations
        current_scale = jnp.max(jnp.abs(currents))
        if current_scale > 1e-12:
            normalized_currents = currents / current_scale
        else:
            normalized_currents = currents
        
        # Create K-matrix based on adjacency and current strength
        adjacency = self.discrete_solver.adjacency_matrix
        
        for i in range(n_nodes):
            for j in range(n_nodes):
                if adjacency[i, j] > 0:
                    # Weight by current contribution
                    r_idx = min(i * len(currents) // n_nodes, len(currents) - 1)
                    current_weight = float(normalized_currents[r_idx])
                    
                    # K_ij represents interaction strength
                    K = K.at[i, j].set(0.1 * current_weight * adjacency[i, j])
        
        return K
    
    def objective_with_quantum(self, params: jnp.ndarray, ansatz_type: str = "gaussian", 
                             alpha: float = 1e-3) -> float:
        """
        Total objective function including quantum geometry penalty.
        
        J_total = J_classical + α * (1/G - 1)²
        
        Args:
            params: Optimization parameters
            ansatz_type: Current distribution ansatz type
            alpha: Quantum penalty weight
            
        Returns:
            Total objective value
        """
        # Classical stress-energy matching objective
        J_classical = self.objective_function(params, ansatz_type)
        
        # Quantum geometry penalty
        J_quantum = alpha * self.quantum_penalty(params, ansatz_type)
        
        return J_classical + J_quantum
    
    def mechanical_penalty(self, params: jnp.ndarray, thickness: float = 0.005, 
                          sigma_yield: float = 300e6) -> float:
        """
        Compute mechanical stress penalty for coil windings.
        
        Hoop stress: σ_θ(r,I) = μ₀I²/(2πrt)
        
        Args:
            params: Coil parameter vector
            thickness: Conductor thickness (m)
            sigma_yield: Yield stress limit (Pa)
            
        Returns:
            Mechanical penalty J_mech = Σ max(0, σ_θ - σ_yield)²
        """
        # Extract current amplitudes from parameter vector
        n_gaussians = len(params) // 3
        A = params.reshape(n_gaussians, 3)[:, 0]  # Amplitude for each Gaussian shell
        r = params.reshape(n_gaussians, 3)[:, 1]  # Center radius
        
        # Hoop stress calculation: σ_θ = μ₀I²/(2πrt)
        mu0 = 4 * jnp.pi * 1e-7  # Permeability of free space
        I_squared = A**2  # Current squared (proportional to amplitude squared)
        
        # Avoid division by zero
        r_safe = jnp.maximum(r, 1e-6)
        thickness_safe = jnp.maximum(thickness, 1e-6)
        
        sigma_theta = mu0 * I_squared / (2 * jnp.pi * r_safe * thickness_safe)
        
        # Penalty for exceeding yield stress
        stress_violation = jnp.maximum(0.0, sigma_theta - sigma_yield)
        penalty = jnp.sum(jnp.square(stress_violation))
        
        return penalty
    
    def thermal_penalty(self, params: jnp.ndarray, rho_cu: float = 1.7e-8,
                       area: float = 1e-6, P_max: float = 1e6) -> float:
        """
        Compute thermal penalty for ohmic heating.
        
        Power loss: P_loss(r,I) = I²R ≈ I²ρ/A
        
        Args:
            params: Coil parameter vector
            rho_cu: Copper resistivity (Ω·m)
            area: Conductor cross-sectional area (m²)
            P_max: Maximum allowable power (W)
            
        Returns:
            Thermal penalty J_therm = Σ (P_loss/P_max)²
        """
        # Extract current amplitudes
        n_gaussians = len(params) // 3
        A = params.reshape(n_gaussians, 3)[:, 0]  # Current amplitude
        
        # Ohmic power loss per coil section
        I_squared = A**2
        area_safe = jnp.maximum(area, 1e-12)
        
        P_loss = I_squared * (rho_cu / area_safe)
        
        # Normalized power penalty
        P_normalized = P_loss / P_max
        penalty = jnp.sum(jnp.square(P_normalized))
        
        return penalty
    
    def objective_full_multiphysics(self, params: jnp.ndarray, 
                                   alpha_q: float = 1e-3,
                                   alpha_m: float = 1e3, 
                                   alpha_t: float = 1e2,
                                   **constraints) -> float:
        """
        Full multi-physics objective function.
        
        J_total = J_classical + α_q·J_quantum + α_m·J_mech + α_t·J_thermal
        
        Args:
            params: Optimization parameters
            alpha_q: Quantum penalty weight
            alpha_m: Mechanical penalty weight  
            alpha_t: Thermal penalty weight
            **constraints: Physical constraint parameters
            
        Returns:
            Total multi-physics objective value
        """
        # Classical electromagnetic objective
        J_classical = self.objective_function(params)
        
        # Quantum geometry penalty
        J_quantum = self.quantum_penalty(params)
        
        # Mechanical stress penalty
        J_mechanical = self.mechanical_penalty(
            params,
            thickness=constraints.get('thickness', 0.005),
            sigma_yield=constraints.get('sigma_yield', 300e6)
        )
        
        # Thermal penalty
        J_thermal = self.thermal_penalty(
            params,
            rho_cu=constraints.get('rho_cu', 1.7e-8),
            area=constraints.get('area', 1e-6),
            P_max=constraints.get('P_max', 1e6)
        )
        
        # Combined multi-physics objective
        J_total = (J_classical + 
                  alpha_q * J_quantum + 
                  alpha_m * J_mechanical + 
                  alpha_t * J_thermal)
        
        return J_total
    
    def get_penalty_components(self, params: jnp.ndarray, **constraints) -> Dict[str, float]:
        """
        Get individual penalty components for analysis.
        
        Returns:
            Dictionary with individual penalty values
        """
        return {
            'classical': float(self.objective_function(params)),
            'quantum': float(self.quantum_penalty(params)),
            'mechanical': float(self.mechanical_penalty(
                params,
                thickness=constraints.get('thickness', 0.005),
                sigma_yield=constraints.get('sigma_yield', 300e6)
            )),
            'thermal': float(self.thermal_penalty(
                params,
                rho_cu=constraints.get('rho_cu', 1.7e-8),
                area=constraints.get('area', 1e-6),
                P_max=constraints.get('P_max', 1e6)
            ))
        }
    
    def compute_momentum_flux_vector(self, params: np.ndarray, 
                                    direction: np.ndarray = None) -> np.ndarray:
        """
        Compute 3D momentum flux vector F⃗ from coil parameters.
        
        Integrates T⁰ⁱ components over the warp bubble volume:
        F_i = ∫ T⁰ⁱ(x) d³x ≈ Σₖ T⁰ⁱₖ ΔVₖ
        
        Args:
            params: Coil optimization parameters
            direction: Optional preferred direction vector
            
        Returns:
            3D momentum flux vector [Fx, Fy, Fz]
        """
        if not hasattr(self, 'target_r_array') or not hasattr(self, 'target_T00'):
            print("⚠️ Target profile not set, using default for momentum flux computation")
            # Create default angular grid
            theta_array = np.linspace(0, np.pi, 32)
            r_array = self.rs
        else:
            theta_array = getattr(self, 'theta_array', np.linspace(0, np.pi, 32))
            r_array = self.target_r_array
        
        # Extract dipole parameters (assuming params includes dipole strength)
        if len(params) >= 4:
            eps = params[3]  # Dipole strength as 4th parameter
        else:
            eps = 0.1  # Default dipole strength
        
        # Generate dipolar warp profile
        from stress_energy.exotic_matter_profile import alcubierre_profile_dipole
        
        # Estimate bubble parameters from optimization params
        R0 = params[1] if len(params) > 1 else 2.0  # Center position as radius
        sigma = 1.0 / (params[2] + 1e-12) if len(params) > 2 else 2.0  # Width^-1 as sharpness
        
        # Compute dipolar profile
        f_profile = alcubierre_profile_dipole(r_array, theta_array, R0, sigma, eps)
        
        # Use profiler to compute momentum flux
        momentum_flux = self.exotic_profiler.compute_momentum_flux_vector(
            f_profile, r_array, theta_array
        )
        
        return momentum_flux
    
    def steering_penalty(self, params: np.ndarray, direction: np.ndarray) -> float:
        """
        Compute steering penalty for directional thrust optimization.
        
        Maximizes thrust component along desired direction:
        J_steer(p) = -(F⃗(p) · d̂)²
        
        Args:
            params: Optimization parameters
            direction: Unit direction vector for desired thrust
            
        Returns:
            Steering penalty (negative for maximization)
        """
        try:
            # Compute momentum flux vector
            momentum_flux = self.compute_momentum_flux_vector(params, direction)
            
            # Normalize direction vector
            direction_norm = direction / (np.linalg.norm(direction) + 1e-12)
            
            # Compute thrust component along desired direction
            thrust_component = np.dot(momentum_flux, direction_norm)
            
            # Return negative squared component (minimization maximizes thrust)
            steering_penalty = -(thrust_component**2)
            
            return steering_penalty
            
        except Exception as e:
            print(f"⚠️ Steering penalty computation failed: {e}")
            return 0.0  # Neutral penalty on failure
    
    def objective_with_steering(self, params: np.ndarray, 
                              alpha_s: float = 1e4,
                              direction: np.ndarray = np.array([1.0, 0.0, 0.0]),
                              **penalty_kwargs) -> float:
        """
        Multi-physics objective function with steering control.
        
        J_total = J_classical + α_q J_quantum + α_m J_mechanical + α_t J_thermal + α_s J_steer
        
        Args:
            params: Optimization parameters
            alpha_s: Steering penalty weight
            direction: Desired thrust direction
            **penalty_kwargs: Additional penalty parameters
            
        Returns:
            Total objective including steering
        """
        # Base multi-physics objective
        base_kwargs = {
            'alpha_q': penalty_kwargs.get('alpha_q', 1e-3),
            'alpha_m': penalty_kwargs.get('alpha_m', 1e3),
            'alpha_t': penalty_kwargs.get('alpha_t', 1e2),
            'thickness': penalty_kwargs.get('thickness', 0.005),
            'sigma_yield': penalty_kwargs.get('sigma_yield', 300e6),
            'rho_cu': penalty_kwargs.get('rho_cu', 1.7e-8),
            'area': penalty_kwargs.get('area', 1e-6),
            'P_max': penalty_kwargs.get('P_max', 1e6)
        }
        
        J_base = self.objective_full_multiphysics(params, **base_kwargs)
        
        # Steering penalty
        J_steer = self.steering_penalty(params, direction)
        
        # Combined objective
        J_total = J_base + alpha_s * J_steer
        
        return J_total
    
    def optimize_steering(self, direction: np.ndarray = np.array([1.0, 0.0, 0.0]),
                         alpha_s: float = 1e4,
                         initial_params: np.ndarray = None,
                         **optimization_kwargs) -> Dict:
        """
        Optimize coil configuration for directional thrust.
        
        Args:
            direction: Desired thrust direction vector
            alpha_s: Steering penalty weight
            initial_params: Initial optimization parameters
            **optimization_kwargs: Additional optimization parameters
            
        Returns:
            Steering optimization results
        """
        print(f"🎯 STEERABLE WARP DRIVE OPTIMIZATION")
        print(f"Target direction: [{direction[0]:.2f}, {direction[1]:.2f}, {direction[2]:.2f}]")
        print(f"Steering weight: α_s = {alpha_s:.0e}")
        
        # Default initial parameters
        if initial_params is None:
            initial_params = np.array([0.1, 2.0, 0.5, 0.1])  # [amplitude, center, width, dipole]
        
        # Ensure we have dipole parameter
        if len(initial_params) < 4:
            initial_params = np.append(initial_params, 0.1)  # Add dipole strength
        
        # Optimization bounds (include dipole bound)
        bounds = [
            (0.01, 5.0),   # Amplitude
            (0.1, 10.0),   # Center
            (0.1, 2.0),    # Width
            (0.0, 0.5)     # Dipole strength
        ]
        
        # Steering objective function
        def steering_objective(params):
            return self.objective_with_steering(
                params, alpha_s=alpha_s, direction=direction, **optimization_kwargs
            )
        
        # Run optimization
        from scipy.optimize import minimize
        
        start_time = time.time()
        
        result = minimize(
            steering_objective,
            initial_params,
            method='L-BFGS-B',
            bounds=bounds,
            options={
                'maxiter': optimization_kwargs.get('maxiter', 200),
                'ftol': optimization_kwargs.get('ftol', 1e-9)
            }
        )
        
        optimization_time = time.time() - start_time
        
        # Analyze results
        if result.success:
            optimal_params = result.x
            momentum_flux = self.compute_momentum_flux_vector(optimal_params, direction)
            thrust_magnitude = np.linalg.norm(momentum_flux)
            thrust_direction = momentum_flux / (thrust_magnitude + 1e-12)
            
            # Compute alignment with desired direction
            direction_norm = direction / np.linalg.norm(direction)
            alignment = np.dot(thrust_direction, direction_norm)
            
            steering_results = {
                'success': True,
                'optimal_params': optimal_params,
                'optimal_objective': result.fun,
                'momentum_flux': momentum_flux,
                'thrust_magnitude': thrust_magnitude,
                'thrust_direction': thrust_direction,
                'direction_alignment': alignment,
                'dipole_strength': optimal_params[3] if len(optimal_params) > 3 else 0.0,
                'optimization_time': optimization_time,
                'iterations': result.nit,
                'message': result.message
            }
            
            print(f"✓ Steering optimization successful!")
            print(f"  Thrust magnitude: {thrust_magnitude:.2e}")
            print(f"  Direction alignment: {alignment:.3f}")
            print(f"  Dipole strength: {steering_results['dipole_strength']:.3f}")
            print(f"  Optimization time: {optimization_time:.3f}s")
            
        else:
            steering_results = {
                'success': False,
                'message': result.message,
                'optimization_time': optimization_time
            }
            
            print(f"❌ Steering optimization failed: {result.message}")
        
        return steering_results
    
    def analyze_steering_performance(self, params: np.ndarray,
                                   direction_grid: List[np.ndarray] = None) -> Dict:
        """
        Analyze steering performance across multiple directions.
        
        Args:
            params: Optimized coil parameters
            direction_grid: List of direction vectors to test
            
        Returns:
            Steering performance analysis
        """
        if direction_grid is None:
            # Default direction grid (6 cardinal directions)
            direction_grid = [
                np.array([1.0, 0.0, 0.0]),   # +X
                np.array([-1.0, 0.0, 0.0]),  # -X
                np.array([0.0, 1.0, 0.0]),   # +Y
                np.array([0.0, -1.0, 0.0]),  # -Y
                np.array([0.0, 0.0, 1.0]),   # +Z
                np.array([0.0, 0.0, -1.0])   # -Z
            ]
        
        performance_analysis = {
            'directions': direction_grid,
            'thrust_magnitudes': [],
            'direction_alignments': [],
            'steering_efficiencies': []
        }
        
        print("📊 Analyzing steering performance across directions...")
        
        for i, direction in enumerate(direction_grid):
            # Compute momentum flux for this direction
            momentum_flux = self.compute_momentum_flux_vector(params, direction)
            thrust_magnitude = np.linalg.norm(momentum_flux)
            
            # Compute alignment
            if thrust_magnitude > 1e-12:
                thrust_direction = momentum_flux / thrust_magnitude
                direction_norm = direction / np.linalg.norm(direction)
                alignment = np.dot(thrust_direction, direction_norm)
            else:
                alignment = 0.0
            
            # Steering efficiency (thrust per unit dipole distortion)
            dipole_strength = params[3] if len(params) > 3 else 0.1
            efficiency = thrust_magnitude / (dipole_strength + 1e-12)
            
            performance_analysis['thrust_magnitudes'].append(thrust_magnitude)
            performance_analysis['direction_alignments'].append(alignment)
            performance_analysis['steering_efficiencies'].append(efficiency)
            
            direction_label = ['X+', 'X-', 'Y+', 'Y-', 'Z+', 'Z-'][i] if i < 6 else f'D{i}'
            print(f"  {direction_label}: |F⃗|={thrust_magnitude:.2e}, align={alignment:.3f}")
        
        # Summary statistics
        performance_analysis['mean_thrust'] = np.mean(performance_analysis['thrust_magnitudes'])
        performance_analysis['thrust_uniformity'] = np.std(performance_analysis['thrust_magnitudes'])
        performance_analysis['mean_alignment'] = np.mean(np.abs(performance_analysis['direction_alignments']))
        
        print(f"✓ Mean thrust: {performance_analysis['mean_thrust']:.2e}")
        print(f"✓ Mean alignment: {performance_analysis['mean_alignment']:.3f}")
        
        return performance_analysis

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
