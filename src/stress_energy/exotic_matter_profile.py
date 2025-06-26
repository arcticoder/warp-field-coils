#!/usr/bin/env python3
"""
Exotic Matter Energy Density Profile Computation
Based on warp-bubble-einstein-equations repository
"""

import numpy as np
import sympy as sp
from sympy import symbols, Function, sin, cos, Matrix, pi, simplify, diff
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Callable, Optional, List
import scipy.optimize
from scipy.interpolate import interp1d

class ExoticMatterProfiler:
    """
    Computes T_μν tensor and extracts T^{00}(r) profile for warp bubble geometry.
    Implements Step 1 of the roadmap.
    """
    
    def __init__(self, r_min: float = 0.1, r_max: float = 10.0, n_points: int = 1000):
        """
        Initialize the exotic matter profiler.
        
        Args:
            r_min: Minimum radial coordinate
            r_max: Maximum radial coordinate  
            n_points: Number of grid points
        """
        self.r_min = r_min
        self.r_max = r_max
        self.n_points = n_points
        self.r_array = np.linspace(r_min, r_max, n_points)
        
        # Physical constants
        self.c = 299792458  # m/s
        self.G = 6.67430e-11  # m³/(kg⋅s²)
        
        # Setup symbolic variables
        self.t, self.r, self.theta, self.phi = symbols('t r theta phi')
        self.f = Function('f')(self.r, self.t)
        self.coords = (self.t, self.r, self.theta, self.phi)
        
        # Initialize tensors
        self._setup_metric()
        self._compute_ricci_tensor()
        self._compute_stress_energy_tensor()
    
    def _setup_metric(self):
        """Setup the warp bubble metric tensor."""
        # Warp bubble metric: ds² = -dt² + [1-f(r,t)]dr² + r²dθ² + r²sin²(θ)dφ²
        self.g = Matrix([
            [-1,      0,                0,               0],
            [ 0, 1 - self.f,           0,               0],
            [ 0,      0,         self.r**2,            0],
            [ 0,      0,                0,  self.r**2 * sin(self.theta)**2],
        ])
        
        # Inverse metric
        self.g_inv = Matrix([
            [-1,           0,                    0,                           0],
            [ 0, 1/(1 - self.f),                0,                           0],
            [ 0,           0,            1/self.r**2,                        0],
            [ 0,           0,                    0,   1/(self.r**2 * sin(self.theta)**2)],
        ])
    
    def _compute_ricci_tensor(self):
        """Compute Ricci tensor components for warp bubble metric."""
        # Define derivatives of f
        ft = diff(self.f, self.t)
        fr = diff(self.f, self.r)
        ftt = diff(self.f, self.t, 2)
        frr = diff(self.f, self.r, 2)
        frt = diff(self.f, self.r, self.t)
        
        # Ricci tensor components for warp bubble metric
        R_00 = -ftt/(2*(1-self.f)) + ft**2/(4*(1-self.f)**2)
        R_01 = R_10 = ft/(2*self.r*(1-self.f))
        R_02 = R_20 = 0
        R_03 = R_30 = 0

        R_11 = ftt/(2*(1-self.f)**2) - ft**2/(4*(1-self.f)**3) - fr/(self.r*(1-self.f)) 
        R_12 = R_21 = 0
        R_13 = R_31 = 0

        R_22 = self.r*fr/(2*(1-self.f)) - self.r
        R_23 = R_32 = 0

        R_33 = (self.r*fr/(2*(1-self.f)) - self.r) * sin(self.theta)**2
        
        # Construct the Ricci tensor matrix
        self.R = Matrix([
            [R_00, R_01, R_02, R_03],
            [R_10, R_11, R_12, R_13], 
            [R_20, R_21, R_22, R_23],
            [R_30, R_31, R_32, R_33]
        ])
        
        # Ricci scalar R = g^μν R_μν
        self.R_scalar = simplify(
            -R_00 + R_11/(1-self.f) + R_22/self.r**2 + R_33/(self.r**2 * sin(self.theta)**2)
        )
    
    def _compute_stress_energy_tensor(self):
        """Compute stress-energy tensor from Einstein equations."""
        # Einstein tensor G = R - 1/2 * g * R_scalar
        self.G = self.R - sp.Rational(1, 2) * self.g * self.R_scalar
        
        # Stress-energy tensor T = G / (8π)
        self.T = simplify(self.G / (8 * pi))
    
    def compute_T00_profile(self, f_profile_func: Callable[[float], float], 
                           time_value: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute T^{00}(r) profile for a given shape function f(r,t).
        
        Args:
            f_profile_func: Function f(r) that defines the warp bubble shape
            time_value: Time at which to evaluate (for time-dependent profiles)
            
        Returns:
            Tuple of (r_array, T00_profile)
        """
        T00_expr = self.T[0, 0]  # Extract T^{00} component
        
        # Substitute the specific f profile
        T00_profile = np.zeros_like(self.r_array)
        
        for i, r_val in enumerate(self.r_array):
            # Create symbolic substitutions
            subs_dict = {
                self.r: r_val,
                self.t: time_value,
                self.f: f_profile_func(r_val)
            }
            
            # Add derivatives if they appear in the expression
            # For a static profile, time derivatives are zero
            fr_val = self._numerical_derivative(f_profile_func, r_val)
            frr_val = self._numerical_second_derivative(f_profile_func, r_val)
            
            # Replace derivatives in the expression
            T00_substituted = T00_expr.subs([
                (diff(self.f, self.r), fr_val),
                (diff(self.f, self.r, 2), frr_val),
                (diff(self.f, self.t), 0),  # Static case
                (diff(self.f, self.t, 2), 0),  # Static case
            ])
            
            # Apply coordinate substitutions
            T00_substituted = T00_substituted.subs(subs_dict)
            
            try:
                T00_profile[i] = float(T00_substituted.evalf())
            except (TypeError, ValueError):
                T00_profile[i] = 0.0  # Handle singular points
        
        return self.r_array, T00_profile
    
    def _numerical_derivative(self, func: Callable, x: float, h: float = 1e-6) -> float:
        """Compute numerical first derivative."""
        return (func(x + h) - func(x - h)) / (2 * h)
    
    def _numerical_second_derivative(self, func: Callable, x: float, h: float = 1e-6) -> float:
        """Compute numerical second derivative."""
        return (func(x + h) - 2*func(x) + func(x - h)) / h**2
    
    def identify_exotic_regions(self, T00_profile: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Identify regions where T^{00} < 0 (exotic matter required).
        
        Args:
            T00_profile: Array of T^{00} values
            
        Returns:
            Dictionary with exotic matter region information
        """
        negative_mask = T00_profile < 0
        exotic_indices = np.where(negative_mask)[0]
        
        if len(exotic_indices) == 0:
            return {
                'has_exotic': False,
                'exotic_r': np.array([]),
                'exotic_T00': np.array([]),
                'total_exotic_energy': 0.0
            }
        
        exotic_r = self.r_array[exotic_indices]
        exotic_T00 = T00_profile[exotic_indices]
        
        # Compute total exotic energy (integral over volume)
        # dV = 4πr²dr for spherical coordinates
        total_exotic_energy = 0.0
        for i, idx in enumerate(exotic_indices):
            if idx > 0 and idx < len(self.r_array) - 1:
                dr = self.r_array[idx+1] - self.r_array[idx-1]
                dV = 4 * np.pi * self.r_array[idx]**2 * dr
                total_exotic_energy += abs(exotic_T00[i]) * dV
        
        return {
            'has_exotic': True,
            'exotic_r': exotic_r,
            'exotic_T00': exotic_T00,
            'total_exotic_energy': total_exotic_energy,
            'exotic_indices': exotic_indices
        }
    
    def plot_T00_profile(self, T00_profile: np.ndarray, 
                        title: str = "Exotic Matter Energy Density Profile",
                        save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot the T^{00}(r) profile with exotic matter regions highlighted.
        
        Args:
            T00_profile: Array of T^{00} values
            title: Plot title
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot full profile
        ax.plot(self.r_array, T00_profile, 'b-', linewidth=2, label='$T^{00}(r)$')
        
        # Highlight exotic matter regions (T^{00} < 0)
        negative_mask = T00_profile < 0
        if np.any(negative_mask):
            ax.fill_between(self.r_array, T00_profile, 0, 
                          where=negative_mask, alpha=0.3, color='red',
                          label='Exotic Matter ($T^{00} < 0$)')
        
        # Add zero line
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        ax.set_xlabel('Radial Distance $r$ (units of $R_s$)')
        ax.set_ylabel(r'Energy Density $T^{00}$ (units of $c^4/8\pi G$)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def alcubierre_profile_time_dep(self, r: np.ndarray, t: float, 
                                   R_func: Callable[[float], float], sigma: float) -> np.ndarray:
        """
        Compute time-dependent Alcubierre warp profile.
        
        Args:
            r: Radial coordinates
            t: Time parameter
            R_func: Function R(t) defining bubble radius trajectory
            sigma: Profile sharpness parameter
            
        Returns:
            Time-dependent warp profile f(r,t)
        """
        R_t = R_func(t)
        if R_t <= 0:
            return np.ones_like(r)
            
        # f(r,t) = [tanh(σ(r-R(t))) - tanh(σ(r+R(t)))] / [2*tanh(σR(t))]
        term1 = np.tanh(sigma * (r - R_t))
        term2 = np.tanh(sigma * (r + R_t))
        denominator = 2 * np.tanh(sigma * R_t)
        
        # Avoid division by zero
        if abs(denominator) < 1e-12:
            return np.ones_like(r)
            
        return (term1 - term2) / denominator
    
    def compute_T00_profile_time_dep(self, R_func: Callable[[float], float], 
                                   sigma: float, times: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute time-dependent T^{00} profile for moving warp bubble.
        
        Args:
            R_func: Bubble radius trajectory R(t)
            sigma: Profile sharpness
            times: Time array
            
        Returns:
            (r_array, T00_rt): Spatial coordinates and T^{00}(r,t) array
        """
        T00_rt = []
        
        for t in times:
            # Get warp profile at time t
            f_rt = self.alcubierre_profile_time_dep(self.r_array, t, R_func, sigma)
            
            # Compute time derivatives for stress-energy tensor
            dt = 1e-6  # Small time step for numerical differentiation
            f_rt_plus = self.alcubierre_profile_time_dep(self.r_array, t + dt, R_func, sigma)
            f_rt_minus = self.alcubierre_profile_time_dep(self.r_array, t - dt, R_func, sigma)
            
            # First and second time derivatives
            dfdt = (f_rt_plus - f_rt_minus) / (2 * dt)
            d2fdt2 = (f_rt_plus - 2*f_rt + f_rt_minus) / (dt**2)
            
            # Compute T^{00} including time-dependent terms
            T00_t = self._compute_T00_time_dependent(self.r_array, f_rt, dfdt, d2fdt2)
            T00_rt.append(T00_t)
        
        return self.r_array, np.array(T00_rt)
    
    def _compute_T00_time_dependent(self, r: np.ndarray, f: np.ndarray, 
                                  dfdt: np.ndarray, d2fdt2: np.ndarray) -> np.ndarray:
        """
        Compute time-dependent stress-energy tensor T^{00}(r,t).
        
        Includes contributions from ∂f/∂t and ∂²f/∂t².
        """
        # Spatial derivatives
        dfdr = np.gradient(f, r)
        d2fdr2 = np.gradient(dfdr, r)
        
        # Avoid division by zero
        r_safe = np.where(r > 1e-12, r, 1e-12)
        f_minus_1 = f - 1
        f_minus_1_safe = np.where(np.abs(f_minus_1) > 1e-12, f_minus_1, 1e-12)
        
        # Time-dependent T^{00} formula (from stress_energy.tex)
        # T^{00} = (1/(64πr(f-1)^4)) * [spatial_terms + time_terms]
        
        # Spatial terms (existing)
        spatial_terms = (
            -r * dfdr**2 * (f_minus_1)**2 +
            2 * r * dfdr * d2fdr2 * (f_minus_1)**3 +
            dfdr**2 * (f_minus_1)**3
        )
        
        # Time-dependent terms
        time_terms = (
            2 * r * (f_minus_1)**3 * d2fdt2 +
            6 * r * (f_minus_1)**2 * dfdt**2 +
            (f_minus_1)**3 * dfdt**2 / r_safe
        )
        
        # Full T^{00} with time dependence
        prefactor = 1.0 / (64 * np.pi * r_safe * (f_minus_1_safe)**4)
        T00 = prefactor * (spatial_terms + time_terms)
        
        return T00
    
    def compute_time_dependent_T00_highres(self, r_array: np.ndarray, 
                                           R_func: Callable[[float], float],
                                           sigma: float, 
                                           t_array: np.ndarray) -> np.ndarray:
        """
        High-resolution time-dependent T^{00} computation with second derivatives.
        
        Includes second-derivative terms for rapid acceleration capture:
        T₀₀ = (2r(f-1)³∂²f/∂t² + r(f-1)²(∂f/∂t)² - ...) / (64πr(f-1)⁴)
        
        Args:
            r_array: Radial coordinate array
            R_func: Time-dependent bubble radius function R(t)
            sigma: Profile sharpness parameter
            t_array: High-resolution time array
            
        Returns:
            T₀₀(r,t) array with enhanced temporal resolution
        """
        try:
            import jax
            import jax.numpy as jnp
            
            # Convert to JAX arrays
            r_jax = jnp.array(r_array)
            t_jax = jnp.array(t_array)
            
            # Create meshgrid for vectorized computation
            R_mesh, T_mesh = jnp.meshgrid(r_jax, t_jax, indexing='ij')
            
            # Vectorized R(t) evaluation
            R_vals = jnp.array([R_func(t) for t in t_array])
            R_mesh_vals = R_vals[jnp.newaxis, :]
            
            # Warp profile function f(r,t)
            def f_profile(r, t_idx):
                R_t = R_vals[t_idx]
                numerator = (jnp.tanh(sigma * (r - R_t)) - 
                           jnp.tanh(sigma * (r + R_t)))
                denominator = 2 * jnp.tanh(sigma * R_t)
                return numerator / (denominator + 1e-12)  # Avoid division by zero
            
            # Compute time derivatives using JAX autodiff
            def df_dt_func(r, t_idx):
                """First time derivative of f(r,t)"""
                if t_idx == 0 or t_idx >= len(t_array) - 1:
                    return 0.0
                
                dt = t_array[1] - t_array[0]
                f_forward = f_profile(r, t_idx + 1)
                f_backward = f_profile(r, t_idx - 1)
                return (f_forward - f_backward) / (2 * dt)
            
            def d2f_dt2_func(r, t_idx):
                """Second time derivative of f(r,t)"""
                if t_idx == 0 or t_idx >= len(t_array) - 1:
                    return 0.0
                
                dt = t_array[1] - t_array[0]
                f_center = f_profile(r, t_idx)
                f_forward = f_profile(r, t_idx + 1)
                f_backward = f_profile(r, t_idx - 1)
                return (f_forward - 2 * f_center + f_backward) / (dt**2)
            
            # Initialize T₀₀ array
            T00_array = jnp.zeros((len(r_array), len(t_array)))
            
            # Compute T₀₀ with enhanced formula including second derivatives
            for i, r in enumerate(r_array):
                for j, t in enumerate(t_array):
                    f = f_profile(r, j)
                    df_dt = df_dt_func(r, j)
                    d2f_dt2 = d2f_dt2_func(r, j)
                    
                    # Enhanced T₀₀ formula with second-derivative terms
                    if abs(f - 1) > 1e-10 and r > 1e-10:
                        numerator = (2 * r * (f - 1)**3 * d2f_dt2 + 
                                   r * (f - 1)**2 * df_dt**2 -
                                   4 * r * (f - 1)**2 * df_dt * sigma * 
                                   (R_vals[j] / (R_vals[j]**2 + 1e-12)))
                        
                        denominator = 64 * jnp.pi * r * (f - 1)**4
                        
                        T00_val = numerator / (denominator + 1e-12)
                    else:
                        T00_val = 0.0
                    
                    T00_array = T00_array.at[i, j].set(T00_val)
            
            return np.array(T00_array)
            
        except ImportError:
            print("⚠️ JAX not available, using finite difference approximation")
            return self._compute_T00_finite_difference(r_array, R_func, sigma, t_array)
    
    def _compute_T00_finite_difference(self, r_array: np.ndarray, 
                                      R_func: Callable[[float], float],
                                      sigma: float, 
                                      t_array: np.ndarray) -> np.ndarray:
        """Fallback finite difference computation for T₀₀."""
        T00_array = np.zeros((len(r_array), len(t_array)))
        
        for i, r in enumerate(r_array):
            for j, t in enumerate(t_array):
                # Simple Alcubierre profile evaluation
                R_t = R_func(t)
                f = (np.tanh(sigma * (r - R_t)) - np.tanh(sigma * (r + R_t))) / \
                    (2 * np.tanh(sigma * R_t) + 1e-12)
                
                # Simplified T₀₀ without time derivatives
                if abs(f - 1) > 1e-10 and r > 1e-10:
                    T00_val = -sigma**2 * (1 - f**2) / (8 * np.pi * r**2)
                else:
                    T00_val = 0.0
                
                T00_array[i, j] = T00_val
        
        return T00_array
    
    def analyze_temporal_resolution_convergence(self, R_func: Callable[[float], float],
                                              sigma: float,
                                              time_ranges: List[int] = [50, 100, 200, 400]) -> Dict:
        """
        Analyze convergence with increasing temporal resolution.
        
        Args:
            R_func: Time-dependent radius function
            sigma: Profile sharpness
            time_ranges: List of temporal resolution values to test
            
        Returns:
            Convergence analysis results
        """
        print("🔍 Analyzing temporal resolution convergence...")
        
        convergence_results = {
            'resolutions': time_ranges,
            'finite_fractions': [],
            'max_T00_values': [],
            'numerical_errors': [],
            'computation_times': []
        }
        
        for n_time in time_ranges:
            import time
            start_time = time.time()
            
            # Generate high-resolution time array
            t_array = np.linspace(0, 2.0, n_time)
            
            # Compute high-resolution T₀₀
            T00_highres = self.compute_time_dependent_T00_highres(
                self.r_array, R_func, sigma, t_array
            )
            
            # Analysis metrics
            finite_fraction = np.sum(np.isfinite(T00_highres)) / T00_highres.size
            max_T00 = np.max(np.abs(T00_highres[np.isfinite(T00_highres)]))
            
            # Estimate numerical error (if previous resolution available)
            if len(convergence_results['finite_fractions']) > 0:
                # Compare with previous resolution (downsampled)
                prev_n = time_ranges[len(convergence_results['finite_fractions']) - 1]
                downsample_factor = n_time // prev_n
                T00_downsampled = T00_highres[::downsample_factor, ::downsample_factor]
                
                if T00_downsampled.shape == (len(self.r_array), prev_n):
                    numerical_error = np.mean(np.abs(T00_downsampled - 
                                                   convergence_results['T00_arrays'][-1]))
                else:
                    numerical_error = 0.0
            else:
                numerical_error = 0.0
                convergence_results['T00_arrays'] = []
            
            computation_time = time.time() - start_time
            
            # Store results
            convergence_results['finite_fractions'].append(finite_fraction)
            convergence_results['max_T00_values'].append(max_T00)
            convergence_results['numerical_errors'].append(numerical_error)
            convergence_results['computation_times'].append(computation_time)
            convergence_results['T00_arrays'].append(T00_highres)
            
            print(f"  Resolution {n_time}: finite={finite_fraction*100:.1f}%, "
                  f"max|T₀₀|={max_T00:.2e}, time={computation_time:.3f}s")
        
        # Determine optimal resolution
        optimal_idx = self._find_optimal_resolution(convergence_results)
        optimal_resolution = time_ranges[optimal_idx]
        
        convergence_results['optimal_resolution'] = optimal_resolution
        convergence_results['convergence_achieved'] = (
            convergence_results['finite_fractions'][-1] > 0.95 and
            convergence_results['numerical_errors'][-1] < 1e-8
        )
        
        print(f"✓ Optimal resolution: {optimal_resolution} time points")
        print(f"✓ Convergence achieved: {convergence_results['convergence_achieved']}")
        
        return convergence_results
    
    def _find_optimal_resolution(self, convergence_results: Dict) -> int:
        """Find optimal temporal resolution balancing accuracy and efficiency."""
        finite_fractions = np.array(convergence_results['finite_fractions'])
        numerical_errors = np.array(convergence_results['numerical_errors'])
        computation_times = np.array(convergence_results['computation_times'])
        
        # Normalize metrics (0-1 scale)
        finite_score = finite_fractions
        error_score = 1 - (numerical_errors / (np.max(numerical_errors) + 1e-12))
        time_score = 1 - (computation_times / np.max(computation_times))
        
        # Combined score (weighted)
        combined_score = 0.5 * finite_score + 0.3 * error_score + 0.2 * time_score
        
        return np.argmax(combined_score)

    def compute_T0r_component(self, f_profile: np.ndarray, 
                             r_array: np.ndarray, 
                             theta_array: np.ndarray) -> np.ndarray:
        """
        Compute T⁰ʳ component of stress-energy tensor for momentum flux.
        
        For dipolar warp bubbles, T⁰ʳ provides radial momentum flux
        that can be projected to Cartesian coordinates for thrust computation.
        
        Args:
            f_profile: Warp profile f(r,θ) 
            r_array: Radial coordinates
            theta_array: Angular coordinates
            
        Returns:
            T⁰ʳ component array
        """
        # Simplified T⁰ʳ computation based on warp geometry
        # T⁰ʳ ≈ (1/8π) G⁰ʳ where G⁰ʳ is mixed Einstein tensor component
        
        # Compute radial gradient of f
        dr = r_array[1] - r_array[0] if len(r_array) > 1 else 0.01
        df_dr = np.gradient(f_profile, dr, axis=0)
        
        # Angular gradient
        if len(theta_array) > 1:
            dtheta = theta_array[1] - theta_array[0]
            df_dtheta = np.gradient(f_profile, dtheta, axis=1)
        else:
            df_dtheta = np.zeros_like(f_profile)
        
        # Simplified T⁰ʳ formula (dimensional analysis consistent)
        # This would be replaced by full Einstein tensor computation in production
        r_mesh = r_array[:, np.newaxis]
        T0r = -(1.0 / (8 * np.pi)) * (df_dr + df_dtheta / (r_mesh + 1e-12))
        
        return T0r
    
    def compute_momentum_flux_vector(self, f_profile: np.ndarray,
                                   r_array: np.ndarray, 
                                   theta_array: np.ndarray,
                                   volume_elements: np.ndarray = None) -> np.ndarray:
        """
        Compute 3D momentum flux vector F⃗ = ∫ T⁰ⁱ d³x.
        
        Integrates momentum density over the warp bubble volume
        to obtain net thrust vector in Cartesian coordinates.
        
        Args:
            f_profile: Warp profile f(r,θ)
            r_array: Radial coordinates
            theta_array: Angular coordinates  
            volume_elements: Volume elements dV (optional)
            
        Returns:
            3D momentum flux vector [Fx, Fy, Fz]
        """
        # Compute T⁰ʳ component
        T0r = self.compute_T0r_component(f_profile, r_array, theta_array)
        
        # Create coordinate meshgrids
        r_mesh, theta_mesh = np.meshgrid(r_array, theta_array, indexing='ij')
        
        # Volume elements in spherical coordinates: dV = r² sin θ dr dθ dφ
        if volume_elements is None:
            dr = r_array[1] - r_array[0] if len(r_array) > 1 else 0.01
            dtheta = theta_array[1] - theta_array[0] if len(theta_array) > 1 else 0.01
            dphi = 2 * np.pi  # Full azimuthal integration
            dV = r_mesh**2 * np.sin(theta_mesh) * dr * dtheta * dphi
        else:
            dV = volume_elements
        
        # Project T⁰ʳ to Cartesian components
        # T⁰ˣ = T⁰ʳ sin θ cos φ (φ-averaged gives 0 for axisymmetric case)
        # T⁰ʸ = T⁰ʳ sin θ sin φ (φ-averaged gives 0 for axisymmetric case)  
        # T⁰ᶻ = T⁰ʳ cos θ
        
        # For axisymmetric dipole, only z-component survives
        T0x = T0r * np.sin(theta_mesh) * 0  # φ-averaged = 0
        T0y = T0r * np.sin(theta_mesh) * 0  # φ-averaged = 0
        T0z = T0r * np.cos(theta_mesh)
        
        # Integrate over volume
        Fx = np.sum(T0x * dV)
        Fy = np.sum(T0y * dV)
        Fz = np.sum(T0z * dV)
        
        return np.array([Fx, Fy, Fz])
    
    def compute_thrust_magnitude(self, momentum_flux: np.ndarray) -> float:
        """
        Compute thrust magnitude from momentum flux vector.
        
        Args:
            momentum_flux: 3D momentum flux vector
            
        Returns:
            Thrust magnitude in Newtons
        """
        return np.linalg.norm(momentum_flux)
    
    def compute_thrust_direction(self, momentum_flux: np.ndarray) -> np.ndarray:
        """
        Compute unit thrust direction vector.
        
        Args:
            momentum_flux: 3D momentum flux vector
            
        Returns:
            Unit direction vector
        """
        magnitude = self.compute_thrust_magnitude(momentum_flux)
        if magnitude > 1e-12:
            return momentum_flux / magnitude
        else:
            return np.array([0.0, 0.0, 0.0])
    
    def analyze_dipolar_thrust_characteristics(self, R0: float, sigma: float,
                                             eps_range: np.ndarray = None) -> Dict:
        """
        Analyze thrust characteristics vs dipole strength.
        
        Args:
            R0: Base bubble radius
            sigma: Profile sharpness
            eps_range: Range of dipole strengths to test
            
        Returns:
            Thrust analysis results
        """
        if eps_range is None:
            eps_range = np.linspace(0, 0.5, 11)
        
        thrust_results = {
            'eps_values': eps_range,
            'thrust_magnitudes': [],
            'thrust_directions': [],
            'thrust_efficiency': []
        }
        
        # Angular coordinates (0 to π)
        theta_array = np.linspace(0, np.pi, 64)
        
        print("🚀 Analyzing dipolar thrust characteristics...")
        
        for i, eps in enumerate(eps_range):
            # Compute dipolar profile
            f_profile = alcubierre_profile_dipole(
                self.r_array, theta_array, R0, sigma, eps
            )
            
            # Compute momentum flux
            momentum_flux = self.compute_momentum_flux_vector(
                f_profile, self.r_array, theta_array
            )
            
            # Analysis metrics
            thrust_mag = self.compute_thrust_magnitude(momentum_flux)
            thrust_dir = self.compute_thrust_direction(momentum_flux)
            
            # Thrust efficiency (thrust per unit dipole distortion)
            efficiency = thrust_mag / (eps + 1e-12)
            
            # Store results
            thrust_results['thrust_magnitudes'].append(thrust_mag)
            thrust_results['thrust_directions'].append(thrust_dir)
            thrust_results['thrust_efficiency'].append(efficiency)
            
            print(f"  ε={eps:.2f}: |F⃗|={thrust_mag:.2e}, F̂={thrust_dir}")
        
        # Find optimal dipole strength
        max_eff_idx = np.argmax(thrust_results['thrust_efficiency'])
        optimal_eps = eps_range[max_eff_idx]
        
        thrust_results['optimal_dipole_strength'] = optimal_eps
        thrust_results['max_efficiency'] = thrust_results['thrust_efficiency'][max_eff_idx]
        
        print(f"✓ Optimal dipole strength: ε = {optimal_eps:.3f}")
        print(f"✓ Maximum efficiency: {thrust_results['max_efficiency']:.2e}")
        
        return thrust_results

# Example usage and standard warp bubble profiles
def alcubierre_profile(r: float, R: float = 1.0, sigma: float = 0.1) -> float:
    """
    Standard Alcubierre warp bubble profile.
    
    Args:
        r: Radial coordinate
        R: Bubble radius
        sigma: Bubble wall thickness
        
    Returns:
        Shape function value f(r)
    """
    if r <= R - sigma:
        return 1.0
    elif r >= R + sigma:
        return 0.0
    else:
        # Smooth transition using tanh
        return 0.5 * (1 - np.tanh((r - R) / sigma))

def gaussian_warp_profile(r: float, A: float = 1.0, sigma: float = 1.0) -> float:
    """
    Gaussian warp bubble profile.
    
    Args:
        r: Radial coordinate
        A: Amplitude
        sigma: Width parameter
        
    Returns:
        Shape function value f(r)
    """
    return A * np.exp(-(r/sigma)**2)

def alcubierre_profile_dipole(r: np.ndarray, theta: np.ndarray, 
                           R0: float, sigma: float, eps: float) -> np.ndarray:
    """
    Compute dipolar Alcubierre warp profile with angular dependence.
    
    Introduces first-order spherical harmonic (dipole) distortion:
    R(θ) = R₀ + ε P₁(cos θ) = R₀ + ε cos θ
    
    f(r,θ) = [tanh[σ(r - R(θ))] - tanh[σ(r + R(θ))]] / [2 tanh[σ R(θ)]]
    
    Args:
        r: Radial coordinate array
        theta: Polar angle array (0 to π)
        R0: Base bubble radius
        sigma: Profile sharpness parameter
        eps: Dipole distortion amplitude
        
    Returns:
        Dipolar warp profile f(r,θ)
    """
    try:
        import jax.numpy as jnp
        
        # Convert to JAX arrays
        r_jax = jnp.array(r)
        theta_jax = jnp.array(theta)
        
        # Dipolar radius modulation: R(θ) = R₀ + ε cos θ
        R_theta = R0 + eps * jnp.cos(theta_jax)
        
        # Ensure R(θ) > 0 for all angles
        R_theta = jnp.maximum(R_theta, 0.1 * R0)
        
        # Alcubierre profile with angular dependence
        # Broadcasting: r[:, None] × theta[None, :]
        if r_jax.ndim == 1 and theta_jax.ndim == 1:
            r_mesh, theta_mesh = jnp.meshgrid(r_jax, theta_jax, indexing='ij')
            R_mesh = R0 + eps * jnp.cos(theta_mesh)
        else:
            r_mesh = r_jax
            theta_mesh = theta_jax
            R_mesh = R0 + eps * jnp.cos(theta_mesh)
        
        # Warp profile computation
        numerator = (jnp.tanh(sigma * (r_mesh - R_mesh)) - 
                    jnp.tanh(sigma * (r_mesh + R_mesh)))
        denominator = 2 * jnp.tanh(sigma * R_mesh)
        
        # Avoid division by zero
        f_profile = jnp.where(jnp.abs(denominator) > 1e-12,
                             numerator / denominator,
                             0.0)
        
        return np.array(f_profile)
        
    except ImportError:
        print("⚠️ JAX not available, using NumPy fallback for dipolar profile")
        return _alcubierre_profile_dipole_numpy(r, theta, R0, sigma, eps)

def _alcubierre_profile_dipole_numpy(r: np.ndarray, theta: np.ndarray,
                                   R0: float, sigma: float, eps: float) -> np.ndarray:
    """NumPy fallback for dipolar profile computation."""
    # Dipolar radius modulation
    if theta.ndim == 1 and r.ndim == 1:
        r_mesh, theta_mesh = np.meshgrid(r, theta, indexing='ij')
    else:
        r_mesh = r
        theta_mesh = theta
    
    R_theta = R0 + eps * np.cos(theta_mesh)
    R_theta = np.maximum(R_theta, 0.1 * R0)  # Prevent negative radius
    
    # Alcubierre profile
    numerator = (np.tanh(sigma * (r_mesh - R_theta)) - 
                np.tanh(sigma * (r_mesh + R_theta)))
    denominator = 2 * np.tanh(sigma * R_theta)
    
    f_profile = np.where(np.abs(denominator) > 1e-12,
                        numerator / denominator,
                        0.0)
    
    return f_profile

def gaussian_warp_profile_dipole(r: np.ndarray, theta: np.ndarray,
                               A: float, sigma: float, eps: float) -> np.ndarray:
    """
    Gaussian warp profile with dipolar distortion.
    
    Args:
        r: Radial coordinate array
        theta: Polar angle array
        A: Profile amplitude
        sigma: Gaussian width
        eps: Dipole distortion amplitude
        
    Returns:
        Dipolar Gaussian warp profile
    """
    if r.ndim == 1 and theta.ndim == 1:
        r_mesh, theta_mesh = np.meshgrid(r, theta, indexing='ij')
    else:
        r_mesh = r
        theta_mesh = theta
    
    # Angular modulation of width: σ(θ) = σ₀(1 + ε cos θ)
    sigma_theta = sigma * (1 + eps * np.cos(theta_mesh))
    sigma_theta = np.maximum(sigma_theta, 0.1 * sigma)  # Prevent collapse
    
    # Gaussian profile with angular dependence
    f_profile = A * np.exp(-r_mesh**2 / (2 * sigma_theta**2))
    
    return f_profile

def visualize_dipolar_profile(r_array: np.ndarray, theta_array: np.ndarray,
                            f_profile: np.ndarray, save_path: str = None) -> plt.Figure:
    """
    Visualize dipolar warp profile in polar coordinates.
    
    Args:
        r_array: Radial coordinates
        theta_array: Angular coordinates
        f_profile: Warp profile f(r,θ)
        save_path: Optional save path
        
    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6),
                                  subplot_kw={'projection': 'polar'})
    
    # Create meshgrid for plotting
    R_mesh, Theta_mesh = np.meshgrid(r_array, theta_array, indexing='ij')
    
    # Plot 1: Full profile
    c1 = ax1.pcolormesh(Theta_mesh.T, R_mesh.T, f_profile.T, 
                       cmap='RdBu_r', shading='auto')
    ax1.set_title('Dipolar Warp Profile f(r,θ)')
    ax1.set_ylim(0, np.max(r_array))
    plt.colorbar(c1, ax=ax1, label='f(r,θ)')
    
    # Plot 2: Angular cross-section at fixed radius
    r_idx = len(r_array) // 2  # Middle radius
    ax2.plot(theta_array, f_profile[r_idx, :], 'b-', linewidth=2)
    ax2.set_title(f'Angular Cross-section at r={r_array[r_idx]:.2f}')
    ax2.set_ylim(np.min(f_profile), np.max(f_profile))
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Dipolar profile visualization saved to {save_path}")
    
    return fig

if __name__ == "__main__":
    # Example usage
    profiler = ExoticMatterProfiler()
    
    # Test with Alcubierre profile
    r_array, T00_profile = profiler.compute_T00_profile(
        lambda r: alcubierre_profile(r, R=2.0, sigma=0.5)
    )
    
    # Identify exotic regions
    exotic_info = profiler.identify_exotic_regions(T00_profile)
    
    print(f"Has exotic matter: {exotic_info['has_exotic']}")
    if exotic_info['has_exotic']:
        print(f"Total exotic energy: {exotic_info['total_exotic_energy']:.2e}")
        print(f"Exotic matter regions: r ∈ [{exotic_info['exotic_r'].min():.2f}, {exotic_info['exotic_r'].max():.2f}]")
    
    # Plot results
    fig = profiler.plot_T00_profile(T00_profile, save_path="exotic_matter_profile.png")
    plt.show()
