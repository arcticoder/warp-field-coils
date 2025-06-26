#!/usr/bin/env python3
"""
Exotic Matter Energy Density Profile Computation
Based on warp-bubble-einstein-equations repository
"""

import numpy as np
import sympy as sp
from sympy import symbols, Function, sin, cos, Matrix, pi, simplify, diff
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Callable, Optional
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
