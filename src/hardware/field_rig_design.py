#!/usr/bin/env python3
"""
Electromagnetic Field Simulation and Safety Analysis
Implements Step 3 of the roadmap: simulate electromagnetic performance & safety margins
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import scipy.constants as const
from scipy.integrate import quad
import warnings

# Physical constants and safety limits (from field_rig_design.py)
E_BREAKDOWN = 1e14     # Dielectric breakdown (V/m)
B_MAX_SAFE = 100       # Maximum safe B-field (T)
I_MAX_SAFE = 1e6       # Maximum safe current (A)
V_MAX_SAFE = 1e7       # Maximum safe voltage (V)
μ0 = const.mu_0        # Permeability of free space (H/m)
ε0 = const.epsilon_0   # Permittivity of free space (F/m)
c = const.c            # Speed of light (m/s)

@dataclass
class FieldRigResults:
    """Results from electromagnetic field simulation."""
    r_char: float          # Characteristic radius (m)
    B_peak: float          # Peak magnetic field (T)
    E_peak: float          # Peak electric field (V/m)
    rho_B: float          # Magnetic energy density (J/m³)
    rho_E: float          # Electric energy density (J/m³)
    stored_energy: float   # Total stored energy (J)
    power_dissipation: float  # Power dissipation (W)
    inductance: float      # Total inductance (H)
    resistance: float      # Total resistance (Ω)
    safety_margins: Dict[str, float]  # Safety factor for each limit

class ElectromagneticFieldSimulator:
    """
    Advanced electromagnetic field simulator for warp field coils.
    Implements Step 3 of the roadmap.
    """
    
    def __init__(self):
        """Initialize the electromagnetic field simulator."""
        # Material properties (superconductor assumptions)
        self.wire_resistivity = 1e-12  # Ω⋅m (superconducting)
        self.wire_critical_current_density = 1e9  # A/m² (high-Tc superconductor)
        self.mu_r = 1.0  # Relative permeability (non-magnetic)
        
    def simulate_inductive_rig(self, L: float, I: float, f_mod: float,
                             mu_r: float = 1.0, geometry: str = 'solenoid',
                             wire_radius: float = 1e-3) -> FieldRigResults:
        """
        Simulate electromagnetic fields for inductive coil rig.
        
        Args:
            L: Inductance (H)
            I: Current (A)
            f_mod: Modulation frequency (Hz)
            mu_r: Relative permeability
            geometry: Coil geometry ('solenoid', 'toroidal', 'planar')
            wire_radius: Wire radius (m)
            
        Returns:
            FieldRigResults with complete electromagnetic analysis
        """
        # Characteristic dimensions based on geometry
        if geometry == 'solenoid':
            # Solenoid approximation: r ≈ √(L/μ₀μᵣ)
            r_char = np.sqrt(L / (μ0 * mu_r))
            length_factor = 1.0
        elif geometry == 'toroidal':
            # Toroidal approximation (more complex)
            r_char = np.sqrt(L / (μ0 * mu_r)) * 0.8  # Correction factor
            length_factor = 2 * np.pi  # Circumferential
        elif geometry == 'planar':
            # Planar spiral coil
            r_char = np.sqrt(L / (μ0 * mu_r)) * 1.2  # Correction factor
            length_factor = 0.5
        else:
            r_char = np.sqrt(L / (μ0 * mu_r))
            length_factor = 1.0
        
        # Peak magnetic field
        if r_char > 0:
            if geometry == 'solenoid':
                # B = μ₀nI for solenoid center
                n_turns = np.sqrt(L * r_char / μ0)  # Approximate turn density
                B_peak = μ0 * mu_r * n_turns * I / r_char
            elif geometry == 'toroidal':
                # B = μ₀nI/(2πr) for toroidal center
                B_peak = μ0 * mu_r * I / (2 * np.pi * r_char)
            else:
                # General approximation
                B_peak = μ0 * mu_r * I / (2 * np.pi * r_char)
        else:
            B_peak = 0
        
        # Electric field from time-varying magnetic field (Faraday's law)
        # E = -dΦ/dt, with Φ = B⋅A and B(t) = B₀sin(2πft)
        E_peak = 2 * np.pi * f_mod * B_peak * r_char
        
        # Energy densities
        rho_B = B_peak**2 / (2 * μ0 * mu_r)  # Magnetic energy density
        rho_E = ε0 * E_peak**2 / 2            # Electric energy density
        
        # Total stored energy
        stored_energy_magnetic = 0.5 * L * I**2
        volume_estimate = np.pi * r_char**2 * r_char * length_factor
        stored_energy_electric = rho_E * volume_estimate
        stored_energy = stored_energy_magnetic + stored_energy_electric
        
        # Resistance and power dissipation
        wire_length = self._estimate_wire_length(r_char, geometry)
        wire_area = np.pi * wire_radius**2
        resistance = self.wire_resistivity * wire_length / wire_area
        power_dissipation = resistance * I**2
        
        # Calculate safety margins
        safety_margins = {
            'magnetic_field': B_MAX_SAFE / B_peak if B_peak > 0 else np.inf,
            'electric_field': E_BREAKDOWN / E_peak if E_peak > 0 else np.inf,
            'current': I_MAX_SAFE / I if I > 0 else np.inf,
            'voltage': V_MAX_SAFE / (E_peak * r_char) if E_peak > 0 else np.inf
        }
        
        return FieldRigResults(
            r_char=r_char,
            B_peak=B_peak,
            E_peak=E_peak,
            rho_B=rho_B,
            rho_E=rho_E,
            stored_energy=stored_energy,
            power_dissipation=power_dissipation,
            inductance=L,
            resistance=resistance,
            safety_margins=safety_margins
        )
    
    def _estimate_wire_length(self, r_char: float, geometry: str) -> float:
        """Estimate total wire length for given geometry."""
        if geometry == 'solenoid':
            # Approximate: N turns, each of circumference 2πr
            N_turns = r_char / (2e-3)  # Assume 2mm spacing
            return N_turns * 2 * np.pi * r_char
        elif geometry == 'toroidal':
            # Toroidal: multiple loops around torus
            N_turns = r_char / (1e-3)  # Assume 1mm spacing
            return N_turns * 2 * np.pi * r_char
        else:
            # Default estimate
            return 2 * np.pi * r_char**2
    
    def safety_analysis(self, results: FieldRigResults) -> Dict[str, str]:
        """
        Perform comprehensive safety analysis.
        
        Args:
            results: FieldRigResults from simulation
            
        Returns:
            Dictionary with safety status for each parameter
        """
        safety_status = {}
        
        # Check each safety margin
        for param, margin in results.safety_margins.items():
            if margin > 10.0:
                safety_status[param] = "SAFE (>10x margin)"
            elif margin > 2.0:
                safety_status[param] = "CAUTION (2-10x margin)"
            elif margin > 1.0:
                safety_status[param] = "WARNING (<2x margin)"
            else:
                safety_status[param] = "DANGER (exceeded limit)"
        
        return safety_status
    
    def sweep_current_geometry(self, L_range: Tuple[float, float], 
                             I_range: Tuple[float, float],
                             n_points: int = 50) -> Dict:
        """
        Sweep current and inductance to map safe operating envelope.
        
        Args:
            L_range: (L_min, L_max) inductance range (H)
            I_range: (I_min, I_max) current range (A)
            n_points: Number of points in each dimension
            
        Returns:
            Dictionary with sweep results and safe operating region
        """
        L_vals = np.linspace(L_range[0], L_range[1], n_points)
        I_vals = np.linspace(I_range[0], I_range[1], n_points)
        
        L_grid, I_grid = np.meshgrid(L_vals, I_vals)
        
        # Initialize result arrays
        B_peak_grid = np.zeros_like(L_grid)
        E_peak_grid = np.zeros_like(L_grid)
        safety_grid = np.zeros_like(L_grid)  # Minimum safety margin
        
        f_mod = 1000.0  # 1 kHz modulation frequency
        
        for i in range(n_points):
            for j in range(n_points):
                L = L_grid[i, j]
                I = I_grid[i, j]
                
                try:
                    results = self.simulate_inductive_rig(L, I, f_mod)
                    B_peak_grid[i, j] = results.B_peak
                    E_peak_grid[i, j] = results.E_peak
                    
                    # Minimum safety margin across all parameters
                    min_margin = min(results.safety_margins.values())
                    safety_grid[i, j] = min_margin
                    
                except Exception as e:
                    B_peak_grid[i, j] = np.nan
                    E_peak_grid[i, j] = np.nan
                    safety_grid[i, j] = 0.0
        
        # Identify safe operating region (safety margin > 2.0)
        safe_mask = safety_grid > 2.0
        
        return {
            'L_grid': L_grid,
            'I_grid': I_grid,
            'B_peak_grid': B_peak_grid,
            'E_peak_grid': E_peak_grid,
            'safety_grid': safety_grid,
            'safe_mask': safe_mask,
            'safe_L_range': (L_vals[safe_mask].min(), L_vals[safe_mask].max()) if np.any(safe_mask) else (0, 0),
            'safe_I_range': (I_vals[safe_mask].min(), I_vals[safe_mask].max()) if np.any(safe_mask) else (0, 0)
        }
    
    def plot_safety_envelope(self, sweep_results: Dict, 
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot the safe operating envelope.
        
        Args:
            sweep_results: Results from sweep_current_geometry
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure object
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        L_grid = sweep_results['L_grid']
        I_grid = sweep_results['I_grid']
        
        # Plot magnetic field
        im1 = ax1.contourf(L_grid, I_grid, sweep_results['B_peak_grid'], 
                          levels=20, cmap='viridis')
        ax1.contour(L_grid, I_grid, sweep_results['B_peak_grid'], 
                   levels=[B_MAX_SAFE], colors='red', linewidths=2)
        ax1.set_xlabel('Inductance L (H)')
        ax1.set_ylabel('Current I (A)')
        ax1.set_title('Peak Magnetic Field B (T)')
        plt.colorbar(im1, ax=ax1)
        
        # Plot electric field
        im2 = ax2.contourf(L_grid, I_grid, sweep_results['E_peak_grid'], 
                          levels=20, cmap='plasma')
        ax2.contour(L_grid, I_grid, sweep_results['E_peak_grid'], 
                   levels=[E_BREAKDOWN], colors='red', linewidths=2)
        ax2.set_xlabel('Inductance L (H)')
        ax2.set_ylabel('Current I (A)')
        ax2.set_title('Peak Electric Field E (V/m)')
        plt.colorbar(im2, ax=ax2)
        
        # Plot safety margins
        safety_levels = [0.5, 1.0, 2.0, 5.0, 10.0]
        im3 = ax3.contourf(L_grid, I_grid, sweep_results['safety_grid'], 
                          levels=safety_levels, cmap='RdYlGn')
        ax3.contour(L_grid, I_grid, sweep_results['safety_grid'], 
                   levels=[1.0, 2.0], colors=['red', 'orange'], linewidths=2)
        ax3.set_xlabel('Inductance L (H)')
        ax3.set_ylabel('Current I (A)')
        ax3.set_title('Safety Margin (minimum across all limits)')
        plt.colorbar(im3, ax=ax3)
        
        # Plot safe operating region
        safe_mask = sweep_results['safe_mask']
        ax4.contourf(L_grid, I_grid, safe_mask.astype(int), 
                    levels=[0, 0.5, 1], colors=['red', 'green'], alpha=0.7)
        ax4.set_xlabel('Inductance L (H)')
        ax4.set_ylabel('Current I (A)')
        ax4.set_title('Safe Operating Region (green = safe)')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def optimize_for_target_field(self, B_target: float, geometry: str = 'solenoid',
                                max_current: float = 1e5) -> Dict:
        """
        Optimize coil parameters to achieve target magnetic field.
        
        Args:
            B_target: Target magnetic field (T)
            geometry: Coil geometry type
            max_current: Maximum allowed current (A)
            
        Returns:
            Optimization results
        """
        def objective(params):
            L, I = params
            if I > max_current or L <= 0:
                return 1e10  # Penalty for invalid parameters
            
            try:
                results = self.simulate_inductive_rig(L, I, 1000.0, geometry=geometry)
                error = (results.B_peak - B_target)**2
                
                # Add penalty for low safety margins
                min_safety = min(results.safety_margins.values())
                if min_safety < 2.0:
                    error += 1e6 * (2.0 - min_safety)**2
                
                return error
            except:
                return 1e10
        
        # Initial guess
        L_guess = 1e-3  # 1 mH
        I_guess = min(B_target / (μ0 * 1000), max_current)  # Rough estimate
        
        from scipy.optimize import minimize
        result = minimize(objective, [L_guess, I_guess], 
                         bounds=[(1e-6, 1.0), (1.0, max_current)],
                         method='L-BFGS-B')
        
        if result.success:
            L_opt, I_opt = result.x
            optimal_results = self.simulate_inductive_rig(L_opt, I_opt, 1000.0, geometry=geometry)
            
            return {
                'success': True,
                'optimal_L': L_opt,
                'optimal_I': I_opt,
                'achieved_B': optimal_results.B_peak,
                'target_B': B_target,
                'error': abs(optimal_results.B_peak - B_target),
                'safety_margins': optimal_results.safety_margins,
                'full_results': optimal_results
            }
        else:
            return {
                'success': False,
                'message': result.message
            }

if __name__ == "__main__":
    # Example usage
    simulator = ElectromagneticFieldSimulator()
    
    # Test single configuration
    L = 1e-3  # 1 mH
    I = 1000  # 1000 A
    f_mod = 1000  # 1 kHz
    
    results = simulator.simulate_inductive_rig(L, I, f_mod, geometry='toroidal')
    
    print("Electromagnetic Field Simulation Results:")
    print(f"Characteristic radius: {results.r_char:.3f} m")
    print(f"Peak magnetic field: {results.B_peak:.2f} T")
    print(f"Peak electric field: {results.E_peak:.2e} V/m")
    print(f"Stored energy: {results.stored_energy:.2e} J")
    print(f"Power dissipation: {results.power_dissipation:.2e} W")
    
    print("\nSafety Analysis:")
    safety_status = simulator.safety_analysis(results)
    for param, status in safety_status.items():
        print(f"  {param}: {status}")
    
    # Perform parameter sweep
    print("\nPerforming parameter sweep...")
    sweep_results = simulator.sweep_current_geometry(
        L_range=(1e-6, 1e-2), I_range=(100, 10000), n_points=20
    )
    
    print(f"Safe inductance range: {sweep_results['safe_L_range']}")
    print(f"Safe current range: {sweep_results['safe_I_range']}")
    
    # Plot results
    fig = simulator.plot_safety_envelope(sweep_results, save_path="safety_envelope.png")
    plt.show()
    
    # Optimize for target field
    print("\nOptimizing for target field of 1 T...")
    opt_results = simulator.optimize_for_target_field(1.0, geometry='toroidal')
    
    if opt_results['success']:
        print(f"Optimal L: {opt_results['optimal_L']:.6f} H")
        print(f"Optimal I: {opt_results['optimal_I']:.1f} A")
        print(f"Achieved B: {opt_results['achieved_B']:.3f} T")
        print(f"Target error: {opt_results['error']:.6f} T")
    else:
        print(f"Optimization failed: {opt_results['message']}")
