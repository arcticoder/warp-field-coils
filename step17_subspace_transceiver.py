#!/usr/bin/env python3
"""
Step 17: Subspace Transceiver Implementation
Standalone module implementing subspace communication with modified wave equation
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass

@dataclass
class SubspaceParams:
    """Parameters for subspace medium."""
    c_s: float = 5e8  # Subspace wave speed (m/s)
    bandwidth: float = 1e12  # Available bandwidth (Hz)
    grid_resolution: int = 64  # Grid resolution for simulations
    subspace_coupling: float = 1e-15  # δ_sub metric coupling
    mu: float = 4e-7 * np.pi  # Permeability
    epsilon: float = 8.854e-12  # Permittivity
    loss_tangent: float = 1e-6  # Loss factor

@dataclass 
class TransmissionParams:
    """Parameters for a transmission."""
    frequency: float  # Carrier frequency (Hz)
    modulation_depth: float  # Modulation depth (0-1)
    duration: float  # Transmission duration (s)
    target_coordinates: Tuple[float, float, float]  # Target location (m)
    priority: int = 5  # Priority level (1-10)

class SubspaceTransceiver:
    """
    Subspace Transceiver implementing modified Helmholtz equation:
    ∇_μ F^{μν} + κ² A^ν = 0
    where κ² = ω² μ ε (1 + δ_sub)
    """
    
    def __init__(self, params: SubspaceParams):
        self.params = params
        self.diagnostics_data = {}
        
    def compute_dispersion_relation(self, omega: np.ndarray) -> np.ndarray:
        """
        Compute dispersion relation: k² = κ² - i σ_subspace ω
        
        Args:
            omega: Angular frequency array (rad/s)
            
        Returns:
            Complex wave vector k
        """
        # Modified permittivity with subspace coupling
        epsilon_eff = self.params.epsilon * (1 + self.params.subspace_coupling)
        
        # Real part of k²
        k_squared_real = omega**2 * self.params.mu * epsilon_eff
        
        # Imaginary part from loss
        k_squared_imag = -1j * self.params.loss_tangent * omega
        
        # Total dispersion
        k_squared = k_squared_real + k_squared_imag
        k = np.sqrt(k_squared)
        
        return k
    
    def compute_group_velocity(self, omega: np.ndarray) -> np.ndarray:
        """Compute group velocity v_g = dω/dk"""
        # Ensure omega is array-like
        omega = np.atleast_1d(omega)
        k = self.compute_dispersion_relation(omega)
        
        if len(omega) == 1:
            # For single frequency, use analytical derivative
            delta_omega = omega[0] * 1e-6  # Small perturbation
            omega_pert = np.array([omega[0] - delta_omega, omega[0] + delta_omega])
            k_pert = self.compute_dispersion_relation(omega_pert)
            dk_domega = (k_pert[1] - k_pert[0]) / (2 * delta_omega)
            v_g = 1.0 / dk_domega.real
            return np.array([v_g])
        else:
            # Numerical derivative for array
            dk_domega = np.gradient(k, omega)
            v_g = 1.0 / dk_domega.real
            return v_g
    
    def run_diagnostics(self) -> Dict:
        """Run comprehensive transceiver diagnostics."""
        print("🔍 Running subspace transceiver diagnostics...")
        
        # Frequency sweep
        omega = np.linspace(1e9, 1e13, 1000)  # 1 GHz to 10 THz
        k = self.compute_dispersion_relation(omega)
        v_g = self.compute_group_velocity(omega)
        
        # Find superluminal regions
        c = 299792458  # Speed of light
        superluminal_mask = np.abs(v_g) > c
        ftl_fraction = np.sum(superluminal_mask) / len(v_g)
        
        # Compute attenuation
        attenuation_db_per_km = 20 * np.log10(np.exp(1)) * k.imag * 1000
        
        # Bandwidth analysis
        usable_bandwidth = np.sum(attenuation_db_per_km < 10) * (omega[1] - omega[0]) / (2*np.pi)
        
        diagnostics = {
            'frequency_range_hz': (omega[0]/(2*np.pi), omega[-1]/(2*np.pi)),
            'superluminal_fraction': ftl_fraction,
            'max_group_velocity_c': np.max(np.abs(v_g)) / c,
            'usable_bandwidth_hz': usable_bandwidth,
            'min_attenuation_db_km': np.min(attenuation_db_per_km),
            'dispersion_data': {
                'omega': omega.tolist(),
                'k_real': k.real.tolist(),
                'k_imag': k.imag.tolist(),
                'group_velocity': v_g.tolist()
            }
        }
        
        self.diagnostics_data = diagnostics
        
        print(f"✓ Superluminal fraction: {ftl_fraction:.1%}")
        print(f"✓ Max group velocity: {np.max(np.abs(v_g))/c:.2f}c")
        print(f"✓ Usable bandwidth: {usable_bandwidth/1e9:.2f} GHz")
        
        return diagnostics
    
    def transmit_message(self, message: str, tx_params: TransmissionParams) -> Dict:
        """
        Simulate transmission of a message through subspace.
        
        Args:
            message: Message to transmit
            tx_params: Transmission parameters
            
        Returns:
            Transmission results
        """
        print(f"📡 Transmitting: '{message}' to {tx_params.target_coordinates}")
        
        # Convert message to bit stream (simplified)
        message_bits = len(message) * 8  # 8 bits per character
        
        # Compute propagation parameters
        omega = 2 * np.pi * tx_params.frequency
        k = self.compute_dispersion_relation(np.array([omega]))[0]
        
        # Distance to target
        target_distance = np.linalg.norm(tx_params.target_coordinates)
        
        # Propagation time
        v_g = self.compute_group_velocity(np.array([omega]))[0]
        propagation_time = target_distance / np.abs(v_g)
        
        # Signal attenuation
        attenuation_nepers = k.imag * target_distance
        signal_strength_linear = np.exp(-attenuation_nepers)
        signal_strength_db = 20 * np.log10(signal_strength_linear + 1e-12)
        
        # Data rate calculation
        snr_linear = 10**(signal_strength_db/10)
        shannon_capacity = self.params.bandwidth * np.log2(1 + snr_linear)
        transmission_time = message_bits / shannon_capacity
        
        # FTL check
        c = 299792458
        is_ftl = np.abs(v_g) > c
        ftl_factor = np.abs(v_g) / c if is_ftl else 1.0
        
        results = {
            'message_length_bits': message_bits,
            'target_distance_m': target_distance,
            'propagation_time_s': propagation_time,
            'signal_strength_db': signal_strength_db,
            'shannon_capacity_bps': shannon_capacity,
            'transmission_time_s': transmission_time,
            'is_superluminal': bool(is_ftl),
            'ftl_factor': ftl_factor,
            'group_velocity_ms': float(v_g),
            'success': signal_strength_db > -60  # -60 dB threshold
        }
        
        print(f"✓ Signal strength: {signal_strength_db:.1f} dB")
        print(f"✓ Propagation time: {propagation_time:.2e} s")
        print(f"✓ FTL factor: {ftl_factor:.2f}x")
        print(f"✓ Data rate: {shannon_capacity/1e9:.2f} Gbps")
        
        return results
    
    def plot_dispersion_curve(self, save_path: str = "step17_dispersion.png"):
        """Plot dispersion relation and group velocity."""
        if not self.diagnostics_data:
            self.run_diagnostics()
        
        omega = np.array(self.diagnostics_data['dispersion_data']['omega'])
        k_real = np.array(self.diagnostics_data['dispersion_data']['k_real'])
        v_g = np.array(self.diagnostics_data['dispersion_data']['group_velocity'])
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Dispersion curve
        ax1.plot(omega/(2*np.pi*1e9), k_real, 'b-', linewidth=2)
        ax1.set_xlabel('Frequency (GHz)')
        ax1.set_ylabel('Wave Vector k (1/m)')
        ax1.set_title('Subspace Dispersion Relation')
        ax1.grid(True, alpha=0.3)
        
        # Group velocity
        c = 299792458
        ax2.plot(omega/(2*np.pi*1e9), v_g/c, 'r-', linewidth=2)
        ax2.axhline(y=1, color='k', linestyle='--', alpha=0.7, label='Speed of light')
        ax2.set_xlabel('Frequency (GHz)')
        ax2.set_ylabel('Group Velocity (c)')
        ax2.set_title('Group Velocity (Superluminal regions > 1c)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Dispersion curve saved to {save_path}")

def main():
    """Demonstration of Step 17: Subspace Transceiver"""
    print("=== STEP 17: SUBSPACE TRANSCEIVER ===")
    
    # Initialize transceiver
    params = SubspaceParams(
        c_s=5e8,  # 1.67x speed of light
        bandwidth=1e12,  # 1 THz bandwidth
        subspace_coupling=1e-15,  # Weak coupling
        loss_tangent=1e-6  # Low loss
    )
    
    transceiver = SubspaceTransceiver(params)
    
    # Run diagnostics
    diagnostics = transceiver.run_diagnostics()
    
    # Plot dispersion
    transceiver.plot_dispersion_curve()
    
    # Test transmission
    tx_params = TransmissionParams(
        frequency=2.4e12,  # 2.4 THz
        modulation_depth=0.8,
        duration=0.05,  # 50 ms
        target_coordinates=(1e16, 0, 0),  # 1 light-year
        priority=7
    )
    
    result = transceiver.transmit_message("Hello, Alpha Centauri!", tx_params)
    
    print("\n📊 TRANSMISSION SUMMARY:")
    print(f"Target: {tx_params.target_coordinates[0]/9.46e15:.1f} light-years")
    print(f"FTL transmission: {result['is_superluminal']}")
    print(f"Speed factor: {result['ftl_factor']:.2f}x light speed")
    print(f"Data rate: {result['shannon_capacity_bps']/1e9:.2f} Gbps")
    print(f"Success: {result['success']}")

if __name__ == "__main__":
    main()
