#!/usr/bin/env python3
"""
FDTD (Finite-Difference Time-Domain) Integration for Full-Wave EM Validation
Provides interface for high-fidelity electromagnetic simulation validation
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
import matplotlib.pyplot as plt
import json

@dataclass
class FDTDSimulationParams:
    """FDTD simulation parameters."""
    cell_size: Tuple[float, float, float]  # (dx, dy, dz) in meters
    domain_size: Tuple[float, float, float]  # Domain dimensions
    resolution: int  # Points per unit length
    time_steps: int  # Number of time steps
    dt: float  # Time step size
    frequency: float  # Source frequency (Hz)
    pml_layers: int = 10  # PML boundary layers

@dataclass
class CurrentSource:
    """Current source specification for FDTD."""
    position: Tuple[float, float, float]  # Source position
    direction: Tuple[float, float, float]  # Current direction
    amplitude: float  # Current amplitude (A)
    frequency: float  # Source frequency (Hz)
    waveform: str  # 'gaussian', 'sinusoidal', 'pulse'

@dataclass  
class FDTDResults:
    """FDTD simulation results."""
    E_field: np.ndarray  # Electric field components (Nx,Ny,Nz,3,Nt)
    B_field: np.ndarray  # Magnetic field components  
    energy_density: np.ndarray  # EM energy density
    power_flow: np.ndarray  # Poynting vector
    frequencies: np.ndarray  # Frequency spectrum
    field_monitors: Dict[str, np.ndarray]  # Field monitor data

class FDTDValidator:
    """
    FDTD electromagnetic validation for warp coil designs.
    
    Provides interface for full-wave electromagnetic simulation
    to validate coil performance beyond quasi-static approximations.
    """
    
    def __init__(self, use_meep: bool = False):
        """
        Initialize FDTD validator.
        
        Args:
            use_meep: Whether to use MEEP for actual simulation
        """
        self.use_meep = use_meep
        self.simulation_results = {}
        
        if use_meep:
            try:
                import meep as mp
                self.mp = mp
                self.meep_available = True
                print("✓ MEEP library loaded for FDTD simulation")
            except ImportError:
                print("⚠️ MEEP not available, using mock simulation")
                self.meep_available = False
        else:
            self.meep_available = False
    
    def setup_coil_current_sources(self, coil_geometry_3d, 
                                  sim_params: FDTDSimulationParams) -> List[CurrentSource]:
        """
        Convert 3D coil geometry to FDTD current sources.
        
        Args:
            coil_geometry_3d: 3D coil path and current specification
            sim_params: FDTD simulation parameters
            
        Returns:
            List of CurrentSource objects for FDTD
        """
        sources = []
        
        # Convert coil path to discrete current sources
        path_points = coil_geometry_3d.path_points
        current = coil_geometry_3d.current
        
        # Create current sources along path
        for i in range(len(path_points) - 1):
            # Current segment
            r1, r2 = path_points[i], path_points[i+1]
            position = (r1 + r2) / 2  # Midpoint
            direction = r2 - r1  # Current direction
            direction_norm = direction / (np.linalg.norm(direction) + 1e-12)
            
            # Create current source
            source = CurrentSource(
                position=tuple(position),
                direction=tuple(direction_norm),
                amplitude=current,
                frequency=sim_params.frequency,
                waveform='sinusoidal'
            )
            sources.append(source)
        
        return sources
    
    def run_meep_simulation(self, current_sources: List[CurrentSource],
                           sim_params: FDTDSimulationParams) -> FDTDResults:
        """
        Run FDTD simulation using MEEP.
        
        Args:
            current_sources: List of current sources
            sim_params: Simulation parameters
            
        Returns:
            FDTD simulation results
        """
        if not self.meep_available:
            return self._run_mock_simulation(current_sources, sim_params)
        
        mp = self.mp
        
        # Setup simulation domain
        cell = mp.Vector3(sim_params.domain_size[0], 
                         sim_params.domain_size[1], 
                         sim_params.domain_size[2])
        
        # PML boundary conditions
        pml_layers = [mp.PML(sim_params.pml_layers)]
        
        # Create current sources
        sources = []
        for src in current_sources:
            # MEEP current source
            meep_source = mp.Source(
                mp.ContinuousSource(frequency=src.frequency/(3e8)),  # Normalize frequency
                component=mp.Jx,  # Current density component
                center=mp.Vector3(src.position[0], src.position[1], src.position[2]),
                amplitude=src.amplitude
            )
            sources.append(meep_source)
        
        # Setup simulation
        sim = mp.Simulation(
            cell_size=cell,
            boundary_layers=pml_layers,
            sources=sources,
            resolution=sim_params.resolution
        )
        
        # Field monitors
        monitor_points = []
        center = mp.Vector3(0, 0, 0)
        
        # Run simulation
        sim.run(until=sim_params.time_steps * sim_params.dt)
        
        # Extract field data
        E_data = sim.get_array(center=center, size=cell, component=mp.Ex)
        B_data = sim.get_array(center=center, size=cell, component=mp.Hx)
        
        # Create results
        results = FDTDResults(
            E_field=E_data[..., np.newaxis, np.newaxis],
            B_field=B_data[..., np.newaxis, np.newaxis], 
            energy_density=0.5 * (E_data**2 + B_data**2),
            power_flow=np.cross(E_data, B_data, axis=-1),
            frequencies=np.fft.fftfreq(sim_params.time_steps, sim_params.dt),
            field_monitors={'center': E_data}
        )
        
        return results
    
    def _run_mock_simulation(self, current_sources: List[CurrentSource],
                            sim_params: FDTDSimulationParams) -> FDTDResults:
        """
        Run mock FDTD simulation for testing.
        
        Args:
            current_sources: List of current sources
            sim_params: Simulation parameters
            
        Returns:
            Mock FDTD results
        """
        # Create mock field data
        nx = int(sim_params.domain_size[0] * sim_params.resolution)
        ny = int(sim_params.domain_size[1] * sim_params.resolution)
        nz = int(sim_params.domain_size[2] * sim_params.resolution)
        nt = sim_params.time_steps
        
        # Mock fields with some realistic structure
        x = np.linspace(-sim_params.domain_size[0]/2, sim_params.domain_size[0]/2, nx)
        y = np.linspace(-sim_params.domain_size[1]/2, sim_params.domain_size[1]/2, ny)
        z = np.linspace(-sim_params.domain_size[2]/2, sim_params.domain_size[2]/2, nz)
        t = np.linspace(0, sim_params.time_steps * sim_params.dt, nt)
        
        X, Y, Z, T = np.meshgrid(x, y, z, t, indexing='ij')
        
        # Mock electric field (dipole-like pattern)
        total_current = sum(src.amplitude for src in current_sources)
        r = np.sqrt(X**2 + Y**2 + Z**2) + 1e-6
        
        # Time-varying field
        omega = 2 * np.pi * sim_params.frequency
        phase = omega * T
        
        E_mock = total_current * np.sin(phase) * np.exp(-r) / r
        B_mock = total_current * np.cos(phase) * np.exp(-r) / r
        
        # Create 4D arrays (x,y,z,3,t) for vector fields
        E_field = np.zeros((nx, ny, nz, 3, nt))
        B_field = np.zeros((nx, ny, nz, 3, nt))
        
        # Fill with mock data
        for i in range(3):
            E_field[:, :, :, i, :] = E_mock
            B_field[:, :, :, i, :] = B_mock
        
        # Energy density and Poynting vector
        energy_density = 0.5 * (np.sum(E_field**2, axis=3) + np.sum(B_field**2, axis=3))
        power_flow = np.cross(E_field, B_field, axis=3)
        
        # Frequency spectrum
        frequencies = np.fft.fftfreq(nt, sim_params.dt)
        
        results = FDTDResults(
            E_field=E_field,
            B_field=B_field,
            energy_density=energy_density,
            power_flow=power_flow,
            frequencies=frequencies,
            field_monitors={'center_z0': E_field[:, :, nz//2, :, :]}
        )
        
        return results
    
    def validate_coil_design(self, coil_geometry_3d, target_frequency: float = 1e6,
                           domain_extent: float = 5.0) -> Dict:
        """
        Validate 3D coil design using FDTD simulation.
        
        Args:
            coil_geometry_3d: 3D coil geometry to validate
            target_frequency: Operating frequency (Hz)
            domain_extent: Simulation domain size (m)
            
        Returns:
            Validation results dictionary
        """
        print(f"🔍 Validating coil design with FDTD simulation...")
        
        # Setup simulation parameters
        sim_params = FDTDSimulationParams(
            cell_size=(0.1, 0.1, 0.1),
            domain_size=(domain_extent, domain_extent, domain_extent),
            resolution=20,  # 20 points per meter
            time_steps=200,
            dt=1e-9,  # 1 ns time step
            frequency=target_frequency
        )
        
        # Convert coil to current sources
        current_sources = self.setup_coil_current_sources(coil_geometry_3d, sim_params)
        print(f"  Created {len(current_sources)} current sources")
        
        # Run FDTD simulation
        results = self.run_meep_simulation(current_sources, sim_params)
        print(f"  Simulation complete: {results.E_field.shape} field grid")
        
        # Analyze results
        analysis = self._analyze_fdtd_results(results, sim_params)
        
        # Store results
        self.simulation_results[f'coil_validation_{target_frequency:.0e}'] = {
            'sim_params': sim_params,
            'results': results,
            'analysis': analysis
        }
        
        return analysis
    
    def _analyze_fdtd_results(self, results: FDTDResults, 
                             sim_params: FDTDSimulationParams) -> Dict:
        """
        Analyze FDTD simulation results.
        
        Args:
            results: FDTD simulation results
            sim_params: Simulation parameters
            
        Returns:
            Analysis dictionary
        """
        # Field statistics
        E_max = np.max(np.abs(results.E_field))
        B_max = np.max(np.abs(results.B_field))
        
        # Energy analysis
        total_energy = np.sum(results.energy_density)
        max_energy_density = np.max(results.energy_density)
        
        # Power flow analysis
        power_magnitude = np.linalg.norm(results.power_flow, axis=3)
        max_power_flow = np.max(power_magnitude)
        
        # Frequency spectrum analysis
        E_fft = np.fft.fft(results.E_field, axis=-1)
        dominant_freq_idx = np.argmax(np.abs(E_fft[..., 0]))
        dominant_frequency = results.frequencies[dominant_freq_idx % len(results.frequencies)]
        
        analysis = {
            'field_maxima': {
                'E_max': float(E_max),
                'B_max': float(B_max)
            },
            'energy_analysis': {
                'total_energy': float(total_energy),
                'max_energy_density': float(max_energy_density),
                'avg_energy_density': float(np.mean(results.energy_density))
            },
            'power_analysis': {
                'max_power_flow': float(max_power_flow),
                'avg_power_flow': float(np.mean(power_magnitude))
            },
            'frequency_analysis': {
                'dominant_frequency': float(dominant_frequency),
                'target_frequency': sim_params.frequency,
                'frequency_accuracy': float(abs(dominant_frequency - sim_params.frequency) / sim_params.frequency)
            },
            'simulation_info': {
                'grid_points': results.E_field.shape[:3],
                'time_steps': results.E_field.shape[-1],
                'domain_size': sim_params.domain_size
            }
        }
        
        return analysis
    
    def create_field_animation(self, results: FDTDResults, 
                              plane: str = 'xy', save_path: Optional[str] = None) -> None:
        """
        Create animated visualization of field evolution.
        
        Args:
            results: FDTD simulation results
            plane: Plane to visualize ('xy', 'xz', 'yz')
            save_path: Path to save animation
        """
        print(f"Creating field animation for {plane} plane...")
        
        # Extract field slice
        if plane == 'xy':
            field_slice = results.E_field[:, :, results.E_field.shape[2]//2, 0, :]
        elif plane == 'xz':
            field_slice = results.E_field[:, results.E_field.shape[1]//2, :, 0, :]
        else:  # yz
            field_slice = results.E_field[results.E_field.shape[0]//2, :, :, 0, :]
        
        # Create static plot (animation would require additional libraries)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Show field at different time steps
        time_indices = [0, field_slice.shape[-1]//3, 2*field_slice.shape[-1]//3]
        
        for i, t_idx in enumerate(time_indices):
            im = axes[i].imshow(field_slice[:, :, t_idx], cmap='RdBu', 
                              origin='lower', aspect='equal')
            axes[i].set_title(f'E-field at t={t_idx}')
            axes[i].set_xlabel('Grid Index')
            axes[i].set_ylabel('Grid Index')
            plt.colorbar(im, ax=axes[i])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        print("✓ Field visualization complete")
    
    def export_results(self, filename: str) -> None:
        """
        Export FDTD validation results.
        
        Args:
            filename: Output filename
        """
        export_data = {}
        
        for sim_name, sim_data in self.simulation_results.items():
            export_data[sim_name] = {
                'analysis': sim_data['analysis'],
                'sim_params': {
                    'domain_size': sim_data['sim_params'].domain_size,
                    'resolution': sim_data['sim_params'].resolution,
                    'frequency': sim_data['sim_params'].frequency,
                    'time_steps': sim_data['sim_params'].time_steps
                }
                # Note: Field data too large for JSON export
            }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"✓ Exported FDTD results to {filename}")

def create_validation_suite() -> FDTDValidator:
    """Create FDTD validation suite."""
    return FDTDValidator(use_meep=False)  # Mock simulation for now

def main():
    """Demonstrate FDTD validation framework."""
    print("🌊 FDTD ELECTROMAGNETIC VALIDATION FRAMEWORK")
    print("=" * 50)
    
    # Create validator
    validator = create_validation_suite()
    print("✓ FDTD validator created")
    
    # Create mock coil geometry
    from dataclasses import dataclass
    @dataclass
    class MockCoilGeometry:
        path_points: np.ndarray
        current: float
    
    # Simple circular coil for testing
    theta = np.linspace(0, 2*np.pi, 50)
    path_points = np.column_stack([np.cos(theta), np.sin(theta), np.zeros(50)])
    
    mock_coil = MockCoilGeometry(
        path_points=path_points,
        current=1000.0
    )
    
    # Run validation
    analysis = validator.validate_coil_design(mock_coil, target_frequency=1e6)
    
    print(f"✓ Validation complete:")
    print(f"  Max E-field: {analysis['field_maxima']['E_max']:.2e}")
    print(f"  Total energy: {analysis['energy_analysis']['total_energy']:.2e}")
    print(f"  Frequency accuracy: {analysis['frequency_analysis']['frequency_accuracy']:.2%}")
    
    print("✅ FDTD validation framework ready!")

if __name__ == "__main__":
    main()
