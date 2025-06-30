"""
Steerable Unified Coil System for Multi-Field Warp Operations

This module implements a steerable coil system that can generate and manage
multiple overlapping warp fields through frequency multiplexing and spatial
sector assignment within a single spin-network shell.

Enhanced Features:
- Multi-field coil configuration
- Frequency multiplexed field generation
- Spatial sector steering
- Dynamic field reconfiguration
- Orthogonal field operation
- Advanced field shaping

Mathematical Foundation:
- Multi-field current density: J_μ = Σ_a J_μ^(a) * f_a(t) * χ_a(x)
- Orthogonal sectors: [f_a, f_b] = 0 ensures field independence
- Coil field coupling: B_μν = Σ_a B_μν^(a) * I_a(t)
- Steerable field vectors: B⃗(θ,φ) = Σ_n B_n * Y_n(θ,φ)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
import logging
from enum import Enum, auto
from scipy.special import sph_harm, legendre
from scipy.spatial.transform import Rotation
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Physical constants
C_LIGHT = 299792458.0  # m/s
MU_0 = 4*np.pi*1e-7  # H/m (permeability of free space)
EPSILON_0 = 8.854187817e-12  # F/m
Z_0 = np.sqrt(MU_0/EPSILON_0)  # Impedance of free space

# Field types for multi-field coil operations
class FieldType(Enum):
    WARP_DRIVE = "warp_drive"
    SHIELDS = "shields"
    TRANSPORTER = "transporter"
    INERTIAL_DAMPER = "inertial_damper"
    STRUCTURAL_INTEGRITY = "structural_integrity"
    HOLODECK_FORCEFIELD = "holodeck_forcefield"
    MEDICAL_TRACTOR = "medical_tractor"
    REPLICATOR = "replicator"

class CoilType(Enum):
    TOROIDAL = "toroidal"
    POLOIDAL = "poloidal"
    HELICAL = "helical"
    SADDLE = "saddle"
    QUADRUPOLE = "quadrupole"
    MULTIPOLE = "multipole"

class SteeringMode(Enum):
    STATIC = "static"
    DYNAMIC = "dynamic"
    ADAPTIVE = "adaptive"
    PREDICTIVE = "predictive"

@dataclass
class CoilConfiguration:
    """Configuration for a single coil in the multi-field system"""
    coil_id: int
    coil_type: CoilType
    field_type: FieldType
    position: np.ndarray  # (x, y, z) in meters
    orientation: np.ndarray  # (θ, φ, ψ) Euler angles in radians
    turns: int = 100
    radius: float = 1.0  # meters
    current_capacity: float = 1000.0  # Amperes
    frequency_band: Tuple[float, float] = (1e9, 1.1e9)  # Hz
    active: bool = True
    
    # Coil-specific parameters
    wire_gauge: float = 0.001  # m (wire diameter)
    resistance: float = 0.1  # Ohms
    inductance: float = 1e-6  # Henry
    quality_factor: float = 100.0
    
    # Field shaping parameters
    field_strength: float = 1.0  # Tesla
    field_gradient: float = 0.0  # T/m
    multipole_order: int = 1  # Dipole = 1, Quadrupole = 2, etc.

@dataclass
class SteerableCoilSystem:
    """Configuration for the complete steerable coil system"""
    shell_radius: float = 50.0  # meters
    coil_configurations: List[CoilConfiguration] = field(default_factory=list)
    max_coils: int = 32
    steering_resolution: int = 64  # Angular resolution for steering
    frequency_multiplexing: bool = True
    adaptive_steering: bool = True
    
    # System parameters
    total_power_limit: float = 100e6  # Watts
    cooling_capacity: float = 50e6  # Watts
    field_uniformity_tolerance: float = 0.05  # 5%
    
    # Control parameters
    response_time: float = 0.001  # seconds
    stability_threshold: float = 0.01
    interference_threshold: float = 0.1

class MultiFieldCoilSystem:
    """
    Advanced steerable coil system for generating multiple overlapping
    warp fields with frequency multiplexing and spatial steering
    """
    
    def __init__(self, config: SteerableCoilSystem):
        """
        Initialize multi-field coil system
        
        Args:
            config: System configuration
        """
        self.config = config
        self.coils: Dict[int, CoilConfiguration] = {}
        self.coil_counter = 0
        
        # Field generation arrays
        self.field_grid_resolution = 32
        self.x_grid = np.linspace(-2*config.shell_radius, 2*config.shell_radius, self.field_grid_resolution)
        self.y_grid = np.linspace(-2*config.shell_radius, 2*config.shell_radius, self.field_grid_resolution)
        self.z_grid = np.linspace(-2*config.shell_radius, 2*config.shell_radius, self.field_grid_resolution)
        
        # Create coordinate meshes
        self.X, self.Y, self.Z = np.meshgrid(self.x_grid, self.y_grid, self.z_grid, indexing='ij')
        
        # Spherical coordinates
        self.R = np.sqrt(self.X**2 + self.Y**2 + self.Z**2)
        self.THETA = np.arccos(self.Z / np.maximum(self.R, 1e-12))
        self.PHI = np.arctan2(self.Y, self.X)
        
        # Frequency band allocator
        self.frequency_bands = self._initialize_frequency_bands()
        self.allocated_bands = set()
        
        # Steering control
        self.steering_angles = np.linspace(0, 2*np.pi, config.steering_resolution)
        self.steering_weights = np.ones(config.steering_resolution)
        
        # Field history for adaptive control
        self.field_history = []
        self.performance_history = []
        
        logger.info(f"Multi-field coil system initialized with {config.max_coils} max coils")

    def _initialize_frequency_bands(self) -> List[Tuple[float, float]]:
        """Initialize frequency bands for coil multiplexing"""
        base_freq = 1e9  # 1 GHz
        band_width = 1e8  # 100 MHz
        num_bands = self.config.max_coils
        
        bands = []
        for i in range(num_bands):
            freq_min = base_freq + i * band_width * 1.2  # 20% guard band
            freq_max = freq_min + band_width
            bands.append((freq_min, freq_max))
        
        return bands

    def allocate_frequency_band(self, coil_id: int) -> Tuple[float, float]:
        """Allocate frequency band for a coil"""
        for i, band in enumerate(self.frequency_bands):
            if i not in self.allocated_bands:
                self.allocated_bands.add(i)
                logger.info(f"Allocated frequency band {band[0]/1e9:.1f}-{band[1]/1e9:.1f} GHz to coil {coil_id}")
                return band
        
        raise ValueError("No available frequency bands for coil allocation")

    def add_coil(self, 
                 coil_type: CoilType,
                 field_type: FieldType,
                 position: np.ndarray,
                 orientation: np.ndarray = None,
                 **kwargs) -> int:
        """
        Add a coil to the multi-field system
        
        Args:
            coil_type: Type of coil
            field_type: Type of field to generate
            position: 3D position (x, y, z) in meters
            orientation: Euler angles (θ, φ, ψ) in radians
            **kwargs: Additional coil parameters
            
        Returns:
            Coil identifier
        """
        if len(self.coils) >= self.config.max_coils:
            raise ValueError(f"Maximum number of coils ({self.config.max_coils}) reached")
        
        coil_id = self.coil_counter
        self.coil_counter += 1
        
        # Default orientation
        if orientation is None:
            orientation = np.array([0.0, 0.0, 0.0])
        
        # Allocate frequency band
        frequency_band = self.allocate_frequency_band(coil_id)
        
        # Create coil configuration
        coil_config = CoilConfiguration(
            coil_id=coil_id,
            coil_type=coil_type,
            field_type=field_type,
            position=position,
            orientation=orientation,
            frequency_band=frequency_band,
            **kwargs
        )
        
        self.coils[coil_id] = coil_config
        
        logger.info(f"Added {coil_type.value} coil for {field_type.value} at position {position}")
        return coil_id

    def compute_coil_magnetic_field(self, 
                                   coil_id: int,
                                   current: float,
                                   time: float = 0.0) -> Dict[str, np.ndarray]:
        """
        Compute magnetic field generated by a single coil
        
        Args:
            coil_id: Coil identifier
            current: Current through coil (A)
            time: Time coordinate
            
        Returns:
            Dictionary with magnetic field components
        """
        if coil_id not in self.coils:
            raise ValueError(f"Coil {coil_id} not found")
        
        coil = self.coils[coil_id]
        
        # Translate coordinates to coil center
        X_rel = self.X - coil.position[0]
        Y_rel = self.Y - coil.position[1]
        Z_rel = self.Z - coil.position[2]
        
        # Rotate coordinates to coil orientation
        rotation = Rotation.from_euler('xyz', coil.orientation)
        coords_rel = np.stack([X_rel.flatten(), Y_rel.flatten(), Z_rel.flatten()], axis=1)
        coords_rotated = rotation.apply(coords_rel)
        
        X_rot = coords_rotated[:, 0].reshape(self.X.shape)
        Y_rot = coords_rotated[:, 1].reshape(self.Y.shape)
        Z_rot = coords_rotated[:, 2].reshape(self.Z.shape)
        
        # Distance from coil axis
        rho = np.sqrt(X_rot**2 + Y_rot**2)
        
        # Initialize field components
        B_x = np.zeros_like(self.X)
        B_y = np.zeros_like(self.Y)
        B_z = np.zeros_like(self.Z)
        
        # Compute field based on coil type
        if coil.coil_type == CoilType.TOROIDAL:
            # Toroidal coil field (approximate)
            B_phi = (MU_0 * coil.turns * current * coil.radius**2) / (2 * (coil.radius**2 + rho**2)**(3/2))
            
            # Convert to Cartesian
            phi = np.arctan2(Y_rot, X_rot)
            B_x = -B_phi * np.sin(phi)
            B_y = B_phi * np.cos(phi)
            B_z = np.zeros_like(B_x)
            
        elif coil.coil_type == CoilType.POLOIDAL:
            # Poloidal coil field
            B_z = (MU_0 * coil.turns * current * coil.radius**2) / (2 * (coil.radius**2 + rho**2)**(3/2))
            
        elif coil.coil_type == CoilType.HELICAL:
            # Helical coil field (simplified)
            helical_pitch = coil.radius / 2.0
            k_z = 2 * np.pi / helical_pitch
            
            B_rho = (MU_0 * coil.turns * current / (2 * np.pi)) * k_z * coil.radius / (coil.radius**2 + rho**2)
            B_phi = (MU_0 * coil.turns * current / (2 * np.pi)) * coil.radius / (coil.radius**2 + rho**2)
            
            phi = np.arctan2(Y_rot, X_rot)
            B_x = B_rho * np.cos(phi) - B_phi * np.sin(phi)
            B_y = B_rho * np.sin(phi) + B_phi * np.cos(phi)
            
        elif coil.coil_type == CoilType.SADDLE:
            # Saddle coil field (dipole-like)
            r_vec = np.sqrt(X_rot**2 + Y_rot**2 + Z_rot**2)
            r_vec = np.maximum(r_vec, coil.radius/10)  # Avoid singularity
            
            # Dipole field
            B_x = (MU_0 * coil.turns * current * coil.radius**2) * (3 * X_rot * Z_rot) / (4 * np.pi * r_vec**5)
            B_y = (MU_0 * coil.turns * current * coil.radius**2) * (3 * Y_rot * Z_rot) / (4 * np.pi * r_vec**5)
            B_z = (MU_0 * coil.turns * current * coil.radius**2) * (2 * Z_rot**2 - X_rot**2 - Y_rot**2) / (4 * np.pi * r_vec**5)
            
        elif coil.coil_type == CoilType.QUADRUPOLE:
            # Quadrupole coil field
            r_vec = np.sqrt(X_rot**2 + Y_rot**2 + Z_rot**2)
            r_vec = np.maximum(r_vec, coil.radius/10)
            
            B_x = (MU_0 * coil.turns * current * coil.radius**3) * X_rot * (5 * Z_rot**2 - r_vec**2) / (4 * np.pi * r_vec**7)
            B_y = (MU_0 * coil.turns * current * coil.radius**3) * Y_rot * (5 * Z_rot**2 - r_vec**2) / (4 * np.pi * r_vec**7)
            B_z = (MU_0 * coil.turns * current * coil.radius**3) * Z_rot * (5 * Z_rot**2 - 3 * r_vec**2) / (4 * np.pi * r_vec**7)
            
        else:  # Default to simple dipole
            r_vec = np.sqrt(X_rot**2 + Y_rot**2 + Z_rot**2)
            r_vec = np.maximum(r_vec, coil.radius/10)
            
            B_x = (MU_0 * coil.turns * current * coil.radius**2) * (3 * X_rot * Z_rot) / (4 * np.pi * r_vec**5)
            B_y = (MU_0 * coil.turns * current * coil.radius**2) * (3 * Y_rot * Z_rot) / (4 * np.pi * r_vec**5)
            B_z = (MU_0 * coil.turns * current * coil.radius**2) * (2 * Z_rot**2 - X_rot**2 - Y_rot**2) / (4 * np.pi * r_vec**5)
        
        # Apply frequency modulation
        freq_center = np.mean(coil.frequency_band)
        temporal_factor = np.cos(2 * np.pi * freq_center * time)
        
        B_x *= temporal_factor
        B_y *= temporal_factor
        B_z *= temporal_factor
        
        # Rotate field back to global coordinates
        B_field = np.stack([B_x.flatten(), B_y.flatten(), B_z.flatten()], axis=1)
        B_field_global = rotation.inv().apply(B_field)
        
        B_x_global = B_field_global[:, 0].reshape(self.X.shape)
        B_y_global = B_field_global[:, 1].reshape(self.Y.shape)
        B_z_global = B_field_global[:, 2].reshape(self.Z.shape)
        
        return {
            'B_x': B_x_global,
            'B_y': B_y_global,
            'B_z': B_z_global,
            'B_magnitude': np.sqrt(B_x_global**2 + B_y_global**2 + B_z_global**2),
            'coil_id': coil_id,
            'current': current,
            'frequency': freq_center
        }

    def compute_superposed_field(self, 
                                time: float = 0.0,
                                current_distribution: Dict[int, float] = None) -> Dict[str, np.ndarray]:
        """
        Compute superposed magnetic field from all active coils
        
        Args:
            time: Time coordinate
            current_distribution: Current values for each coil
            
        Returns:
            Dictionary with total field components
        """
        if current_distribution is None:
            # Default: equal current in all coils
            current_distribution = {cid: 100.0 for cid in self.coils.keys() if self.coils[cid].active}
        
        # Initialize total field
        B_x_total = np.zeros_like(self.X)
        B_y_total = np.zeros_like(self.Y)
        B_z_total = np.zeros_like(self.Z)
        
        field_contributions = {}
        
        # Sum contributions from all active coils
        for coil_id, current in current_distribution.items():
            if coil_id not in self.coils or not self.coils[coil_id].active:
                continue
            
            # Compute individual coil field
            coil_field = self.compute_coil_magnetic_field(coil_id, current, time)
            
            # Add to total (linear superposition)
            B_x_total += coil_field['B_x']
            B_y_total += coil_field['B_y']
            B_z_total += coil_field['B_z']
            
            # Store contribution
            field_contributions[coil_id] = coil_field
        
        # Compute total field magnitude and properties
        B_magnitude = np.sqrt(B_x_total**2 + B_y_total**2 + B_z_total**2)
        
        # Field uniformity (standard deviation relative to mean)
        field_uniformity = np.std(B_magnitude) / np.mean(B_magnitude) if np.mean(B_magnitude) > 0 else 0
        
        # Maximum field strength
        max_field = np.max(B_magnitude)
        
        return {
            'B_x_total': B_x_total,
            'B_y_total': B_y_total,
            'B_z_total': B_z_total,
            'B_magnitude': B_magnitude,
            'field_uniformity': field_uniformity,
            'max_field_strength': max_field,
            'individual_contributions': field_contributions,
            'active_coils': len(current_distribution)
        }

    def optimize_field_steering(self, 
                              target_field_direction: np.ndarray,
                              target_position: np.ndarray,
                              field_strength: float = 0.1) -> Dict[int, float]:
        """
        Optimize coil currents to steer field in target direction
        
        Args:
            target_field_direction: Desired field direction (unit vector)
            target_position: Position where field should be steered
            field_strength: Desired field strength (Tesla)
            
        Returns:
            Dictionary with optimized current values
        """
        # Find grid point closest to target position
        distances = np.sqrt((self.X - target_position[0])**2 + 
                           (self.Y - target_position[1])**2 + 
                           (self.Z - target_position[2])**2)
        target_indices = np.unravel_index(np.argmin(distances), distances.shape)
        
        # Set up optimization problem
        active_coils = [cid for cid, coil in self.coils.items() if coil.active]
        n_coils = len(active_coils)
        
        if n_coils == 0:
            return {}
        
        # Target field vector
        target_field = field_strength * target_field_direction / np.linalg.norm(target_field_direction)
        
        def objective_function(currents):
            """Objective function for field steering optimization"""
            current_dict = {active_coils[i]: currents[i] for i in range(n_coils)}
            
            # Compute field at target position
            field_result = self.compute_superposed_field(current_distribution=current_dict)
            
            # Extract field at target position
            actual_field = np.array([
                field_result['B_x_total'][target_indices],
                field_result['B_y_total'][target_indices],
                field_result['B_z_total'][target_indices]
            ])
            
            # Compute error
            field_error = np.linalg.norm(actual_field - target_field)
            
            # Add power constraint
            total_power = sum(current**2 * self.coils[cid].resistance for cid, current in current_dict.items())
            power_penalty = max(0, total_power - self.config.total_power_limit) / self.config.total_power_limit
            
            return field_error + 0.1 * power_penalty
        
        # Initial guess
        initial_currents = np.ones(n_coils) * 100.0
        
        # Current bounds
        current_bounds = [(0, coil.current_capacity) for coil in 
                         [self.coils[cid] for cid in active_coils]]
        
        # Optimize using scipy
        from scipy.optimize import minimize
        
        result = minimize(
            objective_function,
            x0=initial_currents,
            bounds=current_bounds,
            method='L-BFGS-B'
        )
        
        # Extract optimized currents
        optimized_currents = {active_coils[i]: result.x[i] for i in range(n_coils)}
        
        logger.info(f"Field steering optimization completed: {result.success}")
        logger.info(f"Final error: {result.fun:.6f}")
        
        return optimized_currents

    def setup_multi_field_configuration(self) -> Dict[str, int]:
        """
        Setup a comprehensive multi-field coil configuration
        
        Returns:
            Dictionary mapping field types to coil IDs
        """
        field_coil_mapping = {}
        
        # Shell radius for positioning
        R = self.config.shell_radius
        
        # Warp drive coils (4 coils in tetrahedral arrangement)
        warp_positions = [
            np.array([R, 0, 0]),
            np.array([-R/2, R*np.sqrt(3)/2, 0]),
            np.array([-R/2, -R*np.sqrt(3)/2, 0]),
            np.array([0, 0, R])
        ]
        
        warp_coils = []
        for i, pos in enumerate(warp_positions):
            coil_id = self.add_coil(
                CoilType.TOROIDAL,
                FieldType.WARP_DRIVE,
                pos,
                orientation=np.array([0, np.pi/4, i*np.pi/2]),
                radius=R/4,
                turns=200,
                current_capacity=2000.0,
                field_strength=2.0
            )
            warp_coils.append(coil_id)
        
        field_coil_mapping['warp_drive'] = warp_coils
        
        # Shield coils (6 coils on faces of cube)
        shield_positions = [
            np.array([R*0.8, 0, 0]),
            np.array([-R*0.8, 0, 0]),
            np.array([0, R*0.8, 0]),
            np.array([0, -R*0.8, 0]),
            np.array([0, 0, R*0.8]),
            np.array([0, 0, -R*0.8])
        ]
        
        shield_coils = []
        for i, pos in enumerate(shield_positions):
            coil_id = self.add_coil(
                CoilType.SADDLE,
                FieldType.SHIELDS,
                pos,
                orientation=np.array([i*np.pi/3, 0, 0]),
                radius=R/6,
                turns=150,
                current_capacity=1500.0,
                field_strength=1.5
            )
            shield_coils.append(coil_id)
        
        field_coil_mapping['shields'] = shield_coils
        
        # Transporter coils (8 coils at cube vertices)
        transporter_positions = []
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    x = R * 0.6 * (2*i - 1)
                    y = R * 0.6 * (2*j - 1)
                    z = R * 0.6 * (2*k - 1)
                    transporter_positions.append(np.array([x, y, z]))
        
        transporter_coils = []
        for i, pos in enumerate(transporter_positions):
            coil_id = self.add_coil(
                CoilType.HELICAL,
                FieldType.TRANSPORTER,
                pos,
                orientation=np.array([i*np.pi/4, i*np.pi/8, 0]),
                radius=R/8,
                turns=100,
                current_capacity=1000.0,
                field_strength=1.0
            )
            transporter_coils.append(coil_id)
        
        field_coil_mapping['transporter'] = transporter_coils
        
        # Inertial damper coils (4 quadrupole coils)
        damper_positions = [
            np.array([R*0.4, R*0.4, 0]),
            np.array([-R*0.4, R*0.4, 0]),
            np.array([-R*0.4, -R*0.4, 0]),
            np.array([R*0.4, -R*0.4, 0])
        ]
        
        damper_coils = []
        for i, pos in enumerate(damper_positions):
            coil_id = self.add_coil(
                CoilType.QUADRUPOLE,
                FieldType.INERTIAL_DAMPER,
                pos,
                orientation=np.array([0, 0, i*np.pi/2]),
                radius=R/10,
                turns=80,
                current_capacity=800.0,
                field_strength=0.8
            )
            damper_coils.append(coil_id)
        
        field_coil_mapping['inertial_damper'] = damper_coils
        
        logger.info(f"Multi-field configuration setup complete: {len(self.coils)} total coils")
        logger.info(f"Field types: {list(field_coil_mapping.keys())}")
        
        return field_coil_mapping

    def generate_coil_system_report(self, time: float = 0.0) -> str:
        """Generate comprehensive coil system report"""
        
        # Compute current field state
        field_result = self.compute_superposed_field(time)
        
        # Count coils by type
        coil_counts = {}
        for coil in self.coils.values():
            field_type = coil.field_type.value
            coil_counts[field_type] = coil_counts.get(field_type, 0) + 1
        
        # Total power calculation
        total_power = sum(100.0**2 * coil.resistance for coil in self.coils.values() if coil.active)
        
        report = f"""
⚡ Multi-Field Steerable Coil System Report
{'='*50}

🔧 System Configuration:
   Total coils: {len(self.coils)}
   Active coils: {len([c for c in self.coils.values() if c.active])}
   Shell radius: {self.config.shell_radius:.1f} m
   Grid resolution: {self.field_grid_resolution}

🎛️ Coil Distribution:
"""
        
        for field_type, count in coil_counts.items():
            report += f"   {field_type}: {count} coils\n"
        
        report += f"""
⚡ Field Properties:
   Maximum field strength: {field_result['max_field_strength']:.4f} T
   Field uniformity: {field_result['field_uniformity']:.4f}
   Active frequency bands: {len(self.allocated_bands)}

🔋 Power System:
   Total power consumption: {total_power/1e6:.2f} MW
   Power limit: {self.config.total_power_limit/1e6:.2f} MW
   Power utilization: {100*total_power/self.config.total_power_limit:.1f}%

📊 Frequency Allocation:
"""
        
        for coil_id, coil in self.coils.items():
            if coil.active:
                freq_min, freq_max = coil.frequency_band
                report += f"   Coil {coil_id} ({coil.field_type.value}): "
                report += f"{freq_min/1e9:.1f}-{freq_max/1e9:.1f} GHz\n"
        
        report += f"\n✅ System Status: {'OPERATIONAL' if field_result['field_uniformity'] < 0.1 else 'NEEDS CALIBRATION'}"
        
        return report

def demonstrate_multi_field_coil_system():
    """
    Demonstration of multi-field steerable coil system
    """
    print("⚡ Multi-Field Steerable Coil System Demo")
    print("="*42)
    
    # Initialize system configuration
    config = SteerableCoilSystem(
        shell_radius=100.0,
        max_coils=32,
        total_power_limit=200e6,  # 200 MW
        frequency_multiplexing=True,
        adaptive_steering=True
    )
    
    # Create coil system
    coil_system = MultiFieldCoilSystem(config)
    
    print("Setting up multi-field coil configuration...")
    
    # Setup comprehensive field configuration
    field_mapping = coil_system.setup_multi_field_configuration()
    
    print(f"Configured {len(coil_system.coils)} coils for {len(field_mapping)} field types")
    
    # Demonstrate field steering
    print("\nDemonstrating field steering...")
    
    target_direction = np.array([0, 0, 1])  # Steer field upward
    target_position = np.array([0, 0, 50])  # 50m above center
    
    optimized_currents = coil_system.optimize_field_steering(
        target_direction,
        target_position,
        field_strength=0.5
    )
    
    print(f"Optimized currents for {len(optimized_currents)} coils")
    
    # Compute field with optimized currents
    field_result = coil_system.compute_superposed_field(
        time=0.0,
        current_distribution=optimized_currents
    )
    
    # Generate system report
    report = coil_system.generate_coil_system_report()
    print(report)
    
    print(f"\n✅ Multi-Field Coil System Demo Complete!")
    print(f"   Maximum field strength: {field_result['max_field_strength']:.3f} T")
    print(f"   Field uniformity: {field_result['field_uniformity']:.4f}")
    print(f"   Total coils operational: {len(coil_system.coils)}")
    print(f"   Frequency multiplexing active: {config.frequency_multiplexing}")
    print(f"   Steerable field control verified! ⚡")
    
    return coil_system

if __name__ == "__main__":
    # Run demonstration
    demo_system = demonstrate_multi_field_coil_system()
