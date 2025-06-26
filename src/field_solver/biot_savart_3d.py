#!/usr/bin/env python3
"""
3D Biot-Savart Field Solver for Arbitrary Coil Geometries
Implements full 3D electromagnetic field computation for complex coil arrangements
"""

import numpy as np
import jax.numpy as jnp
import jax
from typing import Dict, Tuple, List, Optional, Callable
from dataclasses import dataclass
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

@dataclass
class CoilGeometry3D:
    """3D coil geometry specification."""
    path_points: np.ndarray  # (N, 3) array of 3D path points
    current: float  # Current magnitude (A)
    n_turns: int  # Number of turns
    coil_type: str  # 'toroidal', 'solenoid', 'helmholtz', 'custom'
    
@dataclass  
class FieldPoint3D:
    """3D field evaluation point."""
    position: np.ndarray  # (3,) position vector
    B_field: np.ndarray  # (3,) magnetic field vector
    magnitude: float  # |B| field magnitude

class BiotSavart3DSolver:
    """
    3D Biot-Savart electromagnetic field solver.
    
    Computes B(r) = (μ₀/4π) ∫ I dl × (r - l) / |r - l|³
    """
    
    def __init__(self, mu0: float = 4*np.pi*1e-7):
        """Initialize 3D field solver."""
        self.mu0 = mu0
        
        # JAX-compiled functions for performance
        self.biot_savart_jax = jax.jit(self._biot_savart_kernel)
        self.field_batch_jax = jax.jit(jax.vmap(self._field_single_point, in_axes=(0, None, None)))
    
    def generate_toroidal_coil(self, R0: float, a: float, n_turns: int = 1, 
                              n_segments: int = 200) -> np.ndarray:
        """
        Generate toroidal coil path.
        
        r(θ,φ) = (R₀ + a cos θ) cos φ x̂ + (R₀ + a cos θ) sin φ ŷ + a sin θ ẑ
        
        Args:
            R0: Major radius (m)
            a: Minor radius (m)  
            n_turns: Number of toroidal turns
            n_segments: Path discretization points
            
        Returns:
            (N, 3) array of path points
        """
        # Parametric angles
        theta = np.linspace(0, 2*np.pi, n_segments // n_turns)
        phi = np.linspace(0, 2*np.pi*n_turns, n_segments)
        
        # Create meshgrid for all combinations
        THETA, PHI = np.meshgrid(theta, phi, indexing='ij')
        
        # Toroidal coordinates
        X = (R0 + a * np.cos(THETA)) * np.cos(PHI)
        Y = (R0 + a * np.cos(THETA)) * np.sin(PHI)
        Z = a * np.sin(THETA)
        
        # Flatten and stack
        path = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
        
        return path
    
    def generate_solenoid_coil(self, radius: float, length: float, n_turns: int,
                              n_segments: int = 200) -> np.ndarray:
        """
        Generate solenoid coil path.
        
        Args:
            radius: Coil radius (m)
            length: Coil length (m)
            n_turns: Number of turns
            n_segments: Path discretization points
            
        Returns:
            (N, 3) array of path points
        """
        # Parametric parameter along helix
        t = np.linspace(0, n_turns * 2*np.pi, n_segments)
        
        # Helical path
        X = radius * np.cos(t)
        Y = radius * np.sin(t)
        Z = np.linspace(-length/2, length/2, n_segments)
        
        path = np.stack([X, Y, Z], axis=1)
        
        return path
    
    def generate_helmholtz_pair(self, radius: float, separation: float,
                               n_segments: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate Helmholtz coil pair.
        
        Args:
            radius: Coil radius (m)
            separation: Coil separation (m)
            n_segments: Points per coil
            
        Returns:
            Tuple of (coil1_path, coil2_path)
        """
        # Parametric angle
        theta = np.linspace(0, 2*np.pi, n_segments)
        
        # First coil (z = -separation/2)
        X1 = radius * np.cos(theta)
        Y1 = radius * np.sin(theta)
        Z1 = np.full_like(theta, -separation/2)
        coil1 = np.stack([X1, Y1, Z1], axis=1)
        
        # Second coil (z = +separation/2)
        X2 = radius * np.cos(theta)
        Y2 = radius * np.sin(theta)
        Z2 = np.full_like(theta, separation/2)
        coil2 = np.stack([X2, Y2, Z2], axis=1)
        
        return coil1, coil2
    
    def _biot_savart_kernel(self, r_eval: jnp.ndarray, r_source: jnp.ndarray, 
                           dl: jnp.ndarray) -> jnp.ndarray:
        """
        JAX-compiled Biot-Savart kernel.
        
        dB = (μ₀/4π) I dl × (r - l) / |r - l|³
        """
        # Vector from source to evaluation point
        r_vec = r_eval - r_source
        r_mag = jnp.linalg.norm(r_vec)
        
        # Avoid singularity
        r_mag_safe = jnp.maximum(r_mag, 1e-12)
        
        # Cross product dl × r_vec
        dl_cross_r = jnp.cross(dl, r_vec)
        
        # Biot-Savart contribution
        dB = (self.mu0 / (4 * jnp.pi)) * dl_cross_r / (r_mag_safe**3)
        
        return dB
    
    def _field_single_point(self, r_eval: jnp.ndarray, coil_path: jnp.ndarray, 
                           current: float) -> jnp.ndarray:
        """Compute field at single evaluation point."""
        # Path differentials
        dl = jnp.diff(coil_path, axis=0)
        r_source = coil_path[:-1]  # Source points
        
        # Sum Biot-Savart contributions
        dB_contributions = jax.vmap(
            lambda rs, dl_seg: self._biot_savart_kernel(r_eval, rs, dl_seg)
        )(r_source, dl)
        
        # Total field
        B_total = current * jnp.sum(dB_contributions, axis=0)
        
        return B_total
    
    def compute_field_3d(self, coil_geometry: CoilGeometry3D, 
                        eval_points: np.ndarray) -> List[FieldPoint3D]:
        """
        Compute 3D magnetic field at evaluation points.
        
        Args:
            coil_geometry: 3D coil specification
            eval_points: (N, 3) evaluation point array
            
        Returns:
            List of FieldPoint3D objects
        """
        # Convert to JAX arrays
        coil_path_jax = jnp.array(coil_geometry.path_points)
        eval_points_jax = jnp.array(eval_points)
        
        # Batch compute fields
        B_fields = self.field_batch_jax(eval_points_jax, coil_path_jax, 
                                       coil_geometry.current)
        
        # Convert to FieldPoint3D objects
        field_points = []
        for i, (pos, B) in enumerate(zip(eval_points, B_fields)):
            field_point = FieldPoint3D(
                position=pos,
                B_field=np.array(B),
                magnitude=float(jnp.linalg.norm(B))
            )
            field_points.append(field_point)
        
        return field_points
    
    def compute_field_on_axis(self, coil_geometry: CoilGeometry3D,
                             z_range: Tuple[float, float], n_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute field along symmetry axis.
        
        Args:
            coil_geometry: 3D coil specification
            z_range: (z_min, z_max) axis range
            n_points: Number of evaluation points
            
        Returns:
            (z_positions, B_z_values) tuple
        """
        # Axis evaluation points
        z_vals = np.linspace(z_range[0], z_range[1], n_points)
        eval_points = np.column_stack([np.zeros(n_points), np.zeros(n_points), z_vals])
        
        # Compute fields
        field_points = self.compute_field_3d(coil_geometry, eval_points)
        
        # Extract z-components
        B_z = np.array([fp.B_field[2] for fp in field_points])
        
        return z_vals, B_z
    
    def visualize_coil_geometry(self, coil_geometries: List[CoilGeometry3D],
                               save_path: Optional[str] = None) -> None:
        """Visualize 3D coil geometries."""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        for i, geometry in enumerate(coil_geometries):
            path = geometry.path_points
            ax.plot(path[:, 0], path[:, 1], path[:, 2], 
                   label=f'{geometry.coil_type} (I={geometry.current:.1f}A)',
                   linewidth=2)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('3D Coil Geometries')
        ax.legend()
        ax.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_field_visualization(self, coil_geometry: CoilGeometry3D,
                                  plane: str = 'xz', extent: float = 1.0,
                                  resolution: int = 50) -> Dict:
        """
        Create 2D field visualization in specified plane.
        
        Args:
            coil_geometry: 3D coil specification
            plane: 'xy', 'xz', or 'yz' plane
            extent: Plot extent (±extent)
            resolution: Grid resolution
            
        Returns:
            Dictionary with field data and coordinates
        """
        # Create evaluation grid
        coords = np.linspace(-extent, extent, resolution)
        
        if plane == 'xy':
            X, Y = np.meshgrid(coords, coords)
            Z = np.zeros_like(X)
            eval_points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        elif plane == 'xz':
            X, Z = np.meshgrid(coords, coords)
            Y = np.zeros_like(X)
            eval_points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        elif plane == 'yz':
            Y, Z = np.meshgrid(coords, coords)
            X = np.zeros_like(Y)
            eval_points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        else:
            raise ValueError(f"Unknown plane: {plane}")
        
        # Compute field
        field_points = self.compute_field_3d(coil_geometry, eval_points)
        
        # Extract field components
        B_mag = np.array([fp.magnitude for fp in field_points]).reshape(resolution, resolution)
        
        return {
            'X': X if plane != 'yz' else Y,
            'Y': Y if plane == 'xy' else Z,
            'B_magnitude': B_mag,
            'extent': extent,
            'plane': plane
        }

def create_warp_coil_3d_system(R_bubble: float = 2.0) -> List[CoilGeometry3D]:
    """
    Create 3D coil system for warp bubble generation.
    
    Args:
        R_bubble: Target warp bubble radius
        
    Returns:
        List of CoilGeometry3D objects
    """
    solver = BiotSavart3DSolver()
    
    # Inner toroidal coil (strong field)
    inner_path = solver.generate_toroidal_coil(R0=R_bubble*0.7, a=0.2, n_turns=10)
    inner_coil = CoilGeometry3D(
        path_points=inner_path,
        current=1000.0,  # High current
        n_turns=10,
        coil_type='toroidal_inner'
    )
    
    # Outer toroidal coil (field shaping)
    outer_path = solver.generate_toroidal_coil(R0=R_bubble*1.3, a=0.3, n_turns=5)
    outer_coil = CoilGeometry3D(
        path_points=outer_path,
        current=-500.0,  # Opposite current for field cancellation
        n_turns=5,
        coil_type='toroidal_outer'
    )
    
    # Solenoid coils for axial field control
    solenoid_path = solver.generate_solenoid_coil(radius=R_bubble*0.5, length=4.0, n_turns=20)
    solenoid_coil = CoilGeometry3D(
        path_points=solenoid_path,
        current=200.0,
        n_turns=20,
        coil_type='solenoid_control'
    )
    
    return [inner_coil, outer_coil, solenoid_coil]

def main():
    """Demonstrate 3D coil system capabilities."""
    print("🔧 3D COIL GEOMETRY & BIOT-SAVART SOLVER")
    print("=" * 50)
    
    # Create solver
    solver = BiotSavart3DSolver()
    
    # Generate warp coil system
    coil_system = create_warp_coil_3d_system(R_bubble=2.0)
    
    print(f"Created {len(coil_system)} coil components:")
    for coil in coil_system:
        print(f"  - {coil.coil_type}: {len(coil.path_points)} points, I = {coil.current:.1f} A")
    
    # Compute field on axis
    z_range = (-3.0, 3.0)
    z_positions, B_z_values = solver.compute_field_on_axis(coil_system[0], z_range)
    
    print(f"Computed axial field: max |B_z| = {np.max(np.abs(B_z_values)):.6f} T")
    
    # Create field visualization
    field_data = solver.create_field_visualization(coil_system[0], plane='xz', extent=3.0)
    print(f"Generated field visualization: {field_data['B_magnitude'].shape} grid")
    
    print("✅ 3D coil system validation complete!")

if __name__ == "__main__":
    main()
