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
from datetime import datetime

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
    
    def generate_helmholtz_pair(self, R: float, spacing: float, n_turns: int,
                               n_segments: int = 200) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate Helmholtz coil pair with optimal field uniformity.
        
        Args:
            R: Coil radius (m)
            spacing: Coil separation distance (m)
            n_turns: Number of turns per coil
            n_segments: Path discretization points per coil
            
        Returns:
            Tuple of (coil1_path, coil2_path) arrays
        """
        # Optimal Helmholtz spacing is R for maximum uniformity
        if abs(spacing - R) > 0.1 * R:
            print(f"⚠️ Non-optimal spacing: {spacing:.3f}m vs optimal {R:.3f}m")
        
        # Generate two circular coils
        theta = np.linspace(0, 2*np.pi * n_turns, n_segments)
        
        # First coil at z = -spacing/2
        X1 = R * np.cos(theta)
        Y1 = R * np.sin(theta)
        Z1 = np.full_like(theta, -spacing/2)
        coil1 = np.stack([X1, Y1, Z1], axis=1)
        
        # Second coil at z = +spacing/2
        X2 = R * np.cos(theta)
        Y2 = R * np.sin(theta)
        Z2 = np.full_like(theta, spacing/2)
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

    def generate_multi_layer_toroidal(self, R0: float, a: float, n_layers: int,
                                     n_turns_per_layer: int, layer_spacing: float = 0.01,
                                     n_segments: int = 200) -> List[np.ndarray]:
        """
        Generate multi-layer toroidal coil for higher field strength.
        
        Args:
            R0: Major radius (m)
            a: Minor radius (m)
            n_layers: Number of concentric layers
            n_turns_per_layer: Turns per layer
            layer_spacing: Radial spacing between layers (m)
            n_segments: Discretization points per layer
            
        Returns:
            List of coil path arrays, one per layer
        """
        coil_layers = []
        
        for layer in range(n_layers):
            # Adjust minor radius for each layer
            layer_a = a + layer * layer_spacing
            
            # Generate toroidal path for this layer
            layer_path = self.generate_toroidal_coil(
                R0, layer_a, n_turns_per_layer, n_segments
            )
            
            coil_layers.append(layer_path)
        
        return coil_layers
    
    def export_coil_to_step(self, coil_path: np.ndarray, filename: str,
                           conductor_radius: float = 0.001) -> None:
        """
        Export 3D coil geometry to STEP file for CAD integration.
        
        Args:
            coil_path: (N, 3) array of 3D path points
            filename: Output STEP filename
            conductor_radius: Conductor cross-section radius (m)
        """
        try:
            # Try to use FreeCAD Python API if available
            import FreeCAD
            import Part
            
            # Create document
            doc = FreeCAD.newDocument()
            
            # Create wire from points
            points = [FreeCAD.Vector(float(p[0]), float(p[1]), float(p[2])) 
                     for p in coil_path]
            
            # Create spline through points
            spline = Part.BSplineCurve()
            spline.interpolate(points)
            
            # Create wire from spline
            wire = Part.Wire(spline.toShape())
            
            # Create pipe (conductor with circular cross-section)
            circle = Part.Circle(FreeCAD.Vector(0, 0, 0), 
                               FreeCAD.Vector(0, 0, 1), 
                               conductor_radius)
            circle_wire = Part.Wire(circle.toShape())
            
            # Sweep circle along wire to create conductor
            conductor = wire.makePipeShell([circle_wire], True, True)
            
            # Add to document
            conductor_obj = doc.addObject("Part::Feature", "Conductor")
            conductor_obj.Shape = conductor
            
            # Export to STEP
            Part.export([conductor_obj], filename)
            
            # Clean up
            FreeCAD.closeDocument(doc.Name)
            
            print(f"✓ Exported coil to {filename}")
            
        except ImportError:
            print("⚠️ FreeCAD not available, generating simple point cloud export")
            self._export_point_cloud(coil_path, filename.replace('.step', '_points.csv'))
    
    def _export_point_cloud(self, coil_path: np.ndarray, filename: str) -> None:
        """Fallback export as CSV point cloud."""
        import csv
        
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['X', 'Y', 'Z'])
            for point in coil_path:
                writer.writerow([f"{point[0]:.6f}", f"{point[1]:.6f}", f"{point[2]:.6f}"])
        
        print(f"✓ Exported point cloud to {filename}")
    
    def batch_export_coil_system(self, coil_system: List[CoilGeometry3D], 
                                output_dir: str = "cad_exports") -> None:
        """
        Batch export complete coil system to CAD files.
        
        Args:
            coil_system: List of 3D coil geometries
            output_dir: Output directory for CAD files
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"📁 Exporting {len(coil_system)} coils to {output_dir}/")
        
        for i, coil in enumerate(coil_system):
            # Generate filename
            filename = f"{output_dir}/{coil.coil_type}_{i:02d}.step"
            
            # Export coil
            self.export_coil_to_step(coil.path_points, filename)
            
            # Export metadata
            metadata_file = filename.replace('.step', '_metadata.json')
            metadata = {
                'coil_type': coil.coil_type,
                'current': coil.current,
                'n_turns': coil.n_turns,
                'n_path_points': len(coil.path_points),
                'bounding_box': {
                    'x_min': float(np.min(coil.path_points[:, 0])),
                    'x_max': float(np.max(coil.path_points[:, 0])),
                    'y_min': float(np.min(coil.path_points[:, 1])),
                    'y_max': float(np.max(coil.path_points[:, 1])),
                    'z_min': float(np.min(coil.path_points[:, 2])),
                    'z_max': float(np.max(coil.path_points[:, 2]))
                }
            }
            
            import json
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        print(f"✓ Batch export complete: {len(coil_system)} coils exported")
    
    def generate_manufacturing_documentation(self, coil_system: List[CoilGeometry3D],
                                           output_file: str = "manufacturing_spec.json") -> None:
        """
        Generate comprehensive manufacturing documentation.
        
        Args:
            coil_system: List of 3D coil geometries
            output_file: Output specification file
        """
        manufacturing_spec = {
            'project_info': {
                'name': 'Warp Field Coil System',
                'version': '1.0',
                'generated_date': str(datetime.now()),
                'total_coils': len(coil_system)
            },
            'coil_specifications': [],
            'material_requirements': {},
            'manufacturing_tolerances': {
                'position_tolerance': '±0.1mm',
                'current_tolerance': '±1%',
                'temperature_rating': '4.2K (superconducting)',
                'magnetic_field_rating': '20T'
            },
            'assembly_instructions': []
        }
        
        total_wire_length = 0.0
        max_current = 0.0
        
        for i, coil in enumerate(coil_system):
            # Calculate wire length
            path_segments = np.diff(coil.path_points, axis=0)
            segment_lengths = np.linalg.norm(path_segments, axis=1)
            wire_length = np.sum(segment_lengths) * coil.n_turns
            total_wire_length += wire_length
            
            max_current = max(max_current, abs(coil.current))
            
            # Coil specification
            coil_spec = {
                'coil_id': f"COIL_{i:02d}",
                'type': coil.coil_type,
                'current_rating': f"{coil.current:.1f}A",
                'turns': coil.n_turns,
                'wire_length': f"{wire_length:.2f}m",
                'center_position': {
                    'x': float(np.mean(coil.path_points[:, 0])),
                    'y': float(np.mean(coil.path_points[:, 1])),
                    'z': float(np.mean(coil.path_points[:, 2]))
                },
                'dimensions': {
                    'x_span': float(np.ptp(coil.path_points[:, 0])),
                    'y_span': float(np.ptp(coil.path_points[:, 1])),
                    'z_span': float(np.ptp(coil.path_points[:, 2]))
                }
            }
            
            manufacturing_spec['coil_specifications'].append(coil_spec)
        
        # Material requirements
        manufacturing_spec['material_requirements'] = {
            'superconducting_wire': {
                'total_length': f"{total_wire_length:.1f}m",
                'recommended_type': 'YBCO or Nb3Sn',
                'cross_section': '1mm²',
                'critical_current': f">{max_current*1.5:.0f}A at 4.2K"
            },
            'support_structure': {
                'material': 'Stainless steel 316L',
                'cryogenic_rating': '4.2K compatible'
            },
            'insulation': {
                'type': 'Kapton polyimide',
                'thickness': '0.1mm',
                'temperature_rating': '2K - 300K'
            }
        }
        
        # Assembly instructions
        manufacturing_spec['assembly_instructions'] = [
            "1. Fabricate support structure according to CAD drawings",
            "2. Wind superconducting coils with specified turn counts",
            "3. Install coils with ±0.1mm position tolerance",
            "4. Connect current leads with low-resistance joints",
            "5. Install thermal insulation and radiation shielding",
            "6. Perform leak testing at room temperature",
            "7. Cool down gradually to operating temperature (4.2K)",
            "8. Verify field profiles before final assembly"
        ]
        
        # Save specification
        import json
        with open(output_file, 'w') as f:
            json.dump(manufacturing_spec, f, indent=2)
        
        print(f"✓ Manufacturing documentation saved to {output_file}")
        print(f"  Total wire length: {total_wire_length:.1f}m")
        print(f"  Max current: {max_current:.1f}A")
        print(f"  Number of coils: {len(coil_system)}")

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
