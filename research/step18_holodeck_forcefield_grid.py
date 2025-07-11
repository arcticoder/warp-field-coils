#!/usr/bin/env python3
"""
Step 18: Holodeck Force-Field Grid Implementation
Adaptive refinement of force-field grid with quadratic potentials
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass

@dataclass
class GridParams:
    """Parameters for the force-field grid."""
    bounds: Dict[str, Tuple[float, float]]  # {'x': (min, max), 'y': (min, max), 'z': (min, max)}
    base_spacing: float = 0.2  # Base grid spacing (m)
    max_nodes: int = 5000  # Maximum number of nodes
    force_threshold: float = 5.0  # Force threshold for refinement (N)
    
@dataclass
class Node:
    """A node in the force-field grid."""
    position: np.ndarray
    material_type: str = "standard"
    spacing: float = 0.2
    force: Optional[np.ndarray] = None
    potential: Optional[float] = None

class ForceFieldGrid:
    """
    Holodeck Force-Field Grid with adaptive refinement.
    
    Implements:
    - Quadratic potentials: V(x) = ½k||x-x₀||²
    - Force computation: F(x) = -∇V = -k(x-x₀)
    - Adaptive mesh refinement where |F| > threshold
    """
    
    def __init__(self, params: GridParams):
        self.params = params
        self.nodes = []
        self.field_centers = []
        self.field_stiffness = []
        self.interaction_zones = []
        
        # Initialize base grid
        self._initialize_base_grid()
        
    def _initialize_base_grid(self):
        """Initialize the base uniform grid."""
        bounds = self.params.bounds
        spacing = self.params.base_spacing
        
        # Create uniform grid points
        x_points = np.arange(bounds['x'][0], bounds['x'][1] + spacing, spacing)
        y_points = np.arange(bounds['y'][0], bounds['y'][1] + spacing, spacing)
        z_points = np.arange(bounds['z'][0], bounds['z'][1] + spacing, spacing)
        
        # Generate all combinations
        for x in x_points:
            for y in y_points:
                for z in z_points:
                    if len(self.nodes) < self.params.max_nodes:
                        node = Node(
                            position=np.array([x, y, z]),
                            material_type="standard",
                            spacing=spacing
                        )
                        self.nodes.append(node)
        
        print(f"✓ Initialized base grid with {len(self.nodes)} nodes")
    
    def add_field_source(self, center: np.ndarray, stiffness: float):
        """Add a quadratic potential field source."""
        self.field_centers.append(center)
        self.field_stiffness.append(stiffness)
        print(f"✓ Added field source at {center} with stiffness {stiffness}")
    
    def compute_potential(self, position: np.ndarray) -> float:
        """
        Compute total potential at a position.
        V(x) = Σᵢ ½kᵢ||x-xᵢ||²
        """
        total_potential = 0.0
        
        for center, k in zip(self.field_centers, self.field_stiffness):
            r_vec = position - center
            V_i = 0.5 * k * np.dot(r_vec, r_vec)
            total_potential += V_i
            
        return total_potential
    
    def compute_force(self, position: np.ndarray) -> np.ndarray:
        """
        Compute total force at a position.
        F(x) = -∇V = -Σᵢ kᵢ(x-xᵢ)
        """
        total_force = np.zeros(3)
        
        for center, k in zip(self.field_centers, self.field_stiffness):
            F_i = -k * (position - center)
            total_force += F_i
            
        return total_force
    
    def evaluate_grid_forces(self):
        """Evaluate forces at all grid nodes."""
        for node in self.nodes:
            node.force = self.compute_force(node.position)
            node.potential = self.compute_potential(node.position)
    
    def find_high_force_zones(self) -> List[np.ndarray]:
        """Find zones where force magnitude exceeds threshold."""
        if not self.nodes[0].force is not None:
            self.evaluate_grid_forces()
        
        high_force_positions = []
        
        for node in self.nodes:
            force_magnitude = np.linalg.norm(node.force)
            if force_magnitude > self.params.force_threshold:
                high_force_positions.append(node.position.copy())
        
        return high_force_positions
    
    def add_interaction_zone(self, center: np.ndarray, radius: float = 0.3, 
                           material_type: str = "high_res", spacing: float = None):
        """Add a high-resolution interaction zone around a point."""
        if spacing is None:
            spacing = self.params.base_spacing / 2  # Higher resolution
        
        # Remove existing nodes in this region
        self.nodes = [node for node in self.nodes 
                     if np.linalg.norm(node.position - center) > radius]
        
        # Add fine-resolution nodes
        refined_points = []
        x_range = np.arange(center[0] - radius, center[0] + radius + spacing, spacing)
        y_range = np.arange(center[1] - radius, center[1] + radius + spacing, spacing)
        z_range = np.arange(center[2] - radius, center[2] + radius + spacing, spacing)
        
        for x in x_range:
            for y in y_range:
                for z in z_range:
                    pos = np.array([x, y, z])
                    if (np.linalg.norm(pos - center) <= radius and 
                        len(self.nodes) < self.params.max_nodes):
                        
                        node = Node(
                            position=pos,
                            material_type=material_type,
                            spacing=spacing
                        )
                        self.nodes.append(node)
                        refined_points.append(pos)
        
        self.interaction_zones.append({
            'center': center,
            'radius': radius,
            'material_type': material_type,
            'nodes_added': len(refined_points)
        })
        
        print(f"✓ Added interaction zone: {len(refined_points)} nodes at {center}")
    
    def adaptive_refinement(self) -> Dict:
        """Perform adaptive mesh refinement based on force magnitude."""
        print("🔍 Performing adaptive mesh refinement...")
        
        initial_node_count = len(self.nodes)
        
        # Evaluate forces
        self.evaluate_grid_forces()
        
        # Find high-force zones
        high_force_zones = self.find_high_force_zones()
        
        # Add interaction zones at high-force locations
        for zone_center in high_force_zones:
            self.add_interaction_zone(
                center=zone_center,
                radius=0.3,
                material_type="high_res"
            )
        
        final_node_count = len(self.nodes)
        
        refinement_results = {
            'initial_nodes': initial_node_count,
            'high_force_zones': len(high_force_zones),
            'final_nodes': final_node_count,
            'nodes_added': final_node_count - initial_node_count,
            'refinement_ratio': final_node_count / initial_node_count
        }
        
        print(f"✓ Refinement complete: {initial_node_count} → {final_node_count} nodes")
        print(f"✓ Found {len(high_force_zones)} high-force zones")
        
        return refinement_results
    
    def run_diagnostics(self) -> Dict:
        """Run comprehensive grid diagnostics."""
        print("🔍 Running force-field grid diagnostics...")
        
        # Evaluate all forces if not done already
        self.evaluate_grid_forces()
        
        # Compute statistics
        forces = np.array([node.force for node in self.nodes])
        potentials = np.array([node.potential for node in self.nodes])
        force_magnitudes = np.linalg.norm(forces, axis=1)
        
        diagnostics = {
            'total_nodes': len(self.nodes),
            'field_sources': len(self.field_centers),
            'interaction_zones': len(self.interaction_zones),
            'force_statistics': {
                'max_force': float(np.max(force_magnitudes)),
                'mean_force': float(np.mean(force_magnitudes)),
                'std_force': float(np.std(force_magnitudes)),
                'nodes_above_threshold': int(np.sum(force_magnitudes > self.params.force_threshold))
            },
            'potential_statistics': {
                'max_potential': float(np.max(potentials)),
                'min_potential': float(np.min(potentials)),
                'mean_potential': float(np.mean(potentials))
            },
            'grid_quality': {
                'resolution_variety': len(set(node.spacing for node in self.nodes)),
                'material_types': len(set(node.material_type for node in self.nodes))
            }
        }
        
        print(f"✓ Grid diagnostics complete")
        print(f"  Max force: {diagnostics['force_statistics']['max_force']:.2f} N")
        print(f"  Nodes above threshold: {diagnostics['force_statistics']['nodes_above_threshold']}")
        print(f"  Resolution levels: {diagnostics['grid_quality']['resolution_variety']}")
        
        return diagnostics
    
    def visualize_grid(self, save_path: str = "step18_holodeck_grid.png"):
        """Visualize the force-field grid and refinement zones."""
        if not hasattr(self.nodes[0], 'force') or self.nodes[0].force is None:
            self.evaluate_grid_forces()
        
        # Extract data for plotting
        positions = np.array([node.position for node in self.nodes])
        forces = np.array([node.force for node in self.nodes])
        force_mags = np.linalg.norm(forces, axis=1)
        
        # Create 3D plot
        fig = plt.figure(figsize=(15, 5))
        
        # 2D projection (XY plane)
        ax1 = fig.add_subplot(131)
        scatter = ax1.scatter(positions[:, 0], positions[:, 1], 
                            c=force_mags, cmap='hot', s=20, alpha=0.7)
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_title('Force Magnitude (XY plane)')
        plt.colorbar(scatter, ax=ax1, label='Force (N)')
        
        # Mark field centers
        for center in self.field_centers:
            ax1.plot(center[0], center[1], 'b*', markersize=15, label='Field Source')
        
        # Mark interaction zones
        for zone in self.interaction_zones:
            circle = plt.Circle((zone['center'][0], zone['center'][1]), 
                              zone['radius'], fill=False, color='red', linestyle='--')
            ax1.add_patch(circle)
        
        # Force vector field (2D slice)
        ax2 = fig.add_subplot(132)
        # Sample subset for vector plot
        step = max(1, len(positions) // 50)
        pos_sample = positions[::step]
        force_sample = forces[::step]
        
        ax2.quiver(pos_sample[:, 0], pos_sample[:, 1], 
                  force_sample[:, 0], force_sample[:, 1], 
                  alpha=0.7, scale=100)
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_title('Force Vector Field')
        
        # Force magnitude histogram
        ax3 = fig.add_subplot(133)
        ax3.hist(force_mags, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax3.axvline(self.params.force_threshold, color='red', linestyle='--', 
                   label=f'Threshold = {self.params.force_threshold}')
        ax3.set_xlabel('Force Magnitude (N)')
        ax3.set_ylabel('Node Count')
        ax3.set_title('Force Distribution')
        ax3.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Grid visualization saved to {save_path}")

def main():
    """Demonstration of Step 18: Holodeck Force-Field Grid"""
    print("=== STEP 18: HOLODECK FORCE-FIELD GRID ===")
    
    # Initialize grid
    grid_params = GridParams(
        bounds={'x': (-2, 2), 'y': (-2, 2), 'z': (-1, 1)},
        base_spacing=0.4,
        max_nodes=2000,
        force_threshold=5.0
    )
    
    grid = ForceFieldGrid(grid_params)
    
    # Add force field sources
    grid.add_field_source(center=np.array([0, 0, 0]), stiffness=1000)
    grid.add_field_source(center=np.array([1, 1, 0]), stiffness=500)
    grid.add_field_source(center=np.array([-1, -1, 0]), stiffness=750)
    
    # Initial diagnostics
    print("\n📊 Initial Grid State:")
    initial_diag = grid.run_diagnostics()
    
    # Perform adaptive refinement
    print("\n🔧 Performing Adaptive Refinement:")
    refinement_results = grid.adaptive_refinement()
    
    # Final diagnostics
    print("\n📊 Final Grid State:")
    final_diag = grid.run_diagnostics()
    
    # Visualize results
    grid.visualize_grid()
    
    print("\n📈 REFINEMENT SUMMARY:")
    print(f"Initial nodes: {refinement_results['initial_nodes']}")
    print(f"High-force zones: {refinement_results['high_force_zones']}")
    print(f"Final nodes: {refinement_results['final_nodes']}")
    print(f"Refinement ratio: {refinement_results['refinement_ratio']:.2f}x")
    print(f"Max force: {final_diag['force_statistics']['max_force']:.2f} N")

if __name__ == "__main__":
    main()
