"""
Holodeck Force-Field Grid Module
===============================

Implements high-density micro tractor beam array for tactile feedback simulation.

Mathematical Foundation:
F_i = -k_i(x - x_i) * exp(-||x - x_i||²/(2σ_i²))

Where:
- F_i: Force from node i
- k_i: Node stiffness/compliance  
- x_i: Node position
- σ_i: Interaction radius
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Callable
import logging
import time
from scipy.spatial import KDTree
import threading
from queue import Queue

@dataclass
class Node:
    """Individual force-field emitter node"""
    position: np.ndarray       # 3D position (m)
    stiffness: float = 1000.0  # Node stiffness (N/m)
    sigma: float = 0.05        # Interaction radius (m)
    max_force: float = 50.0    # Maximum force output (N)
    active: bool = True        # Node activation state
    power_level: float = 1.0   # Power scaling factor (0-1)
    material_type: str = "rigid"  # Material simulation type
    
    # Dynamic properties for material simulation
    damping: float = 10.0      # Damping coefficient (N⋅s/m)
    compliance: float = 1.0    # Compliance factor (m/N)

@dataclass
class GridParams:
    """Parameters for holodeck force-field grid"""
    # Spatial parameters
    bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = ((-2.0, 2.0), (-2.0, 2.0), (-1.0, 3.0))
    base_spacing: float = 0.08   # Base node spacing (m) - 8 cm
    fine_spacing: float = 0.02   # Fine grid spacing (m) - 2 cm for interaction zones
    adaptive_refinement: bool = True
    
    # Performance parameters  
    update_rate: float = 50e3    # Update frequency (Hz) - 50 kHz
    max_nodes: int = 50000       # Maximum number of nodes
    
    # Safety parameters
    global_force_limit: float = 100.0   # Global force limit (N)
    power_limit: float = 10e3           # Total power limit (W)
    emergency_stop_distance: float = 0.001  # Emergency stop if object too close (m)
    
    # Material simulation parameters
    default_materials: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "rigid": {"stiffness": 10000.0, "damping": 50.0, "compliance": 0.1},
        "soft": {"stiffness": 100.0, "damping": 20.0, "compliance": 10.0},
        "liquid": {"stiffness": 10.0, "damping": 100.0, "compliance": 100.0},
        "flesh": {"stiffness": 500.0, "damping": 30.0, "compliance": 2.0},
        "metal": {"stiffness": 50000.0, "damping": 100.0, "compliance": 0.01}
    })

class ForceFieldGrid:
    """
    Advanced holodeck force-field grid system
    
    Features:
    - Variable density node grid (5-10 cm base spacing, 2 cm fine)
    - High-bandwidth real-time updates (10-100 kHz)
    - Material property simulation
    - Safety interlocks and force limiting
    - Adaptive mesh refinement for interaction zones
    - Multi-threaded processing for real-time performance
    """
    
    def __init__(self, params: GridParams):
        """
        Initialize holodeck force-field grid
        
        Args:
            params: Grid configuration parameters
        """
        self.params = params
        self.nodes: List[Node] = []
        self.node_tree: Optional[KDTree] = None
        
        # Tracking and interaction state
        self.tracked_objects = {}  # object_id -> position, velocity
        self.interaction_zones = []  # List of high-detail zones
        self.active_nodes = set()    # Set of currently active node indices
        
        # Performance monitoring
        self.update_time_history = []
        self.force_computation_history = []
        self.last_update_time = 0.0
        
        # Threading for real-time updates
        self.update_thread = None
        self.update_queue = Queue()
        self.running = False
        
        # Safety systems
        self.emergency_stop = False
        self.total_power_usage = 0.0
        
        # Initialize base grid
        self._create_base_grid()
        self._build_spatial_index()
        
        logging.info(f"ForceFieldGrid initialized: {len(self.nodes)} nodes, "
                    f"update rate {params.update_rate/1000:.1f} kHz")

    def _create_base_grid(self):
        """Create the base uniform grid of force-field nodes"""
        x_min, x_max = self.params.bounds[0]
        y_min, y_max = self.params.bounds[1]  
        z_min, z_max = self.params.bounds[2]
        
        spacing = self.params.base_spacing
        
        # Generate grid coordinates
        x_coords = np.arange(x_min, x_max + spacing, spacing)
        y_coords = np.arange(y_min, y_max + spacing, spacing)
        z_coords = np.arange(z_min, z_max + spacing, spacing)
        
        # Create nodes at grid points
        for x in x_coords:
            for y in y_coords:
                for z in z_coords:
                    if len(self.nodes) >= self.params.max_nodes:
                        logging.warning(f"Reached maximum node limit: {self.params.max_nodes}")
                        return
                    
                    position = np.array([x, y, z])
                    node = Node(
                        position=position,
                        stiffness=self.params.default_materials["rigid"]["stiffness"],
                        sigma=spacing / 2,  # Interaction radius half of spacing
                        max_force=self.params.global_force_limit / 1000  # Distribute force limit
                    )
                    self.nodes.append(node)
        
        logging.info(f"Created base grid: {len(self.nodes)} nodes with {spacing:.3f}m spacing")

    def _build_spatial_index(self):
        """Build KDTree for fast spatial queries"""
        if not self.nodes:
            return
        
        positions = np.array([node.position for node in self.nodes])
        self.node_tree = KDTree(positions)
        logging.debug("Built spatial index for fast node lookup")

    def add_interaction_zone(self, center: np.ndarray, radius: float, 
                           material_type: str = "rigid"):
        """
        Add high-detail interaction zone with fine grid spacing
        
        Args:
            center: Center position of interaction zone
            radius: Radius of interaction zone
            material_type: Material type to simulate in this zone
        """
        # Remove any existing nodes in this zone
        nodes_to_remove = []
        for i, node in enumerate(self.nodes):
            if np.linalg.norm(node.position - center) < radius:
                nodes_to_remove.append(i)
        
        # Remove in reverse order to maintain indices
        for i in reversed(nodes_to_remove):
            del self.nodes[i]
        
        # Add fine grid nodes in this zone
        fine_spacing = self.params.fine_spacing
        material_props = self.params.default_materials.get(material_type, 
                                                          self.params.default_materials["rigid"])
        
        # Generate fine grid coordinates around center
        n_points = int(2 * radius / fine_spacing) + 1
        offsets = np.linspace(-radius, radius, n_points)
        
        nodes_added = 0
        for dx in offsets:
            for dy in offsets:
                for dz in offsets:
                    pos = center + np.array([dx, dy, dz])
                    
                    # Check if position is within zone radius
                    if np.linalg.norm(pos - center) <= radius:
                        # Check bounds
                        if (self.params.bounds[0][0] <= pos[0] <= self.params.bounds[0][1] and
                            self.params.bounds[1][0] <= pos[1] <= self.params.bounds[1][1] and
                            self.params.bounds[2][0] <= pos[2] <= self.params.bounds[2][1]):
                            
                            node = Node(
                                position=pos,
                                stiffness=material_props["stiffness"],
                                damping=material_props["damping"],
                                sigma=fine_spacing / 2,
                                material_type=material_type,
                                compliance=material_props["compliance"]
                            )
                            self.nodes.append(node)
                            nodes_added += 1
        
        # Rebuild spatial index
        self._build_spatial_index()
        
        # Record interaction zone
        self.interaction_zones.append({
            'center': center,
            'radius': radius,
            'material_type': material_type,
            'nodes_added': nodes_added
        })
        
        logging.info(f"Added interaction zone: {nodes_added} fine nodes, "
                    f"material={material_type}, spacing={fine_spacing:.3f}m")

    def compute_node_force(self, node: Node, target_point: np.ndarray, 
                          velocity: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute force from individual node
        
        F_i = -k_i(x - x_i) * exp(-||x - x_i||²/(2σ_i²)) - b_i * v
        
        Args:
            node: Force-field node
            target_point: Point where force is computed
            velocity: Optional velocity for damping
            
        Returns:
            3D force vector
        """
        if not node.active:
            return np.zeros(3)
        
        # Vector from node to target
        displacement = target_point - node.position
        distance = np.linalg.norm(displacement)
        
        # Avoid singularity at node position
        if distance < 1e-6:
            return np.zeros(3)
        
        # Gaussian weighting function
        weight = np.exp(-distance**2 / (2 * node.sigma**2))
        
        # Spring force: F = -k * displacement * weight
        spring_force = -node.stiffness * displacement * weight * node.power_level
        
        # Damping force if velocity provided
        damping_force = np.zeros(3)
        if velocity is not None:
            damping_force = -node.damping * velocity * weight * node.power_level
        
        total_force = spring_force + damping_force
        
        # Apply force limiting
        force_magnitude = np.linalg.norm(total_force)
        if force_magnitude > node.max_force:
            total_force = total_force * (node.max_force / force_magnitude)
        
        return total_force

    def compute_total_force(self, point: np.ndarray, 
                           velocity: Optional[np.ndarray] = None,
                           max_range: float = 0.5) -> np.ndarray:
        """
        Compute total force at point from all nearby nodes
        
        Args:
            point: 3D position where force is computed
            velocity: Optional velocity for damping calculations
            max_range: Maximum range for node interaction
            
        Returns:
            Total 3D force vector
        """
        if self.emergency_stop:
            return np.zeros(3)
        
        total_force = np.zeros(3)
        
        if self.node_tree is None:
            return total_force
        
        # Find nearby nodes efficiently using KDTree
        try:
            nearby_indices = self.node_tree.query_ball_point(point, max_range)
        except:
            nearby_indices = range(len(self.nodes))  # Fallback to all nodes
        
        # Compute force contributions
        active_nodes = 0
        for idx in nearby_indices:
            if idx < len(self.nodes):
                node = self.nodes[idx]
                force = self.compute_node_force(node, point, velocity)
                total_force += force
                
                if node.active:
                    active_nodes += 1
        
        # Apply global force limiting
        force_magnitude = np.linalg.norm(total_force)
        if force_magnitude > self.params.global_force_limit:
            total_force = total_force * (self.params.global_force_limit / force_magnitude)
        
        # Emergency stop if object too close to any node
        if self.node_tree:
            min_distance = min([np.linalg.norm(point - self.nodes[idx].position) 
                               for idx in nearby_indices if idx < len(self.nodes)])
            if min_distance < self.params.emergency_stop_distance:
                logging.warning(f"Emergency stop triggered: object at {min_distance:.4f}m from node")
                self.emergency_stop = True
                return np.zeros(3)
        
        return total_force

    def update_object_tracking(self, object_id: str, position: np.ndarray, 
                             velocity: Optional[np.ndarray] = None):
        """
        Update tracked object position and velocity
        
        Args:
            object_id: Unique identifier for tracked object
            position: Current 3D position
            velocity: Current 3D velocity (optional)
        """
        if velocity is None:
            velocity = np.zeros(3)
        
        # Update tracking data
        self.tracked_objects[object_id] = {
            'position': position.copy(),
            'velocity': velocity.copy(),
            'last_update': time.time()
        }
        
        # Adaptive mesh refinement around tracked objects
        if self.params.adaptive_refinement:
            self._update_adaptive_zones(object_id, position)

    def _update_adaptive_zones(self, object_id: str, position: np.ndarray):
        """Update adaptive mesh refinement zones around tracked objects"""
        # Check if we need to create/update interaction zone
        zone_radius = 0.2  # 20 cm interaction zone
        
        # Check if position is near existing interaction zones
        create_new_zone = True
        for zone in self.interaction_zones:
            if np.linalg.norm(position - zone['center']) < zone['radius'] + 0.1:
                create_new_zone = False
                break
        
        if create_new_zone:
            # Add new interaction zone with appropriate material
            material_type = "rigid"  # Default, could be determined by context
            self.add_interaction_zone(position, zone_radius, material_type)

    def step_simulation(self, dt: float) -> Dict:
        """
        Perform one simulation time step
        
        Args:
            dt: Time step size
            
        Returns:
            Step results and performance metrics
        """
        start_time = time.time()
        
        # Update all tracked objects
        total_forces = {}
        power_usage = 0.0
        
        for object_id, obj_data in self.tracked_objects.items():
            position = obj_data['position']
            velocity = obj_data['velocity']
            
            # Compute force at object position
            force = self.compute_total_force(position, velocity)
            total_forces[object_id] = force
            
            # Estimate power usage (simplified)
            force_magnitude = np.linalg.norm(force)
            power_usage += force_magnitude * np.linalg.norm(velocity)
        
        self.total_power_usage = power_usage
        
        # Performance metrics
        computation_time = time.time() - start_time
        self.update_time_history.append(computation_time)
        self.force_computation_history.append(len(total_forces))
        
        # Limit history size
        if len(self.update_time_history) > 1000:
            self.update_time_history = self.update_time_history[-500:]
            self.force_computation_history = self.force_computation_history[-500:]
        
        return {
            'total_forces': total_forces,
            'power_usage': power_usage,
            'computation_time': computation_time,
            'active_nodes': len([n for n in self.nodes if n.active]),
            'emergency_stop': self.emergency_stop
        }

    def start_realtime_updates(self):
        """Start real-time update thread"""
        if self.running:
            return
        
        self.running = True
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
        logging.info("Started real-time force-field updates")

    def stop_realtime_updates(self):
        """Stop real-time update thread"""
        self.running = False
        if self.update_thread:
            self.update_thread.join(timeout=1.0)
        logging.info("Stopped real-time force-field updates")

    def _update_loop(self):
        """Main update loop for real-time operation"""
        dt = 1.0 / self.params.update_rate
        
        while self.running:
            loop_start = time.time()
            
            # Process any queued updates
            while not self.update_queue.empty():
                try:
                    update_data = self.update_queue.get_nowait()
                    if update_data['type'] == 'object_position':
                        self.update_object_tracking(
                            update_data['object_id'],
                            update_data['position'],
                            update_data.get('velocity')
                        )
                except:
                    break
            
            # Perform simulation step
            self.step_simulation(dt)
            
            # Sleep for remaining time to maintain update rate
            elapsed = time.time() - loop_start
            if elapsed < dt:
                time.sleep(dt - elapsed)

    def get_performance_metrics(self) -> Dict:
        """Get system performance metrics"""
        if not self.update_time_history:
            return {}
        
        avg_update_time = np.mean(self.update_time_history)
        max_update_time = np.max(self.update_time_history)
        effective_rate = 1.0 / avg_update_time if avg_update_time > 0 else 0
        
        return {
            'average_update_time': avg_update_time,
            'maximum_update_time': max_update_time,
            'effective_update_rate': effective_rate,
            'target_update_rate': self.params.update_rate,
            'performance_ratio': effective_rate / self.params.update_rate,
            'total_nodes': len(self.nodes),
            'active_nodes': len([n for n in self.nodes if n.active]),
            'interaction_zones': len(self.interaction_zones),
            'tracked_objects': len(self.tracked_objects),
            'total_power_usage': self.total_power_usage,
            'emergency_stop': self.emergency_stop
        }

    def run_diagnostics(self) -> Dict:
        """Run comprehensive system diagnostics"""
        logging.info("Running holodeck force-field grid diagnostics")
        
        # Test basic force computation
        test_point = np.array([0.0, 0.0, 1.0])
        test_force = self.compute_total_force(test_point)
        
        # Test node activation
        active_nodes = sum(1 for node in self.nodes if node.active)
        node_coverage = active_nodes / len(self.nodes) if self.nodes else 0
        
        # Test spatial index
        spatial_index_ok = self.node_tree is not None
        
        # Test material properties
        materials_available = len(self.params.default_materials)
        
        # System health assessment
        diagnostics = {
            'force_computation': 'PASS' if np.all(np.isfinite(test_force)) else 'FAIL',
            'spatial_indexing': 'PASS' if spatial_index_ok else 'FAIL',
            'node_activation': 'PASS' if node_coverage > 0.9 else 'WARN',
            'material_simulation': 'PASS' if materials_available >= 3 else 'FAIL',
            
            'total_nodes': len(self.nodes),
            'active_nodes': active_nodes,
            'node_coverage': node_coverage,
            'materials_available': materials_available,
            'interaction_zones': len(self.interaction_zones),
            'update_rate': self.params.update_rate,
            'force_limit': self.params.global_force_limit,
            'power_limit': self.params.power_limit,
            
            'test_force_magnitude': np.linalg.norm(test_force),
            'emergency_systems': 'ACTIVE' if not self.emergency_stop else 'TRIGGERED'
        }
        
        # Overall health
        critical_systems = ['force_computation', 'spatial_indexing', 'material_simulation']
        all_critical_pass = all(diagnostics[sys] == 'PASS' for sys in critical_systems)
        diagnostics['overall_health'] = 'HEALTHY' if all_critical_pass else 'DEGRADED'
        
        logging.info(f"Diagnostics complete: {diagnostics['overall_health']}")
        
        return diagnostics

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Initialize holodeck grid
    params = GridParams(
        bounds=((-1.0, 1.0), (-1.0, 1.0), (0.0, 2.0)),
        base_spacing=0.1,
        update_rate=10e3  # 10 kHz for testing
    )
    
    grid = ForceFieldGrid(params)
    
    # Run diagnostics
    diag = grid.run_diagnostics()
    print("Holodeck Force-Field Grid Diagnostics:")
    for key, value in diag.items():
        print(f"  {key}: {value}")
    
    # Test force computation
    test_position = np.array([0.0, 0.0, 1.0])
    test_velocity = np.array([0.1, 0.0, 0.0])
    
    force = grid.compute_total_force(test_position, test_velocity)
    print(f"\nTest force at {test_position}: {force} N")
    print(f"Force magnitude: {np.linalg.norm(force):.3f} N")
    
    # Add interaction zone
    grid.add_interaction_zone(test_position, 0.2, "soft")
    
    # Test tracking
    grid.update_object_tracking("hand", test_position, test_velocity)
    
    # Performance test
    step_result = grid.step_simulation(0.0001)  # 0.1 ms step
    print(f"\nSimulation step results:")
    print(f"  Power usage: {step_result['power_usage']:.3f} W")
    print(f"  Computation time: {step_result['computation_time']*1000:.2f} ms")
    print(f"  Active nodes: {step_result['active_nodes']}")
