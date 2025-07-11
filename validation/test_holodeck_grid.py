"""
Test Suite for Holodeck Force-Field Grid
========================================

Comprehensive tests for tactile feedback simulation system.
"""

import numpy as np
import pytest
import time
import logging
from unittest.mock import Mock, patch

# Import the modules we're testing directly
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.holodeck_forcefield_grid.grid import ForceFieldGrid, GridParams, Node

logging.basicConfig(level=logging.INFO)

class TestForceFieldGrid:
    """Test suite for holodeck force-field grid system"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.params = GridParams(
            bounds=((-0.5, 0.5), (-0.5, 0.5), (0.0, 1.0)),
            base_spacing=0.1,
            update_rate=1000.0,  # 1 kHz for testing
            max_nodes=1000
        )
        self.grid = ForceFieldGrid(self.params)
    
    def test_grid_initialization(self):
        """Test proper grid initialization"""
        assert len(self.grid.nodes) > 0
        assert self.grid.node_tree is not None
        assert self.grid.params.update_rate == 1000.0
        assert not self.grid.emergency_stop
        
        # Check node distribution
        positions = np.array([node.position for node in self.grid.nodes])
        assert np.all(positions[:, 0] >= self.params.bounds[0][0])
        assert np.all(positions[:, 0] <= self.params.bounds[0][1])
        assert np.all(positions[:, 1] >= self.params.bounds[1][0])
        assert np.all(positions[:, 1] <= self.params.bounds[1][1])
        assert np.all(positions[:, 2] >= self.params.bounds[2][0])
        assert np.all(positions[:, 2] <= self.params.bounds[2][1])
    
    def test_node_force_computation(self):
        """Test individual node force computation"""
        if not self.grid.nodes:
            pytest.skip("No nodes available for testing")
        
        node = self.grid.nodes[0]
        test_point = node.position + np.array([0.05, 0.0, 0.0])  # 5 cm away
        test_velocity = np.array([0.1, 0.0, 0.0])
        
        # Test force computation
        force = self.grid.compute_node_force(node, test_point, test_velocity)
        
        assert isinstance(force, np.ndarray)
        assert force.shape == (3,)
        assert np.all(np.isfinite(force))
        
        # Force should be non-zero and reasonable magnitude
        force_magnitude = np.linalg.norm(force)
        assert force_magnitude > 0
        assert force_magnitude < node.max_force * 2  # Should be within reasonable bounds
    
    def test_total_force_computation(self):
        """Test total force computation from multiple nodes"""
        test_point = np.array([0.0, 0.0, 0.5])  # Center of grid
        test_velocity = np.array([0.0, 0.1, 0.0])
        
        # Compute total force
        total_force = self.grid.compute_total_force(test_point, test_velocity)
        
        assert isinstance(total_force, np.ndarray)
        assert total_force.shape == (3,)
        assert np.all(np.isfinite(total_force))
        
        # Force magnitude should be within global limits
        force_magnitude = np.linalg.norm(total_force)
        assert force_magnitude <= self.params.global_force_limit
    
    def test_interaction_zone_creation(self):
        """Test creation of high-detail interaction zones"""
        initial_node_count = len(self.grid.nodes)
        
        center = np.array([0.2, 0.1, 0.3])
        radius = 0.15
        material_type = "soft"
        
        # Add interaction zone
        self.grid.add_interaction_zone(center, radius, material_type)
        
        # Should have more nodes after adding interaction zone
        assert len(self.grid.nodes) > initial_node_count
        
        # Check that interaction zone was recorded
        assert len(self.grid.interaction_zones) == 1
        zone = self.grid.interaction_zones[0]
        assert np.allclose(zone['center'], center)
        assert zone['radius'] == radius
        assert zone['material_type'] == material_type
        
        # Check that nodes in the zone have appropriate material properties
        nodes_in_zone = 0
        for node in self.grid.nodes:
            if np.linalg.norm(node.position - center) <= radius:
                if node.material_type == material_type:
                    nodes_in_zone += 1
        
        assert nodes_in_zone > 0
    
    def test_object_tracking(self):
        """Test object position and velocity tracking"""
        object_id = "test_object"
        position = np.array([0.1, 0.2, 0.3])
        velocity = np.array([0.05, 0.0, -0.1])
        
        # Update object tracking
        self.grid.update_object_tracking(object_id, position, velocity)
        
        # Check that object is being tracked
        assert object_id in self.grid.tracked_objects
        tracked_data = self.grid.tracked_objects[object_id]
        assert np.allclose(tracked_data['position'], position)
        assert np.allclose(tracked_data['velocity'], velocity)
        assert tracked_data['last_update'] > 0
    
    def test_simulation_step(self):
        """Test single simulation time step"""
        # Add a tracked object
        object_id = "test_hand"
        position = np.array([0.0, 0.0, 0.5])
        velocity = np.array([0.02, 0.01, 0.0])
        
        self.grid.update_object_tracking(object_id, position, velocity)
        
        # Perform simulation step
        dt = 0.001  # 1 ms
        results = self.grid.step_simulation(dt)
        
        # Check results structure
        assert 'total_forces' in results
        assert 'power_usage' in results
        assert 'computation_time' in results
        assert 'active_nodes' in results
        assert 'emergency_stop' in results
        
        # Check force computation for tracked object
        assert object_id in results['total_forces']
        force = results['total_forces'][object_id]
        assert isinstance(force, np.ndarray)
        assert np.all(np.isfinite(force))
        
        # Check performance metrics
        assert results['computation_time'] > 0
        assert results['computation_time'] < 0.1  # Should be fast
        assert results['active_nodes'] > 0
        assert not results['emergency_stop']
    
    def test_material_properties(self):
        """Test different material property simulation"""
        materials = ["rigid", "soft", "liquid", "flesh", "metal"]
        
        for material in materials:
            # Add interaction zone with this material
            center = np.array([0.0, 0.0, 0.6])
            radius = 0.1
            
            # Clear existing zones
            self.grid.interaction_zones = []
            
            self.grid.add_interaction_zone(center, radius, material)
            
            # Find nodes with this material
            material_nodes = [node for node in self.grid.nodes 
                            if node.material_type == material]
            
            assert len(material_nodes) > 0
            
            # Check material properties are applied
            material_props = self.params.default_materials[material]
            for node in material_nodes[:5]:  # Check first few nodes
                # Material properties should be applied
                assert node.material_type == material
                # Specific property values should match defaults
                if hasattr(node, 'stiffness'):
                    assert abs(node.stiffness - material_props["stiffness"]) < 1e-6
    
    def test_safety_systems(self):
        """Test emergency stop and safety systems"""
        # Test emergency stop activation
        assert not self.grid.emergency_stop
        
        # Force emergency stop by placing object too close to node
        if self.grid.nodes:
            node_pos = self.grid.nodes[0].position
            too_close_pos = node_pos + np.array([0.0001, 0.0, 0.0])  # 0.1 mm away
            
            # This should trigger emergency stop
            force = self.grid.compute_total_force(too_close_pos)
            
            # Emergency stop should be triggered
            assert self.grid.emergency_stop
            
            # Further force computations should return zero
            force_after_stop = self.grid.compute_total_force(np.array([0.0, 0.0, 0.5]))
            assert np.allclose(force_after_stop, np.zeros(3))
    
    def test_performance_monitoring(self):
        """Test performance metrics collection"""
        # Perform multiple simulation steps
        for i in range(10):
            self.grid.step_simulation(0.001)
        
        # Check performance history
        assert len(self.grid.update_time_history) > 0
        assert len(self.grid.force_computation_history) > 0
        
        # Get performance metrics
        metrics = self.grid.get_performance_metrics()
        
        assert 'average_update_time' in metrics
        assert 'maximum_update_time' in metrics
        assert 'effective_update_rate' in metrics
        assert 'target_update_rate' in metrics
        assert 'performance_ratio' in metrics
        assert 'total_nodes' in metrics
        assert 'active_nodes' in metrics
        
        # Check reasonable values
        assert metrics['average_update_time'] > 0
        assert metrics['effective_update_rate'] > 0
        assert metrics['total_nodes'] == len(self.grid.nodes)
        assert metrics['target_update_rate'] == self.params.update_rate
    
    def test_adaptive_refinement(self):
        """Test adaptive mesh refinement around tracked objects"""
        # Enable adaptive refinement
        self.grid.params.adaptive_refinement = True
        
        initial_zones = len(self.grid.interaction_zones)
        
        # Add tracked object in new location
        object_id = "adaptive_test"
        position = np.array([0.3, 0.3, 0.7])  # Away from existing zones
        
        self.grid.update_object_tracking(object_id, position)
        
        # Should have created new interaction zone
        assert len(self.grid.interaction_zones) > initial_zones
        
        # Check that new zone is near the tracked object
        new_zone = self.grid.interaction_zones[-1]
        distance_to_object = np.linalg.norm(new_zone['center'] - position)
        assert distance_to_object < 0.3  # Should be reasonably close
    
    def test_force_limiting(self):
        """Test force limiting and safety constraints"""
        # Create a scenario that would generate high forces
        test_point = np.array([0.0, 0.0, 0.5])
        
        # Temporarily increase all node powers to test limiting
        original_powers = []
        for node in self.grid.nodes:
            original_powers.append(node.stiffness)
            node.stiffness = 1e6  # Very high stiffness
        
        try:
            force = self.grid.compute_total_force(test_point)
            force_magnitude = np.linalg.norm(force)
            
            # Force should be limited to global limit
            assert force_magnitude <= self.params.global_force_limit
            
        finally:
            # Restore original powers
            for i, node in enumerate(self.grid.nodes):
                node.stiffness = original_powers[i]
    
    def test_diagnostics(self):
        """Test comprehensive diagnostics system"""
        diagnostics = self.grid.run_diagnostics()
        
        # Check required diagnostic fields
        required_fields = [
            'force_computation', 'spatial_indexing', 'node_activation',
            'material_simulation', 'total_nodes', 'active_nodes',
            'overall_health', 'emergency_systems'
        ]
        
        for field in required_fields:
            assert field in diagnostics
        
        # Check diagnostic values
        assert diagnostics['total_nodes'] == len(self.grid.nodes)
        assert diagnostics['active_nodes'] >= 0
        assert diagnostics['overall_health'] in ['HEALTHY', 'DEGRADED']
        assert diagnostics['emergency_systems'] in ['ACTIVE', 'TRIGGERED']
        
        # Force computation should pass
        assert diagnostics['force_computation'] == 'PASS'
        
        # Spatial indexing should pass
        assert diagnostics['spatial_indexing'] == 'PASS'

def test_grid_creation_parameters():
    """Test grid creation with different parameters"""
    # Test small grid
    small_params = GridParams(
        bounds=((-0.1, 0.1), (-0.1, 0.1), (0.0, 0.2)),
        base_spacing=0.05,
        max_nodes=50
    )
    small_grid = ForceFieldGrid(small_params)
    assert len(small_grid.nodes) <= 50
    assert len(small_grid.nodes) > 0
    
    # Test large spacing
    large_spacing_params = GridParams(
        bounds=((-1.0, 1.0), (-1.0, 1.0), (0.0, 2.0)),
        base_spacing=0.5,
        max_nodes=100
    )
    large_grid = ForceFieldGrid(large_spacing_params)
    assert len(large_grid.nodes) < 100  # Should be fewer nodes with large spacing

def test_material_force_differences():
    """Test that different materials produce different force characteristics"""
    params = GridParams(
        bounds=((-0.2, 0.2), (-0.2, 0.2), (0.0, 0.4)),
        base_spacing=0.1
    )
    
    # Create two grids with different materials
    grid1 = ForceFieldGrid(params)
    grid2 = ForceFieldGrid(params)
    
    # Add interaction zones with different materials
    center = np.array([0.0, 0.0, 0.2])
    radius = 0.15
    
    grid1.add_interaction_zone(center, radius, "rigid")
    grid2.add_interaction_zone(center, radius, "soft")
    
    # Test force at same position
    test_point = np.array([0.05, 0.0, 0.2])
    
    force1 = grid1.compute_total_force(test_point)
    force2 = grid2.compute_total_force(test_point)
    
    # Forces should be different (rigid vs soft material)
    force_diff = np.linalg.norm(force1 - force2)
    assert force_diff > 1e-6  # Should have measurable difference

if __name__ == "__main__":
    # Run tests directly
    import pytest
    pytest.main([__file__, "-v"])
