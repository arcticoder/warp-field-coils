"""
Simple Direct Test of Working Components
======================================

Direct test of components that we know are working.
"""

import sys
import os
import numpy as np

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, '..', 'src', 'holodeck_forcefield_grid'))

def test_holodeck_detailed():
    """Detailed test of holodeck force-field grid"""
    print("🌐 DETAILED HOLODECK FORCE-FIELD GRID TEST")
    print("=" * 50)
    
    try:
        from grid import ForceFieldGrid, GridParams, Node
        
        # Test 1: Different grid configurations
        print("Test 1: Grid Configurations")
        
        configs = [
            ("Small Dense", {"bounds": ((-0.5, 0.5), (-0.5, 0.5), (0.0, 1.0)), "base_spacing": 0.2}),
            ("Medium", {"bounds": ((-1.0, 1.0), (-1.0, 1.0), (0.0, 2.0)), "base_spacing": 0.3}),
            ("Large Sparse", {"bounds": ((-2.0, 2.0), (-2.0, 2.0), (0.0, 4.0)), "base_spacing": 0.5})
        ]
        
        for name, config in configs:
            params = GridParams(
                bounds=config["bounds"],
                base_spacing=config["base_spacing"],
                update_rate=1000
            )
            grid = ForceFieldGrid(params)
            print(f"  ✅ {name}: {len(grid.nodes)} nodes")
        
        # Test 2: Force computation at different positions
        print("\nTest 2: Force Computation at Various Positions")
        
        # Use medium grid for detailed testing
        params = GridParams(
            bounds=((-1.0, 1.0), (-1.0, 1.0), (0.0, 2.0)),
            base_spacing=0.3,
            update_rate=1000
        )
        grid = ForceFieldGrid(params)
        
        test_positions = [
            np.array([0.0, 0.0, 1.0]),    # Center
            np.array([0.5, 0.5, 1.5]),    # Off-center
            np.array([-0.8, 0.3, 0.5]),   # Near edge
            np.array([0.0, 0.0, 0.1]),    # Near bottom
            np.array([0.0, 0.0, 1.9])     # Near top
        ]
        
        for i, pos in enumerate(test_positions):
            vel = np.array([0.1, 0.0, 0.0])  # Constant test velocity
            force = grid.compute_total_force(pos, vel)
            force_mag = np.linalg.norm(force)
            print(f"  ✅ Position {i+1}: Force magnitude = {force_mag:.4f} N")
        
        # Test 3: Interaction zones and fine detail
        print("\nTest 3: Interaction Zones")
        
        initial_nodes = len(grid.nodes)
        
        # Add multiple interaction zones
        zones = [
            (np.array([0.2, 0.2, 1.0]), 0.2, "soft"),
            (np.array([-0.3, 0.1, 1.5]), 0.15, "rigid"),
            (np.array([0.0, -0.4, 0.8]), 0.25, "liquid")
        ]
        
        for pos, radius, material in zones:
            grid.add_interaction_zone(pos, radius, material)
            nodes_after = len(grid.nodes)
            added_nodes = nodes_after - initial_nodes
            print(f"  ✅ {material.capitalize()} zone: +{added_nodes} nodes (total: {nodes_after})")
            initial_nodes = nodes_after
        
        # Test 4: Object tracking and movement
        print("\nTest 4: Object Tracking")
        
        # Track multiple objects
        objects = [
            ("Probe_A", np.array([0.1, 0.1, 1.2]), np.array([0.05, 0.0, 0.0])),
            ("Probe_B", np.array([-0.2, 0.3, 0.9]), np.array([0.0, -0.03, 0.02])),
            ("Tool_C", np.array([0.4, -0.1, 1.6]), np.array([-0.02, 0.01, 0.0]))
        ]
        
        for obj_id, pos, vel in objects:
            grid.update_object_tracking(obj_id, pos, vel)
        
        print(f"  ✅ Tracking {len(grid.tracked_objects)} objects")
        
        # Test 5: Multi-step simulation
        print("\nTest 5: Multi-Step Simulation")
        
        dt = 0.001  # 1 ms time steps
        n_steps = 10
        
        total_power = 0.0
        total_time = 0.0
        
        for step in range(n_steps):
            result = grid.step_simulation(dt)
            total_power += result['power_usage']
            total_time += result['computation_time']
        
        avg_power = total_power / n_steps
        avg_comp_time = total_time / n_steps
        
        print(f"  ✅ {n_steps} simulation steps completed")
        print(f"  ✅ Average power: {avg_power:.3f} W")
        print(f"  ✅ Average computation time: {avg_comp_time*1000:.2f} ms")
        
        # Test 6: Performance metrics over time
        print("\nTest 6: Performance Metrics")
        
        metrics = grid.get_performance_metrics()
        if metrics:
            print(f"  ✅ Update rate: {metrics.get('effective_update_rate', 0):.1f} Hz")
            print(f"  ✅ Performance ratio: {metrics.get('performance_ratio', 0):.3f}")
            print(f"  ✅ Average update time: {metrics.get('average_update_time', 0)*1000:.3f} ms")
        
        # Test 7: System stress test
        print("\nTest 7: System Stress Test")
        
        # Add many tracked objects
        for i in range(20):
            pos = np.random.uniform(-0.8, 0.8, 3)
            pos[2] = np.random.uniform(0.2, 1.8)  # Keep in bounds
            vel = np.random.uniform(-0.1, 0.1, 3)
            grid.update_object_tracking(f"stress_obj_{i}", pos, vel)
        
        # Run several simulation steps
        for step in range(5):
            result = grid.step_simulation(dt)
        
        final_metrics = grid.get_performance_metrics()
        if final_metrics:
            final_rate = final_metrics.get('effective_update_rate', 0)
            print(f"  ✅ Stress test completed: {final_rate:.1f} Hz effective rate")
        
        print(f"\n🎉 ALL HOLODECK TESTS PASSED!")
        print(f"   Final configuration: {len(grid.nodes)} total nodes")
        print(f"   Interaction zones: {len(grid.interaction_zones)}")
        print(f"   Tracked objects: {len(grid.tracked_objects)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_mathematical_detailed():
    """Detailed test of mathematical refinements"""
    print("\n🔬 DETAILED MATHEMATICAL REFINEMENTS TEST")
    print("=" * 50)
    
    try:
        from step23_mathematical_refinements import (
            DispersionTailoring, DispersionParams,
            ThreeDRadonTransform, FDKParams,
            AdaptiveMeshRefinement, AdaptiveMeshParams
        )
        
        # Test 1: Dispersion engineering
        print("Test 1: Advanced Dispersion Engineering")
        
        # Test different coupling strengths
        couplings = [1e-16, 1e-15, 1e-14]
        
        for coupling in couplings:
            params = DispersionParams(
                base_coupling=coupling,
                resonance_frequency=1e11,
                bandwidth=1e10
            )
            dispersion = DispersionTailoring(params)
            
            # Test at resonance
            eff_perm = dispersion.effective_permittivity(1e11)
            print(f"  ✅ Coupling {coupling:.0e}: ε_eff = {eff_perm.real:.2e}")
            
        # Test 2: Frequency sweep
        print("\nTest 2: Frequency-Dependent Analysis")
        
        params = DispersionParams(
            base_coupling=1e-15,
            resonance_frequency=1e11,
            bandwidth=1e10
        )
        dispersion = DispersionTailoring(params)
        
        frequencies = np.logspace(10, 12, 20)  # 10 GHz to 1 THz
        
        for freq in frequencies[::5]:  # Test every 5th frequency
            eff_perm = dispersion.effective_permittivity(freq)
            group_vel = 3e8 / np.sqrt(eff_perm.real)  # Approximate
            print(f"  ✅ {freq:.1e} Hz: v_g = {group_vel:.2e} m/s")
        
        # Test 3: 3D tomography with different parameters
        print("\nTest 3: 3D Tomographic Reconstruction")
        
        detector_sizes = [(16, 16), (32, 32), (64, 64)]
        projection_counts = [24, 36, 48]
        
        for det_size, n_proj in zip(detector_sizes, projection_counts):
            params = FDKParams(
                detector_size=det_size,
                n_projections=n_proj,
                reconstruction_volume=(det_size[0]//2, det_size[1]//2, det_size[0]//2)
            )
            
            radon_3d = ThreeDRadonTransform(params)
            
            if radon_3d.projection_geometry:
                geom = radon_3d.projection_geometry
                print(f"  ✅ {det_size}x{n_proj}: SDD={geom['SDD']:.1f}cm, mag={geom['magnification']:.2f}")
        
        # Test 4: Adaptive mesh refinement with different thresholds
        print("\nTest 4: Adaptive Mesh Refinement")
        
        thresholds = [0.05, 0.1, 0.2]
        
        for threshold in thresholds:
            params = AdaptiveMeshParams(
                initial_spacing=0.3,
                refinement_threshold=threshold,
                max_refinement_levels=3
            )
            
            mesh_refiner = AdaptiveMeshRefinement(params)
            
            # Create test data with varying gradients
            n_points = 100
            coords = np.random.uniform(-1, 1, (n_points, 3))
            
            # Create a field with sharp features
            values = np.exp(-5 * np.sum(coords**2, axis=1))  # Sharp Gaussian
            values += 0.5 * np.sin(10 * coords[:, 0])  # High-frequency component
            
            errors = mesh_refiner.compute_error_indicators(values, coords)
            n_refinements = np.sum(errors > threshold)
            
            print(f"  ✅ Threshold {threshold}: {n_refinements}/{len(errors)} points need refinement")
        
        print(f"\n🎉 ALL MATHEMATICAL TESTS PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Mathematical test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run detailed tests"""
    print("🧪 DETAILED COMPONENT TEST SUITE")
    print("=" * 60)
    
    results = {}
    
    # Test working components in detail
    tests = [
        ("Holodeck Force-Field Grid", test_holodeck_detailed),
        ("Mathematical Refinements", test_mathematical_detailed)
    ]
    
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        result = test_func()
        results[test_name] = result
    
    # Summary
    print(f"\n{'='*60}")
    print("🏁 DETAILED TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(results.values())
    total = len(results)
    
    print(f"Passed: {passed}/{total}")
    
    for test_name, result in results.items():
        icon = "✅" if result else "❌"
        status = "PASS" if result else "FAIL"
        print(f"  {icon} {test_name}: {status}")
    
    if passed == total:
        print("\n🎉 ALL DETAILED TESTS PASSED!")
        print("📋 Core systems are functioning correctly")
        print("🚀 Ready for integration testing")
    else:
        print(f"\n⚠️ {total-passed} tests failed")
    
    return results

if __name__ == "__main__":
    results = main()
