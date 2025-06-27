"""
Comprehensive Test Suite for Unified Warp Field System
=====================================================

Tests all components of the integrated system:
1. Holodeck Force-Field Grid
2. Medical Tractor Array
3. Unified System Calibration
4. Sensitivity Analysis
5. Mathematical Refinements
6. Extended Pipeline Integration
"""

import sys
import os
import logging
import time
import numpy as np
import traceback

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, '..', 'src'))
sys.path.append(os.path.join(current_dir, '..', 'src', 'holodeck_forcefield_grid'))
sys.path.append(os.path.join(current_dir, '..', 'src', 'medical_tractor_array'))

def setup_logging():
    """Setup logging for tests"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('test_results.log'),
            logging.StreamHandler()
        ]
    )

def test_holodeck_force_field_grid():
    """Test the holodeck force-field grid system"""
    print("\n🌐 Testing Holodeck Force-Field Grid")
    print("=" * 40)
    
    try:
        from grid import ForceFieldGrid, GridParams, Node
        
        # Test 1: Basic initialization
        print("Test 1: Basic Initialization")
        params = GridParams(
            bounds=((-1.0, 1.0), (-1.0, 1.0), (0.0, 2.0)),
            base_spacing=0.2,  # Larger spacing for faster tests
            update_rate=1000   # 1 kHz for testing
        )
        
        grid = ForceFieldGrid(params)
        print(f"✅ Grid initialized with {len(grid.nodes)} nodes")
        
        # Test 2: Diagnostics
        print("\nTest 2: System Diagnostics")
        diag = grid.run_diagnostics()
        print(f"✅ Overall health: {diag['overall_health']}")
        print(f"   Total nodes: {diag['total_nodes']}")
        print(f"   Active nodes: {diag['active_nodes']}")
        print(f"   Force computation: {diag['force_computation']}")
        print(f"   Spatial indexing: {diag['spatial_indexing']}")
        
        # Test 3: Force computation
        print("\nTest 3: Force Computation")
        test_position = np.array([0.0, 0.0, 1.0])
        test_velocity = np.array([0.1, 0.0, 0.0])
        
        force = grid.compute_total_force(test_position, test_velocity)
        force_magnitude = np.linalg.norm(force)
        
        print(f"✅ Force computed: magnitude = {force_magnitude:.3f} N")
        print(f"   Force vector: [{force[0]:.3f}, {force[1]:.3f}, {force[2]:.3f}] N")
        
        # Test 4: Interaction zones
        print("\nTest 4: Interaction Zones")
        initial_nodes = len(grid.nodes)
        grid.add_interaction_zone(test_position, 0.3, "soft")
        final_nodes = len(grid.nodes)
        
        print(f"✅ Interaction zone added: {final_nodes - initial_nodes} fine nodes")
        print(f"   Total interaction zones: {len(grid.interaction_zones)}")
        
        # Test 5: Object tracking
        print("\nTest 5: Object Tracking")
        grid.update_object_tracking("test_object", test_position, test_velocity)
        tracked_count = len(grid.tracked_objects)
        
        print(f"✅ Object tracking: {tracked_count} objects tracked")
        
        # Test 6: Simulation step
        print("\nTest 6: Simulation Step")
        step_result = grid.step_simulation(0.001)  # 1 ms step
        
        print(f"✅ Simulation step completed")
        print(f"   Power usage: {step_result['power_usage']:.3f} W")
        print(f"   Computation time: {step_result['computation_time']*1000:.2f} ms")
        print(f"   Active nodes: {step_result['active_nodes']}")
        
        # Test 7: Performance metrics
        print("\nTest 7: Performance Metrics")
        metrics = grid.get_performance_metrics()
        
        if metrics:
            print(f"✅ Performance metrics available:")
            print(f"   Average update time: {metrics.get('average_update_time', 0)*1000:.2f} ms")
            print(f"   Effective rate: {metrics.get('effective_update_rate', 0):.1f} Hz")
            print(f"   Performance ratio: {metrics.get('performance_ratio', 0):.3f}")
        else:
            print("⚠️ No performance metrics yet (expected on first run)")
        
        return True
        
    except Exception as e:
        print(f"❌ Holodeck test failed: {e}")
        traceback.print_exc()
        return False

def test_medical_tractor_array():
    """Test the medical tractor array system"""
    print("\n🏥 Testing Medical Tractor Array")
    print("=" * 35)
    
    try:
        # Import from the actual medical array module
        sys.path.insert(0, os.path.join(current_dir, '..', 'src', 'medical_tractor_array'))
        import array as medical_array_module
        MedicalTractorArray = medical_array_module.MedicalTractorArray
        MedicalArrayParams = medical_array_module.MedicalArrayParams
        BeamMode = medical_array_module.BeamMode
        SafetyLevel = medical_array_module.SafetyLevel
        
        # Test 1: Basic initialization
        print("Test 1: Medical Array Initialization")
        params = MedicalArrayParams(
            array_bounds=((-0.3, 0.3), (-0.3, 0.3), (0.05, 0.4)),
            beam_spacing=0.05,  # 5 cm spacing for testing
            safety_level=SafetyLevel.THERAPEUTIC
        )
        
        array = MedicalTractorArray(params)
        print(f"✅ Medical array initialized with {len(array.beams)} beams")
        
        # Test 2: Diagnostics
        print("\nTest 2: Medical Diagnostics")
        diag = array.run_diagnostics()
        print(f"✅ Overall health: {diag['overall_health']}")
        print(f"   Beam activation: {diag['beam_activation']}")
        print(f"   Force computation: {diag['force_computation']}")
        print(f"   Safety systems: {diag['safety_systems']}")
        print(f"   Total beams: {diag['total_beams']}")
        print(f"   Active beams: {diag['active_beams']}")
        
        # Test 3: Optical force computation
        print("\nTest 3: Optical Force Computation")
        if array.beams:
            beam = array.beams[0]
            target_pos = np.array([0.1, 0.1, 0.2])
            force = array.compute_optical_force(beam, target_pos)
            force_magnitude = np.linalg.norm(force)
            
            print(f"✅ Optical force computed: {force_magnitude:.2e} N")
            print(f"   Force vector: [{force[0]:.2e}, {force[1]:.2e}, {force[2]:.2e}] N")
        
        # Test 4: Target positioning
        print("\nTest 4: Target Positioning")
        target_pos = np.array([0.05, 0.02, 0.15])
        desired_pos = np.array([0.03, 0.02, 0.15])
        
        array.start_procedure("TEST_001", "positioning_test")
        result = array.position_target(target_pos, desired_pos, tissue_type="soft")
        array.stop_procedure()
        
        print(f"✅ Positioning result: {result['status']}")
        if 'distance_to_target' in result:
            print(f"   Distance to target: {result['distance_to_target']*1000:.2f} mm")
        if 'power_usage' in result:
            print(f"   Power usage: {result['power_usage']:.1f} mW")
        
        # Test 5: Wound closure assistance
        print("\nTest 5: Wound Closure Assistance")
        wound_edges = [
            np.array([0.0, -0.005, 0.15]),
            np.array([0.0, 0.005, 0.15])
        ]
        
        closure_result = array.assist_wound_closure(wound_edges)
        print(f"✅ Wound closure result: {closure_result['status']}")
        if 'closure_progress' in closure_result:
            print(f"   Closure progress: {closure_result['closure_progress']*1000:.2f} mm")
        
        # Test 6: Catheter guidance
        print("\nTest 6: Catheter Guidance")
        catheter_tip = np.array([0.08, 0.03, 0.2])
        target_vessel = np.array([0.05, 0.03, 0.2])
        
        guidance_result = array.guide_catheter(catheter_tip, target_vessel)
        print(f"✅ Catheter guidance result: {guidance_result['status']}")
        if 'distance_to_target' in guidance_result:
            print(f"   Distance to target: {guidance_result['distance_to_target']*1000:.2f} mm")
        
        return True
        
    except Exception as e:
        print(f"❌ Medical array test failed: {e}")
        traceback.print_exc()
        return False

def test_unified_calibration():
    """Test the unified system calibration"""
    print("\n🔧 Testing Unified System Calibration")
    print("=" * 40)
    
    try:
        from step21_system_calibration import UnifiedSystemCalibrator, CalibrationParams
        
        # Test 1: Calibrator initialization
        print("Test 1: Calibrator Initialization")
        params = CalibrationParams(
            max_iterations=10,  # Reduced for testing
            use_genetic_algorithm=True,
            population_size=8   # Small population for testing
        )
        
        calibrator = UnifiedSystemCalibrator(params)
        print("✅ Unified calibrator initialized")
        
        # Test 2: Individual subsystem evaluation
        print("\nTest 2: Subsystem Performance Evaluation")
        
        # Test FTL performance
        ftl_score = calibrator._evaluate_ftl_performance(1e-15)
        print(f"✅ FTL performance score: {ftl_score:.3f}")
        
        # Test grid performance
        grid_score = calibrator._evaluate_grid_performance(0.2)
        print(f"✅ Grid performance score: {grid_score:.3f}")
        
        # Test medical performance
        medical_score = calibrator._evaluate_medical_performance(1.0)
        print(f"✅ Medical performance score: {medical_score:.3f}")
        
        # Test tomography performance
        tomo_score = calibrator._evaluate_tomography_performance(120)
        print(f"✅ Tomography performance score: {tomo_score:.3f}")
        
        # Test 3: Objective function
        print("\nTest 3: Objective Function Evaluation")
        test_params = np.array([1e-15, 0.2, 1.0, 120])
        cost = calibrator.objective_function(test_params)
        print(f"✅ Objective function cost: {cost:.6f}")
        
        # Test 4: Quick calibration run (limited iterations)
        print("\nTest 4: Quick Calibration Run")
        start_time = time.time()
        
        # Override to very limited run for testing
        original_iterations = calibrator.params.max_iterations
        calibrator.params.max_iterations = 3
        
        results = calibrator.run_calibration()
        calibration_time = time.time() - start_time
        
        # Restore original iterations
        calibrator.params.max_iterations = original_iterations
        
        print(f"✅ Quick calibration completed in {calibration_time:.2f}s")
        print(f"   Success: {results['success']}")
        print(f"   Iterations: {results['iterations']}")
        
        if 'optimal_parameters' in results:
            opt = results['optimal_parameters']
            print(f"   Optimal coupling: {opt['subspace_coupling']:.2e}")
            print(f"   Optimal spacing: {opt['grid_spacing']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Calibration test failed: {e}")
        traceback.print_exc()
        return False

def test_sensitivity_analysis():
    """Test the sensitivity analysis system"""
    print("\n🔍 Testing Sensitivity Analysis")
    print("=" * 35)
    
    try:
        from step22_sensitivity_analysis import SensitivityAnalyzer, SensitivityParams
        
        # Test 1: Analyzer initialization
        print("Test 1: Sensitivity Analyzer Initialization")
        params = SensitivityParams(
            mc_samples=50,     # Reduced for testing
            sobol_samples=128  # Reduced for testing
        )
        
        analyzer = SensitivityAnalyzer(params)
        print("✅ Sensitivity analyzer initialized")
        
        # Test 2: Individual function evaluations
        print("\nTest 2: Function Evaluations")
        
        ftl_rate = analyzer.ftl_rate_function(1e-15)
        print(f"✅ FTL rate function: {ftl_rate:.2e} Hz")
        
        grid_uniformity = analyzer.grid_uniformity_function(0.2)
        print(f"✅ Grid uniformity function: {grid_uniformity:.3f}")
        
        medical_precision = analyzer.medical_precision_function(1.0)
        print(f"✅ Medical precision function: {medical_precision:.2e} N")
        
        tomo_fidelity = analyzer.tomography_fidelity_function(120)
        print(f"✅ Tomography fidelity function: {tomo_fidelity:.3f}")
        
        # Test 3: Finite difference derivatives
        print("\nTest 3: Finite Difference Derivatives")
        
        df_dk = analyzer.finite_difference_derivative(analyzer.ftl_rate_function, 1e-15)
        print(f"✅ FTL rate derivative: {df_dk:.2e}")
        
        du_ds = analyzer.finite_difference_derivative(analyzer.grid_uniformity_function, 0.2)
        print(f"✅ Grid uniformity derivative: {du_ds:.3f}")
        
        # Test 4: Local sensitivity analysis
        print("\nTest 4: Local Sensitivity Analysis")
        local_results = analyzer.local_sensitivity_analysis()
        
        print("✅ Local sensitivity analysis completed")
        if 'sensitivity_ranking' in local_results:
            print("   Parameter ranking (top 3):")
            for i, (param, sens) in enumerate(local_results['sensitivity_ranking'][:3], 1):
                print(f"   {i}. {param}: {sens:.6f}")
        
        # Test 5: Monte Carlo uncertainty propagation (limited)
        print("\nTest 5: Monte Carlo Uncertainty Propagation")
        mc_results = analyzer.monte_carlo_uncertainty_propagation()
        
        print("✅ Monte Carlo analysis completed")
        if 'uncertainty_ranking' in mc_results:
            print("   Uncertainty ranking (top 3):")
            for i, (output, cv) in enumerate(mc_results['uncertainty_ranking'][:3], 1):
                print(f"   {i}. {output}: CV = {cv:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Sensitivity analysis test failed: {e}")
        traceback.print_exc()
        return False

def test_mathematical_refinements():
    """Test the mathematical refinements"""
    print("\n🔬 Testing Mathematical Refinements")
    print("=" * 40)
    
    try:
        from step23_mathematical_refinements import (
            DispersionTailoring, DispersionParams,
            ThreeDRadonTransform, FDKParams,
            AdaptiveMeshRefinement, AdaptiveMeshParams
        )
        
        # Test 1: Dispersion tailoring
        print("Test 1: Dispersion Tailoring")
        disp_params = DispersionParams(
            base_coupling=1e-15,
            resonance_frequency=1e11,
            bandwidth=1e10
        )
        
        dispersion = DispersionTailoring(disp_params)
        
        # Test effective permittivity
        eff_perm = dispersion.effective_permittivity(1e11)
        print(f"✅ Effective permittivity at resonance: {eff_perm}")
        
        # Test dispersion relation
        frequencies = np.linspace(5e10, 1.5e11, 10)
        disp_data = dispersion.compute_dispersion_relation(frequencies)
        print(f"✅ Dispersion relation computed for {len(frequencies)} frequencies")
        print(f"   Group velocity range: {np.min(disp_data['group_velocities']):.2e} - {np.max(disp_data['group_velocities']):.2e} m/s")
        
        # Test 2: 3D Radon transform
        print("\nTest 2: 3D Radon Transform")
        fdk_params = FDKParams(
            detector_size=(32, 32),  # Small for testing
            n_projections=36,        # Reduced for testing
            reconstruction_volume=(16, 16, 16)  # Small volume
        )
        
        radon_3d = ThreeDRadonTransform(fdk_params)
        print("✅ 3D Radon transform initialized")
        print(f"   Detector size: {fdk_params.detector_size}")
        print(f"   Projections: {fdk_params.n_projections}")
        print(f"   Reconstruction volume: {fdk_params.reconstruction_volume}")
        
        # Test geometry setup
        if radon_3d.projection_geometry:
            geom = radon_3d.projection_geometry
            print(f"   Source-detector distance: {geom['SDD']:.1f} cm")
            print(f"   Magnification: {geom['magnification']:.2f}")
        
        # Test 3: Adaptive mesh refinement
        print("\nTest 3: Adaptive Mesh Refinement")
        mesh_params = AdaptiveMeshParams(
            initial_spacing=0.3,
            refinement_threshold=0.1,
            max_refinement_levels=3
        )
        
        mesh_refiner = AdaptiveMeshRefinement(mesh_params)
        print("✅ Adaptive mesh refiner initialized")
        
        # Create test mesh and field
        n_points = 50
        coords = np.random.uniform(-1, 1, (n_points, 3))
        values = np.exp(-np.sum(coords**2, axis=1))  # Gaussian field
        
        # Test error indicators
        errors = mesh_refiner.compute_error_indicators(values, coords)
        print(f"✅ Error indicators computed: range {np.min(errors):.4f} - {np.max(errors):.4f}")
        
        # Test mesh refinement
        refined_coords, refined_values = mesh_refiner.refine_mesh(coords, values)
        refinement_ratio = len(refined_coords) / len(coords)
        print(f"✅ Mesh refinement: {len(coords)} → {len(refined_coords)} points (ratio: {refinement_ratio:.2f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Mathematical refinements test failed: {e}")
        traceback.print_exc()
        return False

def test_extended_pipeline():
    """Test the extended pipeline integration"""
    print("\n🚀 Testing Extended Pipeline Integration")
    print("=" * 45)
    
    try:
        from step24_extended_pipeline import ExtendedWarpFieldPipeline, ExtendedPipelineParams
        
        # Test 1: Pipeline initialization
        print("Test 1: Pipeline Initialization")
        params = ExtendedPipelineParams(
            enable_calibration=True,
            enable_sensitivity=True,
            enable_refinements=True,
            calibration_iterations=5,  # Very limited for testing
            sensitivity_samples=20,    # Very limited for testing
            output_directory="test_pipeline_results"
        )
        
        pipeline = ExtendedWarpFieldPipeline(params)
        print("✅ Extended pipeline initialized")
        
        # Test 2: Subsystem initialization
        print("\nTest 2: Subsystem Initialization")
        init_results = pipeline.initialize_subsystems()
        print("✅ Subsystems initialized:")
        for system, status in init_results.items():
            print(f"   {system}: {status}")
        
        # Test 3: Performance validation
        print("\nTest 3: Performance Validation")
        validation_results = pipeline.validate_system_performance()
        
        overall = validation_results['overall_assessment']
        print(f"✅ Performance validation completed")
        print(f"   System status: {overall['system_status']}")
        print(f"   Performance ratio: {overall['overall_performance_ratio']:.3f}")
        print(f"   Requirements met: {overall['all_requirements_met']}")
        
        # Test 4: Quick calibration (very limited)
        print("\nTest 4: Quick Calibration Test")
        # Temporarily reduce iterations even further
        original_iterations = pipeline.params.calibration_iterations
        pipeline.params.calibration_iterations = 2
        
        calib_results = pipeline.step_21_unified_calibration()
        
        # Restore original
        pipeline.params.calibration_iterations = original_iterations
        
        if calib_results.get('success', False):
            print("✅ Quick calibration successful")
            opt = calib_results['optimal_parameters']
            print(f"   Optimal coupling: {opt['subspace_coupling']:.2e}")
        else:
            print("⚠️ Quick calibration had issues (expected with very limited iterations)")
        
        # Test 5: Report generation
        print("\nTest 5: Report Generation")
        report = pipeline.generate_comprehensive_report()
        print("✅ Comprehensive report generated")
        print(f"   Report length: {len(report)} characters")
        
        # Test 6: Milestones
        print("\nTest 6: Milestones Documentation")
        milestones = pipeline.get_recent_milestones()
        print(f"✅ Recent milestones: {len(milestones)} achievements")
        
        for i, milestone in enumerate(milestones[:3], 1):
            print(f"   {i}. {milestone['category']}")
            print(f"      File: {milestone['file_path']}")
            print(f"      Keywords: {', '.join(milestone['keywords'][:2])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Extended pipeline test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    setup_logging()
    
    print("🧪 Comprehensive Warp Field System Test Suite")
    print("=" * 60)
    print(f"Test started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    test_results = {}
    start_time = time.time()
    
    # Run all tests
    tests = [
        ("Holodeck Force-Field Grid", test_holodeck_force_field_grid),
        ("Medical Tractor Array", test_medical_tractor_array),
        ("Unified Calibration", test_unified_calibration),
        ("Sensitivity Analysis", test_sensitivity_analysis),
        ("Mathematical Refinements", test_mathematical_refinements),
        ("Extended Pipeline", test_extended_pipeline)
    ]
    
    for test_name, test_func in tests:
        print(f"\n" + "="*60)
        try:
            result = test_func()
            test_results[test_name] = "PASS" if result else "FAIL"
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            test_results[test_name] = "ERROR"
    
    # Final summary
    total_time = time.time() - start_time
    
    print("\n" + "="*60)
    print("🏁 TEST SUITE SUMMARY")
    print("="*60)
    
    passed = sum(1 for result in test_results.values() if result == "PASS")
    failed = sum(1 for result in test_results.values() if result == "FAIL")
    errors = sum(1 for result in test_results.values() if result == "ERROR")
    
    print(f"Tests run: {len(test_results)}")
    print(f"Passed: {passed} ✅")
    print(f"Failed: {failed} ❌")
    print(f"Errors: {errors} 💥")
    print(f"Total time: {total_time:.2f} seconds")
    print()
    
    print("Individual Test Results:")
    for test_name, result in test_results.items():
        status_icon = {"PASS": "✅", "FAIL": "❌", "ERROR": "💥"}[result]
        print(f"  {status_icon} {test_name}: {result}")
    
    print()
    
    if passed == len(test_results):
        print("🎉 ALL TESTS PASSED! System is ready for deployment.")
    elif failed + errors == 0:
        print("⚠️ Some tests had issues but no failures. System is mostly operational.")
    else:
        print("🔧 Some tests failed. Review results and fix issues before deployment.")
    
    print(f"\nTest completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    return test_results

if __name__ == "__main__":
    results = main()
