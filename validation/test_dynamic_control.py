#!/usr/bin/env python3
"""
Test Dynamic Trajectory Control
Quick validation of the enhanced pipeline with dynamic control capabilities
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, "src")

def main():
    print("🚀 TESTING DYNAMIC TRAJECTORY CONTROL PIPELINE")
    print("=" * 55)
    
    try:
        # Import enhanced pipeline
        from run_unified_pipeline import UnifiedWarpFieldPipeline
        
        # Create configuration for testing
        test_config = {
            'r_min': 0.1,
            'r_max': 5.0,
            'n_points': 100,          # Reduced for faster testing
            'n_points_coil': 50,      # Reduced for faster testing
            'warp_profile_type': 'alcubierre',
            'warp_radius': 2.0,
            'warp_width': 0.8,
            'effective_mass': 1e-18,  # Small effective mass for testing
            'max_dipole_strength': 0.4,
            'control_frequency': 50.0, # Lower frequency for testing
            'optimization_method': 'lbfgs'
        }
        
        # Initialize pipeline
        print("📋 Initializing enhanced pipeline...")
        pipeline = UnifiedWarpFieldPipeline()
        pipeline.config.update(test_config)
        
        # Test core functionality first
        print("\n📋 Testing core exotic matter profile...")
        step1_results = pipeline.step_1_define_exotic_matter_profile()
        
        exotic_info = step1_results['exotic_info']
        print(f"  ✓ Exotic regions detected: {exotic_info['has_exotic']}")
        if exotic_info['has_exotic']:
            print(f"  ✓ Exotic energy: {exotic_info['total_exotic_energy']:.2e} J")
        
        # Test coil optimization
        print("\n📋 Testing coil optimization...")
        step2_results = pipeline.step_2_optimize_coil_geometry()
        
        opt_success = step2_results['optimization_result']['success']
        print(f"  ✓ Optimization success: {opt_success}")
        if opt_success:
            print(f"  ✓ Final objective: {step2_results['optimization_result']['optimal_objective']:.2e}")
        
        # Test dynamic trajectory control
        print("\n📋 Testing dynamic trajectory control...")
        step14_results = pipeline.step_14_dynamic_trajectory_control(
            trajectory_type="smooth_acceleration",
            simulation_time=8.0,
            max_velocity=30.0,
            max_acceleration=6.0
        )
        
        # Analyze results
        perf = step14_results['performance_analysis']
        tracking_performance = perf['tracking_performance']
        efficiency = perf['efficiency_metrics']
        
        print(f"  ✓ Velocity tracking RMS: {tracking_performance['velocity_rms_error']:.3f} m/s")
        print(f"  ✓ Acceleration tracking RMS: {tracking_performance['acceleration_rms_error']:.3f} m/s²")
        print(f"  ✓ Energy efficiency: {efficiency['energy_efficiency']*100:.1f}%")
        print(f"  ✓ Max dipole utilization: {perf['control_authority']['dipole_utilization']*100:.1f}%")
        
        # Test multi-axis maneuvering
        print("\n📋 Testing multi-axis maneuvering...")
        
        # Define simple test maneuver sequence
        test_maneuvers = [
            {
                'time_start': 0.0,
                'time_end': 3.0,
                'maneuver_type': 'acceleration',
                'direction': np.array([0, 0, 1]),
                'target_velocity': 20.0,
                'description': 'Forward acceleration test'
            },
            {
                'time_start': 3.0,
                'time_end': 6.0,
                'maneuver_type': 'steering_turn',
                'direction': np.array([1, 0, 0]),
                'target_velocity': 20.0,
                'description': 'Right turn test'
            },
            {
                'time_start': 6.0,
                'time_end': 8.0,
                'maneuver_type': 'deceleration',
                'direction': np.array([0, 0, -1]),
                'target_velocity': 0.0,
                'description': 'Deceleration test'
            }
        ]
        
        step15_results = pipeline.step_15_multi_axis_maneuvering(
            maneuver_sequence=test_maneuvers,
            simulation_time=8.0
        )
        
        # Analyze multi-axis results
        multi_analysis = step15_results['performance_analysis']
        trajectory_metrics = multi_analysis['trajectory_metrics']
        
        print(f"  ✓ Total distance: {trajectory_metrics['total_distance_traveled']:.1f} m")
        print(f"  ✓ Path efficiency: {trajectory_metrics['path_efficiency']*100:.1f}%")
        print(f"  ✓ Max speed: {trajectory_metrics['max_speed_achieved']:.1f} m/s")
        print(f"  ✓ Max acceleration: {trajectory_metrics['max_acceleration_used']:.1f} m/s²")
        
        # Overall assessment
        print("\n" + "=" * 55)
        print("🎯 DYNAMIC CONTROL VALIDATION RESULTS")
        print("=" * 55)
        
        # Success criteria
        velocity_tracking_good = tracking_performance['velocity_rms_error'] < 2.0
        energy_efficient = efficiency['energy_efficiency'] > 0.05
        control_authority_ok = perf['control_authority']['dipole_utilization'] < 0.9
        path_efficient = trajectory_metrics['path_efficiency'] > 0.3
        
        success_count = sum([
            velocity_tracking_good,
            energy_efficient, 
            control_authority_ok,
            path_efficient
        ])
        
        print(f"Velocity tracking: {'✅ GOOD' if velocity_tracking_good else '⚠️ NEEDS WORK'}")
        print(f"Energy efficiency: {'✅ GOOD' if energy_efficient else '⚠️ NEEDS WORK'}")
        print(f"Control authority: {'✅ GOOD' if control_authority_ok else '⚠️ NEEDS WORK'}")
        print(f"Path efficiency: {'✅ GOOD' if path_efficient else '⚠️ NEEDS WORK'}")
        
        print(f"\nOverall score: {success_count}/4 criteria met")
        
        if success_count >= 3:
            print("🎉 DYNAMIC TRAJECTORY CONTROL: OPERATIONAL")
            print("✅ System ready for steerable warp drive operations")
            return True
        else:
            print("⚠️ DYNAMIC TRAJECTORY CONTROL: NEEDS OPTIMIZATION")
            print("🔧 System functional but requires tuning")
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Check that all required modules are available")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Starting dynamic trajectory control validation...")
    print("This test validates the enhanced pipeline with:")
    print("  - Dynamic trajectory following")
    print("  - Multi-axis maneuvering")
    print("  - Steerable acceleration/deceleration")
    print("  - Control theory integration")
    print()
    
    success = main()
    
    if success:
        print("\n🚀 DYNAMIC CONTROL VALIDATION: PASSED")
        print("   Enhanced warp drive pipeline ready for deployment!")
    else:
        print("\n⚠️ DYNAMIC CONTROL VALIDATION: INCOMPLETE")
        print("   System needs further development before deployment")
    
    print("\nTest completed.")
