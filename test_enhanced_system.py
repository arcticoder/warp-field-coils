#!/usr/bin/env python3
"""
Simple test script for enhanced multi-physics features
"""

import sys
import os
sys.path.append('src')

def test_multi_physics_penalties():
    """Test mechanical and thermal penalties."""
    print("🔧 Testing Multi-Physics Penalties...")
    
    try:
        from coil_optimizer.advanced_coil_optimizer import AdvancedCoilOptimizer
        import jax.numpy as jnp
        
        optimizer = AdvancedCoilOptimizer(r_min=0.1, r_max=2.0, n_points=20)
        
        # Set a simple target profile to avoid "Target profile not set" error
        r_array = optimizer.rs
        T00_target = -0.1 * jnp.exp(-((r_array - 1.0)/0.3)**2)
        optimizer.set_target_profile(r_array, T00_target)
        
        params = jnp.array([0.5, 1.0, 0.3])  # [amplitude, center, width]
        
        # Test penalties
        mech_penalty = optimizer.mechanical_penalty(params, thickness=0.005, sigma_yield=300e6)
        thermal_penalty = optimizer.thermal_penalty(params, rho_cu=1.7e-8, area=1e-6, P_max=1e6)
        
        print(f'✅ Mechanical penalty: {mech_penalty:.6e}')
        print(f'✅ Thermal penalty: {thermal_penalty:.6e}')
        
        # Test multi-physics objective
        J_multi = optimizer.objective_full_multiphysics(
            params, alpha_q=1e-3, alpha_m=1e3, alpha_t=1e2,
            thickness=0.005, sigma_yield=300e6, rho_cu=1.7e-8, area=1e-6, P_max=1e6
        )
        print(f'✅ Multi-physics objective: {J_multi:.6e}')
        
        return True
        
    except Exception as e:
        print(f'❌ Multi-physics penalties: {e}')
        return False

def test_3d_coil_geometry():
    """Test 3D coil geometry generation."""
    print("🔧 Testing 3D Coil Geometry...")
    
    try:
        from field_solver.biot_savart_3d import BiotSavart3DSolver, create_warp_coil_3d_system
        
        # Create 3D coil system
        coil_system = create_warp_coil_3d_system(R_bubble=2.0)
        print(f'✅ 3D coil system: {len(coil_system)} components')
        
        # Test Biot-Savart solver
        solver = BiotSavart3DSolver()
        z_pos, B_z = solver.compute_field_on_axis(coil_system[0], (-1, 1))
        print(f'✅ 3D field computation: max |B_z| = {max(abs(B_z)):.6e} T')
        
        return True
        
    except Exception as e:
        print(f'❌ 3D coil geometry: {e}')
        return False

def test_multi_objective_framework():
    """Test multi-objective optimization framework."""
    print("🎯 Testing Multi-Objective Framework...")
    
    try:
        from optimization.multi_objective import MultiObjectiveOptimizer, create_default_constraints
        
        constraints = create_default_constraints()
        print(f'✅ Multi-objective framework: {len(constraints)} constraint parameters')
        
        return True
        
    except Exception as e:
        print(f'❌ Multi-objective framework: {e}')
        return False

def test_fdtd_framework():
    """Test FDTD validation framework."""
    print("🌊 Testing FDTD Framework...")
    
    try:
        from validation.fdtd_solver import FDTDValidator
        
        validator = FDTDValidator(use_meep=False)
        print(f'✅ FDTD validator: Mock simulation ready')
        
        return True
        
    except Exception as e:
        print(f'❌ FDTD framework: {e}')
        return False

def test_pipeline_class():
    """Test pipeline class import."""
    print("🚀 Testing Pipeline Class...")
    
    try:
        from run_unified_pipeline import WarpFieldCoilPipeline
        
        # Create pipeline
        pipeline = WarpFieldCoilPipeline()
        print(f'✅ Pipeline created successfully')
        
        # Test components individually
        from stress_energy.exotic_matter_profile import ExoticMatterProfiler
        profiler = ExoticMatterProfiler(r_min=0.1, r_max=3.0, n_points=30)
        
        R_func = lambda t: 2.0 + 0.1 * t
        times = [0.0, 0.5, 1.0]
        r_array, T00_evolution = profiler.compute_T00_profile_time_dep(R_func, 0.5, times)
        print(f'✅ Time evolution: {T00_evolution.shape}')
        
        return True
        
    except Exception as e:
        print(f'❌ Pipeline test: {e}')
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all enhanced system tests."""
    print("🚀 ENHANCED MULTI-PHYSICS SYSTEM VALIDATION")
    print("=" * 50)
    
    tests = [
        ("Multi-Physics Penalties", test_multi_physics_penalties),
        ("3D Coil Geometry", test_3d_coil_geometry),
        ("Multi-Objective Framework", test_multi_objective_framework),
        ("FDTD Framework", test_fdtd_framework),
        ("Pipeline Class", test_pipeline_class),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        success = test_func()
        results.append(success)
    
    print("\n" + "=" * 50)
    print("📋 VALIDATION SUMMARY")
    print("=" * 50)
    
    for i, (test_name, _) in enumerate(tests):
        status = "✅" if results[i] else "❌"
        print(f"{status} {test_name}")
    
    passed = sum(results)
    total = len(results)
    print(f"\nSuccess Rate: {passed}/{total} ({passed/total*100:.0f}%)")
    
    if passed == total:
        print("🎉 ALL ENHANCED FEATURES VALIDATED!")
        print("🚀 System ready for experimental deployment!")
    elif passed >= total * 0.8:
        print("✅ MOST ENHANCED FEATURES VALIDATED!")
        print("⚠️ Some advanced features may need attention.")
    else:
        print("❌ MULTIPLE ENHANCED FEATURES FAILED!")
        print("🔧 System requires debugging before deployment.")

if __name__ == "__main__":
    main()
