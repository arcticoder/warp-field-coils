#!/usr/bin/env python3
"""
Simple test script for warp field coil components
"""

import sys
import os
sys.path.append('.')

def test_dependencies():
    """Test all required dependencies."""
    print("=== TESTING WARP FIELD COIL DEPENDENCIES ===\n")
    
    # Test basic scientific libraries
    print("1. Testing basic scientific libraries...")
    try:
        import numpy as np
        import scipy
        import matplotlib.pyplot as plt
        print("✓ NumPy, SciPy, Matplotlib available")
    except ImportError as e:
        print(f"❌ Basic libraries failed: {e}")
        return False
    
    # Test JAX
    print("\n2. Testing JAX...")
    try:
        import jax.numpy as jnp
        from jax import jit
        print("✓ JAX available")
        jax_available = True
    except ImportError:
        print("❌ JAX not available - some optimizations will be disabled")
        jax_available = False
    
    # Test SymPy
    print("\n3. Testing SymPy...")
    try:
        import sympy as sp
        print("✓ SymPy available")
    except ImportError:
        print("❌ SymPy not available - exotic matter profiling will fail")
        return False
    
    # Test control systems
    print("\n4. Testing control library...")
    try:
        import control
        print("✓ Control library available")
    except ImportError:
        print("❌ Control library not available")
    
    return True

def test_exotic_matter_profiler():
    """Test the exotic matter profiler."""
    print("\n=== TESTING EXOTIC MATTER PROFILER ===")
    
    try:
        from src.stress_energy.exotic_matter_profile import ExoticMatterProfiler, alcubierre_profile
        
        # Create profiler
        profiler = ExoticMatterProfiler(r_min=0.1, r_max=5.0, n_points=100)
        print("✓ ExoticMatterProfiler created")
        
        # Test profile computation
        r_array, T00_profile = profiler.compute_T00_profile(
            lambda r: alcubierre_profile(r, R=2.0, sigma=0.5)
        )
        print(f"✓ T00 profile computed: {len(T00_profile)} points")
        
        # Test exotic region identification
        exotic_info = profiler.identify_exotic_regions(T00_profile)
        print(f"✓ Exotic regions identified: {exotic_info['has_exotic']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Exotic matter profiler failed: {e}")
        return False

def test_coil_optimizer():
    """Test the coil optimizer (if JAX is available)."""
    print("\n=== TESTING COIL OPTIMIZER ===")
    
    try:
        import jax.numpy as jnp
        from jax import jit
    except ImportError:
        print("❌ JAX not available - skipping coil optimizer test")
        return False
    
    try:
        from src.coil_optimizer.advanced_coil_optimizer import AdvancedCoilOptimizer
        
        # Create optimizer
        optimizer = AdvancedCoilOptimizer(r_min=0.1, r_max=5.0, n_points=50)
        print("✓ AdvancedCoilOptimizer created")
        
        # Test target profile setting
        r_target = optimizer.rs
        T00_target = -0.1 * jnp.exp(-((r_target - 2.0)/0.5)**2)
        optimizer.set_target_profile(r_target, T00_target)
        print("✓ Target profile set")
        
        # Test objective function
        test_params = jnp.array([0.1, 2.0, 0.5])
        obj_val = optimizer.objective_function(test_params)
        print(f"✓ Objective function evaluated: {obj_val:.6e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Coil optimizer failed: {e}")
        return False

def test_field_simulator():
    """Test the electromagnetic field simulator."""
    print("\n=== TESTING FIELD SIMULATOR ===")
    
    try:
        from src.hardware.field_rig_design import ElectromagneticFieldSimulator
        
        # Create simulator
        simulator = ElectromagneticFieldSimulator()
        print("✓ ElectromagneticFieldSimulator created")
        
        # Test field simulation
        results = simulator.simulate_inductive_rig(
            L=1e-3, I=1000, f_mod=1000, geometry='toroidal'
        )
        print(f"✓ Field simulation completed: B_peak = {results.B_peak:.2f} T")
        
        # Test safety analysis
        safety_status = simulator.safety_analysis(results)
        print(f"✓ Safety analysis completed: {len(safety_status)} checks")
        
        return True
        
    except Exception as e:
        print(f"❌ Field simulator failed: {e}")
        return False

def test_control_system():
    """Test the control system."""
    print("\n=== TESTING CONTROL SYSTEM ===")
    
    try:
        import control
    except ImportError:
        print("❌ Control library not available - skipping control test")
        return False
    
    try:
        from src.control.closed_loop_controller import ClosedLoopFieldController, PlantParams
        
        # Create plant parameters
        plant_params = PlantParams(K=1.0, omega_n=10.0, zeta=0.7)
        print("✓ Plant parameters created")
        
        # Create controller
        controller = ClosedLoopFieldController(plant_params)
        print("✓ ClosedLoopFieldController created")
        
        # Test PID tuning
        pid_params = controller.tune_pid_ziegler_nichols()
        print(f"✓ PID tuned: kp={pid_params.kp:.3f}, ki={pid_params.ki:.3f}, kd={pid_params.kd:.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Control system failed: {e}")
        return False

def run_all_tests():
    """Run all component tests."""
    print("WARP FIELD COIL COMPONENT TESTING")
    print("=" * 50)
    
    # Test dependencies first
    if not test_dependencies():
        print("\n❌ CRITICAL: Basic dependencies failed. Please install requirements.")
        return False
    
    # Test individual components
    results = []
    results.append(test_exotic_matter_profiler())
    results.append(test_coil_optimizer())
    results.append(test_field_simulator())
    results.append(test_control_system())
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print(f"\n{'=' * 50}")
    print(f"TEST SUMMARY: {passed}/{total} components passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! The system is ready for use.")
    elif passed > total // 2:
        print("⚠️  MOST TESTS PASSED. Some components may have limited functionality.")
    else:
        print("❌ MULTIPLE FAILURES. Please check dependencies and installation.")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
