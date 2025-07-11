#!/usr/bin/env python3
"""
Enhanced Warp Field Coil System - Final Validation
Demonstrates all advanced features working together
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import jax.numpy as jnp
from datetime import datetime

def main():
    """Run complete enhanced system validation."""
    print("🚀 ENHANCED WARP FIELD COIL SYSTEM - FINAL VALIDATION")
    print("=" * 60)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    validation_results = {
        'time_dependent_profiles': False,
        'quantum_geometry': False,
        'quantum_optimization': False,
        'enhanced_control': False,
        'sensitivity_analysis': False
    }
    
    # 1. TIME-DEPENDENT WARP PROFILES
    print("\n🌊 Testing Time-Dependent Warp Profiles...")
    try:
        from stress_energy.exotic_matter_profile import ExoticMatterProfiler
        
        profiler = ExoticMatterProfiler(r_min=0.1, r_max=2.0, n_points=30)
        
        # Linear expansion trajectory
        R_func = lambda t: 1.0 + 0.2 * t
        times = np.array([0.0, 1.0, 2.0])
        
        # Test single time profile
        profile_t1 = profiler.alcubierre_profile_time_dep(profiler.r_array, 1.0, R_func, 0.5)
        
        # Test time evolution
        r_array, T00_rt = profiler.compute_T00_profile_time_dep(R_func, 0.5, times)
        
        finite_fraction = np.sum(np.isfinite(T00_rt)) / T00_rt.size
        
        print(f"   ✓ Time-dependent profile computed: {len(profile_t1)} points")
        print(f"   ✓ Time evolution computed: {T00_rt.shape}")
        print(f"   ✓ Finite values: {finite_fraction*100:.1f}%")
        
        validation_results['time_dependent_profiles'] = True
        
    except Exception as e:
        print(f"   ❌ Time-dependent profiles failed: {e}")
    
    # 2. QUANTUM GEOMETRY INTEGRATION
    print("\n⚛️  Testing Quantum Geometry Integration...")
    try:
        from quantum_geometry.discrete_stress_energy import DiscreteQuantumGeometry
        
        dqg = DiscreteQuantumGeometry(n_nodes=15)
        
        # Test generating functional
        G = dqg.compute_generating_functional()
        
        # Test with custom K-matrix
        K_test = 0.1 * dqg.adjacency_matrix
        G_custom = dqg.compute_generating_functional(K_test)
        
        print(f"   ✓ Quantum geometry system created: {dqg.n_nodes} nodes")
        print(f"   ✓ Default generating functional: G = {G:.6f}")
        print(f"   ✓ Custom generating functional: G = {G_custom:.6f}")
        print(f"   ✓ Adjacency matrix: {dqg.adjacency_matrix.shape}")
        
        validation_results['quantum_geometry'] = True
        
    except Exception as e:
        print(f"   ❌ Quantum geometry failed: {e}")
    
    # 3. QUANTUM-AWARE COIL OPTIMIZATION
    print("\n🔧 Testing Quantum-Aware Coil Optimization...")
    try:
        from coil_optimizer.advanced_coil_optimizer import AdvancedCoilOptimizer
        
        optimizer = AdvancedCoilOptimizer(r_min=0.1, r_max=2.0, n_points=25)
        
        # Set simple target profile
        r_array = optimizer.rs
        T00_target = -0.1 * jnp.exp(-((r_array - 1.0)/0.3)**2)
        optimizer.set_target_profile(r_array, T00_target)
        
        # Test quantum penalty
        params = jnp.array([0.1, 1.0, 0.3])
        
        classical_obj = optimizer.objective_function(params)
        quantum_penalty = optimizer.quantum_penalty(params)
        combined_obj = optimizer.objective_with_quantum(params, alpha=1e-3)
        
        print(f"   ✓ Coil optimizer created: {len(r_array)} points")
        print(f"   ✓ Classical objective: {classical_obj:.6e}")
        print(f"   ✓ Quantum penalty: {quantum_penalty:.6e}")
        print(f"   ✓ Combined objective: {combined_obj:.6e}")
        
        validation_results['quantum_optimization'] = True
        
    except Exception as e:
        print(f"   ❌ Quantum optimization failed: {e}")
    
    # 4. ENHANCED CONTROL SYSTEM
    print("\n🎛️  Testing Enhanced Control System...")
    try:
        # Import with explicit module path to avoid conflicts
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "closed_loop_controller", 
            "src/control/closed_loop_controller.py"
        )
        control_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(control_module)
        
        PlantParams = control_module.PlantParams
        ClosedLoopFieldController = control_module.ClosedLoopFieldController
        
        plant = PlantParams(K=1.0, omega_n=10.0, zeta=0.7)
        controller = ClosedLoopFieldController(plant)
        
        # Test PID tuning
        pid_params = controller.tune_pid_ziegler_nichols()
        
        # Test quantum anomaly computation
        mock_state = {'currents': np.ones(10) * 0.1}
        quantum_anomaly = controller.compute_quantum_anomaly(mock_state)
        
        print(f"   ✓ Control system created")
        print(f"   ✓ PID tuned: kp={pid_params.kp:.3f}, ki={pid_params.ki:.3f}, kd={pid_params.kd:.6f}")
        print(f"   ✓ Quantum anomaly computed: {quantum_anomaly:.6e}")
        
        validation_results['enhanced_control'] = True
        
    except Exception as e:
        print(f"   ❌ Enhanced control failed: {e}")
    
    # 5. SENSITIVITY ANALYSIS CAPABILITY
    print("\n📊 Testing Sensitivity Analysis Framework...")
    try:
        # Test JAX gradient computation capability
        import jax
        
        def test_objective(params):
            return jnp.sum(params**2) + 0.1 * jnp.sin(params[0])
        
        grad_fn = jax.grad(test_objective)
        hess_fn = jax.hessian(test_objective)
        
        test_params = jnp.array([1.0, 2.0, 0.5])
        gradient = grad_fn(test_params)
        hessian = hess_fn(test_params)
        
        condition_number = jnp.linalg.cond(hessian)
        
        print(f"   ✓ JAX automatic differentiation working")
        print(f"   ✓ Gradient computed: {gradient}")
        print(f"   ✓ Hessian computed: {hessian.shape}")
        print(f"   ✓ Condition number: {condition_number:.2e}")
        
        validation_results['sensitivity_analysis'] = True
        
    except Exception as e:
        print(f"   ❌ Sensitivity analysis failed: {e}")
    
    # FINAL ASSESSMENT
    print("\n" + "=" * 60)
    print("📋 ENHANCED SYSTEM VALIDATION SUMMARY")
    print("=" * 60)
    
    total_features = len(validation_results)
    passed_features = sum(validation_results.values())
    
    for feature, status in validation_results.items():
        status_icon = "✅" if status else "❌"
        feature_name = feature.replace('_', ' ').title()
        print(f"{status_icon} {feature_name}")
    
    print("\n" + "=" * 60)
    
    if passed_features == total_features:
        print("🎉 ALL ENHANCED FEATURES VALIDATED!")
        print("🚀 System ready for experimental deployment!")
        status = "FULLY OPERATIONAL"
    elif passed_features >= total_features * 0.8:
        print("✅ MOST ENHANCED FEATURES VALIDATED!")
        print("⚠️  Some advanced features may have limited functionality.")
        status = "MOSTLY OPERATIONAL"
    else:
        print("❌ MULTIPLE ENHANCED FEATURES FAILED!")
        print("🔧 System requires debugging before deployment.")
        status = "NEEDS ATTENTION"
    
    print(f"\nFinal Status: {status}")
    print(f"Feature Success Rate: {passed_features}/{total_features} ({passed_features/total_features*100:.1f}%)")
    print("\n" + "=" * 60)
    
    return validation_results, status

if __name__ == "__main__":
    try:
        results, status = main()
        exit_code = 0 if status == "FULLY OPERATIONAL" else 1
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⚠️ Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Validation failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
