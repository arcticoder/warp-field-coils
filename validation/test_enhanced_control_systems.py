"""
Test Enhanced IDF and SIF Control Systems
========================================

Comprehensive testing of Enhanced Inertial Damper Field and Structural Integrity Field
with stress-energy backreaction and curvature coupling.
"""

import sys
import os
import numpy as np
import time
import logging

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, 'src'))
sys.path.insert(0, current_dir)

def test_enhanced_idf():
    """Test Enhanced Inertial Damper Field"""
    print("\n🛡️ Testing Enhanced Inertial Damper Field")
    print("=" * 50)
    
    try:
        from control.enhanced_inertial_damper import EnhancedInertialDamperField
        
        # Test 1: Basic initialization
        print("Test 1: IDF Initialization")
        idf = EnhancedInertialDamperField(
            alpha_max=1e-4 * 9.81,  # 10⁻⁴ g
            j_max=2.0,              # 2 m/s³ max jerk
            lambda_coupling=1e-2,   # Curvature coupling
            effective_density=0.5,  # Effective density
            a_max=5.0               # 5 m/s² safety limit
        )
        print(f"✅ IDF initialized: K_IDF={idf.K_IDF:.2e}")
        print(f"   Safety limit: {idf.a_max:.1f} m/s²")
        
        # Test 2: Basic acceleration computation
        print("\nTest 2: Basic Acceleration Computation")
        
        # Test jerk scenarios
        test_jerks = [
            np.array([0.5, 0.0, 0.0]),   # Mild longitudinal jerk
            np.array([0.0, 1.0, 0.0]),   # Moderate lateral jerk  
            np.array([0.0, 0.0, 2.0]),   # Strong vertical jerk
            np.array([1.5, 1.0, 0.5])    # Combined 3D jerk
        ]
        
        # Mock metric tensor (slightly perturbed Minkowski)
        metric = np.eye(4)
        metric[0, 0] = -1.0  # Time component
        metric += 1e-3 * np.random.random((4, 4))  # Small perturbation
        metric = (metric + metric.T) / 2  # Ensure symmetry
        
        for i, j_res in enumerate(test_jerks):
            result = idf.compute_acceleration(j_res, metric)
            a_total = result['acceleration']
            components = result['components']
            
            print(f"   Test jerk {i+1}: ||j|| = {np.linalg.norm(j_res):.2f} m/s³")
            print(f"   → Total acceleration: ||a|| = {np.linalg.norm(a_total):.3f} m/s²")
            print(f"   → Base component: {np.linalg.norm(components['base']):.3f} m/s²")
            print(f"   → Curvature component: {np.linalg.norm(components['curvature']):.3f} m/s²")
            print(f"   → Backreaction component: {np.linalg.norm(components['backreaction']):.3f} m/s²")
            
            # Verify safety limits
            if np.linalg.norm(a_total) > idf.a_max:
                print(f"   ⚠️ Safety limit active")
            else:
                print(f"   ✅ Within safety limits")
        
        # Test 3: Performance tracking
        print("\nTest 3: Performance Tracking")
        
        # Multiple computations to build performance history
        for _ in range(10):
            j_test = np.random.uniform(-1, 1, 3)
            idf.compute_acceleration(j_test, metric)
        
        metrics = idf.get_performance_metrics()
        print(f"✅ Performance metrics computed:")
        print(f"   Average jerk: {metrics.get('average_jerk', 0):.3f} m/s³")
        print(f"   Average acceleration: {metrics.get('average_acceleration', 0):.3f} m/s²")
        print(f"   Safety violation rate: {metrics.get('safety_violation_rate', 0):.1%}")
        print(f"   Average computation time: {metrics.get('average_computation_time', 0)*1000:.2f} ms")
        
        # Test 4: System diagnostics
        print("\nTest 4: System Diagnostics")
        diag = idf.run_diagnostics()
        print(f"✅ Overall health: {diag['overall_health']}")
        print(f"   Health score: {diag['health_score']:.2f}")
        print(f"   Total computations: {diag['performance_metrics']['total_computations']}")
        print(f"   Einstein equations available: {diag['module_availability']['einstein_equations']}")
        
        return True
        
    except Exception as e:
        print(f"❌ IDF test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_enhanced_sif():
    """Test Enhanced Structural Integrity Field"""
    print("\n🏗️ Testing Enhanced Structural Integrity Field")
    print("=" * 50)
    
    try:
        from control.enhanced_structural_integrity import EnhancedStructuralIntegrityField
        
        # Test 1: Basic initialization
        print("Test 1: SIF Initialization")
        sif = EnhancedStructuralIntegrityField(
            material_coupling=0.8,     # Material coupling
            ricci_coupling=1e-2,       # Ricci tensor coupling
            weyl_coupling=1.2,         # Weyl tensor coupling
            stress_max=1e-6            # 1 μN/m² stress safety limit
        )
        print(f"✅ SIF initialized: μ_mat={sif.mu_mat:.2f}")
        print(f"   Stress limit: {sif.s_max:.2e} N/m²")
        
        # Test 2: Stress compensation computation
        print("\nTest 2: Stress Compensation Computation")
        
        # Mock metric tensors with different curvature characteristics
        test_metrics = []
        
        # Flat space
        metric_flat = np.eye(4)
        metric_flat[0, 0] = -1.0
        test_metrics.append(("Flat space", metric_flat))
        
        # Curved space (weak field)
        metric_curved = np.eye(4)
        metric_curved[0, 0] = -1.0
        metric_curved[3, 3] = 1 + 1e-4  # Small spatial curvature
        test_metrics.append(("Weak curvature", metric_curved))
        
        # Strong curvature
        metric_strong = np.eye(4)
        metric_strong[0, 0] = -1.0
        for i in range(1, 4):
            metric_strong[i, i] = 1 + 1e-3 * i  # Graduated curvature
        test_metrics.append(("Strong curvature", metric_strong))
        
        # Material stress tensor
        material_stress = np.array([[1e-8, 1e-9, 0],
                                   [1e-9, 2e-8, 1e-9],
                                   [0, 1e-9, 1.5e-8]])
        
        for name, metric in test_metrics:
            result = sif.compute_compensation(metric, material_stress)
            sigma_comp = result['stress_compensation']
            components = result['components']
            diagnostics = result['diagnostics']
            
            print(f"   {name}:")
            print(f"   → Compensation magnitude: {np.linalg.norm(sigma_comp):.2e} N/m²")
            print(f"   → Base Weyl stress: {np.linalg.norm(components['base_weyl_stress']):.2e} N/m²")
            print(f"   → Ricci contribution: {np.linalg.norm(components['ricci_contribution']):.2e} N/m²")
            print(f"   → LQG correction: {np.linalg.norm(components['lqg_correction']):.2e} N/m²")
            
            # Check curvature norms
            curv = diagnostics['curvature_tensors']
            print(f"   → Curvature norms: R={curv['riemann_norm']:.2e}, "
                  f"Ric={curv['ricci_norm']:.2e}, W={curv['weyl_norm']:.2e}")
        
        # Test 3: Performance tracking
        print("\nTest 3: Performance Tracking")
        
        # Multiple computations to build performance history
        for _ in range(8):
            # Random metric perturbations
            metric_test = np.eye(4)
            metric_test[0, 0] = -1.0
            metric_test += 1e-4 * np.random.random((4, 4))
            metric_test = (metric_test + metric_test.T) / 2
            sif.compute_compensation(metric_test, material_stress)
        
        metrics = sif.get_performance_metrics()
        print(f"✅ Performance metrics computed:")
        print(f"   Average material stress: {metrics.get('average_material_stress', 0):.2e} N/m²")
        print(f"   Average compensation: {metrics.get('average_compensation', 0):.2e} N/m²")
        print(f"   Safety violation rate: {metrics.get('safety_violation_rate', 0):.1%}")
        print(f"   Average computation time: {metrics.get('average_computation_time', 0)*1000:.2f} ms")
        
        # Test 4: System diagnostics
        print("\nTest 4: System Diagnostics")
        diag = sif.run_diagnostics()
        print(f"✅ Overall health: {diag['overall_health']}")
        print(f"   Health score: {diag['health_score']:.2f}")
        print(f"   Total computations: {diag['performance_metrics']['total_computations']}")
        config = diag['configuration']
        print(f"   Configuration: Weyl coupling = {config['weyl_coupling']:.2f}, "
              f"Ricci coupling = {config['ricci_coupling']:.2e}")
        
        return True
        
    except Exception as e:
        print(f"❌ SIF test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integrated_idf_sif_system():
    """Test integrated IDF-SIF system operation"""
    print("\n🔗 Testing Integrated IDF-SIF System")
    print("=" * 40)
    
    try:
        from control.enhanced_inertial_damper import EnhancedInertialDamperField
        from control.enhanced_structural_integrity import EnhancedStructuralIntegrityField
        
        # Initialize both systems
        idf = EnhancedInertialDamperField(
            alpha_max=1e-4 * 9.81,
            j_max=1.0,
            lambda_coupling=1e-2,
            effective_density=0.5,
            a_max=4.0
        )
        
        sif = EnhancedStructuralIntegrityField(
            material_coupling=0.8,
            ricci_coupling=1e-2,
            weyl_coupling=1.2,
            stress_max=1e-6
        )
        
        print("✅ Integrated system initialized")
        
        # Test 1: Coupled operation simulation
        print("\nTest 1: Coupled Operation Simulation")
        
        # Simulate dynamic scenario
        dt = 0.01  # 10 ms time steps
        n_steps = 20
        
        total_idf_energy = 0.0
        total_sif_energy = 0.0
        
        for step in range(n_steps):
            # Simulate time-varying jerk (e.g., turbulence)
            t = step * dt
            j_res = np.array([
                0.5 * np.sin(2 * np.pi * t),
                0.3 * np.cos(3 * np.pi * t),
                0.2 * np.sin(5 * np.pi * t)
            ])
            
            # Time-varying metric (e.g., gravitational waves)
            metric = np.eye(4)
            metric[0, 0] = -1.0
            wave_amp = 1e-4 * np.sin(10 * np.pi * t)
            metric[1, 1] += wave_amp
            metric[2, 2] -= wave_amp
            
            # Material stress (varying with deformation)
            material_stress = np.array([[1e-8 * (1 + 0.1*t), 1e-9, 0],
                                       [1e-9, 2e-8 * (1 + 0.05*t), 1e-9],
                                       [0, 1e-9, 1.5e-8]])
            
            # Compute IDF response
            idf_result = idf.compute_acceleration(j_res, metric)
            a_idf = idf_result['acceleration']
            
            # Compute SIF response
            sif_result = sif.compute_compensation(metric, material_stress)
            sigma_sif = sif_result['stress_compensation']
            
            # Energy tracking
            idf_energy = 0.5 * np.dot(a_idf, a_idf)
            sif_energy = 0.5 * np.trace(sigma_sif @ sigma_sif)
            
            total_idf_energy += idf_energy * dt
            total_sif_energy += sif_energy * dt
            
            if step % 5 == 0:  # Print every 5th step
                print(f"   t={t:.2f}s: ||j||={np.linalg.norm(j_res):.3f}, "
                      f"||a_IDF||={np.linalg.norm(a_idf):.3f}, "
                      f"||σ_SIF||={np.linalg.norm(sigma_sif):.2e}")
        
        print(f"✅ Simulation completed:")
        print(f"   Total IDF energy: {total_idf_energy:.3f}")
        print(f"   Total SIF energy: {total_sif_energy:.2e}")
        
        # Test 2: Performance comparison
        print("\nTest 2: Performance Comparison")
        
        idf_metrics = idf.get_performance_metrics()
        sif_metrics = sif.get_performance_metrics()
        
        print(f"✅ IDF Performance:")
        print(f"   Computations: {idf_metrics.get('total_computations', 0)}")
        print(f"   Safety violations: {idf_metrics.get('safety_violation_rate', 0):.1%}")
        print(f"   Average acceleration: {idf_metrics.get('average_acceleration', 0):.3f} m/s²")
        
        print(f"✅ SIF Performance:")
        print(f"   Computations: {sif_metrics.get('total_computations', 0)}")
        print(f"   Safety violations: {sif_metrics.get('safety_violation_rate', 0):.1%}")
        print(f"   Average compensation: {sif_metrics.get('average_compensation', 0):.2e} N/m²")
        
        # Test 3: Combined diagnostics
        print("\nTest 3: Combined System Diagnostics")
        
        idf_diag = idf.run_diagnostics()
        sif_diag = sif.run_diagnostics()
        
        overall_health = "HEALTHY" if (idf_diag['overall_health'] == "HEALTHY" and 
                                     sif_diag['overall_health'] == "HEALTHY") else "DEGRADED"
        
        print(f"✅ Overall system health: {overall_health}")
        print(f"   IDF status: {idf_diag['overall_health']} (score: {idf_diag['health_score']:.2f})")
        print(f"   SIF status: {sif_diag['overall_health']} (score: {sif_diag['health_score']:.2f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Integrated system test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all enhanced control system tests"""
    print("🛠️ Enhanced Control Systems Test Suite")
    print("=" * 60)
    print(f"Test started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    test_results = {}
    start_time = time.time()
    
    # Run tests
    tests = [
        ("Enhanced Inertial Damper Field", test_enhanced_idf),
        ("Enhanced Structural Integrity Field", test_enhanced_sif),
        ("Integrated IDF-SIF System", test_integrated_idf_sif_system)
    ]
    
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        try:
            result = test_func()
            test_results[test_name] = "PASS" if result else "FAIL"
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            test_results[test_name] = "ERROR"
    
    # Summary
    total_time = time.time() - start_time
    passed = sum(1 for result in test_results.values() if result == "PASS")
    
    print(f"\n{'='*60}")
    print("🏁 ENHANCED CONTROL SYSTEMS TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Passed: {passed}/{len(tests)} ✅")
    print(f"Time: {total_time:.1f}s")
    
    for test, result in test_results.items():
        icon = {"PASS": "✅", "FAIL": "❌", "ERROR": "💥"}[result]
        print(f"  {icon} {test}: {result}")
    
    if passed == len(tests):
        print("\n🎉 All enhanced control system tests passed!")
        print("🚀 IDF and SIF systems ready for integration!")
    else:
        print(f"\n⚠️ {len(tests)-passed} tests had issues")
    
    return test_results

if __name__ == "__main__":
    results = main()
