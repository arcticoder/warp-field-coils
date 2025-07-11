#!/usr/bin/env python3
"""
Comprehensive Steerable Drive Validation
Final validation of all steerable warp drive components
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

def validate_core_functionality():
    """Validate core steerable drive functionality."""
    print("🔬 CORE FUNCTIONALITY VALIDATION")
    print("-" * 40)
    
    results = {"passed": 0, "total": 0}
    
    # Test 1: Dipolar Profile Generation
    print("Test 1: Dipolar Profile Generation")
    results["total"] += 1
    try:
        from stress_energy.exotic_matter_profile import alcubierre_profile_dipole
        
        r = np.linspace(0.1, 3.0, 20)
        theta = np.linspace(0, np.pi, 16)
        
        # Test multiple configurations
        configs = [
            {"R0": 2.0, "sigma": 1.0, "eps": 0.2, "name": "Standard dipole"},
            {"R0": 1.5, "sigma": 2.0, "eps": 0.0, "name": "Symmetric case"},
            {"R0": 2.5, "sigma": 0.5, "eps": 0.3, "name": "Strong dipole"}
        ]
        
        for config in configs:
            f_profile = alcubierre_profile_dipole(r, theta, **{k: v for k, v in config.items() if k != "name"})
            
            # Validate shape and finiteness
            assert f_profile.shape == (len(r), len(theta)), f"Wrong shape for {config['name']}"
            assert np.isfinite(f_profile).all(), f"Non-finite values in {config['name']}"
            
            # Check asymmetry for non-zero dipole
            if config["eps"] > 0:
                asymmetry = np.max(np.abs(f_profile[:, 0] - f_profile[:, -1]))
                assert asymmetry > 0.01, f"Insufficient asymmetry for {config['name']}"
                print(f"  ✓ {config['name']}: asymmetry = {asymmetry:.3f}")
            else:
                symmetry_error = np.max(np.abs(f_profile[:, 0] - f_profile[:, -1]))
                assert symmetry_error < 1e-10, f"Should be symmetric for {config['name']}"
                print(f"  ✓ {config['name']}: symmetric (error = {symmetry_error:.2e})")
        
        results["passed"] += 1
        print("  ✅ Dipolar profile generation: PASSED")
        
    except Exception as e:
        print(f"  ❌ Dipolar profile generation: FAILED ({e})")
    
    # Test 2: Momentum Flux Computation
    print("\nTest 2: Momentum Flux Computation")
    results["total"] += 1
    try:
        from stress_energy.exotic_matter_profile import ExoticMatterProfiler
        
        profiler = ExoticMatterProfiler(r_min=0.1, r_max=2.5, n_points=25)
        
        # Test with different dipole strengths
        eps_values = [0.0, 0.1, 0.2, 0.3]
        thrust_magnitudes = []
        
        for eps in eps_values:
            f_profile = alcubierre_profile_dipole(
                profiler.r_array, theta, R0=1.5, sigma=1.0, eps=eps
            )
            
            momentum_flux = profiler.compute_momentum_flux_vector(
                f_profile, profiler.r_array, theta
            )
            
            # Validate momentum flux
            assert len(momentum_flux) == 3, "Momentum flux must be 3D"
            assert np.isfinite(momentum_flux).all(), "All components must be finite"
            
            thrust_mag = np.linalg.norm(momentum_flux)
            thrust_magnitudes.append(thrust_mag)
            
            # For axisymmetric dipole, transverse components should be minimal
            if eps > 0:
                assert abs(momentum_flux[0]) < 1e-8, "X component should be minimal"
                assert abs(momentum_flux[1]) < 1e-8, "Y component should be minimal"
            
            print(f"  ε={eps:.1f}: F⃗=[{momentum_flux[0]:.2e}, {momentum_flux[1]:.2e}, {momentum_flux[2]:.2e}], |F⃗|={thrust_mag:.2e}")
        
        # Thrust should increase with dipole strength
        assert thrust_magnitudes[0] < thrust_magnitudes[-1], "Thrust should increase with dipole"
        
        results["passed"] += 1
        print("  ✅ Momentum flux computation: PASSED")
        
    except Exception as e:
        print(f"  ❌ Momentum flux computation: FAILED ({e})")
    
    # Test 3: Thrust Direction Analysis
    print("\nTest 3: Thrust Direction Analysis")
    results["total"] += 1
    try:
        # Test thrust analysis framework
        thrust_analysis = profiler.analyze_dipolar_thrust_characteristics(
            R0=1.5, sigma=1.0, eps_range=np.array([0.0, 0.1, 0.2, 0.3])
        )
        
        # Validate analysis structure
        required_keys = ['eps_values', 'thrust_magnitudes', 'optimal_dipole_strength', 'max_efficiency']
        for key in required_keys:
            assert key in thrust_analysis, f"Missing key: {key}"
        
        eps_vals = thrust_analysis['eps_values']
        thrust_mags = thrust_analysis['thrust_magnitudes']
        
        # Validate data consistency
        assert len(thrust_mags) == len(eps_vals), "Mismatched array lengths"
        assert all(np.isfinite(thrust_mags)), "All thrust values should be finite"
        assert thrust_mags[0] < thrust_mags[-1], "Thrust should increase with dipole"
        
        optimal_eps = thrust_analysis['optimal_dipole_strength']
        max_efficiency = thrust_analysis['max_efficiency']
        
        print(f"  Optimal dipole strength: ε* = {optimal_eps:.3f}")
        print(f"  Maximum efficiency: {max_efficiency:.2e}")
        
        results["passed"] += 1
        print("  ✅ Thrust direction analysis: PASSED")
        
    except Exception as e:
        print(f"  ❌ Thrust direction analysis: FAILED ({e})")
    
    return results

def validate_advanced_components():
    """Validate advanced steering components if available."""
    print("\n🎯 ADVANCED COMPONENTS VALIDATION")
    print("-" * 40)
    
    results = {"passed": 0, "total": 0}
    
    # Test advanced optimizer availability
    print("Test 4: Advanced Optimizer Integration")
    results["total"] += 1
    try:
        from coil_optimizer.advanced_coil_optimizer import AdvancedCoilOptimizer
        from stress_energy.exotic_matter_profile import ExoticMatterProfiler
        
        profiler = ExoticMatterProfiler(r_min=0.1, r_max=2.0, n_points=20)
        optimizer = AdvancedCoilOptimizer(r_min=0.1, r_max=2.0, n_points=20)
        optimizer.exotic_profiler = profiler
        
        # Set dummy target profile
        optimizer.set_target_profile(profiler.r_array, np.zeros_like(profiler.r_array))
        
        # Test steering penalty computation
        params = np.array([0.1, 1.5, 0.5, 0.2])  # [amplitude, center, width, dipole]
        direction = np.array([1.0, 0.0, 0.0])
        
        penalty = optimizer.steering_penalty(params, direction)
        assert np.isfinite(penalty), "Steering penalty should be finite"
        assert penalty <= 0, "Penalty should be negative (maximization)"
        
        print(f"  Steering penalty: {penalty:.2e}")
        
        # Test with different directions
        directions = [
            np.array([1.0, 0.0, 0.0]),  # X
            np.array([0.0, 1.0, 0.0]),  # Y  
            np.array([0.0, 0.0, 1.0])   # Z
        ]
        
        for i, dir_vec in enumerate(directions):
            penalty_i = optimizer.steering_penalty(params, dir_vec)
            print(f"  Direction {['X', 'Y', 'Z'][i]}: penalty = {penalty_i:.2e}")
        
        results["passed"] += 1
        print("  ✅ Advanced optimizer integration: PASSED")
        
    except ImportError:
        print("  ⚠️ Advanced optimizer: Not available (optional component)")
    except Exception as e:
        print(f"  ❌ Advanced optimizer integration: FAILED ({e})")
    
    # Test pipeline integration
    print("\nTest 5: Pipeline Integration")
    results["total"] += 1
    try:
        from run_unified_pipeline import UnifiedWarpFieldPipeline
        
        pipeline = UnifiedWarpFieldPipeline()
        
        # Test Step 12 (simplified)
        print("  Testing Step 12: Directional Profile Analysis")
        result12 = pipeline.step_12_directional_profile_analysis(eps=0.15)
        
        assert "dipole_strength" in result12, "Missing dipole_strength in result"
        assert "optimal_dipole_strength" in result12, "Missing optimal_dipole_strength"
        
        print(f"    Dipole strength: {result12['dipole_strength']:.3f}")
        print(f"    Optimal dipole: {result12['optimal_dipole_strength']:.3f}")
        
        results["passed"] += 1
        print("  ✅ Pipeline integration: PASSED")
        
    except ImportError:
        print("  ⚠️ Pipeline: Not fully available (components missing)")
    except Exception as e:
        print(f"  ❌ Pipeline integration: FAILED ({e})")
    
    return results

def main():
    """Run comprehensive validation."""
    print("🚀 COMPREHENSIVE STEERABLE DRIVE VALIDATION")
    print("=" * 50)
    print(f"Python version: {sys.version}")
    print(f"NumPy version: {np.__version__}")
    
    # Core functionality tests
    core_results = validate_core_functionality()
    
    # Advanced component tests
    advanced_results = validate_advanced_components()
    
    # Summary
    total_passed = core_results["passed"] + advanced_results["passed"]
    total_tests = core_results["total"] + advanced_results["total"]
    
    print("\n📊 VALIDATION SUMMARY")
    print("=" * 30)
    print(f"Core functionality: {core_results['passed']}/{core_results['total']} tests passed")
    print(f"Advanced components: {advanced_results['passed']}/{advanced_results['total']} tests passed")
    print(f"Overall: {total_passed}/{total_tests} tests passed ({100*total_passed/total_tests:.1f}%)")
    
    if core_results["passed"] == core_results["total"]:
        print("\n✅ CORE STEERABLE FUNCTIONALITY: FULLY OPERATIONAL")
        print("   - Dipolar warp profiles: ✓")
        print("   - Momentum flux computation: ✓")
        print("   - Thrust analysis: ✓")
        
        if advanced_results["passed"] > 0:
            print("✅ ADVANCED FEATURES: PARTIALLY OPERATIONAL")
            print("   - Optimization integration: ✓")
            
        print("\n🚀 STEERABLE WARP DRIVE READY FOR DEPLOYMENT!")
        return True
    else:
        print("\n❌ CORE FUNCTIONALITY ISSUES DETECTED")
        print("   System needs debugging before deployment")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
