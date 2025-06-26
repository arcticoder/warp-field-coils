#!/usr/bin/env python3
"""
Quick Steerable Drive Validation
Simple test to verify core steerable functionality works
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

def main():
    print("🚀 QUICK STEERABLE DRIVE VALIDATION")
    print("=" * 45)
    
    try:
        # Test 1: Core dipolar profile
        print("📋 Test 1: Dipolar Profile Generation")
        from stress_energy.exotic_matter_profile import alcubierre_profile_dipole
        
        r = np.linspace(0.1, 3.0, 20)
        theta = np.linspace(0, np.pi, 16)
        
        f_dipolar = alcubierre_profile_dipole(r, theta, R0=2.0, sigma=1.0, eps=0.2)
        
        print(f"  Profile shape: {f_dipolar.shape}")
        print(f"  Profile range: [{np.min(f_dipolar):.3f}, {np.max(f_dipolar):.3f}]")
        
        # Test asymmetry
        asymmetry = np.max(np.abs(f_dipolar[:, 0] - f_dipolar[:, -1]))
        print(f"  Asymmetry: {asymmetry:.3f}")
        
        assert asymmetry > 0.1, "Should have significant asymmetry"
        print("  ✅ Dipolar profile working correctly!")
        
    except Exception as e:
        print(f"  ❌ Dipolar profile failed: {e}")
        return False
    
    try:
        # Test 2: Momentum flux computation
        print("\n📋 Test 2: Momentum Flux Computation")
        from stress_energy.exotic_matter_profile import ExoticMatterProfiler
        
        profiler = ExoticMatterProfiler(r_min=0.1, r_max=2.5, n_points=25)
        
        # Generate dipolar profile
        f_profile = alcubierre_profile_dipole(
            profiler.r_array, theta, R0=1.5, sigma=0.8, eps=0.15
        )
        
        # Compute momentum flux
        momentum_flux = profiler.compute_momentum_flux_vector(
            f_profile, profiler.r_array, theta
        )
        
        print(f"  Momentum flux: [{momentum_flux[0]:.2e}, {momentum_flux[1]:.2e}, {momentum_flux[2]:.2e}]")
        
        thrust_magnitude = np.linalg.norm(momentum_flux)
        print(f"  Thrust magnitude: {thrust_magnitude:.2e}")
        
        # For axisymmetric dipole, x and y components should be zero
        assert abs(momentum_flux[0]) < 1e-10, "X component should be zero"
        assert abs(momentum_flux[1]) < 1e-10, "Y component should be zero"
        assert thrust_magnitude > 1e-6, "Should have measurable thrust"
        
        print("  ✅ Momentum flux computation working!")
        
    except Exception as e:
        print(f"  ❌ Momentum flux failed: {e}")
        return False
    
    try:
        # Test 3: Thrust scaling analysis
        print("\n📋 Test 3: Thrust Scaling Analysis")
        
        eps_values = [0.0, 0.1, 0.2, 0.3]
        thrust_magnitudes = []
        
        for eps in eps_values:
            f_test = alcubierre_profile_dipole(
                profiler.r_array, theta, R0=1.5, sigma=1.0, eps=eps
            )
            
            momentum = profiler.compute_momentum_flux_vector(
                f_test, profiler.r_array, theta
            )
            
            thrust_mag = np.linalg.norm(momentum)
            thrust_magnitudes.append(thrust_mag)
            
            print(f"  ε={eps:.1f}: |F⃗|={thrust_mag:.2e}")
        
        # Thrust should increase with dipole strength
        assert thrust_magnitudes[0] < thrust_magnitudes[-1], "Thrust should increase with dipole"
        print("  ✅ Thrust scaling analysis working!")
        
    except Exception as e:
        print(f"  ❌ Thrust scaling failed: {e}")
        return False
    
    try:
        # Test 4: Advanced optimization components (optional)
        print("\n📋 Test 4: Advanced Components (Optional)")
        
        try:
            from coil_optimizer.advanced_coil_optimizer import AdvancedCoilOptimizer
            print("  ✅ Advanced coil optimizer: Available")
            
            # Quick test of steering penalty
            optimizer = AdvancedCoilOptimizer(r_min=0.1, r_max=2.0, n_points=20)
            optimizer.exotic_profiler = profiler
            
            # Set dummy target
            optimizer.set_target_profile(profiler.r_array, np.zeros_like(profiler.r_array))
            
            params = np.array([0.1, 1.5, 0.5, 0.2])
            direction = np.array([1.0, 0.0, 0.0])
            
            penalty = optimizer.steering_penalty(params, direction)
            print(f"  Steering penalty test: {penalty:.2e}")
            print("  ✅ Advanced optimization components working!")
            
        except ImportError:
            print("  ⚠️ Advanced coil optimizer: Not available (expected)")
        except Exception as e:
            print(f"  ⚠️ Advanced components failed: {e}")
    
    except Exception as e:
        print(f"  ❌ Advanced component test failed: {e}")
    
    print("\n🎉 STEERABLE DRIVE VALIDATION COMPLETE!")
    print("=" * 45)
    print("✅ Core dipolar functionality: Working")
    print("✅ Momentum flux computation: Working") 
    print("✅ Thrust scaling analysis: Working")
    print("✅ System ready for steerable operations!")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
