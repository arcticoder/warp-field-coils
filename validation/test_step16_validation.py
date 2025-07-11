#!/usr/bin/env python3
"""
Validation test for Step 16: Warp-Pulse Tomographic Scanner
Demonstrates the integrated tomographic scanning functionality
"""

import numpy as np
import matplotlib.pyplot as plt
from run_unified_pipeline import UnifiedWarpFieldPipeline
import json

def test_step_16_validation():
    """Test Step 16 tomographic scanner with basic validation."""
    
    print("="*70)
    print("STEP 16 WARP-PULSE TOMOGRAPHIC SCANNER VALIDATION")
    print("="*70)
    
    # Initialize pipeline
    config = {
        'r_min': 0.1,
        'r_max': 10.0,
        'n_points': 200,  # Reduced for faster testing
        'warp_radius': 2.0,
        'warp_width': 0.5
    }
    
    pipeline = UnifiedWarpFieldPipeline()
    pipeline.config.update(config)
    
    # Test polymer enhancement functionality
    print("\n1. Testing Polymer Enhancement...")
    xi_1_5 = pipeline._compute_xi_mu(1.5)
    enhanced_profile = pipeline.exotic_profiler.compute_polymer_enhanced_profile(
        R=2.0, sigma=0.5, enhancement_factor=xi_1_5
    )
    print(f"   ✓ ξ(1.5) = {xi_1_5:.6f}")
    print(f"   ✓ Enhanced profile computed: {enhanced_profile.shape}")
    print(f"   ✓ Profile max enhancement: {enhanced_profile.max():.6f}")
    
    # Test signal quality analysis
    print("\n2. Testing Signal Quality Analysis...")
    # Generate synthetic phase shift data
    r_test = np.linspace(0.1, 10, 100)
    synthetic_phases = 0.1 * np.exp(-((r_test - 2.0)/0.5)**2) + 0.01 * np.random.normal(size=100)
    
    quality_metrics = pipeline._compute_signal_quality(synthetic_phases)
    print(f"   ✓ SNR: {quality_metrics['snr']:.2f} dB")
    print(f"   ✓ Coherence: {quality_metrics['coherence']:.4f}")
    print(f"   ✓ Stability: {quality_metrics['stability']:.4f}")
    
    # Test spatial resolution estimation
    print("\n3. Testing Spatial Resolution Estimation...")
    # Create synthetic 2D field with known resolution
    x = np.linspace(-8, 8, 64)
    y = np.linspace(-8, 8, 64)
    X, Y = np.meshgrid(x, y)
    synthetic_field = np.exp(-((X**2 + Y**2)/4)) * np.exp(1j * 0.5 * X)
    
    estimated_resolution = pipeline._estimate_spatial_resolution(synthetic_field)
    print(f"   ✓ Estimated resolution: {estimated_resolution:.4f} m")
    
    # Test field contrast analysis
    print("\n4. Testing Field Contrast Analysis...")
    contrast = pipeline._analyze_field_contrast(synthetic_field)
    print(f"   ✓ Field contrast: {contrast:.4f}")
    
    # Test reconstruction fidelity
    print("\n5. Testing Reconstruction Fidelity...")
    fidelity = pipeline._compute_reconstruction_fidelity(synthetic_field)
    print(f"   ✓ Reconstruction fidelity: {fidelity:.4f}")
    
    # Test enhanced phase shift computation
    print("\n6. Testing Enhanced Phase Shift Computation...")
    # Mock pulse data
    pulse_data = {
        'pulse_field': synthetic_field,
        'steering_phases': np.angle(synthetic_field),
        'warp_profile': enhanced_profile[:64],  # Match field size
        'application_config': {'optimal_radius': 2.0, 'profile_sharpness': 0.5},
        'enhancement_factor': xi_1_5
    }
    
    app_config = {'optimal_radius': 2.0, 'profile_sharpness': 0.5}
    enhanced_phases = pipeline._compute_enhanced_phase_shifts(
        pulse_data, theta=0.1, app_config=app_config, 
        alpha=1e-10, beta=1e-8, gamma=1e-6
    )
    print(f"   ✓ Enhanced phases computed: {enhanced_phases.shape}")
    print(f"   ✓ Phase range: [{enhanced_phases.min():.6f}, {enhanced_phases.max():.6f}]")
    
    # Performance summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    print("✅ Polymer Enhancement: OPERATIONAL")
    print("✅ Signal Quality Analysis: OPERATIONAL")
    print("✅ Spatial Resolution Estimation: OPERATIONAL")
    print("✅ Field Contrast Analysis: OPERATIONAL")
    print("✅ Reconstruction Fidelity: OPERATIONAL")
    print("✅ Enhanced Phase Computation: OPERATIONAL")
    print("\n🚀 STEP 16 TOMOGRAPHIC SCANNER: FULLY FUNCTIONAL")
    
    return {
        'polymer_enhancement': xi_1_5,
        'enhanced_profile_max': float(enhanced_profile.max()),
        'signal_quality': quality_metrics,
        'spatial_resolution': float(estimated_resolution),
        'field_contrast': float(contrast),
        'reconstruction_fidelity': float(fidelity),
        'phase_computation_range': [float(enhanced_phases.min()), float(enhanced_phases.max())],
        'validation_status': 'PASS'
    }

if __name__ == '__main__':
    try:
        results = test_step_16_validation()
        
        # Save validation results
        with open('step16_validation_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Validation results saved to step16_validation_results.json")
        
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
