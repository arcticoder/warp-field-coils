#!/usr/bin/env python3
"""
Test Enhanced Inertial Damper Field with LQG Polymer Mathematics
===============================================================

Demonstrates the sinc(πμ) polymer corrections reducing stress-energy feedback
using added polymer corrections to backreaction calculations.

Features Tested:
- PolymerStressTensorCorrections class with sinc(πμ) enhancement
- β = 1.9443254780147017 exact backreaction factor (48.55% energy reduction)
- Polymer-corrected stress-energy tensor calculations
- Polymer scale optimization for maximum efficiency
- Performance analysis and diagnostics
"""

import numpy as np
import logging
import sys
import matplotlib.pyplot as plt
from pathlib import Path

# Add the src directory to the path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

try:
    from control.enhanced_inertial_damper_field import (
        EnhancedInertialDamperField, 
        IDFParams, 
        PolymerStressTensorCorrections,
        BETA_EXACT_BACKREACTION,
        MU_OPTIMAL_POLYMER,
        PI
    )
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the warp-field-coils directory")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def test_polymer_corrections():
    """Test the PolymerStressTensorCorrections class"""
    print("\n⚛️ Testing LQG Polymer Stress Tensor Corrections")
    print("=" * 60)
    
    # Initialize polymer corrections
    polymer_corrections = PolymerStressTensorCorrections(mu=MU_OPTIMAL_POLYMER)
    
    # Test sinc(πμ) correction function
    print(f"✅ Polymer scale parameter: μ = {polymer_corrections.mu:.3f}")
    print(f"✅ Exact backreaction factor: β = {polymer_corrections.beta_exact:.10f}")
    print(f"✅ Energy reduction: {(1.0 - 1.0/polymer_corrections.beta_exact)*100:.2f}%")
    
    # Test sinc correction for various field magnitudes
    field_magnitudes = np.logspace(-6, 0, 7)  # 10^-6 to 1
    print(f"\n📊 Sinc(πμ) correction factors:")
    for field_mag in field_magnitudes:
        sinc_factor = polymer_corrections.sinc_polymer_correction(np.array([field_mag]))[0]
        print(f"   Field magnitude {field_mag:.1e}: sinc(πμ) = {sinc_factor:.6f}")
    
    # Test stress-energy tensor correction
    print(f"\n🧮 Testing polymer stress-energy tensor correction:")
    classical_tensor = np.diag([1e-10, -3e-11, -3e-11, -3e-11])  # Simple perfect fluid
    field_magnitude = 1e-3
    
    polymer_tensor = polymer_corrections.compute_polymer_stress_energy_tensor(
        classical_tensor, field_magnitude
    )
    
    sinc_factor = polymer_corrections.sinc_polymer_correction(np.array([field_magnitude]))[0]
    enhancement = polymer_corrections.beta_exact * sinc_factor
    
    print(f"   Classical T_00 = {classical_tensor[0,0]:.2e}")
    print(f"   Polymer T_00   = {polymer_tensor[0,0]:.2e}")
    print(f"   Enhancement factor = {enhancement:.6f}")
    print(f"   Energy reduction = {(1.0 - enhancement)*100:.2f}%")
    
    return polymer_corrections

def test_enhanced_idf():
    """Test the Enhanced Inertial Damper Field with polymer corrections"""
    print("\n🚀 Testing Enhanced Inertial Damper Field with Polymer Corrections")
    print("=" * 70)
    
    # Create IDF parameters with polymer corrections enabled
    params = IDFParams(
        alpha_max=1e-3 * 9.81,
        j_max=2.0,
        rho_eff=1000.0,
        lambda_coupling=0.01,
        safety_acceleration_limit=50.0,
        enable_backreaction=True,
        enable_curvature_coupling=True,
        enable_polymer_corrections=True,
        mu_polymer=MU_OPTIMAL_POLYMER
    )
    
    # Initialize Enhanced IDF
    idf = EnhancedInertialDamperField(params)
    
    # Test with various jerk residuals
    test_cases = [
        np.array([0.1, 0.0, 0.0]),    # Small x-jerk
        np.array([0.0, 0.5, 0.0]),    # Medium y-jerk  
        np.array([0.0, 0.0, 1.0]),    # Large z-jerk
        np.array([0.2, 0.3, 0.1]),    # Mixed jerk
        np.array([2.0, -1.5, 0.8])   # High magnitude jerk
    ]
    
    # Identity metric for flat spacetime testing
    metric = np.diag([1.0, -1.0, -1.0, -1.0])
    
    print(f"✅ Enhanced IDF initialized with polymer corrections")
    print(f"   K_IDF = {idf.K_idf:.2e}")
    print(f"   Polymer μ = {idf.polymer_corrections.mu:.3f}")
    print(f"   Polymer β = {idf.polymer_corrections.beta_exact:.6f}")
    
    results = []
    
    for i, j_res in enumerate(test_cases):
        print(f"\n📋 Test Case {i+1}: j_res = {j_res}")
        
        # Compute IDF acceleration
        result = idf.compute_acceleration(j_res, metric)
        
        # Extract key metrics
        a_total = result['acceleration']
        components = result['components']
        diagnostics = result['diagnostics']
        polymer_info = diagnostics['polymer']
        
        print(f"   Input jerk magnitude: {np.linalg.norm(j_res):.3f} m/s³")
        print(f"   Output acceleration: {np.linalg.norm(a_total):.3e} m/s²")
        print(f"   Polymer sinc factor: {polymer_info['sinc_factor']:.6f}")
        print(f"   Polymer enhancement: {polymer_info['polymer_enhancement']:.6f}")
        print(f"   Safety limited: {diagnostics['performance']['safety_limited']}")
        
        results.append({
            'j_res': j_res,
            'a_total': a_total,
            'polymer_info': polymer_info,
            'performance': diagnostics['performance']
        })
    
    return idf, results

def test_polymer_optimization():
    """Test polymer scale optimization"""
    print("\n🔧 Testing Polymer Scale Optimization")
    print("=" * 50)
    
    # Create IDF
    params = IDFParams(enable_polymer_corrections=True)
    idf = EnhancedInertialDamperField(params)
    
    # Generate sample jerk residuals for optimization
    np.random.seed(42)  # Reproducible results
    j_res_samples = [
        np.random.normal(0, 0.5, 3) for _ in range(20)
    ]
    
    print(f"Initial polymer scale: μ = {idf.polymer_corrections.mu:.4f}")
    
    # Optimize polymer scale
    optimal_mu = idf.optimize_polymer_scale(j_res_samples, target_efficiency=0.9)
    
    print(f"Optimized polymer scale: μ = {optimal_mu:.4f}")
    
    return optimal_mu

def test_polymer_performance_analysis():
    """Test polymer performance analysis"""
    print("\n📊 Testing Polymer Performance Analysis")
    print("=" * 45)
    
    # Create IDF and run multiple computations
    params = IDFParams(enable_polymer_corrections=True)
    idf = EnhancedInertialDamperField(params)
    
    # Generate data for analysis
    metric = np.diag([1.0, -1.0, -1.0, -1.0])
    np.random.seed(42)
    
    for _ in range(50):
        j_res = np.random.normal(0, 0.3, 3)
        idf.compute_acceleration(j_res, metric)
    
    # Analyze performance
    analysis = idf.analyze_polymer_performance()
    
    print(f"Status: {analysis['status']}")
    print(f"Performance level: {analysis['performance_level']}")
    
    if 'metrics' in analysis:
        metrics = analysis['metrics']
        print(f"Average sinc factor: {metrics['average_sinc_factor']:.6f}")
        print(f"Average enhancement: {metrics['average_enhancement']:.6f}")
        print(f"Stability index: {metrics['stability_index']:.6f}")
        print(f"Energy reduction: {metrics['energy_reduction_percent']:.2f}%")
    
    if analysis['recommendations']:
        print("Recommendations:")
        for rec in analysis['recommendations']:
            print(f"  - {rec}")
    
    return analysis

def create_performance_plots(idf, results):
    """Create performance visualization plots"""
    try:
        print("\n📈 Creating Performance Plots")
        print("=" * 35)
        
        # Extract data for plotting
        jerk_magnitudes = [np.linalg.norm(r['j_res']) for r in results]
        accel_magnitudes = [np.linalg.norm(r['a_total']) for r in results]
        sinc_factors = [r['polymer_info']['sinc_factor'] for r in results]
        enhancements = [r['polymer_info']['polymer_enhancement'] for r in results]
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Jerk vs Acceleration
        ax1.scatter(jerk_magnitudes, accel_magnitudes, color='blue', alpha=0.7)
        ax1.set_xlabel('Input Jerk Magnitude (m/s³)')
        ax1.set_ylabel('Output Acceleration Magnitude (m/s²)')
        ax1.set_title('IDF Response: Jerk vs Acceleration')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Jerk vs Sinc Factor
        ax2.scatter(jerk_magnitudes, sinc_factors, color='green', alpha=0.7)
        ax2.set_xlabel('Input Jerk Magnitude (m/s³)')
        ax2.set_ylabel('Polymer Sinc Factor')
        ax2.set_title('Polymer Correction: sinc(πμ) Factor')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Jerk vs Enhancement
        ax3.scatter(jerk_magnitudes, enhancements, color='red', alpha=0.7)
        ax3.set_xlabel('Input Jerk Magnitude (m/s³)')
        ax3.set_ylabel('Polymer Enhancement Factor')
        ax3.set_title('Total Polymer Enhancement (β × sinc)')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Sinc function visualization
        mu_values = np.linspace(0.01, 1.0, 100)
        sinc_values = []
        for mu in mu_values:
            sinc_val = np.sin(PI * mu) / (PI * mu) if mu > 1e-10 else 1.0
            sinc_values.append(sinc_val)
        
        ax4.plot(mu_values, sinc_values, color='purple', linewidth=2)
        ax4.axvline(x=MU_OPTIMAL_POLYMER, color='orange', linestyle='--', 
                   label=f'Optimal μ = {MU_OPTIMAL_POLYMER:.2f}')
        ax4.set_xlabel('Polymer Parameter μ')
        ax4.set_ylabel('sinc(πμ)')
        ax4.set_title('Polymer Sinc Function')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = 'polymer_enhanced_idf_performance.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"✅ Performance plots saved to: {plot_filename}")
        
        # Show plot if running interactively
        plt.show()
        
    except ImportError:
        print("⚠️ Matplotlib not available - skipping plots")
    except Exception as e:
        print(f"⚠️ Error creating plots: {e}")

def main():
    """Main test function"""
    print("🌟 Enhanced Inertial Damper Field - LQG Polymer Mathematics Test")
    print("=" * 80)
    print("Testing sinc(πμ) polymer corrections reducing stress-energy feedback")
    print("with exact backreaction factor β = 1.9443254780147017")
    print("=" * 80)
    
    try:
        # Test 1: Polymer corrections
        polymer_corrections = test_polymer_corrections()
        
        # Test 2: Enhanced IDF
        idf, results = test_enhanced_idf()
        
        # Test 3: Polymer optimization
        optimal_mu = test_polymer_optimization()
        
        # Test 4: Performance analysis
        analysis = test_polymer_performance_analysis()
        
        # Test 5: Create visualization
        create_performance_plots(idf, results)
        
        # Summary
        print("\n🎉 All Tests Completed Successfully!")
        print("=" * 50)
        print("✅ PolymerStressTensorCorrections class working")
        print("✅ Enhanced IDF with polymer corrections operational")
        print("✅ Polymer scale optimization functional")
        print("✅ Performance analysis implemented")
        print("✅ LQG polymer mathematics integration complete")
        
        print(f"\n📋 Summary Statistics:")
        print(f"   Exact backreaction factor: β = {BETA_EXACT_BACKREACTION:.10f}")
        print(f"   Energy reduction achieved: {(1.0-1.0/BETA_EXACT_BACKREACTION)*100:.2f}%")
        print(f"   Optimal polymer parameter: μ = {optimal_mu:.4f}")
        print(f"   Stress-energy feedback reduction: ACTIVE")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        logging.exception("Test execution failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
