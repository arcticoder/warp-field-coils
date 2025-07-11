"""
Test Enhanced Control Integration in Pipeline
===========================================

Quick test of the enhanced IDF and SIF systems integrated into the unified pipeline.
"""

import sys
import os
import numpy as np

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, 'src'))
sys.path.insert(0, current_dir)

def test_enhanced_pipeline_integration():
    """Test enhanced control systems in pipeline"""
    print("🔧 Testing Enhanced Control Pipeline Integration")
    print("=" * 55)
    
    try:
        from run_unified_pipeline import UnifiedWarpFieldPipeline
        
        # Test configuration
        test_config = {
            'alpha_max': 1e-4 * 9.81,
            'j_max': 2.0,
            'lambda_coupling': 1e-2,
            'rho_eff': 0.5,
            'max_acceleration': 5.0,
            'material_coupling': 0.8,
            'ricci_coupling': 1e-2,
            'weyl_coupling': 1.2,
            'stress_max': 1e-6
        }
        
        # Initialize pipeline
        print("Test 1: Pipeline Initialization")
        pipeline = UnifiedWarpFieldPipeline()
        pipeline.config.update(test_config)
        
        # Re-initialize enhanced systems with updated config
        try:
            from src.control.enhanced_inertial_damper import EnhancedInertialDamperField
            from src.control.enhanced_structural_integrity import EnhancedStructuralIntegrityField
            
            pipeline.enhanced_idf = EnhancedInertialDamperField(
                alpha_max=test_config['alpha_max'],
                j_max=test_config['j_max'],
                lambda_coupling=test_config['lambda_coupling'],
                effective_density=test_config['rho_eff'],
                a_max=test_config['max_acceleration']
            )
            
            pipeline.enhanced_sif = EnhancedStructuralIntegrityField(
                material_coupling=test_config['material_coupling'],
                ricci_coupling=test_config['ricci_coupling'],
                weyl_coupling=test_config['weyl_coupling'],
                stress_max=test_config['stress_max']
            )
            
            pipeline.enhanced_control_available = True
            print("✅ Pipeline with enhanced control systems initialized")
            
        except Exception as e:
            print(f"⚠️ Enhanced systems initialization failed: {e}")
            return False
        
        # Test enhanced control integration step
        print("\nTest 2: Enhanced Control Integration Step")
        
        # Create simple test profiles
        simulation_time = 2.0  # Short test
        dt = 0.01
        t = np.arange(0, simulation_time, dt)
        
        # Simple jerk profile
        jerk_profile = np.zeros((len(t), 3))
        jerk_profile[:, 0] = 0.5 * np.sin(2 * np.pi * t)  # 1 Hz oscillation
        jerk_profile[:, 1] = 0.3 * np.cos(3 * np.pi * t)  # 1.5 Hz oscillation
        
        # Simple material stress profile
        material_stress_profile = []
        for i in range(len(t)):
            stress = np.array([[1e-8, 1e-9, 0],
                              [1e-9, 2e-8, 1e-9],
                              [0, 1e-9, 1.5e-8]])
            material_stress_profile.append(stress)
        
        # Run enhanced control integration
        try:
            results = pipeline.step_14_enhanced_control_integration(
                jerk_profile=jerk_profile,
                material_stress_profile=material_stress_profile,
                simulation_time=simulation_time
            )
            
            if results['success']:
                print("✅ Enhanced control integration successful")
                
                # Display results summary
                analysis = results['analysis']
                idf_summary = analysis['idf_summary']
                sif_summary = analysis['sif_summary']
                
                print(f"   IDF average acceleration: {idf_summary['avg_acceleration']:.3f} m/s²")
                print(f"   IDF max acceleration: {idf_summary['max_acceleration']:.3f} m/s²")
                print(f"   IDF safety events: {idf_summary['safety_events']}")
                
                print(f"   SIF average compensation: {sif_summary['avg_compensation']:.2e} N/m²")
                print(f"   SIF max compensation: {sif_summary['max_compensation']:.2e} N/m²")
                print(f"   SIF safety events: {sif_summary['safety_events']}")
                
                # System health
                idf_health = results['system_diagnostics']['idf_diagnostics']['overall_health']
                sif_health = results['system_diagnostics']['sif_diagnostics']['overall_health']
                
                print(f"   IDF system health: {idf_health}")
                print(f"   SIF system health: {sif_health}")
                
                return True
            else:
                print(f"❌ Enhanced control integration failed: {results.get('message', 'Unknown error')}")
                return False
                
        except Exception as e:
            print(f"❌ Enhanced control step failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
    except Exception as e:
        print(f"❌ Pipeline initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run enhanced pipeline integration test"""
    print("🧪 Enhanced Control Pipeline Integration Test")
    print("=" * 50)
    
    success = test_enhanced_pipeline_integration()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 Enhanced control pipeline integration successful!")
        print("🚀 Ready for full system deployment!")
    else:
        print("⚠️ Enhanced control pipeline integration had issues")
        print("🔧 Review logs and fix issues before deployment")
    
    return success

if __name__ == "__main__":
    success = main()
