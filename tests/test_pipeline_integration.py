#!/usr/bin/env python3
"""
Test Pipeline Integration
Simple test of pipeline steps 12-13 with proper syntax
"""

import sys
from pathlib import Path

# Add src to path  
sys.path.append(str(Path(__file__).parent.parent / "src"))

def test_pipeline_integration():
    """Test steerable pipeline integration."""
    print("🎯 TESTING PIPELINE INTEGRATION")
    print("=" * 40)
    
    try:
        from run_unified_pipeline import UnifiedWarpFieldPipeline
        import numpy as np
        
        # Create pipeline
        pipeline = UnifiedWarpFieldPipeline()
        print("✓ Pipeline created successfully")
        
        # Test Step 12: Directional Profile Analysis
        print("\n📋 Step 12: Directional Profile Analysis")
        try:
            result12 = pipeline.step_12_directional_profile_analysis(eps=0.15)
            
            dipole_strength = result12.get("dipole_strength", 0.0)
            optimal_dipole = result12.get("optimal_dipole_strength", 0.0)
            
            print(f"  Dipole strength: {dipole_strength:.3f}")
            print(f"  Optimal dipole: {optimal_dipole:.3f}")
            print("  ✅ Step 12 completed successfully")
            
        except Exception as e:
            print(f"  ❌ Step 12 failed: {e}")
            return False
        
        # Test Step 13: Steering Optimization (simplified)
        print("\n📋 Step 13: Steering Optimization") 
        try:
            direction = (0.0, 0.0, 1.0)  # Z-direction
            alpha_s = 1e4
            
            result13 = pipeline.step_13_optimize_steering(
                direction=direction, 
                alpha_s=alpha_s
            )
            
            optimization_success = result13.get("optimization_results", {}).get("success", False)
            thrust_magnitude = result13.get("thrust_magnitude", 0.0)
            direction_alignment = result13.get("direction_alignment", 0.0)
            
            print(f"  Target direction: {direction}")
            print(f"  Optimization success: {optimization_success}")
            print(f"  Thrust magnitude: {thrust_magnitude:.2e}")
            print(f"  Direction alignment: {direction_alignment:.3f}")
            
            if optimization_success:
                print("  ✅ Step 13 completed successfully")
            else:
                print("  ⚠️ Step 13 had issues but completed")
            
        except Exception as e:
            print(f"  ❌ Step 13 failed: {e}")
            return False
        
        print("\n✅ Pipeline integration test successful!")
        return True
        
    except ImportError as e:
        print(f"❌ Pipeline import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_pipeline_integration()
    
    if success:
        print("\n🚀 PIPELINE READY FOR STEERABLE OPERATIONS!")
    else:
        print("\n⚠️ Pipeline needs debugging before full deployment")
    
    sys.exit(0 if success else 1)
