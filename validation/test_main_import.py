#!/usr/bin/env python3
"""
Test script to debug the main pipeline import
"""

print("Testing main pipeline import...")

try:
    import sys
    import os
    
    # Add src to path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(current_dir, 'src')
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    
    print("Step 1: Testing basic imports...")
    import numpy as np
    import matplotlib.pyplot as plt
    print("✓ Basic imports OK")
    
    print("Step 2: Testing core modules...")
    from src.stress_energy.exotic_matter_profile import ExoticMatterProfiler
    print("✓ ExoticMatterProfiler import OK")
    
    from src.coil_optimizer.advanced_coil_optimizer import AdvancedCoilOptimizer
    print("✓ AdvancedCoilOptimizer import OK")
    
    print("Step 3: Testing main pipeline import...")
    from run_unified_pipeline import UnifiedWarpFieldPipeline
    print("✓ UnifiedWarpFieldPipeline import OK")
    
    print("All imports successful!")
    
except Exception as e:
    import traceback
    print(f"ERROR: {e}")
    print("Full traceback:")
    traceback.print_exc()
