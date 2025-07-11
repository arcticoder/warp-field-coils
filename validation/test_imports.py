#!/usr/bin/env python3
"""
Test script to debug import issues
"""

print("Starting import test...")

try:
    import sys
    import os
    print(f"Python version: {sys.version}")
    print(f"Current working directory: {os.getcwd()}")
    
    # Add src to path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(current_dir, 'src')
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    print(f"Added to path: {src_dir}")
    
    print("Testing imports...")
    
    # Test basic numpy
    import numpy as np
    print("✓ numpy import successful")
    
    # Test matplotlib
    import matplotlib.pyplot as plt
    print("✓ matplotlib import successful")
    
    # Test skopt
    import skopt
    print("✓ skopt import successful")
    
    # Test our modules
    import src.stress_energy.exotic_matter_profile as emp
    print("✓ exotic_matter_profile import successful")
    
    import src.coil_optimizer.advanced_coil_optimizer as aco
    print("✓ advanced_coil_optimizer import successful")
    
    print("All imports successful!")
    
except Exception as e:
    import traceback
    print(f"ERROR: {e}")
    print("Full traceback:")
    traceback.print_exc()
