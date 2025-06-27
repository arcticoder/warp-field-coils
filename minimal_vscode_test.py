#!/usr/bin/env python3
"""
Minimal test to isolate VS Code restart issue
"""

print("=== MINIMAL VS CODE STABILITY TEST ===")
print("Testing if specific operations cause VS Code crashes...")

# Test 1: Basic Python execution
print("✓ Test 1: Basic Python execution - OK")

# Test 2: Small memory allocation
import sys
print(f"✓ Test 2: Python version {sys.version_info.major}.{sys.version_info.minor} - OK")

# Test 3: File operations
import os
print(f"✓ Test 3: Current directory: {os.getcwd()}")

# Test 4: Module imports (most likely culprit)
try:
    import numpy as np
    print("✓ Test 4a: NumPy import - OK")
except Exception as e:
    print(f"❌ Test 4a: NumPy import failed: {e}")

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    print("✓ Test 4b: Matplotlib import - OK")
except Exception as e:
    print(f"❌ Test 4b: Matplotlib import failed: {e}")

# Test 5: Large data structures (potential memory issue)
try:
    large_array = list(range(10000))
    print("✓ Test 5: Large array creation - OK")
    del large_array
except Exception as e:
    print(f"❌ Test 5: Large array creation failed: {e}")

print("=== TEST COMPLETE ===")
print("If VS Code restarted during this test, the issue is in the basic operations.")
print("If not, the issue is specific to the complex pipeline operations.")
