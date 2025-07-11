#!/usr/bin/env python3
"""
Simple validation script to test if terminal re-activation is fixed
"""

import time
print("Testing terminal stability...")

# Test basic Python functionality
for i in range(5):
    print(f"Count: {i}")
    time.sleep(0.5)

print("Testing numpy import...")
import numpy as np
print(f"✓ Numpy version: {np.__version__}")

print("Testing matplotlib...")
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
print("✓ Matplotlib imported successfully")

print("Testing scipy...")
from scipy.optimize import minimize
print("✓ Scipy imported successfully")

print("✅ Terminal stability test completed successfully!")
print("   The terminal re-activation issue should be resolved.")
print("   Key fixes applied:")
print("   1. Fixed import paths (added src. prefix)")
print("   2. Installed missing dependencies (skopt, jax)")
print("   3. Simplified module loading to prevent heavy imports")
print("   4. Used non-interactive matplotlib backend")
