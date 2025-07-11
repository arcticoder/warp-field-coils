#!/usr/bin/env python3
"""
Simple test to explore the control library
"""

try:
    import control
    print(f"Control library version: {control.__version__}")
    print("Available functions:")
    funcs = [attr for attr in dir(control) if not attr.startswith('_')]
    for i, func in enumerate(funcs[:30]):  # Show first 30
        print(f"  {func}")
    
    # Check for transfer function creation
    print("\nTesting transfer function creation:")
    tf = control.tf([1], [1, 1])
    print(f"Transfer function created: {tf}")
    
    # Check for PID-related functions
    pid_funcs = [attr for attr in dir(control) if 'pid' in attr.lower()]
    print(f"\nPID-related functions: {pid_funcs}")
    
except Exception as e:
    print(f"Error: {e}")
