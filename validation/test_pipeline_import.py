#!/usr/bin/env python3
"""Test pipeline import"""

try:
    from run_unified_pipeline import UnifiedWarpFieldPipeline
    print("✅ SUCCESS: Pipeline import worked")
except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
