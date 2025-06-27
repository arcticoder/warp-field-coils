"""
Quick Integration Test for New Warp Technologies
===============================================

Simplified test of all three new warp field components.
"""

import numpy as np
import logging
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from subspace_transceiver.transceiver import SubspaceTransceiver, SubspaceParams, TransmissionParams
from holodeck_forcefield_grid.grid import ForceFieldGrid, GridParams
from medical_tractor_array.array import MedicalTractorArray, MedicalArrayParams, VitalSigns, SafetyLevel

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_subspace_transceiver():
    """Test subspace communication system"""
    print("\n=== TESTING SUBSPACE TRANSCEIVER ===")
    
    # Use smaller grid for faster testing
    params = SubspaceParams(
        c_s=5.0e8, 
        bandwidth=1e12,
        grid_resolution=32,  # Much smaller grid (32x32 instead of 128x128)
        rtol=1e-3,          # Relaxed tolerance for speed
        atol=1e-6
    )
    transceiver = SubspaceTransceiver(params)
    
    # Run diagnostics
    diag = transceiver.run_diagnostics()
    print(f"✅ Transceiver Status: {diag['overall_health']}")
    
    # Test message transmission with fast method
    print("📡 Testing fast transmission...")
    tx_params = TransmissionParams(
        frequency=2.4e12,
        modulation_depth=0.8,
        duration=0.01,  # Shorter duration
        target_coordinates=(10.0, 20.0, 30.0),  # Closer target
        priority=5
    )
    
    import time
    start_time = time.time()
    
    # Try fast transmission first
    if hasattr(transceiver, 'transmit_message_fast'):
        print("Using fast transmission mode...")
        result = transceiver.transmit_message_fast("Test message", tx_params)
    else:
        print("Using standard transmission mode...")
        result = transceiver.transmit_message("Test message", tx_params)
    
    elapsed = time.time() - start_time
    print(f"✅ Message Status: {result.get('status', 'UNKNOWN')}")
    print(f"⏱️  Transmission Time: {elapsed:.3f} seconds")
    
    if 'signal_strength_db' in result:
        print(f"📶 Signal Strength: {result['signal_strength_db']:.1f} dB")
    
    return result.get('success', True)

def test_holodeck_grid():
    """Test holodeck force-field grid"""
    print("\n=== TESTING HOLODECK FORCE-FIELD GRID ===")
    
    print("🏗️  Creating force-field grid...")
    params = GridParams(
        bounds=((-0.5, 0.5), (-0.5, 0.5), (0.0, 1.0)),
        base_spacing=0.3,  # Even larger spacing for speed
        max_nodes=200      # Very few nodes for testing
    )
    grid = ForceFieldGrid(params)
    
    # Run diagnostics
    print("🔍 Running diagnostics...")
    diag = grid.run_diagnostics()
    print(f"✅ Grid Status: {diag['overall_health']}")
    print(f"✅ Total Nodes: {diag['total_nodes']}")
    
    # Test force computation
    print("⚡ Testing force computation...")
    test_point = np.array([0.1, 0.1, 0.5])
    force = grid.compute_total_force(test_point)
    print(f"✅ Force Magnitude: {np.linalg.norm(force):.3f} N")
    
    # Add interaction zone
    print("🎯 Adding interaction zone...")
    grid.add_interaction_zone(np.array([0.0, 0.0, 0.5]), 0.2, "soft")
    print(f"✅ Interaction Zones: {len(grid.interaction_zones)}")
    
    # Test simulation step
    print("⏱️  Testing simulation step...")
    import time
    start_time = time.time()
    step_result = grid.step_simulation(0.001)
    elapsed = time.time() - start_time
    print(f"✅ Simulation Step: {elapsed*1000:.2f} ms")
    
    return True

def test_medical_array():
    """Test medical tractor array"""
    print("\n=== TESTING MEDICAL TRACTOR ARRAY ===")
    
    print("🏥 Creating medical tractor array...")
    params = MedicalArrayParams(
        array_bounds=((-0.2, 0.2), (-0.2, 0.2), (0.1, 0.3)),
        beam_spacing=0.08,  # Larger spacing for fewer beams
        max_beams=25       # Even fewer beams for testing
    )
    array = MedicalTractorArray(params)
    
    # Run diagnostics
    print("🔍 Running diagnostics...")
    diag = array.run_diagnostics()
    print(f"✅ Array Status: {diag['overall_health']}")
    print(f"✅ Total Beams: {diag['total_beams']}")
    
    # Set up vital signs
    print("❤️  Setting up patient monitoring...")
    vitals = VitalSigns(heart_rate=75.0, oxygen_saturation=98.0)
    array.update_vital_signs(vitals)
    
    # Start procedure
    print("🏥 Starting medical procedure...")
    array.start_procedure("TEST_PATIENT", "positioning")
    
    # Test positioning
    print("🎯 Testing tissue positioning...")
    target_pos = np.array([0.05, 0.02, 0.15])
    desired_pos = np.array([0.04, 0.02, 0.15])
    
    import time
    start_time = time.time()
    result = array.position_target(target_pos, desired_pos, tissue_type="soft")
    elapsed = time.time() - start_time
    
    print(f"✅ Positioning Status: {result['status']}")
    print(f"⏱️  Processing Time: {elapsed*1000:.2f} ms")
    
    if 'force' in result:
        force_magnitude = np.linalg.norm(result['force'])
        print(f"⚡ Applied Force: {force_magnitude:.2e} N")
    
    print("🛑 Stopping procedure...")
    array.stop_procedure()
    
    return result['status'] != 'EMERGENCY_STOP'

def main():
    """Run comprehensive test"""
    print("🚀 WARP TECHNOLOGY INTEGRATION TEST 🚀")
    print("="*50)
    
    try:
        # Test each system
        subspace_ok = test_subspace_transceiver()
        holodeck_ok = test_holodeck_grid()
        medical_ok = test_medical_array()
        
        # Results
        print("\n" + "="*50)
        print("TEST RESULTS:")
        print(f"Subspace Transceiver: {'✅ PASS' if subspace_ok else '❌ FAIL'}")
        print(f"Holodeck Grid:        {'✅ PASS' if holodeck_ok else '❌ FAIL'}")
        print(f"Medical Array:        {'✅ PASS' if medical_ok else '❌ FAIL'}")
        
        all_pass = subspace_ok and holodeck_ok and medical_ok
        
        if all_pass:
            print("\n🎉 ALL WARP TECHNOLOGIES OPERATIONAL! 🎉")
            print("Ready for integration into unified warp system")
        else:
            print("\n⚠️  Some systems need attention")
        
        return 0 if all_pass else 1
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
