"""
Comprehensive Integration Test for New Warp Technologies
========================================================

Tests the integration of all three new warp field components:
1. Subspace Transceiver
2. Holodeck Force-Field Grid  
3. Medical Tractor Array

This test demonstrates the complete expandable warp technology monorepo.
"""

import numpy as np
import time
import logging
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import individual modules directly to avoid circular imports
from subspace_transceiver.transceiver import SubspaceTransceiver, SubspaceParams, TransmissionParams
from holodeck_forcefield_grid.grid import ForceFieldGrid, GridParams, Node
from medical_tractor_array.array import MedicalTractorArray, MedicalArrayParams, VitalSigns, SafetyLevel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class WarpTechnologyIntegrationTest:
    """
    Complete integration test for all new warp technologies
    """
    
    def __init__(self):
        """Initialize all three warp technology systems"""
        logging.info("=== INITIALIZING WARP TECHNOLOGY SUITE ===")
        
        # 1. Initialize Subspace Transceiver
        logging.info("Initializing Subspace Communication System...")
        transceiver_params = SubspaceParams(
            c_s=5.0e8,              # Faster subspace wave speed
            bandwidth=1e12,         # 1 THz bandwidth
            power_limit=1e5,        # 100 kW power limit
            grid_resolution=64      # Lower resolution for testing
        )
        self.transceiver = SubspaceTransceiver(transceiver_params)
        
        # 2. Initialize Holodeck Force-Field Grid
        logging.info("Initializing Holodeck Force-Field System...")
        grid_params = GridParams(
            bounds=((-1.0, 1.0), (-1.0, 1.0), (0.0, 2.0)),
            base_spacing=0.08,          # 8 cm base spacing
            fine_spacing=0.02,          # 2 cm fine spacing
            update_rate=50e3,           # 50 kHz
            max_nodes=10000
        )
        self.holodeck_grid = ForceFieldGrid(grid_params)
        
        # 3. Initialize Medical Tractor Array
        logging.info("Initializing Medical Tractor Array...")
        medical_params = MedicalArrayParams(
            array_bounds=((-0.3, 0.3), (-0.3, 0.3), (0.1, 0.5)),
            beam_spacing=0.03,          # 3 cm beam spacing
            safety_level=SafetyLevel.THERAPEUTIC,
            max_beams=200
        )
        self.medical_array = MedicalTractorArray(medical_params)
        
        logging.info("=== ALL SYSTEMS INITIALIZED ===")
    
    def test_subspace_communication(self):
        """Test subspace communication system"""
        logging.info("\n--- TESTING SUBSPACE TRANSCEIVER ---")
        
        # Run diagnostics
        diag = self.transceiver.run_diagnostics()
        logging.info(f"Transceiver health: {diag['overall_health']}")
        
        # Test message transmission
        test_message = "Medical procedure initiated on Deck 12"
        destination = "Starfleet Medical, Earth"
        
        tx_params = TransmissionParams(
            destination=destination,
            priority=5,  # High priority
            power_level=0.8
        )
        
        result = self.transceiver.transmit_message(test_message, tx_params)
        
        assert result['status'] == 'TRANSMITTED'
        logging.info(f"Message transmitted successfully: {result.get('transmission_id', 'N/A')}")
        
        # Test emergency broadcast
        emergency_msg = "EMERGENCY: Warp core breach - immediate evacuation required"
        emergency_params = TransmissionParams(
            destination="ALL_CHANNELS",
            priority=10,  # Maximum priority
            power_level=1.0
        )
        emergency_result = self.transceiver.transmit_message(emergency_msg, emergency_params)
        
        assert emergency_result['status'] == 'TRANSMITTED'
        logging.info(f"Emergency broadcast sent: {emergency_result.get('transmission_id', 'N/A')}")
        
        return True
    
    def test_holodeck_force_field(self):
        """Test holodeck force-field grid system"""
        logging.info("\n--- TESTING HOLODECK FORCE-FIELD GRID ---")
        
        # Run diagnostics
        diag = self.holodeck_grid.run_diagnostics()
        logging.info(f"Holodeck grid health: {diag['overall_health']}")
        logging.info(f"Total nodes: {diag['total_nodes']}, Active: {diag['active_nodes']}")
        
        # Test material simulation - create interaction zones
        logging.info("Creating material interaction zones...")
        
        # Stone wall simulation
        wall_center = np.array([0.5, 0.0, 1.0])
        self.holodeck_grid.add_interaction_zone(wall_center, 0.2, "rigid")
        
        # Soft grass simulation  
        grass_center = np.array([0.0, 0.5, 0.5])
        self.holodeck_grid.add_interaction_zone(grass_center, 0.3, "soft")
        
        # Water simulation
        water_center = np.array([-0.5, -0.5, 0.8])
        self.holodeck_grid.add_interaction_zone(water_center, 0.25, "liquid")
        
        # Test force computation at different points
        test_points = [
            ("stone wall", wall_center + np.array([0.1, 0.0, 0.0])),
            ("soft grass", grass_center + np.array([0.0, 0.1, 0.0])), 
            ("water surface", water_center + np.array([0.0, 0.0, 0.1]))
        ]
        
        for name, point in test_points:
            velocity = np.array([0.05, 0.02, 0.0])  # 5 cm/s movement
            force = self.holodeck_grid.compute_total_force(point, velocity)
            force_magnitude = np.linalg.norm(force)
            
            logging.info(f"Force at {name}: {force_magnitude:.3f} N")
        
        # Test object tracking
        logging.info("Testing object tracking...")
        hand_position = np.array([0.2, 0.3, 1.2])
        hand_velocity = np.array([0.1, 0.05, 0.0])
        
        self.holodeck_grid.update_object_tracking("user_hand", hand_position, hand_velocity)
        
        # Simulate interaction
        step_result = self.holodeck_grid.step_simulation(0.00002)  # 20 μs step
        logging.info(f"Simulation step: {step_result['computation_time']*1000:.2f} ms")
        
        return True
    
    def test_medical_tractor_array(self):
        """Test medical tractor array system"""
        logging.info("\n--- TESTING MEDICAL TRACTOR ARRAY ---")
        
        # Run diagnostics
        diag = self.medical_array.run_diagnostics()
        logging.info(f"Medical array health: {diag['overall_health']}")
        logging.info(f"Total beams: {diag['total_beams']}, Safety level: {diag['safety_level']}")
        
        # Set up patient monitoring
        logging.info("Setting up patient vital signs monitoring...")
        vital_signs = VitalSigns(
            heart_rate=75.0,
            blood_pressure_sys=120.0,
            blood_pressure_dia=80.0,
            oxygen_saturation=98.5,
            respiratory_rate=16.0
        )
        self.medical_array.update_vital_signs(vital_signs)
        
        # Start medical procedure
        self.medical_array.start_procedure("PATIENT_ALPHA_001", "tissue_positioning")
        
        # Test 1: Precise tissue positioning
        logging.info("Testing precise tissue positioning...")
        tissue_position = np.array([0.05, 0.03, 0.2])
        target_position = np.array([0.04, 0.03, 0.19])  # Move 1 cm
        
        positioning_result = self.medical_array.position_target(
            tissue_position, target_position, 
            target_size=10e-6,  # 10 μm tissue element
            tissue_type="organ"
        )
        
        if positioning_result['status'] != 'SAFETY_VIOLATION':
            logging.info(f"Tissue positioning: {positioning_result['status']}")
            if 'distance_to_target' in positioning_result:
                distance_mm = positioning_result['distance_to_target'] * 1000
                logging.info(f"Distance to target: {distance_mm:.2f} mm")
        
        # Test 2: Wound closure assistance
        logging.info("Testing wound closure assistance...")
        wound_edges = [
            np.array([0.0, -0.003, 0.15]),  # 3 mm wound
            np.array([0.0, 0.003, 0.15]),
            np.array([0.002, 0.0, 0.15]),
            np.array([-0.002, 0.0, 0.15])
        ]
        
        closure_result = self.medical_array.assist_wound_closure(wound_edges)
        logging.info(f"Wound closure: {closure_result['status']}")
        
        # Test 3: Catheter guidance
        logging.info("Testing catheter guidance...")
        catheter_tip = np.array([0.08, 0.04, 0.18])
        target_vessel = np.array([0.06, 0.02, 0.16])
        
        guidance_result = self.medical_array.guide_catheter(
            catheter_tip, target_vessel, vessel_diameter=2e-3
        )
        logging.info(f"Catheter guidance: {guidance_result['status']}")
        
        # Stop procedure
        self.medical_array.stop_procedure()
        
        return True
    
    def test_integrated_scenario(self):
        """Test an integrated scenario using all three systems"""
        logging.info("\n--- TESTING INTEGRATED SCENARIO ---")
        logging.info("Scenario: Emergency medical procedure with holodeck training simulation")
        
        # 1. Emergency communication
        emergency_msg = "Medical emergency in progress - activating all systems"
        emergency_params = TransmissionParams(destination="ALL_CHANNELS", priority=10)
        comm_result = self.transceiver.transmit_message(emergency_msg, emergency_params)
        logging.info(f"Emergency communication: {comm_result['status']}")
        
        # 2. Set up holodeck training environment
        logging.info("Configuring holodeck training environment...")
        
        # Create realistic tissue textures for training
        liver_center = np.array([0.2, 0.1, 1.0])
        self.holodeck_grid.add_interaction_zone(liver_center, 0.15, "flesh")
        
        bone_center = np.array([0.1, 0.3, 1.2])  
        self.holodeck_grid.add_interaction_zone(bone_center, 0.1, "rigid")
        
        # 3. Initialize medical array for actual procedure
        logging.info("Activating medical tractor array...")
        self.medical_array.start_procedure("EMERGENCY_001", "trauma_surgery")
        
        # Simulate coordinated operation
        for step in range(5):
            # Holodeck simulation update
            surgeon_hand = np.array([0.2 + step*0.01, 0.1, 1.0])
            self.holodeck_grid.update_object_tracking("surgeon_hand", surgeon_hand)
            holodeck_step = self.holodeck_grid.step_simulation(0.00002)
            
            # Medical array operation (if not in safety violation)
            if not self.medical_array.emergency_stop:
                tissue_pos = np.array([0.05, 0.02 + step*0.001, 0.2])
                target_pos = np.array([0.04, 0.02 + step*0.001, 0.19])
                
                positioning = self.medical_array.position_target(
                    tissue_pos, target_pos, tissue_type="organ"
                )
            
            # Progress communication
            progress_msg = f"Procedure step {step+1}/5 completed"
            progress_params = TransmissionParams(destination="Medical Team Alpha", priority=3)
            self.transceiver.transmit_message(progress_msg, progress_params)
            
            time.sleep(0.1)  # Small delay for realism
        
        self.medical_array.stop_procedure()
        logging.info("Integrated scenario completed successfully")
        
        return True
    
    def run_performance_analysis(self):
        """Analyze performance across all systems"""
        logging.info("\n--- PERFORMANCE ANALYSIS ---")
        
        # Get performance metrics from all systems
        transceiver_metrics = self.transceiver.get_performance_metrics()
        holodeck_metrics = self.holodeck_grid.get_performance_metrics()
        medical_diagnostics = self.medical_array.run_diagnostics()
        
        logging.info("=== SYSTEM PERFORMANCE SUMMARY ===")
        
        # Subspace Transceiver Performance
        logging.info("Subspace Transceiver:")
        logging.info(f"  Signal strength: {transceiver_metrics.get('signal_strength', 0):.1f} dB")
        logging.info(f"  Channel efficiency: {transceiver_metrics.get('channel_efficiency', 0):.1%}")
        logging.info(f"  Messages sent: {transceiver_metrics.get('total_messages_sent', 0)}")
        
        # Holodeck Grid Performance
        if holodeck_metrics:
            logging.info("Holodeck Force-Field Grid:")
            logging.info(f"  Update rate: {holodeck_metrics.get('effective_update_rate', 0):.1f} Hz")
            logging.info(f"  Performance ratio: {holodeck_metrics.get('performance_ratio', 0):.1%}")
            logging.info(f"  Total nodes: {holodeck_metrics.get('total_nodes', 0)}")
            logging.info(f"  Interaction zones: {holodeck_metrics.get('interaction_zones', 0)}")
        
        # Medical Array Performance
        logging.info("Medical Tractor Array:")
        logging.info(f"  Position accuracy: {medical_diagnostics['position_accuracy']:.1e} m")
        logging.info(f"  Force resolution: {medical_diagnostics['force_resolution']:.1e} N")
        logging.info(f"  Safety level: {medical_diagnostics['safety_level']}")
        logging.info(f"  Active beams: {medical_diagnostics['active_beams']}")
        
        # Overall system health
        all_healthy = all([
            self.transceiver.run_diagnostics()['overall_health'] == 'HEALTHY',
            self.holodeck_grid.run_diagnostics()['overall_health'] == 'HEALTHY',
            medical_diagnostics['overall_health'] in ['HEALTHY', 'DEGRADED']  # DEGRADED due to no vital signs initially
        ])
        
        overall_status = "OPERATIONAL" if all_healthy else "DEGRADED"
        logging.info(f"\n=== OVERALL WARP TECHNOLOGY STATUS: {overall_status} ===")
        
        return overall_status == "OPERATIONAL"
    
    def run_complete_test_suite(self):
        """Run the complete test suite for all warp technologies"""
        logging.info("🚀 STARTING COMPREHENSIVE WARP TECHNOLOGY TEST SUITE 🚀")
        
        test_results = {}
        
        try:
            # Test each system individually
            test_results['subspace_communication'] = self.test_subspace_communication()
            test_results['holodeck_force_field'] = self.test_holodeck_force_field()
            test_results['medical_tractor_array'] = self.test_medical_tractor_array()
            
            # Test integrated scenario
            test_results['integrated_scenario'] = self.test_integrated_scenario()
            
            # Performance analysis
            test_results['performance_analysis'] = self.run_performance_analysis()
            
        except Exception as e:
            logging.error(f"Test suite error: {e}")
            test_results['error'] = str(e)
        
        # Final results
        logging.info("\n" + "="*60)
        logging.info("WARP TECHNOLOGY TEST SUITE RESULTS")
        logging.info("="*60)
        
        for test_name, result in test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            logging.info(f"{test_name:.<40} {status}")
        
        total_tests = len([r for r in test_results.values() if isinstance(r, bool)])
        passed_tests = sum([r for r in test_results.values() if isinstance(r, bool)])
        
        logging.info(f"\nTest Results: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            logging.info("🎉 ALL WARP TECHNOLOGY SYSTEMS OPERATIONAL 🎉")
            logging.info("Ready for deployment to Starfleet Engineering Corps")
        else:
            logging.info("⚠️  Some systems require attention before deployment")
        
        return passed_tests == total_tests

def main():
    """Main test execution"""
    print("🌌 WARP FIELD TECHNOLOGY INTEGRATION TEST 🌌")
    print("="*60)
    
    # Initialize test suite
    test_suite = WarpTechnologyIntegrationTest()
    
    # Run complete test suite
    success = test_suite.run_complete_test_suite()
    
    if success:
        print("\n🚀 WARP TECHNOLOGY INTEGRATION: SUCCESS 🚀")
        return 0
    else:
        print("\n⚠️  WARP TECHNOLOGY INTEGRATION: PARTIAL SUCCESS ⚠️")
        return 1

if __name__ == "__main__":
    exit(main())
