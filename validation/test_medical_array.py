"""
Test Suite for Medical Tractor Array
====================================

Comprehensive tests for medical-grade tractor beam system.
"""

import numpy as np
import pytest
import time
import logging
from unittest.mock import Mock, patch

# Import the modules we're testing directly
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.medical_tractor_array.array import (
    MedicalTractorArray, MedicalArrayParams, TractorBeam, 
    BeamMode, SafetyLevel, VitalSigns
)

logging.basicConfig(level=logging.INFO)

class TestMedicalTractorArray:
    """Test suite for medical tractor array system"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.params = MedicalArrayParams(
            array_bounds=((-0.2, 0.2), (-0.2, 0.2), (0.05, 0.3)),
            beam_spacing=0.05,
            safety_level=SafetyLevel.THERAPEUTIC,
            max_beams=50
        )
        self.array = MedicalTractorArray(self.params)
    
    def test_array_initialization(self):
        """Test proper array initialization"""
        assert len(self.array.beams) > 0
        assert len(self.array.beams) <= self.params.max_beams
        assert not self.array.emergency_stop
        assert not self.array.procedure_active
        assert self.array.params.safety_level == SafetyLevel.THERAPEUTIC
        
        # Check beam distribution
        positions = np.array([beam.position for beam in self.array.beams])
        assert np.all(positions[:, 0] >= self.params.array_bounds[0][0])
        assert np.all(positions[:, 0] <= self.params.array_bounds[0][1])
        assert np.all(positions[:, 1] >= self.params.array_bounds[1][0])
        assert np.all(positions[:, 1] <= self.params.array_bounds[1][1])
        assert np.all(positions[:, 2] >= self.params.array_bounds[2][0])
        assert np.all(positions[:, 2] <= self.params.array_bounds[2][1])
    
    def test_optical_force_computation(self):
        """Test optical force computation from individual beam"""
        if not self.array.beams:
            pytest.skip("No beams available for testing")
        
        beam = self.array.beams[0]
        target_position = beam.position + np.array([0.01, 0.0, 0.02])  # 1 cm + 2 cm away
        particle_radius = 1e-6  # 1 μm particle
        
        # Test force computation
        force = self.array.compute_optical_force(beam, target_position, particle_radius)
        
        assert isinstance(force, np.ndarray)
        assert force.shape == (3,)
        assert np.all(np.isfinite(force))
        
        # Force should be reasonable magnitude for medical applications
        force_magnitude = np.linalg.norm(force)
        assert force_magnitude < 1e-5  # Should be micro-Newton range or less
    
    def test_target_positioning(self):
        """Test precise target positioning functionality"""
        target_position = np.array([0.05, 0.02, 0.1])
        desired_position = np.array([0.03, 0.02, 0.1])
        target_size = 5e-6  # 5 μm target
        
        # Start procedure first
        self.array.start_procedure("TEST_001", "positioning_test")
        
        # Test positioning
        result = self.array.position_target(
            target_position, desired_position, target_size, "organ"
        )
        
        assert 'status' in result
        assert 'force' in result
        assert 'distance_to_target' in result
        
        # Should be attempting to position
        assert result['status'] in ['POSITIONING', 'TARGET_REACHED']
        
        # Force should be finite and reasonable
        force = result['force']
        assert isinstance(force, np.ndarray)
        assert np.all(np.isfinite(force))
        
        force_magnitude = np.linalg.norm(force)
        assert force_magnitude < 1e-5  # Medical-grade force limits
        
        # Distance should be computed correctly
        expected_distance = np.linalg.norm(desired_position - target_position)
        assert abs(result['distance_to_target'] - expected_distance) < 1e-6
    
    def test_wound_closure_assistance(self):
        """Test wound closure assistance functionality"""
        # Define wound edges
        wound_edges = [
            np.array([0.0, -0.005, 0.1]),  # 5 mm apart
            np.array([0.0, 0.005, 0.1]),
            np.array([0.005, 0.0, 0.1]),
            np.array([-0.005, 0.0, 0.1])
        ]
        
        # Start procedure
        self.array.start_procedure("PATIENT_001", "wound_closure")
        
        # Test wound closure
        result = self.array.assist_wound_closure(wound_edges)
        
        assert 'status' in result
        assert result['status'] in ['CLOSURE_ACTIVE', 'SAFETY_VIOLATION']
        
        if result['status'] == 'CLOSURE_ACTIVE':
            assert 'wound_center' in result
            assert 'edge_results' in result
            assert 'closure_progress' in result
            
            # Check wound center computation
            expected_center = np.mean(wound_edges, axis=0)
            assert np.allclose(result['wound_center'], expected_center, atol=1e-6)
            
            # Check edge results
            edge_results = result['edge_results']
            assert len(edge_results) == len(wound_edges)
            
            for edge_result in edge_results:
                assert 'edge_position' in edge_result
                assert 'closure_force' in edge_result
                assert 'closure_distance' in edge_result
                
                # Force should be reasonable
                force_mag = np.linalg.norm(edge_result['closure_force'])
                assert force_mag < 1e-6  # Very gentle for wound closure
    
    def test_catheter_guidance(self):
        """Test catheter guidance functionality"""
        catheter_tip = np.array([0.1, 0.05, 0.15])
        target_vessel = np.array([0.08, 0.02, 0.12])
        vessel_diameter = 3e-3  # 3 mm vessel
        
        # Start procedure
        self.array.start_procedure("PATIENT_002", "catheter_insertion")
        
        # Test guidance
        result = self.array.guide_catheter(catheter_tip, target_vessel, vessel_diameter)
        
        assert 'status' in result
        assert result['status'] in ['GUIDING', 'TARGET_REACHED', 'SAFETY_VIOLATION']
        
        if result['status'] == 'GUIDING':
            assert 'guidance_force' in result
            assert 'distance_to_target' in result
            assert 'guidance_accuracy' in result
            
            # Check distance calculation
            expected_distance = np.linalg.norm(target_vessel - catheter_tip)
            assert abs(result['distance_to_target'] - expected_distance) < 1e-6
            
            # Guidance accuracy should be reasonable
            accuracy = result['guidance_accuracy']
            assert accuracy > 0
            assert accuracy < 100  # Should be reasonable ratio
        
        elif result['status'] == 'TARGET_REACHED':
            assert 'distance' in result
            assert result['distance'] < vessel_diameter / 2
    
    def test_vital_signs_monitoring(self):
        """Test vital signs monitoring and alerts"""
        # Test normal vital signs
        normal_vitals = VitalSigns(
            heart_rate=75.0,
            blood_pressure_sys=120.0,
            blood_pressure_dia=80.0,
            oxygen_saturation=98.0,
            respiratory_rate=16.0
        )
        
        self.array.update_vital_signs(normal_vitals)
        assert self.array.vital_signs.heart_rate == 75.0
        
        # Test abnormal vital signs that should trigger alerts
        abnormal_vitals = VitalSigns(
            heart_rate=45.0,  # Too low
            blood_pressure_sys=190.0,  # Too high
            oxygen_saturation=85.0,  # Too low
        )
        
        self.array.update_vital_signs(abnormal_vitals)
        
        # Check for alerts
        alerts = self.array._check_vital_signs_alerts()
        assert len(alerts) > 0
        
        # Should have alerts for low HR, high BP, and low SpO2
        alert_text = ' '.join(alerts)
        assert 'Heart rate' in alert_text
        assert 'Systolic BP' in alert_text
        assert 'Oxygen saturation' in alert_text
    
    def test_safety_systems(self):
        """Test comprehensive safety systems"""
        # Test safety level enforcement
        assert self.array.params.safety_level == SafetyLevel.THERAPEUTIC
        
        # Test safety condition checking
        safety_check = self.array._check_safety_conditions()
        assert 'safe' in safety_check
        assert 'alerts' in safety_check
        
        # Initially should be safe
        assert safety_check['safe']
        
        # Test emergency shutdown
        self.array.emergency_shutdown("Test emergency")
        assert self.array.emergency_stop
        
        # All beams should be deactivated
        for beam in self.array.beams:
            assert not beam.active
            assert beam.power == 0.0
        
        # Power usage should be zero
        assert self.array.total_power_usage == 0.0
    
    def test_procedure_management(self):
        """Test medical procedure start/stop functionality"""
        patient_id = "PATIENT_TEST_001"
        procedure_type = "tissue_manipulation"
        
        # Initially no procedure should be active
        assert not self.array.procedure_active
        
        # Start procedure
        self.array.start_procedure(patient_id, procedure_type)
        
        assert self.array.procedure_active
        assert self.array.patient_id == patient_id
        assert self.array.procedure_type == procedure_type
        assert self.array.procedure_start_time > 0
        assert not self.array.emergency_stop
        
        # All beam exposure times should be reset
        for beam in self.array.beams:
            assert beam.exposure_time == 0.0
        
        # Stop procedure
        self.array.stop_procedure()
        
        assert not self.array.procedure_active
        
        # All beams should be deactivated
        for beam in self.array.beams:
            assert not beam.active
    
    def test_beam_modes(self):
        """Test different beam operating modes"""
        beam = self.array.beams[0] if self.array.beams else None
        if beam is None:
            pytest.skip("No beams available for testing")
        
        # Test all beam modes
        modes = [BeamMode.POSITIONING, BeamMode.CLOSURE, BeamMode.GUIDANCE, 
                BeamMode.MANIPULATION, BeamMode.SCANNING]
        
        for mode in modes:
            beam.mode = mode
            assert beam.mode == mode
            
            # Mode should affect beam behavior (implementation specific)
            # Here we just verify the mode is set correctly
    
    def test_safety_levels(self):
        """Test different safety levels"""
        safety_levels = [SafetyLevel.DIAGNOSTIC, SafetyLevel.THERAPEUTIC, 
                        SafetyLevel.SURGICAL, SafetyLevel.EMERGENCY]
        
        for level in safety_levels:
            params = MedicalArrayParams(safety_level=level)
            array = MedicalTractorArray(params)
            assert array.params.safety_level == level
    
    def test_tissue_type_power_limits(self):
        """Test power limits for different tissue types"""
        tissue_types = ["soft", "bone", "organ", "neural", "vascular", "skin"]
        
        for tissue_type in tissue_types:
            assert tissue_type in self.params.tissue_power_limits
            power_limit = self.params.tissue_power_limits[tissue_type]
            assert power_limit > 0
            assert power_limit < 100  # Reasonable upper bound for medical applications
        
        # Neural tissue should have lowest power limit
        assert (self.params.tissue_power_limits["neural"] <= 
                min(self.params.tissue_power_limits.values()))
    
    def test_force_resolution_and_accuracy(self):
        """Test force resolution and positioning accuracy"""
        # Force resolution should be in picoNewton range
        assert self.params.force_resolution <= 1e-9
        
        # Position accuracy should be in micrometer range
        assert self.params.position_accuracy <= 1e-5
        
        # Test that computed forces respect resolution
        if self.array.beams:
            beam = self.array.beams[0]
            target_pos = beam.position + np.array([0.001, 0.0, 0.002])
            
            force = self.array.compute_optical_force(beam, target_pos)
            force_magnitude = np.linalg.norm(force)
            
            # Force should be within reasonable range for medical applications
            assert force_magnitude < 1e-5  # Less than 10 μN
    
    def test_sterile_field_maintenance(self):
        """Test sterile field monitoring and maintenance"""
        # Test sterile field radius
        assert self.params.sterile_field_radius > 0
        assert self.params.sterile_field_radius < 1.0  # Reasonable size
        
        # Test sterile field activation
        assert not self.array.sterile_field_active  # Initially inactive
        
        # Sterile field should be considered in safety checks
        safety_check = self.array._check_safety_conditions()
        assert isinstance(safety_check, dict)
    
    def test_power_density_limits(self):
        """Test power density safety limits"""
        # Test beam power density limits
        for beam in self.array.beams:
            assert beam.power_density_limit > 0
            assert beam.power_density_limit <= 50.0  # Conservative medical limit
        
        # Test global power limit
        assert self.params.global_power_limit > 0
        assert self.params.global_power_limit < 10000  # Reasonable upper bound
    
    def test_diagnostics(self):
        """Test comprehensive diagnostics system"""
        diagnostics = self.array.run_diagnostics()
        
        # Check required diagnostic fields
        required_fields = [
            'beam_activation', 'force_computation', 'safety_systems',
            'vital_signs', 'total_beams', 'active_beams', 'overall_health',
            'emergency_systems', 'safety_level'
        ]
        
        for field in required_fields:
            assert field in diagnostics
        
        # Check diagnostic values
        assert diagnostics['total_beams'] == len(self.array.beams)
        assert diagnostics['active_beams'] >= 0
        assert diagnostics['overall_health'] in ['HEALTHY', 'DEGRADED']
        assert diagnostics['emergency_systems'] in ['ARMED', 'TRIGGERED']
        assert diagnostics['safety_level'] == self.params.safety_level.value
        
        # Test force magnitude should be reasonable
        assert 'test_force_magnitude' in diagnostics
        assert diagnostics['test_force_magnitude'] >= 0
        assert diagnostics['test_force_magnitude'] < 1e-4  # Should be very small
    
    def test_exposure_time_tracking(self):
        """Test beam exposure time tracking and limits"""
        beam = self.array.beams[0] if self.array.beams else None
        if beam is None:
            pytest.skip("No beams available for testing")
        
        # Initially exposure time should be zero
        assert beam.exposure_time == 0.0
        assert beam.max_exposure > 0
        
        # Simulate exposure by manually incrementing
        beam.exposure_time = 100.0  # 100 seconds
        assert beam.exposure_time < beam.max_exposure  # Should be within limits
        
        # Test exposure limit exceeded
        beam.exposure_time = beam.max_exposure + 10
        safety_check = self.array._check_safety_conditions()
        
        # Should trigger safety alert
        assert not safety_check['safe'] or len(safety_check['alerts']) > 0

def test_beam_parameter_validation():
    """Test beam parameter validation and limits"""
    # Test beam creation with valid parameters
    beam = TractorBeam(
        position=np.array([0.0, 0.0, 0.1]),
        direction=np.array([0.0, 0.0, -1.0]),
        power=5.0,
        wavelength=1064e-9
    )
    
    assert np.allclose(beam.position, np.array([0.0, 0.0, 0.1]))
    assert np.allclose(beam.direction, np.array([0.0, 0.0, -1.0]))
    assert beam.power == 5.0
    assert beam.wavelength == 1064e-9
    assert beam.active
    assert not beam.safety_interlock

def test_vital_signs_validation():
    """Test vital signs parameter validation"""
    vs = VitalSigns(
        heart_rate=72.0,
        blood_pressure_sys=118.0,
        blood_pressure_dia=78.0,
        oxygen_saturation=99.0
    )
    
    assert vs.heart_rate == 72.0
    assert vs.blood_pressure_sys == 118.0
    assert vs.blood_pressure_dia == 78.0
    assert vs.oxygen_saturation == 99.0
    
    # Test threshold values
    assert vs.hr_min < vs.hr_max
    assert vs.spo2_min < 100.0

def test_array_bounds_validation():
    """Test array bounds validation"""
    # Test valid bounds
    bounds = ((-0.3, 0.3), (-0.3, 0.3), (0.0, 0.5))
    params = MedicalArrayParams(array_bounds=bounds)
    array = MedicalTractorArray(params)
    
    # All beam positions should be within bounds
    for beam in array.beams:
        pos = beam.position
        assert bounds[0][0] <= pos[0] <= bounds[0][1]
        assert bounds[1][0] <= pos[1] <= bounds[1][1]
        assert bounds[2][0] <= pos[2] <= bounds[2][1]

if __name__ == "__main__":
    # Run tests directly
    import pytest
    pytest.main([__file__, "-v"])
