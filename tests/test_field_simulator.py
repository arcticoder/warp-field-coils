#!/usr/bin/env python3
"""
Test suite for electromagnetic field simulation
"""

import pytest
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from hardware.field_rig_design import ElectromagneticFieldSimulator, FieldRigResults

class TestElectromagneticFieldSimulator:
    """Test cases for ElectromagneticFieldSimulator class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.simulator = ElectromagneticFieldSimulator()
    
    def test_initialization(self):
        """Test simulator initialization."""
        assert self.simulator.wire_resistivity > 0
        assert self.simulator.wire_critical_current_density > 0
        assert self.simulator.mu_r == 1.0
    
    def test_simulate_inductive_rig_solenoid(self):
        """Test inductive rig simulation with solenoid geometry."""
        L = 1e-3  # 1 mH
        I = 1000  # 1000 A
        f_mod = 1000  # 1 kHz
        
        results = self.simulator.simulate_inductive_rig(
            L=L, I=I, f_mod=f_mod, geometry='solenoid'
        )
        
        assert isinstance(results, FieldRigResults)
        assert results.r_char > 0
        assert results.B_peak > 0
        assert results.E_peak >= 0
        assert results.stored_energy > 0
        assert results.inductance == L
        
        # Check safety margins
        assert 'magnetic_field' in results.safety_margins
        assert 'electric_field' in results.safety_margins
        assert 'current' in results.safety_margins
        assert 'voltage' in results.safety_margins
    
    def test_simulate_inductive_rig_toroidal(self):
        """Test inductive rig simulation with toroidal geometry."""
        L = 5e-4  # 0.5 mH
        I = 2000  # 2000 A
        f_mod = 500   # 500 Hz
        
        results = self.simulator.simulate_inductive_rig(
            L=L, I=I, f_mod=f_mod, geometry='toroidal'
        )
        
        assert isinstance(results, FieldRigResults)
        assert results.r_char > 0
        assert results.B_peak > 0
        assert np.isfinite(results.power_dissipation)
        assert results.resistance >= 0
    
    def test_simulate_inductive_rig_planar(self):
        """Test inductive rig simulation with planar geometry."""
        L = 2e-3  # 2 mH
        I = 500   # 500 A
        f_mod = 2000  # 2 kHz
        
        results = self.simulator.simulate_inductive_rig(
            L=L, I=I, f_mod=f_mod, geometry='planar'
        )
        
        assert isinstance(results, FieldRigResults)
        assert results.r_char > 0
        assert all(margin > 0 for margin in results.safety_margins.values())
    
    def test_safety_analysis(self):
        """Test safety analysis functionality."""
        # Create test results with known safety margins
        test_results = FieldRigResults(
            r_char=1.0,
            B_peak=10.0,     # Well below 100 T limit
            E_peak=1e10,     # Below 1e14 V/m limit
            rho_B=1000.0,
            rho_E=1000.0,
            stored_energy=1000.0,
            power_dissipation=100.0,
            inductance=1e-3,
            resistance=1e-3,
            safety_margins={
                'magnetic_field': 10.0,  # 10x margin
                'electric_field': 100.0,  # 100x margin
                'current': 20.0,         # 20x margin
                'voltage': 50.0          # 50x margin
            }
        )
        
        safety_status = self.simulator.safety_analysis(test_results)
        
        assert safety_status['magnetic_field'] == "CAUTION (2-10x margin)"
        assert safety_status['electric_field'] == "SAFE (>10x margin)"
        assert safety_status['current'] == "SAFE (>10x margin)"
        assert safety_status['voltage'] == "SAFE (>10x margin)"
        
        # Test warning case
        warning_results = FieldRigResults(
            r_char=1.0, B_peak=50.0, E_peak=5e13, rho_B=1000.0, rho_E=1000.0,
            stored_energy=1000.0, power_dissipation=100.0, inductance=1e-3, resistance=1e-3,
            safety_margins={
                'magnetic_field': 1.5,   # <2x margin
                'electric_field': 5.0,
                'current': 10.0,
                'voltage': 10.0
            }
        )
        
        safety_status_warn = self.simulator.safety_analysis(warning_results)
        assert safety_status_warn['magnetic_field'] == "WARNING (<2x margin)"
    
    def test_sweep_current_geometry(self):
        """Test parameter sweep functionality."""
        L_range = (1e-4, 1e-2)  # 0.1 mH to 10 mH
        I_range = (100, 1000)   # 100 A to 1000 A
        
        sweep_results = self.simulator.sweep_current_geometry(
            L_range=L_range, I_range=I_range, n_points=5  # Small for testing
        )
        
        assert 'L_grid' in sweep_results
        assert 'I_grid' in sweep_results
        assert 'B_peak_grid' in sweep_results
        assert 'safety_grid' in sweep_results
        assert 'safe_mask' in sweep_results
        
        # Check grid shapes
        assert sweep_results['L_grid'].shape == (5, 5)
        assert sweep_results['I_grid'].shape == (5, 5)
        assert sweep_results['B_peak_grid'].shape == (5, 5)
        
        # Check that some results are finite
        assert np.any(np.isfinite(sweep_results['B_peak_grid']))
    
    def test_optimize_for_target_field(self):
        """Test field optimization functionality."""
        B_target = 1.0  # Target 1 T field
        
        opt_results = self.simulator.optimize_for_target_field(
            B_target=B_target,
            geometry='solenoid',
            max_current=5000.0
        )
        
        assert 'success' in opt_results
        
        if opt_results['success']:
            assert 'optimal_L' in opt_results
            assert 'optimal_I' in opt_results
            assert 'achieved_B' in opt_results
            
            assert opt_results['optimal_L'] > 0
            assert opt_results['optimal_I'] > 0
            assert opt_results['optimal_I'] <= 5000.0  # Respect max current
            
            # Check that achieved field is close to target
            error = abs(opt_results['achieved_B'] - B_target)
            assert error < 0.1 * B_target  # Within 10% of target

class TestFieldRigResults:
    """Test cases for FieldRigResults dataclass."""
    
    def test_field_rig_results_creation(self):
        """Test FieldRigResults creation and access."""
        results = FieldRigResults(
            r_char=1.5,
            B_peak=25.0,
            E_peak=1e12,
            rho_B=5000.0,
            rho_E=3000.0,
            stored_energy=10000.0,
            power_dissipation=500.0,
            inductance=2e-3,
            resistance=1e-2,
            safety_margins={'magnetic_field': 4.0, 'electric_field': 100.0}
        )
        
        assert results.r_char == 1.5
        assert results.B_peak == 25.0
        assert results.stored_energy == 10000.0
        assert results.safety_margins['magnetic_field'] == 4.0

@pytest.fixture
def configured_simulator():
    """Fixture providing a configured simulator."""
    return ElectromagneticFieldSimulator()

def test_scaling_relationships(configured_simulator):
    """Test that simulation results scale correctly with parameters."""
    L_base = 1e-3
    I_base = 1000
    f_base = 1000
    
    # Base case
    results_base = configured_simulator.simulate_inductive_rig(
        L=L_base, I=I_base, f_mod=f_base, geometry='solenoid'
    )
    
    # Double current case
    results_2I = configured_simulator.simulate_inductive_rig(
        L=L_base, I=2*I_base, f_mod=f_base, geometry='solenoid'
    )
    
    # Magnetic field should scale with current
    assert results_2I.B_peak > results_base.B_peak
    
    # Stored energy should scale as I²
    energy_ratio = results_2I.stored_energy / results_base.stored_energy
    assert 3.5 < energy_ratio < 4.5  # Should be close to 4

def test_frequency_scaling(configured_simulator):
    """Test frequency dependence of electric field."""
    L = 1e-3
    I = 1000
    
    # Low frequency
    results_low = configured_simulator.simulate_inductive_rig(
        L=L, I=I, f_mod=100, geometry='solenoid'
    )
    
    # High frequency
    results_high = configured_simulator.simulate_inductive_rig(
        L=L, I=I, f_mod=10000, geometry='solenoid'
    )
    
    # Electric field should increase with frequency
    assert results_high.E_peak > results_low.E_peak

if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__, "-v"])
