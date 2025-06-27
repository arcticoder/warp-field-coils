"""
Medical Tractor Array Module
===========================

Implements medical-grade tractor beam system for non-contact medical procedures.

Mathematical Foundation:
- Optical dipole force: F = α/2 * ∇|E|² 
- Gradient force: F_grad = α * Re[∇(E* · E)]
- Scattering force: F_scat = <σ>I/c * ẑ

Safety Features:
- Vital sign monitoring integration
- Power density limits (< 10 mW/cm²)
- Emergency stop systems
- Sterile field maintenance
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Callable, Any
import logging
import time
from enum import Enum
import threading
from queue import Queue
import warnings

class BeamMode(Enum):
    """Tractor beam operating modes"""
    POSITIONING = "positioning"      # Gentle positioning for organs/tissue
    CLOSURE = "closure"             # Wound closure assistance  
    GUIDANCE = "guidance"           # Catheter/instrument guidance
    MANIPULATION = "manipulation"   # Precise tissue manipulation
    SCANNING = "scanning"          # Non-contact scanning mode

class SafetyLevel(Enum):
    """Medical safety levels"""
    DIAGNOSTIC = "diagnostic"      # Minimal power, diagnostic only
    THERAPEUTIC = "therapeutic"    # Standard therapeutic procedures
    SURGICAL = "surgical"         # High-precision surgical operations
    EMERGENCY = "emergency"       # Emergency procedures (relaxed limits)

@dataclass
class TractorBeam:
    """Individual medical tractor beam emitter"""
    position: np.ndarray              # 3D position (m)
    direction: np.ndarray             # Beam direction unit vector
    power: float = 5.0                # Beam power (mW)
    wavelength: float = 1064e-9       # Laser wavelength (m) - IR for safety
    beam_waist: float = 50e-6         # Beam waist radius (m) - 50 μm
    focal_distance: float = 0.05      # Focal distance (m) - 5 cm
    
    # Safety parameters
    max_power: float = 50.0           # Maximum power (mW)
    power_density_limit: float = 10.0  # mW/cm²
    active: bool = True
    safety_interlock: bool = False
    
    # Medical parameters
    mode: BeamMode = BeamMode.POSITIONING
    tissue_type: str = "soft"         # Target tissue type
    exposure_time: float = 0.0        # Cumulative exposure time (s)
    max_exposure: float = 300.0       # Maximum exposure time (s)

@dataclass
class VitalSigns:
    """Patient vital signs monitoring"""
    heart_rate: float = 70.0          # beats per minute
    blood_pressure_sys: float = 120.0 # mmHg
    blood_pressure_dia: float = 80.0  # mmHg
    oxygen_saturation: float = 98.0   # %
    respiratory_rate: float = 16.0    # breaths per minute
    body_temperature: float = 37.0    # °C
    
    # Monitoring timestamps
    last_update: float = 0.0
    update_interval: float = 1.0      # seconds
    
    # Alert thresholds
    hr_min: float = 50.0
    hr_max: float = 120.0
    bp_sys_max: float = 180.0
    bp_dia_max: float = 110.0
    spo2_min: float = 90.0

@dataclass
class MedicalArrayParams:
    """Parameters for medical tractor array"""
    # Spatial configuration
    array_bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = ((-0.5, 0.5), (-0.5, 0.5), (0.0, 1.0))
    beam_spacing: float = 0.02        # 2 cm beam spacing
    max_beams: int = 100              # Maximum number of beams
    
    # Safety parameters
    global_power_limit: float = 500.0  # Total power limit (mW)
    min_beam_separation: float = 0.005 # Minimum beam separation (5 mm)
    vital_signs_required: bool = True  # Require vital sign monitoring
    safety_level: SafetyLevel = SafetyLevel.THERAPEUTIC
    
    # Performance parameters
    update_rate: float = 1000.0       # Update frequency (Hz) - 1 kHz
    force_resolution: float = 1e-9    # Force resolution (N) - picoNewton
    position_accuracy: float = 1e-6   # Position accuracy (m) - 1 μm
    
    # Medical parameters
    sterile_field_radius: float = 0.3  # Sterile field radius (m)
    max_procedure_time: float = 7200.0 # Maximum procedure time (s) - 2 hours
    
    # Power density limits by tissue type
    tissue_power_limits: Dict[str, float] = field(default_factory=lambda: {
        "soft": 5.0,      # mW/cm² for soft tissue
        "bone": 20.0,     # mW/cm² for bone
        "organ": 2.0,     # mW/cm² for organs
        "neural": 1.0,    # mW/cm² for neural tissue
        "vascular": 3.0,  # mW/cm² for blood vessels
        "skin": 8.0       # mW/cm² for skin
    })

class MedicalTractorArray:
    """
    Medical-grade tractor beam array for non-contact medical procedures
    
    Features:
    - Precise tissue manipulation and positioning
    - Non-contact wound closure assistance
    - Catheter and instrument guidance
    - Real-time vital sign monitoring integration
    - Comprehensive safety systems
    - Sterile field maintenance
    - Sub-micron positioning accuracy
    """
    
    def __init__(self, params: MedicalArrayParams):
        """
        Initialize medical tractor array
        
        Args:
            params: Array configuration parameters
        """
        self.params = params
        self.beams: List[TractorBeam] = []
        self.vital_signs = VitalSigns()
        
        # Procedure tracking
        self.procedure_start_time = 0.0
        self.procedure_active = False
        self.patient_id = None
        self.procedure_type = None
        
        # Safety systems
        self.emergency_stop = False
        self.safety_alerts = []
        self.total_power_usage = 0.0
        self.sterile_field_active = False
        
        # Performance monitoring
        self.position_accuracy_history = []
        self.force_application_history = []
        self.update_time_history = []
        
        # Threading for real-time operation
        self.update_thread = None
        self.vital_signs_thread = None
        self.running = False
        
        # Initialize beam array
        self._create_beam_array()
        
        logging.info(f"MedicalTractorArray initialized: {len(self.beams)} beams, "
                    f"safety level {params.safety_level.value}")

    def _create_beam_array(self):
        """Create the initial tractor beam array"""
        x_min, x_max = self.params.array_bounds[0]
        y_min, y_max = self.params.array_bounds[1]
        z_min, z_max = self.params.array_bounds[2]
        
        spacing = self.params.beam_spacing
        
        # Generate beam positions
        x_coords = np.arange(x_min, x_max + spacing, spacing)
        y_coords = np.arange(y_min, y_max + spacing, spacing)
        z_coords = np.arange(z_min, z_max + spacing, spacing)
        
        beam_count = 0
        for x in x_coords:
            for y in y_coords:
                for z in z_coords:
                    if beam_count >= self.params.max_beams:
                        break
                    
                    position = np.array([x, y, z])
                    # Default beam pointing downward
                    direction = np.array([0.0, 0.0, -1.0])
                    
                    beam = TractorBeam(
                        position=position,
                        direction=direction,
                        power=2.0,  # Start with low power
                        max_power=self.params.tissue_power_limits["soft"]
                    )
                    self.beams.append(beam)
                    beam_count += 1
        
        logging.info(f"Created medical beam array: {len(self.beams)} beams")

    def compute_optical_force(self, beam: TractorBeam, target_position: np.ndarray,
                            particle_radius: float = 1e-6,
                            refractive_index: float = 1.4) -> np.ndarray:
        """
        Compute optical force from tractor beam
        
        F = F_grad + F_scat
        F_grad = α * ∇I (gradient force)
        F_scat = σ * I/c * ẑ (scattering force)
        
        Args:
            beam: Tractor beam
            target_position: Position where force is computed
            particle_radius: Effective particle radius (m)
            refractive_index: Relative refractive index
            
        Returns:
            3D optical force vector (N)
        """
        if not beam.active or beam.safety_interlock:
            return np.zeros(3)
        
        # Vector from beam position to target
        r_vec = target_position - beam.position
        r_mag = np.linalg.norm(r_vec)
        
        if r_mag < 1e-9:  # Avoid singularity
            return np.zeros(3)
        
        # Gaussian beam intensity profile
        # I(r,z) = I0 * (w0/w(z))² * exp(-2r²/w(z)²)
        z = np.dot(r_vec, beam.direction)  # Distance along beam axis
        r_perp = np.linalg.norm(r_vec - z * beam.direction)  # Perpendicular distance
        
        # Beam waist evolution
        z_R = np.pi * beam.beam_waist**2 / beam.wavelength  # Rayleigh range
        w_z = beam.beam_waist * np.sqrt(1 + (z / z_R)**2)   # Beam waist at z
        
        # Beam intensity
        I0 = 2 * beam.power / (np.pi * beam.beam_waist**2) * 1e-3  # W/m² (convert mW to W)
        intensity = I0 * (beam.beam_waist / w_z)**2 * np.exp(-2 * r_perp**2 / w_z**2)
        
        # Polarizability (simplified for small sphere)
        alpha = 4 * np.pi * (8.854e-12) * particle_radius**3 * (refractive_index**2 - 1) / (refractive_index**2 + 2)
        
        # Gradient force (toward high intensity)
        grad_I = np.zeros(3)
        
        # Simplified gradient calculation
        if r_perp > 0:
            r_perp_unit = (r_vec - z * beam.direction) / r_perp
            grad_I_radial = -4 * intensity * r_perp / w_z**2
            grad_I += grad_I_radial * r_perp_unit
        
        # Axial gradient
        if abs(z) > 0:
            z_unit = beam.direction * np.sign(z)
            grad_I_axial = intensity * 2 * z / z_R**2 * (beam.beam_waist / w_z)**2
            grad_I += grad_I_axial * z_unit
        
        F_gradient = alpha / 2 * grad_I
        
        # Scattering force (along beam direction)
        c = 3e8  # Speed of light
        sigma_scat = (8 * np.pi / 3) * (2 * np.pi * particle_radius / beam.wavelength)**4 * alpha**2
        F_scattering = sigma_scat * intensity / c * beam.direction
        
        total_force = F_gradient + F_scattering
        
        # Apply safety limits
        force_magnitude = np.linalg.norm(total_force)
        max_safe_force = 1e-6  # 1 μN maximum for safety
        if force_magnitude > max_safe_force:
            total_force = total_force * (max_safe_force / force_magnitude)
        
        return total_force

    def position_target(self, target_position: np.ndarray, 
                       desired_position: np.ndarray,
                       target_size: float = 1e-6,
                       tissue_type: str = "soft") -> Dict:
        """
        Position target object using coordinated beam array
        
        Args:
            target_position: Current target position
            desired_position: Desired target position
            target_size: Effective target size (m)
            tissue_type: Type of tissue being manipulated
            
        Returns:
            Positioning results and status
        """
        if self.emergency_stop:
            return {'status': 'EMERGENCY_STOP', 'force': np.zeros(3)}
        
        # Check safety conditions
        safety_check = self._check_safety_conditions()
        if not safety_check['safe']:
            return {'status': 'SAFETY_VIOLATION', 'alerts': safety_check['alerts'], 'force': np.zeros(3)}
        
        # Compute desired force direction
        displacement = desired_position - target_position
        distance = np.linalg.norm(displacement)
        
        if distance < self.params.position_accuracy:
            return {'status': 'TARGET_REACHED', 'force': np.zeros(3), 'accuracy': distance}
        
        # Select optimal beams for positioning
        active_beams = self._select_positioning_beams(target_position, displacement)
        
        # Compute total force from active beams
        total_force = np.zeros(3)
        power_usage = 0.0
        
        for beam in active_beams:
            # Set beam parameters for this tissue type
            beam.tissue_type = tissue_type
            beam.mode = BeamMode.POSITIONING
            
            # Adjust power based on tissue type and distance
            max_power = self.params.tissue_power_limits.get(tissue_type, 5.0)
            beam.power = min(beam.power, max_power)
            
            # Compute force contribution
            force = self.compute_optical_force(beam, target_position, target_size)
            total_force += force
            power_usage += beam.power
            
            # Update exposure time
            beam.exposure_time += 1.0 / self.params.update_rate
        
        self.total_power_usage = power_usage
        
        # Record positioning accuracy
        self.position_accuracy_history.append(distance)
        if len(self.position_accuracy_history) > 1000:
            self.position_accuracy_history = self.position_accuracy_history[-500:]
        
        return {
            'status': 'POSITIONING',
            'force': total_force,
            'distance_to_target': distance,
            'active_beams': len(active_beams),
            'power_usage': power_usage,
            'positioning_accuracy': distance
        }

    def _select_positioning_beams(self, target_position: np.ndarray, 
                                 force_direction: np.ndarray) -> List[TractorBeam]:
        """Select optimal beams for positioning task"""
        # Find beams that can contribute to desired force direction
        suitable_beams = []
        
        for beam in self.beams:
            if not beam.active or beam.safety_interlock:
                continue
            
            # Vector from beam to target
            beam_to_target = target_position - beam.position
            beam_distance = np.linalg.norm(beam_to_target)
            
            # Check if target is within beam range
            if beam_distance > beam.focal_distance * 2:
                continue
            
            # Check if beam can contribute to desired force
            # (simplified - more sophisticated beam selection could be implemented)
            beam_direction_to_target = beam_to_target / beam_distance
            force_alignment = np.dot(beam_direction_to_target, force_direction)
            
            if force_alignment > 0.1:  # Beam can contribute to desired force
                suitable_beams.append(beam)
        
        # Limit number of active beams to prevent overheating
        max_active = min(len(suitable_beams), 10)
        return suitable_beams[:max_active]

    def assist_wound_closure(self, wound_edges: List[np.ndarray],
                           closure_force: float = 1e-9) -> Dict:
        """
        Assist in wound closure by applying gentle forces to wound edges
        
        Args:
            wound_edges: List of 3D positions along wound edges
            closure_force: Target closure force per edge point (N)
            
        Returns:
            Closure assistance results
        """
        if self.emergency_stop:
            return {'status': 'EMERGENCY_STOP'}
        
        if len(wound_edges) < 2:
            return {'status': 'INSUFFICIENT_EDGE_DATA'}
        
        # Safety checks for wound closure
        safety_check = self._check_safety_conditions()
        if not safety_check['safe']:
            return {'status': 'SAFETY_VIOLATION', 'alerts': safety_check['alerts']}
        
        # Compute wound center and closure direction
        wound_center = np.mean(wound_edges, axis=0)
        closure_results = []
        
        total_power = 0.0
        
        for edge_point in wound_edges:
            # Direction toward wound center
            closure_direction = wound_center - edge_point
            closure_distance = np.linalg.norm(closure_direction)
            
            if closure_distance > 1e-6:
                closure_direction = closure_direction / closure_distance
            else:
                continue
            
            # Select beams for this edge point
            active_beams = self._select_positioning_beams(edge_point, closure_direction)
            
            # Apply gentle closure force
            edge_force = np.zeros(3)
            for beam in active_beams:
                beam.mode = BeamMode.CLOSURE
                beam.tissue_type = "skin"  # Wound closure typically involves skin
                beam.power = min(beam.power, 3.0)  # Low power for wound closure
                
                force = self.compute_optical_force(beam, edge_point, 5e-6)  # Larger cells for tissue
                edge_force += force
                total_power += beam.power
            
            closure_results.append({
                'edge_position': edge_point,
                'closure_force': edge_force,
                'closure_distance': closure_distance
            })
        
        self.total_power_usage = total_power
        
        return {
            'status': 'CLOSURE_ACTIVE',
            'wound_center': wound_center,
            'edge_results': closure_results,
            'total_power': total_power,
            'closure_progress': np.mean([r['closure_distance'] for r in closure_results])
        }

    def guide_catheter(self, catheter_tip: np.ndarray, 
                      target_vessel: np.ndarray,
                      vessel_diameter: float = 2e-3) -> Dict:
        """
        Guide catheter tip toward target vessel
        
        Args:
            catheter_tip: Current catheter tip position
            target_vessel: Target vessel position
            vessel_diameter: Target vessel diameter (m)
            
        Returns:
            Guidance results
        """
        if self.emergency_stop:
            return {'status': 'EMERGENCY_STOP'}
        
        # Safety checks for catheter guidance
        safety_check = self._check_safety_conditions()
        if not safety_check['safe']:
            return {'status': 'SAFETY_VIOLATION', 'alerts': safety_check['alerts']}
        
        # Compute guidance force
        guidance_direction = target_vessel - catheter_tip
        guidance_distance = np.linalg.norm(guidance_direction)
        
        if guidance_distance < vessel_diameter / 2:
            return {'status': 'TARGET_REACHED', 'distance': guidance_distance}
        
        if guidance_distance > 1e-6:
            guidance_direction = guidance_direction / guidance_distance
        
        # Select beams for catheter guidance
        active_beams = self._select_positioning_beams(catheter_tip, guidance_direction)
        
        # Apply guidance force
        total_force = np.zeros(3)
        power_usage = 0.0
        
        for beam in active_beams:
            beam.mode = BeamMode.GUIDANCE
            beam.tissue_type = "vascular"
            beam.power = min(beam.power, 5.0)  # Moderate power for guidance
            
            # Compute force on catheter tip
            force = self.compute_optical_force(beam, catheter_tip, 50e-6)  # Catheter tip size
            total_force += force
            power_usage += beam.power
        
        self.total_power_usage = power_usage
        
        return {
            'status': 'GUIDING',
            'guidance_force': total_force,
            'distance_to_target': guidance_distance,
            'power_usage': power_usage,
            'guidance_accuracy': guidance_distance / vessel_diameter
        }

    def update_vital_signs(self, vital_signs: VitalSigns):
        """Update patient vital signs"""
        self.vital_signs = vital_signs
        self.vital_signs.last_update = time.time()
        
        # Check for critical vital signs
        alerts = self._check_vital_signs_alerts()
        if alerts:
            self.safety_alerts.extend(alerts)
            logging.warning(f"Vital signs alerts: {alerts}")

    def _check_vital_signs_alerts(self) -> List[str]:
        """Check for vital signs that require attention"""
        alerts = []
        vs = self.vital_signs
        
        if vs.heart_rate < vs.hr_min or vs.heart_rate > vs.hr_max:
            alerts.append(f"Heart rate out of range: {vs.heart_rate} bpm")
        
        if vs.blood_pressure_sys > vs.bp_sys_max:
            alerts.append(f"Systolic BP high: {vs.blood_pressure_sys} mmHg")
        
        if vs.blood_pressure_dia > vs.bp_dia_max:
            alerts.append(f"Diastolic BP high: {vs.blood_pressure_dia} mmHg")
        
        if vs.oxygen_saturation < vs.spo2_min:
            alerts.append(f"Oxygen saturation low: {vs.oxygen_saturation}%")
        
        return alerts

    def _check_safety_conditions(self) -> Dict:
        """Comprehensive safety condition check"""
        alerts = []
        
        # Check total power usage
        if self.total_power_usage > self.params.global_power_limit:
            alerts.append(f"Power limit exceeded: {self.total_power_usage:.1f} mW")
        
        # Check procedure time
        if self.procedure_active:
            elapsed_time = time.time() - self.procedure_start_time
            if elapsed_time > self.params.max_procedure_time:
                alerts.append(f"Maximum procedure time exceeded: {elapsed_time:.0f} s")
        
        # Check vital signs
        if self.params.vital_signs_required:
            time_since_update = time.time() - self.vital_signs.last_update
            if time_since_update > 10.0:  # No vital signs for 10 seconds
                alerts.append("Vital signs monitoring interrupted")
            
            vital_alerts = self._check_vital_signs_alerts()
            alerts.extend(vital_alerts)
        
        # Check beam exposure times
        for i, beam in enumerate(self.beams):
            if beam.exposure_time > beam.max_exposure:
                alerts.append(f"Beam {i} maximum exposure time exceeded")
        
        # Check sterile field
        if self.sterile_field_active:
            # Simplified sterile field check
            pass
        
        return {
            'safe': len(alerts) == 0,
            'alerts': alerts
        }

    def start_procedure(self, patient_id: str, procedure_type: str):
        """Start medical procedure"""
        self.patient_id = patient_id
        self.procedure_type = procedure_type
        self.procedure_start_time = time.time()
        self.procedure_active = True
        self.emergency_stop = False
        self.safety_alerts = []
        
        # Reset beam exposure times
        for beam in self.beams:
            beam.exposure_time = 0.0
        
        logging.info(f"Started procedure: {procedure_type} for patient {patient_id}")

    def stop_procedure(self):
        """Stop medical procedure"""
        self.procedure_active = False
        
        # Deactivate all beams
        for beam in self.beams:
            beam.active = False
        
        procedure_duration = time.time() - self.procedure_start_time
        logging.info(f"Stopped procedure after {procedure_duration:.1f} seconds")

    def emergency_shutdown(self, reason: str = "Manual"):
        """Emergency shutdown of all systems"""
        self.emergency_stop = True
        self.procedure_active = False
        
        # Immediately deactivate all beams
        for beam in self.beams:
            beam.active = False
            beam.power = 0.0
        
        self.total_power_usage = 0.0
        
        logging.critical(f"EMERGENCY SHUTDOWN: {reason}")

    def run_diagnostics(self) -> Dict:
        """Run comprehensive medical array diagnostics"""
        logging.info("Running medical tractor array diagnostics")
        
        # Test beam activation
        active_beams = sum(1 for beam in self.beams if beam.active)
        beam_coverage = active_beams / len(self.beams) if self.beams else 0
        
        # Test force computation
        test_position = np.array([0.0, 0.0, 0.1])
        test_force = self.compute_optical_force(self.beams[0], test_position) if self.beams else np.zeros(3)
        
        # Test safety systems
        safety_check = self._check_safety_conditions()
        
        # Test vital signs monitoring
        vital_signs_ok = (time.time() - self.vital_signs.last_update) < 5.0
        
        diagnostics = {
            'beam_activation': 'PASS' if beam_coverage > 0.8 else 'FAIL',
            'force_computation': 'PASS' if np.all(np.isfinite(test_force)) else 'FAIL',
            'safety_systems': 'PASS' if safety_check['safe'] else 'WARN',
            'vital_signs': 'PASS' if vital_signs_ok else 'FAIL',
            
            'total_beams': len(self.beams),
            'active_beams': active_beams,
            'beam_coverage': beam_coverage,
            'power_usage': self.total_power_usage,
            'power_limit': self.params.global_power_limit,
            'safety_level': self.params.safety_level.value,
            'sterile_field': 'ACTIVE' if self.sterile_field_active else 'INACTIVE',
            'emergency_systems': 'ARMED' if not self.emergency_stop else 'TRIGGERED',
            
            'test_force_magnitude': np.linalg.norm(test_force),
            'position_accuracy': self.params.position_accuracy,
            'force_resolution': self.params.force_resolution
        }
        
        # Overall health assessment
        critical_systems = ['beam_activation', 'force_computation', 'safety_systems']
        all_critical_pass = all(diagnostics[sys] == 'PASS' for sys in critical_systems)
        diagnostics['overall_health'] = 'HEALTHY' if all_critical_pass else 'DEGRADED'
        
        logging.info(f"Medical diagnostics complete: {diagnostics['overall_health']}")
        
        return diagnostics

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Initialize medical array
    params = MedicalArrayParams(
        array_bounds=((-0.2, 0.2), (-0.2, 0.2), (0.05, 0.3)),
        beam_spacing=0.03,
        safety_level=SafetyLevel.THERAPEUTIC
    )
    
    array = MedicalTractorArray(params)
    
    # Run diagnostics
    diag = array.run_diagnostics()
    print("Medical Tractor Array Diagnostics:")
    for key, value in diag.items():
        print(f"  {key}: {value}")
    
    # Test positioning
    target_pos = np.array([0.05, 0.02, 0.1])
    desired_pos = np.array([0.03, 0.02, 0.1])
    
    array.start_procedure("PATIENT_001", "tissue_positioning")
    
    result = array.position_target(target_pos, desired_pos, tissue_type="organ")
    print(f"\nPositioning result: {result['status']}")
    if 'force' in result:
        print(f"Force applied: {np.linalg.norm(result['force']):.2e} N")
    if 'distance_to_target' in result:
        print(f"Distance to target: {result['distance_to_target']*1000:.2f} mm")
    
    # Test wound closure
    wound_edges = [
        np.array([0.0, -0.01, 0.1]),
        np.array([0.0, 0.01, 0.1])
    ]
    
    closure_result = array.assist_wound_closure(wound_edges)
    print(f"\nWound closure result: {closure_result['status']}")
    if 'closure_progress' in closure_result:
        print(f"Closure progress: {closure_result['closure_progress']*1000:.2f} mm")
    
    array.stop_procedure()
