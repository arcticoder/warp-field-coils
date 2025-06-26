"""
Hardware Control Module

Real-time electromagnetic field control and actuator interfaces.
Provides high-frequency current modulation and field monitoring.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
import threading
import time
import logging
from abc import ABC, abstractmethod


@dataclass
class HardwareConfig:
    """Hardware configuration parameters."""
    max_current: float = 1000.0          # Maximum current (A)
    max_voltage: float = 1000.0          # Maximum voltage (V)
    control_frequency: float = 1e6       # Control loop frequency (Hz)
    safety_margin: float = 0.9           # Safety factor for limits
    thermal_limit: float = 85.0          # Temperature limit (°C)


@dataclass
class ActuatorState:
    """Current state of electromagnetic actuator."""
    current: float = 0.0                 # Current output (A)
    voltage: float = 0.0                 # Voltage output (V)
    temperature: float = 25.0            # Temperature (°C)
    field_strength: float = 0.0          # Measured field (T)
    power: float = 0.0                   # Power consumption (W)
    enabled: bool = False                # Actuator enabled status


class HardwareActuator(ABC):
    """Abstract base class for electromagnetic actuators."""
    
    def __init__(self, actuator_id: str, config: HardwareConfig):
        """Initialize hardware actuator."""
        self.actuator_id = actuator_id
        self.config = config
        self.state = ActuatorState()
        self.logger = logging.getLogger(f"{__name__}.{actuator_id}")
        self._control_thread = None
        self._running = False
    
    @abstractmethod
    def set_output(self, value: float) -> bool:
        """Set actuator output value."""
        pass
    
    @abstractmethod
    def read_feedback(self) -> Dict:
        """Read actuator feedback sensors."""
        pass
    
    @abstractmethod
    def emergency_shutdown(self) -> bool:
        """Emergency shutdown of actuator."""
        pass
    
    def enable(self) -> bool:
        """Enable actuator operation."""
        try:
            self.state.enabled = True
            self.logger.info(f"✅ Actuator {self.actuator_id} enabled")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to enable actuator {self.actuator_id}: {e}")
            return False
    
    def disable(self) -> bool:
        """Disable actuator operation."""
        try:
            self.state.enabled = False
            self.set_output(0.0)  # Set to zero output
            self.logger.info(f"🔴 Actuator {self.actuator_id} disabled")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to disable actuator {self.actuator_id}: {e}")
            return False


class CurrentDriver(HardwareActuator):
    """
    High-current electromagnetic coil driver.
    
    Provides precise current control for electromagnetic coils with
    real-time feedback and safety monitoring.
    """
    
    def __init__(self, actuator_id: str, config: HardwareConfig, 
                 coil_inductance: float = 1e-3, coil_resistance: float = 0.1):
        """Initialize current driver."""
        super().__init__(actuator_id, config)
        self.coil_inductance = coil_inductance  # Henry
        self.coil_resistance = coil_resistance  # Ohm
        self.target_current = 0.0
        self.current_controller = self._create_current_controller()
    
    def _create_current_controller(self):
        """Create PID controller for current regulation."""
        class PIDController:
            def __init__(self, kp=10.0, ki=100.0, kd=0.1):
                self.kp = kp
                self.ki = ki
                self.kd = kd
                self.integral = 0.0
                self.last_error = 0.0
                self.dt = 1.0 / config.control_frequency
            
            def update(self, error):
                self.integral += error * self.dt
                derivative = (error - self.last_error) / self.dt
                
                output = (self.kp * error + 
                         self.ki * self.integral + 
                         self.kd * derivative)
                
                self.last_error = error
                return output
        
        return PIDController()
    
    def set_output(self, current: float) -> bool:
        """Set target current output."""
        if not self.state.enabled:
            self.logger.warning(f"Actuator {self.actuator_id} not enabled")
            return False
        
        # Safety checks
        if abs(current) > self.config.max_current * self.config.safety_margin:
            self.logger.error(f"Current {current} A exceeds safety limit")
            return False
        
        self.target_current = current
        
        # Simulate current regulation
        error = self.target_current - self.state.current
        voltage_adjustment = self.current_controller.update(error)
        
        # Apply voltage limits
        required_voltage = min(abs(voltage_adjustment), 
                             self.config.max_voltage * self.config.safety_margin)
        
        # Simulate current response (first-order system)
        tau = self.coil_inductance / self.coil_resistance
        dt = 1.0 / self.config.control_frequency
        alpha = dt / (tau + dt)
        
        self.state.current += alpha * (self.target_current - self.state.current)
        self.state.voltage = required_voltage * np.sign(voltage_adjustment)
        
        # Update power and temperature
        self.state.power = self.state.current**2 * self.coil_resistance
        self.state.temperature += self.state.power * 0.001  # Simplified heating
        
        return True
    
    def read_feedback(self) -> Dict:
        """Read current driver feedback."""
        # Simulate sensor readings with noise
        current_noise = np.random.normal(0, 0.01)  # 10 mA noise
        voltage_noise = np.random.normal(0, 0.1)   # 0.1 V noise
        
        return {
            'current': self.state.current + current_noise,
            'voltage': self.state.voltage + voltage_noise,
            'power': self.state.power,
            'temperature': self.state.temperature,
            'efficiency': 0.95 if self.state.current > 0 else 0.0
        }
    
    def emergency_shutdown(self) -> bool:
        """Emergency shutdown of current driver."""
        try:
            self.target_current = 0.0
            self.state.current = 0.0
            self.state.voltage = 0.0
            self.state.enabled = False
            self.logger.warning(f"🚨 Emergency shutdown: {self.actuator_id}")
            return True
        except Exception as e:
            self.logger.error(f"❌ Emergency shutdown failed: {e}")
            return False


class FieldActuator(HardwareActuator):
    """
    Electromagnetic field shaping actuator.
    
    Provides precise magnetic field control through multiple coil arrays
    with real-time field monitoring and feedback.
    """
    
    def __init__(self, actuator_id: str, config: HardwareConfig, 
                 coil_geometry: Dict):
        """Initialize field actuator."""
        super().__init__(actuator_id, config)
        self.coil_geometry = coil_geometry
        self.target_field = 0.0
        self.field_controller = self._create_field_controller()
        self.coil_currents = np.zeros(coil_geometry.get('n_coils', 4))
    
    def _create_field_controller(self):
        """Create multi-input controller for field regulation."""
        class MultiInputController:
            def __init__(self, n_inputs=4):
                self.n_inputs = n_inputs
                self.kp = 50.0
                self.ki = 200.0
                self.integral = 0.0
                self.dt = 1.0 / config.control_frequency
            
            def update(self, field_error, current_state):
                self.integral += field_error * self.dt
                
                # Proportional-integral control
                base_adjustment = self.kp * field_error + self.ki * self.integral
                
                # Distribute adjustment across coils
                adjustments = np.ones(self.n_inputs) * base_adjustment / self.n_inputs
                
                return current_state + adjustments
        
        return MultiInputController(self.coil_geometry.get('n_coils', 4))
    
    def set_output(self, field_strength: float) -> bool:
        """Set target field strength."""
        if not self.state.enabled:
            self.logger.warning(f"Actuator {self.actuator_id} not enabled")
            return False
        
        # Safety checks
        max_field = 10.0  # 10 Tesla limit
        if abs(field_strength) > max_field * self.config.safety_margin:
            self.logger.error(f"Field {field_strength} T exceeds safety limit")
            return False
        
        self.target_field = field_strength
        
        # Compute required coil currents
        field_error = self.target_field - self.state.field_strength
        new_currents = self.field_controller.update(field_error, self.coil_currents)
        
        # Apply current limits
        max_current = self.config.max_current * self.config.safety_margin
        self.coil_currents = np.clip(new_currents, -max_current, max_current)
        
        # Simulate field response
        # Simplified: field proportional to sum of currents
        total_current = np.sum(np.abs(self.coil_currents))
        field_constant = 0.001  # T/A
        self.state.field_strength = total_current * field_constant
        
        # Update power consumption
        total_power = np.sum(self.coil_currents**2) * 0.1  # Simplified resistance
        self.state.power = total_power
        self.state.temperature += total_power * 0.0001
        
        return True
    
    def read_feedback(self) -> Dict:
        """Read field actuator feedback."""
        # Simulate field sensor readings
        field_noise = np.random.normal(0, 0.001)  # 1 mT noise
        
        return {
            'field_strength': self.state.field_strength + field_noise,
            'field_uniformity': 0.02 + np.random.normal(0, 0.005),
            'coil_currents': self.coil_currents.tolist(),
            'total_power': self.state.power,
            'temperature': self.state.temperature,
            'field_gradient': np.random.normal(100, 10)  # T/m
        }
    
    def emergency_shutdown(self) -> bool:
        """Emergency shutdown of field actuator."""
        try:
            self.target_field = 0.0
            self.coil_currents.fill(0.0)
            self.state.field_strength = 0.0
            self.state.enabled = False
            self.logger.warning(f"🚨 Emergency shutdown: {self.actuator_id}")
            return True
        except Exception as e:
            self.logger.error(f"❌ Emergency shutdown failed: {e}")
            return False
    
    def get_field_map(self, grid_points: int = 20) -> Dict:
        """Compute magnetic field map around actuator."""
        # Create 3D grid
        x = np.linspace(-0.1, 0.1, grid_points)
        y = np.linspace(-0.1, 0.1, grid_points)
        z = np.linspace(-0.1, 0.1, grid_points)
        
        X, Y, Z = np.meshgrid(x, y, z)
        
        # Simplified field calculation
        # In practice, would use proper electromagnetic field solver
        Bx = np.zeros_like(X)
        By = np.zeros_like(X)
        Bz = np.ones_like(X) * self.state.field_strength
        
        return {
            'coordinates': {'x': X, 'y': Y, 'z': Z},
            'field_components': {'Bx': Bx, 'By': By, 'Bz': Bz},
            'field_magnitude': np.sqrt(Bx**2 + By**2 + Bz**2)
        }


class MultiActuatorController:
    """
    Coordinated control of multiple electromagnetic actuators.
    
    Provides system-level coordination and safety monitoring
    for arrays of current drivers and field actuators.
    """
    
    def __init__(self, config: HardwareConfig):
        """Initialize multi-actuator controller."""
        self.config = config
        self.actuators = {}
        self.logger = logging.getLogger(__name__)
        self.safety_monitor = SafetyMonitor(config)
        self._control_thread = None
        self._running = False
    
    def add_actuator(self, actuator: HardwareActuator) -> bool:
        """Add actuator to control system."""
        try:
            self.actuators[actuator.actuator_id] = actuator
            self.logger.info(f"✅ Added actuator: {actuator.actuator_id}")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to add actuator: {e}")
            return False
    
    def start_control_loop(self) -> bool:
        """Start coordinated control loop."""
        if self._running:
            self.logger.warning("Control loop already running")
            return False
        
        try:
            self._running = True
            self._control_thread = threading.Thread(target=self._control_loop, daemon=True)
            self._control_thread.start()
            self.logger.info("🚀 Control loop started")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to start control loop: {e}")
            self._running = False
            return False
    
    def stop_control_loop(self) -> bool:
        """Stop coordinated control loop."""
        try:
            self._running = False
            if self._control_thread:
                self._control_thread.join(timeout=1.0)
            self.logger.info("🛑 Control loop stopped")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to stop control loop: {e}")
            return False
    
    def _control_loop(self):
        """Main control loop for coordinated actuator control."""
        dt = 1.0 / self.config.control_frequency
        
        while self._running:
            start_time = time.time()
            
            try:
                # Read all actuator states
                actuator_states = {}
                for aid, actuator in self.actuators.items():
                    feedback = actuator.read_feedback()
                    actuator_states[aid] = feedback
                
                # Safety monitoring
                safety_status = self.safety_monitor.check_system_safety(actuator_states)
                
                if not safety_status['safe']:
                    self.logger.warning(f"Safety violation: {safety_status['violations']}")
                    self.emergency_shutdown_all()
                    break
                
                # System-level coordination would go here
                # For now, individual actuators handle their own control
                
            except Exception as e:
                self.logger.error(f"Control loop error: {e}")
            
            # Maintain control frequency
            elapsed = time.time() - start_time
            sleep_time = max(0, dt - elapsed)
            time.sleep(sleep_time)
    
    def emergency_shutdown_all(self) -> bool:
        """Emergency shutdown of all actuators."""
        success = True
        for actuator in self.actuators.values():
            if not actuator.emergency_shutdown():
                success = False
        
        self._running = False
        self.logger.warning("🚨 Emergency shutdown of all actuators")
        return success
    
    def get_system_status(self) -> Dict:
        """Get overall system status."""
        status = {
            'n_actuators': len(self.actuators),
            'control_running': self._running,
            'actuator_states': {},
            'system_health': 'unknown'
        }
        
        enabled_count = 0
        total_power = 0.0
        max_temperature = 0.0
        
        for aid, actuator in self.actuators.items():
            feedback = actuator.read_feedback()
            status['actuator_states'][aid] = {
                'enabled': actuator.state.enabled,
                'power': feedback.get('power', 0.0),
                'temperature': feedback.get('temperature', 25.0)
            }
            
            if actuator.state.enabled:
                enabled_count += 1
            total_power += feedback.get('power', 0.0)
            max_temperature = max(max_temperature, feedback.get('temperature', 25.0))
        
        # Determine system health
        if max_temperature > self.config.thermal_limit:
            status['system_health'] = 'overheating'
        elif total_power > 1e6:  # 1 MW limit
            status['system_health'] = 'high_power'
        elif enabled_count == 0:
            status['system_health'] = 'inactive'
        else:
            status['system_health'] = 'normal'
        
        status['total_power'] = total_power
        status['max_temperature'] = max_temperature
        status['enabled_actuators'] = enabled_count
        
        return status


class SafetyMonitor:
    """Safety monitoring system for electromagnetic actuators."""
    
    def __init__(self, config: HardwareConfig):
        """Initialize safety monitor."""
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.safety")
    
    def check_system_safety(self, actuator_states: Dict) -> Dict:
        """Check system-wide safety conditions."""
        violations = []
        
        total_power = 0.0
        max_temperature = 0.0
        max_current = 0.0
        
        for aid, state in actuator_states.items():
            power = state.get('power', 0.0)
            temperature = state.get('temperature', 25.0)
            current = state.get('current', 0.0)
            
            total_power += power
            max_temperature = max(max_temperature, temperature)
            max_current = max(max_current, abs(current))
            
            # Individual actuator checks
            if temperature > self.config.thermal_limit:
                violations.append(f"Thermal limit exceeded in {aid}: {temperature:.1f}°C")
            
            if abs(current) > self.config.max_current:
                violations.append(f"Current limit exceeded in {aid}: {current:.1f}A")
        
        # System-level checks
        if total_power > 1e6:  # 1 MW system limit
            violations.append(f"Total power limit exceeded: {total_power/1e6:.1f} MW")
        
        return {
            'safe': len(violations) == 0,
            'violations': violations,
            'total_power': total_power,
            'max_temperature': max_temperature,
            'max_current': max_current
        }


def run_hardware_demo():
    """Demonstration of hardware control capabilities."""
    print("⚡ Hardware Control Demo")
    print("=" * 50)
    
    # Create hardware configuration
    config = HardwareConfig(
        max_current=100.0,
        control_frequency=1000.0,  # 1 kHz for demo
        safety_margin=0.8
    )
    
    # Create actuators
    current_driver = CurrentDriver("coil_1", config)
    field_actuator = FieldActuator("field_1", config, {'n_coils': 4})
    
    # Multi-actuator controller
    controller = MultiActuatorController(config)
    controller.add_actuator(current_driver)
    controller.add_actuator(field_actuator)
    
    print("🔌 Enabling actuators...")
    current_driver.enable()
    field_actuator.enable()
    
    # Start control loop
    print("🚀 Starting control loop...")
    controller.start_control_loop()
    time.sleep(0.1)  # Let control loop start
    
    # Test current control
    print("📊 Testing current control...")
    current_driver.set_output(50.0)  # 50 A
    time.sleep(0.1)
    
    feedback = current_driver.read_feedback()
    print(f"   Current: {feedback['current']:.1f} A")
    print(f"   Power: {feedback['power']:.1f} W")
    print(f"   Temperature: {feedback['temperature']:.1f} °C")
    
    # Test field control
    print("\n🧲 Testing field control...")
    field_actuator.set_output(0.1)  # 0.1 T
    time.sleep(0.1)
    
    field_feedback = field_actuator.read_feedback()
    print(f"   Field strength: {field_feedback['field_strength']*1000:.1f} mT")
    print(f"   Field uniformity: {field_feedback['field_uniformity']*100:.1f}%")
    print(f"   Total power: {field_feedback['total_power']:.1f} W")
    
    # System status
    print("\n📈 System status:")
    status = controller.get_system_status()
    print(f"   Active actuators: {status['enabled_actuators']}")
    print(f"   Total power: {status['total_power']:.1f} W")
    print(f"   System health: {status['system_health']}")
    
    # Stop control loop
    print("\n🛑 Stopping control loop...")
    controller.stop_control_loop()
    
    print("\n🎯 Hardware demonstration complete!")


if __name__ == "__main__":
    run_hardware_demo()
