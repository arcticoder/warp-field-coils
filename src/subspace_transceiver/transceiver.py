"""
Subspace Transceiver Module
==========================

Implements subspace communication channel using waveguide dispersion mathematics.

Mathematical Foundation:
∂²ψ/∂t² - c_s²∇²ψ + κ²ψ = 0

Where:
- ψ: transceiver field amplitude
- c_s: subspace wave speed
- κ: dispersion/tuning constant
"""

import numpy as np
from scipy.integrate import solve_ivp
import logging
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass
import time

@dataclass
class SubspaceParams:
    """Parameters for subspace communication channel"""
    c_s: float = 3.0e8        # Subspace wave speed (m/s) - faster than light
    kappa: float = 1e-6       # Dispersion constant (1/m²)
    bandwidth: float = 1e12   # Channel bandwidth (Hz)
    power_limit: float = 1e6  # Maximum transmit power (W)
    noise_floor: float = 1e-12 # Receiver noise floor (W)
    
    # Grid parameters for field computation
    grid_resolution: int = 128
    domain_size: float = 1.0  # Spatial domain size (m)
    
    # Integration parameters
    rtol: float = 1e-6        # Relative tolerance
    atol: float = 1e-9        # Absolute tolerance

@dataclass
class TransmissionParams:
    """Parameters for a specific transmission"""
    frequency: float          # Carrier frequency (Hz)
    modulation_depth: float   # Modulation depth (0-1)
    duration: float           # Transmission duration (s)
    target_coordinates: Tuple[float, float, float]  # Target location
    priority: int = 1         # Message priority (1-10)

class SubspaceTransceiver:
    """
    Advanced subspace communication system
    
    Implements the subspace wave equation:
    ∂²ψ/∂t² = c_s²∇²ψ - κ²ψ
    
    Features:
    - Multi-dimensional wave propagation
    - Frequency modulation and demodulation
    - Signal processing and error correction
    - Power management and safety limits
    """
    
    def __init__(self, params: SubspaceParams):
        """
        Initialize subspace transceiver
        
        Args:
            params: Subspace communication parameters
        """
        self.params = params
        self.transmit_power = 0.0
        self.is_transmitting = False
        self.channel_status = "idle"
        
        # Create spatial grid for field computation
        self.x_grid = np.linspace(-params.domain_size/2, params.domain_size/2, params.grid_resolution)
        self.y_grid = np.linspace(-params.domain_size/2, params.domain_size/2, params.grid_resolution)
        self.X, self.Y = np.meshgrid(self.x_grid, self.y_grid)
        
        # Initialize field state
        self.field_state = np.zeros((params.grid_resolution, params.grid_resolution), dtype=complex)
        self.field_velocity = np.zeros_like(self.field_state)
        
        # Message queue for transmission management
        self.transmission_queue = []
        self.transmission_history = []
        self.total_transmissions = 0
        
        logging.info(f"SubspaceTransceiver initialized: c_s={params.c_s:.2e} m/s, κ={params.kappa:.2e}")

    def compute_laplacian_2d(self, field: np.ndarray) -> np.ndarray:
        """
        Compute 2D Laplacian using finite differences
        
        ∇²ψ = ∂²ψ/∂x² + ∂²ψ/∂y²
        
        Args:
            field: 2D field array
            
        Returns:
            2D Laplacian of the field
        """
        dx = self.x_grid[1] - self.x_grid[0]
        dy = self.y_grid[1] - self.y_grid[0]
        
        # Second derivatives using central differences
        d2_dx2 = np.zeros_like(field)
        d2_dy2 = np.zeros_like(field)
        
        # Interior points
        d2_dx2[1:-1, :] = (field[2:, :] - 2*field[1:-1, :] + field[:-2, :]) / dx**2
        d2_dy2[:, 1:-1] = (field[:, 2:] - 2*field[:, 1:-1] + field[:, :-2]) / dy**2
        
        # Boundary conditions (periodic)
        d2_dx2[0, :] = (field[1, :] - 2*field[0, :] + field[-1, :]) / dx**2
        d2_dx2[-1, :] = (field[0, :] - 2*field[-1, :] + field[-2, :]) / dx**2
        d2_dy2[:, 0] = (field[:, 1] - 2*field[:, 0] + field[:, -1]) / dy**2
        d2_dy2[:, -1] = (field[:, 0] - 2*field[:, -1] + field[:, -2]) / dy**2
        
        return d2_dx2 + d2_dy2

    def wave_equation_rhs(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        Right-hand side of the subspace wave equation
        
        d/dt [ψ, ∂ψ/∂t] = [∂ψ/∂t, c_s²∇²ψ - κ²ψ]
        
        Args:
            t: Current time
            y: State vector [field_real, field_imag, velocity_real, velocity_imag]
            
        Returns:
            Time derivatives
        """
        n = self.params.grid_resolution
        
        # Extract field and velocity components
        field_real = y[:n*n].reshape((n, n))
        field_imag = y[n*n:2*n*n].reshape((n, n))
        vel_real = y[2*n*n:3*n*n].reshape((n, n))
        vel_imag = y[3*n*n:4*n*n].reshape((n, n))
        
        # Complex field and velocity
        field = field_real + 1j * field_imag
        velocity = vel_real + 1j * vel_imag
        
        # Compute Laplacian
        laplacian = self.compute_laplacian_2d(field)
        
        # Wave equation: ∂²ψ/∂t² = c_s²∇²ψ - κ²ψ
        acceleration = self.params.c_s**2 * laplacian - self.params.kappa**2 * field
        
        # Pack derivatives
        derivatives = np.concatenate([
            velocity.real.flatten(),  # dψ_real/dt
            velocity.imag.flatten(),  # dψ_imag/dt
            acceleration.real.flatten(),  # d²ψ_real/dt²
            acceleration.imag.flatten()   # d²ψ_imag/dt²
        ])
        
        return derivatives

    def generate_carrier_wave(self, frequency: float, amplitude: float = 1.0) -> np.ndarray:
        """
        Generate carrier wave pattern in subspace
        
        ψ₀(x,y) = A * exp(i*k*r) * exp(-(r²)/(2σ²))
        
        Args:
            frequency: Carrier frequency
            amplitude: Wave amplitude
            
        Returns:
            Complex carrier wave field
        """
        # Wave number in subspace
        k = 2 * np.pi * frequency / self.params.c_s
        
        # Gaussian envelope to localize the wave
        sigma = self.params.domain_size / 8
        r_squared = self.X**2 + self.Y**2
        envelope = np.exp(-r_squared / (2 * sigma**2))
        
        # Carrier wave with circular wavefront
        r = np.sqrt(r_squared)
        carrier = amplitude * np.exp(1j * k * r) * envelope
        
        return carrier

    def modulate_signal(self, carrier: np.ndarray, message: str, modulation_type: str = "PSK") -> np.ndarray:
        """
        Apply modulation to carrier wave
        
        Args:
            carrier: Carrier wave field
            message: Message to encode
            modulation_type: Modulation scheme ("PSK", "FSK", "QAM")
            
        Returns:
            Modulated field
        """
        # Convert message to binary
        message_binary = ''.join(format(ord(char), '08b') for char in message)
        
        if modulation_type == "PSK":
            # Phase Shift Keying
            modulated = carrier.copy()
            phase_shift = np.pi  # 180 degree phase shift for '1'
            
            for i, bit in enumerate(message_binary):
                if bit == '1':
                    modulated *= np.exp(1j * phase_shift * (i + 1) / len(message_binary))
        
        elif modulation_type == "FSK":
            # Frequency Shift Keying
            modulated = carrier.copy()
            freq_shift = self.params.bandwidth * 0.1  # 10% frequency deviation
            
            for i, bit in enumerate(message_binary):
                if bit == '1':
                    t_local = i / len(message_binary)
                    modulated *= np.exp(1j * 2 * np.pi * freq_shift * t_local)
        
        else:  # Default to simple amplitude modulation
            bit_amplitude = 0.5 if message_binary[0] == '1' else 0.1
            modulated = carrier * bit_amplitude
        
        return modulated

    def transmit_message(self, message: str, transmission_params: TransmissionParams) -> Dict:
        """
        Transmit message through subspace channel
        
        Args:
            message: Message string to transmit
            transmission_params: Transmission parameters
            
        Returns:
            Transmission result dictionary
        """
        if self.is_transmitting:
            return {
                'success': False,
                'status': 'BUSY',
                'error': 'Transceiver busy'
            }
        
        # Check power limits
        estimated_power = len(message) * transmission_params.frequency * 1e-15  # Rough estimate
        if estimated_power > self.params.power_limit:
            return {
                'success': False,
                'error': 'Power limit exceeded',
                'estimated_power': estimated_power,
                'power_limit': self.params.power_limit
            }
        
        logging.info(f"Transmitting message: '{message[:50]}{'...' if len(message) > 50 else ''}'")
        
        start_time = time.time()
        self.is_transmitting = True
        self.transmit_power = estimated_power
        
        try:
            # Generate carrier wave
            carrier = self.generate_carrier_wave(transmission_params.frequency)
            
            # Modulate with message
            modulated_field = self.modulate_signal(carrier, message)
            
            # Set initial conditions for wave propagation
            self.field_state = modulated_field
            self.field_velocity = np.zeros_like(modulated_field)
            
            # Prepare state vector for integration
            n = self.params.grid_resolution
            y0 = np.concatenate([
                modulated_field.real.flatten(),
                modulated_field.imag.flatten(),
                self.field_velocity.real.flatten(),
                self.field_velocity.imag.flatten()
            ])
            
            # Propagate through subspace
            logging.info(f"Starting wave propagation (grid: {n}x{n}, ODE size: {len(y0)})")
            t_span = (0, transmission_params.duration)
            
            # Add progress monitoring
            def progress_callback(t, y):
                progress = t / transmission_params.duration * 100
                if int(progress) % 20 == 0:  # Log every 20%
                    logging.info(f"Wave propagation progress: {progress:.0f}%")
            
            solution = solve_ivp(
                self.wave_equation_rhs,
                t_span,
                y0,
                method='RK45',
                rtol=self.params.rtol,
                atol=self.params.atol,
                max_step=transmission_params.duration / 100,
                dense_output=False  # Save memory
            )
            logging.info("Wave propagation complete")
            
            if not solution.success:
                raise RuntimeError(f"Wave propagation failed: {solution.message}")
            
            # Extract final field state
            final_state = solution.y[:, -1]
            final_field_real = final_state[:n*n].reshape((n, n))
            final_field_imag = final_state[n*n:2*n*n].reshape((n, n))
            final_field = final_field_real + 1j * final_field_imag
            
            # Compute transmission metrics
            initial_energy = np.sum(np.abs(modulated_field)**2)
            final_energy = np.sum(np.abs(final_field)**2)
            transmission_efficiency = final_energy / initial_energy if initial_energy > 0 else 0
            
            signal_strength = np.max(np.abs(final_field))
            snr = signal_strength / self.params.noise_floor if self.params.noise_floor > 0 else float('inf')
            
            computation_time = time.time() - start_time
            
            result = {
                'success': True,
                'message_length': len(message),
                'transmission_time': transmission_params.duration,
                'computation_time': computation_time,
                'carrier_frequency': transmission_params.frequency,
                'transmission_efficiency': transmission_efficiency,
                'signal_to_noise_ratio': snr,
                'final_field_energy': final_energy,
                'power_used': estimated_power,
                'target_coordinates': transmission_params.target_coordinates,
                'solution_points': len(solution.t)
            }
            
            logging.info(f"Transmission complete: efficiency={transmission_efficiency:.3f}, SNR={snr:.1f} dB")
            
            return result
            
        except Exception as e:
            logging.error(f"Transmission failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'computation_time': time.time() - start_time
            }
            
        finally:
            self.is_transmitting = False
            self.transmit_power = 0.0

    def transmit_message_fast(self, message: str, transmission_params: TransmissionParams) -> Dict:
        """
        Fast message transmission with simplified physics (for testing)
        
        Args:
            message: Message string to transmit
            transmission_params: Transmission parameters
            
        Returns:
            Transmission result dictionary
        """
        if self.is_transmitting:
            return {
                'success': False,
                'status': 'BUSY',
                'error': 'Transceiver busy'
            }
        
        logging.info(f"Fast transmitting: '{message[:30]}{'...' if len(message) > 30 else ''}'")
        
        start_time = time.time()
        self.is_transmitting = True
        
        try:
            # Simplified transmission simulation
            estimated_power = len(message) * transmission_params.frequency * 1e-15
            
            if estimated_power > self.params.power_limit:
                return {
                    'success': False,
                    'status': 'POWER_EXCEEDED',
                    'error': 'Power limit exceeded'
                }
            
            # Simulate transmission delay based on distance and subspace speed
            distance = np.linalg.norm(transmission_params.target_coordinates)
            transmission_time = distance / self.params.c_s
            
            # Quick calculation without full wave propagation
            signal_strength = min(1.0, self.params.power_limit / (distance**2 + 1))
            transmission_id = f"TX_{int(time.time() * 1000) % 100000:05d}"
            
            # Record transmission
            self.transmission_history.append({
                'timestamp': time.time(),
                'message_length': len(message),
                'transmission_time': transmission_time,
                'signal_strength': signal_strength,
                'transmission_id': transmission_id
            })
            
            self.total_transmissions += 1
            self.transmit_power = estimated_power
            
            elapsed_time = time.time() - start_time
            
            return {
                'success': True,
                'status': 'TRANSMITTED',
                'transmission_id': transmission_id,
                'signal_strength_db': 20 * np.log10(signal_strength) if signal_strength > 0 else -100,
                'transmission_time': transmission_time,
                'processing_time': elapsed_time,
                'estimated_power': estimated_power
            }
            
        except Exception as e:
            logging.error(f"Transmission failed: {e}")
            return {
                'success': False,
                'status': 'FAILED',
                'error': str(e)
            }
        finally:
            self.is_transmitting = False

    def receive_message(self, duration: float) -> Dict:
        """
        Listen for incoming subspace transmissions
        
        Args:
            duration: Listen duration in seconds
            
        Returns:
            Reception result with decoded message
        """
        logging.info(f"Listening for subspace transmissions for {duration}s")
        
        # Simulate reception by analyzing current field state
        field_energy = np.sum(np.abs(self.field_state)**2)
        
        if field_energy < self.params.noise_floor * 100:
            return {
                'success': False,
                'message': None,
                'reason': 'No signal detected',
                'field_energy': field_energy,
                'noise_floor': self.params.noise_floor
            }
        
        # Simple signal analysis (in practice, would implement full demodulation)
        signal_strength = np.max(np.abs(self.field_state))
        snr = signal_strength / self.params.noise_floor
        
        # Decode based on field pattern (simplified)
        if snr > 10:  # Good signal
            decoded_message = "Incoming transmission detected - signal strong"
        elif snr > 3:  # Weak signal
            decoded_message = "Weak transmission detected - partial data"
        else:
            decoded_message = "Signal too weak to decode"
        
        return {
            'success': snr > 3,
            'message': decoded_message,
            'signal_strength': signal_strength,
            'snr_db': 20 * np.log10(snr),
            'field_energy': field_energy,
            'reception_duration': duration
        }

    def get_channel_status(self) -> Dict:
        """Get current channel status and diagnostics"""
        field_energy = np.sum(np.abs(self.field_state)**2)
        max_field = np.max(np.abs(self.field_state))
        
        return {
            'is_transmitting': self.is_transmitting,
            'transmit_power': self.transmit_power,
            'channel_status': self.channel_status,
            'field_energy': field_energy,
            'max_field_amplitude': max_field,
            'noise_floor': self.params.noise_floor,
            'bandwidth': self.params.bandwidth,
            'power_limit': self.params.power_limit,
            'queue_length': len(self.transmission_queue)
        }

    def run_diagnostics(self) -> Dict:
        """
        Run comprehensive transceiver diagnostics
        
        Returns:
            Diagnostic results
        """
        logging.info("Running subspace transceiver diagnostics")
        
        # Test basic wave propagation
        test_freq = 1e9  # 1 GHz test frequency
        test_carrier = self.generate_carrier_wave(test_freq, amplitude=0.1)
        
        # Measure wave propagation characteristics
        test_laplacian = self.compute_laplacian_2d(test_carrier)
        dispersion_measure = np.std(test_laplacian) / np.mean(np.abs(test_laplacian))
        
        # Test modulation
        test_message = "DIAGNOSTIC TEST"
        modulated = self.modulate_signal(test_carrier, test_message)
        modulation_quality = np.sum(np.abs(modulated)**2) / np.sum(np.abs(test_carrier)**2)
        
        # System health checks
        diagnostics = {
            'carrier_generation': 'PASS' if np.sum(np.abs(test_carrier)**2) > 0 else 'FAIL',
            'laplacian_computation': 'PASS' if np.all(np.isfinite(test_laplacian)) else 'FAIL',
            'modulation_system': 'PASS' if 0.5 < modulation_quality < 2.0 else 'FAIL',
            'dispersion_measure': dispersion_measure,
            'modulation_quality': modulation_quality,
            'wave_speed': self.params.c_s,
            'dispersion_constant': self.params.kappa,
            'grid_resolution': self.params.grid_resolution,
            'system_status': 'OPERATIONAL'
        }
        
        # Overall system health
        all_pass = all(result == 'PASS' for key, result in diagnostics.items() if key.endswith('_system') or key.endswith('_generation') or key.endswith('_computation'))
        diagnostics['overall_health'] = 'HEALTHY' if all_pass else 'DEGRADED'
        
        logging.info(f"Diagnostics complete: {diagnostics['overall_health']}")
        
        return diagnostics

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Initialize transceiver
    params = SubspaceParams(
        c_s=5e8,  # 5x speed of light
        kappa=1e-7,
        bandwidth=1e12
    )
    
    transceiver = SubspaceTransceiver(params)
    
    # Run diagnostics
    diag = transceiver.run_diagnostics()
    print("Subspace Transceiver Diagnostics:")
    for key, value in diag.items():
        print(f"  {key}: {value}")
    
    # Test transmission
    transmission = TransmissionParams(
        frequency=1e10,  # 10 GHz
        modulation_depth=0.8,
        duration=1e-6,   # 1 microsecond
        target_coordinates=(1000, 2000, 3000)  # 1000 km away
    )
    
    result = transceiver.transmit_message("Hello, subspace!", transmission)
    print(f"\nTransmission result: {result['success']}")
    if result['success']:
        print(f"  Efficiency: {result['transmission_efficiency']:.3f}")
        print(f"  SNR: {result['signal_to_noise_ratio']:.1f} dB")
