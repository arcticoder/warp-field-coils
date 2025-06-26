#!/usr/bin/env python3
"""
Superconducting Resonator Diagnostics System
Implements Step 4 of the roadmap: integrate superconducting resonator diagnostics
Based on superconducting_resonator.py from the prototype
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional, List, Callable
from dataclasses import dataclass
import scipy.constants as const
from scipy.signal import hilbert, butter, filtfilt
from scipy.optimize import curve_fit
import warnings

@dataclass 
class ResonatorConfig:
    """Configuration parameters for superconducting resonator."""
    base_frequency: float      # Base resonance frequency (Hz)
    cavity_volume: float       # Cavity volume (m³)
    quality_factor: float      # Quality factor Q
    coupling_strength: float   # Coupling strength
    temperature: float         # Operating temperature (K)
    
@dataclass
class StressEnergyMeasurement:
    """Results from stress-energy tensor measurement."""
    T00_measured: np.ndarray   # Measured T₀₀ component
    T00_vacuum: float          # Vacuum energy density
    field_i: np.ndarray        # In-phase field quadrature
    field_q: np.ndarray        # Quadrature field component
    time_array: np.ndarray     # Time array
    measurement_uncertainty: np.ndarray  # Uncertainty estimates
    signal_to_noise: float     # Signal-to-noise ratio

class SuperconductingResonatorDiagnostics:
    """
    Advanced superconducting resonator system for in-situ stress-energy tensor measurement.
    Implements Step 4 of the roadmap.
    """
    
    def __init__(self, config: ResonatorConfig):
        """
        Initialize resonator diagnostics system.
        
        Args:
            config: ResonatorConfig with system parameters
        """
        self.config = config
        
        # Physical constants
        self.hbar = const.hbar
        self.c = const.c
        self.eps0 = const.epsilon_0
        self.mu0 = const.mu_0
        self.kb = const.k
        
        # Derived parameters
        self.omega0 = 2 * np.pi * config.base_frequency
        self.cavity_length = (config.cavity_volume / np.pi)**(1/3)  # Approximate spherical cavity
        self.vacuum_energy_density = self._compute_vacuum_energy_density()
        
        # Calibration parameters
        self.calibration_factor = 1.0  # To be determined experimentally
        self.field_calibration = {}
        
    def _compute_vacuum_energy_density(self) -> float:
        """Compute vacuum energy density for the resonator cavity."""
        # Vacuum energy density: ρ_vac = (1/2)ℏω₀/V
        return 0.5 * self.hbar * self.omega0 / self.config.cavity_volume
    
    def _extract_stress_energy_tensor(self, field_i: np.ndarray, 
                                    field_q: np.ndarray, 
                                    time_array: np.ndarray) -> np.ndarray:
        """
        Extract T₀₀ component of stress-energy tensor from field measurements.
        
        For electromagnetic fields:
        T₀₀ = (1/2μ₀)[B² + ε₀E²] - ⟨T₀₀⟩_vacuum
        
        Args:
            field_i: In-phase field quadrature measurements
            field_q: Quadrature field component measurements  
            time_array: Time array for measurements
            
        Returns:
            T₀₀ component array
        """
        # Convert field quadratures to E and B fields
        # This is a simplified model - real system requires careful calibration
        
        # Electric field (proportional to field quadratures)
        # E ~ √(ℏω₀/2ε₀V) * field_amplitude
        field_amplitude = np.sqrt(field_i**2 + field_q**2)
        E_field = field_amplitude * np.sqrt(
            self.hbar * self.omega0 / (2 * self.eps0 * self.config.cavity_volume)
        )
        
        # Magnetic field (from time derivative of quadrature)
        if len(field_q) > 1:
            dt = time_array[1] - time_array[0]
            dfield_dt = np.gradient(field_q, dt)
            # B ~ (1/c²) * dE/dt for propagating fields
            B_field = -dfield_dt / self.c**2
        else:
            B_field = np.zeros_like(field_q)
        
        # Stress-energy tensor T₀₀ component
        energy_density = 0.5 * (self.eps0 * E_field**2 + B_field**2 / self.mu0)
        
        # Subtract vacuum contribution
        T00 = energy_density - self.vacuum_energy_density
        
        return T00
    
    def calibrate_field_measurements(self, known_field_values: Dict[str, float],
                                   measured_quadratures: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Calibrate field measurements using known reference values.
        
        Args:
            known_field_values: Dictionary of known field strengths
            measured_quadratures: Corresponding measured quadrature values
            
        Returns:
            Calibration factors for field conversion
        """
        calibration_factors = {}
        
        for field_type in ['E_field', 'B_field']:
            if field_type in known_field_values and field_type in measured_quadratures:
                known_val = known_field_values[field_type]
                measured_val = np.mean(np.abs(measured_quadratures[field_type]))
                
                if measured_val > 0:
                    calibration_factors[field_type] = known_val / measured_val
                else:
                    calibration_factors[field_type] = 1.0
        
        self.field_calibration.update(calibration_factors)
        return calibration_factors
    
    def measure_stress_energy_real_time(self, measurement_duration: float,
                                      sampling_rate: float,
                                      apply_filtering: bool = True) -> StressEnergyMeasurement:
        """
        Perform real-time stress-energy tensor measurement.
        
        Args:
            measurement_duration: Total measurement time (s)
            sampling_rate: Sampling rate (Hz)
            apply_filtering: Whether to apply noise filtering
            
        Returns:
            StressEnergyMeasurement with complete results
        """
        # Generate time array
        n_samples = int(measurement_duration * sampling_rate)
        time_array = np.linspace(0, measurement_duration, n_samples)
        
        # Simulate field measurements (in real system, this would be actual data acquisition)
        field_i, field_q = self._simulate_field_measurements(time_array)
        
        # Apply filtering if requested
        if apply_filtering:
            field_i = self._apply_noise_filter(field_i, sampling_rate)
            field_q = self._apply_noise_filter(field_q, sampling_rate)
        
        # Extract stress-energy tensor
        T00_measured = self._extract_stress_energy_tensor(field_i, field_q, time_array)
        
        # Estimate measurement uncertainty
        measurement_uncertainty = self._estimate_measurement_uncertainty(
            field_i, field_q, time_array
        )
        
        # Calculate signal-to-noise ratio
        signal_power = np.mean(T00_measured**2)
        noise_power = np.mean(measurement_uncertainty**2)
        snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else np.inf
        
        return StressEnergyMeasurement(
            T00_measured=T00_measured,
            T00_vacuum=self.vacuum_energy_density,
            field_i=field_i,
            field_q=field_q,
            time_array=time_array,
            measurement_uncertainty=measurement_uncertainty,
            signal_to_noise=snr
        )
    
    def _simulate_field_measurements(self, time_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate field measurements for testing (replace with real data acquisition).
        
        Args:
            time_array: Time array for simulation
            
        Returns:
            Tuple of (field_i, field_q) simulated measurements
        """
        # Simulate coherent oscillation + noise
        omega = self.omega0
        
        # Signal components
        signal_i = 0.1 * np.cos(omega * time_array + 0.1 * np.sin(10 * omega * time_array))
        signal_q = 0.1 * np.sin(omega * time_array + 0.1 * np.sin(10 * omega * time_array))
        
        # Add thermal noise
        thermal_noise_scale = np.sqrt(self.kb * self.config.temperature / 
                                    (self.hbar * self.omega0))
        noise_i = np.random.normal(0, thermal_noise_scale * 0.01, len(time_array))
        noise_q = np.random.normal(0, thermal_noise_scale * 0.01, len(time_array))
        
        # Add technical noise (1/f, shot noise, etc.)
        technical_noise_i = self._generate_technical_noise(time_array) * 0.005
        technical_noise_q = self._generate_technical_noise(time_array) * 0.005
        
        field_i = signal_i + noise_i + technical_noise_i
        field_q = signal_q + noise_q + technical_noise_q
        
        return field_i, field_q
    
    def _generate_technical_noise(self, time_array: np.ndarray) -> np.ndarray:
        """Generate realistic technical noise (1/f + white)."""
        n_samples = len(time_array)
        frequencies = np.fft.fftfreq(n_samples, time_array[1] - time_array[0])
        frequencies[0] = 1e-10  # Avoid division by zero
        
        # 1/f noise spectrum
        noise_spectrum = 1.0 / np.abs(frequencies)
        noise_spectrum[0] = noise_spectrum[1]  # Fix DC component
        
        # Generate random phases
        phases = np.random.uniform(-np.pi, np.pi, n_samples)
        complex_noise = noise_spectrum * np.exp(1j * phases)
        
        # Take real part and normalize
        technical_noise = np.real(np.fft.ifft(complex_noise))
        technical_noise = technical_noise / np.std(technical_noise)
        
        return technical_noise
    
    def _apply_noise_filter(self, signal: np.ndarray, sampling_rate: float) -> np.ndarray:
        """Apply low-pass filter to reduce high-frequency noise."""
        # Design low-pass Butterworth filter
        nyquist = sampling_rate / 2
        cutoff = min(self.config.base_frequency * 5, nyquist * 0.8)  # 5x resonance frequency
        normalized_cutoff = cutoff / nyquist
        
        b, a = butter(4, normalized_cutoff, btype='low')
        filtered_signal = filtfilt(b, a, signal)
        
        return filtered_signal
    
    def _estimate_measurement_uncertainty(self, field_i: np.ndarray, 
                                        field_q: np.ndarray, 
                                        time_array: np.ndarray) -> np.ndarray:
        """
        Estimate measurement uncertainty using statistical methods.
        
        Args:
            field_i: In-phase field measurements
            field_q: Quadrature field measurements
            time_array: Time array
            
        Returns:
            Uncertainty estimates for each measurement point
        """
        # Estimate noise level from high-frequency components
        dt = time_array[1] - time_array[0]
        
        # High-pass filter to extract noise
        nyquist = 1 / (2 * dt)
        cutoff = self.config.base_frequency * 2
        normalized_cutoff = cutoff / nyquist
        
        if normalized_cutoff < 1.0:
            b, a = butter(2, normalized_cutoff, btype='high')
            noise_i = filtfilt(b, a, field_i)
            noise_q = filtfilt(b, a, field_q)
            
            # Estimate noise standard deviation
            noise_std = np.sqrt(np.var(noise_i) + np.var(noise_q))
        else:
            noise_std = 0.01 * np.std(field_i)  # Fallback estimate
        
        # Propagate uncertainty through stress-energy calculation
        # This is a simplified uncertainty propagation
        field_amplitude = np.sqrt(field_i**2 + field_q**2)
        relative_uncertainty = noise_std / (field_amplitude + 1e-10)
        
        # Convert to stress-energy uncertainty
        T00_base = self._extract_stress_energy_tensor(field_i, field_q, time_array)
        uncertainty = np.abs(T00_base) * relative_uncertainty
        
        return uncertainty
    
    def compare_with_target(self, measurement: StressEnergyMeasurement,
                          target_T00: np.ndarray,
                          target_r_array: np.ndarray) -> Dict:
        """
        Compare measured T₀₀ with target profile.
        
        Args:
            measurement: StressEnergyMeasurement from real-time measurement
            target_T00: Target stress-energy profile
            target_r_array: Radial coordinates for target profile
            
        Returns:
            Comparison metrics and analysis
        """
        # For simplicity, assume spatial measurement corresponds to time evolution
        # In a real system, this would require proper spatial-temporal mapping
        
        # Interpolate measurement to target grid
        from scipy.interpolate import interp1d
        
        if len(measurement.T00_measured) != len(target_r_array):
            # Interpolate to common grid
            measurement_grid = np.linspace(0, 1, len(measurement.T00_measured))
            target_grid = np.linspace(0, 1, len(target_r_array))
            
            interp_func = interp1d(measurement_grid, measurement.T00_measured,
                                 kind='linear', bounds_error=False, fill_value=0)
            T00_interp = interp_func(target_grid)
        else:
            T00_interp = measurement.T00_measured
        
        # Calculate comparison metrics
        rmse = np.sqrt(np.mean((T00_interp - target_T00)**2))
        mae = np.mean(np.abs(T00_interp - target_T00))
        
        # Correlation coefficient
        correlation = np.corrcoef(T00_interp, target_T00)[0, 1]
        
        # Chi-squared test (if uncertainties available)
        if measurement.measurement_uncertainty is not None:
            uncertainty_interp = interp1d(measurement_grid, measurement.measurement_uncertainty,
                                        kind='linear', bounds_error=False, fill_value=np.inf)(target_grid)
            chi_squared = np.sum(((T00_interp - target_T00) / uncertainty_interp)**2)
            dof = len(target_T00) - 1  # degrees of freedom
            reduced_chi_squared = chi_squared / dof
        else:
            chi_squared = np.inf
            reduced_chi_squared = np.inf
        
        return {
            'rmse': rmse,
            'mae': mae,
            'correlation': correlation,
            'chi_squared': chi_squared,
            'reduced_chi_squared': reduced_chi_squared,
            'interpolated_measurement': T00_interp,
            'agreement_quality': 'excellent' if rmse < 0.1 * np.std(target_T00) else
                               'good' if rmse < 0.5 * np.std(target_T00) else
                               'poor'
        }
    
    def plot_measurement_results(self, measurement: StressEnergyMeasurement,
                               target_T00: Optional[np.ndarray] = None,
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot comprehensive measurement results.
        
        Args:
            measurement: StressEnergyMeasurement results
            target_T00: Optional target profile for comparison
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure object
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        time = measurement.time_array
        
        # Plot field quadratures
        ax1.plot(time, measurement.field_i, 'b-', linewidth=1, label='I-quadrature')
        ax1.plot(time, measurement.field_q, 'r-', linewidth=1, label='Q-quadrature')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Field Amplitude')
        ax1.set_title('Field Quadrature Measurements')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot measured T₀₀
        ax2.plot(time, measurement.T00_measured, 'g-', linewidth=2, label='Measured $T_{00}$')
        if measurement.measurement_uncertainty is not None:
            ax2.fill_between(time, 
                           measurement.T00_measured - measurement.measurement_uncertainty,
                           measurement.T00_measured + measurement.measurement_uncertainty,
                           alpha=0.3, color='gray', label='Uncertainty')
        
        ax2.axhline(y=measurement.T00_vacuum, color='k', linestyle='--', 
                   label='Vacuum level')
        
        if target_T00 is not None:
            # Assume target corresponds to spatial profile, map to time
            target_time = np.linspace(time[0], time[-1], len(target_T00))
            ax2.plot(target_time, target_T00, 'r--', linewidth=2, label='Target $T_{00}$')
        
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('$T_{00}$ (J/m³)')
        ax2.set_title(f'Stress-Energy Measurement (SNR: {measurement.signal_to_noise:.1f} dB)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Power spectral density
        from scipy.signal import welch
        f_field, psd_field = welch(measurement.field_i, fs=1/(time[1]-time[0]))
        f_T00, psd_T00 = welch(measurement.T00_measured, fs=1/(time[1]-time[0]))
        
        ax3.loglog(f_field, psd_field, 'b-', label='Field PSD')
        ax3.axvline(x=self.config.base_frequency, color='r', linestyle='--', 
                   label='Resonance frequency')
        ax3.set_xlabel('Frequency (Hz)')
        ax3.set_ylabel('Power Spectral Density')
        ax3.set_title('Field Quadrature PSD')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # T₀₀ power spectral density
        ax4.loglog(f_T00, psd_T00, 'g-', label='$T_{00}$ PSD')
        ax4.set_xlabel('Frequency (Hz)')
        ax4.set_ylabel('Power Spectral Density')
        ax4.set_title('Stress-Energy PSD')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig

if __name__ == "__main__":
    # Example usage
    config = ResonatorConfig(
        base_frequency=1e9,    # 1 GHz
        cavity_volume=1e-6,    # 1 cm³
        quality_factor=1e6,    # High-Q superconducting cavity
        coupling_strength=0.1,
        temperature=0.01       # 10 mK
    )
    
    diagnostics = SuperconductingResonatorDiagnostics(config)
    
    # Perform measurement
    measurement = diagnostics.measure_stress_energy_real_time(
        measurement_duration=1e-3,  # 1 ms
        sampling_rate=10e9,         # 10 GHz sampling
        apply_filtering=True
    )
    
    print("Stress-Energy Measurement Results:")
    print(f"Signal-to-noise ratio: {measurement.signal_to_noise:.1f} dB")
    print(f"Vacuum energy density: {measurement.T00_vacuum:.2e} J/m³")
    print(f"Mean measured T₀₀: {np.mean(measurement.T00_measured):.2e} J/m³")
    print(f"RMS fluctuation: {np.std(measurement.T00_measured):.2e} J/m³")
    
    # Create target profile for comparison
    target_r = np.linspace(0.1, 2.0, len(measurement.T00_measured))
    target_T00 = -0.1 * np.exp(-((target_r - 1.0)/0.3)**2)  # Negative energy shell
    
    # Compare with target
    comparison = diagnostics.compare_with_target(measurement, target_T00, target_r)
    print(f"\nComparison with target:")
    print(f"RMSE: {comparison['rmse']:.2e}")
    print(f"Correlation: {comparison['correlation']:.3f}")
    print(f"Agreement quality: {comparison['agreement_quality']}")
    
    # Plot results
    fig = diagnostics.plot_measurement_results(measurement, target_T00, 
                                             save_path="resonator_diagnostics.png")
    plt.show()
