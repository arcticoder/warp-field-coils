#!/usr/bin/env python3
"""
Dynamic Trajectory Controller
Implements steerable acceleration/deceleration control for warp drive systems
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Callable, Optional, List
from dataclasses import dataclass
from scipy.optimize import minimize_scalar, root_scalar
from scipy.integrate import solve_ivp
import time

@dataclass
class TrajectoryParams:
    """Parameters for dynamic trajectory control."""
    effective_mass: float      # Effective mass of warp bubble system (kg)
    max_acceleration: float    # Maximum safe acceleration (m/s²)
    max_dipole_strength: float # Maximum dipole distortion parameter
    control_frequency: float   # Control loop frequency (Hz)
    integration_tolerance: float  # ODE integration tolerance

@dataclass
class TrajectoryState:
    """Current state of the trajectory."""
    time: float           # Current time (s)
    position: float       # Current position (m)
    velocity: float       # Current velocity (m/s)
    acceleration: float   # Current acceleration (m/s²)
    dipole_strength: float # Current dipole parameter ε
    bubble_radius: float  # Current bubble radius R(t)

class DynamicTrajectoryController:
    """
    Advanced trajectory controller for steerable warp drive acceleration/deceleration.
    
    Implements the control theory bridge between static optimization and dynamic
    trajectory following using:
    
    1. Equation of motion: m_eff * dv/dt = F_z(ε)
    2. Dipole-to-acceleration mapping: ε* = arg min |F_z(ε) - m_eff * a_target|²
    3. Time integration: v(t+Δt) = v(t) + a(t) * Δt
    4. Dynamic bubble radius: R(t) = R₀ + ∫v(τ)dτ
    """
    
    def __init__(self, params: TrajectoryParams, 
                 exotic_profiler, coil_optimizer):
        """
        Initialize dynamic trajectory controller.
        
        Args:
            params: Trajectory control parameters
            exotic_profiler: ExoticMatterProfiler instance
            coil_optimizer: AdvancedCoilOptimizer instance
        """
        self.params = params
        self.exotic_profiler = exotic_profiler
        self.coil_optimizer = coil_optimizer
        
        # Control system parameters
        self.dt = 1.0 / params.control_frequency
        self.history = {
            'time': [],
            'position': [],
            'velocity': [],
            'acceleration': [],
            'dipole_strength': [],
            'bubble_radius': [],
            'thrust_force': [],
            'control_error': []
        }
        
        # Initialize current state
        self.current_state = TrajectoryState(
            time=0.0,
            position=0.0,
            velocity=0.0,
            acceleration=0.0,
            dipole_strength=0.0,
            bubble_radius=2.0  # Default bubble radius
        )
        
        # Cached momentum flux computation for efficiency
        self._momentum_flux_cache = {}
        
    def compute_thrust_force(self, dipole_strength: float, 
                           bubble_radius: float = 2.0,
                           sigma: float = 1.0) -> float:
        """
        Compute thrust force F_z(ε) for given dipole strength.
        
        Implements: F_z(ε) = ∫ T^{0r}(r,θ;ε) cos(θ) r² sin(θ) dr dθ dφ
        
        Args:
            dipole_strength: Dipole parameter ε
            bubble_radius: Bubble radius R₀
            sigma: Profile sharpness parameter
            
        Returns:
            Thrust force in Newtons
        """
        # Check cache first
        cache_key = (dipole_strength, bubble_radius, sigma)
        if cache_key in self._momentum_flux_cache:
            return self._momentum_flux_cache[cache_key]
        
        try:
            from stress_energy.exotic_matter_profile import alcubierre_profile_dipole
            
            # Angular coordinates for integration
            theta_array = np.linspace(0, np.pi, 32)
            r_array = self.exotic_profiler.r_array
            
            # Generate dipolar warp profile
            f_profile = alcubierre_profile_dipole(
                r_array, theta_array, R0=bubble_radius, sigma=sigma, eps=dipole_strength
            )
            
            # Compute 3D momentum flux vector
            momentum_flux = self.exotic_profiler.compute_momentum_flux_vector(
                f_profile, r_array, theta_array
            )
            
            # Extract z-component (thrust force)
            thrust_force = momentum_flux[2]
            
            # Cache result
            self._momentum_flux_cache[cache_key] = thrust_force
            
            return thrust_force
            
        except Exception as e:
            print(f"⚠️ Thrust computation failed: {e}")
            return 0.0
    
    def solve_dipole_for_acceleration(self, target_acceleration: float,
                                    bubble_radius: float = 2.0,
                                    sigma: float = 1.0) -> Tuple[float, bool]:
        """
        Solve inverse problem: find ε* such that F_z(ε*) = m_eff * a_target.
        
        Implements: ε* = arg min |F_z(ε) - m_eff * a_target|²
        
        Args:
            target_acceleration: Desired acceleration (m/s²)
            bubble_radius: Bubble radius R₀
            sigma: Profile sharpness
            
        Returns:
            Tuple of (optimal_dipole_strength, success_flag)
        """
        target_force = self.params.effective_mass * target_acceleration
        
        def objective(eps):
            """Objective function for dipole optimization."""
            current_force = self.compute_thrust_force(eps, bubble_radius, sigma)
            error = (current_force - target_force)**2
            
            # Add penalty for large dipole strengths (physical limits)
            penalty = 1e6 * max(0, eps - self.params.max_dipole_strength)**2
            
            return error + penalty
        
        try:
            # Use bounded optimization to find optimal dipole strength
            result = minimize_scalar(
                objective,
                bounds=(0.0, self.params.max_dipole_strength),
                method='bounded'
            )
            
            if result.success and result.fun < 1e-6:
                return result.x, True
            else:
                # Fallback: use linear approximation
                eps_linear = abs(target_force) / (1e6 * self.params.effective_mass)
                eps_clamped = min(eps_linear, self.params.max_dipole_strength)
                return eps_clamped, False
                
        except Exception as e:
            print(f"⚠️ Dipole optimization failed: {e}")
            return 0.0, False
    
    def define_velocity_profile(self, profile_type: str = "smooth_acceleration",
                              duration: float = 10.0,
                              max_velocity: float = 100.0,
                              **kwargs) -> Callable[[float], float]:
        """
        Define desired velocity profile v(t) for trajectory following.
        
        Args:
            profile_type: Type of velocity profile
            duration: Total trajectory duration
            max_velocity: Maximum velocity
            **kwargs: Additional parameters
            
        Returns:
            Velocity function v(t)
        """
        if profile_type == "smooth_acceleration":
            # Smooth acceleration to max velocity, constant cruise, smooth deceleration
            accel_time = kwargs.get('accel_time', duration * 0.3)
            decel_time = kwargs.get('decel_time', duration * 0.3)
            cruise_time = duration - accel_time - decel_time
            
            def velocity_profile(t):
                if t < 0:
                    return 0.0
                elif t <= accel_time:
                    # Smooth acceleration using tanh profile
                    progress = t / accel_time
                    return max_velocity * 0.5 * (1 + np.tanh(4 * (progress - 0.5)))
                elif t <= accel_time + cruise_time:
                    # Constant cruise velocity
                    return max_velocity
                elif t <= duration:
                    # Smooth deceleration
                    decel_progress = (t - accel_time - cruise_time) / decel_time
                    return max_velocity * 0.5 * (1 - np.tanh(4 * (decel_progress - 0.5)))
                else:
                    return 0.0
                    
        elif profile_type == "sinusoidal":
            # Sinusoidal velocity profile
            frequency = kwargs.get('frequency', 0.1)
            
            def velocity_profile(t):
                if 0 <= t <= duration:
                    return max_velocity * np.sin(2 * np.pi * frequency * t)**2
                else:
                    return 0.0
                    
        elif profile_type == "step_response":
            # Step response for testing
            step_time = kwargs.get('step_time', duration * 0.1)
            
            def velocity_profile(t):
                if step_time <= t <= duration:
                    return max_velocity
                else:
                    return 0.0
                    
        else:
            raise ValueError(f"Unknown velocity profile type: {profile_type}")
        
        return velocity_profile
    
    def compute_acceleration_profile(self, velocity_func: Callable[[float], float],
                                   time_array: np.ndarray) -> np.ndarray:
        """
        Compute acceleration profile a(t) = dv/dt from velocity profile.
        
        Args:
            velocity_func: Velocity function v(t)
            time_array: Time points for evaluation
            
        Returns:
            Acceleration array a(t)
        """
        dt = time_array[1] - time_array[0] if len(time_array) > 1 else 0.01
        
        # Compute numerical derivative
        velocity_array = np.array([velocity_func(t) for t in time_array])
        acceleration_array = np.gradient(velocity_array, dt)
        
        # Clamp to maximum acceleration
        acceleration_array = np.clip(
            acceleration_array, 
            -self.params.max_acceleration, 
            self.params.max_acceleration
        )
        
        return acceleration_array
    
    def simulate_trajectory(self, velocity_func: Callable[[float], float],
                          simulation_time: float = 10.0,
                          initial_conditions: Optional[Dict] = None) -> Dict:
        """
        Simulate complete trajectory with dynamic control.
        
        Implements time integration:
        v(t+Δt) = v(t) + a(t)Δt
        R(t) = R₀ + ∫₀ᵗ v(τ)dτ
        
        Args:
            velocity_func: Desired velocity profile v(t)
            simulation_time: Total simulation duration
            initial_conditions: Initial state (optional)
            
        Returns:
            Simulation results dictionary
        """
        print(f"🚀 Simulating dynamic trajectory for {simulation_time:.1f}s")
        
        # Initialize state
        if initial_conditions:
            self.current_state.velocity = initial_conditions.get('velocity', 0.0)
            self.current_state.position = initial_conditions.get('position', 0.0)
            self.current_state.bubble_radius = initial_conditions.get('bubble_radius', 2.0)
        
        # Clear history
        for key in self.history.keys():
            self.history[key].clear()
        
        # Time array
        time_array = np.arange(0, simulation_time + self.dt, self.dt)
        
        # Pre-compute target acceleration profile
        target_velocities = np.array([velocity_func(t) for t in time_array])
        target_accelerations = self.compute_acceleration_profile(velocity_func, time_array)
        
        print(f"  Control frequency: {self.params.control_frequency:.1f} Hz")
        print(f"  Time steps: {len(time_array)}")
        print(f"  Max target acceleration: {np.max(np.abs(target_accelerations)):.2f} m/s²")
        
        # Main simulation loop
        start_time = time.time()
        
        for i, t in enumerate(time_array):
            # Get target acceleration at current time
            target_accel = target_accelerations[i]
            target_vel = target_velocities[i]
            
            # Solve for required dipole strength
            dipole_strength, solve_success = self.solve_dipole_for_acceleration(
                target_accel, self.current_state.bubble_radius
            )
            
            # Compute actual thrust force
            thrust_force = self.compute_thrust_force(
                dipole_strength, self.current_state.bubble_radius
            )
            
            # Compute actual acceleration
            actual_accel = thrust_force / self.params.effective_mass
            
            # Control error
            control_error = abs(actual_accel - target_accel)
            
            # Update state using Euler integration
            self.current_state.time = t
            self.current_state.acceleration = actual_accel
            self.current_state.dipole_strength = dipole_strength
            
            # Integrate velocity and position
            if i > 0:
                self.current_state.velocity += actual_accel * self.dt
                self.current_state.position += self.current_state.velocity * self.dt
                
                # Update bubble radius based on motion
                # R(t) = R₀ + displacement_factor * position
                displacement_factor = 0.1  # Coupling between motion and bubble size
                self.current_state.bubble_radius = (
                    2.0 + displacement_factor * abs(self.current_state.position)
                )
            
            # Store history
            self.history['time'].append(self.current_state.time)
            self.history['position'].append(self.current_state.position)
            self.history['velocity'].append(self.current_state.velocity)
            self.history['acceleration'].append(actual_accel)
            self.history['dipole_strength'].append(dipole_strength)
            self.history['bubble_radius'].append(self.current_state.bubble_radius)
            self.history['thrust_force'].append(thrust_force)
            self.history['control_error'].append(control_error)
            
            # Progress reporting
            if i % (len(time_array) // 10) == 0:
                progress = 100 * i / len(time_array)
                print(f"  Progress: {progress:.0f}% - "
                      f"v={self.current_state.velocity:.2f} m/s, "
                      f"ε={dipole_strength:.3f}, "
                      f"error={control_error:.2e}")
        
        simulation_time_elapsed = time.time() - start_time
        
        # Convert history to numpy arrays
        results = {}
        for key, values in self.history.items():
            results[key] = np.array(values)
        
        # Add metadata
        results['simulation_metadata'] = {
            'simulation_time': simulation_time,
            'dt': self.dt,
            'n_steps': len(time_array),
            'computation_time': simulation_time_elapsed,
            'target_velocities': target_velocities,
            'target_accelerations': target_accelerations
        }
        
        # Performance metrics
        velocity_tracking_error = np.mean(np.abs(results['velocity'] - target_velocities[:-1]))
        accel_tracking_error = np.mean(results['control_error'])
        max_dipole_used = np.max(results['dipole_strength'])
        
        results['performance_metrics'] = {
            'velocity_tracking_rms': velocity_tracking_error,
            'acceleration_tracking_rms': accel_tracking_error,
            'max_dipole_strength': max_dipole_used,
            'control_success_rate': np.sum(results['control_error'] < 0.1) / len(results['control_error'])
        }
        
        print(f"✓ Trajectory simulation complete")
        print(f"  Computation time: {simulation_time_elapsed:.3f}s")
        print(f"  Velocity tracking RMS: {velocity_tracking_error:.3f} m/s")
        print(f"  Acceleration tracking RMS: {accel_tracking_error:.3f} m/s²")
        print(f"  Max dipole strength: {max_dipole_used:.3f}")
        print(f"  Control success rate: {results['performance_metrics']['control_success_rate']*100:.1f}%")
        
        return results
    
    def plot_trajectory_results(self, results: Dict, 
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot comprehensive trajectory simulation results.
        
        Args:
            results: Simulation results from simulate_trajectory()
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        
        time_array = results['time']
        target_velocities = results['simulation_metadata']['target_velocities'][:-1]
        target_accelerations = results['simulation_metadata']['target_accelerations'][:-1]
        
        # 1. Velocity tracking
        axes[0, 0].plot(time_array, target_velocities, 'b--', linewidth=2, label='Target')
        axes[0, 0].plot(time_array, results['velocity'], 'r-', linewidth=1.5, label='Actual')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Velocity (m/s)')
        axes[0, 0].set_title('Velocity Trajectory Tracking')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Acceleration tracking
        axes[0, 1].plot(time_array, target_accelerations, 'b--', linewidth=2, label='Target')
        axes[0, 1].plot(time_array, results['acceleration'], 'r-', linewidth=1.5, label='Actual')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Acceleration (m/s²)')
        axes[0, 1].set_title('Acceleration Control')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Position evolution
        axes[1, 0].plot(time_array, results['position'], 'g-', linewidth=2)
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Position (m)')
        axes[1, 0].set_title('Position Evolution')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Dipole strength control
        axes[1, 1].plot(time_array, results['dipole_strength'], 'purple', linewidth=2)
        axes[1, 1].axhline(y=self.params.max_dipole_strength, color='r', linestyle='--', 
                          alpha=0.7, label='Max Limit')
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Dipole Strength ε')
        axes[1, 1].set_title('Dipole Control Signal')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 5. Thrust force
        axes[2, 0].plot(time_array, results['thrust_force'], 'orange', linewidth=2)
        axes[2, 0].set_xlabel('Time (s)')
        axes[2, 0].set_ylabel('Thrust Force (N)')
        axes[2, 0].set_title('Generated Thrust Force')
        axes[2, 0].grid(True, alpha=0.3)
        
        # 6. Control error
        axes[2, 1].semilogy(time_array, results['control_error'], 'red', linewidth=1.5)
        axes[2, 1].set_xlabel('Time (s)')
        axes[2, 1].set_ylabel('Control Error (m/s²)')
        axes[2, 1].set_title('Acceleration Tracking Error')
        axes[2, 1].grid(True, alpha=0.3)
        
        plt.suptitle('Dynamic Trajectory Control Results', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Trajectory plots saved to {save_path}")
        
        return fig
    
    def analyze_trajectory_performance(self, results: Dict) -> Dict:
        """
        Analyze trajectory control performance metrics.
        
        Args:
            results: Simulation results
            
        Returns:
            Performance analysis dictionary
        """
        analysis = {
            'tracking_performance': {},
            'control_authority': {},
            'efficiency_metrics': {},
            'stability_analysis': {}
        }
        
        time_array = results['time']
        target_velocities = results['simulation_metadata']['target_velocities'][:-1]
        target_accelerations = results['simulation_metadata']['target_accelerations'][:-1]
        
        # Tracking performance
        velocity_error = results['velocity'] - target_velocities
        acceleration_error = results['acceleration'] - target_accelerations
        
        analysis['tracking_performance'] = {
            'velocity_rms_error': np.sqrt(np.mean(velocity_error**2)),
            'velocity_max_error': np.max(np.abs(velocity_error)),
            'acceleration_rms_error': np.sqrt(np.mean(acceleration_error**2)),
            'acceleration_max_error': np.max(np.abs(acceleration_error)),
            'settling_time': self._compute_settling_time(velocity_error),
            'overshoot_percentage': self._compute_overshoot(results['velocity'], target_velocities)
        }
        
        # Control authority
        analysis['control_authority'] = {
            'max_dipole_strength': np.max(results['dipole_strength']),
            'dipole_utilization': np.max(results['dipole_strength']) / self.params.max_dipole_strength,
            'max_thrust_force': np.max(np.abs(results['thrust_force'])),
            'thrust_to_weight_ratio': np.max(np.abs(results['thrust_force'])) / (self.params.effective_mass * 9.81)
        }
        
        # Efficiency metrics
        total_energy = np.trapz(np.abs(results['thrust_force'] * results['velocity']), time_array)
        useful_kinetic_energy = 0.5 * self.params.effective_mass * np.max(results['velocity'])**2
        
        analysis['efficiency_metrics'] = {
            'total_energy_expenditure': total_energy,
            'useful_kinetic_energy': useful_kinetic_energy,
            'energy_efficiency': useful_kinetic_energy / (total_energy + 1e-12),
            'average_power': total_energy / (time_array[-1] - time_array[0]),
            'peak_power': np.max(np.abs(results['thrust_force'] * results['velocity']))
        }
        
        # Stability analysis
        control_signal_variance = np.var(results['dipole_strength'])
        steady_state_error = np.mean(np.abs(velocity_error[-10:]))  # Last 10 points
        
        analysis['stability_analysis'] = {
            'control_signal_variance': control_signal_variance,
            'steady_state_error': steady_state_error,
            'oscillation_frequency': self._estimate_oscillation_frequency(velocity_error),
            'damping_ratio': self._estimate_damping_ratio(velocity_error)
        }
        
        return analysis
    
    def _compute_settling_time(self, error_signal: np.ndarray, 
                             tolerance: float = 0.02) -> float:
        """Compute settling time for error signal."""
        error_envelope = np.abs(error_signal)
        settled_mask = error_envelope <= tolerance
        
        if np.any(settled_mask):
            first_settled_idx = np.argmax(settled_mask)
            # Check if it stays settled
            if np.all(settled_mask[first_settled_idx:]):
                return first_settled_idx * self.dt
        
        return float('inf')  # Never settled
    
    def _compute_overshoot(self, actual: np.ndarray, target: np.ndarray) -> float:
        """Compute percentage overshoot."""
        max_target = np.max(target)
        max_actual = np.max(actual)
        
        if max_target > 0:
            return 100 * (max_actual - max_target) / max_target
        else:
            return 0.0
    
    def _estimate_oscillation_frequency(self, signal: np.ndarray) -> float:
        """Estimate dominant oscillation frequency in signal."""
        try:
            from scipy import signal as sp_signal
            
            freqs, psd = sp_signal.periodogram(signal, fs=self.params.control_frequency)
            dominant_freq_idx = np.argmax(psd[1:]) + 1  # Skip DC component
            return freqs[dominant_freq_idx]
        except:
            return 0.0
    
    def _estimate_damping_ratio(self, signal: np.ndarray) -> float:
        """Estimate damping ratio from step response."""
        # Simplified estimation based on overshoot
        overshoot = self._compute_overshoot(signal, np.ones_like(signal))
        
        if overshoot > 0:
            # Relationship: overshoot = exp(-π*ζ/√(1-ζ²))
            # Solve for ζ approximately
            zeta = np.sqrt(1 / (1 + (np.pi / np.log(overshoot/100 + 1e-12))**2))
            return min(zeta, 1.0)
        else:
            return 1.0  # Overdamped
