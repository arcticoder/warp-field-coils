#!/usr/bin/env python3
"""
Closed-Loop Field Control System
Implements Step 5 of the roadmap: implement closed-loop field control
Based on warp-bubble-optimizer's control-loop framework
"""

import numpy as np
import scipy.signal as signal
import scipy.optimize as opt
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional, List, Callable
from dataclasses import dataclass
import control
from control import TransferFunction, feedback, step_response, bode_plot
import warnings

@dataclass
class ControllerParams:
    """PID controller parameters."""
    kp: float  # Proportional gain
    ki: float  # Integral gain  
    kd: float  # Derivative gain
    tau_d: float = 0.01  # Derivative filter time constant

@dataclass
class PlantParams:
    """Plant (coil system) parameters."""
    K: float      # DC gain
    omega_n: float  # Natural frequency (rad/s)
    zeta: float   # Damping ratio
    tau_delay: float = 0.0  # Time delay (s)

@dataclass
class ControlPerformance:
    """Control system performance metrics."""
    settling_time: float     # 2% settling time (s)
    overshoot: float        # Percentage overshoot
    steady_state_error: float  # Steady-state error
    gain_margin: float      # Gain margin (dB)
    phase_margin: float     # Phase margin (degrees)
    bandwidth: float        # Closed-loop bandwidth (Hz)
    disturbance_rejection: float  # Disturbance rejection ratio

class ClosedLoopFieldController:
    """
    Closed-loop field control system for warp field coils.
    Implements Step 5 of the roadmap with anomaly tracking and PID control.
    """
    
    def __init__(self, plant_params: PlantParams, sample_time: float = 1e-4):
        """
        Initialize the closed-loop controller.
        
        Args:
            plant_params: Plant model parameters
            sample_time: Control loop sampling time (s)
        """
        self.plant_params = plant_params
        self.sample_time = sample_time
        
        # Build plant transfer function G(s) = K / (s² + 2ζωₙs + ωₙ²)
        self.plant_tf = self._build_plant_model(plant_params)
        
        # Control system state
        self.controller_params = None
        self.closed_loop_tf = None
        self.performance_metrics = None
        
        # Anomaly tracking (from warp-bubble-optimizer framework)
        self.anomaly_history = []
        self.target_anomaly_threshold = 1e-6
        
        # Simulation state
        self.time_history = []
        self.reference_history = []
        self.output_history = []
        self.control_history = []
        self.error_history = []
        self.anomaly_history_time = []
    
    def _build_plant_model(self, params: PlantParams) -> TransferFunction:
        """Build plant transfer function from parameters."""
        # Second-order system: G(s) = K / (s² + 2ζωₙs + ωₙ²)
        num = [params.K]
        den = [1, 2*params.zeta*params.omega_n, params.omega_n**2]
        
        plant_tf = TransferFunction(num, den)
        
        # Add time delay if specified
        if params.tau_delay > 0:
            # Approximate delay with Padé approximation
            delay_num = [1, -params.tau_delay/2]
            delay_den = [1, params.tau_delay/2]
            delay_tf = TransferFunction(delay_num, delay_den)
            plant_tf = plant_tf * delay_tf
        
        return plant_tf
    
    def tune_pid_ziegler_nichols(self, method: str = 'ultimate_gain') -> ControllerParams:
        """
        Tune PID controller using Ziegler-Nichols method.
        
        Args:
            method: Tuning method ('ultimate_gain' or 'step_response')
            
        Returns:
            Tuned PID controller parameters
        """
        if method == 'ultimate_gain':
            # Find ultimate gain and period using root locus/frequency response
            K_u, T_u = self._find_ultimate_gain_period()
            
            # Ziegler-Nichols PID tuning rules
            kp = 0.6 * K_u
            ki = 2 * kp / T_u
            kd = kp * T_u / 8
            
        elif method == 'step_response':
            # Use step response characteristics
            L, T = self._find_step_response_params()
            
            # Ziegler-Nichols rules for step response
            kp = 1.2 * T / L
            ki = kp / (2 * L)
            kd = kp * 0.5 * L
            
        else:
            raise ValueError(f"Unknown tuning method: {method}")
        
        tau_d = 0.01  # Default derivative filter time constant
        
        controller_params = ControllerParams(kp=kp, ki=ki, kd=kd, tau_d=tau_d)
        self.controller_params = controller_params
        
        return controller_params
    
    def _find_ultimate_gain_period(self) -> Tuple[float, float]:
        """Find ultimate gain and period for Ziegler-Nichols tuning."""
        # Sweep gain to find stability boundary
        gains = np.logspace(-2, 2, 1000)
        
        for K_test in gains:
            # Test proportional controller with gain K_test
            controller_tf = TransferFunction([K_test], [1])
            open_loop_tf = controller_tf * self.plant_tf
            
            # Find poles of closed-loop system
            closed_loop_tf = feedback(open_loop_tf, 1)
            poles = closed_loop_tf.poles()
            
            # Check if any pole has zero real part (marginally stable)
            real_parts = np.real(poles)
            if np.any(np.abs(real_parts) < 1e-6):
                K_u = K_test
                # Period of oscillation from imaginary part
                imag_parts = np.imag(poles)
                omega_osc = np.max(np.abs(imag_parts))
                T_u = 2 * np.pi / omega_osc if omega_osc > 0 else 1.0
                return K_u, T_u
        
        # Fallback if no ultimate gain found
        return 1.0, 1.0
    
    def _find_step_response_params(self) -> Tuple[float, float]:
        """Find step response parameters L and T for Ziegler-Nichols tuning."""
        # Generate step response
        time_sim = np.linspace(0, 10/self.plant_params.omega_n, 1000)
        time_step, output_step = step_response(self.plant_tf, time_sim)
        
        # Find inflection point method parameters
        # L = delay time, T = time constant
        
        # Approximate delay and time constant from step response
        final_value = output_step[-1]
        
        # Find 10% and 90% rise times
        idx_10 = np.where(output_step >= 0.1 * final_value)[0]
        idx_90 = np.where(output_step >= 0.9 * final_value)[0]
        
        if len(idx_10) > 0 and len(idx_90) > 0:
            t_10 = time_step[idx_10[0]]
            t_90 = time_step[idx_90[0]]
            
            # Approximate L and T
            L = t_10  # Delay time
            T = (t_90 - t_10) / 0.8  # Time constant approximation
        else:
            # Fallback values
            L = 0.1 / self.plant_params.omega_n
            T = 1.0 / self.plant_params.omega_n
        
        return L, T
    
    def tune_pid_optimization(self, performance_weights: Dict[str, float] = None) -> ControllerParams:
        """
        Tune PID controller using optimization.
        
        Args:
            performance_weights: Weights for different performance criteria
            
        Returns:
            Optimized PID controller parameters
        """
        if performance_weights is None:
            performance_weights = {
                'settling_time': 1.0,
                'overshoot': 2.0,
                'steady_state_error': 3.0,
                'control_effort': 0.5
            }
        
        def objective(params):
            kp, ki, kd = params
            
            # Ensure positive gains
            if kp <= 0 or ki <= 0 or kd <= 0:
                return 1e10
            
            try:
                controller_params = ControllerParams(kp=kp, ki=ki, kd=kd)
                performance = self.analyze_performance(controller_params)
                
                # Weighted objective function
                objective_value = (
                    performance_weights['settling_time'] * performance.settling_time +
                    performance_weights['overshoot'] * performance.overshoot +
                    performance_weights['steady_state_error'] * performance.steady_state_error
                )
                
                # Add penalty for low stability margins
                if performance.gain_margin < 6:  # Less than 6 dB gain margin
                    objective_value += 1000
                if performance.phase_margin < 45:  # Less than 45° phase margin
                    objective_value += 1000
                
                return objective_value
            
            except:
                return 1e10
        
        # Initial guess from Ziegler-Nichols
        zn_params = self.tune_pid_ziegler_nichols()
        initial_guess = [zn_params.kp, zn_params.ki, zn_params.kd]
        
        # Optimization bounds
        bounds = [(1e-3, 100), (1e-3, 1000), (1e-6, 10)]
        
        # Run optimization
        result = opt.minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
        
        if result.success:
            kp_opt, ki_opt, kd_opt = result.x
            controller_params = ControllerParams(kp=kp_opt, ki=ki_opt, kd=kd_opt)
            self.controller_params = controller_params
            return controller_params
        else:
            # Fall back to Ziegler-Nichols if optimization fails
            return self.tune_pid_ziegler_nichols()
    
    def analyze_performance(self, controller_params: ControllerParams) -> ControlPerformance:
        """
        Analyze closed-loop performance metrics.
        
        Args:
            controller_params: PID controller parameters
            
        Returns:
            ControlPerformance with comprehensive metrics
        """
        # Build controller transfer function
        # PID with derivative filter: K(s) = kp + ki/s + kd*s/(τ_d*s + 1)
        pid_tf = self._build_pid_transfer_function(controller_params)
        
        # Form closed-loop system
        open_loop_tf = pid_tf * self.plant_tf
        closed_loop_tf = feedback(open_loop_tf, 1)
        self.closed_loop_tf = closed_loop_tf
        
        # Step response analysis
        time_sim = np.linspace(0, 20/self.plant_params.omega_n, 2000)
        time_step, output_step = step_response(closed_loop_tf, time_sim)
        
        # Calculate performance metrics
        settling_time = self._calculate_settling_time(time_step, output_step)
        overshoot = self._calculate_overshoot(output_step)
        steady_state_error = abs(1.0 - output_step[-1])
        
        # Frequency domain analysis
        try:
            gain_margin, phase_margin, _, _ = control.margin(open_loop_tf)
            gain_margin_db = 20 * np.log10(gain_margin) if gain_margin > 0 else -np.inf
            
            # Bandwidth calculation
            bandwidth = self._calculate_bandwidth(closed_loop_tf)
            
            # Disturbance rejection
            disturbance_tf = feedback(1, open_loop_tf)  # Transfer function from disturbance to output
            disturbance_rejection = np.abs(disturbance_tf.dcgain())
            
        except:
            gain_margin_db = 0
            phase_margin = 0
            bandwidth = 0
            disturbance_rejection = 1
        
        performance = ControlPerformance(
            settling_time=settling_time,
            overshoot=overshoot,
            steady_state_error=steady_state_error,
            gain_margin=gain_margin_db,
            phase_margin=phase_margin,
            bandwidth=bandwidth,
            disturbance_rejection=disturbance_rejection
        )
        
        self.performance_metrics = performance
        return performance
    
    def _build_pid_transfer_function(self, params: ControllerParams) -> TransferFunction:
        """Build PID transfer function with derivative filter."""
        # PID with derivative filter: K(s) = kp + ki/s + kd*s/(τ_d*s + 1)
        
        # Convert to single transfer function
        # K(s) = [kp*(τ_d*s + 1) + ki*(τ_d*s + 1)/s + kd*s] / (τ_d*s + 1)
        # K(s) = [kp*τ_d*s + kp + ki*τ_d + ki/s + kd*s] / (τ_d*s + 1)
        # K(s) = [s²*(kp*τ_d + kd) + s*(kp + ki*τ_d) + ki] / [s*(τ_d*s + 1)]
        
        num = [params.kp*params.tau_d + params.kd, params.kp + params.ki*params.tau_d, params.ki]
        den = [params.tau_d, 1, 0]  # s*(τ_d*s + 1) = τ_d*s² + s
        
        return TransferFunction(num, den)
    
    def _calculate_settling_time(self, time: np.ndarray, output: np.ndarray, 
                               tolerance: float = 0.02) -> float:
        """Calculate 2% settling time."""
        final_value = output[-1]
        settling_band = tolerance * final_value
        
        # Find last time output exits settling band
        outside_band = np.abs(output - final_value) > settling_band
        settling_indices = np.where(outside_band)[0]
        
        if len(settling_indices) > 0:
            settling_time = time[settling_indices[-1]]
        else:
            settling_time = time[0]  # Already settled
        
        return settling_time
    
    def _calculate_overshoot(self, output: np.ndarray) -> float:
        """Calculate percentage overshoot."""
        final_value = output[-1]
        max_value = np.max(output)
        
        if final_value > 0:
            overshoot = 100 * (max_value - final_value) / final_value
        else:
            overshoot = 0
        
        return max(0, overshoot)
    
    def _calculate_bandwidth(self, tf: TransferFunction) -> float:
        """Calculate -3dB bandwidth."""
        try:
            # Generate frequency response
            omega = np.logspace(-2, 4, 1000)
            mag, phase, omega_out = control.bode_plot(tf, omega, plot=False)
            
            # Find -3dB point (magnitude = 1/√2 of DC gain)
            dc_gain = np.abs(tf.dcgain())
            target_mag = dc_gain / np.sqrt(2)
            
            # Find where magnitude crosses target
            crossing_idx = np.where(mag <= target_mag)[0]
            if len(crossing_idx) > 0:
                bandwidth = omega_out[crossing_idx[0]] / (2 * np.pi)  # Convert to Hz
            else:
                bandwidth = omega_out[-1] / (2 * np.pi)  # Use highest frequency
            
        except:
            bandwidth = 1.0  # Fallback
        
        return bandwidth
    
    def compute_anomaly_measure(self, current_T00: np.ndarray, target_T00: np.ndarray,
                              G_tt: np.ndarray) -> float:
        """
        Compute anomaly measure |G_tt - 8π(T_m + T_int)| from warp-bubble framework.
        
        Args:
            current_T00: Current measured T₀₀ profile
            target_T00: Target T₀₀ profile (T_m)
            G_tt: Einstein tensor G_tt component
            
        Returns:
            Anomaly measure value
        """
        # T_int represents interaction term (in discrete framework, this would be 3nj coupling)
        T_int = current_T00 - target_T00  # Difference as interaction term
        
        # Anomaly measure: |G_tt - 8π(T_m + T_int)|
        anomaly = np.mean(np.abs(G_tt - 8 * np.pi * (target_T00 + T_int)))
        
        return anomaly
    
    def simulate_closed_loop(self, simulation_time: float, reference_signal: Callable,
                           disturbances: Optional[Dict] = None) -> Dict:
        """
        Simulate closed-loop system with reference tracking and disturbance rejection.
        
        Args:
            simulation_time: Total simulation time (s)
            reference_signal: Function that takes time and returns reference value
            disturbances: Optional dictionary of disturbance signals
            
        Returns:
            Simulation results dictionary
        """
        if self.controller_params is None:
            raise ValueError("Controller not tuned. Call tune_pid_* method first.")
        
        # Time vector
        time_vec = np.arange(0, simulation_time, self.sample_time)
        n_steps = len(time_vec)
        
        # Initialize arrays
        reference = np.zeros(n_steps)
        output = np.zeros(n_steps)
        control_signal = np.zeros(n_steps)
        error = np.zeros(n_steps)
        anomaly_measure = np.zeros(n_steps)
        
        # Generate reference signal
        for i, t in enumerate(time_vec):
            reference[i] = reference_signal(t)
        
        # PID controller state variables
        integral_error = 0
        previous_error = 0
        
        # Plant state (assuming second-order system)
        plant_state = np.zeros(2)  # [position, velocity] for second-order system
        
        for i in range(n_steps):
            # Current error
            error[i] = reference[i] - output[i]
            
            # PID control law
            integral_error += error[i] * self.sample_time
            derivative_error = (error[i] - previous_error) / self.sample_time
            
            # PID output with derivative filtering
            pid_output = (self.controller_params.kp * error[i] + 
                         self.controller_params.ki * integral_error + 
                         self.controller_params.kd * derivative_error)
            
            control_signal[i] = pid_output
            
            # Apply control limits (saturation)
            control_signal[i] = np.clip(control_signal[i], -100, 100)
            
            # Plant dynamics (simplified second-order system)
            # ẍ + 2ζωₙẋ + ωₙ²x = Ku
            if i < n_steps - 1:
                # State space representation: [x, ẋ]
                A = np.array([[0, 1], 
                             [-self.plant_params.omega_n**2, -2*self.plant_params.zeta*self.plant_params.omega_n]])
                B = np.array([0, self.plant_params.K])
                
                # Add disturbances if specified
                disturbance_input = 0
                if disturbances:
                    for dist_name, dist_func in disturbances.items():
                        disturbance_input += dist_func(time_vec[i])
                
                # Euler integration
                plant_derivative = A @ plant_state + B * (control_signal[i] + disturbance_input)
                plant_state += plant_derivative * self.sample_time
                
                output[i+1] = plant_state[0]
            
            # Compute anomaly measure (simplified)
            # In real system, this would involve actual G_tt calculation
            target_T00_val = reference[i]  # Simplified: reference as target
            current_T00_val = output[i]    # Simplified: output as measured T₀₀
            G_tt_val = target_T00_val + 0.1 * np.sin(2*np.pi*time_vec[i])  # Mock Einstein tensor
            
            anomaly_measure[i] = self.compute_anomaly_measure(
                np.array([current_T00_val]), np.array([target_T00_val]), np.array([G_tt_val])
            )
            
            # Store for next iteration
            previous_error = error[i]
        
        # Store history
        self.time_history = time_vec
        self.reference_history = reference
        self.output_history = output
        self.control_history = control_signal
        self.error_history = error
        self.anomaly_history_time = anomaly_measure
        
        return {
            'time': time_vec,
            'reference': reference,
            'output': output,
            'control_signal': control_signal,
            'error': error,
            'anomaly_measure': anomaly_measure,
            'performance_metrics': self._calculate_simulation_metrics(time_vec, reference, output, error)
        }
    
    def _calculate_simulation_metrics(self, time: np.ndarray, reference: np.ndarray,
                                    output: np.ndarray, error: np.ndarray) -> Dict:
        """Calculate performance metrics from simulation results."""
        rmse = np.sqrt(np.mean(error**2))
        mae = np.mean(np.abs(error))
        max_error = np.max(np.abs(error))
        settling_time_sim = self._calculate_settling_time(time, output)
        
        return {
            'rmse': rmse,
            'mae': mae,
            'max_error': max_error,
            'settling_time': settling_time_sim,
            'final_error': error[-1]
        }
    
    def plot_simulation_results(self, save_path: Optional[str] = None) -> plt.Figure:
        """Plot comprehensive simulation results."""
        if not hasattr(self, 'time_history') or len(self.time_history) == 0:
            raise ValueError("No simulation data available. Run simulate_closed_loop() first.")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        time = self.time_history
        
        # Reference tracking
        ax1.plot(time, self.reference_history, 'r--', linewidth=2, label='Reference')
        ax1.plot(time, self.output_history, 'b-', linewidth=1, label='Output')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude')
        ax1.set_title('Reference Tracking')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Control signal
        ax2.plot(time, self.control_history, 'g-', linewidth=1)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Control Signal')
        ax2.set_title('Control Effort')
        ax2.grid(True, alpha=0.3)
        
        # Tracking error
        ax3.plot(time, self.error_history, 'r-', linewidth=1)
        ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Error')
        ax3.set_title('Tracking Error')
        ax3.grid(True, alpha=0.3)
        
        # Anomaly measure
        ax4.plot(time, self.anomaly_history_time, 'm-', linewidth=1)
        ax4.axhline(y=self.target_anomaly_threshold, color='r', linestyle='--', 
                   label=f'Threshold: {self.target_anomaly_threshold:.2e}')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Anomaly Measure')
        ax4.set_title('Einstein Equation Violation')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_yscale('log')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig

if __name__ == "__main__":
    # Example usage
    
    # Define plant parameters (coil system)
    plant_params = PlantParams(
        K=1.0,          # DC gain
        omega_n=10.0,   # Natural frequency (rad/s)
        zeta=0.1,       # Damping ratio (underdamped)
        tau_delay=0.01  # 10ms delay
    )
    
    # Create controller
    controller = ClosedLoopFieldController(plant_params, sample_time=1e-3)
    
    # Tune PID controller
    print("Tuning PID controller...")
    pid_params = controller.tune_pid_optimization()
    print(f"PID parameters: kp={pid_params.kp:.3f}, ki={pid_params.ki:.3f}, kd={pid_params.kd:.6f}")
    
    # Analyze performance
    performance = controller.analyze_performance(pid_params)
    print(f"Performance metrics:")
    print(f"  Settling time: {performance.settling_time:.3f} s")
    print(f"  Overshoot: {performance.overshoot:.1f} %")
    print(f"  Steady-state error: {performance.steady_state_error:.6f}")
    print(f"  Gain margin: {performance.gain_margin:.1f} dB")
    print(f"  Phase margin: {performance.phase_margin:.1f} degrees")
    
    # Define reference signal (step + sinusoidal exotic matter target)
    def reference_signal(t):
        step = 1.0 if t > 0.5 else 0.0
        sinusoid = 0.2 * np.sin(2 * np.pi * t) if t > 2.0 else 0.0
        return step + sinusoid
    
    # Define disturbances
    def disturbance_func(t):
        return 0.1 * np.sin(20 * np.pi * t)  # High-frequency disturbance
    
    disturbances = {'electromagnetic_noise': disturbance_func}
    
    # Simulate closed-loop system
    print("Running closed-loop simulation...")
    sim_results = controller.simulate_closed_loop(
        simulation_time=5.0,
        reference_signal=reference_signal,
        disturbances=disturbances
    )
    
    print(f"Simulation metrics:")
    metrics = sim_results['performance_metrics']
    print(f"  RMSE: {metrics['rmse']:.6f}")
    print(f"  Max error: {metrics['max_error']:.6f}")
    print(f"  Final error: {metrics['final_error']:.6f}")
    
    # Plot results
    fig = controller.plot_simulation_results(save_path="closed_loop_control.png")
    plt.show()
