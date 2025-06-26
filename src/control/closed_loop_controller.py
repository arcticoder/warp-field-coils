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
        Initialize the closed-loop controller with quantum anomaly tracking.
        
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
        
        # Quantum geometry integration
        from quantum_geometry.discrete_stress_energy import DiscreteQuantumGeometry
        self.quantum_solver = DiscreteQuantumGeometry(n_nodes=20)
        self.quantum_anomaly_history = []
        
        # Quantum-aware control parameters
        self.quantum_feedback_gain = 0.1  # β parameter for quantum reference adjustment
        
        # Simulation state
        self.time_history = []
        self.reference_history = []
        self.output_history = []
        self.control_history = []
        self.error_history = []
        self.anomaly_history_time = []
        self.quantum_anomaly_history_time = []
    
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
    
    def compute_quantum_anomaly(self, current_state: Dict) -> float:
        """
        Compute quantum geometry anomaly (1/G - 1).
        
        Args:
            current_state: Dictionary with current system state including field values
            
        Returns:
            Quantum anomaly measure
        """
        try:
            # Extract current distribution from system state
            if 'currents' in current_state:
                currents = current_state['currents']
            else:
                # Default fallback
                currents = np.ones(self.quantum_solver.n_nodes) * 0.1
            
            # Build K-matrix from current distribution
            K_matrix = self._build_K_from_currents(currents)
            
            # Compute generating functional
            G = self.quantum_solver.su2_calculator.compute_generating_functional(K_matrix)
            
            # Quantum anomaly
            anomaly = abs(1.0/G - 1.0)
            
            return anomaly
            
        except Exception as e:
            print(f"Warning: Quantum anomaly computation failed: {e}")
            return 0.0
    
    def _build_K_from_currents(self, currents: np.ndarray) -> np.ndarray:
        """Build K-matrix from current distribution for quantum calculations."""
        n_nodes = self.quantum_solver.n_nodes
        
        # Ensure currents array has correct size
        if len(currents) != n_nodes:
            # Interpolate or pad to match node count
            currents_interp = np.interp(
                np.linspace(0, 1, n_nodes),
                np.linspace(0, 1, len(currents)),
                currents
            )
        else:
            currents_interp = currents
        
        # Build K-matrix
        K = np.zeros((n_nodes, n_nodes))
        adjacency = self.quantum_solver.adjacency_matrix
        
        # Scale currents appropriately
        current_scale = np.max(np.abs(currents_interp))
        if current_scale > 1e-12:
            normalized_currents = currents_interp / current_scale
        else:
            normalized_currents = currents_interp
        
        for i in range(n_nodes):
            for j in range(n_nodes):
                if adjacency[i, j] > 0:
                    # Weight interaction by current strength
                    current_weight = 0.5 * (normalized_currents[i] + normalized_currents[j])
                    K[i, j] = 0.1 * current_weight * adjacency[i, j]
        
        return K
    
    def quantum_aware_reference(self, reference: float, current_state: Dict) -> float:
        """
        Compute quantum-aware reference signal.
        
        r_quantum(t) = r_0(t) - β * (1/G - 1)
        
        Args:
            reference: Base reference signal r_0(t)
            current_state: Current system state
            
        Returns:
            Quantum-corrected reference signal
        """
        # Compute quantum anomaly
        quantum_anomaly = self.compute_quantum_anomaly(current_state)
        
        # Apply quantum feedback correction
        quantum_correction = self.quantum_feedback_gain * quantum_anomaly
        
        # Adjust reference to compensate for quantum effects
        r_quantum = reference - quantum_correction
        
        # Store for history tracking
        self.quantum_anomaly_history_time.append(quantum_anomaly)
        
        return r_quantum
    
    def simulate_quantum_aware_control(self, time_span: Tuple[float, float], 
                                     reference_func: Callable[[float], float],
                                     disturbance_func: Optional[Callable[[float], float]] = None,
                                     n_points: int = 1000) -> Dict:
        """
        Simulate quantum-aware closed-loop control system.
        
        Includes both Einstein equation anomaly tracking and quantum geometry corrections.
        
        Args:
            time_span: (t_start, t_end) simulation time span
            reference_func: Reference signal r(t)
            disturbance_func: Optional disturbance input d(t)
            n_points: Number of simulation points
            
        Returns:
            Simulation results with quantum corrections
        """
        if self.controller_params is None:
            raise ValueError("Controller not tuned. Call tune_pid_* method first.")
        
        # Time vector
        times = np.linspace(time_span[0], time_span[1], n_points)
        dt = times[1] - times[0]
        
        # Initialize state variables
        output = np.zeros(n_points)
        control = np.zeros(n_points)
        error = np.zeros(n_points)
        reference = np.zeros(n_points)
        quantum_reference = np.zeros(n_points)
        einstein_anomaly = np.zeros(n_points)
        quantum_anomaly = np.zeros(n_points)
        
        # PID state variables
        integral_error = 0.0
        previous_error = 0.0
        
        print("Running quantum-aware control simulation...")
        
        for i, t in enumerate(times):
            # Base reference signal
            ref = reference_func(t)
            reference[i] = ref
            
            # Current system state (simplified model)
            current_state = {
                'time': t,
                'output': output[i-1] if i > 0 else 0.0,
                'control': control[i-1] if i > 0 else 0.0,
                'currents': np.ones(10) * (control[i-1] if i > 0 else 0.1)  # Mock current distribution
            }
            
            # Quantum-aware reference
            ref_quantum = self.quantum_aware_reference(ref, current_state)
            quantum_reference[i] = ref_quantum
            
            # Control error with quantum correction
            error[i] = ref_quantum - output[i-1] if i > 0 else ref_quantum
            
            # PID control computation
            integral_error += error[i] * dt
            derivative_error = (error[i] - previous_error) / dt if i > 0 else 0.0
            
            # PID output
            control[i] = (self.controller_params.kp * error[i] + 
                         self.controller_params.ki * integral_error +
                         self.controller_params.kd * derivative_error)
            
            # Apply control limits
            control[i] = np.clip(control[i], -10.0, 10.0)
            
            # Plant response (simplified second-order model)
            if i > 1:
                # Second-order difference equation approximation
                plant_response = (self.plant_params.K * control[i-1] + 
                                2*output[i-1] - output[i-2])
                output[i] = np.clip(plant_response, -5.0, 5.0)
            elif i == 1:
                output[i] = 0.1 * self.plant_params.K * control[i-1]
            
            # Add disturbance if provided
            if disturbance_func is not None:
                output[i] += disturbance_func(t)
            
            # Compute anomalies
            einstein_anomaly[i] = self._compute_einstein_anomaly(current_state)
            quantum_anomaly[i] = self.compute_quantum_anomaly(current_state)
            
            previous_error = error[i]
        
        # Package results
        results = {
            'time': times,
            'reference': reference,
            'quantum_reference': quantum_reference,
            'output': output,
            'control': control,
            'error': error,
            'einstein_anomaly': einstein_anomaly,
            'quantum_anomaly': quantum_anomaly,
            'controller_params': self.controller_params
        }
        
        # Store in history
        self.time_history = times
        self.reference_history = reference
        self.output_history = output
        self.control_history = control
        self.error_history = error
        self.anomaly_history_time = einstein_anomaly
        self.quantum_anomaly_history_time = quantum_anomaly
        
        print(f"✓ Quantum-aware simulation complete")
        print(f"  Final tracking error: {abs(error[-1]):.6f}")
        print(f"  Final Einstein anomaly: {einstein_anomaly[-1]:.6e}")
        print(f"  Final quantum anomaly: {quantum_anomaly[-1]:.6e}")
        
        return results
    
    def _compute_einstein_anomaly(self, current_state: Dict) -> float:
        """Compute Einstein equation anomaly |G_μν - 8π T_μν|."""
        # Simplified Einstein anomaly computation
        # In practice, this would compute actual curvature vs stress-energy
        
        # Mock calculation based on field strength
        field_strength = abs(current_state.get('output', 0.0))
        target_strength = 1.0  # Target field value
        
        # Einstein anomaly proportional to field deviation
        anomaly = abs(field_strength - target_strength)
        
        return anomaly
