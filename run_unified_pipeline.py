#!/usr/bin/env python3
"""
Unified Warp Field Coil Pipeline
Integrates all components from the comprehensive roadmap
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional, List
import argparse
import json
import os
from pathlib import Path

# Import all our modules
from src.stress_energy.exotic_matter_profile import ExoticMatterProfiler, alcubierre_profile, gaussian_warp_profile
from src.coil_optimizer.advanced_coil_optimizer import AdvancedCoilOptimizer, CoilGeometryParams
from src.hardware.field_rig_design import ElectromagneticFieldSimulator, FieldRigResults
from src.diagnostics.superconducting_resonator import SuperconductingResonatorDiagnostics, ResonatorConfig
from src.control.closed_loop_controller import ClosedLoopFieldController, PlantParams, ControllerParams
from src.quantum_geometry.discrete_stress_energy import SU2GeneratingFunctionalCalculator, DiscreteWarpBubbleSolver

class UnifiedWarpFieldPipeline:
    """
    Unified pipeline implementing the complete roadmap for warp field coil development.
    Integrates exotic matter profiling, coil optimization, field simulation, diagnostics,
    control, and quantum geometry corrections.
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize the unified pipeline.
        
        Args:
            config_file: Optional configuration file path
        """
        # Load configuration
        self.config = self._load_config(config_file)
        
        # Initialize components
        self.exotic_profiler = ExoticMatterProfiler(
            r_min=self.config.get('r_min', 0.1),
            r_max=self.config.get('r_max', 10.0),
            n_points=self.config.get('n_points', 1000)
        )
        
        self.coil_optimizer = AdvancedCoilOptimizer(
            r_min=self.config.get('r_min', 0.1),
            r_max=self.config.get('r_max', 10.0),
            n_points=self.config.get('n_points_coil', 500)
        )
        
        self.field_simulator = ElectromagneticFieldSimulator()
        
        self.resonator_config = ResonatorConfig(
            base_frequency=self.config.get('resonator_frequency', 1e9),
            cavity_volume=self.config.get('cavity_volume', 1e-6),
            quality_factor=self.config.get('quality_factor', 1e6),
            coupling_strength=self.config.get('coupling_strength', 0.1),
            temperature=self.config.get('temperature', 0.01)
        )
        
        self.su2_calculator = SU2GeneratingFunctionalCalculator()
        self.discrete_solver = DiscreteWarpBubbleSolver(self.su2_calculator)
        
        # Results storage
        self.results = {}
        
    def _load_config(self, config_file: Optional[str]) -> Dict:
        """Load configuration from file or use defaults."""
        default_config = {
            'r_min': 0.1,
            'r_max': 10.0,
            'n_points': 1000,
            'n_points_coil': 500,
            'resonator_frequency': 1e9,
            'cavity_volume': 1e-6,
            'quality_factor': 1e6,
            'coupling_strength': 0.1,
            'temperature': 0.01,
            'warp_profile_type': 'alcubierre',
            'warp_radius': 2.0,
            'warp_width': 0.5,
            'optimization_method': 'hybrid',
            'control_sample_time': 1e-4,
            'discrete_mesh_nodes': 50
        }
        
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                file_config = json.load(f)
                default_config.update(file_config)
        
        return default_config
    
    def step_1_define_exotic_matter_profile(self) -> Dict:
        """
        Step 1: Define the required exotic-matter energy-density profile.
        Uses warp-bubble-einstein-equations to compute T^{00}(r) profile.
        """
        print("Step 1: Computing exotic matter energy density profile...")
        
        # Define warp profile function
        profile_type = self.config.get('warp_profile_type', 'alcubierre')
        
        if profile_type == 'alcubierre':
            profile_func = lambda r: alcubierre_profile(
                r, 
                R=self.config.get('warp_radius', 2.0),
                sigma=self.config.get('warp_width', 0.5)
            )
        elif profile_type == 'gaussian':
            profile_func = lambda r: gaussian_warp_profile(
                r,
                A=self.config.get('warp_amplitude', 1.0),
                sigma=self.config.get('warp_width', 1.0)
            )
        else:
            raise ValueError(f"Unknown profile type: {profile_type}")
        
        # Compute T^{00}(r) profile
        r_array, T00_profile = self.exotic_profiler.compute_T00_profile(profile_func)
        
        # Identify exotic regions
        exotic_info = self.exotic_profiler.identify_exotic_regions(T00_profile)
        
        # Plot results
        fig = self.exotic_profiler.plot_T00_profile(
            T00_profile, 
            title=f"Exotic Matter Profile ({profile_type.title()})",
            save_path="results/step1_exotic_matter_profile.png"
        )
        plt.close(fig)
        
        # Store results
        step1_results = {
            'r_array': r_array,
            'T00_profile': T00_profile,
            'exotic_info': exotic_info,
            'profile_type': profile_type
        }
        
        self.results['step1'] = step1_results
        
        print(f"✓ Exotic matter profile computed")
        print(f"  Has exotic regions: {exotic_info['has_exotic']}")
        if exotic_info['has_exotic']:
            print(f"  Total exotic energy: {exotic_info['total_exotic_energy']:.2e}")
        
        return step1_results
    
    def step_2_optimize_coil_geometry(self) -> Dict:
        """
        Step 2: Optimize coil geometry to match exotic matter profile.
        Uses JAX-accelerated optimization with stress-energy matching.
        """
        print("Step 2: Optimizing coil geometry...")
        
        if 'step1' not in self.results:
            raise ValueError("Step 1 must be completed first")
        
        step1_results = self.results['step1']
        
        # Set target profile in optimizer
        self.coil_optimizer.set_target_profile(
            step1_results['r_array'],
            step1_results['T00_profile']
        )
        
        # Initial parameter guess (multi-Gaussian)
        n_gaussians = self.config.get('n_gaussians', 3)
        initial_params = []
        for i in range(n_gaussians):
            A = (-0.1 if i == 1 else 0.05) * (1 + 0.1 * np.random.randn())  # Negative for main shell
            r_center = 0.5 + i * 2.0  # Distributed centers
            sigma = 0.3 + 0.1 * np.random.randn()
            initial_params.extend([A, r_center, sigma])
        
        initial_params = np.array(initial_params)
        
        # Run optimization
        method = self.config.get('optimization_method', 'hybrid')
        
        if method == 'hybrid':
            opt_result = self.coil_optimizer.optimize_hybrid(initial_params)
        elif method == 'lbfgs':
            opt_result = self.coil_optimizer.optimize_lbfgs(initial_params)
        elif method == 'cmaes':
            opt_result = self.coil_optimizer.optimize_cmaes(initial_params)
        else:
            raise ValueError(f"Unknown optimization method: {method}")
        
        # Extract coil geometry
        coil_geometry = self.coil_optimizer.extract_coil_geometry(opt_result['optimal_params'])
        
        # Plot optimization results
        fig = self.coil_optimizer.plot_optimization_result(
            opt_result['optimal_params'],
            save_path="results/step2_coil_optimization.png"
        )
        plt.close(fig)
        
        # Store results
        step2_results = {
            'optimization_result': opt_result,
            'optimal_params': opt_result['optimal_params'],
            'coil_geometry': coil_geometry,
            'optimization_method': method
        }
        
        self.results['step2'] = step2_results
        
        print(f"✓ Coil geometry optimized")
        print(f"  Success: {opt_result['success']}")
        print(f"  Final objective: {opt_result['optimal_objective']:.6e}")
        print(f"  Coil inner radius: {coil_geometry.inner_radius:.3f} m")
        print(f"  Coil outer radius: {coil_geometry.outer_radius:.3f} m")
        
        return step2_results
    
    def step_3_simulate_electromagnetic_performance(self) -> Dict:
        """
        Step 3: Simulate electromagnetic performance & safety margins.
        Uses field_rig_design.py to compute B_peak, E_peak, and safety factors.
        """
        print("Step 3: Simulating electromagnetic performance...")
        
        if 'step2' not in self.results:
            raise ValueError("Step 2 must be completed first")
        
        coil_geometry = self.results['step2']['coil_geometry']
        
        # Estimate inductance from geometry
        # L ≈ μ₀N²A/l for solenoid
        N_turns = coil_geometry.turn_density * coil_geometry.height
        A_coil = np.pi * (coil_geometry.outer_radius**2 - coil_geometry.inner_radius**2)
        L_estimated = 4e-7 * np.pi * N_turns**2 * A_coil / coil_geometry.height
        
        I_current = coil_geometry.current
        f_modulation = self.config.get('modulation_frequency', 1000.0)
        
        # Simulate field performance
        field_results = self.field_simulator.simulate_inductive_rig(
            L=L_estimated,
            I=I_current,
            f_mod=f_modulation,
            geometry='toroidal'
        )
        
        # Safety analysis
        safety_status = self.field_simulator.safety_analysis(field_results)
        
        # Parameter sweep for safe operating envelope
        L_range = (L_estimated * 0.1, L_estimated * 10)
        I_range = (I_current * 0.1, I_current * 10)
        
        sweep_results = self.field_simulator.sweep_current_geometry(
            L_range=L_range,
            I_range=I_range,
            n_points=30
        )
        
        # Plot safety envelope
        fig = self.field_simulator.plot_safety_envelope(
            sweep_results,
            save_path="results/step3_safety_envelope.png"
        )
        plt.close(fig)
        
        # Store results
        step3_results = {
            'field_results': field_results,
            'safety_status': safety_status,
            'sweep_results': sweep_results,
            'estimated_inductance': L_estimated
        }
        
        self.results['step3'] = step3_results
        
        print(f"✓ Electromagnetic performance simulated")
        print(f"  Peak B-field: {field_results.B_peak:.2f} T")
        print(f"  Peak E-field: {field_results.E_peak:.2e} V/m")
        print(f"  Stored energy: {field_results.stored_energy:.2e} J")
        print(f"  Safety status: {min(safety_status.values())}")
        
        return step3_results
    
    def step_4_integrate_resonator_diagnostics(self) -> Dict:
        """
        Step 4: Integrate superconducting resonator diagnostics.
        Measures T₀₀ in situ from field quadratures.
        """
        print("Step 4: Integrating resonator diagnostics...")
        
        # Initialize resonator diagnostics
        resonator = SuperconductingResonatorDiagnostics(self.resonator_config)
        
        # Perform stress-energy measurement
        measurement = resonator.measure_stress_energy_real_time(
            measurement_duration=self.config.get('measurement_duration', 1e-3),
            sampling_rate=self.config.get('sampling_rate', 10e9),
            apply_filtering=True
        )
        
        # Compare with target if available
        comparison_results = None
        if 'step1' in self.results:
            target_T00 = self.results['step1']['T00_profile']
            target_r = self.results['step1']['r_array']
            
            comparison_results = resonator.compare_with_target(
                measurement, target_T00, target_r
            )
        
        # Plot measurement results
        fig = resonator.plot_measurement_results(
            measurement,
            target_T00=target_T00 if 'step1' in self.results else None,
            save_path="results/step4_resonator_diagnostics.png"
        )
        plt.close(fig)
        
        # Store results
        step4_results = {
            'measurement': measurement,
            'comparison': comparison_results,
            'resonator_config': self.resonator_config
        }
        
        self.results['step4'] = step4_results
        
        print(f"✓ Resonator diagnostics integrated")
        print(f"  SNR: {measurement.signal_to_noise:.1f} dB")
        print(f"  Mean T₀₀: {np.mean(measurement.T00_measured):.2e} J/m³")
        if comparison_results:
            print(f"  Agreement quality: {comparison_results['agreement_quality']}")
        
        return step4_results
    
    def step_5_implement_closed_loop_control(self) -> Dict:
        """
        Step 5: Implement closed-loop field control.
        Uses PID control with anomaly tracking.
        """
        print("Step 5: Implementing closed-loop control...")
        
        if 'step3' not in self.results:
            raise ValueError("Step 3 must be completed first")
        
        field_results = self.results['step3']['field_results']
        
        # Define plant parameters from electromagnetic simulation
        plant_params = PlantParams(
            K=1.0,  # Normalized DC gain
            omega_n=2 * np.pi * self.config.get('control_bandwidth', 100.0),  # 100 Hz bandwidth
            zeta=0.7,  # Critically damped
            tau_delay=self.config.get('control_delay', 0.001)  # 1ms delay
        )
        
        # Initialize controller
        controller = ClosedLoopFieldController(
            plant_params,
            sample_time=self.config.get('control_sample_time', 1e-4)
        )
        
        # Tune PID controller
        pid_params = controller.tune_pid_optimization()
        
        # Analyze performance
        performance = controller.analyze_performance(pid_params)
        
        # Define reference signal (exotic matter target tracking)
        def reference_signal(t):
            # Step to exotic matter shell + sinusoidal modulation
            step = -0.1 if t > 0.5 else 0.0  # Negative energy target
            modulation = 0.02 * np.sin(2 * np.pi * 10 * t) if t > 2.0 else 0.0
            return step + modulation
        
        # Define disturbances
        def thermal_disturbance(t):
            return 0.01 * np.sin(2 * np.pi * 50 * t)  # 50 Hz thermal fluctuation
        
        disturbances = {'thermal': thermal_disturbance}
        
        # Run closed-loop simulation
        sim_results = controller.simulate_closed_loop(
            simulation_time=self.config.get('simulation_time', 10.0),
            reference_signal=reference_signal,
            disturbances=disturbances
        )
        
        # Plot control results
        fig = controller.plot_simulation_results(
            save_path="results/step5_closed_loop_control.png"
        )
        plt.close(fig)
        
        # Store results
        step5_results = {
            'pid_params': pid_params,
            'performance': performance,
            'simulation_results': sim_results,
            'plant_params': plant_params
        }
        
        self.results['step5'] = step5_results
        
        print(f"✓ Closed-loop control implemented")
        print(f"  PID gains: kp={pid_params.kp:.3f}, ki={pid_params.ki:.3f}, kd={pid_params.kd:.6f}")
        print(f"  Settling time: {performance.settling_time:.3f} s")
        print(f"  Phase margin: {performance.phase_margin:.1f}°")
        
        return step5_results
    
    def step_6_discrete_quantum_geometry(self) -> Dict:
        """
        Step 6: Implement discrete quantum geometry corrections.
        Uses SU(2) generating functionals and 3nj symbols.
        """
        print("Step 6: Computing discrete quantum geometry corrections...")
        
        # Build discrete mesh
        n_nodes = self.config.get('discrete_mesh_nodes', 50)
        nodes, edges = self.discrete_solver.build_discrete_mesh(
            r_min=self.config.get('r_min', 0.1),
            r_max=self.config.get('r_max', 10.0),
            n_nodes=n_nodes,
            mesh_type="radial"
        )
        
        # Create target profile for discrete optimization
        if 'step1' in self.results:
            # Use continuum profile as target
            continuum_r = self.results['step1']['r_array']
            continuum_T00 = self.results['step1']['T00_profile']
            
            # Interpolate to discrete mesh
            from scipy.interpolate import interp1d
            positions = np.array([node.position for node in nodes])
            r_discrete = np.linalg.norm(positions, axis=1)
            
            interp_func = interp1d(continuum_r, continuum_T00, 
                                 kind='cubic', bounds_error=False, fill_value=0.0)
            target_T00_discrete = interp_func(r_discrete)
        else:
            # Create mock target
            positions = np.array([node.position for node in nodes])
            r_discrete = np.linalg.norm(positions, axis=1)
            target_T00_discrete = -0.1 * np.exp(-((r_discrete - 2.0)/0.5)**2)
        
        # Mock Einstein tensor and matter values
        G_tt_vals = 0.1 * np.sin(np.pi * r_discrete / 3.0)
        Tm_vals = 0.05 * np.ones_like(r_discrete)
        
        # Optimize discrete currents
        opt_result = self.discrete_solver.optimize_discrete_currents(
            target_T00_discrete, G_tt_vals, Tm_vals
        )
        
        # Plot discrete solution
        if opt_result['success']:
            fig = self.discrete_solver.plot_discrete_solution(
                opt_result['optimal_currents'],
                target_T00_discrete,
                save_path="results/step6_discrete_solution.png"
            )
            plt.close(fig)
        
        # Test generating functional computation
        test_currents = np.random.normal(0, 0.1, len(edges))
        K_test = self.su2_calculator.build_K_from_currents(
            self.discrete_solver.adjacency_matrix, test_currents
        )
        G_test = self.su2_calculator.compute_generating_functional(K_test)
        
        # Store results
        step6_results = {
            'discrete_optimization': opt_result,
            'n_nodes': len(nodes),
            'n_edges': len(edges),
            'generating_functional_test': G_test,
            'target_T00_discrete': target_T00_discrete
        }
        
        self.results['step6'] = step6_results
        
        print(f"✓ Discrete quantum geometry computed")
        print(f"  Mesh: {len(nodes)} nodes, {len(edges)} edges")
        if opt_result['success']:
            print(f"  Target error: {opt_result['target_error']:.6f}")
            print(f"  Anomaly: {opt_result['anomaly']:.6e}")
        print(f"  Generating functional: {G_test}")
        
        return step6_results
    
    def run_complete_pipeline(self) -> Dict:
        """Run the complete integrated pipeline."""
        print("=" * 60)
        print("UNIFIED WARP FIELD COIL DEVELOPMENT PIPELINE")
        print("=" * 60)
        
        # Create results directory
        os.makedirs("results", exist_ok=True)
        
        try:
            # Execute all steps
            step1_results = self.step_1_define_exotic_matter_profile()
            step2_results = self.step_2_optimize_coil_geometry()
            step3_results = self.step_3_simulate_electromagnetic_performance()
            step4_results = self.step_4_integrate_resonator_diagnostics()
            step5_results = self.step_5_implement_closed_loop_control()
            step6_results = self.step_6_discrete_quantum_geometry()
            
            # Generate comprehensive summary
            summary = self._generate_pipeline_summary()
            
            # Save results
            self._save_results()
            
            print("=" * 60)
            print("PIPELINE COMPLETED SUCCESSFULLY")
            print("=" * 60)
            print(summary)
            
            return self.results
            
        except Exception as e:
            print(f"Pipeline failed at step: {e}")
            raise
    
    def _generate_pipeline_summary(self) -> str:
        """Generate comprehensive summary of pipeline results."""
        summary = []
        summary.append("PIPELINE SUMMARY")
        summary.append("-" * 40)
        
        if 'step1' in self.results:
            exotic_info = self.results['step1']['exotic_info']
            summary.append(f"Step 1 - Exotic Matter Profile:")
            summary.append(f"  Profile type: {self.results['step1']['profile_type']}")
            summary.append(f"  Has exotic regions: {exotic_info['has_exotic']}")
            if exotic_info['has_exotic']:
                summary.append(f"  Total exotic energy: {exotic_info['total_exotic_energy']:.2e}")
        
        if 'step2' in self.results:
            coil_geom = self.results['step2']['coil_geometry']
            opt_result = self.results['step2']['optimization_result']
            summary.append(f"Step 2 - Coil Optimization:")
            summary.append(f"  Success: {opt_result['success']}")
            summary.append(f"  Inner radius: {coil_geom.inner_radius:.3f} m")
            summary.append(f"  Outer radius: {coil_geom.outer_radius:.3f} m")
            summary.append(f"  Turn density: {coil_geom.turn_density:.1f} turns/m")
        
        if 'step3' in self.results:
            field_results = self.results['step3']['field_results']
            summary.append(f"Step 3 - Electromagnetic Performance:")
            summary.append(f"  Peak B-field: {field_results.B_peak:.2f} T")
            summary.append(f"  Peak E-field: {field_results.E_peak:.2e} V/m")
            summary.append(f"  Min safety margin: {min(field_results.safety_margins.values()):.1f}")
        
        if 'step4' in self.results:
            measurement = self.results['step4']['measurement']
            summary.append(f"Step 4 - Resonator Diagnostics:")
            summary.append(f"  SNR: {measurement.signal_to_noise:.1f} dB")
            if self.results['step4']['comparison']:
                agreement = self.results['step4']['comparison']['agreement_quality']
                summary.append(f"  Agreement: {agreement}")
        
        if 'step5' in self.results:
            performance = self.results['step5']['performance']
            summary.append(f"Step 5 - Closed-Loop Control:")
            summary.append(f"  Settling time: {performance.settling_time:.3f} s")
            summary.append(f"  Phase margin: {performance.phase_margin:.1f}°")
        
        if 'step6' in self.results:
            discrete_opt = self.results['step6']['discrete_optimization']
            summary.append(f"Step 6 - Quantum Geometry:")
            summary.append(f"  Nodes: {self.results['step6']['n_nodes']}")
            if discrete_opt['success']:
                summary.append(f"  Target error: {discrete_opt['target_error']:.6f}")
        
        return "\n".join(summary)
    
    def _save_results(self):
        """Save results to files."""
        # Save configuration
        with open("results/config.json", "w") as f:
            json.dump(self.config, f, indent=2)
        
        # Save summary
        summary = self._generate_pipeline_summary()
        with open("results/summary.txt", "w") as f:
            f.write(summary)
        
        # Save numerical results (simplified)
        results_to_save = {}
        for step, data in self.results.items():
            results_to_save[step] = {}
            for key, value in data.items():
                if isinstance(value, (int, float, str, bool, list)):
                    results_to_save[step][key] = value
                elif isinstance(value, np.ndarray):
                    results_to_save[step][key] = value.tolist()
        
        with open("results/numerical_results.json", "w") as f:
            json.dump(results_to_save, f, indent=2)

def main():
    """Main entry point for the unified pipeline."""
    parser = argparse.ArgumentParser(description="Unified Warp Field Coil Development Pipeline")
    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument("--step", type=int, help="Run specific step only (1-6)")
    parser.add_argument("--profile-type", type=str, choices=['alcubierre', 'gaussian'], 
                       default='alcubierre', help="Warp profile type")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = UnifiedWarpFieldPipeline(args.config)
    
    # Override profile type if specified
    if args.profile_type:
        pipeline.config['warp_profile_type'] = args.profile_type
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.chdir(args.output_dir)
    
    try:
        if args.step:
            # Run specific step
            step_methods = {
                1: pipeline.step_1_define_exotic_matter_profile,
                2: pipeline.step_2_optimize_coil_geometry,
                3: pipeline.step_3_simulate_electromagnetic_performance,
                4: pipeline.step_4_integrate_resonator_diagnostics,
                5: pipeline.step_5_implement_closed_loop_control,
                6: pipeline.step_6_discrete_quantum_geometry
            }
            
            if args.step in step_methods:
                print(f"Running Step {args.step} only...")
                result = step_methods[args.step]()
                print(f"Step {args.step} completed successfully")
            else:
                print(f"Invalid step number: {args.step}")
                return 1
        else:
            # Run complete pipeline
            results = pipeline.run_complete_pipeline()
            
        return 0
        
    except Exception as e:
        print(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
