"""
Step 21: Unified System Calibration
==================================

Multi-objective optimization to find optimal parameters that simultaneously maximize:
- FTL data rate (SubspaceTransceiver)
- Force-field uniformity (HolodeckGrid) 
- Tractor precision & safety margin (MedicalArray)
- Tomographic reconstruction fidelity

Mathematical Foundation:
J(p) = w₁(1 - RateNorm(p)) + w₂(1 - Uniformity(p)) + w₃(1 - Precision(p)) + w₄(1 - Fidelity(p))

Where p = [α, β, γ, δ] are the coupling parameters to optimize.
"""

import numpy as np
import logging
from scipy.optimize import minimize, differential_evolution
from typing import Dict, Tuple, List, Optional
import json
import time
from dataclasses import dataclass

# Import system components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'medical_tractor_array'))

try:
    from medical_tractor_array.array import MedicalArrayParams, MedicalTractorArray
except ImportError:
    # Use mock classes for testing
    from mock_medical_array import MedicalArrayParams, MedicalTractorArray

# Mock implementations for missing components  
from dataclasses import dataclass

@dataclass
class SubspaceParams:
    subspace_coupling: float = 1e-15
    frequency_range: tuple = (1e9, 1e12) 
    antenna_gain: float = 40.0

class SubspaceTransceiver:
    def __init__(self, params: SubspaceParams):
        self.params = params
    
    def run_diagnostics(self) -> dict:
        bandwidth = self.params.subspace_coupling * 1e27
        return {
            'usable_bandwidth_hz': min(bandwidth, 1e12),
            'coupling_efficiency': self.params.subspace_coupling * 1e15
        }

@dataclass
class GridParams:
    bounds: tuple = ((-1.0, 1.0), (-1.0, 1.0), (0.0, 1.0))
    base_spacing: float = 0.2
    max_grid_points: int = 10000

class ForceFieldGrid:
    def __init__(self, params: GridParams):
        self.params = params
    
    def run_diagnostics(self) -> dict:
        n_points = min(int(8 / self.params.base_spacing**3), self.params.max_grid_points)
        mean_force = 1.0 / self.params.base_spacing
        std_force = mean_force * 0.1
        
        return {
            'grid_points': n_points,
            'force_statistics': {
                'mean_force': mean_force,
                'std_force': std_force
            }
        }

@dataclass
class TomographyParams:
    scan_resolution: tuple = (128, 128)
    scan_volume: tuple = (2.0, 2.0, 1.0)
    n_projections: int = 120
    detector_elements: int = 256

class WarpTomographicScanner:
    def __init__(self, params: TomographyParams):
        self.params = params
    
    def compute_radon_transform(self, image):
        # Mock Radon transform
        return np.random.random((self.params.n_projections, self.params.detector_elements))
    
    def filtered_backprojection(self, projections):
        # Mock reconstruction
        return np.random.random(self.params.scan_resolution)

@dataclass
class CalibrationParams:
    """Parameters for unified system calibration"""
    # Parameter bounds for optimization
    subspace_coupling_range: Tuple[float, float] = (1e-17, 1e-12)
    grid_spacing_range: Tuple[float, float] = (0.05, 0.5)
    medical_field_strength_range: Tuple[float, float] = (0.5, 3.0)
    tomography_projections_range: Tuple[int, int] = (60, 240)
    
    # Objective function weights
    ftl_rate_weight: float = 0.25
    grid_uniformity_weight: float = 0.25
    medical_precision_weight: float = 0.25
    tomography_fidelity_weight: float = 0.25
    
    # Performance targets for normalization
    target_ftl_rate: float = 1e12  # 1 THz
    target_force_uniformity: float = 0.95  # 95% uniformity
    target_medical_precision: float = 1e-9  # 1 nN force precision
    target_tomography_correlation: float = 0.99  # 99% correlation
    
    # Optimization settings
    max_iterations: int = 100
    tolerance: float = 1e-6
    use_genetic_algorithm: bool = True
    population_size: int = 50

class UnifiedSystemCalibrator:
    """
    Unified calibration system for all warp field components
    
    Performs multi-objective optimization to find parameters that
    maximize overall system performance across all subsystems.
    """
    
    def __init__(self, params: CalibrationParams):
        """
        Initialize unified calibrator
        
        Args:
            params: Calibration configuration parameters
        """
        self.params = params
        self.optimization_history = []
        self.best_parameters = None
        self.best_performance = None
        
        # Initialize subsystem components with default params
        self._initialize_subsystems()
        
        logging.info("UnifiedSystemCalibrator initialized")

    def _initialize_subsystems(self):
        """Initialize all subsystem components with baseline parameters"""
        # Subspace transceiver
        self.subspace_params = SubspaceParams(
            subspace_coupling=1e-15,
            frequency_range=(1e9, 1e12),
            antenna_gain=40.0
        )
        
        # Holodeck force field grid
        self.grid_params = GridParams(
            bounds=((-1.0, 1.0), (-1.0, 1.0), (0.0, 1.0)),
            base_spacing=0.2,
            max_grid_points=10000
        )
        
        # Medical tractor array  
        self.medical_params = MedicalArrayParams(
            array_bounds=((-0.5, 0.5), (-0.5, 0.5), (0.0, 1.0)),
            beam_spacing=0.02,
            global_power_limit=500.0
        )
        
        # Tomographic scanner
        self.tomography_params = TomographyParams(
            scan_resolution=(128, 128),
            scan_volume=(2.0, 2.0, 1.0),
            n_projections=120,
            detector_elements=256
        )

    def objective_function(self, params: np.ndarray) -> float:
        """
        Multi-objective cost function to minimize
        
        Args:
            params: [subspace_coupling, grid_spacing, medical_field_strength, n_projections]
            
        Returns:
            Combined cost value (lower is better)
        """
        try:
            α, β, γ, δ = params
            δ = int(δ)  # n_projections must be integer
            
            # 1) FTL Communication Rate Assessment
            ftl_score = self._evaluate_ftl_performance(α)
            
            # 2) Holodeck Grid Uniformity Assessment  
            grid_score = self._evaluate_grid_performance(β)
            
            # 3) Medical Tractor Precision Assessment
            medical_score = self._evaluate_medical_performance(γ)
            
            # 4) Tomographic Fidelity Assessment
            tomography_score = self._evaluate_tomography_performance(δ)
            
            # Weighted combination
            total_cost = (
                self.params.ftl_rate_weight * (1 - ftl_score) +
                self.params.grid_uniformity_weight * (1 - grid_score) +
                self.params.medical_precision_weight * (1 - medical_score) +
                self.params.tomography_fidelity_weight * (1 - tomography_score)
            )
            
            # Record optimization step
            self.optimization_history.append({
                'params': params.copy(),
                'scores': {
                    'ftl': ftl_score,
                    'grid': grid_score, 
                    'medical': medical_score,
                    'tomography': tomography_score
                },
                'total_cost': total_cost,
                'timestamp': time.time()
            })
            
            logging.debug(f"Optimization step: params={params}, cost={total_cost:.6f}")
            
            return total_cost
            
        except Exception as e:
            logging.warning(f"Objective function evaluation failed: {e}")
            return 1e6  # High penalty for failed evaluations

    def _evaluate_ftl_performance(self, subspace_coupling: float) -> float:
        """Evaluate FTL communication performance"""
        try:
            # Update subspace coupling parameter
            params = SubspaceParams(
                subspace_coupling=subspace_coupling,
                frequency_range=self.subspace_params.frequency_range,
                antenna_gain=self.subspace_params.antenna_gain
            )
            
            # Create transceiver and run diagnostics
            transceiver = SubspaceTransceiver(params)
            diagnostics = transceiver.run_diagnostics()
            
            # Extract usable bandwidth as performance metric
            rate = diagnostics.get('usable_bandwidth_hz', 0)
            normalized_rate = min(rate / self.params.target_ftl_rate, 1.0)
            
            return normalized_rate
            
        except Exception as e:
            logging.warning(f"FTL performance evaluation failed: {e}")
            return 0.0

    def _evaluate_grid_performance(self, grid_spacing: float) -> float:
        """Evaluate holodeck grid uniformity performance"""
        try:
            # Update grid spacing parameter
            params = GridParams(
                bounds=self.grid_params.bounds,
                base_spacing=grid_spacing,
                max_grid_points=self.grid_params.max_grid_points
            )
            
            # Create grid and run diagnostics
            grid = ForceFieldGrid(params)
            diagnostics = grid.run_diagnostics()
            
            # Extract force uniformity metric
            force_stats = diagnostics.get('force_statistics', {})
            mean_force = force_stats.get('mean_force', 1e-6)
            std_force = force_stats.get('std_force', 1e-6)
            
            # Uniformity = 1 - coefficient_of_variation
            uniformity = max(1 - std_force / mean_force, 0.0)
            normalized_uniformity = min(uniformity / self.params.target_force_uniformity, 1.0)
            
            return normalized_uniformity
            
        except Exception as e:
            logging.warning(f"Grid performance evaluation failed: {e}")
            return 0.0

    def _evaluate_medical_performance(self, field_strength: float) -> float:
        """Evaluate medical tractor array precision"""
        try:
            # Update medical array parameters
            params = MedicalArrayParams(
                array_bounds=self.medical_params.array_bounds,
                beam_spacing=self.medical_params.beam_spacing,
                global_power_limit=field_strength * 100  # Scale power limit
            )
            
            # Create medical array
            array = MedicalTractorArray(params)
            
            # Test positioning precision with sample target
            target_pos = np.array([0.05, 0.02, 0.1])
            desired_pos = np.array([0.03, 0.02, 0.1])
            
            result = array.position_target(target_pos, desired_pos, tissue_type="organ")
            
            # Extract force precision metric
            if 'force' in result:
                force_magnitude = np.linalg.norm(result['force'])
                # Higher force indicates better precision capability
                precision_score = min(force_magnitude / self.params.target_medical_precision, 1.0)
            else:
                precision_score = 0.0
            
            return precision_score
            
        except Exception as e:
            logging.warning(f"Medical performance evaluation failed: {e}")
            return 0.0

    def _evaluate_tomography_performance(self, n_projections: int) -> float:
        """Evaluate tomographic reconstruction fidelity"""
        try:
            # Update tomography parameters
            params = TomographyParams(
                scan_resolution=self.tomography_params.scan_resolution,
                scan_volume=self.tomography_params.scan_volume,
                n_projections=n_projections,
                detector_elements=self.tomography_params.detector_elements
            )
            
            # Create scanner
            scanner = WarpTomographicScanner(params)
            
            # Create test phantom for reconstruction quality assessment
            test_field = np.zeros(self.tomography_params.scan_resolution)
            h, w = test_field.shape
            test_field[h//4:3*h//4, w//4:3*w//4] = -0.5  # Central square region
            
            # Compute forward projection and reconstruction
            projections = scanner.compute_radon_transform(test_field)
            reconstruction = scanner.filtered_backprojection(projections)
            
            # Compute correlation coefficient as fidelity metric
            correlation = np.corrcoef(test_field.flatten(), reconstruction.flatten())[0, 1]
            correlation = max(correlation, 0.0)  # Ensure non-negative
            
            normalized_correlation = min(correlation / self.params.target_tomography_correlation, 1.0)
            
            return normalized_correlation
            
        except Exception as e:
            logging.warning(f"Tomography performance evaluation failed: {e}")
            return 0.0

    def run_calibration(self) -> Dict:
        """
        Run unified system calibration optimization
        
        Returns:
            Optimization results and best parameters
        """
        logging.info("Starting unified system calibration...")
        
        # Define parameter bounds
        bounds = [
            self.params.subspace_coupling_range,
            self.params.grid_spacing_range, 
            self.params.medical_field_strength_range,
            self.params.tomography_projections_range
        ]
        
        # Initial parameter guess
        x0 = np.array([
            1e-15,  # subspace_coupling
            0.2,    # grid_spacing  
            1.0,    # medical_field_strength
            120     # n_projections
        ])
        
        start_time = time.time()
        
        if self.params.use_genetic_algorithm:
            # Use differential evolution for global optimization
            result = differential_evolution(
                self.objective_function,
                bounds,
                maxiter=self.params.max_iterations,
                popsize=self.params.population_size,
                tol=self.params.tolerance,
                seed=42,
                disp=True
            )
        else:
            # Use gradient-based optimization
            result = minimize(
                self.objective_function,
                x0,
                bounds=bounds,
                method='L-BFGS-B',
                options={
                    'maxiter': self.params.max_iterations,
                    'ftol': self.params.tolerance
                }
            )
        
        optimization_time = time.time() - start_time
        
        # Store best results
        self.best_parameters = result.x
        self.best_performance = result.fun
        
        # Final evaluation with best parameters
        final_scores = self._evaluate_final_performance(result.x)
        
        calibration_results = {
            'success': result.success,
            'optimal_parameters': {
                'subspace_coupling': result.x[0],
                'grid_spacing': result.x[1], 
                'medical_field_strength': result.x[2],
                'n_projections': int(result.x[3])
            },
            'performance_scores': final_scores,
            'total_cost': result.fun,
            'optimization_time': optimization_time,
            'iterations': len(self.optimization_history),
            'convergence_message': result.message if hasattr(result, 'message') else 'Success'
        }
        
        logging.info(f"Calibration completed in {optimization_time:.2f}s")
        logging.info(f"Optimal parameters: {calibration_results['optimal_parameters']}")
        logging.info(f"Performance scores: {final_scores}")
        
        return calibration_results

    def _evaluate_final_performance(self, params: np.ndarray) -> Dict[str, float]:
        """Evaluate final performance with optimal parameters"""
        α, β, γ, δ = params
        δ = int(δ)
        
        scores = {
            'ftl_rate': self._evaluate_ftl_performance(α),
            'grid_uniformity': self._evaluate_grid_performance(β), 
            'medical_precision': self._evaluate_medical_performance(γ),
            'tomography_fidelity': self._evaluate_tomography_performance(δ)
        }
        
        # Compute overall performance score
        scores['overall'] = np.mean(list(scores.values()))
        
        return scores

    def save_calibration_results(self, filepath: str):
        """Save calibration results to JSON file"""
        if self.best_parameters is None:
            logging.warning("No calibration results to save")
            return
        
        results = {
            'optimal_parameters': {
                'subspace_coupling': float(self.best_parameters[0]),
                'grid_spacing': float(self.best_parameters[1]),
                'medical_field_strength': float(self.best_parameters[2]), 
                'n_projections': int(self.best_parameters[3])
            },
            'best_performance': float(self.best_performance),
            'optimization_history': [
                {
                    'params': step['params'].tolist(),
                    'scores': step['scores'],
                    'total_cost': step['total_cost'],
                    'timestamp': step['timestamp']
                }
                for step in self.optimization_history
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        logging.info(f"Calibration results saved to {filepath}")

    def generate_performance_report(self) -> str:
        """Generate detailed performance analysis report"""
        if not self.optimization_history:
            return "No optimization data available"
        
        # Extract optimization convergence data
        costs = [step['total_cost'] for step in self.optimization_history]
        best_iteration = np.argmin(costs)
        best_step = self.optimization_history[best_iteration]
        
        report = f"""
🌟 Unified System Calibration Report
=====================================

Optimization Summary:
- Total iterations: {len(self.optimization_history)}
- Best iteration: {best_iteration + 1}
- Best total cost: {best_step['total_cost']:.6f}
- Convergence: {((costs[0] - costs[-1]) / costs[0] * 100):.2f}% improvement

Optimal Parameters:
- Subspace coupling: {best_step['params'][0]:.2e}
- Grid spacing: {best_step['params'][1]:.3f} m
- Medical field strength: {best_step['params'][2]:.3f}
- Tomography projections: {int(best_step['params'][3])}

Performance Scores:
- FTL Rate: {best_step['scores']['ftl']:.3f} ({best_step['scores']['ftl']*100:.1f}%)
- Grid Uniformity: {best_step['scores']['grid']:.3f} ({best_step['scores']['grid']*100:.1f}%)
- Medical Precision: {best_step['scores']['medical']:.3f} ({best_step['scores']['medical']*100:.1f}%)
- Tomography Fidelity: {best_step['scores']['tomography']:.3f} ({best_step['scores']['tomography']*100:.1f}%)

Overall System Performance: {np.mean(list(best_step['scores'].values()))*100:.1f}%
"""
        
        return report

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Initialize calibration parameters
    calib_params = CalibrationParams(
        max_iterations=50,
        use_genetic_algorithm=True,
        population_size=20
    )
    
    # Create calibrator
    calibrator = UnifiedSystemCalibrator(calib_params)
    
    # Run calibration
    results = calibrator.run_calibration()
    
    # Display results
    print("🔧 Unified System Calibration Results:")
    print(f"Success: {results['success']}")
    print(f"Optimal Parameters: {results['optimal_parameters']}")
    print(f"Performance Scores: {results['performance_scores']}")
    
    # Generate and display report
    report = calibrator.generate_performance_report()
    print(report)
    
    # Save results
    calibrator.save_calibration_results("calibration_results.json")
