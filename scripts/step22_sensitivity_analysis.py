"""
Step 22: Cross-Validation & Sensitivity Analysis
===============================================

Quantifies how sensitive each subsystem is to its main parameters using:
- Finite difference approximations
- JAX automatic differentiation (where applicable)  
- Monte Carlo sensitivity sampling
- Sobol indices for global sensitivity

Mathematical Foundation:
∂Fidelity/∂n_proj, ∂Rate/∂κ_subspace, ∂Uniformity/∂spacing, ∂Precision/∂field_strength

Sensitivity metrics:
- Local derivatives at operating point
- Global variance-based sensitivity indices  
- Parameter interaction effects
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Callable, Optional
import json
from dataclasses import dataclass
from scipy.stats import norm
import matplotlib.pyplot as plt

# Import JAX for automatic differentiation (if available)
try:
    import jax
    import jax.numpy as jnp
    from jax import grad, value_and_grad
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    logging.warning("JAX not available - using finite differences only")

# Import SALib for Sobol sensitivity analysis (if available)
try:
    from SALib.sample import sobol
    from SALib.analyze import sobol as sobol_analyze
    SALIB_AVAILABLE = True
except ImportError:
    SALIB_AVAILABLE = False
    logging.warning("SALib not available - Sobol analysis disabled")

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
            'usable_bandwidth_hz': min(bandwidth, 1e12)
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
        mean_force = 1.0 / self.params.base_spacing
        std_force = mean_force * 0.1
        
        return {
            'force_statistics': {
                'mean_force': mean_force,
                'std_force': std_force
            }
        }

@dataclass
class SensitivityParams:
    """Parameters for sensitivity analysis"""
    # Finite difference parameters
    fd_step_size: float = 1e-6
    fd_method: str = 'central'  # 'forward', 'backward', 'central'
    
    # Monte Carlo parameters
    mc_samples: int = 1000
    confidence_level: float = 0.95
    
    # Sobol analysis parameters
    sobol_samples: int = 8192  # Must be power of 2
    sobol_calc_second_order: bool = True
    
    # Parameter perturbation ranges (relative)
    perturbation_range: float = 0.1  # ±10% around nominal values
    
    # Baseline parameter values
    nominal_subspace_coupling: float = 1e-15
    nominal_grid_spacing: float = 0.2
    nominal_medical_field: float = 1.0
    nominal_tomography_projections: int = 120

class SensitivityAnalyzer:
    """
    Comprehensive sensitivity analysis for unified warp field system
    
    Provides local and global sensitivity analysis capabilities:
    - Finite difference derivatives
    - JAX automatic differentiation 
    - Monte Carlo uncertainty propagation
    - Sobol variance-based sensitivity indices
    """
    
    def __init__(self, params: SensitivityParams):
        """
        Initialize sensitivity analyzer
        
        Args:
            params: Sensitivity analysis configuration
        """
        self.params = params
        self.sensitivity_results = {}
        self.baseline_performance = None
        
        logging.info("SensitivityAnalyzer initialized")

    def ftl_rate_function(self, subspace_coupling: float) -> float:
        """FTL communication rate as function of subspace coupling"""
        try:
            params = SubspaceParams(
                subspace_coupling=subspace_coupling,
                frequency_range=(1e9, 1e12),
                antenna_gain=40.0
            )
            transceiver = SubspaceTransceiver(params)
            diagnostics = transceiver.run_diagnostics()
            return diagnostics.get('usable_bandwidth_hz', 0.0)
        except:
            return 0.0

    def grid_uniformity_function(self, grid_spacing: float) -> float:
        """Grid force uniformity as function of spacing"""
        try:
            params = GridParams(
                bounds=((-1.0, 1.0), (-1.0, 1.0), (0.0, 1.0)),
                base_spacing=grid_spacing,
                max_grid_points=10000
            )
            grid = ForceFieldGrid(params)
            diagnostics = grid.run_diagnostics()
            
            force_stats = diagnostics.get('force_statistics', {})
            mean_force = force_stats.get('mean_force', 1e-6)
            std_force = force_stats.get('std_force', 1e-6)
            
            uniformity = max(1 - std_force / mean_force, 0.0)
            return uniformity
        except:
            return 0.0

    def medical_precision_function(self, field_strength: float) -> float:
        """Medical precision as function of field strength"""
        try:
            params = MedicalArrayParams(
                array_bounds=((-0.5, 0.5), (-0.5, 0.5), (0.0, 1.0)),
                beam_spacing=0.02,
                global_power_limit=field_strength * 100
            )
            array = MedicalTractorArray(params)
            
            # Test positioning precision
            target_pos = np.array([0.05, 0.02, 0.1])
            desired_pos = np.array([0.03, 0.02, 0.1])
            
            result = array.position_target(target_pos, desired_pos, tissue_type="organ")
            
            if 'force' in result:
                return np.linalg.norm(result['force'])
            return 0.0
        except:
            return 0.0

    def tomography_fidelity_function(self, n_projections: int) -> float:
        """Tomographic fidelity as function of projection count"""
        try:
            # Simplified fidelity model based on projection count
            # More projections generally improve reconstruction quality
            
            # Empirical model: fidelity approaches 1 asymptotically
            # f(n) = 1 - exp(-n/n₀) where n₀ is characteristic scale
            n0 = 60  # Characteristic projection count
            fidelity = 1 - np.exp(-n_projections / n0)
            
            # Add some noise for realism
            noise = 0.05 * np.random.normal()
            return max(0, min(1, fidelity + noise))
        except:
            return 0.0

    def finite_difference_derivative(self, func: Callable, x0: float, 
                                   step_size: Optional[float] = None) -> float:
        """
        Compute finite difference derivative
        
        Args:
            func: Function to differentiate
            x0: Point at which to evaluate derivative
            step_size: Step size for finite difference
            
        Returns:
            Approximate derivative value
        """
        if step_size is None:
            step_size = self.params.fd_step_size
        
        if self.params.fd_method == 'forward':
            return (func(x0 + step_size) - func(x0)) / step_size
        elif self.params.fd_method == 'backward':
            return (func(x0) - func(x0 - step_size)) / step_size
        elif self.params.fd_method == 'central':
            return (func(x0 + step_size) - func(x0 - step_size)) / (2 * step_size)
        else:
            raise ValueError(f"Unknown finite difference method: {self.params.fd_method}")

    def jax_derivative(self, func: Callable, x0: float) -> float:
        """
        Compute derivative using JAX automatic differentiation
        
        Args:
            func: Function to differentiate (must be JAX-compatible)
            x0: Point at which to evaluate derivative
            
        Returns:
            Exact derivative value
        """
        if not JAX_AVAILABLE:
            logging.warning("JAX not available - falling back to finite differences")
            return self.finite_difference_derivative(func, x0)
        
        try:
            grad_func = grad(func)
            return float(grad_func(x0))
        except Exception as e:
            logging.warning(f"JAX differentiation failed: {e} - using finite differences")
            return self.finite_difference_derivative(func, x0)

    def local_sensitivity_analysis(self) -> Dict:
        """
        Perform local sensitivity analysis at nominal parameter values
        
        Returns:
            Dictionary of sensitivity derivatives and metrics
        """
        logging.info("Running local sensitivity analysis...")
        
        # Evaluate derivatives at nominal points
        derivatives = {}
        
        # FTL rate sensitivity to subspace coupling
        κ_nominal = self.params.nominal_subspace_coupling
        dRate_dκ = self.finite_difference_derivative(
            self.ftl_rate_function, κ_nominal
        )
        derivatives['ftl_rate_wrt_coupling'] = {
            'value': dRate_dκ,
            'units': 'Hz per (coupling unit)',
            'nominal_point': κ_nominal,
            'relative_sensitivity': dRate_dκ * κ_nominal / max(self.ftl_rate_function(κ_nominal), 1e-12)
        }
        
        # Grid uniformity sensitivity to spacing
        spacing_nominal = self.params.nominal_grid_spacing
        dUniformity_dSpacing = self.finite_difference_derivative(
            self.grid_uniformity_function, spacing_nominal
        )
        derivatives['grid_uniformity_wrt_spacing'] = {
            'value': dUniformity_dSpacing,
            'units': 'uniformity per meter',
            'nominal_point': spacing_nominal,
            'relative_sensitivity': dUniformity_dSpacing * spacing_nominal / max(self.grid_uniformity_function(spacing_nominal), 1e-12)
        }
        
        # Medical precision sensitivity to field strength
        field_nominal = self.params.nominal_medical_field
        dPrecision_dField = self.finite_difference_derivative(
            self.medical_precision_function, field_nominal
        )
        derivatives['medical_precision_wrt_field'] = {
            'value': dPrecision_dField,
            'units': 'force per field unit',
            'nominal_point': field_nominal,
            'relative_sensitivity': dPrecision_dField * field_nominal / max(self.medical_precision_function(field_nominal), 1e-12)
        }
        
        # Tomographic fidelity sensitivity to projections
        proj_nominal = self.params.nominal_tomography_projections
        dFidelity_dProj = self.finite_difference_derivative(
            self.tomography_fidelity_function, float(proj_nominal)
        )
        derivatives['tomography_fidelity_wrt_projections'] = {
            'value': dFidelity_dProj,
            'units': 'fidelity per projection',
            'nominal_point': proj_nominal,
            'relative_sensitivity': dFidelity_dProj * proj_nominal / max(self.tomography_fidelity_function(proj_nominal), 1e-12)
        }
        
        # Compute sensitivity rankings
        relative_sensitivities = {
            k: abs(v['relative_sensitivity']) 
            for k, v in derivatives.items()
        }
        sensitivity_ranking = sorted(
            relative_sensitivities.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        results = {
            'derivatives': derivatives,
            'sensitivity_ranking': sensitivity_ranking,
            'analysis_method': 'finite_differences',
            'nominal_parameters': {
                'subspace_coupling': κ_nominal,
                'grid_spacing': spacing_nominal,
                'medical_field': field_nominal,
                'tomography_projections': proj_nominal
            }
        }
        
        self.sensitivity_results['local'] = results
        logging.info("Local sensitivity analysis completed")
        
        return results

    def monte_carlo_uncertainty_propagation(self) -> Dict:
        """
        Perform Monte Carlo uncertainty propagation analysis
        
        Returns:
            Statistical uncertainty propagation results
        """
        logging.info("Running Monte Carlo uncertainty propagation...")
        
        # Generate parameter samples
        n_samples = self.params.mc_samples
        perturbation = self.params.perturbation_range
        
        # Generate perturbed parameter sets
        κ_samples = norm.rvs(
            loc=self.params.nominal_subspace_coupling,
            scale=self.params.nominal_subspace_coupling * perturbation,
            size=n_samples
        )
        
        spacing_samples = norm.rvs(
            loc=self.params.nominal_grid_spacing,
            scale=self.params.nominal_grid_spacing * perturbation,
            size=n_samples
        )
        
        field_samples = norm.rvs(
            loc=self.params.nominal_medical_field,
            scale=self.params.nominal_medical_field * perturbation,
            size=n_samples
        )
        
        proj_samples = np.random.randint(
            max(1, int(self.params.nominal_tomography_projections * (1 - perturbation))),
            int(self.params.nominal_tomography_projections * (1 + perturbation)),
            size=n_samples
        )
        
        # Evaluate performance for each sample
        ftl_samples = []
        grid_samples = []
        medical_samples = []
        tomography_samples = []
        
        for i in range(n_samples):
            if i % 100 == 0:
                logging.debug(f"MC sample {i}/{n_samples}")
            
            ftl_samples.append(self.ftl_rate_function(κ_samples[i]))
            grid_samples.append(self.grid_uniformity_function(spacing_samples[i]))
            medical_samples.append(self.medical_precision_function(field_samples[i]))
            tomography_samples.append(self.tomography_fidelity_function(proj_samples[i]))
        
        # Statistical analysis
        confidence = self.params.confidence_level
        alpha = 1 - confidence
        
        def compute_stats(samples, name):
            samples = np.array(samples)
            return {
                'mean': np.mean(samples),
                'std': np.std(samples),
                'var': np.var(samples),
                'min': np.min(samples),
                'max': np.max(samples),
                'median': np.median(samples),
                'q25': np.percentile(samples, 25),
                'q75': np.percentile(samples, 75),
                'confidence_interval': [
                    np.percentile(samples, 100 * alpha/2),
                    np.percentile(samples, 100 * (1 - alpha/2))
                ],
                'coefficient_of_variation': np.std(samples) / max(np.mean(samples), 1e-12)
            }
        
        results = {
            'ftl_rate': compute_stats(ftl_samples, 'FTL Rate'),
            'grid_uniformity': compute_stats(grid_samples, 'Grid Uniformity'),
            'medical_precision': compute_stats(medical_samples, 'Medical Precision'),
            'tomography_fidelity': compute_stats(tomography_samples, 'Tomography Fidelity'),
            'n_samples': n_samples,
            'confidence_level': confidence,
            'perturbation_range': perturbation
        }
        
        # Rank uncertainties by coefficient of variation
        cv_ranking = sorted([
            ('FTL Rate', results['ftl_rate']['coefficient_of_variation']),
            ('Grid Uniformity', results['grid_uniformity']['coefficient_of_variation']),
            ('Medical Precision', results['medical_precision']['coefficient_of_variation']),
            ('Tomography Fidelity', results['tomography_fidelity']['coefficient_of_variation'])
        ], key=lambda x: x[1], reverse=True)
        
        results['uncertainty_ranking'] = cv_ranking
        
        self.sensitivity_results['monte_carlo'] = results
        logging.info("Monte Carlo uncertainty propagation completed")
        
        return results

    def sobol_sensitivity_analysis(self) -> Dict:
        """
        Perform Sobol variance-based global sensitivity analysis
        
        Returns:
            Sobol sensitivity indices and interaction effects
        """
        if not SALIB_AVAILABLE:
            logging.warning("SALib not available - Sobol analysis skipped")
            return {'status': 'SALib not available'}
        
        logging.info("Running Sobol sensitivity analysis...")
        
        # Define parameter problem
        problem = {
            'num_vars': 4,
            'names': ['subspace_coupling', 'grid_spacing', 'medical_field', 'tomography_projections'],
            'bounds': [
                [self.params.nominal_subspace_coupling * 0.1, self.params.nominal_subspace_coupling * 10],
                [self.params.nominal_grid_spacing * 0.5, self.params.nominal_grid_spacing * 2.0],
                [self.params.nominal_medical_field * 0.5, self.params.nominal_medical_field * 2.0],
                [60, 240]  # Projection count range
            ]
        }
        
        # Generate Sobol samples
        param_values = sobol.sample(problem, self.params.sobol_samples)
        
        # Evaluate model for each output
        def evaluate_combined_performance(params):
            κ, spacing, field, n_proj = params
            n_proj = int(n_proj)
            
            # Evaluate all performance metrics
            ftl = self.ftl_rate_function(κ)
            grid = self.grid_uniformity_function(spacing)
            medical = self.medical_precision_function(field)
            tomography = self.tomography_fidelity_function(n_proj)
            
            # Combined performance score
            combined = np.mean([ftl/1e12, grid, medical/1e-9, tomography])
            
            return [ftl, grid, medical, tomography, combined]
        
        # Evaluate all samples
        logging.info(f"Evaluating {len(param_values)} Sobol samples...")
        outputs = []
        for i, params in enumerate(param_values):
            if i % 500 == 0:
                logging.debug(f"Sobol sample {i}/{len(param_values)}")
            outputs.append(evaluate_combined_performance(params))
        
        outputs = np.array(outputs)
        
        # Analyze sensitivity for each output
        output_names = ['FTL Rate', 'Grid Uniformity', 'Medical Precision', 'Tomography Fidelity', 'Combined']
        sobol_results = {}
        
        for i, output_name in enumerate(output_names):
            try:
                Si = sobol_analyze.analyze(
                    problem, 
                    outputs[:, i], 
                    calc_second_order=self.params.sobol_calc_second_order
                )
                
                sobol_results[output_name] = {
                    'S1': Si['S1'].tolist(),  # First-order indices
                    'S1_conf': Si['S1_conf'].tolist(),  # Confidence intervals
                    'ST': Si['ST'].tolist(),  # Total-order indices
                    'ST_conf': Si['ST_conf'].tolist(),
                }
                
                if self.params.sobol_calc_second_order:
                    sobol_results[output_name]['S2'] = Si['S2'].tolist()  # Second-order indices
                    sobol_results[output_name]['S2_conf'] = Si['S2_conf'].tolist()
                
            except Exception as e:
                logging.warning(f"Sobol analysis failed for {output_name}: {e}")
                sobol_results[output_name] = {'error': str(e)}
        
        results = {
            'sobol_indices': sobol_results,
            'problem_definition': problem,
            'n_samples': len(param_values),
            'parameter_names': problem['names']
        }
        
        # Extract most important parameters for each output
        for output_name, indices in sobol_results.items():
            if 'S1' in indices:
                # Rank parameters by first-order sensitivity
                param_importance = sorted(
                    zip(problem['names'], indices['S1']),
                    key=lambda x: x[1],
                    reverse=True
                )
                results[f'{output_name}_parameter_ranking'] = param_importance
        
        self.sensitivity_results['sobol'] = results
        logging.info("Sobol sensitivity analysis completed")
        
        return results

    def run_comprehensive_analysis(self) -> Dict:
        """
        Run all sensitivity analysis methods
        
        Returns:
            Complete sensitivity analysis results
        """
        logging.info("Starting comprehensive sensitivity analysis...")
        
        # Run all analysis methods
        local_results = self.local_sensitivity_analysis()
        mc_results = self.monte_carlo_uncertainty_propagation()
        sobol_results = self.sobol_sensitivity_analysis()
        
        # Combine results
        comprehensive_results = {
            'local_sensitivity': local_results,
            'monte_carlo_uncertainty': mc_results,
            'sobol_global_sensitivity': sobol_results,
            'analysis_summary': self._generate_analysis_summary()
        }
        
        logging.info("Comprehensive sensitivity analysis completed")
        
        return comprehensive_results

    def _generate_analysis_summary(self) -> Dict:
        """Generate summary of key sensitivity insights"""
        summary = {
            'most_sensitive_parameters': [],
            'least_sensitive_parameters': [],
            'highest_uncertainty_outputs': [],
            'key_insights': []
        }
        
        # Extract insights from local sensitivity
        if 'local' in self.sensitivity_results:
            local = self.sensitivity_results['local']
            if 'sensitivity_ranking' in local:
                ranking = local['sensitivity_ranking']
                summary['most_sensitive_parameters'] = ranking[:2]  # Top 2
                summary['least_sensitive_parameters'] = ranking[-2:]  # Bottom 2
        
        # Extract insights from Monte Carlo
        if 'monte_carlo' in self.sensitivity_results:
            mc = self.sensitivity_results['monte_carlo']
            if 'uncertainty_ranking' in mc:
                summary['highest_uncertainty_outputs'] = mc['uncertainty_ranking'][:2]
        
        # Generate key insights
        insights = [
            "Local derivatives provide instantaneous sensitivity at nominal parameters",
            "Monte Carlo propagation quantifies uncertainty under parameter variations",
            "Higher coefficient of variation indicates greater output uncertainty"
        ]
        
        if SALIB_AVAILABLE:
            insights.append("Sobol indices quantify parameter importance and interaction effects")
        
        summary['key_insights'] = insights
        
        return summary

    def save_results(self, filepath: str):
        """Save sensitivity analysis results to JSON file"""
        if not self.sensitivity_results:
            logging.warning("No sensitivity results to save")
            return
        
        with open(filepath, 'w') as f:
            json.dump(self.sensitivity_results, f, indent=2)
        
        logging.info(f"Sensitivity analysis results saved to {filepath}")

    def generate_sensitivity_report(self) -> str:
        """Generate detailed sensitivity analysis report"""
        if not self.sensitivity_results:
            return "No sensitivity analysis data available"
        
        report = "🔍 Sensitivity Analysis Report\n"
        report += "=" * 50 + "\n\n"
        
        # Local sensitivity summary
        if 'local' in self.sensitivity_results:
            local = self.sensitivity_results['local']
            report += "Local Sensitivity Analysis:\n"
            report += "-" * 30 + "\n"
            
            if 'sensitivity_ranking' in local:
                report += "Parameter Sensitivity Ranking:\n"
                for i, (param, sensitivity) in enumerate(local['sensitivity_ranking'], 1):
                    report += f"{i}. {param}: {sensitivity:.6f}\n"
                report += "\n"
        
        # Monte Carlo uncertainty summary
        if 'monte_carlo' in self.sensitivity_results:
            mc = self.sensitivity_results['monte_carlo']
            report += "Monte Carlo Uncertainty Propagation:\n"
            report += "-" * 40 + "\n"
            
            if 'uncertainty_ranking' in mc:
                report += "Output Uncertainty Ranking (by CV):\n"
                for i, (output, cv) in enumerate(mc['uncertainty_ranking'], 1):
                    report += f"{i}. {output}: {cv:.4f}\n"
                report += "\n"
        
        # Sobol analysis summary
        if 'sobol' in self.sensitivity_results:
            sobol = self.sensitivity_results['sobol']
            report += "Sobol Global Sensitivity Analysis:\n"
            report += "-" * 40 + "\n"
            
            # Show parameter rankings for combined performance
            combined_key = 'Combined_parameter_ranking'
            if combined_key in sobol:
                report += "Most Important Parameters (Combined Performance):\n"
                for i, (param, importance) in enumerate(sobol[combined_key][:3], 1):
                    report += f"{i}. {param}: {importance:.4f}\n"
                report += "\n"
        
        return report

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Initialize sensitivity parameters
    sens_params = SensitivityParams(
        mc_samples=500,  # Reduced for faster testing
        sobol_samples=1024
    )
    
    # Create analyzer
    analyzer = SensitivityAnalyzer(sens_params)
    
    # Run comprehensive analysis
    results = analyzer.run_comprehensive_analysis()
    
    # Display results
    print("🔍 Sensitivity Analysis Results:")
    print(analyzer.generate_sensitivity_report())
    
    # Save results
    analyzer.save_results("sensitivity_analysis_results.json")
