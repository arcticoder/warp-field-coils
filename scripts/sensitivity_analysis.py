#!/usr/bin/env python3
"""
Sensitivity Analysis and Uncertainty Quantification for Warp Field Coil Optimization

Performs comprehensive analysis of parameter sensitivity using JAX automatic differentiation.
"""

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from coil_optimizer.advanced_coil_optimizer import AdvancedCoilOptimizer
from stress_energy.exotic_matter_profile import ExoticMatterProfiler, alcubierre_profile

@dataclass
class SensitivityResults:
    """Results from sensitivity analysis."""
    gradient: np.ndarray
    hessian: np.ndarray
    parameter_names: List[str]
    base_objective: float
    perturbation_results: Dict[str, np.ndarray]
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    condition_number: float

class SensitivityAnalyzer:
    """
    Comprehensive sensitivity analysis for warp field coil optimization.
    
    Uses JAX automatic differentiation for exact gradients and Hessians.
    """
    
    def __init__(self, r_min: float = 0.1, r_max: float = 3.0, n_points: int = 50):
        """Initialize sensitivity analyzer."""
        self.optimizer = AdvancedCoilOptimizer(r_min=r_min, r_max=r_max, n_points=n_points)
        self.profiler = ExoticMatterProfiler(r_min=r_min, r_max=r_max, n_points=n_points)
        
        # Set up target profile
        r_array, T00_target = self.profiler.compute_T00_profile(
            lambda r: alcubierre_profile(r, R=2.0, sigma=0.5)
        )
        self.optimizer.set_target_profile(r_array, T00_target)
        
        # JAX-compiled functions for efficiency
        self.grad_fn = jax.grad(self._objective_wrapper)
        self.hess_fn = jax.hessian(self._objective_wrapper)
        self.value_and_grad_fn = jax.value_and_grad(self._objective_wrapper)
    
    def _objective_wrapper(self, params: jnp.ndarray) -> float:
        """Wrapper for objective function compatible with JAX."""
        return self.optimizer.objective_with_quantum(params, "gaussian", alpha=1e-3)
    
    def analyze_parameter_sensitivity(self, base_params: np.ndarray, 
                                    parameter_names: Optional[List[str]] = None,
                                    perturbation_range: float = 0.1) -> SensitivityResults:
        """
        Perform comprehensive parameter sensitivity analysis.
        
        Args:
            base_params: Base parameter values
            parameter_names: Names for parameters (optional)
            perturbation_range: Relative perturbation range (±10% default)
            
        Returns:
            Complete sensitivity analysis results
        """
        if parameter_names is None:
            parameter_names = [f"param_{i}" for i in range(len(base_params))]
        
        print(f"Analyzing sensitivity for {len(base_params)} parameters...")
        
        # Convert to JAX array
        params_jax = jnp.array(base_params)
        
        # Compute base objective, gradient, and Hessian
        base_objective, gradient = self.value_and_grad_fn(params_jax)
        hessian = self.hess_fn(params_jax)
        
        print(f"Base objective: {base_objective:.6e}")
        print(f"Gradient norm: {jnp.linalg.norm(gradient):.6e}")
        
        # Hessian eigenanalysis
        eigenvalues, eigenvectors = jnp.linalg.eigh(hessian)
        condition_number = jnp.max(eigenvalues) / jnp.max(1e-12, jnp.min(eigenvalues))
        
        # Parameter perturbation analysis
        perturbation_results = self._analyze_perturbations(
            params_jax, perturbation_range, parameter_names
        )
        
        return SensitivityResults(
            gradient=np.array(gradient),
            hessian=np.array(hessian),
            parameter_names=parameter_names,
            base_objective=float(base_objective),
            perturbation_results=perturbation_results,
            eigenvalues=np.array(eigenvalues),
            eigenvectors=np.array(eigenvectors),
            condition_number=float(condition_number)
        )
    
    def _analyze_perturbations(self, base_params: jnp.ndarray, 
                             perturbation_range: float,
                             parameter_names: List[str]) -> Dict[str, np.ndarray]:
        """Analyze objective response to parameter perturbations."""
        results = {}
        
        # Perturbation values to test
        perturbations = np.linspace(-perturbation_range, perturbation_range, 21)
        
        for i, param_name in enumerate(parameter_names):
            print(f"  Analyzing {param_name}...")
            
            objectives = []
            for δ in perturbations:
                # Perturb single parameter
                perturbed_params = base_params.at[i].multiply(1 + δ)
                
                try:
                    obj_val = self._objective_wrapper(perturbed_params)
                    objectives.append(float(obj_val))
                except:
                    objectives.append(np.nan)
            
            results[param_name] = {
                'perturbations': perturbations,
                'objectives': np.array(objectives)
            }
        
        return results
    
    def plot_sensitivity_analysis(self, results: SensitivityResults, 
                                save_dir: Path = Path("results/sensitivity")) -> None:
        """Generate comprehensive sensitivity analysis plots."""
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Gradient magnitude plot
        fig, ax = plt.subplots(figsize=(10, 6))
        grad_magnitudes = np.abs(results.gradient)
        bars = ax.bar(results.parameter_names, grad_magnitudes)
        ax.set_ylabel('|∂J/∂p|')
        ax.set_title('Parameter Sensitivity (Gradient Magnitudes)')
        ax.set_yscale('log')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(save_dir / "gradient_magnitudes.png", dpi=300)
        plt.close()
        
        # 2. Hessian eigenvalue spectrum
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Eigenvalue plot
        ax1.semilogy(np.abs(results.eigenvalues), 'bo-')
        ax1.set_xlabel('Eigenvalue Index')
        ax1.set_ylabel('|λ|')
        ax1.set_title('Hessian Eigenvalue Spectrum')
        ax1.grid(True)
        
        # Condition number
        ax2.text(0.5, 0.5, f'Condition Number:\n{results.condition_number:.2e}', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=14,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis('off')
        ax2.set_title('Optimization Conditioning')
        
        plt.tight_layout()
        plt.savefig(save_dir / "hessian_analysis.png", dpi=300)
        plt.close()
        
        # 3. Parameter perturbation responses
        n_params = len(results.parameter_names)
        n_cols = min(3, n_params)
        n_rows = (n_params + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_params == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, param_name in enumerate(results.parameter_names):
            if i < len(axes):
                ax = axes[i]
                data = results.perturbation_results[param_name]
                perturbations = data['perturbations']
                objectives = data['objectives']
                
                # Filter out NaN values
                valid_mask = ~np.isnan(objectives)
                if np.any(valid_mask):
                    ax.plot(perturbations[valid_mask] * 100, objectives[valid_mask], 'b.-')
                    ax.axhline(results.base_objective, color='r', linestyle='--', alpha=0.7)
                    ax.set_xlabel(f'Δ{param_name} (%)')
                    ax.set_ylabel('Objective J')
                    ax.set_title(f'Response to {param_name}')
                    ax.grid(True, alpha=0.3)
                else:
                    ax.text(0.5, 0.5, 'No valid data', ha='center', va='center',
                           transform=ax.transAxes)
        
        # Hide unused subplots
        for i in range(n_params, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(save_dir / "parameter_responses.png", dpi=300)
        plt.close()
        
        # 4. Hessian heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(results.hessian, cmap='RdBu_r', aspect='auto')
        ax.set_xticks(range(len(results.parameter_names)))
        ax.set_yticks(range(len(results.parameter_names)))
        ax.set_xticklabels(results.parameter_names, rotation=45)
        ax.set_yticklabels(results.parameter_names)
        ax.set_title('Hessian Matrix')
        plt.colorbar(im, label='∂²J/∂p_i∂p_j')
        plt.tight_layout()
        plt.savefig(save_dir / "hessian_heatmap.png", dpi=300)
        plt.close()
        
        print(f"Sensitivity analysis plots saved to {save_dir}")
    
    def save_results(self, results: SensitivityResults, 
                    filepath: Path = Path("results/sensitivity/sensitivity_results.json")) -> None:
        """Save sensitivity analysis results to file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        results_dict = asdict(results)
        results_dict['gradient'] = results.gradient.tolist()
        results_dict['hessian'] = results.hessian.tolist()
        results_dict['eigenvalues'] = results.eigenvalues.tolist()
        results_dict['eigenvectors'] = results.eigenvectors.tolist()
        
        # Convert perturbation results
        for param_name in results_dict['perturbation_results']:
            data = results_dict['perturbation_results'][param_name]
            data['perturbations'] = data['perturbations'].tolist()
            data['objectives'] = data['objectives'].tolist()
        
        with open(filepath, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"Results saved to {filepath}")

def main():
    """Run complete sensitivity analysis."""
    print("🔬 WARP FIELD COIL SENSITIVITY ANALYSIS")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = SensitivityAnalyzer()
    
    # Use example optimal parameters (would load from optimization results)
    base_params = np.array([0.1, 2.0, 0.5])  # [amplitude, center, width]
    param_names = ['Amplitude', 'Center_Position', 'Width']
    
    # Perform sensitivity analysis
    results = analyzer.analyze_parameter_sensitivity(
        base_params, param_names, perturbation_range=0.1
    )
    
    # Generate analysis report
    print(f"\n📊 SENSITIVITY ANALYSIS SUMMARY")
    print(f"Base objective value: {results.base_objective:.6e}")
    print(f"Gradient norm: {np.linalg.norm(results.gradient):.6e}")
    print(f"Condition number: {results.condition_number:.2e}")
    
    print(f"\nParameter sensitivities:")
    for i, name in enumerate(param_names):
        print(f"  {name}: {abs(results.gradient[i]):.6e}")
    
    # Generate plots
    analyzer.plot_sensitivity_analysis(results)
    
    # Save results
    analyzer.save_results(results)
    
    print(f"\n✅ Sensitivity analysis complete!")

if __name__ == "__main__":
    main()
