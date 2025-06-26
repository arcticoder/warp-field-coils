#!/usr/bin/env python3
"""
Multi-Physics Weight Optimization for Warp Field Coil Design
Performs grid search over penalty weights to find Pareto sweet-spot
"""

import numpy as np
import itertools
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import json
from pathlib import Path

class WeightOptimizer:
    """
    Optimize multi-physics penalty weights for warp field coil design.
    
    Performs systematic grid search over (α_q, α_m, α_t) to find
    optimal balance between quantum, mechanical, and thermal constraints.
    """
    
    def __init__(self, coil_optimizer):
        """
        Initialize weight optimizer.
        
        Args:
            coil_optimizer: AdvancedCoilOptimizer instance
        """
        self.coil_optimizer = coil_optimizer
        self.weight_sweep_results = {}
        
    def perform_weight_sweep(self, 
                           alpha_q_range: List[float] = [1e-4, 1e-3, 1e-2],
                           alpha_m_range: List[float] = [1e2, 1e3, 1e4], 
                           alpha_t_range: List[float] = [1e1, 1e2, 1e3],
                           initial_params: Optional[np.ndarray] = None,
                           constraint_params: Optional[Dict] = None) -> Dict:
        """
        Perform systematic weight sweep over penalty parameters.
        
        Args:
            alpha_q_range: Quantum penalty weights to test
            alpha_m_range: Mechanical penalty weights to test
            alpha_t_range: Thermal penalty weights to test
            initial_params: Initial optimization parameters
            constraint_params: Physical constraint parameters
            
        Returns:
            Complete sweep results
        """
        print("🎯 MULTI-PHYSICS WEIGHT OPTIMIZATION")
        print("=" * 50)
        
        # Default parameters
        if initial_params is None:
            initial_params = np.array([0.1, 1.0, 0.3])
        
        if constraint_params is None:
            constraint_params = {
                'thickness': 0.005,      # 5mm conductor
                'sigma_yield': 300e6,    # 300 MPa
                'rho_cu': 1.7e-8,       # Copper resistivity
                'area': 1e-6,           # 1mm² cross-section
                'P_max': 1e6            # 1 MW max power
            }
        
        # Generate all weight combinations
        weight_combinations = list(itertools.product(alpha_q_range, alpha_m_range, alpha_t_range))
        total_combinations = len(weight_combinations)
        
        print(f"Testing {total_combinations} weight combinations...")
        print(f"α_q ∈ {alpha_q_range}")
        print(f"α_m ∈ {alpha_m_range}")  
        print(f"α_t ∈ {alpha_t_range}")
        
        results = {
            'weight_combinations': weight_combinations,
            'objectives': {},
            'penalty_components': {},
            'convergence_status': {},
            'optimization_results': {}
        }
        
        # Sweep over all combinations
        for i, (alpha_q, alpha_m, alpha_t) in enumerate(weight_combinations):
            print(f"\n[{i+1}/{total_combinations}] Testing αq={alpha_q:.0e}, "
                  f"αm={alpha_m:.0e}, αt={alpha_t:.0e}")
            
            try:
                # Define objective function with current weights
                def weighted_objective(params):
                    return self.coil_optimizer.objective_full_multiphysics(
                        params, 
                        alpha_q=alpha_q, 
                        alpha_m=alpha_m, 
                        alpha_t=alpha_t,
                        **constraint_params
                    )
                
                # Run optimization
                from scipy.optimize import minimize
                opt_result = minimize(
                    weighted_objective,
                    initial_params,
                    method='L-BFGS-B',
                    bounds=[(0.01, 5.0), (0.1, 10.0), (0.1, 2.0)],
                    options={'maxiter': 100, 'ftol': 1e-9}
                )
                
                if opt_result.success:
                    # Store results
                    weights = (alpha_q, alpha_m, alpha_t)
                    results['objectives'][weights] = opt_result.fun
                    results['convergence_status'][weights] = 'SUCCESS'
                    results['optimization_results'][weights] = opt_result
                    
                    # Get penalty components
                    components = self.coil_optimizer.get_penalty_components(
                        opt_result.x, **constraint_params
                    )
                    results['penalty_components'][weights] = components
                    
                    print(f"  ✓ J_total = {opt_result.fun:.6e}")
                    print(f"    Classical: {components['classical']:.2e}")
                    print(f"    Quantum: {components['quantum']:.2e}")
                    print(f"    Mechanical: {components['mechanical']:.2e}")
                    print(f"    Thermal: {components['thermal']:.2e}")
                    
                else:
                    weights = (alpha_q, alpha_m, alpha_t)
                    results['objectives'][weights] = np.inf
                    results['convergence_status'][weights] = f'FAILED: {opt_result.message}'
                    print(f"  ❌ Optimization failed: {opt_result.message}")
                
            except Exception as e:
                weights = (alpha_q, alpha_m, alpha_t)
                results['objectives'][weights] = np.inf
                results['convergence_status'][weights] = f'ERROR: {str(e)}'
                print(f"  ❌ Error: {e}")
        
        # Find optimal weights
        optimal_weights = self._find_optimal_weights(results)
        results['optimal_weights'] = optimal_weights
        
        # Store for later analysis
        self.weight_sweep_results = results
        
        print(f"\n✓ Weight sweep complete")
        print(f"Optimal weights: αq={optimal_weights[0]:.0e}, "
              f"αm={optimal_weights[1]:.0e}, αt={optimal_weights[2]:.0e}")
        print(f"Optimal objective: {results['objectives'][optimal_weights]:.6e}")
        
        return results
    
    def _find_optimal_weights(self, results: Dict) -> Tuple[float, float, float]:
        """Find optimal weight combination from sweep results."""
        objectives = results['objectives']
        
        # Filter out failed optimizations
        valid_objectives = {k: v for k, v in objectives.items() if np.isfinite(v)}
        
        if not valid_objectives:
            print("⚠️ No valid weight combinations found")
            return (1e-3, 1e3, 1e2)  # Default fallback
        
        # Find minimum objective
        optimal_weights = min(valid_objectives, key=valid_objectives.get)
        
        return optimal_weights
    
    def analyze_weight_sensitivity(self, results: Dict) -> Dict:
        """
        Analyze sensitivity of optimal objective to weight variations.
        
        Args:
            results: Weight sweep results
            
        Returns:
            Sensitivity analysis results
        """
        print("\n📊 WEIGHT SENSITIVITY ANALYSIS")
        print("-" * 40)
        
        objectives = results['objectives']
        valid_objectives = {k: v for k, v in objectives.items() if np.isfinite(v)}
        
        if len(valid_objectives) < 2:
            print("⚠️ Insufficient data for sensitivity analysis")
            return {}
        
        # Convert to arrays for analysis
        weights_array = np.array(list(valid_objectives.keys()))
        objectives_array = np.array(list(valid_objectives.values()))
        
        # Analyze each weight dimension
        alpha_q_vals = weights_array[:, 0]
        alpha_m_vals = weights_array[:, 1]
        alpha_t_vals = weights_array[:, 2]
        
        sensitivity_analysis = {
            'alpha_q_sensitivity': self._compute_partial_sensitivity(
                alpha_q_vals, objectives_array, 'quantum'
            ),
            'alpha_m_sensitivity': self._compute_partial_sensitivity(
                alpha_m_vals, objectives_array, 'mechanical'
            ),
            'alpha_t_sensitivity': self._compute_partial_sensitivity(
                alpha_t_vals, objectives_array, 'thermal'
            ),
            'weight_correlation_matrix': np.corrcoef(weights_array.T),
            'objective_statistics': {
                'min': np.min(objectives_array),
                'max': np.max(objectives_array),
                'mean': np.mean(objectives_array),
                'std': np.std(objectives_array),
                'range': np.max(objectives_array) - np.min(objectives_array)
            }
        }
        
        # Print summary
        print(f"Objective range: {sensitivity_analysis['objective_statistics']['range']:.2e}")
        print(f"Most sensitive weight: {self._find_most_sensitive_weight(sensitivity_analysis)}")
        
        return sensitivity_analysis
    
    def _compute_partial_sensitivity(self, weight_vals: np.ndarray, 
                                   objectives: np.ndarray, 
                                   weight_name: str) -> Dict:
        """Compute sensitivity of objective to specific weight parameter."""
        unique_weights = np.unique(weight_vals)
        
        if len(unique_weights) < 2:
            return {'sensitivity': 0.0, 'gradient': 0.0}
        
        # Group objectives by weight value
        weight_groups = {}
        for w in unique_weights:
            mask = weight_vals == w
            weight_groups[w] = objectives[mask]
        
        # Compute mean objective for each weight
        mean_objectives = {w: np.mean(objs) for w, objs in weight_groups.items()}
        
        # Estimate gradient (finite difference)
        weights_sorted = sorted(mean_objectives.keys())
        if len(weights_sorted) >= 2:
            dw = weights_sorted[1] - weights_sorted[0]
            dJ = mean_objectives[weights_sorted[1]] - mean_objectives[weights_sorted[0]]
            gradient = dJ / dw if dw > 0 else 0.0
        else:
            gradient = 0.0
        
        # Sensitivity measure (normalized gradient)
        obj_range = max(mean_objectives.values()) - min(mean_objectives.values())
        weight_range = max(weights_sorted) - min(weights_sorted)
        
        sensitivity = abs(gradient * weight_range / obj_range) if obj_range > 0 else 0.0
        
        return {
            'sensitivity': sensitivity,
            'gradient': gradient,
            'mean_objectives': mean_objectives,
            'weight_name': weight_name
        }
    
    def _find_most_sensitive_weight(self, sensitivity_analysis: Dict) -> str:
        """Find which weight parameter has highest sensitivity."""
        sensitivities = {
            'alpha_q': sensitivity_analysis['alpha_q_sensitivity']['sensitivity'],
            'alpha_m': sensitivity_analysis['alpha_m_sensitivity']['sensitivity'],
            'alpha_t': sensitivity_analysis['alpha_t_sensitivity']['sensitivity']
        }
        
        return max(sensitivities, key=sensitivities.get)
    
    def plot_weight_sweep_results(self, results: Dict, save_path: Optional[str] = None) -> None:
        """
        Visualize weight sweep results.
        
        Args:
            results: Weight sweep results
            save_path: Optional path to save plots
        """
        objectives = results['objectives']
        valid_results = {k: v for k, v in objectives.items() if np.isfinite(v)}
        
        if len(valid_results) < 2:
            print("⚠️ Insufficient data for plotting")
            return
        
        # Extract data
        weights_array = np.array(list(valid_results.keys()))
        objectives_array = np.array(list(valid_results.values()))
        
        alpha_q_vals = weights_array[:, 0]
        alpha_m_vals = weights_array[:, 1]
        alpha_t_vals = weights_array[:, 2]
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Objective vs alpha_q
        axes[0, 0].semilogx(alpha_q_vals, objectives_array, 'bo', alpha=0.7)
        axes[0, 0].set_xlabel('α_q (Quantum Weight)')
        axes[0, 0].set_ylabel('Total Objective J_total')
        axes[0, 0].set_title('Objective vs Quantum Weight')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Objective vs alpha_m
        axes[0, 1].semilogx(alpha_m_vals, objectives_array, 'ro', alpha=0.7)
        axes[0, 1].set_xlabel('α_m (Mechanical Weight)')
        axes[0, 1].set_ylabel('Total Objective J_total')
        axes[0, 1].set_title('Objective vs Mechanical Weight')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Objective vs alpha_t
        axes[1, 0].semilogx(alpha_t_vals, objectives_array, 'go', alpha=0.7)
        axes[1, 0].set_xlabel('α_t (Thermal Weight)')
        axes[1, 0].set_ylabel('Total Objective J_total')
        axes[1, 0].set_title('Objective vs Thermal Weight')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 3D scatter plot (if possible)
        if len(np.unique(alpha_q_vals)) > 1 and len(np.unique(alpha_m_vals)) > 1:
            scatter = axes[1, 1].scatter(alpha_q_vals, alpha_m_vals, 
                                       c=objectives_array, cmap='viridis',
                                       alpha=0.7, s=50)
            axes[1, 1].set_xscale('log')
            axes[1, 1].set_yscale('log')
            axes[1, 1].set_xlabel('α_q (Quantum Weight)')
            axes[1, 1].set_ylabel('α_m (Mechanical Weight)')
            axes[1, 1].set_title('Weight Space Exploration')
            plt.colorbar(scatter, ax=axes[1, 1], label='Objective Value')
        else:
            axes[1, 1].text(0.5, 0.5, 'Insufficient data\nfor 2D plot', 
                          ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Weight Space Exploration')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Weight sweep plots saved to {save_path}")
        
        plt.show()
    
    def save_results(self, results: Dict, filename: str) -> None:
        """Save weight optimization results to file."""
        # Convert numpy arrays and complex objects to JSON-serializable format
        serializable_results = {
            'weight_combinations': results['weight_combinations'],
            'objectives': {str(k): float(v) for k, v in results['objectives'].items()},
            'convergence_status': {str(k): v for k, v in results['convergence_status'].items()},
            'optimal_weights': results['optimal_weights'],
            'penalty_components': {}
        }
        
        # Convert penalty components
        for weights, components in results['penalty_components'].items():
            serializable_results['penalty_components'][str(weights)] = {
                k: float(v) for k, v in components.items()
            }
        
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"✓ Weight optimization results saved to {filename}")

def main():
    """Demonstrate weight optimization functionality."""
    print("🎯 MULTI-PHYSICS WEIGHT OPTIMIZER")
    print("=" * 50)
    
    # This would be integrated with actual coil optimizer
    print("Weight optimization framework implemented:")
    print("✓ Systematic grid search over (α_q, α_m, α_t)")
    print("✓ Pareto sweet-spot identification")
    print("✓ Sensitivity analysis")
    print("✓ Visualization tools")
    print("✓ Results export capability")
    
    print("\nReady for integration with AdvancedCoilOptimizer!")

if __name__ == "__main__":
    main()
