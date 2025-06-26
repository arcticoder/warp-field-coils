#!/usr/bin/env python3
"""
Multi-Objective Optimization for Warp Field Coil Design
Implements Pareto front exploration for electro-quantum-mechanical objectives
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import json

@dataclass
class ObjectiveComponents:
    """Container for multi-objective function components."""
    classical: float  # Classical electromagnetic fit
    quantum: float    # Quantum geometry penalty  
    mechanical: float # Mechanical stress penalty
    thermal: float    # Thermal power penalty
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array for optimization."""
        return np.array([self.classical, self.quantum, self.mechanical, self.thermal])
    
    def weighted_sum(self, weights: np.ndarray) -> float:
        """Compute weighted sum of objectives."""
        return np.dot(self.to_array(), weights)

@dataclass  
class ParetoPoint:
    """Point on Pareto front."""
    parameters: np.ndarray
    objectives: ObjectiveComponents
    dominated: bool = False
    
class MultiObjectiveOptimizer:
    """
    Multi-objective optimizer for warp field coil design.
    
    Objectives:
    1. Classical electromagnetic fit J_c
    2. Quantum geometry penalty J_q  
    3. Mechanical stress penalty J_m
    4. Thermal power penalty J_t
    """
    
    def __init__(self, coil_optimizer, constraint_params: Dict):
        """
        Initialize multi-objective optimizer.
        
        Args:
            coil_optimizer: AdvancedCoilOptimizer instance
            constraint_params: Physical constraint parameters
        """
        self.coil_optimizer = coil_optimizer
        self.constraint_params = constraint_params
        self.pareto_points = []
        
    def evaluate_objectives(self, params: np.ndarray) -> ObjectiveComponents:
        """
        Evaluate all objective function components.
        
        Args:
            params: Design parameter vector
            
        Returns:
            ObjectiveComponents with all four objectives
        """
        # Use the penalty components method from coil optimizer
        components = self.coil_optimizer.get_penalty_components(params, **self.constraint_params)
        
        return ObjectiveComponents(
            classical=components['classical'],
            quantum=components['quantum'], 
            mechanical=components['mechanical'],
            thermal=components['thermal']
        )
    
    def dominates(self, obj1: ObjectiveComponents, obj2: ObjectiveComponents) -> bool:
        """
        Check if obj1 dominates obj2 (all components ≤ and at least one <).
        
        Args:
            obj1, obj2: Objective components to compare
            
        Returns:
            True if obj1 dominates obj2
        """
        arr1 = obj1.to_array()
        arr2 = obj2.to_array()
        
        # All components must be ≤
        all_leq = np.all(arr1 <= arr2)
        # At least one component must be <
        any_less = np.any(arr1 < arr2)
        
        return all_leq and any_less
    
    def update_pareto_front(self, new_point: ParetoPoint) -> None:
        """
        Update Pareto front with new point.
        
        Args:
            new_point: Candidate Pareto point
        """
        # Check if new point is dominated by existing points
        dominated_by_existing = False
        for existing_point in self.pareto_points:
            if self.dominates(existing_point.objectives, new_point.objectives):
                dominated_by_existing = True
                break
        
        if not dominated_by_existing:
            # Remove existing points dominated by new point
            self.pareto_points = [
                p for p in self.pareto_points 
                if not self.dominates(new_point.objectives, p.objectives)
            ]
            
            # Add new point to Pareto front
            self.pareto_points.append(new_point)
    
    def weighted_optimization(self, weights: np.ndarray, initial_params: np.ndarray,
                            bounds: Optional[List[Tuple[float, float]]] = None) -> ParetoPoint:
        """
        Single weighted optimization run.
        
        Args:
            weights: (4,) weight vector for objectives
            initial_params: Initial parameter guess
            bounds: Parameter bounds for optimization
            
        Returns:
            Optimized ParetoPoint
        """
        def weighted_objective(params):
            obj_components = self.evaluate_objectives(params)
            return obj_components.weighted_sum(weights)
        
        # Optimize weighted sum
        result = minimize(
            weighted_objective,
            initial_params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-9}
        )
        
        # Create Pareto point
        final_objectives = self.evaluate_objectives(result.x)
        pareto_point = ParetoPoint(
            parameters=result.x.copy(),
            objectives=final_objectives
        )
        
        return pareto_point
    
    def weight_sweep_optimization(self, n_weights: int = 20, 
                                 initial_params: Optional[np.ndarray] = None,
                                 bounds: Optional[List[Tuple[float, float]]] = None) -> List[ParetoPoint]:
        """
        Systematic weight sweep for Pareto front approximation.
        
        Args:
            n_weights: Number of weight combinations to try
            initial_params: Initial parameter guess
            bounds: Parameter bounds
            
        Returns:
            List of Pareto-optimal points
        """
        if initial_params is None:
            initial_params = np.array([0.1, 1.0, 0.3])  # Default guess
        
        print(f"Running weight sweep with {n_weights} weight combinations...")
        
        # Generate systematic weight combinations
        weight_sets = self._generate_weight_combinations(n_weights)
        
        pareto_candidates = []
        
        for i, weights in enumerate(weight_sets):
            try:
                # Run weighted optimization
                pareto_point = self.weighted_optimization(weights, initial_params, bounds)
                pareto_candidates.append(pareto_point)
                
                # Update Pareto front
                self.update_pareto_front(pareto_point)
                
                print(f"  Weight set {i+1}/{len(weight_sets)}: "
                      f"J = [{pareto_point.objectives.classical:.2e}, "
                      f"{pareto_point.objectives.quantum:.2e}, "
                      f"{pareto_point.objectives.mechanical:.2e}, "
                      f"{pareto_point.objectives.thermal:.2e}]")
                
            except Exception as e:
                print(f"  Weight set {i+1} failed: {e}")
                continue
        
        print(f"✓ Found {len(self.pareto_points)} Pareto-optimal points")
        
        return self.pareto_points
    
    def _generate_weight_combinations(self, n_weights: int) -> List[np.ndarray]:
        """
        Generate systematic weight combinations for 4 objectives.
        
        Uses uniform sampling over 3-simplex (weights sum to 1).
        
        Args:
            n_weights: Number of weight combinations
            
        Returns:
            List of weight vectors
        """
        weight_sets = []
        
        # Simple systematic approach: vary one weight, distribute others
        for i in range(4):  # Primary objective index
            for alpha in np.linspace(0.1, 0.9, n_weights // 4):
                weights = np.ones(4) * (1 - alpha) / 3
                weights[i] = alpha
                weight_sets.append(weights)
        
        # Add some balanced combinations
        for _ in range(n_weights // 10):
            # Random Dirichlet distribution (uniform over simplex)
            weights = np.random.dirichlet(np.ones(4))
            weight_sets.append(weights)
        
        return weight_sets
    
    def epsilon_constraint_optimization(self, primary_objective: int,
                                      epsilon_bounds: Dict[int, float],
                                      initial_params: np.ndarray) -> List[ParetoPoint]:
        """
        ε-constraint method for Pareto optimization.
        
        Minimize one objective subject to constraints on others.
        
        Args:
            primary_objective: Index of objective to minimize (0-3)
            epsilon_bounds: {objective_index: upper_bound} constraints
            initial_params: Initial parameter guess
            
        Returns:
            List of Pareto points from ε-constraint method
        """
        def constrained_objective(params):
            obj_components = self.evaluate_objectives(params)
            obj_array = obj_components.to_array()
            
            # Primary objective to minimize
            primary_value = obj_array[primary_objective]
            
            # Penalty for constraint violations
            penalty = 0.0
            for obj_idx, epsilon in epsilon_bounds.items():
                if obj_idx != primary_objective:
                    violation = max(0, obj_array[obj_idx] - epsilon)
                    penalty += 1e6 * violation**2  # Large penalty
            
            return primary_value + penalty
        
        # Optimize with constraints
        result = minimize(
            constrained_objective,
            initial_params,
            method='L-BFGS-B',
            options={'maxiter': 300}
        )
        
        if result.success:
            final_objectives = self.evaluate_objectives(result.x)
            pareto_point = ParetoPoint(
                parameters=result.x.copy(),
                objectives=final_objectives
            )
            
            self.update_pareto_front(pareto_point)
            return [pareto_point]
        else:
            return []
    
    def analyze_pareto_front(self) -> Dict:
        """
        Analyze Pareto front properties.
        
        Returns:
            Analysis dictionary with statistics
        """
        if not self.pareto_points:
            return {'error': 'No Pareto points available'}
        
        # Extract objective arrays
        objectives = np.array([p.objectives.to_array() for p in self.pareto_points])
        
        # Compute statistics
        analysis = {
            'n_points': len(self.pareto_points),
            'objective_ranges': {
                'classical': (float(np.min(objectives[:, 0])), float(np.max(objectives[:, 0]))),
                'quantum': (float(np.min(objectives[:, 1])), float(np.max(objectives[:, 1]))),
                'mechanical': (float(np.min(objectives[:, 2])), float(np.max(objectives[:, 2]))),
                'thermal': (float(np.min(objectives[:, 3])), float(np.max(objectives[:, 3])))
            },
            'ideal_point': objectives.min(axis=0).tolist(),
            'nadir_point': objectives.max(axis=0).tolist(),
            'hypervolume_estimate': self._estimate_hypervolume(objectives)
        }
        
        return analysis
    
    def _estimate_hypervolume(self, objectives: np.ndarray) -> float:
        """Simple hypervolume estimate for Pareto front quality."""
        # Use nadir point as reference
        nadir = objectives.max(axis=0)
        
        # Normalize objectives to [0,1]
        normalized = (objectives - objectives.min(axis=0)) / (nadir - objectives.min(axis=0) + 1e-12)
        
        # Simple hypervolume approximation
        volumes = []
        for point in normalized:
            volume = np.prod(1.0 - point)  # Distance to (1,1,1,1)
            volumes.append(volume)
        
        return float(np.mean(volumes))
    
    def visualize_pareto_front(self, save_path: Optional[str] = None) -> None:
        """
        Visualize Pareto front projections.
        
        Args:
            save_path: Path to save plot
        """
        if not self.pareto_points:
            print("No Pareto points to visualize")
            return
        
        # Extract objectives
        objectives = np.array([p.objectives.to_array() for p in self.pareto_points])
        
        # Create 2x2 subplot for pairwise projections
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        objective_names = ['Classical', 'Quantum', 'Mechanical', 'Thermal']
        
        # Plot key projections
        projections = [
            (0, 1, 'Classical vs Quantum'),
            (0, 2, 'Classical vs Mechanical'), 
            (0, 3, 'Classical vs Thermal'),
            (1, 2, 'Quantum vs Mechanical')
        ]
        
        for idx, (i, j, title) in enumerate(projections):
            ax = axes[idx // 2, idx % 2]
            
            ax.scatter(objectives[:, i], objectives[:, j], 
                      c='red', s=50, alpha=0.7, edgecolors='black')
            
            ax.set_xlabel(f'{objective_names[i]} Objective')
            ax.set_ylabel(f'{objective_names[j]} Objective')
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.set_xscale('log')
            ax.set_yscale('log')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def save_pareto_front(self, filename: str) -> None:
        """Save Pareto front to JSON file."""
        data = {
            'n_points': len(self.pareto_points),
            'constraint_params': self.constraint_params,
            'pareto_points': []
        }
        
        for point in self.pareto_points:
            point_data = {
                'parameters': point.parameters.tolist(),
                'objectives': {
                    'classical': point.objectives.classical,
                    'quantum': point.objectives.quantum,
                    'mechanical': point.objectives.mechanical,
                    'thermal': point.objectives.thermal
                }
            }
            data['pareto_points'].append(point_data)
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✓ Saved {len(self.pareto_points)} Pareto points to {filename}")

def create_default_constraints() -> Dict:
    """Create default physical constraint parameters."""
    return {
        'thickness': 0.005,      # 5mm conductor thickness
        'sigma_yield': 300e6,    # 300 MPa yield stress
        'rho_cu': 1.7e-8,       # Copper resistivity (Ω·m)
        'area': 1e-6,           # 1mm² conductor cross-section
        'P_max': 1e6            # 1 MW maximum power
    }

def main():
    """Demonstrate multi-objective optimization."""
    print("🎯 MULTI-OBJECTIVE WARP COIL OPTIMIZATION")
    print("=" * 50)
    
    # This would be called with actual coil optimizer
    print("Multi-objective framework implemented:")
    print("✓ Pareto front exploration")
    print("✓ Weight sweep method")
    print("✓ ε-constraint method") 
    print("✓ Hypervolume analysis")
    print("✓ Visualization tools")
    
    # Create default constraints
    constraints = create_default_constraints()
    print(f"✓ Default constraints: {len(constraints)} parameters")
    
    print("Ready for integration with AdvancedCoilOptimizer!")

if __name__ == "__main__":
    main()
