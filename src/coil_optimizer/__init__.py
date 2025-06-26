"""
Coil Geometry Optimizer

Multi-objective optimization for electromagnetic coil geometry and current distribution.
Maximizes field strength while minimizing power consumption and material usage.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass
import scipy.optimize as opt
from scipy.constants import mu_0, pi
from skopt import gp_minimize
from skopt.space import Real, Integer


@dataclass
class OptimizationConstraints:
    """Constraints for coil optimization."""
    max_current: float = 1000.0      # Maximum current (A)
    max_power: float = 10000.0       # Maximum power (W)
    min_radius: float = 0.01         # Minimum coil radius (m)
    max_radius: float = 1.0          # Maximum coil radius (m)
    max_coils: int = 10              # Maximum number of coils
    field_uniformity: float = 0.05   # Required field uniformity (5%)


@dataclass
class OptimizationObjectives:
    """Multi-objective optimization targets."""
    target_field_strength: float = 0.1    # Target field strength (T)
    weight_field: float = 1.0             # Field strength weight
    weight_power: float = 0.3             # Power efficiency weight
    weight_uniformity: float = 0.5        # Field uniformity weight
    weight_material: float = 0.2          # Material usage weight


class CoilGeometryOptimizer:
    """
    Advanced multi-objective optimizer for electromagnetic coil geometry.
    
    Optimizes coil positions, sizes, and orientations for maximum field efficiency
    while respecting power, material, and uniformity constraints.
    """
    
    def __init__(self, constraints: OptimizationConstraints, 
                 objectives: OptimizationObjectives):
        """Initialize the coil geometry optimizer."""
        self.constraints = constraints
        self.objectives = objectives
        self.field_solver = None
        self.optimization_history = []
    
    def set_field_solver(self, solver):
        """Set the electromagnetic field solver for evaluation."""
        self.field_solver = solver
    
    def optimize_geometry(self, n_coils: int, target_region: Dict, 
                         method: str = "bayesian") -> Dict:
        """
        Optimize coil geometry for given specifications.
        
        Args:
            n_coils: Number of coils to optimize
            target_region: Target field region specification
            method: Optimization method ("bayesian", "genetic", "gradient")
            
        Returns:
            Optimization results with optimal coil configuration
        """
        if self.field_solver is None:
            raise ValueError("Field solver not set")
        
        # Define optimization space
        space = self._create_optimization_space(n_coils)
        
        # Select optimization method
        if method == "bayesian":
            return self._bayesian_optimization(space, target_region)
        elif method == "genetic":
            return self._genetic_optimization(space, target_region)
        else:
            return self._gradient_optimization(space, target_region)
    
    def _create_optimization_space(self, n_coils: int) -> List:
        """Create optimization parameter space."""
        space = []
        
        for i in range(n_coils):
            # Coil position (x, y, z)
            space.extend([
                Real(-1.0, 1.0, name=f'x_{i}'),
                Real(-1.0, 1.0, name=f'y_{i}'),
                Real(-1.0, 1.0, name=f'z_{i}')
            ])
            
            # Coil geometry
            space.extend([
                Real(self.constraints.min_radius, self.constraints.max_radius, 
                     name=f'radius_{i}'),
                Real(0.001, 0.1, name=f'height_{i}'),
                Integer(1, 100, name=f'turns_{i}')
            ])
            
            # Current
            space.append(Real(0, self.constraints.max_current, name=f'current_{i}'))
        
        return space
    
    def _bayesian_optimization(self, space: List, target_region: Dict) -> Dict:
        """Bayesian optimization using Gaussian processes."""
        def objective(params):
            return self._evaluate_coil_configuration(params, target_region)
        
        # Run Bayesian optimization
        result = gp_minimize(
            func=objective,
            dimensions=space,
            n_calls=100,  # Number of evaluation points
            n_initial_points=20,
            acq_func='EI',  # Expected Improvement
            random_state=42
        )
        
        return self._process_optimization_result(result, space)
    
    def _genetic_optimization(self, space: List, target_region: Dict) -> Dict:
        """Genetic algorithm optimization."""
        # Convert space to bounds for scipy
        bounds = [(dim.low, dim.high) if hasattr(dim, 'low') else (dim.bounds[0], dim.bounds[1]) 
                 for dim in space]
        
        def objective(params):
            return self._evaluate_coil_configuration(params, target_region)
        
        # Use differential evolution as a genetic algorithm
        result = opt.differential_evolution(
            objective,
            bounds,
            maxiter=50,
            popsize=15,
            atol=1e-6,
            seed=42
        )
        
        return {
            'success': result.success,
            'optimal_params': result.x,
            'objective_value': result.fun,
            'n_evaluations': result.nfev
        }
    
    def _gradient_optimization(self, space: List, target_region: Dict) -> Dict:
        """Gradient-based optimization."""
        bounds = [(dim.low, dim.high) if hasattr(dim, 'low') else (dim.bounds[0], dim.bounds[1]) 
                 for dim in space]
        
        def objective(params):
            return self._evaluate_coil_configuration(params, target_region)
        
        # Initial guess - center of bounds
        x0 = [(b[0] + b[1]) / 2 for b in bounds]
        
        result = opt.minimize(
            objective,
            x0,
            bounds=bounds,
            method='L-BFGS-B'
        )
        
        return {
            'success': result.success,
            'optimal_params': result.x,
            'objective_value': result.fun,
            'n_evaluations': result.nfev
        }
    
    def _evaluate_coil_configuration(self, params: List, target_region: Dict) -> float:
        """Evaluate objective function for given coil configuration."""
        try:
            # Convert parameters to coil geometry
            coils = self._params_to_coils(params)
            
            # Set geometry in field solver
            self.field_solver.set_coil_geometry(coils)
            
            # Evaluate field in target region
            field_metrics = self._compute_field_metrics(target_region)
            
            # Compute multi-objective cost
            cost = self._compute_multi_objective_cost(coils, field_metrics)
            
            # Store evaluation
            self.optimization_history.append({
                'params': params.copy(),
                'cost': cost,
                'field_metrics': field_metrics
            })
            
            return cost
            
        except Exception as e:
            # Return high cost for invalid configurations
            return 1e6
    
    def _params_to_coils(self, params: List):
        """Convert optimization parameters to coil geometry objects."""
        from .field_solver import CoilGeometry
        
        coils = []
        n_params_per_coil = 7  # x, y, z, radius, height, turns, current
        n_coils = len(params) // n_params_per_coil
        
        for i in range(n_coils):
            base_idx = i * n_params_per_coil
            
            coil = CoilGeometry(
                position=(params[base_idx], params[base_idx+1], params[base_idx+2]),
                radius=params[base_idx+3],
                height=params[base_idx+4],
                turns=int(params[base_idx+5]),
                current=params[base_idx+6],
                wire_radius=0.001  # Fixed for now
            )
            coils.append(coil)
        
        return coils
    
    def _compute_field_metrics(self, target_region: Dict) -> Dict:
        """Compute field quality metrics in target region."""
        bounds = target_region['bounds']
        
        # Create evaluation grid
        x = np.linspace(bounds['x'][0], bounds['x'][1], 20)
        y = np.linspace(bounds['y'][0], bounds['y'][1], 20)
        z = np.linspace(bounds['z'][0], bounds['z'][1], 20)
        
        X, Y, Z = np.meshgrid(x, y, z)
        points_flat = (X.flatten(), Y.flatten(), Z.flatten())
        
        # Compute magnetic field
        Bx, By, Bz = self.field_solver.compute_magnetic_field(*points_flat)
        B_magnitude = np.sqrt(Bx**2 + By**2 + Bz**2)
        
        # Field metrics
        mean_field = np.mean(B_magnitude)
        std_field = np.std(B_magnitude)
        uniformity = std_field / (mean_field + 1e-12)
        max_field = np.max(B_magnitude)
        
        return {
            'mean_field': mean_field,
            'std_field': std_field,
            'uniformity': uniformity,
            'max_field': max_field,
            'field_strength_score': mean_field / self.objectives.target_field_strength
        }
    
    def _compute_multi_objective_cost(self, coils: List, field_metrics: Dict) -> float:
        """Compute multi-objective cost function."""
        # Field strength objective (minimize deviation from target)
        field_error = abs(field_metrics['mean_field'] - self.objectives.target_field_strength)
        field_cost = field_error / self.objectives.target_field_strength
        
        # Power consumption objective
        total_power = sum(coil.current**2 * 0.1 for coil in coils)  # Simplified resistance
        power_cost = total_power / self.constraints.max_power
        
        # Field uniformity objective
        uniformity_cost = field_metrics['uniformity'] / self.constraints.field_uniformity
        
        # Material usage objective
        total_material = sum(2 * pi * coil.radius * coil.turns for coil in coils)
        material_cost = total_material / 100.0  # Normalize
        
        # Constraint violations
        constraint_penalty = 0.0
        
        # Current constraints
        for coil in coils:
            if coil.current > self.constraints.max_current:
                constraint_penalty += 100 * (coil.current - self.constraints.max_current)
        
        # Power constraint
        if total_power > self.constraints.max_power:
            constraint_penalty += 100 * (total_power - self.constraints.max_power)
        
        # Combine objectives
        total_cost = (
            self.objectives.weight_field * field_cost +
            self.objectives.weight_power * power_cost +
            self.objectives.weight_uniformity * uniformity_cost +
            self.objectives.weight_material * material_cost +
            constraint_penalty
        )
        
        return total_cost
    
    def _process_optimization_result(self, result, space: List) -> Dict:
        """Process and format optimization results."""
        optimal_coils = self._params_to_coils(result.x)
        
        return {
            'success': True,
            'optimal_coils': optimal_coils,
            'optimal_params': result.x,
            'objective_value': result.fun,
            'n_evaluations': len(result.func_vals),
            'convergence_history': result.func_vals,
            'optimization_method': 'bayesian'
        }


class CurrentDistributionOptimizer:
    """
    Optimizer for current distribution in fixed coil geometry.
    
    Given a fixed coil arrangement, optimizes the current in each coil
    to achieve desired field characteristics.
    """
    
    def __init__(self, field_solver):
        """Initialize current distribution optimizer."""
        self.field_solver = field_solver
        self.optimization_history = []
    
    def optimize_currents(self, target_region: Dict, method: str = "least_squares") -> Dict:
        """
        Optimize current distribution for target field.
        
        Args:
            target_region: Target field region and strength
            method: Optimization method ("least_squares", "quadratic")
            
        Returns:
            Optimized current distribution
        """
        if self.field_solver.geometry is None:
            raise ValueError("Coil geometry not set in field solver")
        
        n_coils = len(self.field_solver.geometry)
        
        if method == "least_squares":
            return self._least_squares_optimization(target_region, n_coils)
        else:
            return self._quadratic_optimization(target_region, n_coils)
    
    def _least_squares_optimization(self, target_region: Dict, n_coils: int) -> Dict:
        """Least squares optimization of current distribution."""
        def objective(currents):
            # Update coil currents
            for i, current in enumerate(currents):
                self.field_solver.geometry[i].current = current
            
            # Evaluate field error
            return self._compute_field_error(target_region)
        
        # Initial guess - equal currents
        x0 = np.ones(n_coils) * 10.0
        
        # Current bounds
        bounds = [(0, 1000) for _ in range(n_coils)]
        
        result = opt.minimize(objective, x0, bounds=bounds, method='L-BFGS-B')
        
        return {
            'success': result.success,
            'optimal_currents': result.x,
            'field_error': result.fun,
            'optimization_result': result
        }
    
    def _quadratic_optimization(self, target_region: Dict, n_coils: int) -> Dict:
        """Quadratic programming optimization."""
        # This would implement a more sophisticated QP formulation
        # For now, fall back to least squares
        return self._least_squares_optimization(target_region, n_coils)
    
    def _compute_field_error(self, target_region: Dict) -> float:
        """Compute RMS error between achieved and target field."""
        bounds = target_region['bounds']
        target_field = target_region.get('target_field', 0.1)
        
        # Sample points in target region
        x = np.linspace(bounds['x'][0], bounds['x'][1], 10)
        y = np.linspace(bounds['y'][0], bounds['y'][1], 10)
        z = np.linspace(bounds['z'][0], bounds['z'][1], 10)
        
        X, Y, Z = np.meshgrid(x, y, z)
        points_flat = (X.flatten(), Y.flatten(), Z.flatten())
        
        # Compute field
        Bx, By, Bz = self.field_solver.compute_magnetic_field(*points_flat)
        B_magnitude = np.sqrt(Bx**2 + By**2 + Bz**2)
        
        # RMS error
        error = np.sqrt(np.mean((B_magnitude - target_field)**2))
        return error


def run_coil_optimizer_demo():
    """Demonstration of coil geometry optimization."""
    print("🎯 Coil Geometry Optimizer Demo")
    print("=" * 50)
    
    # Create field solver
    from .field_solver import ElectromagneticFieldSolver, FieldConfiguration
    
    config = FieldConfiguration(resolution=10)
    solver = ElectromagneticFieldSolver(config)
    
    # Set up optimization
    constraints = OptimizationConstraints(
        max_current=100.0,
        max_power=1000.0,
        field_uniformity=0.1
    )
    
    objectives = OptimizationObjectives(
        target_field_strength=0.01,  # 10 mT
        weight_field=1.0,
        weight_power=0.5
    )
    
    optimizer = CoilGeometryOptimizer(constraints, objectives)
    optimizer.set_field_solver(solver)
    
    # Target region
    target_region = {
        'bounds': {'x': [-0.01, 0.01], 'y': [-0.01, 0.01], 'z': [-0.01, 0.01]},
        'target_field': 0.01
    }
    
    print("🔧 Optimizing 2-coil configuration...")
    result = optimizer.optimize_geometry(n_coils=2, target_region=target_region, 
                                       method="gradient")
    
    print(f"✅ Optimization complete!")
    print(f"   Success: {result['success']}")
    print(f"   Objective value: {result['objective_value']:.6f}")
    print(f"   Evaluations: {result['n_evaluations']}")
    
    print("\n🎯 Coil optimizer demonstration complete!")


if __name__ == "__main__":
    run_coil_optimizer_demo()
