#!/usr/bin/env python3
"""
Discrete Stress-Energy from SU(2) Generating Functionals
Implements Steps 8-9 of the roadmap: discrete stress-energy from generating functionals
Based on su2-3nj-generating-functional and su2-node-matrix-elements repositories
"""

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional, List, Callable, Union
from dataclasses import dataclass
import math
from scipy.special import factorial, comb
from scipy.optimize import minimize
import warnings

@dataclass
class DiscreteNode:
    """Represents a discrete node in the warp bubble mesh."""
    node_id: int
    position: np.ndarray  # 3D coordinates
    j_values: List[float]  # Spin quantum numbers on edges
    magnetic_quantum_nums: List[float]  # Magnetic quantum numbers
    current_density: float  # Local current density
    coupling_strength: float  # Node coupling strength

@dataclass
class DiscreteEdge:
    """Represents an edge connecting two nodes."""
    edge_id: int
    node_a: int  # Source node ID
    node_b: int  # Target node ID
    j_edge: float  # Edge spin quantum number
    x_edge: float  # Edge coupling variable (current/turn density)
    rho_edge: float  # Edge parameter for 3nj calculation

class SU2GeneratingFunctionalCalculator:
    """
    Calculator for SU(2) generating functionals and 3nj symbols.
    Implements the master generating functional approach from the roadmap.
    """
    
    def __init__(self, max_j: float = 10.0, numerical_precision: float = 1e-12):
        """
        Initialize the SU(2) calculator.
        
        Args:
            max_j: Maximum spin quantum number to support
            numerical_precision: Numerical precision for calculations
        """
        self.max_j = max_j
        self.precision = numerical_precision
        
        # Cache for computed values
        self._factorial_cache = {}
        self._3nj_cache = {}
        self._generating_functional_cache = {}
    
    def compute_generating_functional(self, K: np.ndarray) -> complex:
        """
        Compute the master generating functional G({x_e}) = 1/√det(I - K({x_e})).
        
        Args:
            K: NxN antisymmetric matrix of edge coupling variables x_e
            
        Returns:
            Generating functional value G
        """
        # Check if K is antisymmetric
        if not np.allclose(K, -K.T, atol=self.precision):
            warnings.warn("Matrix K should be antisymmetric. Symmetrizing...")
            K = 0.5 * (K - K.T)
        
        # Compute I - K
        I_minus_K = np.eye(K.shape[0]) - K
        
        # Compute determinant
        try:
            det_val = la.det(I_minus_K)
            
            # Handle numerical issues
            if np.abs(det_val) < self.precision:
                warnings.warn("Determinant near zero. System may be unstable.")
                det_val = self.precision
            
            # G = 1/√det(I - K)
            G = 1.0 / np.sqrt(det_val)
            
        except la.LinAlgError:
            warnings.warn("Singular matrix encountered. Returning fallback value.")
            G = complex(1.0)
        
        return G
    
    def build_K_from_currents(self, adjacency_matrix: np.ndarray, 
                            currents: np.ndarray) -> np.ndarray:
        """
        Build antisymmetric kernel K from coil currents/couplings.
        
        Args:
            adjacency_matrix: Network adjacency matrix
            currents: Current values on edges
            
        Returns:
            Antisymmetric coupling matrix K
        """
        n_nodes = adjacency_matrix.shape[0]
        K = np.zeros((n_nodes, n_nodes))
        
        # Fill K based on adjacency and current values
        edge_idx = 0
        for i in range(n_nodes):
            for j in range(i+1, n_nodes):
                if adjacency_matrix[i, j] > 0:  # Edge exists
                    if edge_idx < len(currents):
                        coupling_value = currents[edge_idx]
                        K[i, j] = coupling_value
                        K[j, i] = -coupling_value  # Antisymmetric
                        edge_idx += 1
        
        return K
    
    def calculate_3nj_hypergeometric(self, j_list: List[float], 
                                   rho_list: List[float]) -> float:
        """
        Calculate 3nj symbol using hypergeometric product formula.
        
        From su2-3nj-closedform: {3nj}({j_e}) = ∏_e [1/(2j_e)!] * 2F1(-2j_e, 1/2; 1; -ρ_e)
        
        Args:
            j_list: List of spin quantum numbers on edges
            rho_list: List of ρ parameters for each edge
            
        Returns:
            3nj symbol value
        """
        # Input validation
        if len(j_list) != len(rho_list):
            raise ValueError("j_list and rho_list must have same length")
        
        # Check cache
        cache_key = (tuple(j_list), tuple(rho_list))
        if cache_key in self._3nj_cache:
            return self._3nj_cache[cache_key]
        
        # Compute product formula
        product = 1.0
        
        for j, rho in zip(j_list, rho_list):
            if j < 0 or not np.isclose(j, round(j*2)/2):  # Check if j is valid (non-negative half-integer)
                return 0.0
            
            # Factorial term: 1/(2j)!
            factorial_term = 1.0 / self._cached_factorial(int(2*j))
            
            # Hypergeometric function 2F1(-2j, 1/2; 1; -ρ)
            # For specific case, this can be computed using known formulas
            hypergeom_term = self._hypergeometric_2F1(-2*j, 0.5, 1.0, -rho)
            
            product *= factorial_term * hypergeom_term
        
        # Cache result
        self._3nj_cache[cache_key] = product
        
        return product
    
    def _cached_factorial(self, n: int) -> float:
        """Cached factorial calculation."""
        if n in self._factorial_cache:
            return self._factorial_cache[n]
        
        if n < 0:
            raise ValueError("Factorial undefined for negative numbers")
        
        if n <= 20:  # Use exact calculation for small numbers
            result = float(factorial(n))
        else:  # Use Stirling's approximation for large numbers
            result = np.sqrt(2*np.pi*n) * (n/np.e)**n
        
        self._factorial_cache[n] = result
        return result
    
    def _hypergeometric_2F1(self, a: float, b: float, c: float, z: float) -> float:
        """
        Compute hypergeometric function 2F1(a,b;c;z).
        
        For the specific case 2F1(-2j, 1/2; 1; -ρ), we can use series expansion
        or known closed forms for certain parameter ranges.
        """
        # For |z| < 1, use series expansion: 2F1(a,b;c;z) = Σ (a)_n(b)_n/((c)_n n!) z^n
        # where (x)_n is the Pochhammer symbol
        
        if abs(z) >= 1:
            # Use transformation formulas or approximations
            if z < -1:
                # For z < -1, use transformation 2F1(a,b;c;z) = (1-z)^(-a) 2F1(a,c-b;c;z/(z-1))
                z_trans = z / (z - 1)
                if abs(z_trans) < 1:
                    factor = (1 - z)**(-a)
                    return factor * self._hypergeometric_2F1(a, c - b, c, z_trans)
                else:
                    # Fallback to approximation
                    return 1.0
            else:
                # For |z| ≥ 1, use approximation or special cases
                return 1.0
        
        # Series expansion for |z| < 1
        max_terms = 100
        sum_val = 1.0  # n=0 term
        term = 1.0
        
        for n in range(1, max_terms):
            # Update term: term *= (a+n-1)(b+n-1)z / ((c+n-1)n)
            term *= (a + n - 1) * (b + n - 1) * z / ((c + n - 1) * n)
            sum_val += term
            
            # Check convergence
            if abs(term) < self.precision:
                break
        
        return sum_val
    
    def extract_3nj_from_generating_functional(self, K: np.ndarray, j_list: List[float],
                                             edge_idx_map: List[Tuple[int, int]]) -> float:
        """
        Extract 3nj coefficient from generating functional using differentiation.
        
        {3nj}({j_e}) = (1/∏(2j_e)!) * ∂^(2j_e)G/∂x_e^(2j_e)|_{x=0}
        
        Args:
            K: Base coupling matrix structure
            j_list: Spin quantum numbers for each edge
            edge_idx_map: Mapping from edge index to matrix positions (i,j)
            
        Returns:
            3nj symbol value from generating functional
        """
        # For numerical differentiation, use finite differences
        h = 1e-6  # Small step size
        
        # Total derivative order
        total_order = sum(int(2*j) for j in j_list)
        
        if total_order == 0:
            return float(self.compute_generating_functional(K))
        
        # Use symmetric finite differences for better accuracy
        # This is a simplified implementation - full implementation would use
        # automatic differentiation or symbolic computation
        
        def G_at_x(x_vals):
            """Evaluate G at given edge values."""
            Kx = K.copy()
            for e, x in enumerate(x_vals):
                if e < len(edge_idx_map):
                    i, j = edge_idx_map[e]
                    Kx[i, j] = x
                    Kx[j, i] = -x
            return self.compute_generating_functional(Kx)
        
        # For high-order derivatives, use recursive finite differences
        # This is computationally expensive but works for small systems
        
        n_edges = len(j_list)
        x_base = np.zeros(n_edges)
        
        # Compute high-order derivative using nested finite differences
        if total_order <= 4:  # Only for small orders due to computational cost
            derivative = self._compute_high_order_derivative(G_at_x, x_base, 
                                                           [int(2*j) for j in j_list], h)
        else:
            # For high orders, use approximation or fallback to hypergeometric formula
            edge_params = [0.1] * len(j_list)  # Default ρ parameters
            derivative = self.calculate_3nj_hypergeometric(j_list, edge_params)
        
        # Normalize by factorial factors
        normalization = 1.0
        for j in j_list:
            normalization /= self._cached_factorial(int(2*j))
        
        return float(derivative * normalization)
    
    def _compute_high_order_derivative(self, func: Callable, x0: np.ndarray, 
                                     orders: List[int], h: float) -> float:
        """
        Compute high-order mixed partial derivative using finite differences.
        
        This is a simplified implementation for demonstration.
        Real implementation would use more sophisticated methods.
        """
        if all(order == 0 for order in orders):
            return func(x0)
        
        # Find first non-zero order
        for i, order in enumerate(orders):
            if order > 0:
                # Compute partial derivative with respect to x_i
                orders_minus = orders.copy()
                orders_minus[i] -= 1
                
                x_plus = x0.copy()
                x_plus[i] += h
                x_minus = x0.copy()
                x_minus[i] -= h
                
                derivative = (
                    self._compute_high_order_derivative(func, x_plus, orders_minus, h) -
                    self._compute_high_order_derivative(func, x_minus, orders_minus, h)
                ) / (2 * h)
                
                return derivative
        
        return func(x0)

class DiscreteWarpBubbleSolver:
    """
    Discrete warp bubble solver using quantum geometry and SU(2) generating functionals.
    Implements the complete discrete framework from the roadmap.
    """
    
    def __init__(self, calculator: SU2GeneratingFunctionalCalculator):
        """
        Initialize discrete solver.
        
        Args:
            calculator: SU(2) generating functional calculator
        """
        self.calculator = calculator
        self.nodes = []
        self.edges = []
        self.adjacency_matrix = None
        
    def build_discrete_mesh(self, r_min: float, r_max: float, n_nodes: int,
                          mesh_type: str = "radial") -> Tuple[List[DiscreteNode], List[DiscreteEdge]]:
        """
        Build discrete mesh for warp bubble region.
        
        Args:
            r_min: Minimum radius
            r_max: Maximum radius
            n_nodes: Number of nodes
            mesh_type: Type of mesh ("radial", "cartesian", "spherical")
            
        Returns:
            Tuple of (nodes, edges)
        """
        nodes = []
        edges = []
        
        if mesh_type == "radial":
            # Create radial mesh with nodes along radius
            r_values = np.linspace(r_min, r_max, n_nodes)
            
            for i, r in enumerate(r_values):
                position = np.array([r, 0, 0])  # Along x-axis
                j_values = [0.5, 1.0]  # Default spin values
                magnetic_nums = [0.0, 0.0]  # Default magnetic quantum numbers
                
                node = DiscreteNode(
                    node_id=i,
                    position=position,
                    j_values=j_values,
                    magnetic_quantum_nums=magnetic_nums,
                    current_density=0.0,
                    coupling_strength=1.0
                )
                nodes.append(node)
                
                # Create edges to neighboring nodes
                if i > 0:
                    edge = DiscreteEdge(
                        edge_id=len(edges),
                        node_a=i-1,
                        node_b=i,
                        j_edge=0.5,
                        x_edge=0.1,  # Default coupling
                        rho_edge=0.1  # Default ρ parameter
                    )
                    edges.append(edge)
        
        elif mesh_type == "spherical":
            # Create spherical mesh
            n_radial = int(np.sqrt(n_nodes))
            n_angular = n_nodes // n_radial
            
            node_id = 0
            for i in range(n_radial):
                r = r_min + (r_max - r_min) * i / (n_radial - 1)
                for j in range(n_angular):
                    theta = np.pi * j / (n_angular - 1)
                    phi = 0  # Simplified 2D
                    
                    position = np.array([
                        r * np.sin(theta) * np.cos(phi),
                        r * np.sin(theta) * np.sin(phi),
                        r * np.cos(theta)
                    ])
                    
                    node = DiscreteNode(
                        node_id=node_id,
                        position=position,
                        j_values=[0.5],
                        magnetic_quantum_nums=[0.0],
                        current_density=0.0,
                        coupling_strength=1.0
                    )
                    nodes.append(node)
                    node_id += 1
            
            # Create edges (simplified nearest-neighbor)
            for i, node_a in enumerate(nodes):
                for j, node_b in enumerate(nodes):
                    if i < j:
                        distance = np.linalg.norm(node_a.position - node_b.position)
                        if distance < 1.5 * (r_max - r_min) / n_radial:  # Nearest neighbors
                            edge = DiscreteEdge(
                                edge_id=len(edges),
                                node_a=i,
                                node_b=j,
                                j_edge=0.5,
                                x_edge=0.1,
                                rho_edge=0.1
                            )
                            edges.append(edge)
        
        self.nodes = nodes
        self.edges = edges
        
        # Build adjacency matrix
        self.adjacency_matrix = np.zeros((len(nodes), len(nodes)))
        for edge in edges:
            self.adjacency_matrix[edge.node_a, edge.node_b] = 1
            self.adjacency_matrix[edge.node_b, edge.node_a] = 1
        
        return nodes, edges
    
    def compute_discrete_T00(self, currents: np.ndarray) -> np.ndarray:
        """
        Compute discrete T₀₀ using generating functional approach.
        
        Args:
            currents: Current values on edges
            
        Returns:
            T₀₀ values at each node
        """
        if self.adjacency_matrix is None:
            raise ValueError("Mesh not built. Call build_discrete_mesh() first.")
        
        # Build coupling matrix K from currents
        K = self.calculator.build_K_from_currents(self.adjacency_matrix, currents)
        
        # Compute generating functional
        G = self.calculator.compute_generating_functional(K)
        
        # Extract T₀₀ at each node using matrix element extraction
        T00_nodes = np.zeros(len(self.nodes))
        
        for i, node in enumerate(self.nodes):
            # Use generating functional to compute local stress-energy
            # This is a simplified model - full implementation would involve
            # proper source differentiation and spin network evaluation
            
            # Local contribution from generating functional
            local_K = K[i:i+1, i:i+1] if K.shape[0] > i else np.array([[0.0]])
            local_G = self.calculator.compute_generating_functional(local_K)
            
            # Convert to stress-energy (simplified geometric units)
            geometric_factor = 1.0 / (8 * np.pi)  # c⁴/8πG normalization
            T00_nodes[i] = geometric_factor * np.real(local_G - 1.0)  # Subtract vacuum
        
        return T00_nodes
    
    def discrete_anomaly(self, currents: np.ndarray, G_tt_vals: np.ndarray, 
                        Tm_vals: np.ndarray) -> float:
        """
        Compute discrete anomaly measure |G_tt - 8π(T_m + T_int)|.
        
        Args:
            currents: Current distribution on edges
            G_tt_vals: Einstein tensor G_tt values at nodes
            Tm_vals: Matter stress-energy T_m values at nodes
            
        Returns:
            Total anomaly measure
        """
        # Compute interaction stress-energy from 3nj recoupling
        T_int_vals = np.zeros(len(self.nodes))
        
        for i, node in enumerate(self.nodes):
            # Get local edge parameters
            node_edges = [edge for edge in self.edges if edge.node_a == i or edge.node_b == i]
            
            if len(node_edges) > 0:
                j_vals = [edge.j_edge for edge in node_edges]
                rho_vals = [edge.rho_edge for edge in node_edges]
                
                # Compute 3nj interaction term
                T_int_vals[i] = self.calculator.calculate_3nj_hypergeometric(j_vals, rho_vals)
            else:
                T_int_vals[i] = 0.0
        
        # Compute discrete T₀₀ from currents
        T00_discrete = self.compute_discrete_T00(currents)
        
        # Total anomaly measure
        anomaly = 0.0
        for i in range(len(self.nodes)):
            local_anomaly = abs(G_tt_vals[i] - 8 * np.pi * (Tm_vals[i] + T_int_vals[i] + T00_discrete[i]))
            anomaly += local_anomaly
        
        return anomaly / len(self.nodes)  # Average anomaly
    
    def optimize_discrete_currents(self, target_T00: np.ndarray, G_tt_vals: np.ndarray,
                                 Tm_vals: np.ndarray) -> Dict:
        """
        Optimize current distribution to minimize anomaly and match target T₀₀.
        
        Args:
            target_T00: Target stress-energy profile at nodes
            G_tt_vals: Einstein tensor values
            Tm_vals: Matter stress-energy values
            
        Returns:
            Optimization results
        """
        n_edges = len(self.edges)
        
        def objective(currents):
            try:
                # Compute discrete T₀₀
                T00_computed = self.compute_discrete_T00(currents)
                
                # L2 difference from target
                profile_error = np.sum((T00_computed - target_T00)**2)
                
                # Anomaly penalty
                anomaly = self.discrete_anomaly(currents, G_tt_vals, Tm_vals)
                anomaly_penalty = 1000 * anomaly  # Large penalty for Einstein equation violation
                
                return profile_error + anomaly_penalty
            
            except Exception as e:
                return 1e10  # Penalty for numerical issues
        
        # Initial guess: small random currents
        initial_currents = np.random.normal(0, 0.1, n_edges)
        
        # Bounds: reasonable current values
        bounds = [(-10.0, 10.0) for _ in range(n_edges)]
        
        # Optimize
        result = minimize(objective, initial_currents, bounds=bounds, method='L-BFGS-B')
        
        if result.success:
            optimal_currents = result.x
            optimal_T00 = self.compute_discrete_T00(optimal_currents)
            optimal_anomaly = self.discrete_anomaly(optimal_currents, G_tt_vals, Tm_vals)
            
            return {
                'success': True,
                'optimal_currents': optimal_currents,
                'optimal_T00': optimal_T00,
                'target_error': np.sqrt(np.mean((optimal_T00 - target_T00)**2)),
                'anomaly': optimal_anomaly,
                'objective_value': result.fun
            }
        else:
            return {
                'success': False,
                'message': result.message
            }
    
    def plot_discrete_solution(self, currents: np.ndarray, target_T00: Optional[np.ndarray] = None,
                             save_path: Optional[str] = None) -> plt.Figure:
        """Plot discrete solution results."""
        T00_computed = self.compute_discrete_T00(currents)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Node positions
        positions = np.array([node.position for node in self.nodes])
        r_values = np.linalg.norm(positions, axis=1)
        
        # Plot T₀₀ profile
        ax1.scatter(r_values, T00_computed, c='blue', s=50, label='Computed $T_{00}$')
        if target_T00 is not None:
            ax1.scatter(r_values, target_T00, c='red', s=30, alpha=0.7, label='Target $T_{00}$')
        ax1.set_xlabel('Radial Distance')
        ax1.set_ylabel('$T_{00}$')
        ax1.set_title('Discrete Stress-Energy Profile')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot current distribution
        edge_positions = []
        for edge in self.edges:
            pos_a = self.nodes[edge.node_a].position
            pos_b = self.nodes[edge.node_b].position
            edge_pos = 0.5 * (pos_a + pos_b)
            edge_positions.append(np.linalg.norm(edge_pos))
        
        ax2.scatter(edge_positions, currents, c='green', s=50)
        ax2.set_xlabel('Edge Position (radius)')
        ax2.set_ylabel('Current')
        ax2.set_title('Current Distribution on Edges')
        ax2.grid(True, alpha=0.3)
        
        # Plot mesh structure
        ax3.scatter(positions[:, 0], positions[:, 1], c='blue', s=30, label='Nodes')
        for edge in self.edges:
            pos_a = self.nodes[edge.node_a].position
            pos_b = self.nodes[edge.node_b].position
            ax3.plot([pos_a[0], pos_b[0]], [pos_a[1], pos_b[1]], 'k-', alpha=0.3, linewidth=1)
        ax3.set_xlabel('X coordinate')
        ax3.set_ylabel('Y coordinate')
        ax3.set_title('Discrete Mesh Structure')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_aspect('equal')
        
        # Plot generating functional contribution
        G_values = []
        for i in range(len(self.nodes)):
            # Local generating functional value (simplified)
            local_K = np.array([[currents[i] if i < len(currents) else 0.0]])
            G_local = self.calculator.compute_generating_functional(local_K)
            G_values.append(np.real(G_local))
        
        ax4.scatter(r_values, G_values, c='purple', s=50)
        ax4.set_xlabel('Radial Distance')
        ax4.set_ylabel('Generating Functional G')
        ax4.set_title('Local Generating Functional Values')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig

class DiscreteQuantumGeometry:
    """
    Unified interface for discrete quantum geometry calculations.
    Combines SU(2) generating functional calculator with discrete mesh solver.
    """
    
    def __init__(self, n_nodes: int = 20, max_j: float = 5.0):
        """
        Initialize discrete quantum geometry system.
        
        Args:
            n_nodes: Number of discrete nodes in the mesh
            max_j: Maximum spin quantum number
        """
        self.n_nodes = n_nodes
        self.max_j = max_j
        
        # Initialize SU(2) calculator
        self.su2_calculator = SU2GeneratingFunctionalCalculator(max_j=max_j)
        
        # Initialize discrete solver
        self.discrete_solver = DiscreteWarpBubbleSolver(self.su2_calculator)
        
        # Generate adjacency matrix for mesh connectivity
        self.adjacency_matrix = self._generate_adjacency_matrix()
    
    def _generate_adjacency_matrix(self) -> np.ndarray:
        """Generate adjacency matrix for node connectivity."""
        # Simple nearest-neighbor connectivity
        adj = np.zeros((self.n_nodes, self.n_nodes))
        
        for i in range(self.n_nodes):
            # Connect to next neighbor (circular)
            next_node = (i + 1) % self.n_nodes
            adj[i, next_node] = 1.0
            adj[next_node, i] = 1.0
            
            # Connect to second nearest neighbor for better connectivity
            if self.n_nodes > 4:
                second_next = (i + 2) % self.n_nodes
                adj[i, second_next] = 0.5
                adj[second_next, i] = 0.5
        
        return adj
    
    def compute_generating_functional(self, K_matrix: Optional[np.ndarray] = None) -> float:
        """
        Compute SU(2) generating functional G(K).
        
        Args:
            K_matrix: Interaction matrix (uses default if None)
            
        Returns:
            Generating functional value G
        """
        if K_matrix is None:
            # Default K matrix based on adjacency
            K_matrix = 0.1 * self.adjacency_matrix
        
        return self.su2_calculator.compute_generating_functional(K_matrix)
    
    def compute_quantum_corrected_stress_energy(self, currents: np.ndarray) -> np.ndarray:
        """
        Compute quantum-corrected stress-energy tensor.
        
        Args:
            currents: Current distribution array
            
        Returns:
            Quantum-corrected stress-energy values
        """
        # Build K-matrix from currents
        K_matrix = self._build_K_from_currents(currents)
        
        # Compute generating functional
        G = self.compute_generating_functional(K_matrix)
        
        # Apply quantum correction factor
        quantum_correction = 1.0 / G if G > 1e-12 else 1.0
        
        # Classical stress-energy (placeholder - would compute from currents)
        classical_T00 = self._compute_classical_stress_energy(currents)
        
        # Apply quantum correction
        corrected_T00 = classical_T00 * quantum_correction
        
        return corrected_T00
    
    def _build_K_from_currents(self, currents: np.ndarray) -> np.ndarray:
        """Build interaction matrix from current distribution."""
        # Ensure currents match node count
        if len(currents) != self.n_nodes:
            # Interpolate to match
            currents_interp = np.interp(
                np.linspace(0, 1, self.n_nodes),
                np.linspace(0, 1, len(currents)),
                currents
            )
        else:
            currents_interp = currents
        
        # Build K-matrix
        K = np.zeros((self.n_nodes, self.n_nodes))
        
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if self.adjacency_matrix[i, j] > 0:
                    # Weight by current strength
                    current_factor = 0.5 * (currents_interp[i] + currents_interp[j])
                    K[i, j] = 0.1 * current_factor * self.adjacency_matrix[i, j]
        
        return K
    
    def _compute_classical_stress_energy(self, currents: np.ndarray) -> np.ndarray:
        """Compute classical stress-energy from currents (simplified)."""
        # Simplified classical computation
        # In practice, this would use Maxwell stress-energy tensor
        current_scale = np.max(np.abs(currents)) if len(currents) > 0 else 1.0
        
        # Mock stress-energy proportional to current squared
        if current_scale > 1e-12:
            normalized_currents = currents / current_scale
        else:
            normalized_currents = currents
        
        T00_classical = -0.1 * normalized_currents**2  # Negative for exotic matter
        
        return T00_classical
    
    def analyze_mesh_convergence(self, node_counts: List[int] = [50, 100, 200, 400]) -> Dict:
        """
        Analyze quantum geometry convergence with increasing mesh resolution.
        
        Monitor max|1/G - 1| until < 10⁻⁸ for discretization error control.
        
        Args:
            node_counts: List of node counts to test
            
        Returns:
            Convergence analysis with anomaly tracking
        """
        print("🔬 Analyzing quantum geometry mesh convergence...")
        
        convergence_data = {
            'node_counts': node_counts,
            'anomalies': [],
            'generating_functionals': [],
            'mesh_qualities': [],
            'computation_times': [],
            'memory_usage': []
        }
        
        for N in node_counts:
            import time
            import psutil
            
            start_time = time.time()
            process = psutil.Process()
            start_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            try:
                # Generate mesh with N nodes
                nodes, edges = self.discrete_solver.build_discrete_mesh(
                    r_min=0.1, r_max=10.0, n_nodes=N, mesh_type="radial"
                )
                
                # Create adjacency matrix
                adjacency = self.discrete_solver._build_adjacency_matrix(nodes, edges)
                
                # Generate test current configuration
                test_currents = np.random.normal(0, 0.1, len(edges))
                
                # Build K-matrix from currents
                K_matrix = self.su2_calculator.build_K_from_currents(
                    adjacency, test_currents
                )
                
                # Compute generating functional
                G = self.su2_calculator.compute_generating_functional(K_matrix)
                
                # Calculate anomaly
                anomaly = abs(1.0 / G - 1.0)
                
                # Mesh quality metrics
                mesh_quality = self._assess_mesh_quality(nodes, edges)
                
                computation_time = time.time() - start_time
                end_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_usage = end_memory - start_memory
                
                # Store results
                convergence_data['anomalies'].append(anomaly)
                convergence_data['generating_functionals'].append(G)
                convergence_data['mesh_qualities'].append(mesh_quality)
                convergence_data['computation_times'].append(computation_time)
                convergence_data['memory_usage'].append(memory_usage)
                
                print(f"  N={N:3d}: anomaly={anomaly:.2e}, G={G:.6f}, "
                      f"time={computation_time:.3f}s, mem={memory_usage:.1f}MB")
                
                # Check convergence criterion
                if anomaly < 1e-8:
                    print(f"  ✓ Convergence achieved at N={N}")
                
            except Exception as e:
                print(f"  ❌ Failed at N={N}: {e}")
                convergence_data['anomalies'].append(np.inf)
                convergence_data['generating_functionals'].append(np.nan)
                convergence_data['mesh_qualities'].append(0.0)
                convergence_data['computation_times'].append(np.inf)
                convergence_data['memory_usage'].append(np.inf)
        
        # Analysis
        optimal_N = self._find_optimal_mesh_size(convergence_data)
        convergence_achieved = min(convergence_data['anomalies']) < 1e-8
        
        convergence_data['optimal_node_count'] = optimal_N
        convergence_data['convergence_achieved'] = convergence_achieved
        convergence_data['min_anomaly'] = min(convergence_data['anomalies'])
        
        print(f"✓ Optimal mesh size: N={optimal_N}")
        print(f"✓ Min anomaly: {min(convergence_data['anomalies']):.2e}")
        print(f"✓ Convergence: {convergence_achieved}")
        
        return convergence_data
    
    def _assess_mesh_quality(self, nodes: List, edges: List) -> float:
        """Assess mesh quality metrics."""
        if len(nodes) < 3:
            return 0.0
        
        # Calculate node spacing uniformity
        positions = np.array([node.position for node in nodes])
        r_coords = np.linalg.norm(positions, axis=1)
        
        # Spacing uniformity (smaller is better)
        spacing = np.diff(np.sort(r_coords))
        spacing_uniformity = np.std(spacing) / np.mean(spacing) if len(spacing) > 0 else 1.0
        
        # Connectivity ratio
        max_possible_edges = len(nodes) * (len(nodes) - 1) // 2
        connectivity_ratio = len(edges) / max_possible_edges if max_possible_edges > 0 else 0.0
        
        # Combined quality score (higher is better)
        quality = connectivity_ratio / (1 + spacing_uniformity)
        
        return quality
    
    def _find_optimal_mesh_size(self, convergence_data: Dict) -> int:
        """Find optimal mesh size balancing accuracy and efficiency."""
        node_counts = np.array(convergence_data['node_counts'])
        anomalies = np.array(convergence_data['anomalies'])
        times = np.array(convergence_data['computation_times'])
        
        # Remove failed cases
        valid_mask = np.isfinite(anomalies) & np.isfinite(times)
        if not np.any(valid_mask):
            return node_counts[0]  # Fallback
        
        valid_nodes = node_counts[valid_mask]
        valid_anomalies = anomalies[valid_mask]
        valid_times = times[valid_mask]
        
        # Find first point where anomaly < 1e-6 (relaxed criterion)
        converged_mask = valid_anomalies < 1e-6
        if np.any(converged_mask):
            converged_indices = np.where(converged_mask)[0]
            # Among converged solutions, pick the one with best time/accuracy trade-off
            scores = 1.0 / (valid_anomalies[converged_indices] * valid_times[converged_indices])
            best_idx = converged_indices[np.argmax(scores)]
            return valid_nodes[best_idx]
        else:
            # If no convergence, pick best anomaly
            best_idx = np.argmin(valid_anomalies)
            return valid_nodes[best_idx]
    
    def set_optimal_mesh_resolution(self, convergence_data: Dict) -> None:
        """Set the optimal mesh resolution based on convergence analysis."""
        optimal_N = convergence_data['optimal_node_count']
        self.optimal_mesh_nodes = optimal_N
        
        print(f"✓ Set optimal mesh resolution: {optimal_N} nodes")
        print(f"  Anomaly at optimal resolution: "
              f"{convergence_data['anomalies'][convergence_data['node_counts'].index(optimal_N)]:.2e}")
    
    def generate_optimal_mesh(self, r_min: float = 0.1, r_max: float = 10.0) -> Tuple:
        """Generate mesh using optimal resolution determined from convergence analysis."""
        if hasattr(self, 'optimal_mesh_nodes'):
            n_nodes = self.optimal_mesh_nodes
        else:
            print("⚠️ Optimal mesh resolution not set, using default")
            n_nodes = 100  # Reasonable default
        
        return self.build_discrete_mesh(r_min, r_max, n_nodes, "radial")

if __name__ == "__main__":
    # Example usage
    
    # Initialize SU(2) calculator
    calculator = SU2GeneratingFunctionalCalculator()
    
    # Test 3nj calculation
    j_list = [0.5, 1.0, 0.5]
    rho_list = [0.1, 0.2, 0.1]
    threej_val = calculator.calculate_3nj_hypergeometric(j_list, rho_list)
    print(f"3nj symbol value: {threej_val:.6e}")
    
    # Initialize discrete solver
    solver = DiscreteWarpBubbleSolver(calculator)
    
    # Build discrete mesh
    nodes, edges = solver.build_discrete_mesh(r_min=0.5, r_max=3.0, n_nodes=20, mesh_type="radial")
    print(f"Built mesh with {len(nodes)} nodes and {len(edges)} edges")
    
    # Create target profile (negative energy shell)
    positions = np.array([node.position for node in nodes])
    r_values = np.linalg.norm(positions, axis=1)
    target_T00 = -0.1 * np.exp(-((r_values - 2.0)/0.5)**2)
    
    # Mock Einstein tensor and matter values
    G_tt_vals = 0.1 * np.sin(np.pi * r_values / 3.0)  # Oscillating background
    Tm_vals = 0.05 * np.ones_like(r_values)  # Constant matter density
    
    # Optimize currents
    print("Optimizing discrete current distribution...")
    opt_result = solver.optimize_discrete_currents(target_T00, G_tt_vals, Tm_vals)
    
    if opt_result['success']:
        print(f"Optimization successful!")
        print(f"Target error (RMSE): {opt_result['target_error']:.6f}")
        print(f"Final anomaly: {opt_result['anomaly']:.6e}")
        
        # Plot results
        fig = solver.plot_discrete_solution(opt_result['optimal_currents'], target_T00,
                                          save_path="discrete_warp_solution.png")
        plt.show()
    else:
        print(f"Optimization failed: {opt_result['message']}")
    
    # Test generating functional directly
    print("\nTesting generating functional computation...")
    test_currents = np.random.normal(0, 0.1, len(edges))
    K_test = calculator.build_K_from_currents(solver.adjacency_matrix, test_currents)
    G_test = calculator.compute_generating_functional(K_test)
    print(f"Generating functional G = {G_test}")
    
    # Test discrete T₀₀ computation
    T00_test = solver.compute_discrete_T00(test_currents)
    print(f"Discrete T₀₀ range: [{T00_test.min():.6f}, {T00_test.max():.6f}]")
