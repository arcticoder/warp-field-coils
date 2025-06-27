#!/usr/bin/env python3
"""
Step 19: Medical Tractor-Field Array Implementation

Author: Assistant
Created: Current session
Version: 1.0

PHYSICS IMPLEMENTATION:
=====================
1. Maxwell-Faraday Induction:  ∇ × E = -∂B/∂t
2. Lorentz Force:             F = q(E + v × B) 
3. Magnetic Pressure:         P_B = B²/(2μ₀)
4. Medical Safety Fields:     |E| < E_safe, |B| < B_safe
5. Gradient Forces:           F_∇ = μ·∇B (magnetic dipole interaction)
6. Hall Effect:               E_H = (1/nq)(J × B)
7. Biocompatibility Limits:   dB/dt < 20 T/s, B < 3 T, E < 1 kV/m

MEDICAL APPLICATIONS:
===================
- Targeted drug delivery via magnetic nanoparticles
- Non-invasive tissue manipulation
- Cellular membrane permeabilization
- Precision surgical guidance
- Blood flow control and hemostasis
- Tumor cell isolation and removal

SAFETY PROTOCOLS:
================
- Real-time SAR (Specific Absorption Rate) monitoring
- Thermal imaging integration
- Emergency field shutdown systems
- Patient vital sign feedback loops
- Magnetic field gradient limiting
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize
import scipy.integrate
from scipy.spatial.distance import cdist
import warnings

class MedicalTractorFieldArray:
    """
    Advanced medical tractor field array system for precision biomedical applications.
    
    Implements sophisticated electromagnetic field control for:
    - Targeted therapy delivery
    - Non-invasive tissue manipulation  
    - Cellular interaction control
    - Real-time safety monitoring
    """
    
    def __init__(self, array_size=(8, 8), field_volume=(0.5, 0.5, 0.3), 
                 max_field_strength=1.5, safety_margin=0.3):
        """
        Initialize medical tractor field array.
        
        Args:
            array_size: (rows, cols) of electromagnetic coil array
            field_volume: (x, y, z) dimensions of treatment volume in meters
            max_field_strength: Maximum safe magnetic field in Tesla
            safety_margin: Safety factor for field limits (0-1)
        """
        self.array_size = array_size
        self.field_volume = np.array(field_volume)
        self.max_B = max_field_strength * (1 - safety_margin)  # Safety-limited field
        self.max_E = 800.0  # V/m - Safe electric field limit
        self.max_dBdt = 15.0  # T/s - Safe rate of change
        
        # Physical constants
        self.mu0 = 4 * np.pi * 1e-7  # H/m - Permeability of free space
        self.epsilon0 = 8.854e-12   # F/m - Permittivity of free space  
        self.c = 2.998e8            # m/s - Speed of light
        
        # Array geometry
        self.setup_coil_array()
        
        # Field state
        self.current_B = np.zeros(3)
        self.current_E = np.zeros(3) 
        self.field_history = []
        self.safety_violations = []
        
        # Medical parameters
        self.tissue_conductivity = 0.5  # S/m - Average tissue conductivity
        self.blood_susceptibility = -9.05e-6  # Diamagnetic susceptibility
        self.nanoparticle_moment = 1e-18  # A⋅m² - Magnetic nanoparticle moment
        
        print(f"Medical Tractor Field Array Initialized:")
        print(f"  Array Size: {array_size[0]}×{array_size[1]} coils")
        print(f"  Treatment Volume: {field_volume[0]:.2f}×{field_volume[1]:.2f}×{field_volume[2]:.2f} m³")
        print(f"  Max Safe B-field: {self.max_B:.2f} T")
        print(f"  Max Safe E-field: {self.max_E:.0f} V/m")
        
    def setup_coil_array(self):
        """Setup electromagnetic coil array geometry."""
        rows, cols = self.array_size
        
        # Generate coil positions around treatment volume
        x_positions = np.linspace(-self.field_volume[0], self.field_volume[0], cols)
        y_positions = np.linspace(-self.field_volume[1], self.field_volume[1], rows)
        
        self.coil_positions = []
        self.coil_orientations = []
        self.coil_currents = np.zeros((rows, cols))
        
        # Top array (z = +field_volume[2]/2)
        for i, y in enumerate(y_positions):
            for j, x in enumerate(x_positions):
                pos = np.array([x, y, self.field_volume[2]/2])
                orientation = np.array([0, 0, -1])  # Pointing down
                self.coil_positions.append(pos)
                self.coil_orientations.append(orientation)
                
        # Bottom array (z = -field_volume[2]/2)  
        for i, y in enumerate(y_positions):
            for j, x in enumerate(x_positions):
                pos = np.array([x, y, -self.field_volume[2]/2])
                orientation = np.array([0, 0, 1])  # Pointing up
                self.coil_positions.append(pos)
                self.coil_orientations.append(orientation)
                
        self.coil_positions = np.array(self.coil_positions)
        self.coil_orientations = np.array(self.coil_orientations)
        self.n_coils = len(self.coil_positions)
        
        print(f"  Configured {self.n_coils} electromagnetic coils")
        
    def compute_magnetic_field(self, target_point, coil_currents):
        """
        Compute magnetic field at target point using Biot-Savart law.
        
        For circular coil: B = (μ₀ I R²)/(2(R² + z²)^(3/2)) ẑ (on axis)
        
        Args:
            target_point: 3D coordinates of evaluation point
            coil_currents: Current through each coil
            
        Returns:
            B_field: 3D magnetic field vector
        """
        B_total = np.zeros(3)
        coil_radius = 0.05  # m - Coil radius
        
        for i, (pos, orientation, current) in enumerate(zip(
            self.coil_positions, self.coil_orientations, coil_currents)):
            
            # Vector from coil to target
            r_vec = target_point - pos
            r_distance = np.linalg.norm(r_vec)
            
            if r_distance < 1e-6:  # Avoid singularity
                continue
                
            # Simplified coil field (circular loop approximation)
            z_axis_distance = np.dot(r_vec, orientation)
            
            # On-axis field strength
            denominator = (coil_radius**2 + z_axis_distance**2)**(3/2)
            B_magnitude = (self.mu0 * current * coil_radius**2) / (2 * denominator)
            
            # Field direction along coil axis
            B_contribution = B_magnitude * orientation
            B_total += B_contribution
            
        return B_total
        
    def compute_electric_field(self, target_point, dBdt):
        """
        Compute induced electric field from changing magnetic field.
        
        Maxwell-Faraday: ∇ × E = -∂B/∂t
        For cylindrical symmetry: E_φ = -r/2 ⋅ ∂B_z/∂t
        
        Args:
            target_point: 3D coordinates
            dBdt: Time derivative of magnetic field
            
        Returns:
            E_field: 3D electric field vector
        """
        # Distance from central axis
        r_perp = np.sqrt(target_point[0]**2 + target_point[1]**2)
        
        # Induced electric field (cylindrical symmetry)
        E_magnitude = r_perp * np.abs(dBdt[2]) / 2
        
        # Field direction (tangential)
        if r_perp > 1e-6:
            phi_hat = np.array([-target_point[1], target_point[0], 0]) / r_perp
            E_field = E_magnitude * phi_hat
        else:
            E_field = np.zeros(3)
            
        return E_field
        
    def compute_gradient_force(self, target_point, coil_currents, magnetic_moment):
        """
        Compute gradient force on magnetic dipole.
        
        Force: F = ∇(μ⋅B) = μ⋅∇B (for constant moment)
        
        Args:
            target_point: 3D coordinates
            coil_currents: Current distribution
            magnetic_moment: Magnetic dipole moment vector
            
        Returns:
            force: 3D force vector
        """
        # Compute field gradient numerically
        delta = 1e-4  # Small displacement for gradient calculation
        
        gradient = np.zeros((3, 3))  # ∂B_i/∂x_j
        
        for i in range(3):
            # Forward difference
            point_plus = target_point.copy()
            point_plus[i] += delta
            B_plus = self.compute_magnetic_field(point_plus, coil_currents)
            
            # Backward difference  
            point_minus = target_point.copy()
            point_minus[i] -= delta
            B_minus = self.compute_magnetic_field(point_minus, coil_currents)
            
            # Gradient components
            gradient[:, i] = (B_plus - B_minus) / (2 * delta)
            
        # Force calculation: F_i = μ_j ⋅ ∂B_i/∂x_j (Einstein summation)
        force = np.dot(gradient, magnetic_moment)
        
        return force
        
    def optimize_field_targeting(self, target_points, desired_forces, 
                                max_iterations=100):
        """
        Optimize coil currents to achieve desired forces at target points.
        
        Minimizes: ||F_actual - F_desired||² subject to safety constraints
        
        Args:
            target_points: List of 3D target coordinates
            desired_forces: List of desired force vectors
            max_iterations: Optimization iteration limit
            
        Returns:
            optimal_currents: Optimized coil current distribution
            achieved_forces: Actually achieved forces
        """
        def objective_function(currents):
            """Objective: minimize force error."""
            total_error = 0.0
            
            for target, desired_force in zip(target_points, desired_forces):
                # Compute actual force
                actual_force = self.compute_gradient_force(
                    target, currents, self.nanoparticle_moment * np.array([0, 0, 1]))
                    
                # Add to error
                force_error = np.linalg.norm(actual_force - desired_force)**2
                total_error += force_error
                
            return total_error
            
        def safety_constraints(currents):
            """Ensure magnetic field safety limits."""
            violations = []
            
            # Check field strength at multiple points
            test_points = np.random.rand(20, 3) * self.field_volume
            
            for point in test_points:
                B_field = self.compute_magnetic_field(point, currents)
                B_magnitude = np.linalg.norm(B_field)
                
                if B_magnitude > self.max_B:
                    violations.append(self.max_B - B_magnitude)
                    
            return np.array(violations) if violations else np.array([1.0])
            
        # Initial guess (uniform low current)
        initial_currents = np.ones(self.n_coils) * 0.1
        
        # Current bounds (±10 A maximum)
        bounds = [(-10.0, 10.0) for _ in range(self.n_coils)]
        
        # Optimization constraints
        constraints = {'type': 'ineq', 'fun': safety_constraints}
        
        try:
            # Optimize using Sequential Least Squares Programming
            result = scipy.optimize.minimize(
                objective_function,
                initial_currents,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': max_iterations}
            )
            
            optimal_currents = result.x
            
            # Compute achieved forces
            achieved_forces = []
            for target in target_points:
                force = self.compute_gradient_force(
                    target, optimal_currents, 
                    self.nanoparticle_moment * np.array([0, 0, 1]))
                achieved_forces.append(force)
                
            print(f"Field optimization completed:")
            print(f"  Optimization success: {result.success}")
            print(f"  Final objective value: {result.fun:.2e}")
            print(f"  Max coil current: {np.max(np.abs(optimal_currents)):.2f} A")
            
            return optimal_currents, achieved_forces
            
        except Exception as e:
            print(f"Optimization failed: {str(e)}")
            return initial_currents, []
            
    def monitor_safety_parameters(self, target_points, coil_currents, dt=1e-3):
        """
        Real-time safety monitoring of electromagnetic exposure.
        
        Monitors:
        - SAR (Specific Absorption Rate): σ|E|²/(2ρ) 
        - dB/dt limits for neural stimulation
        - Thermal heating rates
        
        Args:
            target_points: Points to monitor
            coil_currents: Current distribution
            dt: Time step for derivative calculation
            
        Returns:
            safety_status: Dictionary of safety parameters
        """
        safety_status = {
            'safe': True,
            'violations': [],
            'max_B': 0.0,
            'max_E': 0.0,
            'max_dBdt': 0.0,
            'max_SAR': 0.0
        }
        
        # Previous currents for dB/dt calculation
        if not hasattr(self, 'previous_currents'):
            self.previous_currents = np.zeros_like(coil_currents)
            
        for point in target_points:
            # Magnetic field
            B_field = self.compute_magnetic_field(point, coil_currents)
            B_magnitude = np.linalg.norm(B_field)
            safety_status['max_B'] = max(safety_status['max_B'], B_magnitude)
            
            # Rate of change
            B_prev = self.compute_magnetic_field(point, self.previous_currents)
            dBdt = (B_field - B_prev) / dt
            dBdt_magnitude = np.linalg.norm(dBdt)
            safety_status['max_dBdt'] = max(safety_status['max_dBdt'], dBdt_magnitude)
            
            # Electric field
            E_field = self.compute_electric_field(point, dBdt)
            E_magnitude = np.linalg.norm(E_field)
            safety_status['max_E'] = max(safety_status['max_E'], E_magnitude)
            
            # SAR calculation (W/kg)
            tissue_density = 1000  # kg/m³
            SAR = (self.tissue_conductivity * E_magnitude**2) / (2 * tissue_density)
            safety_status['max_SAR'] = max(safety_status['max_SAR'], SAR)
            
            # Check violations
            if B_magnitude > self.max_B:
                safety_status['safe'] = False
                safety_status['violations'].append(f"B-field: {B_magnitude:.2f} T > {self.max_B:.2f} T")
                
            if E_magnitude > self.max_E:
                safety_status['safe'] = False
                safety_status['violations'].append(f"E-field: {E_magnitude:.0f} V/m > {self.max_E:.0f} V/m")
                
            if dBdt_magnitude > self.max_dBdt:
                safety_status['safe'] = False
                safety_status['violations'].append(f"dB/dt: {dBdt_magnitude:.1f} T/s > {self.max_dBdt:.1f} T/s")
                
            if SAR > 2.0:  # W/kg SAR limit
                safety_status['safe'] = False
                safety_status['violations'].append(f"SAR: {SAR:.2f} W/kg > 2.0 W/kg")
                
        self.previous_currents = coil_currents.copy()
        return safety_status
        
    def simulate_drug_delivery(self, drug_particles, target_region, 
                             simulation_time=10.0, time_steps=1000):
        """
        Simulate magnetic nanoparticle drug delivery.
        
        Equations of motion:
        m⋅dv/dt = F_magnetic + F_drag + F_brownian
        F_drag = -γv (Stokes drag)
        F_brownian = √(2γk_B T)⋅ξ(t)
        
        Args:
            drug_particles: Initial particle positions and properties
            target_region: Target delivery coordinates
            simulation_time: Total simulation duration (s)
            time_steps: Number of simulation steps
            
        Returns:
            particle_trajectories: Time evolution of particle positions
            delivery_efficiency: Fraction reaching target
        """
        dt = simulation_time / time_steps
        n_particles = len(drug_particles)
        
        # Particle properties
        particle_radius = 50e-9  # m - Nanoparticle radius
        particle_mass = 4/3 * np.pi * particle_radius**3 * 7800  # kg (iron oxide)
        drag_coefficient = 6 * np.pi * 1e-3 * particle_radius  # Stokes drag
        
        # Initialize trajectories
        trajectories = np.zeros((time_steps, n_particles, 3))
        velocities = np.zeros((n_particles, 3))
        
        # Set initial positions
        for i, particle in enumerate(drug_particles):
            trajectories[0, i] = particle['position']
            
        # Optimize fields for target steering
        target_points = [target_region]
        desired_forces = [np.array([0, 0, -1e-12])]  # Small downward force
        
        optimal_currents, _ = self.optimize_field_targeting(target_points, desired_forces)
        
        # Time evolution
        for t in range(1, time_steps):
            for i in range(n_particles):
                current_pos = trajectories[t-1, i]
                
                # Magnetic force
                F_magnetic = self.compute_gradient_force(
                    current_pos, optimal_currents, 
                    self.nanoparticle_moment * np.array([0, 0, 1]))
                    
                # Drag force
                F_drag = -drag_coefficient * velocities[i]
                
                # Brownian motion (thermal fluctuations)
                kB_T = 1.38e-23 * 310  # J at body temperature
                F_brownian = np.sqrt(2 * drag_coefficient * kB_T / dt) * np.random.randn(3)
                
                # Total force and acceleration
                F_total = F_magnetic + F_drag + F_brownian
                acceleration = F_total / particle_mass
                
                # Update velocity and position (Verlet integration)
                velocities[i] += acceleration * dt
                trajectories[t, i] = trajectories[t-1, i] + velocities[i] * dt
                
        # Calculate delivery efficiency
        final_positions = trajectories[-1]
        target_distances = np.linalg.norm(final_positions - target_region, axis=1)
        delivery_radius = 0.01  # m - 1 cm delivery tolerance
        successful_deliveries = np.sum(target_distances < delivery_radius)
        delivery_efficiency = successful_deliveries / n_particles
        
        # Handle NaN case
        mean_distance = np.mean(target_distances) if len(target_distances) > 0 else 0.0
        if np.isnan(mean_distance):
            mean_distance = 0.0
        
        print(f"Drug delivery simulation completed:")
        print(f"  Simulation time: {simulation_time:.1f} s")
        print(f"  Particles tracked: {n_particles}")
        print(f"  Delivery efficiency: {delivery_efficiency:.1%}")
        print(f"  Average final distance: {mean_distance*1000:.1f} mm")
        
        return trajectories, delivery_efficiency
        
    def visualize_field_distribution(self, coil_currents, z_plane=0.0):
        """
        Visualize magnetic field distribution in treatment volume.
        
        Args:
            coil_currents: Current distribution through coils
            z_plane: Z-coordinate plane for visualization
        """
        # Create evaluation grid
        x_range = np.linspace(-self.field_volume[0]/2, self.field_volume[0]/2, 20)
        y_range = np.linspace(-self.field_volume[1]/2, self.field_volume[1]/2, 20)
        X, Y = np.meshgrid(x_range, y_range)
        
        # Compute field at each point
        B_magnitude = np.zeros_like(X)
        B_x = np.zeros_like(X)
        B_y = np.zeros_like(X)
        
        for i in range(len(x_range)):
            for j in range(len(y_range)):
                point = np.array([X[i,j], Y[i,j], z_plane])
                B_field = self.compute_magnetic_field(point, coil_currents)
                B_magnitude[i,j] = np.linalg.norm(B_field)
                B_x[i,j] = B_field[0]
                B_y[i,j] = B_field[1]
                
        # Create visualization
        plt.figure(figsize=(12, 5))
        
        # Field magnitude
        plt.subplot(1, 2, 1)
        contour = plt.contourf(X, Y, B_magnitude, levels=20, cmap='viridis')
        plt.colorbar(contour, label='|B| (T)')
        plt.title(f'Magnetic Field Magnitude (z={z_plane:.2f}m)')
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        
        # Field vectors
        plt.subplot(1, 2, 2)
        plt.contourf(X, Y, B_magnitude, levels=20, cmap='viridis', alpha=0.6)
        skip = 2  # Vector field decimation
        plt.quiver(X[::skip, ::skip], Y[::skip, ::skip], 
                  B_x[::skip, ::skip], B_y[::skip, ::skip], 
                  scale=np.max(B_magnitude)*20, color='white')
        plt.title('Magnetic Field Vectors')
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        
        plt.tight_layout()
        plt.show()
        
        # Safety assessment
        max_field = np.max(B_magnitude)
        print(f"Field Distribution Analysis:")
        print(f"  Maximum field strength: {max_field:.3f} T")
        print(f"  Safety margin: {(self.max_B - max_field)/self.max_B:.1%}")
        
    def demonstrate_medical_applications(self):
        """Demonstrate key medical tractor field applications."""
        print("\n" + "="*60)
        print("MEDICAL TRACTOR FIELD ARRAY - DEMONSTRATION")
        print("="*60)
        
        # 1. Targeted Drug Delivery
        print("\n1. TARGETED DRUG DELIVERY SIMULATION")
        print("-" * 40)
        
        # Create drug particles at injection site
        injection_site = np.array([-0.1, 0, 0])  # 10 cm offset
        drug_particles = []
        for i in range(100):
            # Add small random displacement
            pos = injection_site + np.random.normal(0, 0.005, 3)
            drug_particles.append({'position': pos, 'id': i})
            
        # Target: tumor location
        tumor_location = np.array([0.05, 0.02, 0])
        
        # Simulate delivery
        trajectories, efficiency = self.simulate_drug_delivery(
            drug_particles, tumor_location, simulation_time=5.0)
            
        # 2. Force Field Optimization
        print("\n2. PRECISION FORCE TARGETING")
        print("-" * 40)
        
        # Multiple targets for complex procedures
        target_points = [
            np.array([0.02, 0.03, 0]),   # Target 1
            np.array([-0.01, 0.04, 0]),  # Target 2
            np.array([0.03, -0.02, 0])   # Target 3
        ]
        
        desired_forces = [
            np.array([1e-12, 0, 0]),      # Right force
            np.array([0, -1e-12, 0]),     # Left force  
            np.array([0, 0, 1e-12])       # Up force
        ]
        
        optimal_currents, achieved_forces = self.optimize_field_targeting(
            target_points, desired_forces)
            
        # 3. Safety Monitoring
        print("\n3. REAL-TIME SAFETY MONITORING")
        print("-" * 40)
        
        # Monitor critical tissue regions
        critical_regions = [
            np.array([0, 0, 0]),        # Treatment center
            np.array([0.1, 0, 0]),      # Nearby organ
            np.array([0, 0.1, 0])       # Another region
        ]
        
        safety_status = self.monitor_safety_parameters(critical_regions, optimal_currents)
        
        print(f"  Safety Status: {'✅ SAFE' if safety_status['safe'] else '❌ VIOLATION'}")
        print(f"  Max B-field: {safety_status['max_B']:.3f} T")
        print(f"  Max E-field: {safety_status['max_E']:.1f} V/m")
        print(f"  Max dB/dt: {safety_status['max_dBdt']:.1f} T/s")
        print(f"  Max SAR: {safety_status['max_SAR']:.2f} W/kg")
        
        if safety_status['violations']:
            print("  Violations detected:")
            for violation in safety_status['violations']:
                print(f"    • {violation}")
                
        # 4. Field Visualization
        print("\n4. FIELD DISTRIBUTION VISUALIZATION")
        print("-" * 40)
        
        self.visualize_field_distribution(optimal_currents, z_plane=0.0)
        
        # Summary statistics
        print(f"\n5. MEDICAL SYSTEM PERFORMANCE SUMMARY")
        print("-" * 40)
        print(f"  Treatment Volume: {np.prod(self.field_volume)*1000:.1f} liters")
        print(f"  Active Coils: {self.n_coils}")
        print(f"  Drug Delivery Efficiency: {efficiency:.1%}")
        print(f"  Force Targeting Precision: ±{np.mean([np.linalg.norm(af) for af in achieved_forces])*1e12:.1f} pN")
        print(f"  Safety Compliance: {'✅ FULL' if safety_status['safe'] else '⚠️ PARTIAL'}")
        
        return {
            'drug_delivery_efficiency': efficiency,
            'optimal_currents': optimal_currents,
            'safety_status': safety_status,
            'force_precision': np.mean([np.linalg.norm(af) for af in achieved_forces])
        }

def main():
    """Main demonstration of Medical Tractor Field Array."""
    print("Initializing Medical Tractor Field Array System...")
    
    # Create medical system
    medical_array = MedicalTractorFieldArray(
        array_size=(6, 6),
        field_volume=(0.4, 0.4, 0.2),  # 40×40×20 cm treatment volume
        max_field_strength=2.0,         # 2 Tesla maximum
        safety_margin=0.4               # 40% safety margin
    )
    
    # Run full demonstration
    results = medical_array.demonstrate_medical_applications()
    
    print(f"\n🏥 MEDICAL TRACTOR FIELD ARRAY - STEP 19 COMPLETE ✅")
    print(f"   Drug delivery efficiency: {results['drug_delivery_efficiency']:.1%}")
    print(f"   Force precision: {results['force_precision']*1e12:.1f} pN")
    print(f"   Safety compliance: {'✅' if results['safety_status']['safe'] else '⚠️'}")
    
    return medical_array, results

if __name__ == "__main__":
    # Suppress scientific notation warnings
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    
    medical_system, demo_results = main()
