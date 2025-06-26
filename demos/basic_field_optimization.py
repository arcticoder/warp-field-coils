"""
Basic Field Optimization Demo

Demonstrates core electromagnetic field optimization capabilities
of the warp field coils system.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt


def main():
    """Run basic field optimization demonstration."""
    print("🌟 Warp Field Coils - Basic Demo")
    print("=" * 50)
    
    # Import modules with fallback for missing dependencies
    try:
        from field_solver import (
            ElectromagneticFieldSolver, 
            FieldConfiguration, 
            create_helmholtz_coils
        )
        field_solver_available = True
    except ImportError as e:
        print(f"⚠️  Field solver not available: {e}")
        field_solver_available = False
    
    try:
        from coil_optimizer import (
            CoilGeometryOptimizer,
            OptimizationConstraints,
            OptimizationObjectives
        )
        optimizer_available = True
    except ImportError as e:
        print(f"⚠️  Coil optimizer not available: {e}")
        optimizer_available = False
    
    try:
        from integration import NegativeEnergyInterface, IntegrationConfig
        integration_available = True
    except ImportError as e:
        print(f"⚠️  Integration interface not available: {e}")
        integration_available = False
    
    try:
        from hardware import CurrentDriver, HardwareConfig
        hardware_available = True
    except ImportError as e:
        print(f"⚠️  Hardware interface not available: {e}")
        hardware_available = False
    
    # Demo 1: Basic field computation
    if field_solver_available:
        print("\n🔬 Demo 1: Electromagnetic Field Computation")
        print("-" * 45)
        
        config = FieldConfiguration(resolution=10)
        solver = ElectromagneticFieldSolver(config)
        
        # Create simple coil configuration
        coils = create_helmholtz_coils(radius=0.05, separation=0.05, current=10.0)
        solver.set_coil_geometry(coils)
        
        # Compute field along axis
        z_points = np.linspace(-0.1, 0.1, 50)
        x_points = np.zeros_like(z_points)
        y_points = np.zeros_like(z_points)
        
        Bx, By, Bz = solver.compute_magnetic_field(x_points, y_points, z_points)
        
        print(f"✅ Computed field at {len(z_points)} points")
        print(f"   Max field: {np.max(np.sqrt(Bx**2 + By**2 + Bz**2))*1000:.2f} mT")
        print(f"   Field uniformity: {np.std(Bz)/np.mean(Bz)*100:.1f}%")
        
        # Plot field profile
        plt.figure(figsize=(10, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(z_points*1000, Bz*1000, 'b-', linewidth=2, label='Bz')
        plt.xlabel('Position (mm)')
        plt.ylabel('Magnetic Field (mT)')
        plt.title('Helmholtz Coil Field Profile')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 2D field map
        plt.subplot(1, 2, 2)
        y_grid = np.linspace(-0.05, 0.05, 20)
        z_grid = np.linspace(-0.1, 0.1, 40)
        Y_grid, Z_grid = np.meshgrid(y_grid, z_grid)
        X_grid = np.zeros_like(Y_grid)
        
        Bx_grid, By_grid, Bz_grid = solver.compute_magnetic_field(
            X_grid.flatten(), Y_grid.flatten(), Z_grid.flatten()
        )
        B_mag = np.sqrt(Bx_grid**2 + By_grid**2 + Bz_grid**2).reshape(Y_grid.shape)
        
        im = plt.contourf(Z_grid*1000, Y_grid*1000, B_mag*1000, levels=20, cmap='viridis')
        plt.colorbar(im, label='Field Magnitude (mT)')
        plt.xlabel('Z Position (mm)')
        plt.ylabel('Y Position (mm)')
        plt.title('2D Field Map (X=0)')
        
        plt.tight_layout()
        plt.savefig('field_optimization_demo.png', dpi=150, bbox_inches='tight')
        print("   📊 Field plot saved as 'field_optimization_demo.png'")
    
    # Demo 2: Field optimization
    if field_solver_available and optimizer_available:
        print("\n🎯 Demo 2: Field Optimization")
        print("-" * 35)
        
        # Setup optimization
        constraints = OptimizationConstraints(
            max_current=50.0,
            field_uniformity=0.05
        )
        
        objectives = OptimizationObjectives(
            target_field_strength=0.01,  # 10 mT
            weight_field=1.0,
            weight_power=0.3
        )
        
        optimizer = CoilGeometryOptimizer(constraints, objectives)
        optimizer.set_field_solver(solver)
        
        target_region = {
            'bounds': {'x': [-0.01, 0.01], 'y': [-0.01, 0.01], 'z': [-0.01, 0.01]},
            'target_field': 0.01
        }
        
        print("   🔧 Running optimization (simplified)...")
        
        # Simplified optimization demo
        result = optimizer.optimize_geometry(
            n_coils=2, 
            target_region=target_region, 
            method="gradient"
        )
        
        print(f"   ✅ Optimization complete!")
        print(f"      Success: {result['success']}")
        print(f"      Objective: {result['objective_value']:.6f}")
        print(f"      Evaluations: {result['n_evaluations']}")
    
    # Demo 3: Integration interface
    if integration_available:
        print("\n🔗 Demo 3: System Integration")
        print("-" * 35)
        
        config = IntegrationConfig(control_frequency=1e6)
        interface = NegativeEnergyInterface(config)
        
        # Mock integration
        class MockChamber:
            pass
        
        success = interface.connect_to_chamber_array(MockChamber())
        print(f"   Chamber connection: {'✅ Success' if success else '❌ Failed'}")
        
        if success:
            coil_fields = {'magnitude': 0.05, 'uniformity': 0.02}
            coupling = interface.synchronize_field_energy_coupling(coil_fields)
            print(f"   Coupling efficiency: {coupling['coupling_efficiency']:.1%}")
            print(f"   Energy enhancement: {coupling['energy_enhancement']:.2f}x")
    
    # Demo 4: Hardware interface
    if hardware_available:
        print("\n⚡ Demo 4: Hardware Control")
        print("-" * 30)
        
        config = HardwareConfig(max_current=100.0, control_frequency=1000.0)
        driver = CurrentDriver("demo_coil", config)
        
        print("   🔌 Enabling current driver...")
        driver.enable()
        
        print("   📊 Setting 25A output...")
        driver.set_output(25.0)
        
        feedback = driver.read_feedback()
        print(f"      Current: {feedback['current']:.1f} A")
        print(f"      Power: {feedback['power']:.1f} W")
        print(f"      Efficiency: {feedback['efficiency']:.1%}")
        
        driver.disable()
        print("   🔴 Driver disabled")
    
    # Summary
    print("\n🎉 Demo Summary")
    print("=" * 20)
    print(f"✅ Field Solver: {'Available' if field_solver_available else 'Not Available'}")
    print(f"✅ Optimizer: {'Available' if optimizer_available else 'Not Available'}")
    print(f"✅ Integration: {'Available' if integration_available else 'Not Available'}")
    print(f"✅ Hardware: {'Available' if hardware_available else 'Not Available'}")
    
    print("\n🚀 Next Steps:")
    print("   1. Install missing dependencies (see requirements.txt)")
    print("   2. Connect to real hardware systems")
    print("   3. Integrate with negative energy generators")
    print("   4. Scale up to full warp drive configuration")
    
    print("\n🌟 Warp Field Coils demo complete!")


if __name__ == "__main__":
    main()
