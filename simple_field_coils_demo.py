#!/usr/bin/env python3
"""
Simple Enhanced Field Coils Demonstration
=========================================

A Windows-compatible demonstration of the Enhanced Field Coils system.
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from field_solver.lqg_enhanced_fields import LQGEnhancedFieldGenerator, LQGFieldConfig
    from integration.lqg_framework_integration import setup_enhanced_field_coils_with_lqg
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure you're running from the warp-field-coils directory")
    sys.exit(1)

class SimpleFieldCoilsDemo:
    """Simple demonstration of Enhanced Field Coils."""
    
    def __init__(self):
        # Setup basic logging without emojis
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            stream=sys.stdout
        )
        self.logger = logging.getLogger("simple_demo")
        
    async def run_demo(self):
        """Run a simple demonstration."""
        print("=" * 60)
        print("Enhanced Field Coils Simple Demonstration")
        print("=" * 60)
        
        # Step 1: Create field generator
        print("\nStep 1: Creating LQG Enhanced Field Generator...")
        
        # Create default configuration
        config = LQGFieldConfig()
        field_generator = LQGEnhancedFieldGenerator(config)
        print("Field generator created successfully!")
        
        # Step 2: Basic field generation test
        print("\nStep 2: Testing basic field generation...")
        try:
            # Create test field geometry
            import numpy as np
            
            # Field evaluation points (e.g., around a toroidal coil)
            theta = np.linspace(0, 2*np.pi, 10)
            phi = np.linspace(0, np.pi, 5)
            r = 1.0  # 1 meter radius
            
            positions = np.array([
                [r * np.cos(t), r * np.sin(t), 0.5 * np.cos(p)]
                for t in theta for p in phi
            ])
            
            # Classical current distribution (1000 Amperes in 8 coils)
            classical_currents = np.array([1000.0] * 8)
            
            # Coil positions (toroidal arrangement)
            coil_theta = np.linspace(0, 2*np.pi, 8, endpoint=False)
            coil_positions = np.array([
                [1.2 * np.cos(t), 1.2 * np.sin(t), 0.0]
                for t in coil_theta
            ])
            
            result = field_generator.generate_lqg_corrected_field(
                positions, classical_currents, coil_positions
            )
            print(f"Field generation successful!")
            print(f"- Enhancement ratio: {result.enhancement_ratio:.3f}")
            print(f"- Stability metric: {result.stability_metric:.4f}")
            print(f"- Field strength: {result.field_strength:.2e}")
            
        except Exception as e:
            print(f"Field generation failed: {e}")
            
        # Step 3: LQG integration test
        print("\nStep 3: Testing LQG ecosystem integration...")
        try:
            integration = await setup_enhanced_field_coils_with_lqg(field_generator)
            
            if integration:
                print("LQG integration successful!")
                summary = await self._get_integration_summary(integration)
                print(f"- Polymer field generator: {summary.get('polymer_connected', False)}")
                print(f"- Hardware abstraction: {summary.get('hardware_connected', False)}")
                print(f"- System ready: {summary.get('system_ready', False)}")
            else:
                print("LQG integration setup completed with fallbacks")
                
        except Exception as e:
            print(f"LQG integration test failed: {e}")
            
        # Step 4: Performance demonstration
        print("\nStep 4: Performance demonstration...")
        await self._performance_demo(field_generator)
        
        print("\n" + "=" * 60)
        print("Enhanced Field Coils demonstration completed!")
        print("=" * 60)
        
    async def _get_integration_summary(self, integration):
        """Get integration summary safely."""
        try:
            if hasattr(integration, 'get_integration_summary'):
                return integration.get_integration_summary()
            else:
                return {
                    'polymer_connected': hasattr(integration, 'polymer_field_generator'),
                    'hardware_connected': hasattr(integration, 'hardware_abstraction'),
                    'system_ready': True
                }
        except Exception as e:
            print(f"Could not get integration summary: {e}")
            return {}
            
    async def _performance_demo(self, field_generator):
        """Demonstrate field generation performance."""
        try:
            print("Running field generation benchmark...")
            import numpy as np
            
            # Create test geometry for benchmarks
            positions = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
            coil_positions = np.array([[1.2, 0.0, 0.0], [0.0, 1.2, 0.0]])
            
            test_currents = [
                np.array([500.0, 600.0]),    # Test 1: Lower currents
                np.array([800.0, 900.0]),    # Test 2: Medium currents  
                np.array([1000.0, 1100.0])  # Test 3: Higher currents
            ]
            
            total_time = 0
            successful_generations = 0
            
            for i, currents in enumerate(test_currents):
                start_time = time.time()
                try:
                    result = field_generator.generate_lqg_corrected_field(
                        positions, currents, coil_positions
                    )
                    end_time = time.time()
                    generation_time = end_time - start_time
                    total_time += generation_time
                    successful_generations += 1
                    
                    print(f"  Test {i+1}: {generation_time:.3f}s - Enhancement: {result.enhancement_ratio:.3f}")
                    
                except Exception as e:
                    print(f"  Test {i+1}: Failed - {e}")
                    
            if successful_generations > 0:
                avg_time = total_time / successful_generations
                print(f"Average generation time: {avg_time:.3f}s")
                print(f"Successful generations: {successful_generations}/{len(test_currents)}")
            else:
                print("No successful field generations in benchmark")
                
        except Exception as e:
            print(f"Performance demo failed: {e}")

async def main():
    """Main demonstration function."""
    demo = SimpleFieldCoilsDemo()
    await demo.run_demo()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        sys.exit(1)
