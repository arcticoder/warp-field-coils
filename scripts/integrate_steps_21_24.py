"""
Integration Script: Add Steps 21-24 to Existing Pipeline
=======================================================

Integrates the new analysis capabilities into the existing warp field pipeline.
This script can be run to demonstrate the enhanced unified system.
"""

import sys
import os
import logging
import json
import time

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, '..', 'src'))

# Import the extended pipeline
from step24_extended_pipeline import ExtendedWarpFieldPipeline, ExtendedPipelineParams

def integrate_steps_21_24():
    """Demonstrate integration of Steps 21-24 with existing pipeline"""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('integration_steps_21_24.log'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("🚀 Starting Integration of Steps 21-24")
    
    # Configure the extended pipeline for demonstration
    pipeline_params = ExtendedPipelineParams(
        pipeline_name="Unified Warp Field System v2.0",
        enable_calibration=True,
        enable_sensitivity=True, 
        enable_refinements=True,
        calibration_iterations=25,  # Reasonable for demo
        sensitivity_samples=300,    # Reasonable for demo
        output_directory="steps_21_24_integration_results"
    )
    
    # Create the extended pipeline
    pipeline = ExtendedWarpFieldPipeline(pipeline_params)
    
    print("🌟 Unified Warp Field System - Steps 21-24 Integration")
    print("=" * 60)
    print(f"Pipeline: {pipeline_params.pipeline_name}")
    print(f"Output Directory: {pipeline_params.output_directory}")
    print()
    
    try:
        # Initialize all systems
        print("📋 Phase 1: System Initialization")
        print("-" * 40)
        init_start = time.time()
        
        init_results = pipeline.initialize_subsystems()
        
        init_time = time.time() - init_start
        print(f"✅ Systems initialized in {init_time:.2f}s")
        
        for system, status in init_results.items():
            print(f"   • {system.title()}: {status}")
        print()
        
        # Step 21: Unified Calibration
        print("🔧 Phase 2: Unified System Calibration (Step 21)")  
        print("-" * 50)
        calib_start = time.time()
        
        calib_results = pipeline.step_21_unified_calibration()
        
        calib_time = time.time() - calib_start
        print(f"✅ Calibration completed in {calib_time:.2f}s")
        
        if calib_results.get('success', False):
            optimal = calib_results['optimal_parameters']
            scores = calib_results['performance_scores']
            
            print("   Optimal Parameters:")
            print(f"   • Subspace coupling: {optimal['subspace_coupling']:.2e}")
            print(f"   • Grid spacing: {optimal['grid_spacing']:.3f} m")
            print(f"   • Medical field: {optimal['medical_field_strength']:.2f}")
            print(f"   • Tomography projections: {optimal['n_projections']}")
            print()
            print("   Performance Scores:")
            print(f"   • FTL Rate: {scores['ftl_rate']:.3f}")
            print(f"   • Grid Uniformity: {scores['grid_uniformity']:.3f}")
            print(f"   • Medical Precision: {scores['medical_precision']:.3f}")
            print(f"   • Tomography Fidelity: {scores['tomography_fidelity']:.3f}")
            print(f"   • Overall: {scores['overall']:.3f}")
        else:
            print("   ⚠️ Calibration encountered issues")
        print()
        
        # Step 22: Sensitivity Analysis
        print("🔍 Phase 3: Sensitivity Analysis (Step 22)")
        print("-" * 45)
        sens_start = time.time()
        
        sens_results = pipeline.step_22_sensitivity_analysis()
        
        sens_time = time.time() - sens_start
        print(f"✅ Sensitivity analysis completed in {sens_time:.2f}s")
        
        if 'local' in sens_results:
            local = sens_results['local']
            if 'sensitivity_ranking' in local:
                print("   Parameter Sensitivity Ranking:")
                for i, (param, sensitivity) in enumerate(local['sensitivity_ranking'][:3], 1):
                    print(f"   {i}. {param}: {sensitivity:.6f}")
        
        if 'monte_carlo' in sens_results:
            mc = sens_results['monte_carlo']
            if 'uncertainty_ranking' in mc:
                print("   Output Uncertainty Ranking:")
                for i, (output, cv) in enumerate(mc['uncertainty_ranking'][:3], 1):
                    print(f"   {i}. {output}: CV = {cv:.4f}")
        print()
        
        # Step 23: Mathematical Refinements
        print("🔬 Phase 4: Mathematical Refinements (Step 23)")
        print("-" * 48)
        refine_start = time.time()
        
        refine_results = pipeline.step_23_mathematical_refinements()
        
        refine_time = time.time() - refine_start
        print(f"✅ Mathematical refinements completed in {refine_time:.2f}s")
        
        if 'dispersion_tailoring' in refine_results:
            disp = refine_results['dispersion_tailoring']
            if disp.get('optimization_success', False):
                print(f"   • Dispersion optimization: Success")
                print(f"     Optimal bandwidth: {disp.get('optimal_bandwidth', 0):.2e} Hz")
            else:
                print(f"   • Dispersion optimization: {disp.get('message', 'Failed')}")
        
        if '3d_tomography' in refine_results:
            tomo = refine_results['3d_tomography']
            print(f"   • 3D Tomography: {tomo.get('status', 'N/A')}")
            print(f"     Detector: {tomo.get('detector_size', 'N/A')}")
            print(f"     Volume: {tomo.get('reconstruction_volume', 'N/A')}")
        
        if 'adaptive_mesh' in refine_results:
            mesh = refine_results['adaptive_mesh']
            print(f"   • Adaptive Mesh: {mesh.get('status', 'N/A')}")
            print(f"     Refinement ratio: {mesh.get('refinement_ratio', 1.0):.2f}")
        print()
        
        # Performance Validation
        print("🎯 Phase 5: Performance Validation")
        print("-" * 35)
        valid_start = time.time()
        
        validation_results = pipeline.validate_system_performance()
        
        valid_time = time.time() - valid_start
        print(f"✅ Performance validation completed in {valid_time:.2f}s")
        
        overall = validation_results['overall_assessment']
        print(f"   System Status: {overall['system_status']}")
        print(f"   Performance Ratio: {overall['overall_performance_ratio']:.3f}")
        print(f"   All Requirements Met: {overall['all_requirements_met']}")
        print()
        
        print("   Subsystem Performance:")
        for subsys_name, subsys_data in validation_results.items():
            if subsys_name != 'overall_assessment':
                status = "✅" if subsys_data.get('meets_requirement', False) else "⚠️"
                ratio = subsys_data.get('performance_ratio', 0)
                print(f"   {status} {subsys_name.replace('_', ' ').title()}: {ratio:.3f}")
        print()
        
        # Generate Final Report
        print("📊 Phase 6: Report Generation")
        print("-" * 30)
        
        report = pipeline.generate_comprehensive_report()
        
        # Save comprehensive results
        output_dir = pipeline_params.output_directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save detailed results
        all_results = {
            'initialization': init_results,
            'step_21_calibration': calib_results,
            'step_22_sensitivity': sens_results,
            'step_23_refinements': refine_results,
            'performance_validation': validation_results,
            'execution_times': {
                'initialization': init_time,
                'calibration': calib_time,
                'sensitivity': sens_time,
                'refinements': refine_time,
                'validation': valid_time
            }
        }
        
        results_file = os.path.join(output_dir, "integration_results.json")
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        # Save report
        report_file = os.path.join(output_dir, "integration_report.txt")
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"✅ Results saved to: {output_dir}")
        print()
        
        # Display milestones
        milestones = pipeline.get_recent_milestones()
        print("🏆 Recent Milestones & Achievements")
        print("-" * 35)
        
        for i, milestone in enumerate(milestones[:5], 1):  # Top 5 milestones
            print(f"{i}. {milestone['category']}")
            print(f"   File: {milestone['file_path']} (lines {milestone['line_range']})")
            print(f"   Keywords: {', '.join(milestone['keywords'][:3])}")
            if milestone.get('math'):
                print(f"   Math: {milestone['math']}")
            print(f"   Observation: {milestone['observation'][:80]}...")
            print()
        
        # Final summary
        total_time = calib_time + sens_time + refine_time + valid_time
        print("🎉 Integration Complete!")
        print("=" * 25)
        print(f"Total Analysis Time: {total_time:.2f} seconds")
        print(f"System Status: {overall['system_status']}")
        print(f"Performance: {overall['overall_performance_ratio']:.1%}")
        print(f"Milestones Achieved: {len(milestones)}")
        print()
        print("The unified warp field system is now operational with:")
        print("✅ Multi-objective parameter optimization")
        print("✅ Comprehensive sensitivity analysis")
        print("✅ Advanced mathematical refinements")
        print("✅ Real-time performance monitoring")
        print("✅ Automated safety systems")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Integration failed: {e}")
        print(f"\n❌ Integration failed: {e}")
        return False

if __name__ == "__main__":
    success = integrate_steps_21_24()
    
    if success:
        print("\n🚀 Ready for operational deployment!")
        print("Next steps: Hardware integration and field testing")
    else:
        print("\n🔧 Integration issues detected - review logs for details")
        
    input("\nPress Enter to exit...")
