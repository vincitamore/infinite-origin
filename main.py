"""
Infinite Origin Framework - Main Entry Point

This module serves as the main entry point for the Infinite Origin framework,
a computational implementation of a geometric system with infinity at the origin.
"""

import argparse
import sys
import os
import time
import math

from configuration_space import Configuration, Point
from dynamics_engine.examples import (
    harmonic_oscillator_example,
    three_body_example,
    collapsing_system_example
)
from visualization_tools import (
    plot_configuration,
    plot_trajectory,
    animate_trajectory,
    animate_dual_view
)


def print_header():
    """Print framework header information."""
    print("=" * 70)
    print("Infinite Origin Framework")
    print("A geometric system with infinity at the origin")
    print("=" * 70)
    

def run_example(example_name):
    """Run a specific example simulation."""
    if example_name == 'harmonic':
        print("\nRunning harmonic oscillator example:")
        final_config, trajectory = harmonic_oscillator_example()
        return final_config, trajectory
        
    elif example_name == 'three-body':
        print("\nRunning three-body example:")
        final_config, trajectory = three_body_example()
        return final_config, trajectory
        
    elif example_name == 'collapse':
        print("\nRunning collapsing system example:")
        final_config, trajectory = collapsing_system_example()
        return final_config, trajectory
        
    elif example_name == 'all':
        results = {}
        
        print("\nRunning harmonic oscillator example:")
        results['harmonic'] = harmonic_oscillator_example()
        
        print("\n" + "=" * 50)
        print("\nRunning three-body example:")
        results['three-body'] = three_body_example()
        
        print("\n" + "=" * 50)
        print("\nRunning collapsing system example:")
        results['collapse'] = collapsing_system_example()
        
        return results
    
    else:
        print(f"Unknown example: {example_name}")
        return None, None


def visualize_results(config, trajectory, vis_type, save_path=None):
    """Visualize simulation results."""
    os.makedirs("output", exist_ok=True)
    
    if vis_type == 'static':
        # Create static visualization
        print("\nGenerating static visualization...")
        fig, ax = plot_configuration(
            config, 
            plane='r', 
            title="Configuration Visualization",
            show_scale=True,
            show_center_of_mass=True
        )
        
        if save_path:
            fig.savefig(f"output/{save_path}_config.png")
            print(f"Static visualization saved to output/{save_path}_config.png")
    
    elif vis_type == 'trajectory':
        # Create trajectory plot
        print("\nGenerating trajectory visualization...")
        fig, axes = plot_trajectory(
            trajectory,
            time_var='tau',
            show_sigma=True,
            show_physical_time=True,
            title="Trajectory Visualization"
        )
        
        if save_path:
            fig.savefig(f"output/{save_path}_trajectory.png")
            print(f"Trajectory visualization saved to output/{save_path}_trajectory.png")
    
    elif vis_type == 'animation':
        # Create animation
        print("\nGenerating animation...")
        try:
            anim = animate_trajectory(
                trajectory,
                interval=50,
                figsize=(8, 6),
                save_path=f"output/{save_path}_animation.mp4" if save_path else None,
                title="Configuration Animation",
                show_time=True,
                plane='r',
                fps=30
            )
            if save_path:
                print(f"Animation saved to output/{save_path}_animation.mp4")
        except Exception as e:
            print(f"Animation error: {e}")
            print("Make sure ffmpeg is installed for saving animations.")
    
    elif vis_type == 'dual':
        # Create dual view animation
        print("\nGenerating dual view animation...")
        try:
            anim = animate_dual_view(
                trajectory,
                interval=50,
                figsize=(12, 6),
                save_path=f"output/{save_path}_dual_animation.mp4" if save_path else None,
                title="Dual View Animation",
                show_time=True,
                fps=30,
                dynamic_scaling=True  # Enable dynamic r-plane scaling
            )
            if save_path:
                print(f"Dual view animation saved to output/{save_path}_dual_animation.mp4")
        except Exception as e:
            print(f"Animation error: {e}")
            print("Make sure ffmpeg is installed for saving animations.")
    
    elif vis_type == 'all':
        # Generate all visualization types
        visualize_results(config, trajectory, 'static', save_path)
        visualize_results(config, trajectory, 'trajectory', save_path)
        visualize_results(config, trajectory, 'animation', save_path)
        visualize_results(config, trajectory, 'dual', save_path)
    
    else:
        print(f"Unknown visualization type: {vis_type}")


def run_profile():
    """Run performance profiling."""
    print("\nRunning performance profiling...")
    
    # Import here to avoid overhead if not used
    from performance_profiling import (
        profile_hyperreal_arithmetic,
        profile_mapping_functions,
        profile_configuration_operations,
        profile_dynamics_simulation,
        profile_visualization,
        find_optimization_opportunities
    )
    
    # Run profiling
    print("\nProfiling basic operations...")
    profile_hyperreal_arithmetic(n_iterations=1000)
    profile_mapping_functions(n_iterations=1000)
    profile_configuration_operations(n_iterations=100)
    
    print("\nProfiling simulation and visualization...")
    final_config, trajectory = profile_dynamics_simulation()
    profile_visualization(final_config, trajectory)
    
    # Suggest optimizations
    find_optimization_opportunities()


def run_tests():
    """Run all tests."""
    print("\nRunning all tests...")
    
    import unittest
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('tests')
    
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    
    return result.wasSuccessful()


def create_custom_configuration(points_desc):
    """Create a custom configuration based on description."""
    points = []
    
    try:
        # Parse points description (format: "x1,y1,w1;x2,y2,w2;...")
        for point_desc in points_desc.split(';'):
            coords = point_desc.split(',')
            if len(coords) >= 2:
                if len(coords) >= 3:
                    # With weight
                    points.append(Point(
                        [float(coords[0]), float(coords[1])], 
                        weight=float(coords[2])
                    ))
                else:
                    # Without weight
                    points.append(Point([float(coords[0]), float(coords[1])]))
        
        # Create configuration
        config = Configuration(points)
        config.fix_center_of_mass()
        return config
    except Exception as e:
        print(f"Error creating custom configuration: {e}")
        print("Format should be: x1,y1,w1;x2,y2,w2;...")
        return None


def create_dynamics_function(function_type, param=None):
    """
    Create a dynamics function F(σ) based on the specified type.
    
    Args:
        function_type: Type of dynamics ('gravity', 'harmonic', 'collapse', 'oscillating', 'expansion', 'custom')
        param: Optional parameter to customize the function
    
    Returns:
        Function F(σ)
    """
    import numpy as np
    
    if function_type == 'gravity':
        # Classic gravitational (F ~ -e^σ)
        return lambda s: -np.exp(s)
        
    elif function_type == 'harmonic':
        # Harmonic oscillator (F ~ -e^(2σ))
        return lambda s: -np.exp(2*s)
        
    elif function_type == 'collapse':
        # Stronger collapse (F ~ -2e^σ)
        strength = 2.0 if param is None else float(param)
        return lambda s: -strength * np.exp(s)
        
    elif function_type == 'oscillating':
        # Oscillating force (F ~ -e^σ * cos(s))
        freq = 1.0 if param is None else float(param)
        return lambda s: -np.exp(s) * np.cos(freq * s)
        
    elif function_type == 'expansion':
        # Expansion force with stability improvements
        strength = 1.0 if param is None else float(param)
        # Add damping and limit growth to prevent numerical instability
        return lambda s: strength * np.exp(s) * np.exp(-0.1 * s**2) * np.tanh(s) 
        
    elif function_type == 'custom':
        # Evaluate custom expression with 's' as the variable
        if param is None:
            raise ValueError("Custom function requires an expression parameter")
        
        # Replace common mathematical functions
        param = param.replace('sin', 'np.sin')
        param = param.replace('cos', 'np.cos')
        param = param.replace('exp', 'np.exp')
        param = param.replace('log', 'np.log')
        param = param.replace('sqrt', 'np.sqrt')
        
        # Create lambda function
        try:
            return eval(f"lambda s: {param}")
        except Exception as e:
            print(f"Error creating custom function: {e}")
            print("Defaulting to gravitational dynamics")
            return lambda s: -np.exp(s)
            
    else:
        print(f"Unknown function type '{function_type}', defaulting to gravitational dynamics")
        return lambda s: -np.exp(s)


def create_time_transformation(transform_type, param=None):
    """
    Create a time transformation function f(σ) based on the specified type.
    
    Args:
        transform_type: Type of transformation ('standard', 'half', 'inverse', 'scale', 'custom')
        param: Optional parameter to customize the transformation
    
    Returns:
        TimeTransformation object
    """
    from dynamics_engine import TimeTransformation
    
    if transform_type == 'standard':
        # Standard f(σ) = σ (dt/dτ = e^σ)
        return TimeTransformation(lambda s: s)
        
    elif transform_type == 'half':
        # f(σ) = σ/2 (dt/dτ = e^(σ/2))
        return TimeTransformation(lambda s: s/2)
        
    elif transform_type == 'inverse':
        # f(σ) = -σ (dt/dτ = e^(-σ))
        return TimeTransformation(lambda s: -s)
        
    elif transform_type == 'scale':
        # f(σ) = k*σ
        k = 1.0 if param is None else float(param)
        return TimeTransformation(lambda s: k * s)
        
    elif transform_type == 'custom':
        # Evaluate custom expression with 's' as the variable
        if param is None:
            raise ValueError("Custom transformation requires an expression parameter")
        
        # Replace common mathematical functions
        param = param.replace('sin', 'math.sin')
        param = param.replace('cos', 'math.cos')
        param = param.replace('exp', 'math.exp')
        param = param.replace('log', 'math.log')
        param = param.replace('sqrt', 'math.sqrt')
        
        # Create lambda function
        try:
            return TimeTransformation(eval(f"lambda s: {param}"))
        except Exception as e:
            print(f"Error creating custom transformation: {e}")
            print("Defaulting to standard transformation")
            return TimeTransformation(lambda s: s/2)
            
    else:
        print(f"Unknown transformation type '{transform_type}', defaulting to half-scale")
        return TimeTransformation(lambda s: s/2)


def main():
    """Main entry point for the framework."""
    parser = argparse.ArgumentParser(
        description='Infinite Origin Framework - A geometric system with infinity at the origin',
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # Main operation mode
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        '--example', type=str, choices=['harmonic', 'three-body', 'collapse', 'all'],
        help='Example simulation to run'
    )
    mode_group.add_argument(
        '--custom', type=str, metavar='POINTS',
        help=('Create a custom configuration (format: "x1,y1,w1;x2,y2,w2;...")\n'
              'Example: "0,0,2;1,0,1;0,1,1" creates a three-point system')
    )
    mode_group.add_argument(
        '--profile', action='store_true',
        help='Run performance profiling'
    )
    mode_group.add_argument(
        '--test', action='store_true',
        help='Run all tests'
    )
    
    # Simulation options
    sim_group = parser.add_argument_group('Simulation options')
    sim_group.add_argument(
        '--tau-max', type=float, default=5.0, 
        help='Maximum tau value for simulation (default: 5.0)'
    )
    sim_group.add_argument(
        '--num-steps', type=int, default=200,
        help='Number of simulation steps (default: 200)'
    )
    sim_group.add_argument(
        '--dynamics', type=str, 
        choices=['gravity', 'harmonic', 'collapse', 'oscillating', 'expansion', 'custom'],
        default='gravity',
        help='Type of dynamics function to use (default: gravity)'
    )
    sim_group.add_argument(
        '--dynamics-param', type=str,
        help='Parameter for dynamics function (strength, frequency, or custom expression)'
    )
    sim_group.add_argument(
        '--time-transform', type=str,
        choices=['standard', 'half', 'inverse', 'scale', 'custom'],
        default='half',
        help='Type of time transformation to use (default: half)'
    )
    sim_group.add_argument(
        '--transform-param', type=str,
        help='Parameter for time transformation (scale factor or custom expression)'
    )
    
    # Visualization options
    vis_group = parser.add_argument_group('Visualization options')
    vis_group.add_argument(
        '--visualize', type=str, choices=['static', 'trajectory', 'animation', 'dual', 'all'],
        help='Visualization type to generate'
    )
    vis_group.add_argument(
        '--save', type=str, metavar='PREFIX',
        help='Save visualization with given prefix'
    )
    
    # Parse arguments
    if len(sys.argv) == 1:
        parser.print_help()
        return
    
    args = parser.parse_args()
    print_header()
    
    # Handle different operation modes
    start_time = time.time()
    
    if args.profile:
        run_profile()
    
    elif args.test:
        success = run_tests()
        if not success:
            print("\nSome tests failed!")
            sys.exit(1)
    
    elif args.custom:
        config = create_custom_configuration(args.custom)
        if config:
            print(f"\nCreated custom configuration: {config}")
            
            # Run simulation with this configuration
            from dynamics_engine import TimeTransformation, simulate
            import numpy as np
            
            # Create dynamics function
            F = create_dynamics_function(args.dynamics, args.dynamics_param)
            
            # Create time transformation
            transform = create_time_transformation(args.time_transform, args.transform_param)
            
            # Print dynamics information
            dynamics_info = f"{args.dynamics}"
            if args.dynamics_param:
                dynamics_info += f" (param: {args.dynamics_param})"
                
            transform_info = f"{args.time_transform}"
            if args.transform_param:
                transform_info += f" (param: {args.transform_param})"
                
            print(f"\nDynamics: {dynamics_info}")
            print(f"Time transformation: {transform_info}")
            
            # Run simulation
            tau_max = args.tau_max
            num_steps = args.num_steps
            print(f"Running simulation with tau_max={tau_max}, steps={num_steps}...")
            final_config, trajectory = simulate(
                config, F, tau_max=tau_max, num_steps=num_steps, 
                time_transform=transform
            )
            
            print(f"Simulation complete! Final sigma: {final_config.sigma:.4f}")
            
            # Visualize if requested
            if args.visualize:
                visualize_results(final_config, trajectory, args.visualize, args.save or "custom")
    
    else:  # Run example
        example = args.example or 'harmonic'
        results = run_example(example)
        
        # Handle visualization for examples
        if args.visualize:
            if example == 'all':
                # Handle each example
                for ex_name, (config, traj) in results.items():
                    save_prefix = f"{args.save}_{ex_name}" if args.save else ex_name
                    visualize_results(config, traj, args.visualize, save_prefix)
            else:
                # Single example
                config, traj = results
                save_prefix = args.save or example
                visualize_results(config, traj, args.visualize, save_prefix)
    
    elapsed_time = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"Operation completed in {elapsed_time:.2f} seconds!")
    print("=" * 70)


if __name__ == "__main__":
    main() 