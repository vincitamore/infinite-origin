"""
Extreme Scale Simulation

This script demonstrates the power of the Infinite Origin Framework by
simulating extreme scale scenarios like gravitational collapses and expansions,
which traditional numerical methods struggle to handle.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse
from matplotlib.animation import FuncAnimation
import time

# Add the current directory to path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configuration_space import Configuration, Point
from dynamics_engine import TimeTransformation, simulate
from visualization_tools import animate_dual_view

def create_unstable_configuration(scenario='collapse'):
    """
    Create an unstable configuration that will demonstrate extreme scale changes.
    
    Args:
        scenario: Type of scenario ('collapse', 'expansion', 'binary_collapse')
        
    Returns:
        Configuration object
    """
    if scenario == 'collapse':
        # Create a system that will gravitationally collapse
        # A central massive object and several smaller objects in unstable orbits
        points = [
            # Massive central object
            Point([0.0, 0.0], weight=1.0, properties={"name": "Central Mass"}),
            
            # Smaller objects in close, unstable orbits
            Point([0.5, 0.0], weight=0.001, properties={"name": "Object 1", "velocity": [0.0, 0.7]}),
            Point([-0.3, 0.4], weight=0.001, properties={"name": "Object 2", "velocity": [0.6, 0.0]}),
            Point([0.0, -0.6], weight=0.001, properties={"name": "Object 3", "velocity": [-0.5, 0.0]}),
            Point([0.2, 0.2], weight=0.0005, properties={"name": "Object 4", "velocity": [0.0, 0.5]}),
            Point([-0.4, -0.2], weight=0.0005, properties={"name": "Object 5", "velocity": [0.3, -0.4]}),
        ]
    
    elif scenario == 'expansion':
        # Create a system that will rapidly expand
        # Objects with high initial velocities
        points = [
            # Central object
            Point([0.0, 0.0], weight=0.5, properties={"name": "Central Mass"}),
            
            # Fast-moving objects
            Point([0.5, 0.0], weight=0.1, properties={"name": "Object 1", "velocity": [0.0, 3.0]}),
            Point([-0.3, 0.4], weight=0.1, properties={"name": "Object 2", "velocity": [2.5, 0.0]}),
            Point([0.0, -0.6], weight=0.1, properties={"name": "Object 3", "velocity": [-2.0, 0.0]}),
            Point([0.2, 0.2], weight=0.05, properties={"name": "Object 4", "velocity": [0.0, 2.2]}),
            Point([-0.4, -0.2], weight=0.05, properties={"name": "Object 5", "velocity": [1.8, -1.8]}),
        ]
    
    elif scenario == 'binary_collapse':
        # Create a binary system that will merge
        points = [
            # Two massive objects
            Point([0.5, 0.0], weight=0.5, properties={"name": "Mass 1", "velocity": [0.0, 0.4]}),
            Point([-0.5, 0.0], weight=0.5, properties={"name": "Mass 2", "velocity": [0.0, -0.4]}),
            
            # Smaller orbiting objects
            Point([0.7, 0.0], weight=0.01, properties={"name": "Satellite 1", "velocity": [0.0, 0.8]}),
            Point([-0.7, 0.0], weight=0.01, properties={"name": "Satellite 2", "velocity": [0.0, -0.8]}),
            Point([0.0, 0.8], weight=0.005, properties={"name": "Satellite 3", "velocity": [0.7, 0.0]}),
        ]
    
    elif scenario == 'multi_scale':
        # Create a system with multiple scales
        # A main system with orbiting objects and a distant small subsystem
        points = [
            # Main system
            Point([0.0, 0.0], weight=1.0, properties={"name": "Central Star"}),
            Point([1.0, 0.0], weight=0.001, properties={"name": "Planet 1", "velocity": [0.0, 1.0]}),
            Point([0.0, 1.5], weight=0.001, properties={"name": "Planet 2", "velocity": [-0.8, 0.0]}),
            
            # Distant small subsystem (a binary)
            Point([15.0, 15.0], weight=0.1, properties={"name": "Small Star 1", "velocity": [0.0, 0.1]}),
            Point([15.5, 15.0], weight=0.05, properties={"name": "Small Star 2", "velocity": [0.0, -0.2]}),
            
            # Extremely distant object on highly eccentric orbit
            Point([50.0, 0.0], weight=0.01, properties={"name": "Comet", "velocity": [0.0, 0.15]}),
        ]
    
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
    
    # Create configuration
    config = Configuration(points)
    
    return config

def create_custom_dynamics(scenario='collapse'):
    """
    Create a dynamics function tailored to the specific scenario.
    
    Args:
        scenario: Type of scenario
        
    Returns:
        Dynamics function F(σ)
    """
    if scenario == 'collapse':
        # Strong attractive force
        return lambda s: -1.2 * np.exp(s) * (1.0 - np.exp(-0.1 * s**2))
    
    elif scenario == 'expansion':
        # Initially repulsive force that decreases with expansion
        return lambda s: 0.8 * np.exp(s) * np.exp(-0.2 * s**2)
    
    elif scenario == 'binary_collapse':
        # Stronger attraction for binary mergers
        return lambda s: -1.5 * np.exp(s) * (1.0 - np.exp(-0.15 * s**2))
    
    elif scenario == 'multi_scale':
        # Balanced force for stable multi-scale system
        return lambda s: -1.0 * np.exp(s) * (1.0 - np.exp(-0.05 * s**2))
    
    else:
        # Default gravitational
        return lambda s: -1.0 * np.exp(s)

def create_custom_time_transformation(scenario='collapse'):
    """
    Create a time transformation suited to the specific scenario.
    
    Args:
        scenario: Type of scenario
        
    Returns:
        TimeTransformation object
    """
    from dynamics_engine import TimeTransformation
    
    if scenario == 'collapse':
        # Slower time evolution near collapse
        return TimeTransformation(lambda s: 0.3 * s)
    
    elif scenario == 'expansion':
        # Faster time evolution during expansion
        return TimeTransformation(lambda s: 0.7 * s)
    
    elif scenario == 'binary_collapse':
        # More precise timing for binary interaction
        return TimeTransformation(lambda s: 0.4 * s)
    
    elif scenario == 'multi_scale':
        # Balanced for multi-scale system
        return TimeTransformation(lambda s: 0.5 * s)
    
    else:
        # Default
        return TimeTransformation(lambda s: 0.5 * s)

def run_extreme_simulation(scenario='collapse', tau_max=20.0, num_steps=1000):
    """
    Run a simulation of an extreme scale scenario.
    
    Args:
        scenario: Type of scenario
        tau_max: Maximum tau-time for the simulation
        num_steps: Number of simulation steps
        
    Returns:
        tuple: (final_config, trajectory)
    """
    print(f"Creating extreme {scenario} configuration...")
    config = create_unstable_configuration(scenario)
    
    print(f"Creating specialized dynamics function...")
    F = create_custom_dynamics(scenario)
    
    print(f"Creating specialized time transformation...")
    transform = create_custom_time_transformation(scenario)
    
    # Extract velocities for custom initial conditions
    initial_velocities = {}
    for i, point in enumerate(config.points):
        if "velocity" in point.properties:
            initial_velocities[i] = point.properties["velocity"]
    
    print(f"Running simulation with tau_max={tau_max}, steps={num_steps}...")
    start_time = time.time()
    
    # Custom state derivative function with velocities
    from dynamics_engine.integrator import create_state_derivative_function, integrate
    from solar_system_simulation import create_custom_state_derivative_function
    
    # Create custom state derivative function
    deriv_fn = create_custom_state_derivative_function(config, F, transform)
    
    # Run integration
    trajectory = integrate(config, F, tau_max, num_steps, transform, use_adaptive_steps=True)
    
    # Extract final configuration (we need to reconstruct from the final state)
    final_config, _ = simulate(config, F, tau_max, num_steps, transform)
    
    elapsed_time = time.time() - start_time
    print(f"Simulation complete! Final sigma: {final_config.sigma:.4f}")
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    
    # Calculate scale ratio
    min_scale = np.exp(np.min(trajectory['sigma']))
    max_scale = np.exp(np.max(trajectory['sigma']))
    scale_ratio = max_scale / min_scale if min_scale > 0 else float('inf')
    
    print(f"Scale metrics:")
    print(f"  Minimum scale: {min_scale:.2e}")
    print(f"  Maximum scale: {max_scale:.2e}")
    print(f"  Scale ratio: {scale_ratio:.2e}")
    
    return final_config, trajectory

def visualize_extreme_simulation(final_config, trajectory, scenario, save_prefix=None):
    """
    Create visualizations for the extreme simulation.
    
    Args:
        final_config: Final configuration
        trajectory: Trajectory data
        scenario: Scenario name
        save_prefix: Prefix for saving output files
    """
    os.makedirs("output", exist_ok=True)
    
    save_path = f"output/{save_prefix}_{scenario}_dual_animation.mp4" if save_prefix else None
    
    print("\nGenerating dual view animation...")
    try:
        anim = animate_dual_view(
            trajectory,
            interval=50,
            figsize=(12, 6),
            save_path=save_path,
            title=f"Extreme {scenario.title()} Simulation",
            show_time=True,
            fps=30,
            dynamic_scaling=True
        )
        if save_path:
            print(f"Animation saved to {save_path}")
    except Exception as e:
        print(f"Animation error: {e}")
        print("Make sure ffmpeg is installed for saving animations.")
        
    # Create additional analysis plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot scale evolution
    ax1.plot(trajectory['tau'], np.exp(trajectory['sigma']), 'b-')
    ax1.set_title('Scale Evolution')
    ax1.set_xlabel('τ-time')
    ax1.set_ylabel('Scale factor (e^σ)')
    ax1.set_yscale('log')
    
    # Plot time transformation
    ax2.plot(trajectory['tau'], trajectory['t'], 'g-')
    ax2.set_title('Physical Time Evolution')
    ax2.set_xlabel('τ-time')
    ax2.set_ylabel('Physical time')
    
    plt.tight_layout()
    
    # Save the plot
    if save_prefix:
        analysis_path = f"output/{save_prefix}_{scenario}_analysis.png"
        plt.savefig(analysis_path)
        print(f"Analysis plot saved to {analysis_path}")
        plt.close(fig)
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(
        description='Extreme Scale Simulation using the Infinite Origin Framework',
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        '--scenario', type=str,
        choices=['collapse', 'expansion', 'binary_collapse', 'multi_scale'],
        default='collapse',
        help='Type of extreme scenario to simulate'
    )
    
    parser.add_argument(
        '--tau-max', type=float, default=20.0, 
        help='Maximum tau value for simulation (default: 20.0)'
    )
    
    parser.add_argument(
        '--num-steps', type=int, default=1000,
        help='Number of simulation steps (default: 1000)'
    )
    
    parser.add_argument(
        '--save', type=str, metavar='PREFIX',
        default='extreme',
        help='Save visualization with given prefix (default: extreme)'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Extreme Scale Simulation using the Infinite Origin Framework")
    print("=" * 70)
    
    # Print scenario description
    scenario_descriptions = {
        'collapse': 'Gravitational collapse of objects around a central mass',
        'expansion': 'Rapid expansion of a system with high initial velocities',
        'binary_collapse': 'Merger of a binary system with orbiting satellites',
        'multi_scale': 'System with structures at vastly different scales'
    }
    
    print(f"\nScenario: {args.scenario}")
    print(f"Description: {scenario_descriptions.get(args.scenario, 'Custom scenario')}")
    
    # Run simulation
    final_config, trajectory = run_extreme_simulation(
        scenario=args.scenario,
        tau_max=args.tau_max,
        num_steps=args.num_steps
    )
    
    # Visualize results
    visualize_extreme_simulation(
        final_config,
        trajectory,
        args.scenario,
        save_prefix=args.save
    )
    
    print("\n" + "=" * 70)
    print(f"Extreme scale simulation completed successfully!")
    print("=" * 70)

if __name__ == "__main__":
    main() 