"""
Benchmark Comparison: Hyperreal Framework vs Traditional Methods

This script compares the performance and accuracy of the Infinite Origin Framework
against traditional N-body integration methods for astronomical simulations.
"""

import os
import sys
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse
from scipy.integrate import solve_ivp

# Add the current directory to path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configuration_space import Configuration, Point
from dynamics_engine import TimeTransformation, simulate
from dynamics_engine.integrator import integrate
from extreme_simulation import create_unstable_configuration, create_custom_dynamics
from solar_system_simulation import create_custom_state_derivative_function

# Constants for traditional simulation
G = 1.0  # Gravitational constant in simulation units

def traditional_nbody_derivatives(t, state, masses):
    """
    Compute derivatives for traditional N-body integration using ODE solver.
    
    Args:
        t: Current time
        state: State vector [x1, y1, vx1, vy1, x2, y2, vx2, vy2, ...]
        masses: List of masses for each body
        
    Returns:
        Derivatives of state vector
    """
    n_bodies = len(masses)
    derivatives = np.zeros_like(state)
    
    # Extract positions and velocities
    positions = state.reshape(n_bodies, 4)[:, 0:2]  # [x, y] for each body
    velocities = state.reshape(n_bodies, 4)[:, 2:4]  # [vx, vy] for each body
    
    # Velocities are derivatives of positions
    for i in range(n_bodies):
        derivatives[i*4] = state[i*4+2]  # dx/dt = vx
        derivatives[i*4+1] = state[i*4+3]  # dy/dt = vy
    
    # Calculate accelerations (derivatives of velocities)
    for i in range(n_bodies):
        ax, ay = 0.0, 0.0
        for j in range(n_bodies):
            if i != j:
                dx = positions[j, 0] - positions[i, 0]
                dy = positions[j, 1] - positions[i, 1]
                r_squared = dx*dx + dy*dy
                
                # Add softening to prevent singularity
                softening = 1e-6
                r_squared_soft = r_squared + softening
                
                # Force magnitude: F = G*m1*m2/r^2
                # Acceleration: a = F/m1 = G*m2/r^2
                factor = G * masses[j] / (r_squared_soft * np.sqrt(r_squared_soft))
                
                ax += factor * dx
                ay += factor * dy
        
        derivatives[i*4+2] = ax  # dvx/dt = ax
        derivatives[i*4+3] = ay  # dvy/dt = ay
    
    return derivatives

def run_traditional_nbody(config, t_max, num_steps):
    """
    Run a traditional N-body simulation.
    
    Args:
        config: Configuration object with initial positions
        t_max: Maximum physical time to simulate
        num_steps: Number of steps for output
        
    Returns:
        Dictionary containing trajectory data
    """
    # Extract masses and initial conditions
    n_bodies = len(config.points)
    masses = [point.weight for point in config.points]
    
    # Build initial state vector [x1, y1, vx1, vy1, x2, y2, vx2, vy2, ...]
    initial_state = np.zeros(n_bodies * 4)
    
    for i, point in enumerate(config.points):
        # Position
        initial_state[i*4] = point.position[0]
        initial_state[i*4+1] = point.position[1]
        
        # Velocity
        if "velocity" in point.properties:
            initial_state[i*4+2] = point.properties["velocity"][0]
            initial_state[i*4+3] = point.properties["velocity"][1]
    
    # Set up time points
    t_eval = np.linspace(0, t_max, num_steps)
    
    print(f"Running traditional N-body simulation...")
    start_time = time.time()
    
    # Solve using scipy's ODE solver
    result = solve_ivp(
        lambda t, y: traditional_nbody_derivatives(t, y, masses),
        [0, t_max],
        initial_state,
        method='RK45',
        t_eval=t_eval,
        rtol=1e-6,
        atol=1e-9
    )
    
    elapsed_time = time.time() - start_time
    print(f"Traditional simulation completed in {elapsed_time:.2f} seconds")
    
    # Process results
    positions = []
    for i in range(num_steps):
        step_positions = []
        for j in range(n_bodies):
            step_positions.append([
                result.y[j*4, i],    # x
                result.y[j*4+1, i]   # y
            ])
        positions.append(step_positions)
    
    # Calculate scale for each time step
    scales = []
    for step_positions in positions:
        # Compute r_rms from positions
        r_squared_sum = 0
        for pos in step_positions:
            r_squared_sum += pos[0]**2 + pos[1]**2
        scale = np.sqrt(r_squared_sum / n_bodies)
        scales.append(scale)
    
    # Check for failure (NaN or Inf)
    has_failure = np.isnan(np.array(scales)).any() or np.isinf(np.array(scales)).any()
    
    return {
        't': result.t,
        'positions': np.array(positions),
        'scales': np.array(scales),
        'elapsed_time': elapsed_time,
        'success': result.success and not has_failure,
        'has_failure': has_failure,
        'message': result.message
    }

def run_hyperreal_simulation(config, tau_max, num_steps, scenario='binary_collapse'):
    """
    Run a simulation using the Hyperreal Framework.
    
    Args:
        config: Configuration object
        tau_max: Maximum tau-time for the simulation
        num_steps: Number of simulation steps
        scenario: Scenario type
        
    Returns:
        Dictionary containing results and trajectory
    """
    # Create dynamics function
    F = create_custom_dynamics(scenario)
    
    # Create time transformation
    transform = TimeTransformation(lambda s: 0.4 * s)
    
    print(f"Running Hyperreal Framework simulation...")
    start_time = time.time()
    
    # Create custom state derivative function
    deriv_fn = create_custom_state_derivative_function(config, F, transform)
    
    # Run integration
    trajectory = integrate(config, F, tau_max, num_steps, transform, use_adaptive_steps=True)
    
    # Extract final configuration
    final_config, _ = simulate(config, F, tau_max, num_steps, transform)
    
    elapsed_time = time.time() - start_time
    print(f"Hyperreal simulation completed in {elapsed_time:.2f} seconds")
    
    # Calculate scale ratio
    scales = np.exp(trajectory['sigma'])
    min_scale = np.min(scales)
    max_scale = np.max(scales)
    scale_ratio = max_scale / min_scale if min_scale > 0 else float('inf')
    
    # Check for failure (NaN or Inf)
    has_failure = np.isnan(np.array(scales)).any() or np.isinf(np.array(scales)).any()
    
    return {
        'trajectory': trajectory,
        'final_config': final_config,
        'elapsed_time': elapsed_time,
        'min_scale': min_scale,
        'max_scale': max_scale,
        'scale_ratio': scale_ratio,
        'has_failure': has_failure,
        'success': not has_failure
    }

def extract_simulation_data(config, trad_results, hyper_results):
    """
    Extract comparable data from both simulations for comparison.
    
    Args:
        config: Initial configuration
        trad_results: Results from traditional simulation
        hyper_results: Results from hyperreal simulation
        
    Returns:
        Dictionary of comparable metrics
    """
    n_bodies = len(config.points)
    
    # Extract data from traditional simulation
    trad_times = trad_results['t']
    trad_positions = trad_results['positions']
    trad_scales = trad_results['scales']
    
    # Extract data from hyperreal simulation
    hyper_trajectory = hyper_results['trajectory']
    hyper_times = hyper_trajectory['t']
    hyper_scales = np.exp(hyper_trajectory['sigma'])
    
    # Compute final position differences if both simulations succeeded
    if trad_results['success'] and hyper_results['success']:
        # Get final positions from traditional simulation
        trad_final_positions = trad_positions[-1]
        
        # Get final positions from hyperreal simulation
        # Need to extract from final_config
        hyper_final_config = hyper_results['final_config']
        hyper_final_positions = [point.position for point in hyper_final_config.points]
        
        # Compute RMS position difference
        rms_pos_diff = 0.0
        for i in range(n_bodies):
            dx = trad_final_positions[i][0] - hyper_final_positions[i][0]
            dy = trad_final_positions[i][1] - hyper_final_positions[i][1]
            rms_pos_diff += dx*dx + dy*dy
        rms_pos_diff = np.sqrt(rms_pos_diff / n_bodies)
    else:
        rms_pos_diff = None
    
    return {
        'trad_times': trad_times,
        'trad_scales': trad_scales,
        'hyper_times': hyper_times,
        'hyper_scales': hyper_scales,
        'trad_success': trad_results['success'],
        'hyper_success': hyper_results['success'],
        'trad_elapsed': trad_results['elapsed_time'],
        'hyper_elapsed': hyper_results['elapsed_time'],
        'speedup': trad_results['elapsed_time'] / hyper_results['elapsed_time'] if hyper_results['elapsed_time'] > 0 else float('inf'),
        'rms_position_difference': rms_pos_diff,
        'trad_has_failure': trad_results['has_failure'],
        'hyper_has_failure': hyper_results['has_failure'],
        'trad_message': trad_results.get('message', '')
    }

def plot_comparison(config, data, scenario, save_path=None):
    """
    Create comparison plots between the two methods.
    
    Args:
        config: Initial configuration
        data: Extracted comparison data
        scenario: Name of the scenario
        save_path: Path to save plot
    """
    fig = plt.figure(figsize=(14, 10))
    
    # Create grid layout
    gs = plt.GridSpec(3, 2, figure=fig)
    
    # Plot 1: Scale evolution over time
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Plot traditional simulation scale
    if data['trad_success']:
        ax1.plot(data['trad_times'], data['trad_scales'], 'b-', label='Traditional')
    
    # Plot hyperreal simulation scale
    if data['hyper_success']:
        ax1.plot(data['hyper_times'], data['hyper_scales'], 'r-', label='Hyperreal')
    
    ax1.set_title('Scale Evolution')
    ax1.set_xlabel('Physical Time')
    ax1.set_ylabel('Scale Factor')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Scale evolution with log scale
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Plot traditional simulation scale
    if data['trad_success']:
        ax2.plot(data['trad_times'], data['trad_scales'], 'b-', label='Traditional')
    
    # Plot hyperreal simulation scale
    if data['hyper_success']:
        ax2.plot(data['hyper_times'], data['hyper_scales'], 'r-', label='Hyperreal')
    
    ax2.set_title('Scale Evolution (Log Scale)')
    ax2.set_xlabel('Physical Time')
    ax2.set_ylabel('Scale Factor')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True)
    
    # Plot 3: Performance bar chart
    ax3 = fig.add_subplot(gs[1, 0])
    methods = ['Traditional', 'Hyperreal']
    times = [data['trad_elapsed'], data['hyper_elapsed']]
    colors = ['blue', 'red']
    
    # Add status indicators to the labels
    if not data['trad_success']:
        methods[0] += ' (FAILED)'
        colors[0] = 'gray'
    if not data['hyper_success']:
        methods[1] += ' (FAILED)'
        colors[1] = 'gray'
    
    ax3.bar(methods, times, color=colors)
    ax3.set_title('Computation Time Comparison')
    ax3.set_ylabel('Time (seconds)')
    for i, v in enumerate(times):
        ax3.text(i, v + 0.05, f"{v:.2f}s", ha='center')
    
    # Plot 4: Initial configuration
    ax4 = fig.add_subplot(gs[1, 1])
    positions = np.array([point.position for point in config.points])
    weights = np.array([point.weight for point in config.points])
    
    # Scale marker sizes by weight
    sizes = 20 + 180 * weights/np.max(weights)
    
    ax4.scatter(positions[:, 0], positions[:, 1], s=sizes, c='green', alpha=0.7)
    ax4.set_title('Initial Configuration')
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.grid(True)
    ax4.set_aspect('equal')
    
    # Plot 5: Text summary
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')
    
    # Prepare summary text
    summary_lines = [
        f"Benchmark Results: {scenario.replace('_', ' ').title()} Scenario",
        f"--------------------------------------------------------",
        f"Traditional Method: {'Successful' if data['trad_success'] else 'Failed'}",
        f"  - Computation time: {data['trad_elapsed']:.4f} seconds",
    ]
    
    if not data['trad_success']:
        summary_lines.append(f"  - Failure reason: {data['trad_message'] if data['trad_has_failure'] else 'Unknown'}")
    
    summary_lines.extend([
        f"Hyperreal Framework: {'Successful' if data['hyper_success'] else 'Failed'}",
        f"  - Computation time: {data['hyper_elapsed']:.4f} seconds",
        f"  - Speedup factor: {data['speedup']:.2f}x",
    ])
    
    if data['trad_success'] and data['hyper_success']:
        summary_lines.append(f"  - RMS position difference: {data['rms_position_difference']:.6f}")
    
    summary_text = "\n".join(summary_lines)
    ax5.text(0.5, 0.5, summary_text, ha='center', va='center', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Comparison plot saved to {save_path}")
    
    plt.show()

def run_benchmark(scenario='binary_collapse', save_results=True):
    """
    Run benchmark comparing traditional and hyperreal methods.
    
    Args:
        scenario: Type of scenario to test
        save_results: Whether to save results to file
        
    Returns:
        Dictionary with benchmark results
    """
    # Create configuration for testing
    config = create_unstable_configuration(scenario)
    
    # Adjust simulation parameters based on scenario
    if scenario == 'binary_collapse':
        t_max = 3.0
        tau_max = 20.0
    elif scenario == 'collapse':
        t_max = 2.0
        tau_max = 20.0
    elif scenario == 'expansion':
        t_max = 1.0
        tau_max = 10.0
    elif scenario == 'multi_scale':
        t_max = 5.0
        tau_max = 30.0
    else:
        t_max = 2.0
        tau_max = 20.0
    
    num_steps = 500
    
    print(f"Running benchmark for {scenario} scenario...")
    print(f"Traditional simulation will run to t_max={t_max}")
    print(f"Hyperreal simulation will run to tau_max={tau_max}")
    
    # Run traditional N-body simulation
    try:
        trad_results = run_traditional_nbody(config, t_max, num_steps)
    except Exception as e:
        print(f"Traditional simulation failed with error: {str(e)}")
        trad_results = {
            'elapsed_time': 0.0,
            'success': False,
            'has_failure': True,
            'message': str(e)
        }
    
    # Run hyperreal simulation
    try:
        hyper_results = run_hyperreal_simulation(config, tau_max, num_steps, scenario)
    except Exception as e:
        print(f"Hyperreal simulation failed with error: {str(e)}")
        hyper_results = {
            'elapsed_time': 0.0,
            'success': False,
            'has_failure': True
        }
    
    # Extract comparable data
    comparison_data = extract_simulation_data(config, trad_results, hyper_results)
    
    # Plot comparison
    if save_results:
        save_path = f"output/benchmark_{scenario}.png"
    else:
        save_path = None
        
    plot_comparison(config, comparison_data, scenario, save_path)
    
    # Prepare final results
    results = {
        'scenario': scenario,
        'config': config,
        'comparison_data': comparison_data,
        'trad_results': trad_results,
        'hyper_results': hyper_results
    }
    
    return results

def main():
    parser = argparse.ArgumentParser(
        description='Benchmark Comparison: Hyperreal Framework vs Traditional Methods',
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        '--scenario', type=str,
        choices=['collapse', 'expansion', 'binary_collapse', 'multi_scale'],
        default='binary_collapse',
        help='Type of scenario to benchmark'
    )
    
    parser.add_argument(
        '--no-save', action='store_true',
        help='Do not save results to files'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Benchmark Comparison: Hyperreal Framework vs Traditional Methods")
    print("=" * 70)
    
    # Run benchmark
    results = run_benchmark(args.scenario, not args.no_save)
    
    # Print summary
    comp_data = results['comparison_data']
    print("\nBenchmark Summary:")
    print(f"Scenario: {args.scenario}")
    print(f"Traditional method: {'Successful' if comp_data['trad_success'] else 'Failed'}")
    print(f"  - Time: {comp_data['trad_elapsed']:.4f} seconds")
    print(f"Hyperreal Framework: {'Successful' if comp_data['hyper_success'] else 'Failed'}")
    print(f"  - Time: {comp_data['hyper_elapsed']:.4f} seconds")
    
    if comp_data['trad_success'] and comp_data['hyper_success']:
        print(f"Performance: Hyperreal is {comp_data['speedup']:.2f}x faster")
        print(f"Accuracy: RMS position difference = {comp_data['rms_position_difference']:.6f}")
    
    print("\n" + "=" * 70)
    print("Benchmark completed successfully!")
    print("=" * 70)

if __name__ == "__main__":
    main() 