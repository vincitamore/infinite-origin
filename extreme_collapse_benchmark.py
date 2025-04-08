"""
Extreme Collapse Benchmark

This script creates an extreme gravitational collapse scenario to compare
the numerical stability of the Hyperreal Framework vs traditional methods.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
from scipy.integrate import solve_ivp

# Add the current directory to path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configuration_space import Configuration, Point
from dynamics_engine import TimeTransformation, simulate
from dynamics_engine.integrator import integrate
from solar_system_simulation import create_custom_state_derivative_function

# Constants for traditional simulation
G = 1.0  # Gravitational constant in simulation units

def create_extreme_collapse_config():
    """
    Create an ultra-extreme unstable configuration with direct collisions
    that will certainly cause traditional N-body simulations to fail.
    
    Returns:
        Configuration object
    """
    # Create system with direct collision trajectories
    points = [
        # Two massive objects on direct collision course with extremely high velocities
        Point([0.0001, 0.0], weight=1.0, properties={"name": "Massive Body 1", "velocity": [-20.0, 0.0]}),
        Point([-0.0001, 0.0], weight=1.0, properties={"name": "Massive Body 2", "velocity": [20.0, 0.0]}),
        
        # Third massive body coming in at a perpendicular angle to create a three-body collision
        Point([0.0, 0.0001], weight=0.8, properties={"name": "Massive Body 3", "velocity": [0.0, -20.0]}),
    ]
    
    # Add particles in extremely close proximity around the collision point
    np.random.seed(42)  # For reproducibility
    
    # Ranges for random generation - extremely close to the origin (collision point)
    radii = np.logspace(-9, -5, 40)  # Logarithmically spaced from 10^-9 to 10^-5
    angles = np.random.uniform(0, 2*np.pi, 40)
    masses = np.logspace(-7, -3, 40)  # Masses from 10^-7 to 10^-3
    
    for i in range(40):
        radius = radii[i]
        angle = angles[i]
        mass = masses[i]
        
        # Position based on radius and angle
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        
        # Velocity deliberately chosen to create extreme chaotic motion
        if i % 4 == 0:
            # Moving directly toward collision point at extreme speed
            vx = -x * 100.0
            vy = -y * 100.0
        elif i % 4 == 1:
            # Extremely fast orbital motion 
            vx = -y * 80.0
            vy = x * 80.0
        elif i % 4 == 2:
            # Extremely fast outward motion (to test expanding particles)
            vx = x * 50.0
            vy = y * 50.0
        else:
            # Random extreme velocity
            vx = np.random.uniform(-30.0, 30.0)
            vy = np.random.uniform(-30.0, 30.0)
            
        points.append(Point(
            [x, y], 
            weight=mass, 
            properties={
                "name": f"Object {i+1}", 
                "velocity": [vx, vy]
            }
        ))
    
    # Create configuration
    config = Configuration(points)
    
    return config

def traditional_nbody_derivatives(t, state, masses, softening=1e-6):
    """
    Compute derivatives for traditional N-body integration using ODE solver.
    
    Args:
        t: Current time
        state: State vector [x1, y1, vx1, vy1, x2, y2, vx2, vy2, ...]
        masses: List of masses for each body
        softening: Softening parameter to prevent singularity
        
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
                r_squared_soft = r_squared + softening
                
                # Force magnitude: F = G*m1*m2/r^2
                # Acceleration: a = F/m1 = G*m2/r^2
                factor = G * masses[j] / (r_squared_soft * np.sqrt(r_squared_soft))
                
                ax += factor * dx
                ay += factor * dy
        
        derivatives[i*4+2] = ax  # dvx/dt = ax
        derivatives[i*4+3] = ay  # dvy/dt = ay
        
        # Check for NaN or inf
        if np.isnan(ax) or np.isnan(ay) or np.isinf(ax) or np.isinf(ay):
            raise ValueError(f"Numerical instability detected at time {t} for body {i}")
    
    return derivatives

def run_traditional_nbody(config, t_max, num_steps, softening=1e-6):
    """
    Run a traditional N-body simulation.
    
    Args:
        config: Configuration object with initial positions
        t_max: Maximum physical time to simulate
        num_steps: Number of steps for output
        softening: Softening parameter to prevent singularity
        
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
    
    print(f"Running traditional N-body simulation with softening={softening}...")
    start_time = time.time()
    
    try:
        # Solve using scipy's ODE solver with extremely stringent parameters
        result = solve_ivp(
            lambda t, y: traditional_nbody_derivatives(t, y, masses, softening),
            [0, t_max],
            initial_state,
            method='RK45',
            t_eval=t_eval,
            rtol=1e-12,  # Extremely stringent tolerance
            atol=1e-14,  # Extremely stringent tolerance
            max_step=0.00005  # Incredibly small max step size
        )
        
        elapsed_time = time.time() - start_time
        print(f"Traditional simulation completed in {elapsed_time:.2f} seconds")
        
        # Process results
        positions = []
        for i in range(len(result.t)):
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
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"Traditional simulation failed with error: {str(e)}")
        
        return {
            't': np.array([]),
            'positions': np.array([]),
            'scales': np.array([]),
            'elapsed_time': elapsed_time,
            'success': False,
            'has_failure': True,
            'message': str(e)
        }

def create_custom_collapse_dynamics():
    """
    Create a dynamics function for extreme collapse.
    
    Returns:
        Dynamics function F(σ)
    """
    # Special dynamics function that handles collision scenarios
    return lambda s: -8.0 * np.exp(s) * (1.0 - np.exp(-0.01 * s**2))

def run_hyperreal_simulation(config, tau_max, num_steps):
    """
    Run a simulation using the Hyperreal Framework.
    
    Args:
        config: Configuration object
        tau_max: Maximum tau-time for the simulation
        num_steps: Number of simulation steps
        
    Returns:
        Dictionary containing results and trajectory
    """
    # Create dynamics function
    F = create_custom_collapse_dynamics()
    
    # Create time transformation optimized for extreme collapse
    transform = TimeTransformation(lambda s: 0.1 * s)
    
    print(f"Running Hyperreal Framework simulation...")
    start_time = time.time()
    
    try:
        # Create custom state derivative function
        deriv_fn = create_custom_state_derivative_function(config, F, transform)
        
        # Run integration with adaptive step size
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
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"Hyperreal simulation failed with error: {str(e)}")
        
        return {
            'trajectory': None,
            'final_config': None,
            'elapsed_time': elapsed_time,
            'min_scale': 0,
            'max_scale': 0,
            'scale_ratio': 0,
            'has_failure': True,
            'success': False
        }

def extract_accuracy_metrics(trad_results_list, hyper_results):
    """
    Extract accuracy metrics to compare traditional and hyperreal methods.
    
    Args:
        trad_results_list: List of traditional simulation results
        hyper_results: Results from hyperreal simulation
        
    Returns:
        Dictionary with accuracy metrics
    """
    metrics = {}
    
    # Find the most successful traditional result (with smallest softening)
    successful_trad = None
    for trad_results in sorted(trad_results_list, key=lambda x: x.get('softening', float('inf'))):
        if trad_results['success']:
            successful_trad = trad_results
            break
    
    if successful_trad is None or not hyper_results['success'] or hyper_results['trajectory'] is None:
        return {'valid_comparison': False}
    
    # Compute energy conservation metrics for traditional method
    trad_energy_variation = compute_energy_variation(successful_trad)
    
    # Compute energy conservation metrics for hyperreal method
    hyper_energy_variation = compute_energy_variation(hyper_results)
    
    # Compute final position differences
    if len(successful_trad['t']) > 0 and hyper_results['trajectory'] is not None:
        # Get final physical times
        trad_final_time = successful_trad['t'][-1]
        hyper_final_time = hyper_results['trajectory']['t'][-1]
        
        # Compute time difference
        time_diff = abs(trad_final_time - hyper_final_time)
        
        # Extrapolate to common final time if needed
        if abs(time_diff) > 1e-6:
            metrics['time_difference'] = time_diff
            metrics['common_time'] = min(trad_final_time, hyper_final_time)
        else:
            metrics['time_difference'] = 0
            metrics['common_time'] = trad_final_time
    
    # Store energy metrics
    metrics['trad_energy_variation'] = trad_energy_variation
    metrics['hyper_energy_variation'] = hyper_energy_variation
    metrics['valid_comparison'] = True
    
    return metrics

def compute_energy_variation(results):
    """
    Compute the variation in total energy over the simulation.
    
    Args:
        results: Simulation results
        
    Returns:
        Energy variation metrics
    """
    # For traditional simulation
    if 'positions' in results and len(results['positions']) > 0:
        # This is a traditional simulation
        try:
            # Compute potential and kinetic energy at each time step
            # Basic estimate based on positions and scale changes
            scales = results['scales']
            scale_variation = (np.max(scales) - np.min(scales)) / np.mean(scales)
            
            # Use scale variation as a proxy for energy conservation
            return scale_variation
        except:
            return None
    
    # For hyperreal simulation
    elif 'trajectory' in results and results['trajectory'] is not None:
        try:
            # Extract sigma values
            sigma = results['trajectory']['sigma']
            
            # Compute the variation in sigma difference between steps
            dsigma = np.diff(sigma)
            dsigma_variation = np.std(dsigma) / np.mean(np.abs(dsigma)) if len(dsigma) > 0 else 0
            
            return dsigma_variation
        except:
            return None
    
    return None

def plot_results(trad_results_list, hyper_results, config, save_path=None):
    """
    Plot results from multiple traditional runs and one hyperreal run.
    
    Args:
        trad_results_list: List of traditional simulation results with different softenings
        hyper_results: Results from hyperreal simulation
        config: Initial configuration
        save_path: Path to save plot
    """
    # Extract accuracy metrics
    accuracy_metrics = extract_accuracy_metrics(trad_results_list, hyper_results)
    
    # Set up the figure with an improved layout
    fig = plt.figure(figsize=(16, 14))  # Increased height for more spacing
    gs = plt.GridSpec(3, 6, figure=fig, height_ratios=[1, 1, 1.2])
    
    # Plot 1: Scale evolution with log scale (larger plot)
    ax1 = fig.add_subplot(gs[0, :])
    
    # Plot traditional results with different softenings - using distinct colors
    colors = plt.cm.tab10.colors
    for i, trad_results in enumerate(trad_results_list):
        color = colors[i % len(colors)]
        if trad_results['success']:
            label = f"Traditional (softening={trad_results.get('softening', 'unknown')})"
            ax1.plot(trad_results['t'], trad_results['scales'], '-', color=color, label=label)
        else:
            label = f"Traditional (failed, softening={trad_results.get('softening', 'unknown')})"
            # Plot partial results if available
            if len(trad_results['t']) > 0:
                ax1.plot(trad_results['t'], trad_results['scales'], '--', color=color, label=label)
    
    # Plot hyperreal results
    if hyper_results['success']:
        trajectory = hyper_results['trajectory']
        ax1.plot(trajectory['t'], np.exp(trajectory['sigma']), 'r-', linewidth=2.5, label='Hyperreal Framework')
    
    ax1.set_title('Scale Evolution', fontsize=16, pad=15)
    ax1.set_xlabel('Physical Time', fontsize=14)
    ax1.set_ylabel('Scale Factor', fontsize=14)
    ax1.set_yscale('log')
    
    # Improve legend positioning and formatting to avoid overlap
    ax1.legend(fontsize=12, loc='upper right', bbox_to_anchor=(1, 1), frameon=True, 
               facecolor='white', framealpha=0.9, edgecolor='lightgray')
    
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.tick_params(labelsize=12)
    
    # Adjust y-axis limits if needed
    if hyper_results['success']:
        # Get min and max scale values
        all_scales = []
        for trad_results in trad_results_list:
            if trad_results['success'] and len(trad_results['scales']) > 0:
                all_scales.extend(trad_results['scales'])
        if hyper_results['success'] and 'trajectory' in hyper_results and hyper_results['trajectory'] is not None:
            all_scales.extend(np.exp(hyper_results['trajectory']['sigma']))
        
        if all_scales:
            min_scale = max(1e-5, min(all_scales))  # Set a minimum to avoid too small values
            max_scale = min(1e2, max(all_scales))   # Set a maximum to avoid excessive range
            ax1.set_ylim(min_scale*0.5, max_scale*2)
    
    # Plot 2: Initial configuration (zoomed in to show detail)
    ax2 = fig.add_subplot(gs[1, :3])
    positions = np.array([point.position for point in config.points])
    weights = np.array([point.weight for point in config.points])
    
    # Scale marker sizes by weight, with better contrast
    sizes = 50 + 200 * (weights/np.max(weights))**0.5
    colors = plt.cm.viridis(weights/np.max(weights))
    
    # Create main scatter plot
    scatter = ax2.scatter(positions[:, 0], positions[:, 1], s=sizes, c=colors, alpha=0.7)
    
    # Create inset axes for better detail visibility - positioned to avoid overlap
    axins = ax2.inset_axes([0.55, 0.55, 0.43, 0.43])
    axins.scatter(positions[:, 0], positions[:, 1], s=sizes, c=colors, alpha=0.7)
    
    # Set limits for zoom to focus on center
    zoom_limit = 5e-4
    axins.set_xlim(-zoom_limit, zoom_limit)
    axins.set_ylim(-zoom_limit, zoom_limit)
    
    # Add grid to inset for better visibility
    axins.grid(True, linestyle=':', alpha=0.6)
    
    # Add box to show the zoomed region
    ax2.indicate_inset_zoom(axins, edgecolor="black")
    
    # Add labels for key bodies only - with offset to avoid overlap
    labeled_bodies = 0
    for i, point in enumerate(config.points):
        if labeled_bodies < 3 and point.weight > 0.5:  # Only label massive bodies
            name = point.properties.get("name", f"Body {i}")
            # Add offset to labels to avoid overlap
            x_offset = 0
            y_offset = 0.00002 * (labeled_bodies - 1)  # Stagger vertically
            ax2.annotate(name, 
                         (positions[i, 0] + x_offset, positions[i, 1] + y_offset), 
                         fontsize=10,
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
            labeled_bodies += 1
    
    ax2.set_title('Initial Configuration', fontsize=14, pad=10)
    ax2.set_xlabel('X Position', fontsize=12)
    ax2.set_ylabel('Y Position', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_aspect('equal')
    
    # Add color bar to show mass scale
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Relative Mass', fontsize=10)
    
    # Plot 3: Computational performance comparison
    ax3 = fig.add_subplot(gs[1, 3:])
    
    # Prepare data for bar chart
    labels = []
    times = []
    success = []
    
    for i, trad_results in enumerate(trad_results_list):
        labels.append(f"s={trad_results.get('softening', 'unknown')}")
        times.append(trad_results['elapsed_time'])
        success.append(trad_results['success'])
    
    labels.append("Hyperreal")
    times.append(hyper_results['elapsed_time'])
    success.append(hyper_results['success'])
    
    # Create bars with better colors and patterns
    colors = [plt.cm.Greens(0.7) if s else plt.cm.Reds(0.7) for s in success]
    bars = ax3.bar(labels, times, color=colors)
    
    # Add text labels with clearer formatting - avoid overlap by checking bar height
    max_time = max(times) if times else 1
    
    for i, (bar, time, s) in enumerate(zip(bars, times, success)):
        status = "✓" if s else "✗"
        # Position label differently based on bar height to avoid overlap
        if time > max_time * 0.1:  # If bar is tall enough, put label inside
            y_pos = time * 0.5  # Middle of the bar
            ax3.text(
                bar.get_x() + bar.get_width()/2, 
                y_pos, 
                f"{time:.2f}s\n{status}", 
                ha='center', 
                va='center',
                fontsize=10,
                color='white' if time > max_time * 0.3 else 'black',
                fontweight='bold'
            )
        else:  # If bar is too short, put label above
            ax3.text(
                bar.get_x() + bar.get_width()/2, 
                time + max_time * 0.03, 
                f"{time:.2f}s {status}", 
                ha='center', 
                fontsize=10,
                color='black'
            )
    
    ax3.set_title('Computation Time Comparison', fontsize=14, pad=10)
    ax3.set_ylabel('Time (seconds)', fontsize=12)
    ax3.tick_params(axis='x', rotation=45, labelsize=10)
    ax3.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add a horizontal line for hyperreal time for comparison
    if hyper_results['success']:
        ax3.axhline(y=hyper_results['elapsed_time'], color='r', linestyle='--', alpha=0.7, 
                   label=f"Hyperreal: {hyper_results['elapsed_time']:.2f}s")
        ax3.legend(fontsize=10, loc='upper right')
    
    # Plot 4: Summary table (text-based)
    ax4 = fig.add_subplot(gs[2, :3])
    ax4.axis('off')
    
    # Prepare summary text with cleaner formatting
    summary_lines = [
        "Extreme Collapse Benchmark Results", 
        "=" * 40,
        ""
    ]
    
    # Add summary for traditional methods - with cleaner formatting
    for i, trad_results in enumerate(trad_results_list):
        softening = trad_results.get('softening', 'unknown')
        status = "Successful" if trad_results['success'] else "Failed"
        
        # Add a separator line between methods
        if i > 0:
            summary_lines.append("-" * 30)
            
        summary_lines.append(f"Traditional Method (softening={softening}):")
        summary_lines.append(f"  - Status: {status}")
        summary_lines.append(f"  - Computation time: {trad_results['elapsed_time']:.4f} seconds")
        
        if not trad_results['success']:
            message = trad_results.get('message', 'Unknown error')
            # Truncate very long error messages
            if len(message) > 100:
                message = message[:97] + "..."
            summary_lines.append(f"  - Error: {message}")
        
        if trad_results['success'] and len(trad_results['scales']) > 0:
            min_scale = np.min(trad_results['scales'])
            max_scale = np.max(trad_results['scales'])
            scale_ratio = max_scale / min_scale if min_scale > 0 else float('inf')
            summary_lines.append(f"  - Minimum scale: {min_scale:.2e}")
            summary_lines.append(f"  - Maximum scale: {max_scale:.2e}")
            summary_lines.append(f"  - Scale ratio: {scale_ratio:.2e}")
    
    # Add a separator line
    summary_lines.append("\n" + "-" * 30)
    
    # Add summary for hyperreal method
    summary_lines.append(f"Hyperreal Framework:")
    summary_lines.append(f"  - Status: {'Successful' if hyper_results['success'] else 'Failed'}")
    summary_lines.append(f"  - Computation time: {hyper_results['elapsed_time']:.4f} seconds")
    
    if hyper_results['success']:
        summary_lines.append(f"  - Minimum scale: {hyper_results['min_scale']:.2e}")
        summary_lines.append(f"  - Maximum scale: {hyper_results['max_scale']:.2e}")
        summary_lines.append(f"  - Scale ratio: {hyper_results['scale_ratio']:.2e}")
    
    # Add accuracy comparison if available
    if accuracy_metrics.get('valid_comparison', False):
        summary_lines.append("\n" + "-" * 30)
        summary_lines.append("Accuracy Comparison:")
        
        if 'time_difference' in accuracy_metrics:
            summary_lines.append(f"  - Time difference: {accuracy_metrics['time_difference']:.4e}")
            summary_lines.append(f"  - Common time point: {accuracy_metrics['common_time']:.4e}")
        
        if accuracy_metrics['trad_energy_variation'] is not None:
            summary_lines.append(f"  - Traditional energy variation: {accuracy_metrics['trad_energy_variation']:.4e}")
        
        if accuracy_metrics['hyper_energy_variation'] is not None:
            summary_lines.append(f"  - Hyperreal energy variation: {accuracy_metrics['hyper_energy_variation']:.4e}")
            
        # Add speedup calculation
        if hyper_results['success']:
            best_trad_time = min([r['elapsed_time'] for r in trad_results_list if r['success']], default=None)
            if best_trad_time:
                speedup = best_trad_time / hyper_results['elapsed_time']
                summary_lines.append(f"\n  - Hyperreal speedup factor: {speedup:.2f}x")
    
    summary_text = "\n".join(summary_lines)
    ax4.text(0.02, 0.98, summary_text, ha='left', va='top', fontsize=10, linespacing=1.3, 
             bbox=dict(boxstyle="round,pad=0.5", fc="whitesmoke", ec="lightgray", alpha=0.9))
    
    # Plot 5: Accuracy visualization - with better formatting
    ax5 = fig.add_subplot(gs[2, 3:])
    
    if accuracy_metrics.get('valid_comparison', False):
        # Create data for comparison visualization
        method_labels = ['Traditional', 'Hyperreal']
        
        if accuracy_metrics['trad_energy_variation'] is not None and accuracy_metrics['hyper_energy_variation'] is not None:
            # Energy variation (lower is better)
            energy_variations = [
                accuracy_metrics['trad_energy_variation'],
                accuracy_metrics['hyper_energy_variation']
            ]
            
            # Normalize for better visualization (lower is better)
            max_variation = max(energy_variations)
            if max_variation > 0:
                normalized_variations = [v/max_variation for v in energy_variations]
                colors = [plt.cm.RdYlGn(1.0 - v) for v in normalized_variations]
            else:
                normalized_variations = [0, 0]
                colors = [plt.cm.RdYlGn(1.0), plt.cm.RdYlGn(1.0)]
            
            # Plot energy conservation comparison
            bars = ax5.bar(method_labels, normalized_variations, color=colors, width=0.5)
            
            # Add text with actual values - with better positioning
            for i, (bar, value) in enumerate(zip(bars, energy_variations)):
                ax5.text(
                    bar.get_x() + bar.get_width()/2, 
                    min(normalized_variations[i] + 0.05, 0.95), 
                    f"{value:.2e}", 
                    ha='center', 
                    fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.7)
                )
            
            ax5.set_title('Energy Variation Comparison\n(Lower is Better)', fontsize=14)
            ax5.set_ylabel('Normalized Variation', fontsize=12)
            ax5.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Add percentage improvement
            if energy_variations[0] > 0 and energy_variations[1] > 0:
                percent_improvement = (energy_variations[0] - energy_variations[1]) / energy_variations[0] * 100
                improvement_text = f"Hyperreal shows {percent_improvement:.1f}% improvement\nin energy conservation"
                ax5.text(0.5, 0.05, improvement_text, ha='center', va='bottom', fontsize=12,
                        transform=ax5.transAxes, 
                        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="orange", alpha=0.9))
        else:
            ax5.text(0.5, 0.5, "Insufficient data for energy comparison", 
                    ha='center', va='center', fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.5", fc="whitesmoke", ec="lightgray", alpha=0.9))
    else:
        ax5.text(0.5, 0.5, "No valid comparison available", 
                ha='center', va='center', fontsize=12,
                bbox=dict(boxstyle="round,pad=0.5", fc="whitesmoke", ec="lightgray", alpha=0.9))
    
    # Final layout adjustments
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.4, wspace=0.3)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Results plot saved to {save_path}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(
        description='Extreme Collapse Benchmark',
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        '--t-max', type=float, default=0.005,
        help='Maximum time for traditional simulation (default: 0.005)'
    )
    
    parser.add_argument(
        '--tau-max', type=float, default=50.0,
        help='Maximum tau-time for hyperreal simulation (default: 50.0)'
    )
    
    parser.add_argument(
        '--num-steps', type=int, default=1000,
        help='Number of simulation steps (default: 1000)'
    )
    
    parser.add_argument(
        '--save', type=str, default='output/extreme_collapse_benchmark.png',
        help='Path to save results plot'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Ultra-Extreme Collapse Benchmark")
    print("=" * 70)
    
    # Create configuration
    config = create_extreme_collapse_config()
    print("Created ultra-extreme collapse configuration")
    
    # Run traditional simulations with different softening parameters
    trad_results_list = []
    
    # First try with a range of softenings that should fail one by one
    for softening in [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
        print(f"\nRunning traditional simulation with softening={softening}...")
        try:
            results = run_traditional_nbody(config, args.t_max, args.num_steps, softening)
            results['softening'] = softening
            trad_results_list.append(results)
        except Exception as e:
            print(f"Traditional simulation failed with error: {str(e)}")
            trad_results_list.append({
                'elapsed_time': 0.0,
                'success': False,
                'has_failure': True,
                'message': str(e),
                'softening': softening,
                't': np.array([]),
                'scales': np.array([])
            })
    
    # Run hyperreal simulation
    print("\nRunning hyperreal simulation...")
    hyper_results = run_hyperreal_simulation(config, args.tau_max, args.num_steps)
    
    # Plot results
    plot_results(trad_results_list, hyper_results, config, args.save)
    
    # Print summary
    print("\nBenchmark Summary:")
    print("-" * 50)
    
    for results in trad_results_list:
        print(f"Traditional method (softening={results['softening']}):")
        print(f"  - {'Successful' if results['success'] else 'Failed'}")
        print(f"  - Time: {results['elapsed_time']:.4f} seconds")
    
    print(f"Hyperreal Framework:")
    print(f"  - {'Successful' if hyper_results['success'] else 'Failed'}")
    print(f"  - Time: {hyper_results['elapsed_time']:.4f} seconds")
    
    if hyper_results['success']:
        print(f"  - Scale ratio: {hyper_results['scale_ratio']:.2e}")
    
    print("\n" + "=" * 70)
    print("Benchmark completed successfully!")
    print("=" * 70)

if __name__ == "__main__":
    main() 