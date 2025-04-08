"""
Test script to generate benchmark plots with dummy data
without having to rerun the simulations.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from configuration_space import Configuration, Point

def create_dummy_config():
    """Create a dummy configuration similar to the extreme collapse config."""
    # Create system with direct collision trajectories
    points = [
        # Two massive objects on direct collision course with extremely high velocities
        Point([0.0001, 0.0], weight=1.0, properties={"name": "Massive Body 1", "velocity": [-20.0, 0.0]}),
        Point([-0.0001, 0.0], weight=1.0, properties={"name": "Massive Body 2", "velocity": [20.0, 0.0]}),
        Point([0.0, 0.0001], weight=0.8, properties={"name": "Massive Body 3", "velocity": [0.0, -20.0]}),
    ]
    
    # Add some random smaller particles
    np.random.seed(42)
    for i in range(40):
        radius = np.random.uniform(1e-9, 1e-5)
        angle = np.random.uniform(0, 2*np.pi)
        mass = np.random.uniform(1e-7, 1e-3)
        
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
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
    
    return Configuration(points)

def create_dummy_trad_results(softening, elapsed_time, success=True):
    """Create dummy traditional method results."""
    t_max = 0.005
    num_steps = 1000
    
    # Create time points
    t = np.linspace(0, t_max, num_steps)
    
    # Create dummy scale values
    if success:
        # Initial spike followed by relatively constant scale
        scales = np.ones(num_steps) * 2.65e-5
        scales[0:5] = np.logspace(-2, np.log10(2.65e-5), 5)
        
        # Add some random noise to make it look realistic
        scales += np.random.normal(0, 1e-6, num_steps)
        
        # Make sure scales are positive
        scales = np.abs(scales)
        
        # Create dummy positions (not actually used in the plot)
        positions = np.zeros((num_steps, 43, 2))
        
        return {
            't': t,
            'positions': positions,
            'scales': scales,
            'elapsed_time': elapsed_time,
            'success': success,
            'has_failure': not success,
            'message': "Success" if success else "Failed with numerical instability",
            'softening': softening
        }
    else:
        # Return failed result
        return {
            't': np.array([]),
            'positions': np.array([]),
            'scales': np.array([]),
            'elapsed_time': elapsed_time,
            'success': success,
            'has_failure': not success,
            'message': "Failed with numerical instability",
            'softening': softening
        }

def create_dummy_hyperreal_results(elapsed_time):
    """Create dummy hyperreal method results."""
    tau_max = 50.0
    num_steps = 1000
    
    # Create time points up to t_max=20
    t = np.linspace(0, 20.0, num_steps)
    
    # Create sigma values (log of scales)
    sigma = np.ones(num_steps) * np.log(1e-4)
    
    # Add some slight variations to make it look realistic
    sigma += np.random.normal(0, 0.01, num_steps)
    
    # Create trajectory dictionary
    trajectory = {
        't': t,
        'sigma': sigma
    }
    
    return {
        'trajectory': trajectory,
        'final_config': None,
        'elapsed_time': elapsed_time,
        'min_scale': 7.91e-5,
        'max_scale': 9.58e-5,
        'scale_ratio': 1.21,
        'has_failure': False,
        'success': True
    }

def extract_accuracy_metrics(trad_results_list, hyper_results):
    """
    Create dummy accuracy metrics based on typical values
    """
    return {
        'valid_comparison': True,
        'time_difference': 1.9687e+01,
        'common_time': 5.0000e-03,
        'trad_energy_variation': 2.0017e+00,
        'hyper_energy_variation': 8.8881e-01
    }

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
    plt.rcParams.update({'font.size': 12})
    fig = plt.figure(figsize=(12, 16))  # Taller figure to allow for proper spacing
    
    # Create a grid layout with fixed row heights
    gs = plt.GridSpec(4, 1, figure=fig, height_ratios=[1, 3, 1.5, 0.1])
    
    # Plot 1: Scale evolution with log scale (top row)
    ax1 = fig.add_subplot(gs[0, 0])
    
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
        ax1.plot(trajectory['t'], np.exp(trajectory['sigma']), 'r-', linewidth=2, label='Hyperreal Framework')
    
    ax1.set_title('Scale Evolution', fontsize=14)
    ax1.set_xlabel('Physical Time', fontsize=12)
    ax1.set_ylabel('Scale Factor', fontsize=12)
    ax1.set_yscale('log')
    
    # Improve legend positioning
    legend = ax1.legend(
        loc='upper right', 
        frameon=True, 
        facecolor='white', 
        framealpha=0.9, 
        fontsize=10
    )
    legend.get_frame().set_linewidth(0.5)
    
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.tick_params(labelsize=10)
    ax1.set_ylim(1e-5, 1e-1)
    
    # Plot 2: Initial configuration - EXACTLY as shown in the blue box
    ax2 = fig.add_subplot(gs[1, 0])  # Take up the middle row, full width
    positions = np.array([point.position for point in config.points])
    weights = np.array([point.weight for point in config.points])
    
    # Scale marker sizes by weight
    sizes = 300 + 3000 * (weights/np.max(weights))**0.5
    colors = plt.cm.viridis(weights/np.max(weights))
    
    # Create main scatter plot with larger points
    scatter = ax2.scatter(positions[:, 0], positions[:, 1], s=sizes, c=colors, alpha=0.8)
    
    # Set exact limits to match reference image
    ax2.set_xlim(-0.0004, 0.0004)
    ax2.set_ylim(-0.0004, 0.0004)
    
    # Create inset axes for detail
    axins = ax2.inset_axes([0.68, 0.55, 0.3, 0.3])
    axins.scatter(positions[:, 0], positions[:, 1], s=sizes*0.6, c=colors, alpha=0.8)
    
    # Set inset zoom limits
    zoom_limit = 5e-5
    axins.set_xlim(-zoom_limit, zoom_limit)
    axins.set_ylim(-zoom_limit, zoom_limit)
    
    # Add grid to inset
    axins.grid(True, linestyle=':', alpha=0.6)
    axins.tick_params(labelsize=8)
    
    # Draw connection patch
    rect = plt.Rectangle((-zoom_limit, -zoom_limit), 2*zoom_limit, 2*zoom_limit, 
                         fill=False, edgecolor='gray', linestyle='-', linewidth=1)
    ax2.add_patch(rect)
    
    # Add connecting line
    con_box = plt.matplotlib.patches.ConnectionPatch(
        xyA=(-zoom_limit, -zoom_limit), xyB=(rect.get_x(), rect.get_y()),
        coordsA="data", coordsB="data", axesA=axins, axesB=ax2,
        color='gray', linestyle='--', alpha=0.4, linewidth=0.5
    )
    ax2.add_artist(con_box)
    
    # Add labels for key bodies
    labeled = {}
    for i, point in enumerate(config.points):
        if i < 3:
            name = point.properties.get("name", f"Body {i}")
            pos = positions[i]
            too_close = False
            for prev_pos in labeled.values():
                dist = np.sqrt((pos[0] - prev_pos[0])**2 + (pos[1] - prev_pos[1])**2)
                if dist < 0.00008:
                    too_close = True
                    break
                    
            if not too_close:
                ax2.annotate(
                    name, 
                    (pos[0], pos[1]), 
                    fontsize=10,
                    xytext=(10, 10),
                    textcoords='offset points',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9)
                )
                labeled[name] = pos
    
    ax2.set_title('Initial Configuration', fontsize=14)
    ax2.set_xlabel('X Position', fontsize=12)
    ax2.set_ylabel('Y Position', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_aspect('equal')
    ax2.tick_params(labelsize=10)
    
    # Add color bar
    cbar = plt.colorbar(scatter, ax=ax2, pad=0.01, fraction=0.05)
    cbar.set_label('Relative Mass', fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    
    # Add some additional particles for visual complexity
    np.random.seed(43)
    for _ in range(30):
        r = np.random.uniform(0.0001, 0.0003)
        theta = np.random.uniform(0, 2*np.pi)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        ax2.scatter(x, y, s=30, color='lightgray', alpha=0.6, zorder=1)
    
    # Bottom section with 2 plots side by side
    gs_bottom = gs[2, 0].subgridspec(1, 2)
    
    # Plot 3: Computational performance comparison (bottom left)
    ax3 = fig.add_subplot(gs_bottom[0, 0])
    
    # Prepare data for bar chart
    labels = []
    times = []
    success = []
    
    for i, trad_results in enumerate(trad_results_list):
        softening = trad_results.get('softening', 'unknown')
        labels.append(f"s={softening:.0e}")
        times.append(trad_results['elapsed_time'])
        success.append(trad_results['success'])
    
    labels.append("Hyperreal")
    times.append(hyper_results['elapsed_time'])
    success.append(hyper_results['success'])
    
    # Create bars
    colors = [plt.cm.Greens(0.7) if s else plt.cm.Reds(0.7) for s in success]
    bars = ax3.bar(labels, times, color=colors, width=0.7)
    
    # Add text labels
    max_time = max(times) if times else 1
    
    for i, (bar, time, s) in enumerate(zip(bars, times, success)):
        status = "✓" if s else "✗"
        if time > max_time * 0.08:
            y_pos = time * 0.5
            ax3.text(
                bar.get_x() + bar.get_width()/2, 
                y_pos, 
                f"{time:.2f}s\n{status}", 
                ha='center', 
                va='center',
                fontsize=8,
                color='white' if time > max_time * 0.3 else 'black',
                fontweight='bold'
            )
        else:
            ax3.text(
                bar.get_x() + bar.get_width()/2, 
                time + max_time * 0.02, 
                f"{time:.2f}s {status}", 
                ha='center', 
                fontsize=8,
                color='black'
            )
    
    ax3.set_title('Computation Time Comparison', fontsize=14)
    ax3.set_ylabel('Time (seconds)', fontsize=12)
    ax3.tick_params(axis='x', rotation=45, labelsize=8)
    ax3.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add a horizontal line for hyperreal time
    if hyper_results['success']:
        ax3.axhline(
            y=hyper_results['elapsed_time'], 
            color='r', 
            linestyle='--', 
            alpha=0.7, 
            label=f"Hyperreal: {hyper_results['elapsed_time']:.2f}s"
        )
        ax3.legend(fontsize=8, loc='upper right')
    
    # Plot 4: Energy variation comparison (bottom right)
    ax5 = fig.add_subplot(gs_bottom[0, 1])
    
    if accuracy_metrics.get('valid_comparison', False):
        # Create data for comparison visualization
        method_labels = ['Traditional', 'Hyperreal']
        
        if accuracy_metrics['trad_energy_variation'] is not None and accuracy_metrics['hyper_energy_variation'] is not None:
            # Energy variation (lower is better)
            energy_variations = [
                accuracy_metrics['trad_energy_variation'],
                accuracy_metrics['hyper_energy_variation']
            ]
            
            # Normalize for visualization
            max_variation = max(energy_variations)
            if max_variation > 0:
                normalized_variations = [v/max_variation for v in energy_variations]
                colors = [plt.cm.RdYlGn(1.0 - v) for v in normalized_variations]
            else:
                normalized_variations = [0, 0]
                colors = [plt.cm.RdYlGn(1.0), plt.cm.RdYlGn(1.0)]
            
            # Plot bars
            bars = ax5.bar(method_labels, normalized_variations, color=colors, width=0.35)
            
            # Add text with values
            for i, (bar, value) in enumerate(zip(bars, energy_variations)):
                text_y = min(normalized_variations[i] + 0.05, 0.95)
                text_box = dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
                ax5.text(
                    bar.get_x() + bar.get_width()/2, 
                    text_y, 
                    f"{value:.2e}", 
                    ha='center', 
                    fontsize=10,
                    bbox=text_box
                )
            
            ax5.set_title('Energy Variation Comparison\n(Lower is Better)', fontsize=14)
            ax5.set_ylabel('Normalized Variation', fontsize=12)
            ax5.grid(axis='y', linestyle='--', alpha=0.7)
            ax5.tick_params(labelsize=10)
            
            # Add percentage improvement
            if energy_variations[0] > 0 and energy_variations[1] > 0:
                percent_improvement = (energy_variations[0] - energy_variations[1]) / energy_variations[0] * 100
                improvement_text = f"Hyperreal shows {percent_improvement:.1f}% improvement\nin energy conservation"
                
                imp_box = dict(boxstyle="round,pad=0.5", fc="lightyellow", ec="orange", alpha=0.9)
                ax5.text(
                    0.5, 0.3,
                    improvement_text, 
                    ha='center', 
                    va='center', 
                    fontsize=10,
                    transform=ax5.transAxes,
                    bbox=imp_box,
                    zorder=10
                )
        else:
            ax5.text(
                0.5, 0.5, 
                "Insufficient data for energy comparison", 
                ha='center', 
                va='center', 
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.5", fc="whitesmoke", ec="lightgray", alpha=0.9)
            )
    else:
        ax5.text(
            0.5, 0.5, 
            "No valid comparison available", 
            ha='center', 
            va='center', 
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", fc="whitesmoke", ec="lightgray", alpha=0.9)
        )
    
    # Add benchmark results table at the bottom
    # Create summary data
    summary_data = []
    for i, trad_results in enumerate(trad_results_list):
        softening = trad_results.get('softening', 'unknown')
        status = "Successful" if trad_results['success'] else "Failed"
        time = trad_results['elapsed_time']
        
        if trad_results['success'] and len(trad_results['scales']) > 0:
            min_scale = np.min(trad_results['scales'])
            max_scale = np.max(trad_results['scales'])
            scale_ratio = max_scale / min_scale if min_scale > 0 else float('inf')
            
            summary_data.append({
                'method': f"Traditional (s={softening:.0e})",
                'status': status,
                'time': f"{time:.2f}s",
                'min_scale': f"{min_scale:.2e}",
                'max_scale': f"{max_scale:.2e}",
                'scale_ratio': f"{scale_ratio:.2e}"
            })
        else:
            summary_data.append({
                'method': f"Traditional (s={softening:.0e})",
                'status': status,
                'time': f"{time:.2f}s",
                'min_scale': "N/A",
                'max_scale': "N/A",
                'scale_ratio': "N/A"
            })
    
    # Add hyperreal results
    if hyper_results['success']:
        summary_data.append({
            'method': "Hyperreal Framework",
            'status': "Successful",
            'time': f"{hyper_results['elapsed_time']:.2f}s",
            'min_scale': f"{hyper_results['min_scale']:.2e}",
            'max_scale': f"{hyper_results['max_scale']:.2e}",
            'scale_ratio': f"{hyper_results['scale_ratio']:.2e}"
        })
    
    # Create table format
    col_widths = {
        'method': max(len(item['method']) for item in summary_data) + 2,
        'status': 12,
        'time': 10,
        'min_scale': 14,
        'max_scale': 14,
        'scale_ratio': 14
    }
    
    header = (
        f"{'Method':<{col_widths['method']}}"
        f"{'Status':<{col_widths['status']}}"
        f"{'Time':<{col_widths['time']}}"
        f"{'Min Scale':<{col_widths['min_scale']}}"
        f"{'Max Scale':<{col_widths['max_scale']}}"
        f"{'Scale Ratio':<{col_widths['scale_ratio']}}"
    )
    
    separator = "-" * (sum(col_widths.values()))
    
    table_lines = ["Benchmark Results", "=" * 40, "", header, separator]
    
    for item in summary_data:
        line = (
            f"{item['method']:<{col_widths['method']}}"
            f"{item['status']:<{col_widths['status']}}"
            f"{item['time']:<{col_widths['time']}}"
            f"{item['min_scale']:<{col_widths['min_scale']}}"
            f"{item['max_scale']:<{col_widths['max_scale']}}"
            f"{item['scale_ratio']:<{col_widths['scale_ratio']}}"
        )
        table_lines.append(line)
    
    # Add accuracy metrics if available
    if accuracy_metrics.get('valid_comparison', False):
        table_lines.extend([
            "",
            separator,
            "Accuracy Comparison:",
            f"  Time difference: {accuracy_metrics['time_difference']:.4e}",
            f"  Common time point: {accuracy_metrics['common_time']:.4e}",
            f"  Traditional energy variation: {accuracy_metrics['trad_energy_variation']:.4e}",
            f"  Hyperreal energy variation: {accuracy_metrics['hyper_energy_variation']:.4e}"
        ])
        
        # Add speedup calculation
        if hyper_results['success']:
            best_trad_time = min([r['elapsed_time'] for r in trad_results_list if r['success']], default=None)
            if best_trad_time:
                speedup = best_trad_time / hyper_results['elapsed_time']
                table_lines.append(f"\n  Hyperreal speedup factor: {speedup:.2f}x")
    
    summary_text = "\n".join(table_lines)
    
    # Add table in its own axis at the bottom
    ax4 = fig.add_subplot(gs[3, 0])
    ax4.axis('off')
    
    # Display the table with proper formatting
    props = dict(boxstyle='round', facecolor='whitesmoke', alpha=0.95, pad=1.0, edgecolor='lightgray')
    ax4.text(
        0.5, 0.5, 
        summary_text, 
        ha='center', 
        va='center', 
        fontsize=8,
        linespacing=1.5,
        family='monospace',
        bbox=props,
        transform=ax4.transAxes
    )
    
    # Add overall title
    fig.suptitle('Ultra-Extreme Collapse Benchmark: Hyperreal vs Traditional Methods', fontsize=16, y=0.98)
    
    # Layout adjustments
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    fig.subplots_adjust(hspace=0.4)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Results plot saved to {save_path}")

def main():
    """Main function to generate test plots."""
    print("Generating test plots with dummy data...")
    
    # Create dummy config
    config = create_dummy_config()
    
    # Create dummy traditional results
    trad_results_list = [
        create_dummy_trad_results(1e-3, 1.8079),
        create_dummy_trad_results(1e-4, 3.1859),
        create_dummy_trad_results(1e-5, 23.7719),
        create_dummy_trad_results(1e-6, 195.0552),
        create_dummy_trad_results(1e-7, 1231.4383)
    ]
    
    # Create dummy hyperreal results
    hyper_results = create_dummy_hyperreal_results(0.0845)
    
    # Plot results
    plot_results(trad_results_list, hyper_results, config, 'output/test_plot_format.png')
    
    print("Test plot generated successfully!")

if __name__ == "__main__":
    main() 