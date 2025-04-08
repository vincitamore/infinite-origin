"""
Simulation Analysis Tool

This script analyzes the results of solar system simulations performed with
the Infinite Origin Framework, focusing on multi-scale dynamics.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import argparse
import json

# Add the current directory to path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def load_trajectory_data(scenario='standard'):
    """
    Load trajectory data from a simulation output file.
    
    Args:
        scenario: Name of the scenario to analyze
        
    Returns:
        Dictionary containing trajectory data
    """
    # For now, simulate loading from a file
    # In a real implementation, we would load from saved data
    # Instead, we'll run a quick simulation
    
    from solar_system_simulation import run_solar_system_simulation
    
    print(f"Running {scenario} simulation to gather data...")
    _, trajectory = run_solar_system_simulation(
        include_bodies='major',
        tau_max=15.0,
        num_steps=500,
        use_velocities=True,
        scenario=scenario
    )
    
    return trajectory

def calculate_scale_metrics(trajectory):
    """
    Calculate metrics related to scale handling in the simulation.
    
    Args:
        trajectory: Dictionary containing trajectory data
        
    Returns:
        Dictionary of scale metrics
    """
    # Extract data
    sigma_values = trajectory['sigma']
    t_values = trajectory['t']
    
    # Get scale range (min and max e^σ values)
    scale_min = np.exp(np.min(sigma_values))
    scale_max = np.exp(np.max(sigma_values))
    scale_ratio = scale_max / scale_min if scale_min > 0 else float('inf')
    
    # Scale evolution rate
    sigma_diff = np.diff(sigma_values)
    tau_diff = np.diff(trajectory['tau'])
    max_rate = np.max(np.abs(sigma_diff / tau_diff)) if len(tau_diff) > 0 else 0
    
    # Time transformation effectiveness
    # Compare physical time progression to τ-time
    time_ratio = t_values[-1] / trajectory['tau'][-1] if trajectory['tau'][-1] > 0 else 0
    
    # Identify extreme episodes
    # Times when scale change exceeds a threshold
    threshold = 0.1  # Adjust based on your needs
    extreme_episodes = np.where(np.abs(sigma_diff) > threshold)[0]
    
    return {
        'scale_min': scale_min,
        'scale_max': scale_max,
        'scale_ratio': scale_ratio,
        'max_rate': max_rate,
        'time_ratio': time_ratio,
        'extreme_episodes': extreme_episodes,
        'extreme_count': len(extreme_episodes)
    }

def plot_scale_analysis(trajectory, metrics, scenario='standard', save_path=None, show_plot=False):
    """
    Create plots showing scale analysis.
    
    Args:
        trajectory: Dictionary containing trajectory data
        metrics: Dictionary of scale metrics
        scenario: Name of the scenario
        save_path: Path to save the plot
        show_plot: Whether to display the plot interactively
    """
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(3, 2, figure=fig)
    
    # Plot 1: Sigma vs Tau
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(trajectory['tau'], trajectory['sigma'], 'b-')
    ax1.set_title('Scale Evolution (σ vs τ)')
    ax1.set_xlabel('τ-time')
    ax1.set_ylabel('σ (log scale)')
    
    # Highlight extreme episodes
    if len(metrics['extreme_episodes']) > 0:
        extreme_tau = [trajectory['tau'][i] for i in metrics['extreme_episodes']]
        extreme_sigma = [trajectory['sigma'][i] for i in metrics['extreme_episodes']]
        ax1.plot(extreme_tau, extreme_sigma, 'ro', label='Extreme scale changes')
        ax1.legend()
    
    # Plot 2: Physical time vs Tau
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(trajectory['tau'], trajectory['t'], 'g-')
    ax2.set_title('Physical Time vs τ')
    ax2.set_xlabel('τ-time')
    ax2.set_ylabel('Physical time')
    
    # Plot 3: Scale factor vs Physical time
    ax3 = fig.add_subplot(gs[1, 0])
    scale_factor = np.exp(trajectory['sigma'])
    ax3.plot(trajectory['t'], scale_factor, 'r-')
    ax3.set_title('Scale Factor vs Physical Time')
    ax3.set_xlabel('Physical time')
    ax3.set_ylabel('Scale factor (e^σ)')
    
    # Plot 4: Rate of scale change
    ax4 = fig.add_subplot(gs[1, 1])
    sigma_diff = np.diff(trajectory['sigma'])
    tau_diff = np.diff(trajectory['tau'])
    rate = sigma_diff / tau_diff
    ax4.plot(trajectory['tau'][1:], rate, 'b-')
    ax4.set_title('Rate of Scale Change (dσ/dτ)')
    ax4.set_xlabel('τ-time')
    ax4.set_ylabel('dσ/dτ')
    
    # Plot 5: Scale distribution histogram
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.hist(trajectory['sigma'], bins=30, color='green', alpha=0.7)
    ax5.set_title('Distribution of Scale Values')
    ax5.set_xlabel('σ (log scale)')
    ax5.set_ylabel('Frequency')
    
    # Plot 6: Text summary
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis('off')
    summary_text = f"""
    Scale Analysis Summary ({scenario})
    ----------------------------------
    Minimum scale: {metrics['scale_min']:.2e}
    Maximum scale: {metrics['scale_max']:.2e}
    Scale ratio: {metrics['scale_ratio']:.2e}
    
    Maximum rate of change: {metrics['max_rate']:.2f}
    Time transformation ratio: {metrics['time_ratio']:.2f}
    
    Extreme scale changes: {metrics['extreme_count']}
    """
    ax6.text(0.05, 0.95, summary_text, fontsize=12, verticalalignment='top')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Scale analysis saved to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)

def compare_scenarios(scenarios=None, save_path=None, show_plot=False):
    """
    Compare scale metrics across different scenarios.
    
    Args:
        scenarios: List of scenarios to compare
        save_path: Path to save the comparison plot
        show_plot: Whether to display the plot interactively
    """
    if scenarios is None:
        scenarios = ['standard', 'comet', 'close_approach', 'binary_planet', 'rogue_star']
    
    results = {}
    for scenario in scenarios:
        print(f"\nAnalyzing {scenario} scenario...")
        trajectory = load_trajectory_data(scenario)
        metrics = calculate_scale_metrics(trajectory)
        results[scenario] = metrics
        
        # Save individual scenario analysis
        scenario_save_path = save_path.replace('_comparison', f'_{scenario}') if save_path else None
        if scenario_save_path:
            plot_scale_analysis(trajectory, metrics, scenario, scenario_save_path, False)
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Scale ratio comparison
    scenario_names = list(results.keys())
    scale_ratios = [results[s]['scale_ratio'] for s in scenario_names]
    axes[0, 0].bar(scenario_names, scale_ratios, color='blue', alpha=0.7)
    axes[0, 0].set_title('Scale Ratio Comparison')
    axes[0, 0].set_ylabel('Max/Min Scale Ratio')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Plot 2: Max rate comparison
    max_rates = [results[s]['max_rate'] for s in scenario_names]
    axes[0, 1].bar(scenario_names, max_rates, color='red', alpha=0.7)
    axes[0, 1].set_title('Maximum Rate of Scale Change')
    axes[0, 1].set_ylabel('dσ/dτ max')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Plot 3: Time ratio comparison
    time_ratios = [results[s]['time_ratio'] for s in scenario_names]
    axes[1, 0].bar(scenario_names, time_ratios, color='green', alpha=0.7)
    axes[1, 0].set_title('Time Transformation Ratio')
    axes[1, 0].set_ylabel('Physical time / τ-time')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Plot 4: Extreme episodes comparison
    extreme_counts = [results[s]['extreme_count'] for s in scenario_names]
    axes[1, 1].bar(scenario_names, extreme_counts, color='purple', alpha=0.7)
    axes[1, 1].set_title('Extreme Scale Change Episodes')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Scenario comparison saved to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    # Print summary table
    print("\nScenario Comparison Summary:")
    print("-" * 80)
    print(f"{'Scenario':<15} {'Scale Ratio':<15} {'Max Rate':<15} {'Time Ratio':<15} {'Extreme Events':<15}")
    print("-" * 80)
    
    for scenario in scenario_names:
        print(f"{scenario:<15} {results[scenario]['scale_ratio']:<15.2e} {results[scenario]['max_rate']:<15.2f} "
              f"{results[scenario]['time_ratio']:<15.2f} {results[scenario]['extreme_count']:<15}")
    
    # Save metrics to a JSON file
    if save_path:
        metrics_path = save_path.replace('.png', '_metrics.json')
        serializable_results = {}
        for scenario, metrics in results.items():
            serializable_results[scenario] = {
                'scale_min': float(metrics['scale_min']),
                'scale_max': float(metrics['scale_max']),
                'scale_ratio': float(metrics['scale_ratio']),
                'max_rate': float(metrics['max_rate']),
                'time_ratio': float(metrics['time_ratio']),
                'extreme_count': int(metrics['extreme_count'])
            }
        
        with open(metrics_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        print(f"Metrics saved to {metrics_path}")
    
    return results

def main():
    parser = argparse.ArgumentParser(
        description='Analyze simulation results from the Infinite Origin Framework',
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        '--scenario', type=str,
        default='standard',
        help='Scenario to analyze'
    )
    
    parser.add_argument(
        '--compare', action='store_true',
        help='Compare multiple scenarios'
    )
    
    parser.add_argument(
        '--scenarios', type=str,
        help='Comma-separated list of scenarios to compare'
    )
    
    parser.add_argument(
        '--save', type=str,
        default='output/analysis',
        help='Save the analysis plots with the given prefix'
    )
    
    parser.add_argument(
        '--show', action='store_true',
        help='Show interactive plots (default: False)'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Scale Analysis Tool for Infinite Origin Framework")
    print("=" * 70)
    
    if args.compare:
        scenarios = args.scenarios.split(',') if args.scenarios else None
        save_path = f"{args.save}_comparison.png" 
        compare_scenarios(scenarios, save_path, args.show)
    else:
        trajectory = load_trajectory_data(args.scenario)
        metrics = calculate_scale_metrics(trajectory)
        
        save_path = f"{args.save}_{args.scenario}.png"
        plot_scale_analysis(trajectory, metrics, args.scenario, save_path, args.show)
        
        # Print summary
        print(f"\nScale Analysis Summary for {args.scenario}:")
        print("-" * 60)
        print(f"Minimum scale: {metrics['scale_min']:.2e}")
        print(f"Maximum scale: {metrics['scale_max']:.2e}")
        print(f"Scale ratio: {metrics['scale_ratio']:.2e}")
        print(f"Maximum rate of change: {metrics['max_rate']:.2f}")
        print(f"Time transformation ratio: {metrics['time_ratio']:.2f}")
        print(f"Extreme scale changes: {metrics['extreme_count']}")
    
    print("\n" + "=" * 70)
    print("Analysis completed successfully!")
    print("=" * 70)

if __name__ == "__main__":
    main() 