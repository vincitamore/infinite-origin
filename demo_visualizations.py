"""
Visualization Demo Script

This script showcases the visualization tools for the Infinite Origin framework.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from configuration_space import Configuration, Point
from dynamics_engine.examples import harmonic_oscillator_example, three_body_example
from visualization_tools import (
    plot_configuration,
    plot_configuration_comparison,
    plot_trajectory,
    plot_trajectory_shape,
    animate_trajectory,
    animate_dual_view
)

# Create output directory for plots if it doesn't exist
os.makedirs("output", exist_ok=True)

# 1. Configuration Visualization
print("Demonstrating static configuration visualizations...")

# Create a sample configuration
p1 = Point([0, 0], weight=2.0)      # Central point
p2 = Point([1, 0], weight=1.0)      # Point on x-axis
p3 = Point([0, 1], weight=1.0)      # Point on y-axis
p4 = Point([0.7, 0.7], weight=0.5)  # Point in first quadrant
config = Configuration([p1, p2, p3, p4])

# Fix center of mass at origin
config.fix_center_of_mass()

# 1.1 r-plane visualization
print("1.1 Plotting configuration in r-plane...")
fig1, ax1 = plot_configuration(
    config, 
    plane='r', 
    title="Configuration in r-plane", 
    show_scale=True,
    show_center_of_mass=True
)
plt.figure(fig1.number)
plt.savefig('output/config_r_plane.png')
plt.close(fig1)

# 1.2 τ-plane visualization
print("1.2 Plotting configuration in τ-plane...")
fig2, ax2 = plot_configuration(
    config, 
    plane='tau', 
    title="Configuration in τ-plane", 
    show_scale=True
)
plt.figure(fig2.number)
plt.savefig('output/config_tau_plane.png')
plt.close(fig2)

# 1.3 Side-by-side comparison
print("1.3 Creating side-by-side comparison...")
fig3, (ax3_1, ax3_2) = plot_configuration_comparison(
    config,
    show_scale=True,
    show_center_of_mass=True
)
plt.figure(fig3.number)
plt.savefig('output/config_comparison.png')
plt.close(fig3)

# 2. Trajectory Visualization
print("\nDemonstrating trajectory visualizations...")

# 2.1 Harmonic oscillator trajectory
print("2.1 Running harmonic oscillator example...")
final_config, trajectory = harmonic_oscillator_example()

# Plot trajectory variables
print("2.2 Plotting trajectory variables...")
fig4, axes4 = plot_trajectory(
    trajectory, 
    time_var='tau', 
    show_sigma=True, 
    show_physical_time=True,
    title="Harmonic Oscillator Evolution"
)
plt.figure(fig4.number)
plt.savefig('output/trajectory_vars.png')
plt.close(fig4)

# Reconstruct positions for shape visualization
print("2.3 Reconstructing positions for shape visualization...")
theta = trajectory.get('theta', None)
sigma = trajectory.get('sigma', None)
n_steps = len(sigma)
n_coords = theta.shape[1]
dim = 2  # Harmonic oscillator is 2D
n_points = n_coords // dim
positions = np.zeros((n_steps, n_points, dim))

for i in range(n_steps):
    scale = np.exp(sigma[i])
    points_at_t = theta[i].reshape(n_points, dim)
    positions[i] = points_at_t * scale

trajectory['positions'] = positions

# Plot shape evolution
print("2.4 Plotting shape evolution...")
fig5 = plot_trajectory_shape(
    trajectory, 
    num_points=10, 
    title="Harmonic Oscillator Shape Evolution"
)
plt.figure(fig5.number)
plt.savefig('output/shape_evolution.png')
plt.close(fig5)

# 3. Three-body Example
print("\nDemonstrating three-body example...")

# 3.1 Run three-body example
print("3.1 Running three-body simulation...")
final_config_3b, trajectory_3b = three_body_example()

# 3.2 Plot trajectory
print("3.2 Plotting three-body trajectory variables...")
fig6, axes6 = plot_trajectory(
    trajectory_3b,
    title="Three-Body System Evolution"
)
plt.figure(fig6.number)
plt.savefig('output/three_body_trajectory.png')
plt.close(fig6)

# Reconstruct positions for three-body
print("3.3 Reconstructing positions for three-body...")
theta_3b = trajectory_3b.get('theta', None)
sigma_3b = trajectory_3b.get('sigma', None)
n_steps_3b = len(sigma_3b)
n_coords_3b = theta_3b.shape[1]
dim_3b = 2  # Three-body example is 2D
n_points_3b = n_coords_3b // dim_3b
positions_3b = np.zeros((n_steps_3b, n_points_3b, dim_3b))

for i in range(n_steps_3b):
    scale = np.exp(sigma_3b[i])
    points_at_t = theta_3b[i].reshape(n_points_3b, dim_3b)
    positions_3b[i] = points_at_t * scale

trajectory_3b['positions'] = positions_3b

# Plot shape evolution for three-body
print("3.4 Plotting three-body shape evolution...")
fig7 = plot_trajectory_shape(
    trajectory_3b, 
    num_points=20, 
    title="Three-Body Shape Evolution"
)
plt.figure(fig7.number)
plt.savefig('output/three_body_shape.png')
plt.close(fig7)

# 4. Animations
print("\nDemonstrating animations...")
print("Note: This may take a while, and requires ffmpeg to save animations.")
print("If you don't have ffmpeg installed, animations will be displayed but not saved.")

# 4.1 Animate harmonic oscillator
print("4.1 Creating harmonic oscillator animation...")
try:
    anim1 = animate_trajectory(
        trajectory,
        interval=50,
        figsize=(8, 6),
        save_path="output/harmonic_animation.mp4",
        title="Harmonic Oscillator",
        show_time=True,
        plane='r',
        fps=30
    )
    plt.close()
    print("   Animation saved to output/harmonic_animation.mp4")
except Exception as e:
    print(f"   Error saving animation: {e}")
    print("   Make sure ffmpeg is installed for saving animations.")

# 4.2 Dual view animation
print("4.2 Creating dual view animation...")
try:
    anim2 = animate_dual_view(
        trajectory_3b,
        interval=50,
        figsize=(12, 6),
        save_path="output/three_body_dual_animation.mp4",
        title="Three-Body System",
        show_time=True,
        fps=30
    )
    plt.close()
    print("   Animation saved to output/three_body_dual_animation.mp4")
except Exception as e:
    print(f"   Error saving animation: {e}")
    print("   Make sure ffmpeg is installed for saving animations.")

print("\nAll visualizations complete! Check the 'output' directory for results.")
print("Static plots saved as PNG files.")
print("Animations saved as MP4 files if ffmpeg is installed.") 