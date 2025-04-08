"""
Animation Test Script

A simplified script to test animation functionality.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from configuration_space import Configuration, Point
from dynamics_engine.examples import harmonic_oscillator_example

# Create output directory if it doesn't exist
os.makedirs("output", exist_ok=True)

# Run the harmonic oscillator example to get a trajectory
print("Running harmonic oscillator simulation...")
final_config, trajectory = harmonic_oscillator_example()

# Reconstruct positions for animation
print("Reconstructing positions...")
theta = trajectory.get('theta', None)
sigma = trajectory.get('sigma', None)
tau_values = trajectory.get('tau', None)

n_steps = len(sigma)
n_coords = theta.shape[1]
dim = 2  # Harmonic oscillator is 2D
n_points = n_coords // dim
positions = np.zeros((n_steps, n_points, dim))

for i in range(n_steps):
    scale = np.exp(sigma[i])
    points_at_t = theta[i].reshape(n_points, dim)
    positions[i] = points_at_t * scale

# Now let's create a manual animation using matplotlib's animation module
print("Creating animation...")

# Prepare figure and axis
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_aspect('equal')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.grid(True, alpha=0.3)

# Colors for the points
colors = plt.cm.viridis(np.linspace(0, 1, n_points))

# Initialize with the first frame data instead of empty
scatter = ax.scatter(
    positions[0, :, 0], 
    positions[0, :, 1], 
    s=100, 
    c=colors
)

# Title with time information
title_text = ax.set_title(f"Time: {tau_values[0]:.2f}")

def animate(i):
    """Animate one frame."""
    # Update the scatter plot
    scatter.set_offsets(positions[i])
    
    # Update the title
    title_text.set_text(f"Time: {tau_values[i]:.2f}")
    
    return scatter, title_text

# Create animation
print("Setting up animation...")
anim = animation.FuncAnimation(
    fig, animate, frames=len(positions),
    blit=True, interval=50
)

# Save the animation
print("Saving animation...")
try:
    # Save as mp4 (requires ffmpeg)
    writer = animation.FFMpegWriter(fps=30, metadata=dict(artist='Infinite Origin Framework'))
    anim.save('output/simple_animation.mp4', writer=writer)
    print("Animation saved successfully to output/simple_animation.mp4")
except Exception as e:
    print(f"Error saving animation: {e}")
    print("Make sure ffmpeg is installed and properly configured.")

# Display the animation in an interactive window
print("Displaying animation (close window to continue)...")
plt.show()
print("Animation test complete!") 