# Getting Started with the Infinite Origin Framework

This tutorial provides a step-by-step introduction to the Infinite Origin framework, a computational implementation of a geometric system with infinity at the origin. We'll explore the key components, demonstrate basic usage, and create visualizations.

## Introduction

The Infinite Origin framework implements a geometric system where:
- Infinity is positioned at the origin
- The τ-plane maps to the r-plane through inversion
- Multi-point configurations manage scale and shape separately
- Dynamics are simulated with regularized time transformations

This allows for:
- Handling extreme scales through hyperreal arithmetic
- Analyzing asymptotic behavior at infinity
- Simulating multi-body dynamics with stable numerical properties
- Visualizing configurations in different coordinate systems

## Installation

To install the framework, clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/infinite-origin.git
cd infinite-origin
pip install -r requirements.txt
```

## Basic Usage

The framework can be used as a command-line tool or as a Python library. Let's start with the command-line interface:

```bash
# Run a harmonic oscillator example
python main.py --example harmonic

# Run all examples
python main.py --example all

# Run an example and create visualizations
python main.py --example three-body --visualize all --save my_results
```

## Core Concepts

### 1. The τ-Plane and r-Plane

The τ-plane and r-plane are related through inversion. Let's explore this relationship:

```python
from hyperreal_arithmetic.numerical import HyperrealNum
from mapping_functions import tau_to_r, r_to_tau

# Create τ-plane coordinates
tau_coords = [HyperrealNum(1.0, 0), HyperrealNum(2.0, 0)]

# Map to r-plane
r_coords = tau_to_r(tau_coords)
print(f"τ-plane: {tau_coords} → r-plane: {r_coords}")

# Map back to τ-plane
tau_coords_2 = r_to_tau(r_coords)
print(f"r-plane: {r_coords} → τ-plane: {tau_coords_2}")
```

Key points:
- Points near the origin in the τ-plane map to points at infinity in the r-plane
- Points at infinity in the τ-plane map to points near the origin in the r-plane
- The relationship is $r_i = \frac{1}{\tau_i}$ for each coordinate

### 2. Configurations

Configurations represent multi-point systems with scale, shape, and orientation:

```python
from configuration_space import Configuration, Point

# Create points
p1 = Point([0, 0], weight=2.0)  # Center point with weight 2.0
p2 = Point([1, 0], weight=1.0)  # Point on x-axis
p3 = Point([0, 1], weight=1.0)  # Point on y-axis

# Create configuration
config = Configuration([p1, p2, p3])

# Fix center of mass at origin
config.fix_center_of_mass()

# Access configuration properties
print(f"Scale factor: {config.scale_factor}")
print(f"Sigma (log scale): {config.sigma}")
print(f"Center of mass: ({config.center_of_mass_x}, {config.center_of_mass_y})")

# Get shape coordinates (scale-invariant)
shape_coords = config.get_shape_coordinates()
print(f"Shape coordinates: {shape_coords}")
```

### 3. Dynamics

Dynamics simulate the evolution of configurations over time:

```python
import numpy as np
from dynamics_engine import TimeTransformation, simulate

# Create configuration
p1 = Point([0, 0], weight=2.0)
p2 = Point([1, 0], weight=1.0)
p3 = Point([0, 1], weight=1.0)
config = Configuration([p1, p2, p3])
config.fix_center_of_mass()

# Define driving function F(σ) = -e^σ (gravitational-like)
F = lambda s: -np.exp(s)

# Create time transformation with f(σ) = σ/2
transform = TimeTransformation(lambda s: s/2)

# Simulate for τ ∈ [0, 10]
tau_max = 10.0
final_config, trajectory = simulate(
    config, F, tau_max=tau_max, num_steps=200, 
    time_transform=transform
)

# Examine trajectory
print(f"Initial sigma: {trajectory['sigma'][0]}")
print(f"Final sigma: {trajectory['sigma'][-1]}")
print(f"Physical time elapsed: {trajectory['t'][-1]}")
```

### 4. Visualization

The framework provides various visualization tools:

```python
import matplotlib.pyplot as plt
from visualization_tools import (
    plot_configuration,
    plot_trajectory,
    animate_trajectory,
    animate_dual_view
)

# Static visualization in r-plane
fig1, ax1 = plot_configuration(
    config, 
    plane='r', 
    title="Configuration in r-plane", 
    show_scale=True,
    show_center_of_mass=True
)

# Trajectory visualization
fig2, axes2 = plot_trajectory(
    trajectory,
    time_var='tau',
    show_sigma=True,
    show_physical_time=True,
    title="Trajectory Visualization"
)

# Animation
anim = animate_trajectory(
    trajectory,
    interval=50,
    figsize=(8, 6),
    save_path="tutorial_animation.mp4",
    title="Configuration Animation",
    show_time=True,
    plane='r',
    fps=30
)

# Show plots
plt.show()
```

## Example Applications

### Harmonic Oscillator

Let's analyze a harmonic oscillator using the framework:

```python
from dynamics_engine.examples import harmonic_oscillator_example

# Run harmonic oscillator example
final_config, trajectory = harmonic_oscillator_example()

# Plot trajectory
import matplotlib.pyplot as plt
from visualization_tools import plot_trajectory

fig, axes = plot_trajectory(
    trajectory, 
    time_var='tau', 
    show_sigma=True, 
    show_physical_time=True,
    title="Harmonic Oscillator Evolution"
)
plt.show()
```

Key observations:
- The oscillator follows a periodic pattern in τ-time
- Scale (σ) oscillates between maximum and minimum values
- Physical time (t) relates to τ-time through the transformation

### Three-Body System

The framework excels at multi-body dynamics:

```python
from dynamics_engine.examples import three_body_example

# Run three-body example
final_config, trajectory = three_body_example()

# Visualize with dual view
from visualization_tools import animate_dual_view

anim = animate_dual_view(
    trajectory,
    interval=50,
    figsize=(12, 6),
    save_path="three_body_dual.mp4",
    title="Three-Body System",
    show_time=True,
    fps=30
)
```

This shows:
- The motion of three bodies under gravitational-like forces
- The corresponding representation in both r-plane and τ-plane
- How the scale evolves over time

## Advanced Usage

### Custom Configurations

Create and simulate custom configurations:

```python
# Create a pentagon of equal-weight points
import numpy as np
from configuration_space import Configuration, Point

points = [
    Point([np.cos(2*np.pi*i/5), np.sin(2*np.pi*i/5)]) 
    for i in range(5)
]
config = Configuration(points)
config.fix_center_of_mass()

# Define a custom driving function
F = lambda s: -2 * np.exp(s) * np.cos(s)  # Oscillating collapse force

# Create time transformation
transform = TimeTransformation(lambda s: s/3)

# Simulate
final_config, trajectory = simulate(
    config, F, tau_max=20.0, num_steps=400, 
    time_transform=transform
)

# Visualize
from visualization_tools import animate_trajectory
animate_trajectory(
    trajectory,
    interval=50,
    figsize=(8, 6),
    save_path="pentagon_animation.mp4",
    title="Pentagon Configuration",
    show_time=True,
    plane='r',
    fps=30
)
```

### Performance Profiling

Optimize your simulations with performance profiling:

```bash
# Run performance profiling
python main.py --profile
```

## Conclusion

The Infinite Origin framework provides a powerful toolset for exploring geometric systems across scales. By inverting the traditional paradigm and placing infinity at the origin, it offers unique advantages for numerical simulation, asymptotic analysis, and visualization of multi-point dynamics.

For more examples and detailed documentation, see:

- [API Reference](../api/index.md)
- [Mathematical Background](../theory/index.md)
- [Example Gallery](../examples/index.md)

Happy exploring! 