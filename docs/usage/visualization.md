# Visualization Tools

The visualization tools module provides a comprehensive set of functions for visualizing configurations and trajectories in both the r-plane and τ-plane representations. These tools are essential for understanding and analyzing the behavior of systems in the Infinite Origin framework.

## Static Configuration Visualization

### Plot Configuration

The `plot_configuration` function allows you to visualize a configuration in either the r-plane or τ-plane.

```python
from visualization_tools import plot_configuration
from configuration_space import Configuration, Point

# Create a configuration
p1 = Point([0, 0], weight=2.0)
p2 = Point([1, 0], weight=1.0)
p3 = Point([0, 1], weight=1.0)
config = Configuration([p1, p2, p3])

# Visualize in r-plane using matplotlib
fig, ax = plot_configuration(
    config, 
    plane='r',               # 'r' or 'tau'
    title="My Configuration", 
    figsize=(10, 6),
    show_scale=True,         # Show scale factor (σ)
    show_center_of_mass=True,
    point_labels=True,
    axes_equal=True,
    use_plotly=False         # Use matplotlib
)

# Visualize in τ-plane using Plotly (interactive)
fig = plot_configuration(
    config, 
    plane='tau',
    use_plotly=True          # Use Plotly for interactive visualization
)
```

### Compare Configurations

The `plot_configuration_comparison` function allows you to visualize a configuration in both planes side by side.

```python
from visualization_tools import plot_configuration_comparison

# Create a side-by-side comparison using matplotlib
fig, (ax_r, ax_tau) = plot_configuration_comparison(
    config,
    figsize=(12, 5),
    show_scale=True,
    show_center_of_mass=True,
    point_labels=True,
    use_plotly=False
)

# Create an interactive comparison using Plotly
fig = plot_configuration_comparison(
    config,
    use_plotly=True
)
```

## Trajectory Visualization

### Plot Trajectory Evolution

The `plot_trajectory` function visualizes the evolution of key variables in a trajectory over time.

```python
from visualization_tools import plot_trajectory
from dynamics_engine.examples import harmonic_oscillator_example

# Run a simulation to get a trajectory
_, trajectory = harmonic_oscillator_example()

# Plot trajectory evolution using matplotlib
fig, axes = plot_trajectory(
    trajectory,
    time_var='tau',          # 'tau' or 't'
    figsize=(10, 8),
    show_sigma=True,         # Show scale evolution
    show_physical_time=True, # Show physical time vs tau
    use_plotly=False,
    title="Harmonic Oscillator Evolution"
)

# Create an interactive plot using Plotly
fig = plot_trajectory(
    trajectory,
    time_var='tau',
    use_plotly=True,
    show_sigma=True,
    show_physical_time=True
)
```

### Visualize Shape Evolution

The `plot_trajectory_shape` function shows the shape evolution of a configuration over time.

```python
from visualization_tools import plot_trajectory_shape

# Visualize shape evolution for selected time points
fig = plot_trajectory_shape(
    trajectory,
    highlight_points=[0, 10, 20, 30, 40], # Specific points to highlight
    figsize=(10, 8),
    use_plotly=False,
    title="Configuration Shape Evolution"
)

# Or select a number of evenly spaced points
fig = plot_trajectory_shape(
    trajectory,
    num_points=10,           # Number of time points to show
    use_plotly=True
)
```

## Animations

### Animate Trajectory

The `animate_trajectory` function creates an animation of a configuration evolving over time.

```python
from visualization_tools import animate_trajectory

# Create animation in r-plane
anim = animate_trajectory(
    trajectory,
    interval=50,             # Milliseconds between frames
    figsize=(10, 6),
    save_path="animation.mp4", # Optional path to save animation
    title="Configuration Evolution",
    show_time=True,          # Show time in title
    plane='r',               # 'r' or 'tau'
    fps=30                   # Frames per second for saved animation
)

# Display the animation in a Jupyter notebook
from IPython.display import HTML
HTML(anim.to_jshtml())
```

### Dual View Animation

The `animate_dual_view` function creates a side-by-side animation showing a trajectory in both r-plane and τ-plane.

```python
from visualization_tools import animate_dual_view

# Create dual view animation
anim = animate_dual_view(
    trajectory,
    interval=50,
    figsize=(12, 6),
    save_path="dual_animation.mp4",
    title="Dual View Animation",
    show_time=True,
    fps=30
)

# Display in a notebook
from IPython.display import HTML
HTML(anim.to_jshtml())
```

## Practical Tips

1. **Interactive Exploration**: Use `use_plotly=True` for interactive exploration where you need to zoom, pan, or hover for data inspection.

2. **Static Visualizations**: Use matplotlib (default) for static visualizations suitable for publications and reports.

3. **Saving Figures**: For matplotlib figures, use `plt.savefig()` after getting the figure.
   ```python
   fig, ax = plot_configuration(config)
   plt.figure(fig.number)  # Activate the figure
   plt.savefig('my_config.png', dpi=300)
   plt.close(fig)
   ```

4. **3D Visualizations**: Both matplotlib and Plotly support 3D configurations. The tools automatically detect the dimension from the configuration.

5. **Large Trajectories**: For large trajectories with many time steps, use `plot_trajectory_shape` with a smaller `num_points` value to avoid cluttered plots. 