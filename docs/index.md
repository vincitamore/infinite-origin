# Infinite Origin Framework Documentation

Welcome to the documentation for the Infinite Origin framework, a computational implementation of a geometric system with infinity at the origin.

## Overview

This framework provides tools for working with a novel geometric approach that places infinity at the origin and an infinitesimal boundary at the extremes. Based on hyperreal numbers, it enables exploration and computation across vast scales with precision and clarity.

## Core Components

### Hyperreal Arithmetic

The foundation of the framework is a hyperreal arithmetic system that extends real numbers to include infinitesimals and infinities.

- [Hyperreal Arithmetic Documentation](usage/hyperreal_arithmetic.md)

### Mapping Functions

Transformations between the τ-plane (where infinity is at the origin) and the r-plane (traditional Cartesian geometry).

- [Mapping Functions Documentation](usage/mapping_functions.md)

### Configuration Space

Tools for managing multi-point systems with scale, shape, and orientation.

- [Configuration Space Documentation](usage/configuration_space.md)

### Dynamics Engine

Simulation capabilities for evolving configurations over transformed time.

- [Dynamics Engine Documentation](usage/dynamics.md)

### Visualization Tools

Interactive and static tools for visualizing configurations and trajectories in both τ-plane and r-plane.

- [Visualization Tools Documentation](usage/visualization.md)

## Getting Started

To get started with the framework, follow these steps:

1. Install the framework according to the [README.md](../README.md) instructions
2. Explore the basic examples provided in each module
3. Run the pre-built simulations to understand the dynamics

```python
# Basic example
from configuration_space import Configuration, Point
from dynamics_engine import simulate
from visualization_tools import plot_configuration, plot_trajectory
import numpy as np

# Create a simple configuration
p1 = Point([0, 0])
p2 = Point([1, 0])
config = Configuration([p1, p2])

# Visualize the initial configuration
fig, ax = plot_configuration(config, plane='r')
plt.show()

# Define a driving function
F = lambda sigma: -np.exp(2*sigma)

# Simulate dynamics
final_config, trajectory = simulate(config, F, tau_max=10.0)

# Visualize the trajectory
fig, axes = plot_trajectory(trajectory, show_sigma=True)
plt.show()

# Print results
print(f"Initial sigma: {config.sigma}")
print(f"Final sigma: {final_config.sigma}")
```

## Development

For those interested in contributing to the framework:

1. Read the [development plan](../development-plan.md) to understand the project structure
2. Check the [dev-progress.md](../dev-progress.md) file for current status
3. Run the tests to ensure your changes maintain compatibility

## Mathematical Background

The framework is based on a geometric system with infinity at the origin, as described in detail in the [hyperreal-geometric-infinite-framework-axioms.md](../hyperreal-geometric-infinite-framework-axioms.md) document. 