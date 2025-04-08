# Dynamics Engine Usage Guide

This guide explains how to use the dynamics simulation engine to model and simulate the evolution of configurations in transformed time.

## Overview

The dynamics engine implements the time transformation capabilities and integration methods defined in the geometric framework with infinity at the origin. It allows for simulating multi-point configurations with proper handling of scale evolution, shape preservation, and time regularization.

## Mathematical Foundation

### Scale-Shape Decomposition

At the core of the dynamics engine is the decomposition of a configuration into scale (σ) and shape (θ) components:

```
x = e^σ · θ
```

Where:
- `x` is the physical configuration in Cartesian coordinates
- `σ` (sigma) is the logarithmic scale coordinate
- `θ` (theta) is the shape coordinate on the unit sphere
- `e^σ` represents the overall scale of the system

This decomposition allows tracking the system's size and shape components independently, which is crucial for handling extreme scale variations.

### Time Transformation

The time transformation is defined by the relation:

```
dt/dτ = e^(f(σ))
```

Where:
- `t` is physical time
- `τ` (tau) is transformed time
- `f(σ)` is a function of the logarithmic scale coordinate `σ` (sigma)

This transformation regularizes dynamics at extreme scales, preventing runaway behavior and enabling stable numerical integration.

### Equations of Motion

The transformed equations of motion in τ-time are:

```
d²σ/dτ² = F(σ) + |dθ/dτ|²
dθ/dτ = -∇_θ V(θ) - 2(dσ/dτ)(dθ/dτ)
```

Where:
- `F(σ)` is the driving function determining scale evolution
- `V(θ)` is the shape potential (typically from physical forces)
- `∇_θ` is the gradient with respect to shape coordinates

This formulation ensures conservation of energy and momentum in the transformed system.

## Driving Functions

Driving functions `F(σ)` determine the evolution of the system. They represent forces or potentials that drive the dynamics.

Common examples:
- Harmonic oscillator: `F(σ) = -e^(2σ)`
- Gravitational attraction: `F(σ) = -e^σ`
- Collapsing system: `F(σ) = -2e^σ`

The homogeneity degree of the driving function determines its scaling behavior:
- Degree 0: Scale-invariant forces (like the strong nuclear force)
- Degree -1: Coulomb/gravitational forces (inverse-square law)
- Degree -2: Harmonic oscillator (spring forces)

## Using the TimeTransformation Class

### Creating a Time Transformation

```python
from dynamics_engine import TimeTransformation

# Create with an explicit function f(σ)
f_sigma = lambda s: -s  # Example: f(σ) = -σ
transform = TimeTransformation(f_sigma)

# Create from a driving function
driving_function = lambda s: -np.exp(2*s)  # Harmonic oscillator
transform = TimeTransformation.from_driving_function(
    driving_function,
    homogeneity_degree=2,  # Default for second-order systems
    regularization_strategy='exponential'  # Analyzes the exponential dependence
)
```

### Computing Transformation Factors

```python
# Get dt/dτ for a specific σ value
sigma_value = 1.5
dt_dtau = transform.dt_dtau(sigma_value)
print(f"dt/dτ at σ = {sigma_value}: {dt_dtau}")

# Get the inverse dτ/dt
dtau_dt = transform.dtau_dt(sigma_value)
print(f"dτ/dt at σ = {sigma_value}: {dtau_dt}")
```

### Converting Between t and τ

```python
# Array of τ values
tau_values = np.linspace(0, 10, 100)

# Array of corresponding σ values along the trajectory
sigma_values = some_trajectory_function(tau_values)

# Convert to physical time t
t_values = transform.integrate_transformation(sigma_values, tau_values)

# Now you can plot against physical time
plt.plot(t_values, sigma_values)
plt.xlabel('Physical time (t)')
plt.ylabel('Logarithmic scale (σ)')
```

## Configuration Space

### Creating and Manipulating Configurations

```python
from dynamics_engine.configuration import Configuration, Point

# Create points in configuration space
p1 = Point([0.0, 0.0, 0.0])  # Origin in 3D
p2 = Point([1.0, 0.0, 0.0])  # Point on x-axis
p3 = Point([0.0, 1.0, 0.0])  # Point on y-axis

# Create a configuration from points
config = Configuration([p1, p2, p3])

# Access configuration properties
sigma = config.sigma  # Logarithmic scale
theta = config.theta  # Shape coordinates (on unit sphere)
dim = config.dimension  # Dimension of configuration space
num_points = config.num_points  # Number of points

# Convert to and from physical coordinates
physical_coords = config.to_physical()
config_from_physical = Configuration.from_physical(physical_coords)

# Apply transformations
config_scaled = config.scale(2.0)  # Scale by factor of 2
config_rotated = config.rotate(angle=np.pi/4, axis=[0, 0, 1])  # Rotate 45° around z-axis
```

### Computing Configuration Properties

```python
# Compute moment of inertia tensor
inertia_tensor = config.moment_of_inertia()

# Compute center of mass
com = config.center_of_mass()

# Compute configuration energy
energy = config.energy(
    potential_function=lambda r: -1.0/r,  # Gravitational-like potential
    mass=[1.0, 1.0, 1.0]  # Mass of each point
)

# Compute relative distances
distances = config.distances()
min_distance = np.min(distances)
```

## Integration and Simulation

### Basic Integration

```python
from dynamics_engine import integrate
from dynamics_engine.configuration import Configuration, Point

# Create a configuration
p1 = Point([0, 0])
p2 = Point([1, 0])
config = Configuration([p1, p2])

# Define a driving function
F = lambda s: -np.exp(2*s)  # Harmonic oscillator

# Integrate the dynamics
results = integrate(
    config,
    driving_function=F,
    tau_max=10.0,  # Maximum τ-time
    num_steps=1000,  # Number of integration steps
    time_transform=None,  # Will be created automatically if None
    use_adaptive_steps=True  # Use adaptive step sizing for better accuracy
)

# Access the results
tau_values = results['tau']
sigma_values = results['sigma']
physical_time = results['t']
theta_evolution = results['theta']  # Shape evolution
energy = results['energy']  # Energy at each step
```

### Integrator Options

```python
# Specify integrator options
from dynamics_engine.integrators import IntegratorOptions

options = IntegratorOptions(
    method='rk45',           # Integration method: 'rk45', 'dopri', 'euler'
    rtol=1e-6,               # Relative tolerance
    atol=1e-8,               # Absolute tolerance
    max_step=0.1,            # Maximum step size
    min_step=1e-10,          # Minimum step size
    store_intermediate=True  # Store all intermediate steps
)

# Use these options in integration
results = integrate(
    config,
    driving_function=F,
    tau_max=10.0,
    num_steps=1000,
    integrator_options=options
)
```

### Complete Simulation

```python
from dynamics_engine import simulate

# Simulate and get the final configuration
final_config, trajectory = simulate(
    config,
    driving_function=F,
    tau_max=10.0,
    num_steps=500
)

# The trajectory contains all the intermediate values
print(f"Initial sigma: {config.sigma}")
print(f"Final sigma: {final_config.sigma}")
print(f"Physical time elapsed: {trajectory['t'][-1]}")

# You can access the full evolution
sigma_evolution = trajectory['sigma']
shape_evolution = trajectory['theta']

# Plot the results
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))

# Plot scale evolution
plt.subplot(2, 2, 1)
plt.plot(trajectory['tau'], trajectory['sigma'])
plt.xlabel('τ-time')
plt.ylabel('Scale (σ)')
plt.title('Scale Evolution')

# Plot physical time vs τ-time
plt.subplot(2, 2, 2)
plt.plot(trajectory['tau'], trajectory['t'])
plt.xlabel('τ-time')
plt.ylabel('Physical time (t)')
plt.title('Time Transformation')

# Plot energy conservation
plt.subplot(2, 2, 3)
plt.plot(trajectory['tau'], trajectory['energy'])
plt.xlabel('τ-time')
plt.ylabel('Energy')
plt.title('Energy Conservation')

# Plot minimum distance between particles
plt.subplot(2, 2, 4)
min_distances = [config.min_distance() for config in trajectory['configurations']]
plt.plot(trajectory['tau'], min_distances)
plt.xlabel('τ-time')
plt.ylabel('Minimum distance')
plt.title('Minimum Particle Distance')

plt.tight_layout()
plt.show()
```

### Running Examples

The `dynamics_engine.examples` module provides ready-to-use examples:

```python
from dynamics_engine.examples import (
    harmonic_oscillator_example,
    three_body_example,
    collapsing_system_example
)

# Run the harmonic oscillator example
final_config, trajectory = harmonic_oscillator_example()

# Run the three-body example
final_config, trajectory = three_body_example()

# Run the collapsing system example
final_config, trajectory = collapsing_system_example()
```

You can also run examples from the command line:

```bash
# Run a specific example
python main.py --example harmonic
python main.py --example three-body
python main.py --example collapse

# Run all examples
python main.py --example all
```

## Advanced Usage

### Implementing Custom Driving Functions

You can create driving functions based on physical potentials:

```python
import numpy as np

# Gravitational N-body driving function
def gravitational_driving(sigma, theta, masses):
    """
    Compute driving function for gravitational N-body problem.
    
    Args:
        sigma: Scale coordinate
        theta: Shape coordinates (normalized)
        masses: List of masses for each point
    
    Returns:
        F(sigma): Driving function value
    """
    # Convert to physical configuration
    exp_sigma = np.exp(sigma)
    points = theta * exp_sigma
    
    # Compute gravitational potential energy
    G = 1.0  # Gravitational constant
    potential = 0.0
    n_points = len(masses)
    
    for i in range(n_points):
        for j in range(i+1, n_points):
            # Distance between points i and j
            r_ij = np.linalg.norm(points[i] - points[j])
            potential -= G * masses[i] * masses[j] / r_ij
    
    # Driving function for gravitational potential (homogeneity degree -1)
    return -potential / exp_sigma
```

### Custom Damped Systems

You can create more complex driving functions with damping:

```python
# Damped harmonic oscillator
def damped_harmonic(state):
    """
    Damped harmonic oscillator state derivative function.
    
    Args:
        state: Current state [sigma, dsigma_dtau, theta, dtheta_dtau]
    
    Returns:
        State derivatives [dsigma_dtau, d2sigma_dtau2, dtheta_dtau, d2theta_dtau2]
    """
    sigma, dsigma_dtau, theta, dtheta_dtau = state
    
    # Scale driving function with damping term
    F_sigma = -np.exp(2*sigma) - 0.1 * dsigma_dtau
    
    # Compute shape force (usually depends on physical forces)
    F_theta = np.zeros_like(theta)
    
    # Second derivatives
    d2sigma_dtau2 = F_sigma + np.sum(dtheta_dtau**2)
    d2theta_dtau2 = F_theta - 2 * dsigma_dtau * dtheta_dtau
    
    return [dsigma_dtau, d2sigma_dtau2, dtheta_dtau, d2theta_dtau2]

# Use with a custom integrator
from dynamics_engine.integrators import CustomIntegrator
integrator = CustomIntegrator(damped_harmonic)
results = integrator.integrate(initial_state, tau_max=10.0, num_steps=1000)
```

### Hyperreal Integration

For systems involving singularities, you can use hyperreal arithmetic:

```python
from hyperreal_arithmetic import HyperrealNum
from dynamics_engine import compute_time_transformation

# Define functions with hyperreal arguments
def f_sigma_hyperreal(sigma):
    if isinstance(sigma, HyperrealNum) and sigma.is_infinitesimal():
        # Special handling for infinitesimal scales
        return HyperrealNum(-1.0, 0)  # Regular value for infinitesimal input
    return -sigma  # Regular case

# Compute transformation with hyperreal numbers
sigma = HyperrealNum.infinitesimal()
dt_dtau = compute_time_transformation(sigma, f_sigma_hyperreal)

# This enables precise calculations near singularities
```

### Combining with τ-Plane Mapping

The dynamics engine can be used with the τ-plane mapping for handling multi-scale systems:

```python
from mapping_functions import r_to_tau, tau_to_r
from dynamics_engine import integrate

# Initial configuration in r-plane
r_config = [
    [0.0, 0.0],  # First point
    [1.0, 0.0]   # Second point
]

# Convert to τ-plane
tau_config = [r_to_tau(r) for r in r_config]

# Create dynamics configuration from τ-coordinates
from dynamics_engine.configuration import Configuration, Point
points = [Point(tau) for tau in tau_config]
config = Configuration(points)

# Run simulation in τ-time
final_config, trajectory = simulate(
    config,
    driving_function=lambda s: -np.exp(s),  # Gravitational-like
    tau_max=10.0
)

# Convert back to r-plane for visualization
final_r_config = [tau_to_r(point.coordinates) for point in final_config.points]
```

## Performance Tips

1. For most simulations, adaptive step sizing provides the best balance of accuracy and performance.
2. If you know your system has smooth, predictable dynamics, fixed step size can be faster.
3. When dealing with near-singular behavior, adjust the time transformation to regularize the evolution.
4. For production simulations, consider using the `cProfile` module to identify bottlenecks.
5. Pre-calculate as much as possible before the integration loop for optimal performance.
6. For large N-body systems, consider approximation methods like Barnes-Hut or Fast Multipole Method.

## Regularization Strategies

The dynamics engine offers several regularization strategies for different types of singularities:

1. **Logarithmic regularization**: `f(σ) = -σ`
   - Good for gravitational/Coulomb-like singularities
   - Effectively spreads out the singularity in τ-time

2. **Double-logarithmic regularization**: `f(σ) = -2σ`
   - More aggressive regularization for stronger singularities
   - Used in the collapsing system example

3. **Exponential regularization**: `f(σ) = -e^(kσ)`
   - Provides even stronger regularization for extreme collapse
   - Parameter k controls regularization strength

## See Also

- [Configuration Space Documentation](./configuration_space.md)
- [Hyperreal Arithmetic Documentation](./hyperreal_arithmetic.md)
- [Mapping Functions Documentation](./mapping_functions.md)
- [Extreme Collapse Benchmark](./extreme-collapse-scenario-analysis.md) 