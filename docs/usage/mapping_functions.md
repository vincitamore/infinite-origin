# Mapping Functions Documentation

This document provides an overview of the mapping functions between the τ-plane and r-plane in the Infinite Origin framework.

## Overview

The mapping functions module implements transformations between the τ-plane (where infinity is at the origin) and the r-plane (traditional Cartesian plane). These mappings allow for working with both infinitesimal and infinite scales within a unified framework.

## Mathematical Foundation

### The Mapping Relationship

The core mapping between the τ-plane and r-plane is defined by the reciprocal relationship:

```
r = 1/τ and τ = 1/r
```

Where:
- `r` represents coordinates in the traditional Cartesian (r-plane)
- `τ` represents coordinates in the transformed τ-plane

This simple relationship has profound implications:
- The origin in the τ-plane (τ = 0) corresponds to infinity in the r-plane
- Infinity in the τ-plane corresponds to the origin in the r-plane (r = 0)
- Points near the origin in the τ-plane represent very distant points in the r-plane
- Points far from the origin in the τ-plane represent points near the origin in the r-plane

### Geometric Interpretation

The τ-plane transformation effectively "turns the universe inside out" by:

1. Mapping the infinite Cartesian space to a finite region
2. Placing infinity at a definite location (the origin)
3. Preserving angles and relative positions (conformal mapping)
4. Inversing the scale relationship (large becomes small, small becomes large)

This enables numerical representations of both infinitesimal and infinite values without numerical overflow or underflow.

## Basic Transformations

```python
from mapping_functions import tau_to_r, r_to_tau
import numpy as np

# Create coordinates in τ-plane
tau_coords = np.array([1.0, 2.0])

# Map to r-plane
r_coords = tau_to_r(tau_coords)
print(f"τ-coordinates {tau_coords} map to r-coordinates {r_coords}")
# Output: τ-coordinates [1. 2.] map to r-coordinates [1.  0.5]

# Map back to τ-plane
tau_coords_again = r_to_tau(r_coords)
print(f"r-coordinates {r_coords} map to τ-coordinates {tau_coords_again}")
# Output: r-coordinates [1.  0.5] map to τ-coordinates [1. 2.]
```

### Using Hyperreals

The mapping functions support hyperreal numbers for precise handling of extreme values:

```python
from mapping_functions import tau_to_r, r_to_tau
from hyperreal_arithmetic import Hyperreal, HyperrealNum

# Using symbolic hyperreals
tau_symbolic = [Hyperreal(1), Hyperreal.infinitesimal()]
r_symbolic = tau_to_r(tau_symbolic)
# r_symbolic will contain [Hyperreal(1), Hyperreal.infinite()]

# Using numerical hyperreals
tau_numerical = [HyperrealNum(1.0, 0), HyperrealNum(1.0, -2)]  # [1, ε²]
r_numerical = tau_to_r(tau_numerical)
# r_numerical will contain [HyperrealNum(1.0, 0), HyperrealNum(1.0, 2)]  # [1, ∞²]
```

## Handling Singularities

The τ-plane representation introduces singularities at specific points that require careful handling:

### Origin Singularity (τ = 0)

In the τ-plane, the origin represents infinity in the r-plane and requires special treatment:

```python
from mapping_functions import handle_origin_tau, is_near_singularity

# A point very close to the origin in τ-plane
tau_coords = [1e-15, 2e-16]

# Check if near singularity
if is_near_singularity(tau_coords, epsilon=1e-10):
    # Apply regularization
    safe_tau = handle_origin_tau(tau_coords, epsilon=1e-10)
    # safe_tau will be regularized to avoid the exact singularity
    
# Using hyperreals for precise handling
from hyperreal_arithmetic import HyperrealNum
tau_epsilon = [HyperrealNum(1.0, -10), HyperrealNum(2.0, -10)]  # Very small
safe_tau = handle_origin_tau(tau_epsilon)
# safe_tau will use hyperreal arithmetic to properly represent the near-singular point
```

### Infinity Handling (r → ∞)

Points approaching infinity in the r-plane (origin in τ-plane) are handled with:

```python
from mapping_functions import handle_infinity_r

# Extremely large coordinates in r-plane
r_coords = [1e20, 2e18]

# Apply regularization
safe_r = handle_infinity_r(r_coords, max_magnitude=1e10)
# safe_r will contain bounded values that preserve directional information

# Using hyperreals
from hyperreal_arithmetic import HyperrealNum
r_huge = [HyperrealNum(1.0, 15), HyperrealNum(2.0, 14)]  # Very large values
safe_r = handle_infinity_r(r_huge)
# safe_r will use hyperreal arithmetic to properly represent the near-infinite point
```

## Distance Calculations

Computing distances in either plane requires special consideration of scale:

```python
from mapping_functions import compute_distance_tau, compute_distance_r

# Points in τ-plane
tau1 = [1.0, 1.0]
tau2 = [2.0, 2.0]

# Distance in τ-plane
tau_distance = compute_distance_tau(tau1, tau2)
print(f"Distance in τ-plane: {tau_distance}")

# Map to r-plane and compute distance
from mapping_functions import tau_to_r
r1 = tau_to_r(tau1)
r2 = tau_to_r(tau2)
r_distance = compute_distance_r(r1, r2)
print(f"Distance in r-plane: {r_distance}")
```

### Scale-Dependent Distance

The relationship between distances in the two planes is scale-dependent:

```python
# Demonstration of scale dependence
tau_pairs = [
    ([0.1, 0.1], [0.2, 0.2]),   # Far from origin in τ-plane (near origin in r-plane)
    ([10.0, 10.0], [11.0, 11.0]),  # Near origin in τ-plane (far in r-plane)
]

for tau_a, tau_b in tau_pairs:
    # Distance in τ-plane
    d_tau = compute_distance_tau(tau_a, tau_b)
    
    # Map to r-plane
    r_a = tau_to_r(tau_a)
    r_b = tau_to_r(tau_b)
    
    # Distance in r-plane
    d_r = compute_distance_r(r_a, r_b)
    
    print(f"τ-plane: {tau_a} to {tau_b}, Distance: {d_tau}")
    print(f"r-plane: {r_a} to {r_b}, Distance: {d_r}")
    print(f"Ratio d_r/d_tau: {d_r/d_tau}")
    print()
```

## Advanced Usage

### Multi-Scale Configurations

The τ-plane representation excels in handling multi-scale configurations:

```python
# Create a multi-scale configuration in r-plane
r_multi_scale = [
    [1.0, 0.0],           # Regular scale
    [1e-10, 0.0],         # Microscopic scale
    [1e10, 0.0]           # Cosmic scale
]

# Convert to τ-plane
tau_multi_scale = [r_to_tau(point) for point in r_multi_scale]
print("Multi-scale configuration in τ-plane:")
for point in tau_multi_scale:
    print(point)
    
# All points are now represented with similar magnitudes in τ-plane!
```

### Computing Vector Fields

Scale-shape decomposition in the τ-plane simplifies field calculations:

```python
import numpy as np
from mapping_functions import tau_to_r, r_to_tau

# Define a field in r-plane (e.g., gravitational field)
def gravity_field_r(r_coords):
    """Compute gravitational field at r-coordinates."""
    r_norm = np.sqrt(sum(x**2 for x in r_coords))
    if r_norm < 1e-10:  # Avoid singularity
        return [0.0, 0.0]
    # Field proportional to 1/r²
    field = [-x/(r_norm**3) for x in r_coords]
    return field

# Compute the same field in τ-plane
def gravity_field_tau(tau_coords):
    """Compute gravitational field at τ-coordinates."""
    # Convert to r-plane
    r_coords = tau_to_r(tau_coords)
    
    # Compute field in r-plane
    field_r = gravity_field_r(r_coords)
    
    # Transform field vector back to τ-plane (requires Jacobian of transformation)
    # For the reciprocal transformation, this simplifies considerably
    tau_norm_squared = sum(x**2 for x in tau_coords)
    field_tau = [-x * tau_norm_squared for x in field_r]
    
    return field_tau
```

### Creating a Uniform Grid

Working with grids is often easier in the τ-plane:

```python
import numpy as np
from mapping_functions import tau_to_r

# Create a uniform grid in τ-plane
tau_grid_x = np.linspace(0.1, 10.0, 100)
tau_grid_y = np.linspace(0.1, 10.0, 100)
tau_grid = np.meshgrid(tau_grid_x, tau_grid_y)

# Transform to r-plane
r_grid_x = np.zeros_like(tau_grid_x)
r_grid_y = np.zeros_like(tau_grid_y)

for i in range(len(tau_grid_x)):
    for j in range(len(tau_grid_y)):
        r_coords = tau_to_r([tau_grid_x[i], tau_grid_y[j]])
        r_grid_x[i, j] = r_coords[0]
        r_grid_y[i, j] = r_coords[1]

# The r-plane grid will be non-uniform with higher density near origin
```

## Examples and Practical Applications

### Handling the Three-Body Collapse Problem

The r-plane mapping helps avoid singularities in the classic three-body problem:

```python
from mapping_functions import r_to_tau, tau_to_r
import numpy as np

# Initial configuration in r-plane - three bodies on collision course
r_config = [
    [0.0, 0.0],    # Body 1 at origin
    [1.0, 0.0],    # Body 2 approaching
    [0.0, 1.0]     # Body 3 approaching
]

# Convert to τ-plane for simulation
tau_config = [r_to_tau(r) for r in r_config]

# As the bodies approach collision in r-plane (distance → 0)
# In τ-plane they move toward infinity (distance → ∞)
# This spreads out the singularity and makes it numerically tractable
```

### Simulating Expansive Systems

The τ-plane representation is ideal for systems that expand to infinity:

```python
from mapping_functions import r_to_tau, tau_to_r
import numpy as np

# Define expanding system in r-plane
def expanding_system_r(r_config, time):
    """Simulate expansion in r-plane."""
    # Simple expansion model: each point moves radially outward
    result = []
    for point in r_config:
        # Exponential expansion
        expanded = [x * np.exp(time) for x in point]
        result.append(expanded)
    return result

# The same system in τ-plane
def expanding_system_tau(tau_config, time):
    """Simulate expansion in τ-plane."""
    result = []
    for point in tau_config:
        # In τ-plane, expansion is a contraction toward origin
        contracted = [x * np.exp(-time) for x in point]
        result.append(contracted)
    return result

# The τ-plane representation remains bounded even as the
# r-plane representation approaches infinity
```

## See Also

- [Dynamics Engine Documentation](./dynamics.md)
- [Hyperreal Arithmetic Documentation](./hyperreal_arithmetic.md)
- [Extreme Collapse Benchmark](./extreme-collapse-scenario-analysis.md) 