# Configuration Space Management

This module provides tools for managing multi-point systems in configuration space, with support for scale, shape, and orientation calculations.

## Basic Usage

### Creating Points

Points can be created with position coordinates, optional weights, and additional properties:

```python
from configuration_space import Point

# Create a 2D point
point_2d = Point([1.0, 0.0])

# Create a 3D point with weight and properties
point_3d = Point(
    position=[1.0, 0.0, 1.0],
    weight=2.0,
    properties={"mass": 1.0, "charge": -1.0}
)
```

### Working with Configurations

Configurations represent collections of points and provide methods for analyzing their geometric properties:

```python
from configuration_space import Configuration
import numpy as np

# Create a simple 2D configuration
p1 = Point([1.0, 0.0])
p2 = Point([-1.0, 0.0])
config = Configuration([p1, p2])

# Access basic properties
print(f"Center of mass: {config.center_of_mass}")
print(f"Scale factor: {config.scale_factor}")
print(f"Logarithmic scale (sigma): {config.sigma}")

# Get shape coordinates (normalized positions)
shape_coords = config.get_shape_coordinates()
print(f"Shape coordinates:\n{shape_coords}")

# Get orientation (2D only)
orientation = config.get_orientation()
print(f"Orientation angle: {np.degrees(orientation)} degrees")
```

## Advanced Usage

### Working with Weighted Points

Points can have different weights, affecting the configuration's properties:

```python
# Create a triangular configuration with different weights
p1 = Point([0.0, 0.0], weight=2.0)
p2 = Point([1.0, 0.0], weight=1.0)
p3 = Point([0.5, np.sqrt(3)/2], weight=1.0)

config = Configuration([p1, p2, p3])
print(f"Center of mass: {config.center_of_mass}")
```

### Fixing Center of Mass

You can adjust the configuration to place its center of mass at the origin:

```python
# Create an off-center configuration
p1 = Point([1.0, 1.0])
p2 = Point([2.0, 2.0])
config = Configuration([p1, p2])

print(f"Original center: {config.center_of_mass}")
config.fix_center_of_mass()
print(f"New center: {config.center_of_mass}")
```

### Adding Properties to Points

Points can carry additional properties useful for physics simulations or other applications:

```python
# Create particles with physical properties
electron = Point(
    position=[0.0, 0.0, 0.0],
    weight=1.0,
    properties={
        "mass": 9.1093837015e-31,  # kg
        "charge": -1.60217663e-19,  # C
        "spin": -0.5
    }
)

proton = Point(
    position=[1e-10, 0.0, 0.0],
    weight=1836.15,  # relative to electron
    properties={
        "mass": 1.67262192369e-27,  # kg
        "charge": 1.60217663e-19,   # C
        "spin": 0.5
    }
)

# Create hydrogen atom configuration
hydrogen = Configuration([electron, proton])
```

## Mathematical Background

The configuration space implementation is based on the following concepts:

1. **Scale Factor (s)**:
   - Computed as \( s = \sqrt{\frac{\sum w_i |\mathbf{r}_i|^2}{W}} \)
   - Where \( W = \sum w_i \) is the total weight

2. **Logarithmic Scale (σ)**:
   - Defined as \( \sigma = \ln(s) \)
   - Maps scale changes to the real line

3. **Shape Coordinates**:
   - Normalized positions relative to center of mass
   - Scale-invariant representation of the configuration

4. **Orientation (2D)**:
   - Computed using principal component analysis
   - Represents angle between first principal axis and x-axis 