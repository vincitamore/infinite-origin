# Infinite Origin Framework

A computational framework implementing a geometric system with infinity at the origin, based on hyperreal numbers and multi-plane mappings.

## Overview

The Infinite Origin Framework provides tools for simulating physical systems across extreme scales with unprecedented numerical stability and computational efficiency. It's particularly effective for scenarios involving:

- **Multi-scale phenomena**: Systems spanning many orders of magnitude in scale
- **Gravitational dynamics**: N-body simulations without traditional softening issues
- **Singular configurations**: Direct collisions and other near-singular scenarios
- **High-precision simulations**: Maintaining energy conservation across extreme scale changes

Core components include:
- Hyperreal arithmetic operations
- τ-plane to r-plane mappings
- Configuration space management
- Dynamic system simulations with time transformation
- Visualization tools

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/infinite-origin.git
cd infinite-origin
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # Windows PowerShell
# Or for Linux/macOS:
# source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

- `hyperreal_arithmetic/` - Implementation of hyperreal number operations
  - `hyperreal.py` - Symbolic hyperreal implementation
  - `numerical.py` - Numerical hyperreal implementation for efficient computation
- `mapping_functions/` - Plane-to-plane transformation functions
  - `plane_maps.py` - Core mapping functions between τ-plane and r-plane
  - `singularities.py` - Handling of special cases and singularities
- `configuration_space/` - Multi-point system configuration management
- `dynamics_engine/` - Time evolution simulations
  - `integrator.py` - Numerical integration algorithms
  - `time_transformation.py` - Scale-dependent time transformations
  - `examples.py` - Pre-configured simulation examples
- `visualization_tools/` - Plotting and animation utilities
- `tests/` - Unit and integration tests
- `docs/` - Documentation
  - `usage/` - Usage guides and tutorials
  - `examples/` - Example simulations and benchmarks

## Usage

### Basic Example

```python
from main import main

if __name__ == "__main__":
    main()
```

### Running Simulations
The main entry point supports running different example simulations:

```bash
# Run the harmonic oscillator example
python main.py --example harmonic

# Run the three-body example
python main.py --example three-body

# Run the collapsing system example
python main.py --example collapse

# Run the extreme collapse benchmark
python main.py --example extreme-collapse

# Run all examples
python main.py --example all
```

### Using the Dynamics Engine

The dynamics engine enables simulating systems over transformed time:

```python
from configuration_space import Configuration, Point
from dynamics_engine import TimeTransformation, simulate
import numpy as np

# Create a configuration of points
p1 = Point([0, 0], weight=2.0)
p2 = Point([1, 0], weight=1.0)
config = Configuration([p1, p2])

# Define a driving function
F = lambda sigma: -np.exp(2*sigma)  # Harmonic oscillator-like

# Create a time transformation
transform = TimeTransformation(lambda sigma: sigma/2)

# Run a simulation
final_config, trajectory = simulate(
    config, F, tau_max=10.0, num_steps=500, time_transform=transform
)

# Access trajectory data
sigma_values = trajectory['sigma']  # Scale evolution
t_values = trajectory['t']          # Physical time values
```

### Hyperreal Arithmetic

The framework provides both symbolic and numerical hyperreal arithmetic:

```python
from hyperreal_arithmetic import Hyperreal, HyperrealNum

# Symbolic hyperreals for mathematical proofs
epsilon = Hyperreal.infinitesimal()  # Infinitesimal value
infinity = Hyperreal.infinite()      # Infinite value
result = 1 + epsilon                 # 1 + ε

# Numerical hyperreals for efficient computation
a = HyperrealNum(3.0, 1)    # 3.0∞¹ (3.0 times first-order infinity)
b = HyperrealNum(2.0, -1)   # 2.0ε¹ (2.0 times first-order infinitesimal)
product = a * b             # 6.0∞⁰ (6.0 times infinity⁰)
```

### τ-Plane Mappings

Transform between traditional coordinates (r-plane) and the τ-plane where infinity is at the origin:

```python
from mapping_functions import tau_to_r, r_to_tau

# Convert from r-plane to τ-plane
r_coords = [1.0, 0.0]           # Point at distance 1 from origin
tau_coords = r_to_tau(r_coords)  # [1.0, 0.0] in τ-plane

# Convert coordinates approaching infinity to finite τ-plane values
r_huge = [1.0e15, 0.0]          # Extremely distant point
tau_tiny = r_to_tau(r_huge)     # Represented as point near origin

# Handle singularities explicitly
from mapping_functions import handle_origin_tau, handle_infinity_r
safe_coords = handle_infinity_r([1.0e20, 0])  # Regularized representation
```

### Custom Driving Functions

You can create custom driving functions to model different physical systems:

```python
# Harmonic oscillator: F(σ) = -e^(2σ)
F_harmonic = lambda s: -np.exp(2*s)

# Gravitational-like: F(σ) = -e^σ
F_gravity = lambda s: -np.exp(s)

# Collapsing system: F(σ) = -2e^σ
F_collapse = lambda s: -2 * np.exp(s)

# Extreme collapse with regularization:
F_extreme = lambda s: -8.0 * np.exp(s) * (1.0 - np.exp(-0.01 * s**2))
```

## Performance Benchmarks

The Infinite Origin Framework demonstrates exceptional performance in extreme scenarios like gravitational collapse:

- **21.4× faster** than traditional N-body methods in extreme collapse scenarios
- **55.6% better energy conservation** compared to traditional methods
- **Several orders of magnitude better scale handling** for multi-scale systems

For detailed benchmarks, see [Extreme Collapse Benchmark](docs/usage/extreme-collapse-scenario-analysis.md).

## Documentation

Explore the full documentation:

- [Hyperreal Arithmetic](docs/usage/hyperreal_arithmetic.md)
- [Mapping Functions](docs/usage/mapping_functions.md)
- [Dynamics Engine](docs/usage/dynamics.md)
- [Extreme Collapse Benchmark](docs/usage/extreme-collapse-scenario-analysis.md)

## Development

To contribute to the project:
1. Create a new branch for your feature
2. Make your changes
3. Run tests: `pytest`
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 