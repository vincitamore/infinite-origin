# Hyperreal Arithmetic Documentation

This document provides an overview of the hyperreal arithmetic capabilities in the Infinite Origin framework.

## Overview

The hyperreal arithmetic module implements operations with hyperreal numbers, which extend the real numbers to include infinitesimals (positive numbers smaller than any real number) and infinities (larger than any real number). This extension provides a rigorous foundation for handling extreme scale phenomena that occur in simulations of physical systems operating at the edge of numerical precision.

The module provides two implementations:
1. **Symbolic Hyperreal (Hyperreal)**: For mathematical rigor and symbolic manipulation
2. **Numerical Hyperreal (HyperrealNum)**: For efficient computation in simulation contexts

## Core Concepts

### Infinitesimals

An infinitesimal is a number that is:
- Positive (or negative)
- Smaller in absolute value than any positive real number
- Not equal to zero

Infinitesimals are represented as ε (epsilon) in mathematical notation and can be raised to powers to represent different orders of infinitesimals.

### Infinities

Hyperreal infinities are reciprocals of infinitesimals and represent numbers larger than any real number. They can have different orders, providing a more nuanced representation of infinite quantities than traditional approaches.

## Symbolic Hyperreal Implementation

The `Hyperreal` class provides a symbolic implementation using SymPy, ideal for formal proofs and analytical applications.

### Creating Hyperreal Values

```python
from hyperreal_arithmetic import Hyperreal

# Standard creation from values
x = Hyperreal(3.5)
y = Hyperreal("y")  # Symbolic variable

# Create a positive infinitesimal
epsilon = Hyperreal.infinitesimal()  # Creates the symbolic infinitesimal ε

# Create an infinite value
infinity = Hyperreal.infinite()  # Represents 1/ε

# Convert from a real number
z = Hyperreal.from_real(42)
```

### Arithmetic Operations

```python
# Addition and subtraction
sum_result = x + y
diff_result = x - y

# Multiplication and division
product = x * y
quotient = x / y  # Raises ValueError if y.value == 0

# Comparison operations
equals = x == y
less_than = x < y

# Negation
neg_x = -x
```

### Special Properties

```python
# Check if a value is infinite
if infinity.is_infinite():
    print("Value is infinite")
    
# Check if a value is infinitesimal
if epsilon.is_infinitesimal():
    print("Value is infinitesimal")
    
# Representation
print(str(epsilon))  # String representation
print(repr(epsilon))  # Detailed representation
```

## Numerical Hyperreal Implementation

The `HyperrealNum` class provides a computational implementation designed for efficiency in numerical simulations.

### Creating Numerical Hyperreals

```python
from hyperreal_arithmetic import HyperrealNum

# Standard creation with real part and infinity order
# real_part * (infinity^inf_order)
a = HyperrealNum(3.0, 0)        # Just the real number 3.0
b = HyperrealNum(2.5, 1)        # 2.5∞ (first order infinity)
c = HyperrealNum(1.0, -1)       # 1.0ε (first order infinitesimal)
d = HyperrealNum(4.0, -2)       # 4.0ε² (second order infinitesimal)

# Factory methods
standard = HyperrealNum.from_real(3.14)
epsilon = HyperrealNum.infinitesimal(order=1)  # First order infinitesimal
infinity = HyperrealNum.infinite(order=2)      # Second order infinity
```

### Arithmetic Operations

```python
# Addition follows dominance rules
sum1 = HyperrealNum(3.0, 2) + HyperrealNum(5.0, 2)   # 8.0∞²
sum2 = HyperrealNum(3.0, 2) + HyperrealNum(5.0, 1)   # 3.0∞² (higher order dominates)
sum3 = HyperrealNum(3.0, -1) + HyperrealNum(5.0, 0)  # 5.0 (real dominates infinitesimal)

# Multiplication combines orders
product1 = HyperrealNum(2.0, 1) * HyperrealNum(3.0, 2)  # 6.0∞³
product2 = HyperrealNum(2.0, -1) * HyperrealNum(3.0, 2) # 6.0∞¹ (∞² * ε = ∞)

# Division
quotient1 = HyperrealNum(4.0, 3) / HyperrealNum(2.0, 1)  # 2.0∞²
quotient2 = HyperrealNum(4.0, 2) / HyperrealNum(2.0, -1) # 2.0∞³ (∞²/ε = ∞³)

# Equality and comparison
equals = HyperrealNum(3.0, 1) == HyperrealNum(3.0, 1)  # True
less_than = HyperrealNum(3.0, 1) < HyperrealNum(2.0, 2)  # True (order dominates)
```

### Special Properties

```python
# Check if infinite or infinitesimal
if infinity.is_infinite():
    print("Value is infinite")
    
if epsilon.is_infinitesimal():
    print("Value is infinitesimal")

# String representations
print(str(HyperrealNum(2.5, 0)))   # "2.5"
print(str(HyperrealNum(2.0, 2)))   # "2.0∞^2"
print(str(HyperrealNum(3.0, -1)))  # "3.0ε^1"
```

## Use Cases

### Singularity Handling

```python
# Handle division by zero-like situations
x = HyperrealNum(1.0, 0)
y = HyperrealNum.infinitesimal()
result = x / y  # Results in a finite, well-defined infinity

# Analyze behavior near singularities
singularity_approach = []
for k in range(1, 10):
    eps = HyperrealNum(1.0, -k)
    singularity_approach.append(1.0 / eps)
    
# Result contains well-defined infinities of increasing order
```

### Multi-scale Physics

```python
# Model phenomena at vastly different scales
galaxy_scale = HyperrealNum(1.0, 20)  # Galactic scale
atomic_scale = HyperrealNum(1.0, -10)  # Atomic scale
ratio = galaxy_scale / atomic_scale    # Well-defined ratio of 10^30

# Perform calculations across these scales
force = HyperrealNum(6.67e-11, 0) * mass1 * mass2 / (distance * distance)
# Force remains well-defined even as distance approaches zero
```

### Integration with Dynamics Engine

```python
from dynamics_engine import TimeTransformation
from hyperreal_arithmetic import HyperrealNum

# Define time transformation with hyperreal support
def f_sigma(sigma):
    if isinstance(sigma, HyperrealNum) and sigma.is_infinitesimal():
        # Special handling for infinitesimal scale
        return HyperrealNum(-1.0, 0)
    return -sigma
    
transform = TimeTransformation(f_sigma)

# Now transform works correctly even at singular points
singularity_time_factor = transform.dt_dtau(HyperrealNum.infinitesimal())
```

## Advanced Topics

### Error Analysis

Hyperreal arithmetic can be used for rigorous error analysis:

```python
from hyperreal_arithmetic import Hyperreal

# Define a function
def f(x):
    return x**2 + 3*x + 2

# Compute derivative using infinitesimals
def df_dx(x_0):
    epsilon = Hyperreal.infinitesimal()
    x = Hyperreal(x_0) + epsilon
    return (f(x) - f(x_0)) / epsilon

# The result gives the exact derivative
derivative_at_3 = df_dx(3)  # 2*3 + 3 = 9
```

### Non-standard Analysis

```python
# Define an infinite integer
omega = HyperrealNum(1.0, 1)

# The transfer principle ensures that properties of real numbers
# extend to hyperreals in a consistent way
is_greater = omega > 1000000  # True, infinity is greater than any real number
```

## See Also

- [Dynamics Engine Documentation](./dynamics.md)
- [Mapping Functions Documentation](./mapping_functions.md)
- [Extreme Collapse Benchmark](./extreme-collapse-scenario-analysis.md) 