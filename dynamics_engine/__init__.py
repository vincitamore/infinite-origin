"""
Dynamics Simulation Engine Package

This package provides tools for simulating system dynamics over transformed time,
implementing the time transformation and integration capabilities defined in the
geometric framework with infinity at the origin.
"""

from .time_transform import compute_time_transformation, TimeTransformation
from .integrator import integrate, simulate

__all__ = ['compute_time_transformation', 'TimeTransformation', 'integrate', 'simulate'] 