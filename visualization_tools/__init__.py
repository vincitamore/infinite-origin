"""
Visualization Tools Module

This module provides visualization tools for the Infinite Origin framework,
allowing users to visualize configurations and trajectories in both τ-plane 
and r-plane representations.
"""

from .static_plots import plot_configuration, plot_configuration_comparison
from .trajectory_plots import plot_trajectory, plot_trajectory_shape
from .animations import animate_trajectory, animate_dual_view

__all__ = [
    'plot_configuration',
    'plot_configuration_comparison',
    'plot_trajectory',
    'plot_trajectory_shape',
    'animate_trajectory',
    'animate_dual_view'
] 