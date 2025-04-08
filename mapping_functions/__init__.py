"""
Mapping Functions Package

This package provides functions for mapping between the τ-plane and r-plane,
including distance calculations and singularity handling.
"""

from .plane_maps import tau_to_r, r_to_tau, compute_distance_tau, compute_distance_r
from .singularities import handle_origin_tau, handle_infinity_r, is_near_singularity

__all__ = [
    'tau_to_r',
    'r_to_tau',
    'compute_distance_tau',
    'compute_distance_r',
    'handle_origin_tau',
    'handle_infinity_r',
    'is_near_singularity'
] 