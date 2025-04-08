"""
Singularity Handling

This module provides functions to handle special cases and singularities
in the mappings between τ-plane and r-plane.
"""

from typing import List, Union, Tuple, Optional
import numpy as np
from hyperreal_arithmetic import Hyperreal, HyperrealNum

def handle_origin_tau(
    tau_coords: Union[List[Hyperreal], List[HyperrealNum]],
    epsilon: Optional[float] = None
) -> Union[List[Hyperreal], List[HyperrealNum]]:
    """
    Handle points near the origin in τ-plane (corresponding to infinity in r-plane).
    
    Args:
        tau_coords: List of τ-plane coordinates
        epsilon: Optional small value for numerical computations
    
    Returns:
        Regularized coordinates that avoid the origin singularity
    """
    # For symbolic computations (Hyperreal)
    if all(isinstance(t, Hyperreal) for t in tau_coords):
        eps = Hyperreal.infinitesimal()
        return [t if not t.is_infinitesimal() else eps for t in tau_coords]
    
    # For numerical computations (HyperrealNum)
    if epsilon is None:
        epsilon = 1e-10
        
    result = []
    for tau in tau_coords:
        if isinstance(tau, (int, float)):
            if abs(tau) < epsilon:
                result.append(epsilon if tau >= 0 else -epsilon)
            else:
                result.append(tau)
        else:  # HyperrealNum
            if abs(tau.real_part) < epsilon and tau.inf_order <= 0:
                result.append(HyperrealNum(
                    epsilon if tau.real_part >= 0 else -epsilon,
                    0
                ))
            else:
                result.append(tau)
    return result

def handle_infinity_r(
    r_coords: Union[List[Hyperreal], List[HyperrealNum]],
    max_magnitude: Optional[float] = None
) -> Union[List[Hyperreal], List[HyperrealNum]]:
    """
    Handle points near infinity in r-plane (corresponding to origin in τ-plane).
    
    Args:
        r_coords: List of r-plane coordinates
        max_magnitude: Optional maximum allowed magnitude
    
    Returns:
        Regularized coordinates that avoid infinite values
    """
    # For symbolic computations (Hyperreal)
    if all(isinstance(r, Hyperreal) for r in r_coords):
        return [r if not r.is_infinite() else Hyperreal.infinite() for r in r_coords]
    
    # For numerical computations (HyperrealNum)
    if max_magnitude is None:
        max_magnitude = 1e10
        
    result = []
    for r in r_coords:
        if isinstance(r, (int, float)):
            if abs(r) > max_magnitude:
                result.append(max_magnitude if r > 0 else -max_magnitude)
            else:
                result.append(r)
        else:  # HyperrealNum
            if abs(r.real_part) > max_magnitude or r.inf_order > 0:
                result.append(HyperrealNum(
                    max_magnitude if r.real_part >= 0 else -max_magnitude,
                    1
                ))
            else:
                result.append(r)
    return result

def is_near_singularity(
    coords: Union[List[Hyperreal], List[HyperrealNum]],
    epsilon: Optional[float] = None
) -> bool:
    """
    Check if a point is near a singularity in either plane.
    
    Args:
        coords: List of coordinates to check
        epsilon: Optional threshold for "nearness"
    
    Returns:
        True if the point is near a singularity, False otherwise
    """
    if epsilon is None:
        epsilon = 1e-10
        
    for coord in coords:
        if isinstance(coord, (int, float)):
            if abs(coord) < epsilon:
                return True
        elif isinstance(coord, Hyperreal):
            if coord.is_infinitesimal() or coord.is_infinite():
                return True
        else:  # HyperrealNum
            if coord.inf_order != 0 or abs(coord.real_part) < epsilon:
                return True
    return False 