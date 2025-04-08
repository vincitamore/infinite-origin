"""
Plane Mapping Functions

This module provides the core functions for mapping between the τ-plane
and r-plane, including distance calculations in both spaces.
"""

from typing import List, Union, Tuple
import numpy as np
from hyperreal_arithmetic import Hyperreal, HyperrealNum

def tau_to_r(
    tau_coords: Union[List[Hyperreal], List[HyperrealNum]]
) -> List[Union[Hyperreal, HyperrealNum]]:
    """
    Map coordinates from τ-plane to r-plane.
    
    Args:
        tau_coords: List of τ-plane coordinates [τ₁, τ₂, ..., τₙ]
    
    Returns:
        List of r-plane coordinates [r₁, r₂, ..., rₙ]
        where rᵢ = 1/τᵢ for each i
    
    Raises:
        ValueError: If any τᵢ = 0
    """
    result = []
    for tau in tau_coords:
        if isinstance(tau, (int, float)):
            if tau == 0:
                raise ValueError("Cannot map τ = 0 to r-plane (corresponds to infinity)")
            result.append(Hyperreal(1) / Hyperreal(tau))
        elif hasattr(tau, 'value'):  # Handle symbolic Hyperreal
            if tau.value == 0:
                raise ValueError("Cannot map τ = 0 to r-plane (corresponds to infinity)")
            result.append(Hyperreal(1) / tau)
        elif hasattr(tau, 'real_part'):  # Handle numerical HyperrealNum
            if tau.real_part == 0 and tau.inf_order == 0:
                raise ValueError("Cannot map τ = 0 to r-plane (corresponds to infinity)")
            result.append(HyperrealNum(1) / tau)
        else:
            raise TypeError(f"Unsupported type for tau_coords: {type(tau)}")
    return result

def r_to_tau(
    r_coords: Union[List[Hyperreal], List[HyperrealNum]]
) -> List[Union[Hyperreal, HyperrealNum]]:
    """
    Map coordinates from r-plane to τ-plane.
    
    Args:
        r_coords: List of r-plane coordinates [r₁, r₂, ..., rₙ]
    
    Returns:
        List of τ-plane coordinates [τ₁, τ₂, ..., τₙ]
        where τᵢ = 1/rᵢ for each i
    
    Raises:
        ValueError: If any rᵢ = 0
    """
    result = []
    for r in r_coords:
        if isinstance(r, (int, float)):
            if r == 0:
                raise ValueError("Cannot map r = 0 to τ-plane (corresponds to infinity)")
            result.append(Hyperreal(1) / Hyperreal(r))
        elif hasattr(r, 'value'):  # Handle symbolic Hyperreal
            if r.value == 0:
                raise ValueError("Cannot map r = 0 to τ-plane (corresponds to infinity)")
            result.append(Hyperreal(1) / r)
        elif hasattr(r, 'real_part'):  # Handle numerical HyperrealNum
            if r.real_part == 0 and r.inf_order == 0:
                raise ValueError("Cannot map r = 0 to τ-plane (corresponds to infinity)")
            result.append(HyperrealNum(1) / r)
        else:
            raise TypeError(f"Unsupported type for r_coords: {type(r)}")
    return result

def compute_distance_tau(
    tau1: Union[List[Hyperreal], List[HyperrealNum]],
    tau2: Union[List[Hyperreal], List[HyperrealNum]]
) -> Union[Hyperreal, HyperrealNum]:
    """
    Compute Euclidean distance between two points in τ-plane.
    
    Args:
        tau1: First point coordinates [τ₁₁, τ₁₂, ..., τ₁ₙ]
        tau2: Second point coordinates [τ₂₁, τ₂₂, ..., τ₂ₙ]
    
    Returns:
        Euclidean distance √(Σ(τ₁ᵢ - τ₂ᵢ)²)
    """
    if len(tau1) != len(tau2):
        raise ValueError("Points must have same dimension")
        
    # Use the same type (Hyperreal or HyperrealNum) as the inputs
    squared_sum = None
    for t1, t2 in zip(tau1, tau2):
        if isinstance(t1, (int, float)):
            t1 = Hyperreal(t1)
        if isinstance(t2, (int, float)):
            t2 = Hyperreal(t2)
        diff = t1 - t2
        term = diff * diff
        if squared_sum is None:
            squared_sum = term
        else:
            squared_sum = squared_sum + term
            
    # For HyperrealNum, we can use numerical methods
    if isinstance(squared_sum, HyperrealNum):
        return HyperrealNum(
            np.sqrt(squared_sum.real_part),
            squared_sum.inf_order // 2
        )
    # For Hyperreal, we use SymPy's sqrt
    else:
        from sympy import sqrt
        return Hyperreal(sqrt(squared_sum.value))

def compute_distance_r(
    r1: Union[List[Hyperreal], List[HyperrealNum]],
    r2: Union[List[Hyperreal], List[HyperrealNum]]
) -> Union[Hyperreal, HyperrealNum]:
    """
    Compute Euclidean distance between two points in r-plane.
    
    Args:
        r1: First point coordinates [r₁₁, r₁₂, ..., r₁ₙ]
        r2: Second point coordinates [r₂₁, r₂₂, ..., r₂ₙ]
    
    Returns:
        Euclidean distance √(Σ(r₁ᵢ - r₂ᵢ)²)
    """
    if len(r1) != len(r2):
        raise ValueError("Points must have same dimension")
        
    # Use the same type (Hyperreal or HyperrealNum) as the inputs
    squared_sum = None
    for r1_i, r2_i in zip(r1, r2):
        if isinstance(r1_i, (int, float)):
            r1_i = Hyperreal(r1_i)
        if isinstance(r2_i, (int, float)):
            r2_i = Hyperreal(r2_i)
        diff = r1_i - r2_i
        term = diff * diff
        if squared_sum is None:
            squared_sum = term
        else:
            squared_sum = squared_sum + term
            
    # For HyperrealNum, we can use numerical methods
    if isinstance(squared_sum, HyperrealNum):
        return HyperrealNum(
            np.sqrt(squared_sum.real_part),
            squared_sum.inf_order // 2
        )
    # For Hyperreal, we use SymPy's sqrt
    else:
        from sympy import sqrt
        return Hyperreal(sqrt(squared_sum.value)) 