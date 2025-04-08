"""
Time Transformation Module

This module implements the time transformation capability as defined in Axiom 16:
dt/dτ = e^(f(σ)), allowing for regularization of dynamics.
"""

from typing import Callable, Union, Optional, Dict, Any
import numpy as np
from hyperreal_arithmetic import Hyperreal, HyperrealNum
from sympy import Symbol, exp, lambdify
import sympy as sp


class TimeTransformation:
    """
    Implements the time transformation defined by dt/dτ = e^(f(σ)).
    
    This class handles the transformation between physical time t and
    regularized time τ, based on a sigma-dependent function f(σ).
    
    Attributes:
        f_sigma (Callable): The function f(σ) used in the transformation
        homogeneity_degree (int): The homogeneity degree in derivatives
    """
    
    def __init__(
        self, 
        f_sigma: Callable[[float], float], 
        homogeneity_degree: int = 2
    ) -> None:
        """
        Initialize a TimeTransformation with the specified f(σ) function.
        
        Args:
            f_sigma: Function f(σ) used in dt/dτ = e^(f(σ))
            homogeneity_degree: Homogeneity degree in derivatives (default: 2)
                For mechanical systems, typically 2 for second-order equations
        """
        self.f_sigma = f_sigma
        self.homogeneity_degree = homogeneity_degree
        
    def dt_dtau(self, sigma: float) -> float:
        """
        Compute dt/dτ for a given value of σ.
        
        Args:
            sigma: The logarithmic scale coordinate
            
        Returns:
            The time transformation factor dt/dτ
        """
        return np.exp(self.f_sigma(sigma))
    
    def dtau_dt(self, sigma: float) -> float:
        """
        Compute dτ/dt for a given value of σ.
        
        Args:
            sigma: The logarithmic scale coordinate
            
        Returns:
            The inverse time transformation factor dτ/dt
        """
        return 1.0 / self.dt_dtau(sigma)
    
    def integrate_transformation(
        self, 
        sigma_values: np.ndarray, 
        tau_values: np.ndarray
    ) -> np.ndarray:
        """
        Integrate the time transformation to get t values from τ values.
        
        Args:
            sigma_values: Array of σ values along the trajectory
            tau_values: Array of τ values corresponding to sigma_values
            
        Returns:
            Array of t values corresponding to the input τ values
        """
        # Compute dt/dτ at each step
        dt_dtau_values = np.array([self.dt_dtau(s) for s in sigma_values])
        
        # Initialize t array
        t_values = np.zeros_like(tau_values)
        
        # Integrate dt/dτ = e^(f(σ)) using trapezoidal rule
        for i in range(1, len(tau_values)):
            delta_tau = tau_values[i] - tau_values[i-1]
            avg_dt_dtau = (dt_dtau_values[i] + dt_dtau_values[i-1]) / 2
            t_values[i] = t_values[i-1] + avg_dt_dtau * delta_tau
            
        return t_values
    
    @classmethod
    def from_driving_function(
        cls, 
        driving_function: Callable[[float], float],
        homogeneity_degree: int = 2,
        regularization_strategy: str = 'exponential'
    ) -> 'TimeTransformation':
        """
        Create a TimeTransformation from a driving function F(σ).
        
        For systems where F ~ e^(kσ)G(θ,φ), we set f(σ) = (k/m)σ
        where m is the homogeneity degree in derivatives.
        
        Args:
            driving_function: The driving function F(σ) to regularize
            homogeneity_degree: Homogeneity degree in derivatives (default: 2)
            regularization_strategy: Strategy for finding k ('exponential', 'symbolic', or 'numerical')
            
        Returns:
            A TimeTransformation instance that regularizes the driving function
        """
        if regularization_strategy == 'exponential':
            # Analyze the driving function to estimate k
            sigma_values = np.linspace(-2, 2, 100)
            f_values = np.array([driving_function(s) for s in sigma_values])
            
            # Check if the function is expansion-type (mostly positive)
            is_expansion = np.mean(f_values) > 0
            
            # Find k by linear regression on log(|F|) vs σ
            valid_indices = np.abs(f_values) > 1e-10  # Need non-zero values for log
            if np.sum(valid_indices) < 2:
                # Default to k=1 if not enough valid points
                k = 1
            else:
                log_f = np.log(np.abs(f_values[valid_indices]))
                slope, _ = np.polyfit(sigma_values[valid_indices], log_f, 1)
                k = slope
                
            # For expansion dynamics (positive k), we need stronger regularization
            if is_expansion and k > 0:
                # Use f(σ) = (k/m)σ - 0.1σ²
                f_sigma = lambda s: (k / homogeneity_degree) * s - 0.1 * s**2
            else:
                # Create f(σ) = (k/m)σ
                f_sigma = lambda s: (k / homogeneity_degree) * s
            
        elif regularization_strategy == 'symbolic':
            # Use SymPy for symbolic analysis
            sigma = Symbol('sigma')
            expr = driving_function(sigma)
            
            # Try to find the coefficient of e^(kσ)
            try:
                k = expr.as_coefficient(exp(sigma))
                if k is None:  # Not a simple exponential
                    k = 1  # Default
            except Exception:
                k = 1  # Default if symbolic analysis fails
                
            # Check if k is positive (expansion-like)
            if k > 0:
                f_sigma = lambda s: (k / homogeneity_degree) * s - 0.1 * s**2
            else:
                f_sigma = lambda s: (k / homogeneity_degree) * s
            
        else:  # 'numerical' or other
            # Default: assume k=1 with quadratic damping for stability
            f_sigma = lambda s: (1 / homogeneity_degree) * s - 0.05 * s**2
            
        return cls(f_sigma, homogeneity_degree)


def compute_time_transformation(
    sigma: Union[float, np.ndarray, Hyperreal, HyperrealNum], 
    f_sigma: Callable[[Union[float, Hyperreal, HyperrealNum]], 
                      Union[float, Hyperreal, HyperrealNum]]
) -> Union[float, np.ndarray, Hyperreal, HyperrealNum]:
    """
    Compute the time transformation factor dt/dτ = e^(f(σ)).
    
    This implementation supports both standard floating-point and hyperreal numbers.
    
    Args:
        sigma: The logarithmic scale coordinate (scalar or array)
        f_sigma: Function f(σ) used in the transformation
        
    Returns:
        The time transformation factor dt/dτ (same type as input)
    """
    if isinstance(sigma, (Hyperreal, HyperrealNum)):
        # Use hyperreal arithmetic
        return sp.exp(f_sigma(sigma))
    elif isinstance(sigma, np.ndarray):
        # Element-wise calculation for arrays
        return np.exp(np.vectorize(f_sigma)(sigma))
    else:
        # Standard floating-point
        return np.exp(f_sigma(sigma)) 