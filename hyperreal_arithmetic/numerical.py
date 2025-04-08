"""
Numerical Hyperreal Number Implementation

This module provides a numerical implementation of hyperreal numbers
for practical computations, using floating-point arithmetic with
explicit tracking of infinitesimal orders.
"""

from typing import Union, Tuple
import numpy as np
from math import isclose

class HyperrealNum:
    """
    A numerical hyperreal number representation.
    
    Attributes:
        real_part: The standard real part of the number
        inf_order: The order of infinity (negative for infinitesimals)
    """
    
    def __init__(self, real_part: float, inf_order: int = 0):
        """
        Initialize a numerical hyperreal number.
        
        Args:
            real_part: The real coefficient
            inf_order: Order of infinity (negative for infinitesimals)
        """
        self.real_part = float(real_part)
        self.inf_order = int(inf_order)
        
    @classmethod
    def from_real(cls, value: float) -> 'HyperrealNum':
        """Create a hyperreal from a real number."""
        return cls(value, 0)
    
    @classmethod
    def infinitesimal(cls, order: int = 1) -> 'HyperrealNum':
        """Create an infinitesimal of given order."""
        if order <= 0:
            raise ValueError("Infinitesimal order must be positive")
        return cls(1.0, -order)
    
    @classmethod
    def infinite(cls, order: int = 1) -> 'HyperrealNum':
        """Create an infinite value of given order."""
        if order <= 0:
            raise ValueError("Infinite order must be positive")
        return cls(1.0, order)
    
    def __add__(self, other: Union['HyperrealNum', float]) -> 'HyperrealNum':
        """Add two hyperreal numbers."""
        if isinstance(other, (int, float)):
            other = HyperrealNum.from_real(float(other))
            
        if self.inf_order == other.inf_order:
            return HyperrealNum(self.real_part + other.real_part, self.inf_order)
        elif self.inf_order > other.inf_order:
            return HyperrealNum(self.real_part, self.inf_order)
        else:
            return HyperrealNum(other.real_part, other.inf_order)
    
    def __mul__(self, other: Union['HyperrealNum', float]) -> 'HyperrealNum':
        """Multiply two hyperreal numbers."""
        if isinstance(other, (int, float)):
            other = HyperrealNum.from_real(float(other))
        return HyperrealNum(
            self.real_part * other.real_part,
            self.inf_order + other.inf_order
        )
    
    def __truediv__(self, other: Union['HyperrealNum', float]) -> 'HyperrealNum':
        """Divide two hyperreal numbers."""
        if isinstance(other, (int, float)):
            other = HyperrealNum.from_real(float(other))
        if isclose(other.real_part, 0.0):
            raise ValueError("Division by zero in hyperreal arithmetic")
        return HyperrealNum(
            self.real_part / other.real_part,
            self.inf_order - other.inf_order
        )
    
    def __neg__(self) -> 'HyperrealNum':
        """Negate a hyperreal number."""
        return HyperrealNum(-self.real_part, self.inf_order)
    
    def __sub__(self, other: Union['HyperrealNum', float]) -> 'HyperrealNum':
        """Subtract two hyperreal numbers."""
        return self + (-other)
    
    def __eq__(self, other: Union['HyperrealNum', float]) -> bool:
        """Check if two hyperreal numbers are equal."""
        if isinstance(other, (int, float)):
            other = HyperrealNum.from_real(float(other))
        return (isclose(self.real_part, other.real_part) and 
                self.inf_order == other.inf_order)
    
    def __lt__(self, other: Union['HyperrealNum', float]) -> bool:
        """Check if self is less than other."""
        if isinstance(other, (int, float)):
            other = HyperrealNum.from_real(float(other))
        if self.inf_order != other.inf_order:
            return self.inf_order < other.inf_order
        return self.real_part < other.real_part
    
    def is_infinite(self) -> bool:
        """Check if the number is infinite."""
        return self.inf_order > 0
    
    def is_infinitesimal(self) -> bool:
        """Check if the number is infinitesimal."""
        return self.inf_order < 0
    
    def __str__(self) -> str:
        """String representation of the hyperreal number."""
        if self.inf_order == 0:
            return str(self.real_part)
        elif self.inf_order > 0:
            return f"{self.real_part}∞^{self.inf_order}"
        else:
            return f"{self.real_part}ε^{-self.inf_order}"
    
    def __repr__(self) -> str:
        """Detailed string representation of the hyperreal number."""
        return f"HyperrealNum({self.real_part}, {self.inf_order})" 