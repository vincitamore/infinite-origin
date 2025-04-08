"""
Symbolic Hyperreal Number Implementation

This module provides a symbolic implementation of hyperreal numbers using SymPy.
It supports operations with infinitesimals and infinite values.
"""

from typing import Union, Optional
import sympy as sp
from sympy import Symbol, Expr, oo

class Hyperreal:
    """
    A symbolic hyperreal number representation.
    
    Attributes:
        value: The symbolic value of the hyperreal number
        infinitesimal: The infinitesimal component (if any)
    """
    
    def __init__(
        self, 
        value: Union[int, float, str, Symbol, Expr],
        infinitesimal: Optional[Symbol] = None
    ):
        """
        Initialize a hyperreal number.
        
        Args:
            value: The real part of the number
            infinitesimal: Optional infinitesimal component
        """
        if isinstance(value, (int, float)):
            self.value = sp.sympify(value)
        elif isinstance(value, str):
            self.value = sp.Symbol(value)
        else:
            self.value = value
            
        if infinitesimal is None:
            self.infinitesimal = sp.Symbol('eps', positive=True)
        else:
            self.infinitesimal = infinitesimal
            
    @classmethod
    def from_real(cls, value: Union[int, float]) -> 'Hyperreal':
        """Create a hyperreal from a real number."""
        return cls(value)
    
    @classmethod
    def infinitesimal(cls) -> 'Hyperreal':
        """Create a positive infinitesimal."""
        eps = sp.Symbol('eps', positive=True)
        return cls(eps)
    
    @classmethod
    def infinite(cls) -> 'Hyperreal':
        """Create an infinite value."""
        eps = sp.Symbol('eps', positive=True)
        return cls(1/eps)
    
    def __add__(self, other: Union['Hyperreal', int, float]) -> 'Hyperreal':
        """Add two hyperreal numbers."""
        if isinstance(other, (int, float)):
            other = Hyperreal.from_real(other)
        return Hyperreal(self.value + other.value)
    
    def __mul__(self, other: Union['Hyperreal', int, float]) -> 'Hyperreal':
        """Multiply two hyperreal numbers."""
        if isinstance(other, (int, float)):
            other = Hyperreal.from_real(other)
        return Hyperreal(self.value * other.value)
    
    def __truediv__(self, other: Union['Hyperreal', int, float]) -> 'Hyperreal':
        """Divide two hyperreal numbers."""
        if isinstance(other, (int, float)):
            other = Hyperreal.from_real(other)
        if other.value == 0:
            raise ValueError("Division by zero in hyperreal arithmetic")
        return Hyperreal(self.value / other.value)
    
    def __neg__(self) -> 'Hyperreal':
        """Negate a hyperreal number."""
        return Hyperreal(-self.value)
    
    def __sub__(self, other: Union['Hyperreal', int, float]) -> 'Hyperreal':
        """Subtract two hyperreal numbers."""
        return self + (-other)
    
    def __eq__(self, other: Union['Hyperreal', int, float]) -> bool:
        """Check if two hyperreal numbers are equal."""
        if isinstance(other, (int, float)):
            other = Hyperreal.from_real(other)
        return sp.simplify(self.value - other.value) == 0
    
    def __lt__(self, other: Union['Hyperreal', int, float]) -> bool:
        """Check if self is less than other."""
        if isinstance(other, (int, float)):
            other = Hyperreal.from_real(other)
        return sp.simplify(self.value - other.value) < 0
    
    def is_infinite(self) -> bool:
        """Check if the number is infinite."""
        # A number is infinite if it contains negative powers of the infinitesimal symbol
        expr = sp.simplify(self.value)
        if isinstance(expr, sp.Pow):
            base, exp = expr.as_base_exp()
            return base == self.infinitesimal and exp < 0
        return False
    
    def is_infinitesimal(self) -> bool:
        """Check if the number is infinitesimal."""
        # An infinitesimal is a value that, when simplified, contains only
        # positive powers of the infinitesimal symbol in the numerator
        expr = sp.simplify(self.value)
        if isinstance(expr, sp.Symbol) and expr.name == 'eps':
            return True
        if isinstance(expr, sp.Pow):
            base, exp = expr.as_base_exp()
            return base == self.infinitesimal and exp > 0
        return False
    
    def __str__(self) -> str:
        """String representation of the hyperreal number."""
        return str(self.value)
    
    def __repr__(self) -> str:
        """Detailed string representation of the hyperreal number."""
        return f"Hyperreal({self.value})" 