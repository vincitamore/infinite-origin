"""
Hyperreal Arithmetic Package

This package provides tools for working with hyperreal numbers,
including both symbolic and numerical implementations.
"""

from .hyperreal import Hyperreal
from .numerical import HyperrealNum

__all__ = ['Hyperreal', 'HyperrealNum'] 