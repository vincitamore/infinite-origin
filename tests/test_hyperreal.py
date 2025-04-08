"""
Tests for hyperreal arithmetic implementations.
"""

import unittest
from hyperreal_arithmetic import Hyperreal, HyperrealNum
import sympy as sp

class TestHyperreal(unittest.TestCase):
    """Test cases for symbolic hyperreal implementation."""
    
    def test_creation(self):
        """Test creation of hyperreal numbers."""
        x = Hyperreal(2)
        self.assertEqual(str(x), "2")
        
        eps = Hyperreal.infinitesimal()
        self.assertTrue(eps.is_infinitesimal())
        
        inf = Hyperreal.infinite()
        self.assertTrue(inf.is_infinite())
    
    def test_arithmetic(self):
        """Test basic arithmetic operations."""
        x = Hyperreal(2)
        y = Hyperreal(3)
        
        self.assertEqual(str(x + y), "5")
        self.assertEqual(str(x * y), "6")
        self.assertEqual(str(-x), "-2")
        self.assertEqual(str(y / x), "3/2")
    
    def test_comparison(self):
        """Test comparison operations."""
        x = Hyperreal(2)
        y = Hyperreal(3)
        
        self.assertTrue(x < y)
        self.assertFalse(y < x)
        self.assertTrue(x == x)
        self.assertFalse(x == y)

class TestHyperrealNum(unittest.TestCase):
    """Test cases for numerical hyperreal implementation."""
    
    def test_creation(self):
        """Test creation of numerical hyperreal numbers."""
        x = HyperrealNum(2.0)
        self.assertEqual(str(x), "2.0")
        
        eps = HyperrealNum.infinitesimal()
        self.assertTrue(eps.is_infinitesimal())
        
        inf = HyperrealNum.infinite()
        self.assertTrue(inf.is_infinite())
    
    def test_arithmetic(self):
        """Test basic arithmetic operations."""
        x = HyperrealNum(2.0)
        y = HyperrealNum(3.0)
        
        self.assertEqual(str(x + y), "5.0")
        self.assertEqual(str(x * y), "6.0")
        self.assertEqual(str(-x), "-2.0")
        self.assertEqual(str(y / x), "1.5")
    
    def test_infinitesimal_arithmetic(self):
        """Test arithmetic with infinitesimals."""
        eps1 = HyperrealNum.infinitesimal(1)  # ε
        eps2 = HyperrealNum.infinitesimal(2)  # ε²
        
        # ε + ε² = ε (higher order infinitesimal is absorbed)
        result = eps1 + eps2
        self.assertEqual(result.inf_order, -1)
        
        # ε * ε = ε²
        result = eps1 * eps1
        self.assertEqual(result.inf_order, -2)
    
    def test_infinite_arithmetic(self):
        """Test arithmetic with infinite values."""
        inf1 = HyperrealNum.infinite(1)  # ∞
        inf2 = HyperrealNum.infinite(2)  # ∞²
        
        # ∞ + ∞² = ∞² (higher order infinity dominates)
        result = inf1 + inf2
        self.assertEqual(result.inf_order, 2)
        
        # ∞ * ∞ = ∞²
        result = inf1 * inf1
        self.assertEqual(result.inf_order, 2)

if __name__ == '__main__':
    unittest.main() 