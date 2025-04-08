"""
Tests for mapping functions between τ-plane and r-plane.
"""

import unittest
import numpy as np
from hyperreal_arithmetic import Hyperreal, HyperrealNum
from mapping_functions import (
    tau_to_r,
    r_to_tau,
    compute_distance_tau,
    compute_distance_r,
    handle_origin_tau,
    handle_infinity_r,
    is_near_singularity
)

class TestMappingFunctions(unittest.TestCase):
    """Test cases for plane mapping functions."""
    
    def test_tau_to_r_basic(self):
        """Test basic τ to r mapping."""
        tau_coords = [Hyperreal(2), Hyperreal(3)]
        r_coords = tau_to_r(tau_coords)
        
        self.assertEqual(str(r_coords[0]), "1/2")
        self.assertEqual(str(r_coords[1]), "1/3")
    
    def test_r_to_tau_basic(self):
        """Test basic r to τ mapping."""
        r_coords = [Hyperreal(2), Hyperreal(3)]
        tau_coords = r_to_tau(r_coords)
        
        self.assertEqual(str(tau_coords[0]), "1/2")
        self.assertEqual(str(tau_coords[1]), "1/3")
    
    def test_mapping_composition(self):
        """Test that r_to_tau(tau_to_r(x)) = x."""
        tau_coords = [Hyperreal(2), Hyperreal(3)]
        composed = r_to_tau(tau_to_r(tau_coords))
        
        for original, mapped in zip(tau_coords, composed):
            self.assertTrue(original == mapped)
    
    def test_distance_tau(self):
        """Test distance calculation in τ-plane."""
        tau1 = [HyperrealNum(0.0, 1), HyperrealNum(0.0, 1)]  # Point at infinity
        tau2 = [HyperrealNum(1.0, 0), HyperrealNum(1.0, 0)]  # Finite point
        
        dist = compute_distance_tau(tau1, tau2)
        self.assertTrue(dist.is_infinite())
    
    def test_distance_r(self):
        """Test distance calculation in r-plane."""
        r1 = [HyperrealNum(1.0, 0), HyperrealNum(0.0, 0)]
        r2 = [HyperrealNum(0.0, 0), HyperrealNum(1.0, 0)]
        
        dist = compute_distance_r(r1, r2)
        self.assertEqual(dist.real_part, np.sqrt(2))
        self.assertEqual(dist.inf_order, 0)
    
    def test_handle_origin_tau(self):
        """Test handling of points near origin in τ-plane."""
        tau_coords = [HyperrealNum(1e-15, 0), HyperrealNum(1.0, 0)]
        regularized = handle_origin_tau(tau_coords)
        
        self.assertGreater(abs(regularized[0].real_part), 1e-15)
        self.assertEqual(regularized[1].real_part, 1.0)
    
    def test_handle_infinity_r(self):
        """Test handling of infinite points in r-plane."""
        r_coords = [HyperrealNum(1e15, 0), HyperrealNum(1.0, 0)]
        regularized = handle_infinity_r(r_coords)
        
        self.assertLess(abs(regularized[0].real_part), 1e15)
        self.assertEqual(regularized[1].real_part, 1.0)
    
    def test_singularity_detection(self):
        """Test detection of points near singularities."""
        # Point near origin in τ-plane
        tau_near_origin = [HyperrealNum(1e-15, 0)]
        self.assertTrue(is_near_singularity(tau_near_origin))
        
        # Regular point
        regular_point = [HyperrealNum(1.0, 0)]
        self.assertFalse(is_near_singularity(regular_point))
        
        # Infinite point
        infinite_point = [HyperrealNum(1.0, 1)]  # Order 1 infinity
        self.assertTrue(is_near_singularity(infinite_point))

if __name__ == '__main__':
    unittest.main() 