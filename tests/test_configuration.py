import unittest
import numpy as np
from configuration_space.point import Point
from configuration_space.configuration import Configuration

class TestConfiguration(unittest.TestCase):
    def setUp(self):
        """Set up test configurations."""
        # Simple 2D configuration
        self.p1 = Point([1.0, 0.0], weight=1.0)
        self.p2 = Point([-1.0, 0.0], weight=1.0)
        self.config_2d = Configuration([self.p1, self.p2])

        # 3D configuration with different weights
        self.p3 = Point([1.0, 0.0, 0.0], weight=2.0)
        self.p4 = Point([0.0, 1.0, 0.0], weight=1.0)
        self.p5 = Point([0.0, 0.0, 1.0], weight=1.0)
        self.config_3d = Configuration([self.p3, self.p4, self.p5])

    def test_init_empty(self):
        """Test that empty configuration raises ValueError."""
        with self.assertRaises(ValueError):
            Configuration([])

    def test_init_different_dimensions(self):
        """Test that points with different dimensions raise ValueError."""
        p1 = Point([1.0, 0.0])
        p2 = Point([1.0, 0.0, 0.0])
        with self.assertRaises(ValueError):
            Configuration([p1, p2])

    def test_center_of_mass_2d(self):
        """Test center of mass calculation for 2D configuration."""
        np.testing.assert_array_almost_equal(
            self.config_2d.center_of_mass,
            np.array([0.0, 0.0])
        )

    def test_center_of_mass_3d_weighted(self):
        """Test center of mass calculation for 3D weighted configuration."""
        expected_com = np.array([0.5, 0.25, 0.25])  # Due to weights
        np.testing.assert_array_almost_equal(
            self.config_3d.center_of_mass,
            expected_com
        )

    def test_scale_factor_2d(self):
        """Test scale factor calculation for 2D configuration."""
        # For points at (-1,0) and (1,0), scale factor should be 1
        self.assertAlmostEqual(self.config_2d.scale_factor, 1.0)
        self.assertAlmostEqual(self.config_2d.sigma, 0.0)

    def test_shape_coordinates(self):
        """Test shape coordinate calculation."""
        shape_coords = self.config_2d.get_shape_coordinates()
        expected = np.array([[1.0, 0.0], [-1.0, 0.0]])
        np.testing.assert_array_almost_equal(shape_coords, expected)

    def test_orientation_2d(self):
        """Test orientation calculation for 2D configuration."""
        # For points at (-1,0) and (1,0), principal axis should align with x-axis
        self.assertAlmostEqual(self.config_2d.get_orientation(), 0.0)

        # Create a configuration rotated 45 degrees
        p1 = Point([1.0, 1.0])
        p2 = Point([-1.0, -1.0])
        config_45 = Configuration([p1, p2])
        self.assertAlmostEqual(config_45.get_orientation(), np.pi/4)

    def test_orientation_3d(self):
        """Test orientation calculation for 3D configuration."""
        # Test 1: Points along x-axis
        p1 = Point([1.0, 0.0, 0.0])
        p2 = Point([-1.0, 0.0, 0.0])
        config = Configuration([p1, p2])
        phi, theta, psi = config.get_orientation()
        # Principal axis should align with x-axis
        self.assertAlmostEqual(theta, np.pi/2)  # 90 degrees from z-axis
        self.assertAlmostEqual(phi, 0.0)        # No rotation around z-axis needed
        self.assertAlmostEqual(psi, 0.0)        # No final rotation needed

        # Test 2: Points along diagonal in xy-plane
        p1 = Point([1.0, 1.0, 0.0])
        p2 = Point([-1.0, -1.0, 0.0])
        config = Configuration([p1, p2])
        phi, theta, psi = config.get_orientation()
        # Principal axis should be 45 degrees in xy-plane
        self.assertAlmostEqual(theta, np.pi/2)  # 90 degrees from z-axis
        self.assertAlmostEqual(phi, np.pi/4)    # 45 degrees in xy-plane
        self.assertAlmostEqual(psi, 0.0)        # No final rotation needed

        # Test 3: Points in general position
        p1 = Point([1.0, 1.0, 1.0])
        p2 = Point([-1.0, -1.0, -1.0])
        config = Configuration([p1, p2])
        phi, theta, psi = config.get_orientation()
        # Just verify we get valid angles (actual values depend on convention)
        self.assertTrue(-np.pi <= phi <= np.pi)
        self.assertTrue(0 <= theta <= np.pi)
        self.assertTrue(-np.pi <= psi <= np.pi)

    def test_orientation_invalid_dimension(self):
        """Test that orientation calculation raises error for invalid dimensions."""
        # Create a 4D point configuration
        p1 = Point([1.0, 0.0, 0.0, 0.0])
        p2 = Point([-1.0, 0.0, 0.0, 0.0])
        config = Configuration([p1, p2])
        with self.assertRaises(ValueError):
            config.get_orientation()

    def test_fix_center_of_mass(self):
        """Test fixing center of mass at origin."""
        self.config_3d.fix_center_of_mass()
        np.testing.assert_array_almost_equal(
            self.config_3d.center_of_mass,
            np.zeros(3)
        )

    def test_repr(self):
        """Test string representation."""
        # Just verify it contains essential information
        repr_str = repr(self.config_2d)
        self.assertIn("scale=", repr_str)
        self.assertIn("sigma=", repr_str)

if __name__ == '__main__':
    unittest.main() 