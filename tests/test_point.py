import unittest
import numpy as np
from configuration_space.point import Point

class TestPoint(unittest.TestCase):
    def test_init_with_list(self):
        """Test Point initialization with a list of coordinates."""
        point = Point([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(point.position, np.array([1.0, 2.0, 3.0]))
        self.assertEqual(point.weight, 1.0)
        self.assertEqual(point.properties, {})

    def test_init_with_numpy_array(self):
        """Test Point initialization with a numpy array."""
        pos = np.array([1.0, 2.0])
        point = Point(pos, weight=2.0)
        np.testing.assert_array_equal(point.position, pos)
        self.assertEqual(point.weight, 2.0)

    def test_init_with_properties(self):
        """Test Point initialization with properties."""
        props = {"mass": 1.0, "charge": -1.0}
        point = Point([0.0, 0.0], properties=props)
        self.assertEqual(point.properties, props)

    def test_invalid_weight(self):
        """Test that negative weight raises ValueError."""
        with self.assertRaises(ValueError):
            Point([1.0, 1.0], weight=-1.0)
        with self.assertRaises(ValueError):
            Point([1.0, 1.0], weight=0.0)

    def test_dimension(self):
        """Test dimension calculation."""
        point_2d = Point([1.0, 2.0])
        point_3d = Point([1.0, 2.0, 3.0])
        self.assertEqual(point_2d.dimension(), 2)
        self.assertEqual(point_3d.dimension(), 3)

    def test_repr(self):
        """Test string representation."""
        point = Point([1.0, 2.0], weight=2.0, properties={"mass": 1.0})
        expected = "Point(position=[1.0, 2.0], weight=2.0, properties={'mass': 1.0})"
        self.assertEqual(repr(point), expected)

if __name__ == '__main__':
    unittest.main() 