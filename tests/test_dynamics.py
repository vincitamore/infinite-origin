"""
Test the dynamics simulation engine.
"""

import unittest
import sys
import os
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_allclose
import sympy as sp

# Add parent directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hyperreal_arithmetic import Hyperreal, HyperrealNum
from configuration_space import Configuration, Point
from dynamics_engine import TimeTransformation, compute_time_transformation, integrate, simulate


class TestTimeTransformation(unittest.TestCase):
    """Test time transformation functionality."""
    
    def test_basic_transformation(self):
        """Test basic time transformation calculations."""
        # Create a simple f(σ) = -σ
        f_sigma = lambda s: -s
        
        # Create transformation
        transform = TimeTransformation(f_sigma)
        
        # Test dt/dτ calculation
        self.assertAlmostEqual(transform.dt_dtau(0), 1.0)  # e^0 = 1
        self.assertAlmostEqual(transform.dt_dtau(1), np.exp(-1))
        self.assertAlmostEqual(transform.dt_dtau(-2), np.exp(2))
        
        # Test dτ/dt calculation (inverse)
        self.assertAlmostEqual(transform.dtau_dt(0), 1.0)
        self.assertAlmostEqual(transform.dtau_dt(1), np.exp(1))
        self.assertAlmostEqual(transform.dtau_dt(-2), np.exp(-2))
    
    def test_driving_function_analysis(self):
        """Test creation from driving function."""
        # Create a simple exponential driving function
        F = lambda s: 2.5 * np.exp(3 * s)
        
        # Create transformation
        transform = TimeTransformation.from_driving_function(
            F, homogeneity_degree=2, regularization_strategy='exponential'
        )
        
        # The function should approximately identify k=3
        # So f_sigma should be approximately (3/2)σ
        expected_value = 3/2 * 1.5  # For sigma=1.5
        actual_value = transform.f_sigma(1.5)
        self.assertAlmostEqual(actual_value, expected_value, places=1)
    
    def test_integration(self):
        """Test time integration."""
        f_sigma = lambda s: -s  # Simple case where dt/dτ = e^(-σ)
        transform = TimeTransformation(f_sigma)
        
        # Create sigma and tau arrays
        tau_values = np.linspace(0, 1, 5)
        # Constant sigma = 1 for simplicity
        sigma_values = np.ones_like(tau_values)
        
        # Integrate
        t_values = transform.integrate_transformation(sigma_values, tau_values)
        
        # Expected: t(τ) = (1/e) * τ when σ = 1
        expected_t = np.exp(-1) * tau_values
        assert_array_almost_equal(t_values, expected_t)


class TestHarmonicOscillator(unittest.TestCase):
    """Test integration for a simple harmonic oscillator."""
    
    def test_harmonic_oscillator(self):
        """Test integration of a harmonic oscillator."""
        # Create a two-point configuration
        # One at origin, one at (1, 0)
        p1 = Point([0, 0])
        p2 = Point([1, 0])
        config = Configuration([p1, p2])
        
        # Simple harmonic driving function
        # F(σ) = -e^(2σ), so we should get a harmonic oscillator in tau time
        F = lambda s: -np.exp(2*s)
        
        # Create transformation with f(σ) = σ (so dt/dτ = e^σ)
        transform = TimeTransformation(lambda s: s)
        
        # Integrate for τ ∈ [0, 2π]
        tau_max = 2 * np.pi
        results = integrate(config, F, tau_max, 100, transform, False)
        
        # In τ-time, we should get approximately sinusoidal behavior
        # due to the harmonic potential
        tau_values = results['tau']
        sigma_values = results['sigma']
        
        # Just check that we have the expected keys in the results
        expected_keys = ['tau', 'sigma', 'dsigma_dtau', 'theta', 'phi', 't']
        for key in expected_keys:
            self.assertIn(key, results)
            
        # Verify trajectory has right shape
        self.assertEqual(len(results['tau']), len(results['sigma']))
        self.assertEqual(len(results['tau']), len(results['t']))
        
        # Verify we have reasonable sigma values (shouldn't diverge to infinity)
        self.assertTrue(np.all(np.isfinite(sigma_values)))
    
    def test_simulate_function(self):
        """Test the simulate function to ensure it returns a valid configuration."""
        # Create a three-point configuration in 2D
        p1 = Point([0, 0])
        p2 = Point([1, 0])
        p3 = Point([0, 1])
        config = Configuration([p1, p2, p3])
        
        # A simple driving function
        F = lambda s: -s
        
        # Run simulation
        final_config, trajectory = simulate(config, F, tau_max=2.0, num_steps=50)
        
        # Verify the final configuration is valid
        self.assertEqual(len(final_config.points), 3)
        self.assertEqual(final_config.points[0].dimension(), 2)
        
        # Verify trajectory contains expected keys
        expected_keys = ['tau', 'sigma', 'dsigma_dtau', 'theta', 'phi', 't']
        for key in expected_keys:
            self.assertIn(key, trajectory)
        
        # Verify trajectory lengths match
        tau_length = len(trajectory['tau'])
        self.assertGreater(tau_length, 0)
        self.assertEqual(len(trajectory['sigma']), tau_length)
        self.assertEqual(len(trajectory['t']), tau_length)


if __name__ == '__main__':
    unittest.main() 