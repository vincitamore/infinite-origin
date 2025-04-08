"""
Tests for visualization tools.

This module contains tests for the visualization tools implemented in
the visualization_tools module.
"""

import os
import numpy as np
import pytest
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import plotly.graph_objects as go

from configuration_space import Configuration, Point
from dynamics_engine.examples import harmonic_oscillator_example, three_body_example
from visualization_tools import (
    plot_configuration, 
    plot_configuration_comparison,
    plot_trajectory,
    plot_trajectory_shape
)


class TestStaticPlots:
    """Tests for static plotting functions."""
    
    @pytest.fixture
    def sample_config(self):
        """Create a sample configuration for testing."""
        p1 = Point([0, 0], weight=2.0)
        p2 = Point([1, 0], weight=1.0)
        p3 = Point([0, 1], weight=1.0)
        return Configuration([p1, p2, p3])
    
    def test_plot_configuration_matplotlib(self, sample_config):
        """Test the plot_configuration function with matplotlib."""
        # Test r-plane
        fig, ax = plot_configuration(
            sample_config, plane='r', use_plotly=False,
            figsize=(6, 4), show_scale=True
        )
        assert isinstance(fig, Figure)
        plt.close(fig)
        
        # Test tau-plane
        fig, ax = plot_configuration(
            sample_config, plane='tau', use_plotly=False,
            figsize=(6, 4), show_scale=False
        )
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_plot_configuration_plotly(self, sample_config):
        """Test the plot_configuration function with plotly."""
        # Test r-plane
        fig = plot_configuration(
            sample_config, plane='r', use_plotly=True,
            show_scale=True, point_labels=True
        )
        assert isinstance(fig, go.Figure)
        
        # Test tau-plane
        fig = plot_configuration(
            sample_config, plane='tau', use_plotly=True,
            show_scale=False, point_labels=False
        )
        assert isinstance(fig, go.Figure)
    
    def test_plot_configuration_comparison_matplotlib(self, sample_config):
        """Test the plot_configuration_comparison function with matplotlib."""
        fig, (ax1, ax2) = plot_configuration_comparison(
            sample_config, figsize=(10, 5), use_plotly=False,
            show_scale=True, point_labels=True
        )
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_plot_configuration_comparison_plotly(self, sample_config):
        """Test the plot_configuration_comparison function with plotly."""
        fig = plot_configuration_comparison(
            sample_config, use_plotly=True,
            show_scale=True, point_labels=True
        )
        assert isinstance(fig, go.Figure)


class TestTrajectoryPlots:
    """Tests for trajectory plotting functions."""
    
    @pytest.fixture
    def sample_trajectory(self):
        """Create a sample trajectory for testing."""
        # Run harmonic oscillator example to get a trajectory
        _, trajectory = harmonic_oscillator_example()
        return trajectory
    
    def test_plot_trajectory_matplotlib(self, sample_trajectory):
        """Test the plot_trajectory function with matplotlib."""
        fig, axes = plot_trajectory(
            sample_trajectory, time_var='tau', use_plotly=False,
            figsize=(8, 6), show_sigma=True, show_physical_time=True
        )
        assert isinstance(fig, Figure)
        assert len(axes) > 0
        plt.close(fig)
        
        fig, axes = plot_trajectory(
            sample_trajectory, time_var='t', use_plotly=False,
            figsize=(8, 6), show_sigma=False, show_physical_time=False
        )
        assert isinstance(fig, Figure)
        assert len(axes) > 0
        plt.close(fig)
    
    def test_plot_trajectory_plotly(self, sample_trajectory):
        """Test the plot_trajectory function with plotly."""
        fig = plot_trajectory(
            sample_trajectory, time_var='tau', use_plotly=True,
            show_sigma=True, show_physical_time=True
        )
        assert isinstance(fig, go.Figure)
        
        fig = plot_trajectory(
            sample_trajectory, time_var='t', use_plotly=True,
            show_sigma=False, show_physical_time=False
        )
        assert isinstance(fig, go.Figure)
    
    def test_plot_trajectory_shape(self, sample_trajectory):
        """Test the plot_trajectory_shape function."""
        # First add positions to the trajectory
        # Reconstructing positions from theta and sigma
        theta = sample_trajectory.get('theta', None)
        sigma = sample_trajectory.get('sigma', None)
        
        # Mock positions for the test
        n_steps = len(sigma)
        n_coords = theta.shape[1]
        dim = 2  # Assuming 2D
        n_points = n_coords // dim
        positions = np.zeros((n_steps, n_points, dim))
        
        for i in range(n_steps):
            scale = np.exp(sigma[i])
            points_at_t = theta[i].reshape(n_points, dim)
            positions[i] = points_at_t * scale
        
        # Add positions to trajectory
        sample_trajectory['positions'] = positions
        
        # Test with matplotlib
        fig = plot_trajectory_shape(
            sample_trajectory, num_points=5, use_plotly=False,
            figsize=(8, 6)
        )
        assert isinstance(fig, Figure)
        plt.close(fig)
        
        # Test with plotly
        fig = plot_trajectory_shape(
            sample_trajectory, num_points=5, use_plotly=True
        )
        assert isinstance(fig, go.Figure)


class TestAnimations:
    """Tests for animation functions."""
    
    @pytest.fixture
    def sample_trajectory(self):
        """Create a sample trajectory for testing."""
        # Run three body example to get a trajectory
        _, trajectory = three_body_example()
        
        # Reconstruct positions from theta and sigma
        theta = trajectory.get('theta', None)
        sigma = trajectory.get('sigma', None)
        
        n_steps = len(sigma)
        n_coords = theta.shape[1]
        dim = 2  # Assuming 2D
        n_points = n_coords // dim
        positions = np.zeros((n_steps, n_points, dim))
        
        for i in range(n_steps):
            scale = np.exp(sigma[i])
            points_at_t = theta[i].reshape(n_points, dim)
            positions[i] = points_at_t * scale
        
        # Add positions to trajectory
        trajectory['positions'] = positions
        
        return trajectory
    
    @pytest.mark.skip(reason="Animation tests are not run automatically")
    def test_animate_trajectory(self, sample_trajectory):
        """Test the animate_trajectory function (skipped in automated tests)."""
        from visualization_tools import animate_trajectory
        
        # Test basic animation
        anim = animate_trajectory(
            sample_trajectory, interval=100, figsize=(8, 6),
            show_time=True, plane='r'
        )
        assert anim is not None
    
    @pytest.mark.skip(reason="Animation tests are not run automatically")
    def test_animate_dual_view(self, sample_trajectory):
        """Test the animate_dual_view function (skipped in automated tests)."""
        from visualization_tools import animate_dual_view
        
        # Test dual view animation
        anim = animate_dual_view(
            sample_trajectory, interval=100, figsize=(12, 6),
            show_time=True
        )
        assert anim is not None


if __name__ == "__main__":
    # Run selected tests manually
    # Create a sample config
    p1 = Point([0, 0], weight=2.0)
    p2 = Point([1, 0], weight=1.0)
    p3 = Point([0, 1], weight=1.0)
    config = Configuration([p1, p2, p3])
    
    # Test static plots
    print("Testing static plots...")
    fig1, ax1 = plot_configuration(config, plane='r', use_plotly=False)
    plt.figure(fig1.number)
    plt.savefig('test_r_plane.png')
    plt.close(fig1)
    
    fig2, ax2 = plot_configuration(config, plane='tau', use_plotly=False)
    plt.figure(fig2.number)
    plt.savefig('test_tau_plane.png')
    plt.close(fig2)
    
    fig3, (ax3_1, ax3_2) = plot_configuration_comparison(config, use_plotly=False)
    plt.figure(fig3.number)
    plt.savefig('test_comparison.png')
    plt.close(fig3)
    
    # Test trajectory plots
    print("Testing trajectory plots...")
    _, trajectory = harmonic_oscillator_example()
    
    fig4, axes4 = plot_trajectory(trajectory, use_plotly=False)
    plt.figure(fig4.number)
    plt.savefig('test_trajectory.png')
    plt.close(fig4)
    
    # Add positions to trajectory for shape plot
    theta = trajectory.get('theta', None)
    sigma = trajectory.get('sigma', None)
    n_steps = len(sigma)
    n_coords = theta.shape[1]
    dim = 2
    n_points = n_coords // dim
    positions = np.zeros((n_steps, n_points, dim))
    for i in range(n_steps):
        scale = np.exp(sigma[i])
        points_at_t = theta[i].reshape(n_points, dim)
        positions[i] = points_at_t * scale
    trajectory['positions'] = positions
    
    fig5 = plot_trajectory_shape(trajectory, use_plotly=False, num_points=10)
    plt.figure(fig5.number)
    plt.savefig('test_shape_trajectory.png')
    plt.close(fig5)
    
    print("Tests completed and plots saved.") 