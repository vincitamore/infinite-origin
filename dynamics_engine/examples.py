"""
Dynamics Engine Examples

This module provides ready-to-use examples to demonstrate the dynamics engine functionality.
"""

import numpy as np
from configuration_space import Configuration, Point
from dynamics_engine import TimeTransformation, integrate, simulate


def harmonic_oscillator_example():
    """
    Simulate a simple harmonic oscillator using a two-point configuration.
    
    Returns:
        Tuple of (final_config, trajectory)
    """
    print("Simulating harmonic oscillator...")
    
    # Create two points: one at origin, one at (1, 0)
    p1 = Point([0, 0])
    p2 = Point([1, 0])
    config = Configuration([p1, p2])
    
    # Fix center of mass at origin
    config.fix_center_of_mass()
    
    print(f"Initial configuration: {config}")
    print(f"Initial sigma: {config.sigma:.4f}")
    print("Initial positions:")
    for i, p in enumerate(config.points):
        print(f"  Point {i+1}: {p.position}")
    
    # Harmonic driving function: F(σ) = -e^(2σ)
    # This gives a Hooke's Law-like force
    F = lambda s: -np.exp(2*s)
    
    # Create time transformation with f(σ) = σ
    # This means dt/dτ = e^σ
    transform = TimeTransformation(lambda s: s)
    
    # Integrate for τ ∈ [0, 2π] (one full oscillation in τ-time)
    tau_max = 2 * np.pi
    final_config, trajectory = simulate(
        config, F, tau_max=tau_max, num_steps=200, time_transform=transform
    )
    
    print("\nSimulation complete!")
    print(f"Final configuration: {final_config}")
    print(f"Final sigma: {final_config.sigma:.4f}")
    print("Final positions:")
    for i, p in enumerate(final_config.points):
        print(f"  Point {i+1}: {p.position}")
    
    print(f"\ntau-time range: [0, {tau_max:.2f}]")
    print(f"Physical time range: [0, {trajectory['t'][-1]:.4f}]")
    
    # Print a few key points in the trajectory
    num_steps = len(trajectory['tau'])
    points_to_print = min(5, num_steps)
    step_size = max(1, num_steps // points_to_print)
    
    print("\nTrajectory highlights:")
    print("  tau      sigma        t")
    print("------------------------")
    for i in range(0, num_steps, step_size):
        tau = trajectory['tau'][i]
        sigma = trajectory['sigma'][i]
        t = trajectory['t'][i]
        print(f"  {tau:.2f}   {sigma:.4f}   {t:.4f}")
    
    return final_config, trajectory


def three_body_example():
    """
    Simulate a simple three-body system.
    
    Returns:
        Tuple of (final_config, trajectory)
    """
    print("Simulating three-body system...")
    
    # Create three points in a triangular configuration
    p1 = Point([0, 0], weight=2.0)        # Heavier central mass
    p2 = Point([1, 0], weight=1.0)        # Orbiting body 1
    p3 = Point([0, 1], weight=1.0)        # Orbiting body 2
    config = Configuration([p1, p2, p3])
    
    # Fix center of mass at origin
    config.fix_center_of_mass()
    
    print(f"Initial configuration: {config}")
    print(f"Initial sigma: {config.sigma:.4f}")
    print("Initial positions:")
    for i, p in enumerate(config.points):
        print(f"  Point {i+1}: {p.position} (weight: {p.weight})")
    
    # Gravitational-like driving function
    # F(σ) = -e^σ to simulate inverse square attraction
    F = lambda s: -np.exp(s)
    
    # Create time transformation with f(σ) = σ/2
    # This regularizes the dynamics (dt/dτ = e^(σ/2))
    transform = TimeTransformation(lambda s: s/2)
    
    # Integrate for τ ∈ [0, 10]
    tau_max = 10.0
    final_config, trajectory = simulate(
        config, F, tau_max=tau_max, num_steps=500, time_transform=transform
    )
    
    print("\nSimulation complete!")
    print(f"Final configuration: {final_config}")
    print(f"Final sigma: {final_config.sigma:.4f}")
    print("Final positions:")
    for i, p in enumerate(final_config.points):
        print(f"  Point {i+1}: {p.position} (weight: {p.weight})")
    
    print(f"\ntau-time range: [0, {tau_max:.2f}]")
    print(f"Physical time range: [0, {trajectory['t'][-1]:.4f}]")
    
    return final_config, trajectory


def collapsing_system_example():
    """
    Simulate a collapsing system to demonstrate handling of approaching singularities.
    
    Returns:
        Tuple of (final_config, trajectory)
    """
    print("Simulating collapsing system...")
    
    # Create a simple configuration of points
    p1 = Point([-1, -1])
    p2 = Point([1, -1])
    p3 = Point([0, 1])
    config = Configuration([p1, p2, p3])
    
    # Original scale factor
    print(f"Initial configuration: {config}")
    print(f"Initial sigma: {config.sigma:.4f}")
    
    # Driving function with strong inward collapse
    # F(σ) = -2e^(σ) to make the system collapse rapidly
    F = lambda s: -2 * np.exp(s)
    
    # Time transformation with f(σ) = -σ
    # This means dt/dτ = e^(-σ), which slows physical time as σ decreases
    transform = TimeTransformation(lambda s: -s)
    
    # Integrate for τ ∈ [0, 5]
    tau_max = 5.0
    final_config, trajectory = simulate(
        config, F, tau_max=tau_max, num_steps=200, time_transform=transform
    )
    
    print("\nSimulation complete!")
    print(f"Final configuration: {final_config}")
    print(f"Final sigma: {final_config.sigma:.4f}")
    
    print(f"\ntau-time range: [0, {tau_max:.2f}]")
    print(f"Physical time range: [0, {trajectory['t'][-1]:.4f}]")
    
    return final_config, trajectory


if __name__ == "__main__":
    # Run a simple example
    harmonic_oscillator_example()
    print("\n" + "="*50 + "\n")
    three_body_example()
    print("\n" + "="*50 + "\n")
    collapsing_system_example() 