"""
Performance Profiling Script

This script profiles the performance of key components of the Infinite Origin framework,
identifying bottlenecks and opportunities for optimization.
"""

import cProfile
import pstats
import time
import numpy as np
import os
import matplotlib.pyplot as plt
from io import StringIO

from configuration_space import Configuration, Point
from dynamics_engine import TimeTransformation, simulate
from visualization_tools import plot_configuration, animate_trajectory
from hyperreal_arithmetic.numerical import HyperrealNum
from mapping_functions import tau_to_r, r_to_tau


def profile_hyperreal_arithmetic(n_iterations=10000):
    """Profile hyperreal arithmetic operations."""
    print("\n=== Profiling Hyperreal Arithmetic ===")
    
    # Setup hyperreal numbers
    a = HyperrealNum(1.0, 0.1)
    b = HyperrealNum(2.0, 0.2)
    
    # Profile addition
    start_time = time.time()
    for _ in range(n_iterations):
        c = a + b
    add_time = time.time() - start_time
    print(f"Addition: {add_time:.6f} seconds for {n_iterations} iterations")
    
    # Profile multiplication
    start_time = time.time()
    for _ in range(n_iterations):
        c = a * b
    mul_time = time.time() - start_time
    print(f"Multiplication: {mul_time:.6f} seconds for {n_iterations} iterations")
    
    # Profile division
    start_time = time.time()
    for _ in range(n_iterations):
        c = a / b
    div_time = time.time() - start_time
    print(f"Division: {div_time:.6f} seconds for {n_iterations} iterations")


def profile_mapping_functions(n_iterations=10000):
    """Profile mapping between τ-plane and r-plane."""
    print("\n=== Profiling Mapping Functions ===")
    
    # Setup coordinates
    tau_coords = [HyperrealNum(1.0, 0), HyperrealNum(2.0, 0)]
    
    # Profile tau to r mapping
    start_time = time.time()
    for _ in range(n_iterations):
        r_coords = tau_to_r(tau_coords)
    tau_to_r_time = time.time() - start_time
    print(f"tau_to_r: {tau_to_r_time:.6f} seconds for {n_iterations} iterations")
    
    # Profile r to tau mapping
    r_coords = tau_to_r(tau_coords)
    start_time = time.time()
    for _ in range(n_iterations):
        tau_coords_2 = r_to_tau(r_coords)
    r_to_tau_time = time.time() - start_time
    print(f"r_to_tau: {r_to_tau_time:.6f} seconds for {n_iterations} iterations")


def profile_configuration_operations(n_iterations=1000):
    """Profile configuration space operations."""
    print("\n=== Profiling Configuration Operations ===")
    
    # Setup
    n_points = 10
    points = [
        Point([np.cos(2*np.pi*i/n_points), np.sin(2*np.pi*i/n_points)]) 
        for i in range(n_points)
    ]
    
    # Profile configuration creation
    start_time = time.time()
    for _ in range(n_iterations):
        config = Configuration(points)
    create_time = time.time() - start_time
    print(f"Configuration creation: {create_time:.6f} seconds for {n_iterations} iterations")
    
    # Profile fixing center of mass
    config = Configuration(points)
    start_time = time.time()
    for _ in range(n_iterations):
        config.fix_center_of_mass()
    fix_cm_time = time.time() - start_time
    print(f"Fixing center of mass: {fix_cm_time:.6f} seconds for {n_iterations} iterations")
    
    # Profile scale and shape calculations
    config = Configuration(points)
    start_time = time.time()
    for _ in range(n_iterations):
        s = config.scale_factor
        sigma = config.sigma
        theta = config.get_shape_coordinates()
    calc_time = time.time() - start_time
    print(f"Scale/shape calculation: {calc_time:.6f} seconds for {n_iterations} iterations")


def profile_dynamics_simulation():
    """Profile dynamics simulation."""
    print("\n=== Profiling Dynamics Simulation ===")
    
    # Create profiler
    pr = cProfile.Profile()
    pr.enable()
    
    # Setup simple configuration
    p1 = Point([0, 0], weight=2.0)
    p2 = Point([1, 0], weight=1.0)
    p3 = Point([0, 1], weight=1.0)
    config = Configuration([p1, p2, p3])
    config.fix_center_of_mass()
    
    # Setup simulation
    F = lambda s: -np.exp(s)
    transform = TimeTransformation(lambda s: s/2)
    
    # Run simulation
    print("Running simulation...")
    tau_max = 5.0
    final_config, trajectory = simulate(
        config, F, tau_max=tau_max, num_steps=200, 
        time_transform=transform
    )
    
    # Disable profiler and print stats
    pr.disable()
    s = StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(20)  # Print top 20 functions by cumulative time
    print(s.getvalue())
    
    return final_config, trajectory


def profile_visualization(config, trajectory):
    """Profile visualization operations."""
    print("\n=== Profiling Visualization ===")
    
    # Create profiler
    pr = cProfile.Profile()
    pr.enable()
    
    # Profile static plotting
    print("Testing static plotting...")
    fig, ax = plot_configuration(config, plane='r', show_scale=True)
    plt.close(fig)
    
    # Profile animation (without display)
    print("Testing animation generation...")
    anim = animate_trajectory(
        trajectory,
        interval=100,
        figsize=(8, 6),
        plane='r',
        show_time=False  # Don't show time to simplify
    )
    
    # Disable profiler and print stats
    pr.disable()
    s = StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(20)  # Print top 20 functions by cumulative time
    print(s.getvalue())


def find_optimization_opportunities():
    """Find and report optimization opportunities."""
    print("\n=== Optimization Opportunities ===")
    
    # Check vectorization opportunities
    print("Checking for vectorization opportunities:")
    print("- Consider replacing loops in configuration.py with NumPy operations")
    print("- Use array operations where possible in integrator.py")
    
    # Check for memory usage
    print("\nMemory usage optimization:")
    print("- Consider using generators for large trajectories")
    print("- Implement selective trajectory storage to reduce memory")
    
    # Check for algorithm optimizations
    print("\nAlgorithm optimizations:")
    print("- Consider implementing adaptive step size in integration")
    print("- Check for redundant calculations in visualization tools")


if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
    
    print("=" * 70)
    print("Performance Profiling for Infinite Origin Framework")
    print("=" * 70)
    
    # Profile basic operations
    profile_hyperreal_arithmetic()
    profile_mapping_functions()
    profile_configuration_operations()
    
    # Profile dynamics and visualization
    final_config, trajectory = profile_dynamics_simulation()
    profile_visualization(final_config, trajectory)
    
    # Find optimization opportunities
    find_optimization_opportunities()
    
    print("=" * 70)
    print("Profiling complete! Check output for detailed results.")
    print("=" * 70) 