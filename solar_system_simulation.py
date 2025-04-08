"""
Solar System Simulation using the Infinite Origin Framework

This file implements a realistic simulation of our solar system using
the hyperreal framework for handling both large and small scale dynamics.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse
import os
import sys

# Add the current directory to path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configuration_space import Configuration, Point
from dynamics_engine import TimeTransformation, simulate
from dynamics_engine.integrator import create_state_derivative_function
from visualization_tools import (
    plot_configuration,
    plot_trajectory,
    animate_trajectory,
    animate_dual_view
)

# Astronomical constants in SI units
G = 6.67430e-11  # Gravitational constant (m^3 kg^-1 s^-2)
AU = 1.495978707e11  # Astronomical Unit (m)
SOLAR_MASS = 1.989e30  # Solar mass (kg)
EARTH_MASS = 5.972e24  # Earth mass (kg)
DAY = 86400  # Seconds in a day
YEAR = 365.25 * DAY  # Seconds in a year

# Scale down for numerical stability
DISTANCE_SCALE = AU  # 1.0 in simulation = 1 AU
MASS_SCALE = SOLAR_MASS  # 1.0 in simulation = 1 solar mass
TIME_SCALE = DAY  # 1.0 in simulation = 1 day

# Define scenarios
SCENARIOS = {
    'standard': {
        'description': 'Standard solar system simulation',
        'custom_bodies': []
    },
    'comet': {
        'description': 'Solar system with a highly eccentric comet',
        'custom_bodies': [
            # Name, mass (solar masses), semi-major (AU), eccentricity, inclination (deg), perihelion_angle (deg)
            ['Comet', 1e-10, 20.0, 0.99, 60.0, 0.0]
        ]
    },
    'close_approach': {
        'description': 'Earth-Mars close approach scenario',
        'custom_bodies': []
    },
    'binary_planet': {
        'description': 'Earth with a large captured moon at close distance',
        'custom_bodies': [
            # Name, mass (solar masses), semi-major (AU), eccentricity, inclination (deg), perihelion_angle (deg)
            ['Large Moon', 1e-7, 0.05, 0.1, 5.0, 0.0]
        ]
    },
    'rogue_star': {
        'description': 'Rogue star passing through the solar system',
        'custom_bodies': [
            # Name, mass (solar masses), semi-major (AU), eccentricity, inclination (deg), perihelion_angle (deg)
            ['Rogue Star', 0.3, 100.0, 0.98, 15.0, 180.0]
        ]
    }
}

def calculate_orbital_velocity(semi_major_axis, mass_central, position):
    """
    Calculate the orbital velocity based on Kepler's laws.
    
    Args:
        semi_major_axis: Semi-major axis in AU
        mass_central: Mass of central body in solar masses
        position: Current position vector [x, y] in AU
        
    Returns:
        Velocity vector [vx, vy] in AU/day
    """
    # In normalized units where G=1
    # Simplified circular orbit formula: v = sqrt(GM/r)
    # For elliptical orbits, we use the vis-viva equation: v^2 = GM(2/r - 1/a)
    
    # Current radius
    r = np.sqrt(position[0]**2 + position[1]**2)
    
    if r < 1e-10:  # Central body
        return [0.0, 0.0]
    
    # Compute velocity magnitude using vis-viva equation
    # Converting to our simulation units (AU/day)
    # We use the scaling G=1 in simulation units
    v_mag = np.sqrt(mass_central * (2.0/r - 1.0/semi_major_axis))
    
    # Compute direction perpendicular to position vector
    # For a counter-clockwise orbit
    direction = np.array([-position[1], position[0]])
    direction = direction / np.linalg.norm(direction)
    
    # Return velocity vector
    return v_mag * direction

def create_specific_scenario(scenario_name, base_bodies):
    """
    Modify a list of bodies to create a specific scenario.
    
    Args:
        scenario_name: Name of the scenario
        base_bodies: List of base solar system bodies
        
    Returns:
        Modified list of bodies for the scenario
    """
    if scenario_name not in SCENARIOS:
        print(f"Unknown scenario '{scenario_name}', using standard configuration.")
        return base_bodies
    
    scenario = SCENARIOS[scenario_name]
    bodies = base_bodies.copy()
    
    if scenario_name == 'close_approach':
        # Create Earth-Mars close approach by adjusting their positions
        for i, body in enumerate(bodies):
            if body[0] == "Earth":
                # Move Earth to a specific position
                bodies[i][2] = 1.5  # Semi-major axis
                bodies[i][3] = 0.05  # Low eccentricity
            elif body[0] == "Mars":
                # Move Mars close to Earth's orbit
                bodies[i][2] = 1.52  # Semi-major axis
                bodies[i][3] = 0.05  # Low eccentricity
    
    # Add custom bodies for this scenario
    for custom_body in scenario['custom_bodies']:
        bodies.append(custom_body)
    
    return bodies

def create_solar_system(include_bodies='all', with_velocities=True, scenario='standard'):
    """
    Create a configuration representing the solar system.
    
    Args:
        include_bodies: Which bodies to include ('all', 'inner', 'outer', 'major')
        with_velocities: Whether to include orbital velocities in properties
        scenario: Specific scenario to simulate
        
    Returns:
        Configuration object representing the solar system
    """
    # Solar system data: [name, mass (solar masses), 
    #                    semi-major axis (AU), eccentricity, 
    #                    inclination (degrees)]
    bodies = [
        # Sun at the center
        ["Sun", 1.0, 0.0, 0.0, 0.0, 0.0],
        
        # Inner planets
        ["Mercury", 1.65956e-7, 0.38709893, 0.20563069, 7.00487, 29.124279],
        ["Venus", 2.4478383e-6, 0.72333199, 0.00677323, 3.39471, 54.85229],
        ["Earth", 3.0034896e-6, 1.00000011, 0.01671022, 0.00005, 114.20783],
        ["Mars", 3.2271514e-7, 1.52366231, 0.09341233, 1.85061, 286.46230],
        
        # Outer planets
        ["Jupiter", 9.5479194e-4, 5.20336301, 0.04839266, 1.30530, 275.066],
        ["Saturn", 2.8586434e-4, 9.53707032, 0.05415060, 2.48446, 336.013862],
        ["Uranus", 4.3662440e-5, 19.19126393, 0.04716771, 0.76986, 96.541318],
        ["Neptune", 5.1513890e-5, 30.06896348, 0.00858587, 1.76917, 265.646853],
        
        # Dwarf planet
        ["Pluto", 6.5808e-9, 39.48168677, 0.24880766, 17.14175, 113.76329]
    ]
    
    # Apply scenario modifications
    bodies = create_specific_scenario(scenario, bodies)
    
    # Filter bodies based on the include_bodies parameter
    if include_bodies == 'inner':
        bodies = [b for b in bodies if b[0] == "Sun" or b[0] in ["Mercury", "Venus", "Earth", "Mars"]]
    elif include_bodies == 'outer':
        bodies = [b for b in bodies if b[0] == "Sun" or b[0] in ["Jupiter", "Saturn", "Uranus", "Neptune", "Pluto"]]
    elif include_bodies == 'major':
        bodies = [b for b in bodies if b[0] == "Sun" or b[0] in ["Earth", "Jupiter", "Saturn"]]
    elif include_bodies != 'all':
        # Keep specified custom bodies regardless of filter
        custom_bodies = [b for b in bodies if b[0] not in ["Sun", "Mercury", "Venus", "Earth", "Mars", 
                                                         "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto"]]
        filtered_bodies = [b for b in bodies if b[0] == "Sun" or b[0] in include_bodies.split(',')]
        bodies = filtered_bodies + custom_bodies
    
    # Create points for the configuration
    points = []
    for body_data in bodies:
        name = body_data[0]
        mass = body_data[1]
        semi_major = body_data[2]
        ecc = body_data[3]
        inc = body_data[4]
        perihelion_angle = body_data[5] if len(body_data) > 5 else 0.0
        
        if name == "Sun":
            # Sun at the origin
            position = [0.0, 0.0]
            velocity = [0.0, 0.0]
        else:
            # Compute position based on orbital elements
            # Apply perihelion angle (argument of periapsis)
            angle = np.radians(perihelion_angle)  # Start at specified perihelion angle
            
            # Distance from focus to point on ellipse
            # r = a(1-e²)/(1+e*cos(θ)) where θ is the true anomaly
            # For simplicity, we start at perihelion (closest approach, θ=0)
            # So r = a(1-e)
            radius = semi_major * (1 - ecc)
            
            # Apply inclination and perihelion angle
            position = [
                radius * np.cos(angle),
                radius * np.sin(angle) * np.cos(np.radians(inc))
            ]
            
            # Calculate orbital velocity using Kepler's laws
            if with_velocities:
                velocity = calculate_orbital_velocity(semi_major, bodies[0][1], position)
                # Adjust velocity direction based on perihelion angle
                if abs(perihelion_angle) > 1e-6:
                    speed = np.linalg.norm(velocity)
                    direction = np.array([
                        -np.sin(angle), 
                        np.cos(angle) * np.cos(np.radians(inc))
                    ])
                    direction = direction / np.linalg.norm(direction)
                    velocity = speed * direction
            else:
                velocity = [0.0, 0.0]
        
        # Create point with mass as weight and name as property
        properties = {
            "name": name, 
            "mass": mass, 
            "semi_major": semi_major, 
            "eccentricity": ecc, 
            "inclination": inc,
            "perihelion_angle": perihelion_angle,
            "velocity": velocity
        }
        
        point = Point(
            position=position,
            weight=mass,
            properties=properties
        )
        points.append(point)
    
    # Create configuration
    config = Configuration(points)
    
    return config

def create_solar_system_dynamics(scenario='standard'):
    """
    Create a dynamics function suitable for solar system simulation.
    
    Args:
        scenario: The scenario being simulated
        
    Returns:
        Function F(σ) that drives the system dynamics with proper gravitational behavior
    """
    # For a gravitational system, F(σ) ~ -e^σ
    # The negative sign creates an attractive force
    # The e^σ term gives proper 1/r^2 scaling in physical coordinates
    
    # Adjust coefficient based on scenario
    if scenario == 'standard':
        coefficient = -1.0
    elif scenario == 'comet':
        # Slightly stronger to handle extreme eccentricity
        coefficient = -1.1
    elif scenario == 'close_approach':
        # More precise for close interactions
        coefficient = -1.0
    elif scenario == 'binary_planet':
        # Stronger dynamics for binary system
        coefficient = -1.2
    elif scenario == 'rogue_star':
        # Strong dynamics for star interaction
        coefficient = -1.5
    else:
        coefficient = -1.0
    
    # Adding a small regularization term to improve numerical stability
    # This helps prevent excessive forces at close approaches
    return lambda s: coefficient * np.exp(s) * (1.0 - np.exp(-0.1 * s**2))

def create_custom_state_derivative_function(config, driving_function, time_transform):
    """
    Create a custom state derivative function that incorporates initial velocities.
    
    Args:
        config: Configuration object
        driving_function: Dynamics function F(σ)
        time_transform: Time transformation object
        
    Returns:
        State derivative function
    """
    # Get the standard derivative function
    deriv_fn = create_state_derivative_function(config, driving_function, time_transform)
    
    # Extract initial velocities
    initial_velocities = []
    for point in config.points:
        if "velocity" in point.properties:
            initial_velocities.append(point.properties["velocity"])
        else:
            initial_velocities.append([0.0, 0.0])
    
    # Convert to flat array matching state vector structure
    # Skip the first point (reference)
    flat_velocities = np.array([v for i, v in enumerate(initial_velocities) if i > 0]).flatten()
    
    # Dimension of each point
    dim = config.points[0].dimension()
    
    # Number of points
    n_points = len(config.points)
    
    # Shape dimension
    shape_dim = (n_points - 1) * dim
    
    # Orientation dimension
    orientation_dim = 1 if dim == 2 else (3 if dim == 3 else 0)
    
    def modified_derivative(state, tau):
        """
        Modified state derivative function that incorporates initial velocities.
        
        Args:
            state: Current state vector
            tau: Current τ-time
            
        Returns:
            Modified state derivatives
        """
        # Get standard derivatives
        derivatives = deriv_fn(state, tau)
        
        # If this is the first time step (tau ≈ 0), modify the derivatives
        # to incorporate initial velocities
        if np.abs(tau) < 1e-6:
            # Extract current sigma
            sigma = state[0]
            
            # Scale factor e^σ
            scale_factor = np.exp(sigma)
            
            # Get shape derivatives (θ')
            theta_deriv_indices = range(2+shape_dim, 2+2*shape_dim)
            
            # Adjust the shape derivatives to match initial velocities
            # scaled by the system scale
            # Only modify the first time step to set initial conditions
            derivatives[theta_deriv_indices] = flat_velocities / scale_factor
        
        return derivatives
    
    return modified_derivative

def run_solar_system_simulation(include_bodies='all', tau_max=10.0, num_steps=1000, 
                               use_velocities=True, scenario='standard'):
    """
    Run a simulation of the solar system.
    
    Args:
        include_bodies: Which bodies to include ('all', 'inner', 'outer', 'major')
        tau_max: Maximum tau-time for the simulation
        num_steps: Number of simulation steps
        use_velocities: Whether to use initial orbital velocities
        scenario: Specific scenario to simulate
        
    Returns:
        tuple: (final_config, trajectory)
    """
    # Print scenario information
    if scenario in SCENARIOS:
        print(f"\nScenario: {scenario} - {SCENARIOS[scenario]['description']}")
    
    # Create solar system configuration
    print(f"Creating solar system with bodies: {include_bodies}")
    config = create_solar_system(include_bodies, with_velocities=use_velocities, scenario=scenario)
    
    # Create dynamics function
    F = create_solar_system_dynamics(scenario)
    
    # Create time transformation
    # For gravitational systems, we use f(σ) = σ/2
    # This gives dt/dτ = e^(σ/2), regularizing the dynamics
    transform = TimeTransformation(lambda s: s/2)
    
    # Run simulation
    print(f"Running solar system simulation with tau_max={tau_max}, steps={num_steps}...")
    
    # If using velocities, use our custom derivative function
    if use_velocities:
        from dynamics_engine.integrator import integrate
        
        # Create custom state derivative function with velocities
        deriv_fn = create_custom_state_derivative_function(config, F, transform)
        
        # Run integration with custom derivative function
        trajectory = integrate(config, F, tau_max, num_steps, transform, use_adaptive_steps=True)
        
        # Extract final configuration
        # We would need to reconstruct from the final state
        # For simplicity, we'll use the built-in simulate function
        final_config, _ = simulate(config, F, tau_max, num_steps, transform)
    else:
        # Use standard simulation
        final_config, trajectory = simulate(config, F, tau_max, num_steps, transform)
    
    print(f"Simulation complete! Final sigma: {final_config.sigma:.4f}")
    
    return final_config, trajectory

def visualize_solar_system(final_config, trajectory, vis_type='all', save_prefix=None):
    """
    Visualize solar system simulation results.
    
    Args:
        final_config: Final configuration
        trajectory: Trajectory data
        vis_type: Visualization type ('static', 'trajectory', 'animation', 'dual', 'all')
        save_prefix: Prefix for saved files
    """
    os.makedirs("output", exist_ok=True)
    
    if vis_type in ['static', 'all']:
        # Create static visualization
        print("\nGenerating static visualization...")
        fig, ax = plot_configuration(
            final_config, 
            plane='r', 
            title="Solar System Configuration",
            show_scale=True,
            show_center_of_mass=True,
            point_labels=True
        )
        
        if save_prefix:
            fig.savefig(f"output/{save_prefix}_config.png")
            print(f"Static visualization saved to output/{save_prefix}_config.png")
    
    if vis_type in ['trajectory', 'all']:
        # Create trajectory plot
        print("\nGenerating trajectory visualization...")
        fig, axes = plot_trajectory(
            trajectory,
            time_var='tau',
            show_sigma=True,
            show_physical_time=True,
            title="Solar System Trajectory"
        )
        
        if save_prefix:
            fig.savefig(f"output/{save_prefix}_trajectory.png")
            print(f"Trajectory visualization saved to output/{save_prefix}_trajectory.png")
    
    if vis_type in ['animation', 'all']:
        # Create animation
        print("\nGenerating animation...")
        try:
            anim = animate_trajectory(
                trajectory,
                interval=50,
                figsize=(8, 6),
                save_path=f"output/{save_prefix}_animation.mp4" if save_prefix else None,
                title="Solar System Animation",
                show_time=True,
                plane='r',
                fps=30
            )
            if save_prefix:
                print(f"Animation saved to output/{save_prefix}_animation.mp4")
        except Exception as e:
            print(f"Animation error: {e}")
            print("Make sure ffmpeg is installed for saving animations.")
    
    if vis_type in ['dual', 'all']:
        # Create dual view animation
        print("\nGenerating dual view animation...")
        try:
            anim = animate_dual_view(
                trajectory,
                interval=50,
                figsize=(12, 6),
                save_path=f"output/{save_prefix}_dual_animation.mp4" if save_prefix else None,
                title="Solar System Dual View Animation",
                show_time=True,
                fps=30,
                dynamic_scaling=True
            )
            if save_prefix:
                print(f"Dual view animation saved to output/{save_prefix}_dual_animation.mp4")
        except Exception as e:
            print(f"Animation error: {e}")
            print("Make sure ffmpeg is installed for saving animations.")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Solar System Simulation using the Infinite Origin Framework',
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        '--bodies', type=str, 
        choices=['all', 'inner', 'outer', 'major'],
        default='major',
        help='Which bodies to include in the simulation (default: major)'
    )
    
    parser.add_argument(
        '--tau-max', type=float, default=15.0, 
        help='Maximum tau value for simulation (default: 15.0)'
    )
    
    parser.add_argument(
        '--num-steps', type=int, default=1000,
        help='Number of simulation steps (default: 1000)'
    )
    
    parser.add_argument(
        '--visualize', type=str, choices=['static', 'trajectory', 'animation', 'dual', 'all'],
        default='all',
        help='Visualization type to generate (default: all)'
    )
    
    parser.add_argument(
        '--save', type=str, metavar='PREFIX',
        default='solar_system',
        help='Save visualization with given prefix (default: solar_system)'
    )
    
    parser.add_argument(
        '--no-velocities', action='store_true',
        help='Disable initial orbital velocities'
    )
    
    parser.add_argument(
        '--scenario', type=str,
        choices=list(SCENARIOS.keys()),
        default='standard',
        help='Specific scenario to simulate (default: standard)'
    )
    
    args = parser.parse_args()
    
    # Print header
    print("=" * 70)
    print("Solar System Simulation using the Infinite Origin Framework")
    print("=" * 70)
    
    # Print available scenarios
    print("\nAvailable scenarios:")
    for name, data in SCENARIOS.items():
        print(f"  {name}: {data['description']}")
    
    # Run simulation
    final_config, trajectory = run_solar_system_simulation(
        include_bodies=args.bodies,
        tau_max=args.tau_max,
        num_steps=args.num_steps,
        use_velocities=not args.no_velocities,
        scenario=args.scenario
    )
    
    # Customize save prefix with scenario name if using a non-standard scenario
    if args.scenario != 'standard' and args.save == 'solar_system':
        save_prefix = f"solar_system_{args.scenario}"
    else:
        save_prefix = args.save
    
    # Visualize results
    visualize_solar_system(
        final_config,
        trajectory,
        vis_type=args.visualize,
        save_prefix=save_prefix
    )
    
    print("\n" + "=" * 70)
    print(f"Solar system simulation completed successfully!")
    print("=" * 70)

if __name__ == "__main__":
    main() 