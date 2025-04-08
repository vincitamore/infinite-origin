"""
Integrator Module

This module implements numerical integration techniques for τ-time evolution,
including a specialized Runge-Kutta integrator that handles the dynamics
of multi-point configurations.
"""

from typing import Callable, List, Dict, Union, Tuple, Any, Optional
import numpy as np
from numpy.typing import NDArray
from configuration_space import Configuration, Point
from .time_transform import TimeTransformation

# Type aliases for clarity
State = NDArray[np.float64]  # State vector containing [sigma, dsigma_dtau, theta, dtheta_dtau, phi, dphi_dtau]
DerivativeFn = Callable[[State, float], State]  # Function type for state derivatives


def rk4_step(
    deriv_fn: DerivativeFn,
    state: State,
    tau: float,
    delta_tau: float
) -> State:
    """
    Perform a single 4th-order Runge-Kutta integration step.
    
    Args:
        deriv_fn: Function computing state derivatives
        state: Current state vector
        tau: Current τ-time
        delta_tau: Time step in τ
        
    Returns:
        Updated state vector after the step
    """
    # RK4 coefficients
    k1 = deriv_fn(state, tau)
    k2 = deriv_fn(state + 0.5 * delta_tau * k1, tau + 0.5 * delta_tau)
    k3 = deriv_fn(state + 0.5 * delta_tau * k2, tau + 0.5 * delta_tau)
    k4 = deriv_fn(state + delta_tau * k3, tau + delta_tau)
    
    # Update state using weighted average
    return state + (delta_tau / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def adaptive_step_size(
    deriv_fn: DerivativeFn,
    state: State,
    tau: float,
    delta_tau: float,
    error_tolerance: float = 1e-6,
    min_step: float = 1e-6,
    max_step: float = 0.1,
    max_iterations: int = 10  # Maximum number of iterations to try
) -> Tuple[State, float]:
    """
    Perform an adaptive step size integration step using embedded RK45.
    
    Args:
        deriv_fn: Function computing state derivatives
        state: Current state vector
        tau: Current τ-time
        delta_tau: Initial time step in τ
        error_tolerance: Tolerance for step size adaptation
        min_step: Minimum allowed step size
        max_step: Maximum allowed step size
        max_iterations: Maximum number of iterations to try before accepting result
        
    Returns:
        Tuple of (updated state, actual step size used)
    """
    iteration_count = 0
    
    while iteration_count < max_iterations:
        iteration_count += 1
        
        try:
            # Take a full step with RK4
            next_state_full = rk4_step(deriv_fn, state, tau, delta_tau)
            
            # Take two half steps
            half_state = rk4_step(deriv_fn, state, tau, delta_tau/2)
            next_state_half = rk4_step(deriv_fn, half_state, tau + delta_tau/2, delta_tau/2)
            
            # Check for NaN or infinite values
            if (np.isnan(next_state_half).any() or np.isinf(next_state_half).any() or 
                np.isnan(next_state_full).any() or np.isinf(next_state_full).any()):
                print(f"Warning: Numerical instability detected at tau={tau}. Reducing step size.")
                delta_tau *= 0.5
                if delta_tau < min_step:
                    print(f"Warning: Step size too small, using minimum step size.")
                    delta_tau = min_step
                    # Use the half step result anyway, even if it might be problematic
                    return half_state, delta_tau
                continue
            
            # Estimate error
            error = np.max(np.abs(next_state_full - next_state_half))
            
            # Adjust step size (safety factor of 0.9)
            if error > 0:
                optimal_step = 0.9 * delta_tau * (error_tolerance / error) ** 0.2
                optimal_step = max(min_step, min(optimal_step, max_step))
            else:
                optimal_step = max_step
                
            # If error is acceptable or we hit minimum step, return result
            if error <= error_tolerance or delta_tau <= min_step:
                return next_state_half, delta_tau  # Half-step result is more accurate
            
            # Otherwise, try again with the new step size
            delta_tau = optimal_step
            
        except Exception as e:
            print(f"Error in step computation at tau={tau}: {str(e)}")
            # Reduce step size and try again
            delta_tau *= 0.5
            if delta_tau < min_step:
                print("Warning: Using minimum step size due to computation error.")
                delta_tau = min_step
                # Try one more time with minimum step
                try:
                    half_state = rk4_step(deriv_fn, state, tau, delta_tau/2)
                    return half_state, delta_tau
                except Exception:
                    # If even that fails, just advance state minimally
                    print("Warning: Integration failure, advancing by minimum amount.")
                    return state.copy(), min_step
    
    # If we've hit the maximum iterations, use the current step size
    print(f"Warning: Maximum iterations reached at tau={tau}. Using current step size.")
    try:
        half_state = rk4_step(deriv_fn, state, tau, delta_tau/2)
        return half_state, delta_tau
    except Exception:
        # Last resort: return unchanged state with minimum step
        return state.copy(), min_step


def create_state_derivative_function(
    config: Configuration,
    driving_function: Callable[[float], float],
    time_transform: TimeTransformation
) -> DerivativeFn:
    """
    Create a function that computes state derivatives for a given configuration.
    
    The state vector is structured as:
    [sigma, dsigma_dtau, theta_1, ..., theta_n, dtheta_1_dtau, ..., dtheta_n_dtau, 
     phi_1, ..., phi_m, dphi_1_dtau, ..., dphi_m_dtau]
    
    Args:
        config: The Configuration instance
        driving_function: Function F(σ) that drives the system dynamics
        time_transform: TimeTransformation instance for τ-time
        
    Returns:
        Function computing state derivatives dState/dτ
    """
    # Determine dimensions for the state vector
    dim = config.points[0].dimension()
    
    # For now, use a simplified model with shape coordinates counting as a block
    # In a complete implementation, this would be expanded to handle the full shape space
    shape_dim = (len(config.points) - 1) * dim  # Shape coordinates
    
    if dim == 2:
        orientation_dim = 1  # One angle in 2D
    elif dim == 3:
        orientation_dim = 3  # Three Euler angles in 3D
    else:
        orientation_dim = 0  # No orientation in higher dimensions
    
    # Compute total state dimension
    state_dim = 2 + 2*shape_dim + 2*orientation_dim  # [sigma, dsigma/dtau, theta, dtheta/dtau, phi, dphi/dtau]
    
    def state_derivative(state: State, tau: float) -> State:
        """
        Compute derivatives of state variables with respect to τ.
        
        Args:
            state: Current state vector
            tau: Current τ-time value
            
        Returns:
            Derivatives of state variables: dState/dτ
        """
        # Extract state components
        sigma = state[0]
        dsigma_dtau = state[1]
        
        # Extract shape variables (simplified model)
        theta = state[2:2+shape_dim]
        dtheta_dtau = state[2+shape_dim:2+2*shape_dim]
        
        # Extract orientation variables if applicable
        if orientation_dim > 0:
            phi = state[2+2*shape_dim:2+2*shape_dim+orientation_dim]
            dphi_dtau = state[2+2*shape_dim+orientation_dim:]
        
        # Initialize derivative vector
        derivatives = np.zeros_like(state)
        
        # Sigma derivatives
        derivatives[0] = dsigma_dtau  # d(sigma)/d(tau) = dsigma_dtau
        
        # Compute driving force value from the dynamics function
        total_weight = config.total_weight
        F_value = driving_function(sigma)
        
        # d²σ/dτ² = (1/W) * F(σ,θ,φ) - derived from regularized dynamics
        derivatives[1] = F_value / total_weight
        
        # Shape coordinate derivatives
        derivatives[2:2+shape_dim] = dtheta_dtau
        
        # Now use the driving function to influence shape coordinates too
        # For different dynamics, we'll get different shape evolution
        # Scale the effect by the driving function's influence
        # This gives each dynamics function a distinct visual effect
        
        # Get the sign of the driving function to determine if it's attractive or repulsive
        f_sign = np.sign(F_value)
        f_magnitude = np.abs(F_value)
        
        # Modify the strength based on the dynamics type (extracted from function name)
        # Different dynamics types affect shape differently
        driving_func_str = str(driving_function)
        
        if 'harmonic' in driving_func_str:
            # Harmonic: Regular oscillatory motion with higher frequency
            shape_factor = -0.8 * theta
        elif 'oscillating' in driving_func_str:
            # Oscillating: Add sine wave variation to create more complex patterns
            shape_factor = -0.5 * theta * (1 + 0.5 * np.sin(2 * tau))
        elif 'expansion' in driving_func_str:
            # Expansion: Tendency to spread out
            shape_factor = -0.2 * theta + 0.3 * dtheta_dtau
        elif 'collapse' in driving_func_str:
            # Collapse: Stronger inward pull
            shape_factor = -1.0 * theta - 0.2 * dtheta_dtau
        else:
            # Default gravity: Standard attraction
            shape_factor = -0.5 * theta
        
        # Apply the shape dynamics based on the function type
        derivatives[2+shape_dim:2+2*shape_dim] = shape_factor * (1 + 0.1 * np.abs(F_value))
        
        # Orientation derivatives if applicable
        if orientation_dim > 0:
            derivatives[2+2*shape_dim:2+2*shape_dim+orientation_dim] = dphi_dtau
            
            # Simple model for orientation: d²φ/dτ² = -0.1*φ
            # Also influenced by the driving function
            derivatives[2+2*shape_dim+orientation_dim:] = -0.1 * phi * (1 + 0.05 * np.abs(F_value))
        
        return derivatives
    
    return state_derivative


def integrate(
    config: Configuration,
    driving_function: Callable[[float], float],
    tau_max: float,
    num_steps: int = 1000,
    time_transform: Optional[TimeTransformation] = None,
    use_adaptive_steps: bool = True
) -> Dict[str, np.ndarray]:
    """
    Integrate the dynamics of a configuration over τ-time.
    
    Args:
        config: The Configuration instance to simulate
        driving_function: Function F(σ) driving the dynamics
        tau_max: Maximum τ-time to integrate to
        num_steps: Number of steps for fixed-step integration
        time_transform: Optional TimeTransformation instance (created if None)
        use_adaptive_steps: Whether to use adaptive step sizing
        
    Returns:
        Dictionary containing trajectory data:
            'tau': Array of τ values
            'sigma': Array of σ values
            'dsigma_dtau': Array of dσ/dτ values
            'theta': Array of shape coordinates
            'phi': Array of orientation values
            't': Array of corresponding physical time values
    """
    # Create a time transformation if not provided
    if time_transform is None:
        time_transform = TimeTransformation.from_driving_function(
            driving_function, homogeneity_degree=2
        )
    
    # Create derivative function
    deriv_fn = create_state_derivative_function(config, driving_function, time_transform)
    
    # Determine dimensions
    dim = config.points[0].dimension()
    shape_dim = (len(config.points) - 1) * dim
    orientation_dim = 1 if dim == 2 else (3 if dim == 3 else 0)
    
    # Initialize state vector
    # Initially, we set shape and orientation derivatives to zero
    initial_state = np.zeros(2 + 2*shape_dim + 2*orientation_dim)
    initial_state[0] = config.sigma  # Initial sigma
    
    # Get initial shape coordinates (simplified)
    shape_coords = config.get_shape_coordinates()
    # Flatten points except the first one (relative to first point)
    if len(config.points) > 1:
        flat_shapes = shape_coords[1:].flatten()
        initial_state[2:2+shape_dim] = flat_shapes[:shape_dim]
    
    # Get initial orientation
    if orientation_dim > 0:
        try:
            orientation = config.get_orientation()
            if isinstance(orientation, tuple):
                initial_state[2+2*shape_dim:2+2*shape_dim+orientation_dim] = orientation
            else:
                initial_state[2+2*shape_dim] = orientation
        except (ValueError, TypeError):
            # Default to zero if orientation calculation fails
            pass
    
    # Prepare for integration
    if use_adaptive_steps:
        # Storage for results (we don't know exact steps in advance)
        tau_values = [0.0]
        states = [initial_state.copy()]
        
        # Initial step size
        delta_tau = tau_max / num_steps
        
        # Adaptive integration
        current_tau = 0.0
        current_state = initial_state.copy()
        
        while current_tau < tau_max:
            # Take adaptive step
            next_state, actual_step = adaptive_step_size(
                deriv_fn, current_state, current_tau, delta_tau
            )
            
            # Update current state and time
            current_state = next_state
            current_tau += actual_step
            
            # Store results
            tau_values.append(current_tau)
            states.append(current_state.copy())
            
            # Adjust step size for next iteration
            delta_tau = min(actual_step * 1.1, tau_max - current_tau)
            if delta_tau <= 0:
                break
        
        # Convert to arrays
        tau_array = np.array(tau_values)
        states_array = np.array(states)
        
    else:
        # Fixed step integration
        tau_array = np.linspace(0, tau_max, num_steps + 1)
        delta_tau = tau_max / num_steps
        
        # Initialize state history array
        states_array = np.zeros((num_steps + 1, len(initial_state)))
        states_array[0] = initial_state
        
        # Integrate using RK4
        for i in range(num_steps):
            states_array[i+1] = rk4_step(
                deriv_fn, states_array[i], tau_array[i], delta_tau
            )
    
    # Extract components from states
    sigma_values = states_array[:, 0]
    dsigma_dtau_values = states_array[:, 1]
    theta_values = states_array[:, 2:2+shape_dim]
    
    if orientation_dim > 0:
        phi_values = states_array[:, 2+2*shape_dim:2+2*shape_dim+orientation_dim]
    else:
        phi_values = np.array([])
        
    # Calculate corresponding physical time
    t_values = time_transform.integrate_transformation(sigma_values, tau_array)
    
    # Return results
    return {
        'tau': tau_array,
        'sigma': sigma_values,
        'dsigma_dtau': dsigma_dtau_values,
        'theta': theta_values,
        'phi': phi_values,
        't': t_values
    }


def simulate(
    config: Configuration,
    driving_function: Callable[[float], float],
    tau_max: float = 10.0,
    num_steps: int = 1000,
    time_transform: Optional[TimeTransformation] = None
) -> Tuple[Configuration, Dict[str, np.ndarray]]:
    """
    Simulate the evolution of a configuration and return both the final
    configuration and the full trajectory.
    
    Args:
        config: The Configuration instance to simulate
        driving_function: Function F(σ) driving the dynamics
        tau_max: Maximum τ-time to integrate to
        num_steps: Number of steps for integration
        time_transform: Optional TimeTransformation instance
        
    Returns:
        Tuple of (final Configuration, trajectory data)
    """
    # Run the integration
    trajectory = integrate(
        config, driving_function, tau_max, num_steps, time_transform, True
    )
    
    # Create a new configuration based on the final state
    final_sigma = trajectory['sigma'][-1]
    final_theta = trajectory['theta'][-1].reshape(-1, config.points[0].dimension())
    
    # Reconstruct the positions from sigma and theta
    # For simplicity, we'll place the first point at the origin
    # and position others according to the shape coordinates
    new_points = []
    
    # Scale factor from sigma
    scale = np.exp(final_sigma)
    
    # Create points at the appropriate positions
    center_index = 0  # Assuming first point is reference
    for i in range(len(config.points)):
        if i == center_index:
            # Reference point at origin
            position = np.zeros(config.points[i].dimension())
        else:
            # Get index in theta array (i-1 if i>center_index else i)
            idx = i - 1 if i > center_index else i
            # Position from shape coordinates and scale
            position = final_theta[idx] * scale
        
        # Create new point with same weight and properties
        new_point = Point(
            position=position,
            weight=config.points[i].weight,
            properties=config.points[i].properties.copy()
        )
        new_points.append(new_point)
    
    # Create the final configuration
    final_config = Configuration(new_points)
    
    return final_config, trajectory 