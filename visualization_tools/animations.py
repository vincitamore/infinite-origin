"""
Animations Module

This module provides functions for creating animations of trajectories
from simulations in both τ-plane and r-plane representations.
"""

from typing import List, Optional, Tuple, Dict, Union, Any, Callable
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.animation as animation
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from configuration_space import Configuration, Point


def animate_trajectory(
    trajectory: Dict[str, np.ndarray],
    interval: int = 50,
    figsize: Tuple[float, float] = (10, 6),
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    show_time: bool = True,
    plane: str = 'r',
    fps: int = 30
) -> Union[animation.FuncAnimation, go.Figure]:
    """
    Create an animation of a configuration trajectory.
    
    Args:
        trajectory: Dictionary containing trajectory data from simulation
        interval: Interval between frames in milliseconds (for matplotlib)
        figsize: Figure size as (width, height) in inches
        save_path: Optional path to save the animation (e.g., 'animation.mp4')
        title: Optional animation title
        show_time: Whether to display time in the animation
        plane: Which plane to visualize in ('r' or 'tau')
        fps: Frames per second when saving animation
    
    Returns:
        Matplotlib animation object or Plotly figure with animation
    """
    # Extract necessary data from trajectory
    positions = trajectory.get('positions', None)
    
    # If positions are not available but theta (shape) and sigma (scale) are,
    # reconstruct positions from theta and sigma
    if positions is None:
        theta = trajectory.get('theta', None)
        sigma = trajectory.get('sigma', None)
        
        if theta is not None and sigma is not None:
            # For simplicity, assume 2D configuration and reshape theta
            n_steps = len(sigma)
            if len(theta.shape) > 1:
                n_coords = theta.shape[1]
                # Detect dimension (assuming theta contains all coordinates)
                # and reshape theta to (n_steps, n_points, dim)
                if n_coords % 2 == 0:  # 2D
                    dim = 2
                elif n_coords % 3 == 0:  # 3D
                    dim = 3
                else:
                    raise ValueError("Cannot determine dimension from theta shape")
                
                n_points = n_coords // dim
                
                # Reshape theta for each time step and compute positions using scale
                positions = np.zeros((n_steps, n_points, dim))
                
                for i in range(n_steps):
                    scale = np.exp(sigma[i])
                    points_at_t = theta[i].reshape(n_points, dim)
                    positions[i] = points_at_t * scale
            else:
                raise ValueError("Cannot reconstruct positions: theta shape not supported")
                
    if positions is None:
        raise ValueError("Cannot animate: position data not available in trajectory")
    
    # Get time data
    tau_values = trajectory.get('tau', None)
    t_values = trajectory.get('t', None)
    
    # Get number of points and dimensions
    n_frames, n_points, dim = positions.shape
    
    if dim not in (2, 3):
        raise ValueError("Animation only supported for 2D and 3D configurations")
    
    # Set the plane label for titles
    plane_label = 'r-plane' if plane == 'r' else 'τ-plane'
        
    # Transform to tau-plane if needed
    if plane == 'tau':
        # Create tau_positions with proper error handling
        tau_positions = np.zeros_like(positions)
        for i in range(n_frames):
            for j in range(n_points):
                for k in range(dim):
                    coord = positions[i, j, k]
                    if abs(coord) < 1e-10:  # Near zero
                        tau_coord = 1e10 * np.sign(coord) if coord != 0 else 1e10
                    else:
                        tau_coord = 1.0 / coord
                    tau_positions[i, j, k] = tau_coord
        positions = tau_positions
    
    # Create a matplotlib animation
    fig = plt.figure(figsize=figsize)
    
    # Prepare colors for points
    colors = plt.cm.viridis(np.linspace(0, 1, n_points))
    
    if dim == 2:
        ax = fig.add_subplot(111)
        
        # Get limits for consistent view
        max_x = np.max(np.abs(positions[:, :, 0]))
        max_y = np.max(np.abs(positions[:, :, 1]))
        margin = 0.1 * max(max_x, max_y)
        
        # Set plot limits for consistent view
        if plane == 'r':  # Regular r-plane
            ax.set_xlim(-max_x - margin, max_x + margin)
            ax.set_ylim(-max_y - margin, max_y + margin)
        else:  # tau-plane needs special handling
            # Calculate appropriate limits for tau-plane display
            min_tau_x = np.min(positions[:, :, 0][~np.isnan(positions[:, :, 0])])
            max_tau_x = np.max(positions[:, :, 0][~np.isnan(positions[:, :, 0])])
            min_tau_y = np.min(positions[:, :, 1][~np.isnan(positions[:, :, 1])])
            max_tau_y = np.max(positions[:, :, 1][~np.isnan(positions[:, :, 1])])
            
            # Ensure proper limits even if there are extreme values
            xlimit = max(abs(min_tau_x), abs(max_tau_x)) * 1.2
            ylimit = max(abs(min_tau_y), abs(max_tau_y)) * 1.2
            
            # Use logarithmic scale for tau-plane
            ax.set_xscale('symlog')
            ax.set_yscale('symlog')
            ax.set_xlim(-xlimit, xlimit)
            ax.set_ylim(-ylimit, ylimit)
            ax.grid(True, which='both', linestyle='--', alpha=0.6)
        
        # Initialize with the first frame's data
        scatter = ax.scatter(
            positions[0, :, 0],
            positions[0, :, 1],
            s=30, 
            c=colors
        )
        
        ax.set_xlabel(f'x ({plane_label})')
        ax.set_ylabel(f'y ({plane_label})')
        ax.set_aspect('equal')
    
    else:  # 3D
        ax = fig.add_subplot(111, projection='3d')
        
        # Get limits for consistent view
        max_x = np.max(np.abs(positions[:, :, 0]))
        max_y = np.max(np.abs(positions[:, :, 1]))
        max_z = np.max(np.abs(positions[:, :, 2]))
        margin = 0.1 * max(max_x, max_y, max_z)
        
        # Set plot limits for consistent view
        if plane == 'r':  # Regular r-plane
            ax.set_xlim(-max_x - margin, max_x + margin)
            ax.set_ylim(-max_y - margin, max_y + margin)
            ax.set_zlim(-max_z - margin, max_z + margin)
        else:  # tau-plane needs special handling
            # Calculate appropriate limits for tau-plane display
            min_tau_x = np.min(positions[:, :, 0][~np.isnan(positions[:, :, 0])])
            max_tau_x = np.max(positions[:, :, 0][~np.isnan(positions[:, :, 0])])
            min_tau_y = np.min(positions[:, :, 1][~np.isnan(positions[:, :, 1])])
            max_tau_y = np.max(positions[:, :, 1][~np.isnan(positions[:, :, 1])])
            
            # Ensure proper limits even if there are extreme values
            xlimit = max(abs(min_tau_x), abs(max_tau_x)) * 1.2
            ylimit = max(abs(min_tau_y), abs(max_tau_y)) * 1.2
            
            # Use logarithmic scale for tau-plane
            ax.set_xscale('symlog')
            ax.set_yscale('symlog')
            ax.set_xlim(-xlimit, xlimit)
            ax.set_ylim(-ylimit, ylimit)
            ax.grid(True, which='both', linestyle='--', alpha=0.6)
        
        # Initialize with the first frame's data
        scatter = ax.scatter(
            positions[0, :, 0],
            positions[0, :, 1],
            positions[0, :, 2],
            s=30, 
            c=colors
        )
        
        ax.set_xlabel(f'x ({plane_label})')
        ax.set_ylabel(f'y ({plane_label})')
        ax.set_zlabel(f'z ({plane_label})')
    
    # Set title with time if requested
    if title:
        title_text = ax.set_title(title)
    elif show_time:
        if tau_values is not None and t_values is not None:
            title_text = ax.set_title(
                f'Configuration in {plane_label} (τ={tau_values[0]:.2f}, t={t_values[0]:.2f})'
            )
        elif tau_values is not None:
            title_text = ax.set_title(f'Configuration in {plane_label} (τ={tau_values[0]:.2f})')
        elif t_values is not None:
            title_text = ax.set_title(f'Configuration in {plane_label} (t={t_values[0]:.2f})')
        else:
            title_text = ax.set_title(f'Configuration in {plane_label} (frame 0)')
    else:
        title_text = ax.set_title("")
    
    # Create path lines for tau-plane to show trajectory
    path_lines = []
    if plane == 'tau':
        for i in range(n_points):
            line, = ax.plot([], [], '-', lw=1, alpha=0.5, color=colors[i])
            path_lines.append(line)
    
    def update(frame):
        """Update function for animation."""
        # Update positions
        if dim == 2:
            scatter.set_offsets(positions[frame])
        else:  # 3D
            scatter._offsets3d = (
                positions[frame, :, 0],
                positions[frame, :, 1],
                positions[frame, :, 2]
            )
        
        # Update path lines for tau-plane
        if plane == 'tau' and frame > 0:
            for i in range(n_points):
                x_path = positions[:frame+1, i, 0]
                y_path = positions[:frame+1, i, 1]
                # Filter out any infinity or NaN values
                valid = np.isfinite(x_path) & np.isfinite(y_path)
                path_lines[i].set_data(x_path[valid], y_path[valid])
        
        # Update title with time if requested
        if show_time and title_text:
            if tau_values is not None and t_values is not None:
                title_text.set_text(
                    f'Configuration in {plane_label} (τ={tau_values[frame]:.2f}, t={t_values[frame]:.2f})'
                )
            elif tau_values is not None:
                title_text.set_text(f'Configuration in {plane_label} (τ={tau_values[frame]:.2f})')
            elif t_values is not None:
                title_text.set_text(f'Configuration in {plane_label} (t={t_values[frame]:.2f})')
            else:
                title_text.set_text(f'Configuration in {plane_label} (frame {frame})')
        
        if plane == 'tau':
            return [scatter, title_text] + path_lines
        else:
            return scatter, title_text
    
    # Create animation
    anim = animation.FuncAnimation(
        fig, update, frames=n_frames,
        blit=True, interval=interval
    )
    
    # Save animation if path provided
    if save_path:
        try:
            # Make sure to install ffmpeg for video export
            writer = animation.FFMpegWriter(fps=fps, metadata=dict(artist='Infinite Origin Framework'))
            anim.save(save_path, writer=writer)
            print(f"Animation saved to {save_path}")
        except Exception as e:
            print(f"Error saving animation: {e}. Make sure ffmpeg is installed.")
    
    plt.tight_layout()
    return anim

def animate_dual_view(
    trajectory: Dict[str, np.ndarray],
    interval: int = 50,
    figsize: Tuple[float, float] = (12, 6),
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    show_time: bool = True,
    fps: int = 30,
    dynamic_scaling: bool = True
) -> animation.FuncAnimation:
    """
    Create a side-by-side animation showing a trajectory in both r-plane and τ-plane.
    
    Args:
        trajectory: Dictionary containing trajectory data from simulation
        interval: Interval between frames in milliseconds
        figsize: Figure size as (width, height) in inches
        save_path: Optional path to save the animation (e.g., 'dual_animation.mp4')
        title: Optional animation title
        show_time: Whether to display time in the animation
        fps: Frames per second when saving animation
        dynamic_scaling: Whether to dynamically adjust r-plane scaling as points collapse
    
    Returns:
        Matplotlib animation object
    """
    # Extract necessary data from trajectory
    positions = trajectory.get('positions', None)
    
    # If positions are not available but theta (shape) and sigma (scale) are,
    # reconstruct positions from theta and sigma
    if positions is None:
        theta = trajectory.get('theta', None)
        sigma = trajectory.get('sigma', None)
        
        if theta is not None and sigma is not None:
            # For simplicity, assume 2D configuration and reshape theta
            n_steps = len(sigma)
            if len(theta.shape) > 1:
                n_coords = theta.shape[1]
                # Detect dimension (assuming theta contains all coordinates)
                # and reshape theta to (n_steps, n_points, dim)
                if n_coords % 2 == 0:  # 2D
                    dim = 2
                elif n_coords % 3 == 0:  # 3D
                    dim = 3
                else:
                    raise ValueError("Cannot determine dimension from theta shape")
                
                n_points = n_coords // dim
                
                # Reshape theta for each time step and compute positions using scale
                positions = np.zeros((n_steps, n_points, dim))
                
                for i in range(n_steps):
                    scale = np.exp(sigma[i])
                    points_at_t = theta[i].reshape(n_points, dim)
                    positions[i] = points_at_t * scale
            else:
                raise ValueError("Cannot reconstruct positions: theta shape not supported")
                
    if positions is None:
        raise ValueError("Cannot animate: position data not available in trajectory")
    
    # Get time data
    tau_values = trajectory.get('tau', None)
    t_values = trajectory.get('t', None)
    
    # Get number of points and dimensions
    n_frames, n_points, dim = positions.shape
    
    if dim not in (2, 3):
        raise ValueError("Animation only supported for 2D and 3D configurations")
    
    # Prepare colors for points
    colors_r = plt.cm.viridis(np.linspace(0, 1, n_points))
    colors_tau = plt.cm.plasma(np.linspace(0, 1, n_points))
    
    # Create a matplotlib figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Set titles for each subplot
    ax1.set_title("r-plane")
    ax2.set_title("τ-plane")
    
    # Get limits for consistent view in r-plane
    max_x_r = np.max(np.abs(positions[:, :, 0]))
    max_y_r = np.max(np.abs(positions[:, :, 1]))
    margin_r = 0.1 * max(max_x_r, max_y_r)
    
    # For dynamic scaling, calculate distance from origin for each frame
    if dynamic_scaling:
        # Calculate distances from origin for each point at each frame
        distances = np.sqrt(np.sum(positions**2, axis=2))
        # For each frame, get the maximum distance (used for r-plane scaling)
        max_distances = np.max(distances, axis=1)
        # Add buffer to prevent zero or very small scales
        max_distances = np.maximum(max_distances, 1e-6)
        # Calculate appropriate scale factors
        # We'll use these to dynamically adjust the r-plane view
        r_scales = np.maximum(max_distances * 1.5, 0.01)
    else:
        # Use static scaling if dynamic scaling is disabled
        r_scales = np.ones(n_frames) * (max_x_r + margin_r)
    
    # Create tau positions for visualization
    tau_positions = np.zeros_like(positions)
    for i in range(n_frames):
        for j in range(n_points):
            for k in range(dim):
                coord = positions[i, j, k]
                if abs(coord) < 1e-10:  # Near zero
                    tau_coord = 1e10 * np.sign(coord) if coord != 0 else 1e10
                else:
                    tau_coord = 1.0 / coord
                tau_positions[i, j, k] = tau_coord
    
    # Set up r-plane scatter plot (initialize with first frame data)
    initial_scale = r_scales[0]
    ax1.set_xlim(-initial_scale, initial_scale)
    ax1.set_ylim(-initial_scale, initial_scale)
    ax1.set_xlabel('x (r-plane)')
    ax1.set_ylabel('y (r-plane)')
    ax1.set_aspect('equal')
    ax1.grid(True, which='both', linestyle='--', alpha=0.4)
    scatter1 = ax1.scatter(
        positions[0, :, 0], 
        positions[0, :, 1], 
        s=30, 
        c=colors_r
    )
    
    # Create trace lines array to show paths in r-plane
    r_path_lines = []
    for i in range(n_points):
        # Empty line to start
        line, = ax1.plot([], [], '-', lw=1, alpha=0.5, color=colors_r[i])
        r_path_lines.append(line)
    
    # Set up tau-plane scatter plot (with logarithmic scale)
    ax2.set_xscale('symlog')
    ax2.set_yscale('symlog')
    
    # Calculate appropriate limits for tau-plane display
    # Using standard logarithmic increments to ensure visibility
    min_tau_x = np.min(tau_positions[:, :, 0][~np.isnan(tau_positions[:, :, 0])])
    max_tau_x = np.max(tau_positions[:, :, 0][~np.isnan(tau_positions[:, :, 0])])
    min_tau_y = np.min(tau_positions[:, :, 1][~np.isnan(tau_positions[:, :, 1])])
    max_tau_y = np.max(tau_positions[:, :, 1][~np.isnan(tau_positions[:, :, 1])])
    
    # Ensure proper limits even if there are extreme values
    xlimit = max(abs(min_tau_x), abs(max_tau_x)) * 1.2
    ylimit = max(abs(min_tau_y), abs(max_tau_y)) * 1.2
    
    # Create more viewable limits for tau-plane
    linthresh = 1.0  # Linear threshold for symlog scale
    ax2.set_xlim(-xlimit, xlimit)
    ax2.set_ylim(-ylimit, ylimit)
    
    # Set grid for better visibility
    ax2.grid(True, which='both', linestyle='--', alpha=0.6)
    
    ax2.set_xlabel('x (τ-plane)')
    ax2.set_ylabel('y (τ-plane)')
    scatter2 = ax2.scatter(
        tau_positions[0, :, 0], 
        tau_positions[0, :, 1], 
        s=30, 
        c=colors_tau
    )
    
    # Create trace lines array to show paths in tau plane
    path_lines = []
    for i in range(n_points):
        # Empty line to start
        line, = ax2.plot([], [], '-', lw=1, alpha=0.5, color=colors_tau[i])
        path_lines.append(line)
    
    # Set main title
    if title:
        title_text = fig.suptitle(title)
    elif show_time:
        if tau_values is not None and t_values is not None:
            title_text = fig.suptitle(f'Dual View (τ={tau_values[0]:.2f}, t={t_values[0]:.2f})')
        elif tau_values is not None:
            title_text = fig.suptitle(f'Dual View (τ={tau_values[0]:.2f})')
        elif t_values is not None:
            title_text = fig.suptitle(f'Dual View (t={t_values[0]:.2f})')
        else:
            title_text = fig.suptitle(f'Dual View (frame 0)')
    else:
        title_text = fig.suptitle("")
    
    def update(frame):
        """Update function for animation."""
        # Update r-plane positions
        scatter1.set_offsets(positions[frame])
        
        # Update r-plane limits if dynamic scaling is enabled
        if dynamic_scaling:
            current_scale = r_scales[frame]
            # Make sure the scale doesn't change too abruptly (smooth transitions)
            # Using exponential smoothing between frames
            if frame > 0:
                prev_scale = r_scales[frame-1]
                smoothed_scale = prev_scale * 0.7 + current_scale * 0.3
                current_scale = smoothed_scale
                r_scales[frame] = current_scale
            
            # Update limits
            ax1.set_xlim(-current_scale, current_scale)
            ax1.set_ylim(-current_scale, current_scale)
        
        # Update r-plane path lines
        if frame > 0:
            for i in range(n_points):
                x_path = positions[:frame+1, i, 0]
                y_path = positions[:frame+1, i, 1]
                # Filter out any infinity or NaN values
                valid = np.isfinite(x_path) & np.isfinite(y_path)
                r_path_lines[i].set_data(x_path[valid], y_path[valid])
        
        # Update tau-plane positions
        scatter2.set_offsets(tau_positions[frame])
        
        # Update trace lines (show path up to current frame)
        if frame > 0:
            for i in range(n_points):
                x_path = tau_positions[:frame+1, i, 0]
                y_path = tau_positions[:frame+1, i, 1]
                # Filter out any infinity or NaN values
                valid = np.isfinite(x_path) & np.isfinite(y_path)
                path_lines[i].set_data(x_path[valid], y_path[valid])
        
        # Update title with time if requested
        if show_time and title_text:
            if tau_values is not None and t_values is not None:
                title_text.set_text(f'Dual View (τ={tau_values[frame]:.2f}, t={t_values[frame]:.2f})')
            elif tau_values is not None:
                title_text.set_text(f'Dual View (τ={tau_values[frame]:.2f})')
            elif t_values is not None:
                title_text.set_text(f'Dual View (t={t_values[frame]:.2f})')
            else:
                title_text.set_text(f'Dual View (frame {frame})')
        
        return [scatter1, scatter2, title_text] + path_lines + r_path_lines
    
    # Create animation
    anim = animation.FuncAnimation(
        fig, update, frames=n_frames,
        blit=True, interval=interval
    )
    
    # Save animation if path provided
    if save_path:
        try:
            # Make sure to install ffmpeg for video export
            writer = animation.FFMpegWriter(fps=fps, metadata=dict(artist='Infinite Origin Framework'))
            anim.save(save_path, writer=writer)
            print(f"Animation saved to {save_path}")
        except Exception as e:
            print(f"Error saving animation: {e}. Make sure ffmpeg is installed.")
    
    plt.tight_layout()
    return anim 