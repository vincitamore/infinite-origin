"""
Trajectory Plots Module

This module provides functions for visualizing trajectories from simulations
in both τ-plane and r-plane representations.
"""

from typing import List, Optional, Tuple, Dict, Union, Any
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from configuration_space import Configuration, Point


def plot_trajectory(
    trajectory: Dict[str, np.ndarray],
    config: Optional[Configuration] = None,
    time_var: str = 'tau',
    figsize: Tuple[float, float] = (10, 8),
    show_sigma: bool = True,
    show_physical_time: bool = True,
    use_plotly: bool = False,
    title: Optional[str] = None,
    plot_kwargs: Optional[Dict[str, Any]] = None
) -> Union[Tuple[Figure, List[Axes]], go.Figure]:
    """
    Plot key variables from a trajectory over time.
    
    Args:
        trajectory: Dictionary containing trajectory data from simulation
        config: Optional final Configuration object
        time_var: Time variable to use for x-axis ('tau' or 't')
        figsize: Figure size as (width, height) in inches
        show_sigma: Whether to plot sigma (scale) evolution
        show_physical_time: Whether to plot physical time vs tau
        use_plotly: Use Plotly for interactive visualization
        title: Optional plot title
        plot_kwargs: Additional keyword arguments for plot customization
    
    Returns:
        Matplotlib Figure and list of Axes objects, or Plotly Figure if use_plotly=True
    """
    if plot_kwargs is None:
        plot_kwargs = {}
        
    # Determine which time variable to use on x-axis
    if time_var not in ('tau', 't'):
        raise ValueError("time_var must be 'tau' or 't'")
        
    time_values = trajectory[time_var]
    
    if use_plotly:
        return _plot_trajectory_plotly(
            trajectory, time_var, time_values, title, 
            show_sigma, show_physical_time, **plot_kwargs
        )
    else:
        return _plot_trajectory_matplotlib(
            trajectory, time_var, time_values, figsize, title,
            show_sigma, show_physical_time, **plot_kwargs
        )

def _plot_trajectory_matplotlib(
    trajectory: Dict[str, np.ndarray],
    time_var: str,
    time_values: np.ndarray,
    figsize: Tuple[float, float],
    title: Optional[str],
    show_sigma: bool,
    show_physical_time: bool,
    **kwargs
) -> Tuple[Figure, List[Axes]]:
    """Helper function for matplotlib trajectory plotting."""
    # Determine how many subplots we need
    num_plots = 1  # Start with at least one plot
    if show_sigma:
        num_plots += 1
    if show_physical_time and time_var == 'tau':
        num_plots += 1
    
    # Create figure with subplots
    fig, axes = plt.subplots(num_plots, 1, figsize=figsize, sharex=True)
    if num_plots == 1:
        axes = [axes]  # Convert to list for consistency
    
    # Track current subplot index
    plot_idx = 0
    
    # Plot physical time vs tau if requested and appropriate
    if show_physical_time and time_var == 'tau':
        ax = axes[plot_idx]
        ax.plot(time_values, trajectory['t'], lw=2, **kwargs)
        ax.set_ylabel('Physical Time (t)')
        ax.grid(True, alpha=0.3)
        plot_idx += 1
    
    # Plot sigma evolution if requested
    if show_sigma:
        ax = axes[plot_idx]
        ax.plot(time_values, trajectory['sigma'], lw=2, color='red', **kwargs)
        ax.set_ylabel('Scale (σ)')
        ax.grid(True, alpha=0.3)
        plot_idx += 1
    
    # Plot shape coordinates on the last subplot
    ax = axes[plot_idx]
    
    # Get shape coordinates (theta) from trajectory
    # If theta is multidimensional, plot each component
    theta = trajectory.get('theta', None)
    if theta is not None and len(theta.shape) > 1:
        # Plot each component of theta
        n_components = theta.shape[1]
        for i in range(n_components):
            ax.plot(
                time_values, theta[:, i], 
                label=f'θ[{i}]', 
                lw=1.5, 
                **kwargs
            )
        ax.legend()
    elif theta is not None:
        # Plot single theta value
        ax.plot(time_values, theta, lw=2, color='blue', label='θ', **kwargs)
        ax.legend()
        
    ax.set_ylabel('Shape Coordinates (θ)')
    ax.grid(True, alpha=0.3)
    
    # Set x-axis label on the bottom plot
    axes[-1].set_xlabel(f'τ-time' if time_var == 'tau' else 'Physical time (t)')
    
    # Set overall title
    if title:
        fig.suptitle(title)
    else:
        fig.suptitle(f'Trajectory Evolution over {"τ-time" if time_var == "tau" else "Physical time"}')
    
    plt.tight_layout()
    return fig, axes

def _plot_trajectory_plotly(
    trajectory: Dict[str, np.ndarray],
    time_var: str,
    time_values: np.ndarray,
    title: Optional[str],
    show_sigma: bool,
    show_physical_time: bool,
    **kwargs
) -> go.Figure:
    """Helper function for plotly trajectory plotting."""
    # Create plotly figure
    fig = make_subplots(
        rows=3 if (show_sigma and show_physical_time and time_var == 'tau') else 
             2 if (show_sigma or (show_physical_time and time_var == 'tau')) else 1,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05
    )
    
    # Track current subplot index
    row_idx = 1
    
    # Plot physical time vs tau if requested and appropriate
    if show_physical_time and time_var == 'tau':
        fig.add_trace(
            go.Scatter(
                x=time_values, 
                y=trajectory['t'], 
                mode='lines',
                name='Physical Time (t)',
                line=dict(width=2, color='green')
            ),
            row=row_idx, col=1
        )
        fig.update_yaxes(title_text='Physical Time (t)', row=row_idx, col=1, gridcolor='lightgray')
        row_idx += 1
    
    # Plot sigma evolution if requested
    if show_sigma:
        fig.add_trace(
            go.Scatter(
                x=time_values, 
                y=trajectory['sigma'], 
                mode='lines',
                name='Scale (σ)',
                line=dict(width=2, color='red')
            ),
            row=row_idx, col=1
        )
        fig.update_yaxes(title_text='Scale (σ)', row=row_idx, col=1, gridcolor='lightgray')
        row_idx += 1
    
    # Plot shape coordinates
    theta = trajectory.get('theta', None)
    if theta is not None and len(theta.shape) > 1:
        # Plot each component of theta
        n_components = theta.shape[1]
        for i in range(n_components):
            fig.add_trace(
                go.Scatter(
                    x=time_values, 
                    y=theta[:, i], 
                    mode='lines',
                    name=f'θ[{i}]',
                    line=dict(width=1.5)
                ),
                row=row_idx, col=1
            )
    elif theta is not None:
        # Plot single theta value
        fig.add_trace(
            go.Scatter(
                x=time_values, 
                y=theta, 
                mode='lines',
                name='θ',
                line=dict(width=2, color='blue')
            ),
            row=row_idx, col=1
        )
    
    fig.update_yaxes(title_text='Shape Coordinates (θ)', row=row_idx, col=1, gridcolor='lightgray')
    
    # Update layout
    time_label = 'τ-time' if time_var == 'tau' else 'Physical time (t)'
    fig.update_xaxes(title_text=time_label, row=row_idx, col=1)
    
    # Set title
    if title:
        fig.update_layout(title_text=title)
    else:
        fig.update_layout(title_text=f'Trajectory Evolution over {time_label}')
    
    # Update layout for better appearance
    fig.update_layout(
        showlegend=True,
        height=200 * (row_idx + 1),  # Adjust height based on number of subplots
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified"
    )
    
    return fig

def plot_trajectory_shape(
    trajectory: Dict[str, np.ndarray],
    highlight_points: Optional[List[int]] = None,
    num_points: int = 20,
    figsize: Tuple[float, float] = (10, 8),
    use_plotly: bool = False,
    title: Optional[str] = None,
    plot_kwargs: Optional[Dict[str, Any]] = None
) -> Union[Figure, go.Figure]:
    """
    Plot the shape evolution of a trajectory, showing points at regular intervals.
    
    Args:
        trajectory: Dictionary containing trajectory data
        highlight_points: List of point indices to highlight (e.g., [0, 10, 20])
        num_points: Number of trajectory points to show if highlight_points is None
        figsize: Figure size as (width, height) in inches
        use_plotly: Use Plotly for interactive visualization
        title: Optional plot title
        plot_kwargs: Additional keyword arguments for plot customization
    
    Returns:
        Matplotlib Figure or Plotly Figure if use_plotly=True
    """
    if plot_kwargs is None:
        plot_kwargs = {}
    
    # Get positions if available, or theta if positions aren't directly stored
    positions = trajectory.get('positions', None)
    theta = trajectory.get('theta', None)
    sigma = trajectory.get('sigma', None)
    
    if positions is None and theta is not None and sigma is not None:
        # Reconstruct positions from theta and sigma
        # This assumes theta is a 2D array with shape (time_steps, num_points * dim)
        # and that we can reshape it to (time_steps, num_points, dim)
        
        # For simplicity, we'll assume 2D or detect from theta shape
        if len(theta.shape) > 1 and theta.shape[1] % 2 == 0:
            dim = 2
            # Reshape and scale by e^sigma to get positions
            n_shapes = theta.shape[0]
            n_coords = theta.shape[1]
            n_points = n_coords // dim
            positions = np.zeros((n_shapes, n_points, dim))
            
            for i in range(n_shapes):
                scale = np.exp(sigma[i])
                points_at_t = theta[i].reshape(n_points, dim)
                positions[i] = points_at_t * scale
    
    if positions is None:
        raise ValueError("Cannot visualize trajectory shape: positions data not available")
    
    # Get subset of points to plot
    if highlight_points is not None:
        indices = highlight_points
    else:
        # Select evenly spaced points
        total_points = len(positions)
        indices = np.linspace(0, total_points - 1, num_points, dtype=int)
    
    if use_plotly:
        return _plot_trajectory_shape_plotly(
            trajectory, positions, indices, title, **plot_kwargs
        )
    else:
        return _plot_trajectory_shape_matplotlib(
            trajectory, positions, indices, figsize, title, **plot_kwargs
        )

def _plot_trajectory_shape_matplotlib(
    trajectory: Dict[str, np.ndarray],
    positions: np.ndarray,
    indices: List[int],
    figsize: Tuple[float, float],
    title: Optional[str],
    **kwargs
) -> Figure:
    """Helper function for matplotlib shape trajectory plotting."""
    # Determine dimension from positions
    dim = positions.shape[2] if len(positions.shape) > 2 else 2
    
    # Create figure
    if dim == 2:
        fig, ax = plt.subplots(figsize=figsize)
    else:  # 3D
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
    
    # Get a colormap for time progression
    cmap = plt.get_cmap('viridis')
    colors = [cmap(i / len(indices)) for i in range(len(indices))]
    
    # Plot each configuration
    for i, idx in enumerate(indices):
        # Get positions at this time
        pos = positions[idx]
        
        # For 2D
        if dim == 2:
            # Plot points
            ax.scatter(
                pos[:, 0], pos[:, 1], 
                color=colors[i], 
                alpha=0.7, 
                label=f't={idx}' if i == 0 or i == len(indices)-1 else None,
                **kwargs
            )
            
            # Connect points to show configuration
            ax.plot(
                pos[:, 0], pos[:, 1], 
                '-', 
                color=colors[i], 
                alpha=0.5,
                **kwargs
            )
            
        # For 3D
        else:
            # Plot points
            ax.scatter(
                pos[:, 0], pos[:, 1], pos[:, 2],
                color=colors[i], 
                alpha=0.7, 
                label=f't={idx}' if i == 0 or i == len(indices)-1 else None,
                **kwargs
            )
            
            # Connect points to show configuration
            ax.plot(
                pos[:, 0], pos[:, 1], pos[:, 2],
                '-', 
                color=colors[i], 
                alpha=0.5,
                **kwargs
            )
    
    # Set labels
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    if dim == 3:
        ax.set_zlabel('z')
    
    # Set title
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Configuration Shape Evolution')
    
    # Add colorbar to show time progression
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Time progression')
    
    # Set equal aspect ratio for 2D
    if dim == 2:
        ax.set_aspect('equal')
    
    plt.tight_layout()
    return fig

def _plot_trajectory_shape_plotly(
    trajectory: Dict[str, np.ndarray],
    positions: np.ndarray,
    indices: List[int],
    title: Optional[str],
    **kwargs
) -> go.Figure:
    """Helper function for plotly shape trajectory plotting."""
    # Determine dimension from positions
    dim = positions.shape[2] if len(positions.shape) > 2 else 2
    
    # Create figure
    fig = go.Figure()
    
    # Plot each configuration
    for i, idx in enumerate(indices):
        # Get positions at this time
        pos = positions[idx]
        
        # Calculate color based on time progression
        color_val = i / len(indices)
        
        # Get tau or t value if available
        time_label = f"Step {idx}"
        if 'tau' in trajectory:
            time_label = f"τ = {trajectory['tau'][idx]:.2f}"
        elif 't' in trajectory:
            time_label = f"t = {trajectory['t'][idx]:.2f}"
        
        # For 2D
        if dim == 2:
            # Plot points
            fig.add_trace(
                go.Scatter(
                    x=pos[:, 0], 
                    y=pos[:, 1],
                    mode='markers+lines',
                    marker=dict(
                        size=10,
                        color=color_val,
                        colorscale='Viridis',
                        showscale=i == 0,  # Only show colorbar once
                        colorbar=dict(title='Time progression') if i == 0 else None
                    ),
                    line=dict(
                        color=f'rgba({int(255*(1-color_val))}, {int(255*color_val)}, 255, 0.5)',
                        width=2
                    ),
                    name=time_label,
                    showlegend=i == 0 or i == len(indices)-1 or i % (len(indices)//5) == 0
                )
            )
            
        # For 3D
        else:
            # Plot points and lines
            fig.add_trace(
                go.Scatter3d(
                    x=pos[:, 0], 
                    y=pos[:, 1],
                    z=pos[:, 2],
                    mode='markers+lines',
                    marker=dict(
                        size=5,
                        color=color_val,
                        colorscale='Viridis',
                        showscale=i == 0,  # Only show colorbar once
                        colorbar=dict(title='Time progression') if i == 0 else None
                    ),
                    line=dict(
                        color=f'rgba({int(255*(1-color_val))}, {int(255*color_val)}, 255, 0.5)',
                        width=2
                    ),
                    name=time_label,
                    showlegend=i == 0 or i == len(indices)-1 or i % (len(indices)//5) == 0
                )
            )
    
    # Update layout
    if dim == 2:
        fig.update_layout(
            title=title if title else 'Configuration Shape Evolution',
            xaxis_title='x',
            yaxis_title='y',
            hovermode='closest',
            legend_title='Time Steps',
            xaxis=dict(scaleanchor='y', scaleratio=1),  # Equal aspect ratio
            yaxis=dict(scaleanchor='x', scaleratio=1)
        )
    else:
        fig.update_layout(
            title=title if title else 'Configuration Shape Evolution',
            scene=dict(
                xaxis_title='x',
                yaxis_title='y',
                zaxis_title='z',
                aspectmode='cube'  # Equal aspect ratio
            ),
            legend_title='Time Steps'
        )
        
    # Add slider for time progression
    steps = []
    for i, idx in enumerate(indices):
        step = dict(
            method="update",
            args=[
                {"visible": [j == i for j in range(len(indices))]},
                {"title": f"Configuration at Step {idx}"}
            ],
            label=f"{i}"
        )
        steps.append(step)
    
    sliders = [dict(
        active=0,
        steps=steps,
        currentvalue={"prefix": "Time Step: "},
        pad={"t": 50}
    )]
    
    # Only add slider if there are enough time points
    if len(indices) > 5:
        fig.update_layout(sliders=sliders)
    
    return fig 