"""
Static Plots Module

This module provides functions for static visualization of configurations
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
from mapping_functions import tau_to_r, r_to_tau


def plot_configuration(
    config: Configuration,
    plane: str = 'r',
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 6),
    show_scale: bool = True,
    show_center_of_mass: bool = True,
    point_labels: bool = True,
    axes_equal: bool = True,
    use_plotly: bool = False,
    plot_kwargs: Optional[Dict[str, Any]] = None
) -> Union[Tuple[Figure, Axes], go.Figure]:
    """
    Plot a static visualization of a configuration.
    
    Args:
        config: The Configuration to visualize
        plane: Which plane to visualize in ('r' or 'tau')
        title: Optional plot title
        figsize: Figure size as (width, height) in inches
        show_scale: Whether to display scale information
        show_center_of_mass: Whether to show the center of mass
        point_labels: Whether to show point labels
        axes_equal: Whether to set equal aspect ratio for axes
        use_plotly: Use Plotly for interactive visualization
        plot_kwargs: Additional keyword arguments for plot customization
    
    Returns:
        Matplotlib Figure and Axes objects, or Plotly Figure object if use_plotly=True
    """
    if plot_kwargs is None:
        plot_kwargs = {}
    
    # Get dimensions and positions
    dim = config.points[0].dimension()
    if dim not in (2, 3):
        raise ValueError("Visualization only supported for 2D and 3D configurations")
    
    # Extract positions
    positions = np.array([p.position for p in config.points])
    weights = np.array([p.weight for p in config.points])
    
    # Transform to tau-plane if needed
    if plane == 'tau':
        # Apply r_to_tau mapping to each position
        # Note: Real transformations would use the mapping_functions module with Hyperreal
        # but for visualization we can use a simple numerical approximation
        try:
            # Handle positions that may be at or near origin specially
            # For visualization, we'll use a numerical approximation
            tau_positions = np.zeros_like(positions)
            for i, pos in enumerate(positions):
                tau_pos = []
                for coord in pos:
                    if abs(coord) < 1e-10:  # Near zero
                        tau_coord = 1e10 * np.sign(coord) if coord != 0 else 1e10
                    else:
                        tau_coord = 1.0 / coord
                    tau_pos.append(tau_coord)
                tau_positions[i] = tau_pos
            positions = tau_positions
        except Exception as e:
            raise ValueError(f"Error transforming to τ-plane: {e}")
    
    if use_plotly:
        return _plot_configuration_plotly(
            positions, weights, dim, config, plane, title, 
            show_scale, show_center_of_mass, point_labels, **plot_kwargs
        )
    else:
        return _plot_configuration_matplotlib(
            positions, weights, dim, config, plane, title, figsize,
            show_scale, show_center_of_mass, point_labels, axes_equal, **plot_kwargs
        )

def _plot_configuration_matplotlib(
    positions: np.ndarray,
    weights: np.ndarray,
    dim: int,
    config: Configuration,
    plane: str,
    title: Optional[str],
    figsize: Tuple[float, float],
    show_scale: bool,
    show_center_of_mass: bool,
    point_labels: bool,
    axes_equal: bool,
    **kwargs
) -> Tuple[Figure, Axes]:
    """Helper function for matplotlib plotting."""
    fig, ax = plt.subplots(figsize=figsize)
    
    # Normalize weights for plotting
    norm_weights = 30 * (weights / max(weights))
    
    # Choose color scheme based on plane
    colors = 'viridis' if plane == 'r' else 'plasma'
    cmap = plt.get_cmap(colors)
    
    if dim == 2:
        # 2D plot
        scatter = ax.scatter(
            positions[:, 0], positions[:, 1], 
            s=norm_weights, c=np.arange(len(positions)), 
            cmap=cmap, alpha=0.7, **kwargs
        )
        
        # Add labels if requested
        if point_labels:
            for i, pos in enumerate(positions):
                ax.text(pos[0], pos[1], f" P{i+1}", fontsize=9)
        
        # Show center of mass
        if show_center_of_mass and plane == 'r':
            ax.scatter(
                [config.center_of_mass[0]], [config.center_of_mass[1]],
                marker='x', color='red', s=100, label='Center of Mass'
            )
        
        # Set equal aspect ratio if requested
        if axes_equal:
            ax.set_aspect('equal')
            
    elif dim == 3:
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(
            positions[:, 0], positions[:, 1], positions[:, 2],
            s=norm_weights, c=np.arange(len(positions)), 
            cmap=cmap, alpha=0.7, **kwargs
        )
        
        # Add labels if requested
        if point_labels:
            for i, pos in enumerate(positions):
                ax.text(pos[0], pos[1], pos[2], f" P{i+1}", fontsize=9)
        
        # Show center of mass
        if show_center_of_mass and plane == 'r':
            ax.scatter(
                [config.center_of_mass[0]], 
                [config.center_of_mass[1]], 
                [config.center_of_mass[2]],
                marker='x', color='red', s=100, label='Center of Mass'
            )
    
    # Set labels and title
    plane_label = 'r-plane' if plane == 'r' else 'τ-plane'
    ax.set_xlabel(f'x ({plane_label})')
    ax.set_ylabel(f'y ({plane_label})')
    if dim == 3:
        ax.set_zlabel(f'z ({plane_label})')
    
    # Set plot title
    if title:
        ax.set_title(title)
    else:
        scale_info = f", σ={config.sigma:.3f}" if show_scale else ""
        ax.set_title(f"Configuration in {plane_label}{scale_info}")
    
    # Add legend
    if show_center_of_mass and plane == 'r':
        ax.legend()
    
    # Adjust limits to make small values visible
    if plane == 'tau':
        # In τ-plane, we may need to handle very large values
        # Use logarithmic scale or adjust limits based on data
        ax.set_xscale('symlog')
        ax.set_yscale('symlog')
        if dim == 3:
            ax.set_zscale('symlog')
    
    plt.tight_layout()
    return fig, ax

def _plot_configuration_plotly(
    positions: np.ndarray,
    weights: np.ndarray,
    dim: int,
    config: Configuration,
    plane: str,
    title: Optional[str],
    show_scale: bool,
    show_center_of_mass: bool,
    point_labels: bool,
    **kwargs
) -> go.Figure:
    """Helper function for plotly interactive plotting."""
    # Normalize weights for plotting
    norm_weights = 10 * (weights / max(weights))
    
    # Choose color scheme based on plane
    colorscale = 'Viridis' if plane == 'r' else 'Plasma'
    
    # Initialize figure
    plane_label = 'r-plane' if plane == 'r' else 'τ-plane'
    
    if title is None:
        scale_info = f", σ={config.sigma:.3f}" if show_scale else ""
        title = f"Configuration in {plane_label}{scale_info}"
    
    # Create figure
    fig = go.Figure()
    
    # Add scatter plot for points
    if dim == 2:
        # 2D plot
        marker_data = {
            'size': norm_weights,
            'color': np.arange(len(positions)),
            'colorscale': colorscale,
            'opacity': 0.7,
            'line': {'width': 1, 'color': 'DarkSlateGrey'}
        }
        
        # Add points
        scatter = go.Scatter(
            x=positions[:, 0], 
            y=positions[:, 1],
            mode='markers' if not point_labels else 'markers+text',
            marker=marker_data,
            text=[f"P{i+1}" for i in range(len(positions))] if point_labels else None,
            textposition="top center" if point_labels else None,
            name='Points'
        )
        fig.add_trace(scatter)
        
        # Add center of mass
        if show_center_of_mass and plane == 'r':
            com = go.Scatter(
                x=[config.center_of_mass[0]],
                y=[config.center_of_mass[1]],
                mode='markers',
                marker={'size': 12, 'symbol': 'x', 'color': 'red'},
                name='Center of Mass'
            )
            fig.add_trace(com)
            
        # Set layout
        fig.update_layout(
            title=title,
            xaxis_title=f'x ({plane_label})',
            yaxis_title=f'y ({plane_label})',
            hovermode='closest'
        )
        
    elif dim == 3:
        # 3D plot
        marker_data = {
            'size': norm_weights,
            'color': np.arange(len(positions)),
            'colorscale': colorscale,
            'opacity': 0.7,
            'line': {'width': 1, 'color': 'DarkSlateGrey'}
        }
        
        # Add points
        scatter = go.Scatter3d(
            x=positions[:, 0], 
            y=positions[:, 1],
            z=positions[:, 2],
            mode='markers' if not point_labels else 'markers+text',
            marker=marker_data,
            text=[f"P{i+1}" for i in range(len(positions))] if point_labels else None,
            name='Points'
        )
        fig.add_trace(scatter)
        
        # Add center of mass
        if show_center_of_mass and plane == 'r':
            com = go.Scatter3d(
                x=[config.center_of_mass[0]],
                y=[config.center_of_mass[1]],
                z=[config.center_of_mass[2]],
                mode='markers',
                marker={'size': 8, 'symbol': 'x', 'color': 'red'},
                name='Center of Mass'
            )
            fig.add_trace(com)
            
        # Set layout
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title=f'x ({plane_label})',
                yaxis_title=f'y ({plane_label})',
                zaxis_title=f'z ({plane_label})',
                aspectmode='cube'
            )
        )
    
    # Handle special scaling for tau-plane
    if plane == 'tau':
        # Use logarithmic scale for tau-plane visualization
        if dim == 2:
            fig.update_xaxes(type="log", autorange=True)
            fig.update_yaxes(type="log", autorange=True)
        else:
            fig.update_layout(
                scene=dict(
                    xaxis_type="log",
                    yaxis_type="log",
                    zaxis_type="log",
                    aspectmode='cube'
                )
            )
    
    # Add buttons to switch between linear and log scales
    axis_names = ['xaxis', 'yaxis'] if dim == 2 else ['scene.xaxis', 'scene.yaxis', 'scene.zaxis']
    buttons = []
    buttons.append(dict(
        label="Linear Scale", 
        method="relayout",
        args=[{f"{axis}.type": "linear" for axis in axis_names}]
    ))
    buttons.append(dict(
        label="Log Scale", 
        method="relayout",
        args=[{f"{axis}.type": "log" for axis in axis_names}]
    ))
    
    fig.update_layout(
        updatemenus=[dict(
            type="buttons",
            direction="right",
            buttons=buttons,
            pad={"r": 10, "t": 10},
            showactive=True,
            x=0.1,
            y=1.15,
            xanchor="left",
            yanchor="top"
        )]
    )
    
    return fig

def plot_configuration_comparison(
    config: Configuration,
    figsize: Tuple[float, float] = (12, 5),
    show_scale: bool = True,
    show_center_of_mass: bool = True,
    point_labels: bool = True,
    use_plotly: bool = False,
    **kwargs
) -> Union[Tuple[Figure, Tuple[Axes, Axes]], go.Figure]:
    """
    Plot a side-by-side comparison of a configuration in both r-plane and τ-plane.
    
    Args:
        config: The Configuration to visualize
        figsize: Figure size as (width, height) in inches
        show_scale: Whether to display scale information
        show_center_of_mass: Whether to show the center of mass
        point_labels: Whether to show point labels
        use_plotly: Use Plotly for interactive visualization
        **kwargs: Additional keyword arguments passed to the plotting functions
    
    Returns:
        If use_plotly=False: Tuple of (Figure, (left_axes, right_axes))
        If use_plotly=True: Plotly Figure object with two subplots
    """
    if use_plotly:
        # Create subplot figure with two plots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("r-plane", "τ-plane"),
            specs=[[{"type": "scatter"}, {"type": "scatter"}]] if config.points[0].dimension() == 2 
                  else [[{"type": "scene"}, {"type": "scene"}]]
        )
        
        # Get r-plane plot and add its traces to the first subplot
        r_fig = _plot_configuration_plotly(
            np.array([p.position for p in config.points]),
            np.array([p.weight for p in config.points]),
            config.points[0].dimension(),
            config,
            'r',
            None,
            show_scale,
            show_center_of_mass,
            point_labels,
            **kwargs
        )
        
        # Get tau-plane positions
        tau_positions = np.array([
            [1.0/coord if abs(coord) > 1e-10 else 1e10 * np.sign(coord) if coord != 0 else 1e10
             for coord in p.position]
            for p in config.points
        ])
        
        # Get tau-plane plot and add its traces to the second subplot
        tau_fig = _plot_configuration_plotly(
            tau_positions,
            np.array([p.weight for p in config.points]),
            config.points[0].dimension(),
            config,
            'tau',
            None,
            show_scale,
            False,  # No center of mass in tau-plane
            point_labels,
            **kwargs
        )
        
        # Add traces from individual figures to the subplots
        for trace in r_fig.data:
            fig.add_trace(trace, row=1, col=1)
            
        for trace in tau_fig.data:
            fig.add_trace(trace, row=1, col=2)
        
        # Update layout
        fig.update_layout(
            title_text=f"Configuration Comparison (σ={config.sigma:.3f})" if show_scale 
                      else "Configuration Comparison",
            height=500
        )
        
        return fig
        
    else:
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot r-plane on the left
        _, _ = _plot_configuration_matplotlib(
            np.array([p.position for p in config.points]),
            np.array([p.weight for p in config.points]),
            config.points[0].dimension(),
            config,
            'r',
            "r-plane",
            figsize,
            show_scale,
            show_center_of_mass,
            point_labels,
            True,  # axes_equal
            **kwargs
        )
        
        # Get tau-plane positions
        tau_positions = np.array([
            [1.0/coord if abs(coord) > 1e-10 else 1e10 * np.sign(coord) if coord != 0 else 1e10
             for coord in p.position]
            for p in config.points
        ])
        
        # Plot tau-plane on the right
        _, _ = _plot_configuration_matplotlib(
            tau_positions,
            np.array([p.weight for p in config.points]),
            config.points[0].dimension(),
            config,
            'tau',
            "τ-plane",
            figsize,
            show_scale,
            False,  # No center of mass in tau-plane
            point_labels,
            True,  # axes_equal
            **kwargs
        )
        
        # Set title for the whole figure
        if show_scale:
            fig.suptitle(f"Configuration Comparison (σ={config.sigma:.3f})")
        else:
            fig.suptitle("Configuration Comparison")
            
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        
        return fig, (ax1, ax2) 