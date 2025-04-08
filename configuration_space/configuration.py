from typing import List, Optional, Tuple, Union
import numpy as np
from numpy.typing import NDArray
from .point import Point

class Configuration:
    """
    Represents a configuration of multiple points with scale, shape, and orientation.
    
    Attributes:
        points (List[Point]): List of points in the configuration
        center_of_mass (NDArray[np.float64]): Center of mass of the configuration
        total_weight (float): Sum of all point weights
        scale_factor (float): Scale factor s = sqrt(sum(w_i |r_i|^2) / W)
        sigma (float): Logarithmic scale coordinate ln(s)
    """
    
    def __init__(self, points: List[Point]) -> None:
        """
        Initialize a Configuration instance.
        
        Args:
            points: List of Point objects forming the configuration
        
        Raises:
            ValueError: If points list is empty or points have different dimensions
        """
        if not points:
            raise ValueError("Configuration must contain at least one point")
            
        self.points = points
        self._validate_dimensions()
        self.total_weight = sum(p.weight for p in points)
        self._compute_center_of_mass()
        self._compute_scale_factor()
        
    def _validate_dimensions(self) -> None:
        """Ensure all points have the same dimension."""
        dim = self.points[0].dimension()
        if not all(p.dimension() == dim for p in self.points):
            raise ValueError("All points must have the same dimension")
            
    def _compute_center_of_mass(self) -> None:
        """Compute the center of mass of the configuration."""
        weighted_sum = sum(
            p.weight * p.position for p in self.points
        )
        self.center_of_mass = weighted_sum / self.total_weight
        
    def _compute_scale_factor(self) -> None:
        """Compute scale factor s and logarithmic scale coordinate sigma."""
        squared_distances = sum(
            p.weight * np.sum((p.position - self.center_of_mass) ** 2)
            for p in self.points
        )
        self.scale_factor = np.sqrt(squared_distances / self.total_weight)
        self.sigma = np.log(self.scale_factor) if self.scale_factor > 0 else float('-inf')
        
    def get_shape_coordinates(self) -> NDArray[np.float64]:
        """
        Compute shape coordinates (normalized positions relative to center of mass).
        
        Returns:
            Array of shape coordinates for each point
        """
        return np.array([
            (p.position - self.center_of_mass) / self.scale_factor
            for p in self.points
        ])
        
    def get_orientation(self) -> Union[float, Tuple[float, float, float]]:
        """
        Compute orientation relative to reference axes.
        
        For 2D: Returns angle in radians from x-axis to first principal component.
        For 3D: Returns tuple of Euler angles (phi, theta, psi) in radians:
            - phi: rotation around z-axis (yaw)
            - theta: rotation around y'-axis (pitch)
            - psi: rotation around x''-axis (roll)
        
        Returns:
            For 2D: Single angle in radians
            For 3D: Tuple of three Euler angles in radians (phi, theta, psi)
            
        Raises:
            ValueError: If configuration is not 2D or 3D
        """
        dim = self.points[0].dimension()
        if dim not in (2, 3):
            raise ValueError("Orientation calculation only supported for 2D and 3D")
            
        # Get positions relative to center of mass
        positions = self.get_shape_coordinates()
        
        # Compute covariance matrix
        cov = np.cov(positions.T)
        
        # Get eigenvectors and eigenvalues
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        # Sort by eigenvalues in descending order
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        if dim == 2:
            # For 2D, return angle with x-axis
            principal_component = eigenvectors[:, 0]
            return np.arctan2(principal_component[1], principal_component[0])
        else:
            # For 3D, compute Euler angles
            # Get the principal direction (first eigenvector)
            principal_direction = eigenvectors[:, 0]
            
            # Normalize the direction vector
            principal_direction = principal_direction / np.linalg.norm(principal_direction)
            
            # Convert to spherical coordinates first
            r = np.linalg.norm(principal_direction)
            theta = np.arccos(principal_direction[2] / r)  # Polar angle from z-axis
            phi = np.arctan2(principal_direction[1], principal_direction[0])  # Azimuthal angle in x-y plane
            
            # Handle special cases
            if abs(principal_direction[2]) > 1 - 1e-10:  # Almost parallel to z-axis
                phi = 0.0
                theta = 0.0 if principal_direction[2] > 0 else np.pi
                psi = 0.0
            elif abs(principal_direction[2]) < 1e-10:  # In x-y plane
                theta = np.pi/2
                psi = 0.0
                # phi is already correct from arctan2
            else:
                # For general case, we keep theta and phi from spherical coordinates
                # and set psi based on secondary direction
                secondary_direction = eigenvectors[:, 1]
                # Project secondary direction onto plane perpendicular to principal direction
                proj = secondary_direction - np.dot(secondary_direction, principal_direction) * principal_direction
                proj = proj / np.linalg.norm(proj)
                # Compute psi as angle between projected vector and reference direction
                ref = np.cross([0, 0, 1], principal_direction)
                ref = ref / np.linalg.norm(ref)
                psi = np.arctan2(np.dot(np.cross(ref, proj), principal_direction), np.dot(ref, proj))
            
            return (phi, theta, psi)
        
    def fix_center_of_mass(self) -> None:
        """Adjust positions to fix center of mass at origin."""
        for point in self.points:
            point.position = point.position - self.center_of_mass
        self.center_of_mass = np.zeros_like(self.center_of_mass)
        self._compute_scale_factor()
        
    def __repr__(self) -> str:
        """Return string representation of the Configuration."""
        return (f"Configuration(points={self.points}, "
                f"scale={self.scale_factor:.3f}, "
                f"sigma={self.sigma:.3f})") 