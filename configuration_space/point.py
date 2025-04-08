from typing import List, Dict, Union, Optional
import numpy as np
from numpy.typing import NDArray

class Point:
    """
    Represents a point in configuration space with position, weight, and additional properties.
    
    Attributes:
        position (NDArray[np.float64]): Array of coordinates representing position
        weight (float): Weight factor for the point (default: 1.0)
        properties (Dict): Additional properties/metadata for the point
    """
    
    def __init__(
        self,
        position: Union[List[float], NDArray[np.float64]],
        weight: float = 1.0,
        properties: Optional[Dict] = None
    ) -> None:
        """
        Initialize a Point instance.
        
        Args:
            position: List or numpy array of coordinates
            weight: Weight factor for the point (default: 1.0)
            properties: Optional dictionary of additional properties
        """
        self.position = np.array(position, dtype=np.float64)
        self.weight = float(weight)
        self.properties = properties if properties is not None else {}
        
        if weight <= 0:
            raise ValueError("Weight must be positive")
            
    def dimension(self) -> int:
        """Return the dimension of the space this point exists in."""
        return len(self.position)
    
    def __repr__(self) -> str:
        """Return string representation of the Point."""
        return f"Point(position={self.position.tolist()}, weight={self.weight}, properties={self.properties})" 