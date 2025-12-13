"""
Urban Environment Simulation

Simulates a GPS-challenged urban environment with buildings,
GPS-denied zones, and configurable drone dynamics.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field


@dataclass
class GPSDeniedZone:
    """A circular GPS-denied zone."""
    center_x: float
    center_y: float
    radius: float
    
    def contains(self, x: float, y: float) -> bool:
        """Check if point is inside the zone."""
        dist = np.sqrt((x - self.center_x)**2 + (y - self.center_y)**2)
        return dist < self.radius
    
    def get_quality(self, x: float, y: float) -> float:
        """Get GPS quality (0-1) based on distance from zone."""
        dist = np.sqrt((x - self.center_x)**2 + (y - self.center_y)**2)
        if dist >= self.radius * 1.5:
            return 1.0
        elif dist < self.radius:
            return max(0.05, 0.3 * (dist / self.radius))
        else:
            # Transition zone
            return 0.3 + 0.7 * ((dist - self.radius) / (0.5 * self.radius))


@dataclass
class Building:
    """A rectangular building obstacle."""
    x: float
    y: float
    width: float
    height: float
    building_height: float  # Height in meters
    
    def contains_2d(self, x: float, y: float) -> bool:
        """Check if point is inside building footprint."""
        return (self.x <= x <= self.x + self.width and
                self.y <= y <= self.y + self.height)


class UrbanEnvironment:
    """
    Urban environment with GPS-denied zones and building obstacles.
    
    Simulates realistic GPS degradation in "urban canyons" where
    tall buildings obstruct satellite signals.
    
    Parameters
    ----------
    size : float
        Environment size in meters (square)
    n_buildings : int
        Number of randomly placed buildings
    gps_denied_zones : List[Tuple[float, float, float]]
        List of (center_x, center_y, radius) tuples
    seed : int, optional
        Random seed for reproducibility
        
    Example
    -------
    >>> env = UrbanEnvironment(size=1000, n_buildings=30)
    >>> quality = env.get_gps_quality(500, 500, 50)
    >>> print(f"GPS quality: {quality:.2f}")
    """
    
    def __init__(
        self,
        size: float = 1000.0,
        n_buildings: int = 30,
        gps_denied_zones: Optional[List[Tuple[float, float, float]]] = None,
        seed: Optional[int] = None
    ):
        self.size = size
        self.n_buildings = n_buildings
        
        if seed is not None:
            np.random.seed(seed)
        
        # Create GPS-denied zones
        if gps_denied_zones is None:
            gps_denied_zones = [
                (size * 0.3, size * 0.3, 80),
                (size * 0.5, size * 0.5, 100),
                (size * 0.7, size * 0.2, 60),
            ]
        
        self.gps_zones = [
            GPSDeniedZone(x, y, r) for x, y, r in gps_denied_zones
        ]
        
        # Create buildings
        self.buildings = self._generate_buildings()
        
    def _generate_buildings(self) -> List[Building]:
        """Generate random building obstacles."""
        buildings = []
        for _ in range(self.n_buildings):
            width = np.random.uniform(20, 80)
            height = np.random.uniform(20, 80)
            x = np.random.uniform(0, self.size - width)
            y = np.random.uniform(0, self.size - height)
            building_height = np.random.uniform(20, 100)
            
            buildings.append(Building(x, y, width, height, building_height))
            
        return buildings
    
    def get_gps_quality(self, x: float, y: float, z: float = 50.0) -> float:
        """
        Get GPS signal quality at a position.
        
        Parameters
        ----------
        x : float
            X position in meters
        y : float
            Y position in meters  
        z : float
            Altitude in meters
            
        Returns
        -------
        float
            GPS quality from 0 (no signal) to 1 (perfect signal)
        """
        quality = 1.0
        
        # Check GPS-denied zones
        for zone in self.gps_zones:
            zone_quality = zone.get_quality(x, y)
            quality = min(quality, zone_quality)
            
        # Altitude improves quality (above buildings)
        if z > 80:
            quality = min(1.0, quality + 0.3)
            
        return quality
    
    def check_collision(
        self, 
        x: float, 
        y: float, 
        z: float,
        safety_margin: float = 5.0
    ) -> bool:
        """
        Check if position collides with a building.
        
        Parameters
        ----------
        x, y, z : float
            Position to check
        safety_margin : float
            Safety buffer around buildings
            
        Returns
        -------
        bool
            True if collision detected
        """
        for building in self.buildings:
            # Check if inside building footprint with margin
            in_footprint = (
                building.x - safety_margin <= x <= building.x + building.width + safety_margin and
                building.y - safety_margin <= y <= building.y + building.height + safety_margin
            )
            if in_footprint and z < building.building_height + safety_margin:
                return True
                
        return False
    
    def get_random_free_position(self, altitude: float = 50.0) -> np.ndarray:
        """Get a random position that doesn't collide with buildings."""
        for _ in range(100):
            x = np.random.uniform(50, self.size - 50)
            y = np.random.uniform(50, self.size - 50)
            if not self.check_collision(x, y, altitude):
                return np.array([x, y, altitude])
        
        # Fallback
        return np.array([self.size / 2, self.size / 2, altitude])
    
    def get_info(self) -> Dict:
        """Get environment information."""
        return {
            "size": self.size,
            "n_buildings": len(self.buildings),
            "n_gps_zones": len(self.gps_zones),
            "gps_zones": [(z.center_x, z.center_y, z.radius) for z in self.gps_zones]
        }
    
    def __repr__(self) -> str:
        return (
            f"UrbanEnvironment(size={self.size}, n_buildings={len(self.buildings)}, "
            f"n_gps_zones={len(self.gps_zones)})"
        )
