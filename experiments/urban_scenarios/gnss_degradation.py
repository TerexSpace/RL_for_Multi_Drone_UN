"""
GNSS Signal Degradation Model
Models GPS signal quality based on urban geometry, ionospheric conditions, and multipath effects.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
import json


@dataclass
class IonosphericConditions:
    """Ionospheric delay model parameters."""
    tec_value: float = 20.0  # Total Electron Content (TECU)
    solar_activity: float = 0.5  # 0-1 scale
    time_of_day: float = 12.0  # Hour (0-24)
    
    def get_delay_factor(self) -> float:
        """Compute ionospheric delay factor."""
        # Simplified Klobuchar model
        base_delay = self.tec_value / 10.0
        solar_factor = 1.0 + 0.5 * self.solar_activity
        
        # Diurnal variation
        hour_rad = (self.time_of_day - 14.0) * np.pi / 12.0
        diurnal = 1.0 + 0.3 * np.cos(hour_rad)
        
        return base_delay * solar_factor * diurnal


@dataclass
class MultipathEnvironment:
    """Multipath propagation model for urban environments."""
    building_density: float = 0.5  # 0-1 scale
    avg_building_height: float = 50.0  # meters
    street_width: float = 20.0  # meters
    surface_reflectivity: float = 0.3  # 0-1 scale
    
    def compute_multipath_error(self, pos: np.ndarray, satellites_visible: int) -> float:
        """Compute multipath-induced position error."""
        # Urban canyon effect
        canyon_factor = self.avg_building_height / max(self.street_width, 1.0)
        
        # Reflectivity contribution
        reflection_error = self.surface_reflectivity * canyon_factor * 2.0
        
        # Satellite geometry degradation
        geometry_factor = 1.0 + (8 - min(satellites_visible, 8)) * 0.2
        
        return reflection_error * geometry_factor * self.building_density


@dataclass
class GNSSReceiver:
    """GNSS receiver characteristics."""
    receiver_noise: float = 1.0  # meters
    tracking_threshold: float = 28.0  # dB-Hz
    channels: int = 12
    
    
class GNSSDegradationModel:
    """Comprehensive GNSS signal degradation model."""
    
    def __init__(self, 
                 ionosphere: Optional[IonosphericConditions] = None,
                 multipath: Optional[MultipathEnvironment] = None,
                 receiver: Optional[GNSSReceiver] = None):
        self.ionosphere = ionosphere or IonosphericConditions()
        self.multipath = multipath or MultipathEnvironment()
        self.receiver = receiver or GNSSReceiver()
        
        # Satellite constellation (simplified GPS)
        self.num_satellites = 31
        
    def compute_satellite_visibility(self, pos: np.ndarray, 
                                     building_mask: np.ndarray = None) -> int:
        """Compute number of visible satellites based on position and obstructions."""
        base_visible = np.random.randint(6, 12)
        
        # Altitude bonus
        altitude = pos[2] if len(pos) > 2 else 0
        altitude_bonus = min(int(altitude / 20), 4)
        
        # Urban penalty based on multipath environment
        urban_penalty = int(self.multipath.building_density * 4)
        
        visible = max(0, min(12, base_visible + altitude_bonus - urban_penalty))
        return visible
    
    def compute_pdop(self, satellites_visible: int) -> float:
        """Compute Position Dilution of Precision."""
        if satellites_visible < 4:
            return float('inf')
        
        # Simplified PDOP model
        base_pdop = 2.0
        geometry_factor = 1.0 + (8 - min(satellites_visible, 8)) * 0.5
        return base_pdop * geometry_factor
    
    def compute_signal_quality(self, pos: np.ndarray,
                               weather_factor: float = 1.0) -> dict:
        """Compute comprehensive GNSS signal quality metrics."""
        satellites = self.compute_satellite_visibility(pos)
        pdop = self.compute_pdop(satellites)
        
        # Ionospheric delay
        iono_delay = self.ionosphere.get_delay_factor()
        
        # Multipath error
        multipath_error = self.multipath.compute_multipath_error(pos, satellites)
        
        # Total position error estimate
        if satellites >= 4:
            base_error = self.receiver.receiver_noise * pdop
            total_error = np.sqrt(base_error**2 + iono_delay**2 + multipath_error**2)
            total_error *= weather_factor
        else:
            total_error = float('inf')
            
        # Convert to quality score (0-1)
        if total_error == float('inf'):
            quality = 0.0
        else:
            quality = 1.0 / (1.0 + total_error / 5.0)
            
        return {
            'quality': quality,
            'satellites_visible': satellites,
            'pdop': pdop,
            'iono_delay': iono_delay,
            'multipath_error': multipath_error,
            'total_error_m': total_error
        }
    
    def simulate_position_estimate(self, true_pos: np.ndarray,
                                   weather_factor: float = 1.0) -> Tuple[np.ndarray, dict]:
        """Simulate GNSS position estimate with realistic errors."""
        metrics = self.compute_signal_quality(true_pos, weather_factor)
        
        if metrics['quality'] < 0.1:
            # No fix available
            return None, metrics
            
        # Add position error
        error_std = metrics['total_error_m']
        error = np.random.normal(0, error_std, 3)
        error[2] *= 1.5  # Vertical error typically larger
        
        estimated_pos = true_pos + error
        
        return estimated_pos, metrics


class DynamicGNSSEnvironment:
    """Dynamic GNSS environment with time-varying conditions."""
    
    def __init__(self, urban_area_size: float = 1000.0):
        self.size = urban_area_size
        self.time = 0.0
        
        # Define urban canyon zones with varying severity
        self.canyon_zones = [
            {'bounds': (100, 100, 250, 350), 'severity': 0.9, 'type': 'deep_canyon'},
            {'bounds': (400, 200, 550, 450), 'severity': 0.8, 'type': 'moderate_canyon'},
            {'bounds': (700, 50, 850, 200), 'severity': 0.6, 'type': 'open_plaza'},
            {'bounds': (150, 600, 400, 800), 'severity': 0.85, 'type': 'tunnel_entrance'},
            {'bounds': (600, 550, 900, 750), 'severity': 0.7, 'type': 'suburban'},
        ]
        
    def get_zone_model(self, pos: np.ndarray) -> GNSSDegradationModel:
        """Get appropriate degradation model for position."""
        x, y = pos[0], pos[1]
        
        for zone in self.canyon_zones:
            bx, by, ex, ey = zone['bounds']
            if bx <= x <= ex and by <= y <= ey:
                severity = zone['severity']
                return GNSSDegradationModel(
                    ionosphere=IonosphericConditions(
                        tec_value=20.0 + 10.0 * severity,
                        time_of_day=self.time % 24
                    ),
                    multipath=MultipathEnvironment(
                        building_density=severity,
                        avg_building_height=40 + 80 * severity,
                        street_width=30 - 20 * severity
                    )
                )
                
        # Default open sky model
        return GNSSDegradationModel()
    
    def update_time(self, dt: float):
        """Update simulation time."""
        self.time += dt / 3600.0  # Convert to hours
        
    def get_quality_at(self, pos: np.ndarray, weather: float = 1.0) -> float:
        """Get GNSS quality at position."""
        model = self.get_zone_model(pos)
        metrics = model.compute_signal_quality(pos, weather)
        return metrics['quality']


def create_quality_heatmap(env: DynamicGNSSEnvironment, 
                           resolution: int = 50) -> np.ndarray:
    """Generate GNSS quality heatmap for visualization."""
    heatmap = np.zeros((resolution, resolution))
    step = env.size / resolution
    
    for i in range(resolution):
        for j in range(resolution):
            pos = np.array([i * step, j * step, 30.0])
            heatmap[j, i] = env.get_quality_at(pos)
            
    return heatmap


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    print("Testing GNSS Degradation Model...")
    
    env = DynamicGNSSEnvironment(1000.0)
    heatmap = create_quality_heatmap(env, 100)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(heatmap, origin='lower', cmap='RdYlGn', 
               extent=[0, 1000, 0, 1000], vmin=0, vmax=1)
    plt.colorbar(label='GNSS Quality')
    plt.title('Urban GNSS Signal Quality Map')
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    
    # Mark canyon zones
    for zone in env.canyon_zones:
        bx, by, ex, ey = zone['bounds']
        rect = plt.Rectangle((bx, by), ex-bx, ey-by, 
                             fill=False, edgecolor='black', linestyle='--')
        plt.gca().add_patch(rect)
        
    plt.savefig('experiments/gnss_quality_map.png', dpi=150)
    print("Saved GNSS quality map to experiments/gnss_quality_map.png")
