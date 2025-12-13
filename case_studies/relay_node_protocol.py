"""
GPS-Denied Navigation Case Study
Demonstrates cooperative re-localization using relay nodes in urban canyons.
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Tuple, Dict
import json


@dataclass
class RelayNode:
    """GPS relay node for cooperative localization."""
    id: int
    pos: np.ndarray
    gps_quality: float = 1.0
    is_anchor: bool = False
    range_measurements: Dict[int, float] = field(default_factory=dict)
    

class GPSDeniedNavigationStudy:
    """
    Case Study: Cooperative Navigation Through GPS-Denied Urban Canyons
    
    Demonstrates:
    - GPS relay node positioning
    - Cooperative ranging for localization
    - Formation maintenance without GPS
    - Graceful degradation
    """
    
    def __init__(self, num_drones: int = 6):
        self.num_drones = num_drones
        
        # Canyon geometry
        self.canyon_entry = np.array([50, 500, 30])
        self.canyon_exit = np.array([950, 500, 30])
        self.canyon_width = 50
        
        # GPS quality along canyon
        self.gps_profile = [
            (0, 150, 0.9),      # Entry - good GPS
            (150, 400, 0.1),   # Deep canyon - very poor
            (400, 600, 0.05),  # Deepest - near zero
            (600, 800, 0.15),  # Recovery zone
            (800, 1000, 0.85)  # Exit - good GPS
        ]
        
        self.drones: List[RelayNode] = []
        
    def setup_scenario(self):
        """Initialize swarm with relay configuration."""
        # Assign roles: first 2 are anchors (GPS relay), rest are followers
        for i in range(self.num_drones):
            start_offset = np.array([i * 8, 0, i * 3])
            drone = RelayNode(
                id=i,
                pos=self.canyon_entry + start_offset,
                is_anchor=(i < 2)  # First 2 drones are anchors
            )
            self.drones.append(drone)
            
    def get_gps_quality(self, x: float) -> float:
        """Get GPS quality based on x-position."""
        for start, end, quality in self.gps_profile:
            if start <= x <= end:
                return quality + np.random.uniform(-0.02, 0.02)
        return 0.9
    
    def measure_ranges(self):
        """Simulate UWB/acoustic range measurements between drones."""
        for d in self.drones:
            d.range_measurements = {}
            for other in self.drones:
                if d.id != other.id:
                    true_dist = np.linalg.norm(d.pos - other.pos)
                    # Add measurement noise
                    noise = np.random.normal(0, 0.5)  # 0.5m std dev
                    d.range_measurements[other.id] = true_dist + noise
                    
    def cooperative_localization(self, drone: RelayNode) -> np.ndarray:
        """Estimate position using cooperative ranging."""
        if drone.is_anchor:
            # Anchors use their GPS (with noise based on quality)
            gps_q = self.get_gps_quality(drone.pos[0])
            noise = np.random.normal(0, (1.0 - gps_q) * 5.0, 3)
            return drone.pos + noise
            
        # Non-anchors use multilateration from anchors
        anchor_positions = []
        ranges = []
        
        for other in self.drones:
            if other.is_anchor and other.id in drone.range_measurements:
                anchor_positions.append(other.pos.copy())
                ranges.append(drone.range_measurements[other.id])
                
        if len(anchor_positions) < 2:
            # Not enough anchors - use dead reckoning
            return drone.pos + np.random.normal(0, 3.0, 3)
            
        # Simple weighted average based on range (simplified multilateration)
        weights = [1.0 / (r + 1.0) for r in ranges]
        weight_sum = sum(weights)
        
        estimated_pos = np.zeros(3)
        for pos, w, r in zip(anchor_positions, weights, ranges):
            # Move from anchor position toward estimated range
            direction = drone.pos - pos
            dist = np.linalg.norm(direction)
            if dist > 0:
                direction = direction / dist
            estimated_pos += w * (pos + direction * r)
            
        return estimated_pos / weight_sum
    
    def run_simulation(self, steps: int = 300) -> Dict:
        """Run canyon transit simulation."""
        self.setup_scenario()
        np.random.seed(42)
        
        dt = 0.5
        target = self.canyon_exit.copy()
        
        results = {
            'paths': [[] for _ in self.drones],
            'localization_errors': [[] for _ in self.drones],
            'gps_qualities': [],
            'formation_errors': [],
            'anchor_separations': []
        }
        
        for t in range(steps):
            # Measure ranges
            self.measure_ranges()
            
            # Average GPS quality at current swarm position
            avg_x = np.mean([d.pos[0] for d in self.drones])
            avg_gps = self.get_gps_quality(avg_x)
            results['gps_qualities'].append(avg_gps)
            
            # Update each drone
            for d in self.drones:
                # Get localization estimate
                d.gps_quality = self.get_gps_quality(d.pos[0])
                estimated_pos = self.cooperative_localization(d)
                
                # Compute localization error
                error = np.linalg.norm(estimated_pos - d.pos)
                results['localization_errors'][d.id].append(error)
                
                # Navigation toward target
                to_target = target - estimated_pos
                dist = np.linalg.norm(to_target)
                
                if dist > 10:
                    direction = to_target / dist
                    
                    # Anchors lead, followers maintain formation
                    if d.is_anchor:
                        speed = 4.0
                    else:
                        # Formation keeping: stay near centroid
                        centroid = np.mean([dr.pos for dr in self.drones], axis=0)
                        to_centroid = centroid - d.pos
                        direction = 0.7 * direction + 0.3 * to_centroid / (np.linalg.norm(to_centroid) + 1e-6)
                        direction = direction / (np.linalg.norm(direction) + 1e-6)
                        speed = 3.5
                        
                    d.pos = d.pos + direction * speed * dt
                    
                results['paths'][d.id].append(d.pos.copy())
                
            # Formation error (spread from centroid)
            centroid = np.mean([d.pos for d in self.drones], axis=0)
            formation_error = np.mean([np.linalg.norm(d.pos - centroid) for d in self.drones])
            results['formation_errors'].append(formation_error)
            
            # Anchor separation (for relay coverage)
            anchors = [d for d in self.drones if d.is_anchor]
            if len(anchors) >= 2:
                anchor_sep = np.linalg.norm(anchors[0].pos - anchors[1].pos)
                results['anchor_separations'].append(anchor_sep)
                
        # Summary
        results['summary'] = {
            'avg_localization_error': np.mean([np.mean(e) for e in results['localization_errors']]),
            'max_localization_error': np.max([np.max(e) for e in results['localization_errors']]),
            'avg_formation_error': np.mean(results['formation_errors']),
            'min_gps_quality': min(results['gps_qualities']),
            'mission_complete': all(np.linalg.norm(d.pos - target) < 50 for d in self.drones)
        }
        
        return results
    
    def visualize_results(self, results: Dict, filename: str = "gps_denied_navigation.png"):
        """Generate visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Drone paths through canyon
        ax = axes[0, 0]
        
        # Draw canyon walls
        ax.fill_between([0, 1000], [450, 450], [0, 0], color='gray', alpha=0.2)
        ax.fill_between([0, 1000], [550, 550], [1000, 1000], color='gray', alpha=0.2)
        
        # Draw GPS quality gradient
        for start, end, quality in self.gps_profile:
            color = plt.cm.RdYlGn(quality)
            ax.axvspan(start, end, ymin=0.45, ymax=0.55, color=color, alpha=0.3)
            
        # Draw paths
        colors = ['blue', 'cyan', 'green', 'orange', 'red', 'purple']
        for i, (path, color) in enumerate(zip(results['paths'], colors)):
            if path:
                path_arr = np.array(path)
                label = f'Anchor {i}' if i < 2 else f'Drone {i}'
                linestyle = '-' if i < 2 else '--'
                ax.plot(path_arr[:, 0], path_arr[:, 1], color=color, 
                       linewidth=2 if i < 2 else 1, label=label, linestyle=linestyle)
                       
        ax.axvline(50, color='green', linestyle=':', label='Entry')
        ax.axvline(950, color='red', linestyle=':', label='Exit')
        
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_title('Canyon Transit Trajectories')
        ax.legend(loc='upper right', fontsize=8)
        ax.set_xlim(-20, 1020)
        ax.set_ylim(400, 600)
        
        # Plot 2: Localization error over time
        ax2 = axes[0, 1]
        
        for i, errors in enumerate(results['localization_errors']):
            label = f'Anchor {i}' if i < 2 else f'Drone {i}'
            ax2.plot(errors, label=label, alpha=0.7)
            
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Localization Error (m)')
        ax2.set_title('Localization Error Over Time')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: GPS quality profile
        ax3 = axes[1, 0]
        
        ax3.plot(results['gps_qualities'], 'b-', linewidth=2)
        ax3.fill_between(range(len(results['gps_qualities'])), 
                         results['gps_qualities'], alpha=0.3)
        ax3.axhline(0.5, color='orange', linestyle='--', label='Degraded Threshold')
        ax3.axhline(0.2, color='red', linestyle='--', label='Denied Threshold')
        
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('GPS Quality')
        ax3.set_title('GPS Signal Quality During Transit')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1)
        
        # Plot 4: Formation error
        ax4 = axes[1, 1]
        
        ax4.plot(results['formation_errors'], 'g-', linewidth=2)
        ax4.set_xlabel('Time Step')
        ax4.set_ylabel('Formation Spread (m)')
        ax4.set_title('Formation Maintenance')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        return filename


def run_case_study():
    """Execute GPS-denied navigation case study."""
    print("=" * 60)
    print("CASE STUDY: GPS-Denied Urban Canyon Navigation")
    print("=" * 60)
    
    study = GPSDeniedNavigationStudy(num_drones=6)
    
    print("\nSimulating canyon transit...")
    results = study.run_simulation(steps=300)
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    summary = results['summary']
    print(f"\nMission Complete: {'Yes' if summary['mission_complete'] else 'No'}")
    print(f"Average Localization Error: {summary['avg_localization_error']:.2f} m")
    print(f"Maximum Localization Error: {summary['max_localization_error']:.2f} m")
    print(f"Average Formation Error: {summary['avg_formation_error']:.2f} m")
    print(f"Minimum GPS Quality Encountered: {summary['min_gps_quality']:.2f}")
    
    print("\nGenerating visualization...")
    study.visualize_results(results, "case_studies/gps_denied_navigation/results/canyon_navigation.png")
    
    with open("case_studies/gps_denied_navigation/results/navigation_results.json", "w") as f:
        json.dump(results['summary'], f, indent=2)
        
    print("\nCase study complete!")
    print("Results saved to case_studies/gps_denied_navigation/results/")
    
    return results


if __name__ == "__main__":
    run_case_study()
