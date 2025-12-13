"""
Safety Protocols Module
Implements geofencing, kill-switch, and ground-truth tracking for CIGRL validation.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Callable
from enum import Enum
import json


class SafetyLevel(Enum):
    """Safety severity levels."""
    NORMAL = 0
    WARNING = 1
    CRITICAL = 2
    EMERGENCY = 3


@dataclass
class Geofence:
    """3D geofence boundary."""
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float
    soft_margin: float = 20.0  # Soft boundary warning margin
    
    def check_position(self, pos: np.ndarray) -> Tuple[SafetyLevel, str]:
        """Check if position is within geofence."""
        x, y, z = pos[0], pos[1], pos[2]
        
        # Hard boundary violations
        if x < self.x_min or x > self.x_max:
            return SafetyLevel.EMERGENCY, f"X boundary violation: {x:.1f}"
        if y < self.y_min or y > self.y_max:
            return SafetyLevel.EMERGENCY, f"Y boundary violation: {y:.1f}"
        if z < self.z_min or z > self.z_max:
            return SafetyLevel.EMERGENCY, f"Z boundary violation: {z:.1f}"
            
        # Soft boundary warnings
        margin = self.soft_margin
        warnings = []
        
        if x < self.x_min + margin or x > self.x_max - margin:
            warnings.append("approaching X boundary")
        if y < self.y_min + margin or y > self.y_max - margin:
            warnings.append("approaching Y boundary")
        if z < self.z_min + margin or z > self.z_max - margin:
            warnings.append("approaching Z boundary")
            
        if warnings:
            return SafetyLevel.WARNING, "; ".join(warnings)
            
        return SafetyLevel.NORMAL, "within safe zone"


@dataclass
class KillSwitch:
    """Emergency kill switch system."""
    armed: bool = True
    triggered: bool = False
    trigger_reason: str = ""
    affected_drones: List[int] = field(default_factory=list)
    
    def trigger(self, reason: str, drone_ids: List[int] = None):
        """Trigger kill switch."""
        if self.armed:
            self.triggered = True
            self.trigger_reason = reason
            self.affected_drones = drone_ids or []
            return True
        return False
    
    def reset(self):
        """Reset kill switch (requires explicit action)."""
        self.triggered = False
        self.trigger_reason = ""
        self.affected_drones = []


@dataclass
class GroundTruthTracker:
    """Simulates VICON/RTK ground truth tracking."""
    update_rate_hz: float = 100.0
    position_accuracy_m: float = 0.001  # 1mm accuracy (VICON-like)
    coverage_area: Tuple[float, float, float, float] = (0, 0, 1000, 1000)
    
    recorded_positions: Dict[int, List[np.ndarray]] = field(default_factory=dict)
    timestamps: List[float] = field(default_factory=list)
    
    def record(self, drone_id: int, true_pos: np.ndarray, time: float):
        """Record ground truth position."""
        if drone_id not in self.recorded_positions:
            self.recorded_positions[drone_id] = []
            
        # Add tiny noise to simulate real measurement
        noise = np.random.normal(0, self.position_accuracy_m, 3)
        self.recorded_positions[drone_id].append(true_pos + noise)
        
        if time not in self.timestamps:
            self.timestamps.append(time)
            
    def get_trajectory(self, drone_id: int) -> np.ndarray:
        """Get recorded trajectory for a drone."""
        if drone_id in self.recorded_positions:
            return np.array(self.recorded_positions[drone_id])
        return np.array([])
    
    def compute_tracking_error(self, drone_id: int, 
                                estimated_positions: List[np.ndarray]) -> Dict:
        """Compute error between estimated and ground truth."""
        true_traj = self.get_trajectory(drone_id)
        
        if len(true_traj) == 0 or len(estimated_positions) == 0:
            return {'error': 'no data'}
            
        # Align lengths
        min_len = min(len(true_traj), len(estimated_positions))
        true_traj = true_traj[:min_len]
        est_traj = np.array(estimated_positions[:min_len])
        
        errors = np.linalg.norm(true_traj - est_traj, axis=1)
        
        return {
            'mean_error': float(np.mean(errors)),
            'max_error': float(np.max(errors)),
            'std_error': float(np.std(errors)),
            'rmse': float(np.sqrt(np.mean(errors**2)))
        }


class SafetyMonitor:
    """Central safety monitoring system."""
    
    def __init__(self, geofence: Geofence, num_drones: int):
        self.geofence = geofence
        self.kill_switch = KillSwitch()
        self.ground_truth = GroundTruthTracker()
        self.num_drones = num_drones
        
        self.safety_log: List[Dict] = []
        self.collision_threshold = 2.0  # meters
        self.near_miss_threshold = 5.0
        
        # Status tracking
        self.drone_status = {i: SafetyLevel.NORMAL for i in range(num_drones)}
        
    def check_collision(self, positions: List[np.ndarray]) -> List[Tuple[int, int, float]]:
        """Check for collisions between drones."""
        collisions = []
        
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist < self.collision_threshold:
                    collisions.append((i, j, dist))
                    
        return collisions
    
    def check_near_misses(self, positions: List[np.ndarray]) -> List[Tuple[int, int, float]]:
        """Check for near-miss events."""
        near_misses = []
        
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                dist = np.linalg.norm(positions[i] - positions[j])
                if self.collision_threshold <= dist < self.near_miss_threshold:
                    near_misses.append((i, j, dist))
                    
        return near_misses
    
    def update(self, positions: List[np.ndarray], time: float) -> Dict:
        """Main safety update loop."""
        events = {
            'geofence_violations': [],
            'collisions': [],
            'near_misses': [],
            'kill_switch_triggered': False
        }
        
        # Check geofence for each drone
        for i, pos in enumerate(positions):
            level, msg = self.geofence.check_position(pos)
            self.drone_status[i] = level
            
            if level.value >= SafetyLevel.CRITICAL.value:
                events['geofence_violations'].append({
                    'drone_id': i,
                    'position': pos.tolist(),
                    'message': msg,
                    'time': time
                })
                
            # Record ground truth
            self.ground_truth.record(i, pos, time)
            
        # Check collisions
        collisions = self.check_collision(positions)
        for d1, d2, dist in collisions:
            events['collisions'].append({
                'drones': [d1, d2],
                'separation': dist,
                'time': time
            })
            
            # Trigger kill switch on collision
            self.kill_switch.trigger(
                f"Collision between drones {d1} and {d2}",
                [d1, d2]
            )
            events['kill_switch_triggered'] = True
            
        # Check near misses
        near_misses = self.check_near_misses(positions)
        for d1, d2, dist in near_misses:
            events['near_misses'].append({
                'drones': [d1, d2],
                'separation': dist,
                'time': time
            })
            
        # Log events
        if events['geofence_violations'] or events['collisions'] or events['near_misses']:
            self.safety_log.append({
                'time': time,
                'events': events
            })
            
        return events
    
    def get_safety_report(self) -> Dict:
        """Generate comprehensive safety report."""
        total_violations = sum(len(e['events']['geofence_violations']) 
                              for e in self.safety_log)
        total_collisions = sum(len(e['events']['collisions']) 
                              for e in self.safety_log)
        total_near_misses = sum(len(e['events']['near_misses']) 
                               for e in self.safety_log)
        
        return {
            'total_geofence_violations': total_violations,
            'total_collisions': total_collisions,
            'total_near_misses': total_near_misses,
            'kill_switch_triggered': self.kill_switch.triggered,
            'kill_switch_reason': self.kill_switch.trigger_reason,
            'event_log': self.safety_log
        }


class LiveOverrideSystem:
    """Live override system for manual intervention."""
    
    def __init__(self):
        self.override_active = False
        self.override_commands: Dict[int, np.ndarray] = {}
        self.override_mode = "none"  # none, hover, rtl (return to launch), land
        
    def activate_hover(self, drone_ids: List[int]):
        """Force specified drones to hover."""
        self.override_active = True
        self.override_mode = "hover"
        for d_id in drone_ids:
            self.override_commands[d_id] = np.zeros(3)  # Zero velocity
            
    def activate_return_to_launch(self, drone_ids: List[int], launch_pos: np.ndarray):
        """Initiate return to launch for specified drones."""
        self.override_active = True
        self.override_mode = "rtl"
        for d_id in drone_ids:
            self.override_commands[d_id] = launch_pos.copy()
            
    def activate_emergency_land(self, drone_ids: List[int]):
        """Initiate emergency landing."""
        self.override_active = True
        self.override_mode = "land"
        for d_id in drone_ids:
            self.override_commands[d_id] = np.array([0, 0, -2.0])  # Descend
            
    def deactivate(self):
        """Deactivate override system."""
        self.override_active = False
        self.override_commands = {}
        self.override_mode = "none"
        
    def get_override_command(self, drone_id: int) -> Optional[np.ndarray]:
        """Get override command for a drone if active."""
        if self.override_active and drone_id in self.override_commands:
            return self.override_commands[drone_id]
        return None


def run_safety_test():
    """Test safety protocols."""
    print("=" * 60)
    print("SAFETY PROTOCOLS TEST")
    print("=" * 60)
    
    # Create safety monitor
    geofence = Geofence(
        x_min=0, x_max=1000,
        y_min=0, y_max=1000,
        z_min=10, z_max=200,
        soft_margin=30
    )
    
    monitor = SafetyMonitor(geofence, num_drones=5)
    override = LiveOverrideSystem()
    
    # Simulate positions
    np.random.seed(42)
    
    print("\nSimulating 100 timesteps...")
    
    for t in range(100):
        # Generate test positions
        positions = []
        for i in range(5):
            pos = np.array([
                100 + t * 5 + np.random.normal(0, 10),
                500 + np.sin(t * 0.1) * 50 + i * 20,
                50 + np.random.normal(0, 5)
            ])
            positions.append(pos)
            
        # Simulate near-collision at t=50
        if t == 50:
            positions[0] = positions[1] + np.array([3, 0, 0])
            
        # Simulate geofence violation at t=80
        if t == 80:
            positions[2] = np.array([990, 500, 50])  # Near boundary
            
        events = monitor.update(positions, t * 0.5)
        
        if events['kill_switch_triggered']:
            print(f"  [t={t}] KILL SWITCH TRIGGERED!")
            override.activate_hover([0, 1, 2, 3, 4])
            
    # Generate report
    report = monitor.get_safety_report()
    
    print("\n" + "=" * 60)
    print("SAFETY REPORT")
    print("=" * 60)
    print(f"Total Geofence Violations: {report['total_geofence_violations']}")
    print(f"Total Collisions: {report['total_collisions']}")
    print(f"Total Near Misses: {report['total_near_misses']}")
    print(f"Kill Switch Triggered: {report['kill_switch_triggered']}")
    if report['kill_switch_triggered']:
        print(f"  Reason: {report['kill_switch_reason']}")
        
    # Compute tracking accuracy
    print("\n--- Ground Truth Tracking Accuracy ---")
    for i in range(5):
        traj = monitor.ground_truth.get_trajectory(i)
        print(f"Drone {i}: {len(traj)} positions recorded")
        
    # Save report
    with open("case_studies/safety_protocols/safety_report.json", "w") as f:
        # Convert numpy arrays for JSON serialization
        safe_report = {
            'total_geofence_violations': report['total_geofence_violations'],
            'total_collisions': report['total_collisions'],
            'total_near_misses': report['total_near_misses'],
            'kill_switch_triggered': report['kill_switch_triggered'],
            'kill_switch_reason': report['kill_switch_reason']
        }
        json.dump(safe_report, f, indent=2)
        
    print("\nSafety test complete!")
    print("Report saved to case_studies/safety_protocols/safety_report.json")


if __name__ == "__main__":
    run_safety_test()
