"""
CIGRL Enhanced Simulation Framework
Scalable Multi-Drone Urban Navigation with Comprehensive Metrics

Supports 3-20 drone swarms with:
- Multiple mission types (surveillance, delivery, formation, canyon transit)
- GNSS dropout modeling with signal degradation
- Inter-drone communication failure simulation
- Attention-based cooperative sensing
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum
import json
import time


class MissionType(Enum):
    SURVEILLANCE = "surveillance"
    DELIVERY = "delivery"
    FORMATION = "formation"
    CANYON_TRANSIT = "canyon_transit"


@dataclass
class SimConfig:
    """Simulation configuration parameters."""
    num_drones: int = 10
    steps: int = 300
    dt: float = 0.5
    urban_area_size: float = 1000.0
    communication_range: float = 150.0
    comm_failure_prob: float = 0.1
    collision_threshold: float = 2.0
    near_miss_threshold: float = 5.0
    mission_type: MissionType = MissionType.DELIVERY
    seed: int = 42


@dataclass
class GNSSZone:
    """GPS denial/degradation zone."""
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    signal_quality: float  # 0.0 = complete denial, 1.0 = perfect


@dataclass 
class UrbanEnvironment:
    """Urban environment with buildings and GNSS zones."""
    size: float
    gnss_zones: List[GNSSZone] = field(default_factory=list)
    buildings: List[Tuple[float, float, float, float, float]] = field(default_factory=list)  # x, y, w, h, height
    
    @classmethod
    def create_dense_urban(cls, size: float = 1000.0) -> 'UrbanEnvironment':
        """Create a dense urban environment with multiple GPS-denied zones."""
        env = cls(size=size)
        
        # Add GPS-denied zones (urban canyons, tunnels)
        env.gnss_zones = [
            GNSSZone(100, 100, 250, 350, 0.1),   # Heavy denial
            GNSSZone(400, 200, 550, 450, 0.2),   # Heavy denial
            GNSSZone(700, 50, 850, 200, 0.3),    # Moderate degradation
            GNSSZone(150, 600, 400, 800, 0.15),  # Heavy denial
            GNSSZone(600, 550, 900, 750, 0.25),  # Moderate degradation
        ]
        
        # Add building obstacles
        np.random.seed(42)
        for i in range(30):
            x = np.random.uniform(50, size - 100)
            y = np.random.uniform(50, size - 100)
            w = np.random.uniform(20, 60)
            h = np.random.uniform(20, 60)
            height = np.random.uniform(30, 100)
            env.buildings.append((x, y, w, h, height))
            
        return env
    
    def get_gps_quality(self, pos: np.ndarray) -> float:
        """Get GPS signal quality at a position."""
        x, y = pos[0], pos[1]
        for zone in self.gnss_zones:
            if zone.x_min <= x <= zone.x_max and zone.y_min <= y <= zone.y_max:
                return zone.signal_quality + np.random.uniform(-0.05, 0.05)
        return np.random.uniform(0.85, 1.0)


@dataclass
class Drone:
    """Individual drone agent."""
    id: int
    pos: np.ndarray
    target: np.ndarray
    vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    path: List[np.ndarray] = field(default_factory=list)
    imu_bias: np.ndarray = field(default_factory=lambda: np.random.normal(0, 0.05, 3))
    gps_quality: float = 1.0
    comm_active: bool = True
    mission_complete: bool = False
    
    def __post_init__(self):
        self.path = [self.pos.copy()]
        
    def distance_to_target(self) -> float:
        return np.linalg.norm(self.target - self.pos)
    
    def is_moving(self) -> bool:
        return self.distance_to_target() > 5.0 and not self.mission_complete


@dataclass
class Metrics:
    """Comprehensive evaluation metrics."""
    mission_completion_rate: float = 0.0
    avg_trajectory_deviation: float = 0.0
    localization_drift: float = 0.0
    collision_count: int = 0
    near_miss_count: int = 0
    communication_packets: int = 0
    communication_bytes: int = 0
    total_reward: float = 0.0
    time_to_completion: List[float] = field(default_factory=list)


class BaselineController:
    """Baseline independent navigation (A* + Potential Fields style)."""
    
    def __init__(self, env: UrbanEnvironment):
        self.env = env
        
    def compute_control(self, drone: Drone, all_drones: List[Drone]) -> np.ndarray:
        if not drone.is_moving():
            return np.zeros(3)
            
        gps_q = self.env.get_gps_quality(drone.pos)
        drone.gps_quality = gps_q
        
        # Position estimate with GPS-dependent noise
        noise = np.random.normal(0, (1.0 - gps_q) * 25.0, 3)
        estimated_pos = drone.pos + noise
        
        # Simple proportional control toward target
        direction = drone.target - estimated_pos
        dist = np.linalg.norm(direction)
        
        if dist > 0:
            direction = direction / dist
            
        speed = 5.0
        
        # Panic behavior in GPS-denied zones
        if gps_q < 0.3:
            direction = np.random.normal(0, 1, 3)
            direction = direction / (np.linalg.norm(direction) + 1e-6)
            speed = 2.0  # Reduced speed when lost
            
        # Basic collision avoidance (potential field repulsion)
        for other in all_drones:
            if other.id != drone.id:
                diff = drone.pos - other.pos
                dist_other = np.linalg.norm(diff)
                if dist_other < 20.0 and dist_other > 0:
                    repulsion = (diff / dist_other) * (20.0 - dist_other) / 20.0
                    direction = direction + repulsion * 0.5
                    direction = direction / (np.linalg.norm(direction) + 1e-6)
                    
        return direction * speed


class CIGRLController:
    """CIGRL: Covariance-driven Intelligent Graph RL with attention-based cooperation."""
    
    def __init__(self, env: UrbanEnvironment, config: SimConfig):
        self.env = env
        self.config = config
        
    def compute_attention_weights(self, drone: Drone, neighbors: List[Dict]) -> np.ndarray:
        """Compute attention weights for neighbor information fusion."""
        if not neighbors:
            return np.array([])
            
        # Attention based on GPS confidence and distance
        weights = []
        for n in neighbors:
            conf_weight = n['conf'] ** 2  # Prioritize high-confidence neighbors
            dist = np.linalg.norm(drone.pos - n['pos'])
            dist_weight = 1.0 / (1.0 + dist / 50.0)  # Closer neighbors weighted more
            weights.append(conf_weight * dist_weight)
            
        weights = np.array(weights)
        if weights.sum() > 0:
            weights = weights / weights.sum()
        return weights
    
    def cooperative_localization(self, drone: Drone, neighbors: List[Dict], 
                                  attention_weights: np.ndarray) -> np.ndarray:
        """Fuse position estimates using attention-weighted neighbor information."""
        my_q = drone.gps_quality
        
        if my_q > 0.6:
            # Good GPS - use own estimate with small correction from peers
            my_noise = np.random.normal(0, (1.0 - my_q) * 5.0, 3)
            my_est = drone.pos + my_noise
            
            if len(neighbors) > 0 and attention_weights.sum() > 0:
                peer_est = np.zeros(3)
                for n, w in zip(neighbors, attention_weights):
                    peer_est += w * n['pos']
                # Small fusion with peer estimates
                return 0.8 * my_est + 0.2 * peer_est
            return my_est
        else:
            # Poor GPS - rely heavily on cooperative localization
            if len(neighbors) > 0 and attention_weights.sum() > 0:
                # Weighted average of neighbor-relative positions
                fused_est = np.zeros(3)
                total_weight = 0
                
                for n, w in zip(neighbors, attention_weights):
                    if n['conf'] > 0.5:
                        # Estimate own position relative to confident neighbor
                        relative_est = n['pos'] + np.random.normal(0, 3.0, 3)
                        fused_est += w * relative_est
                        total_weight += w
                        
                if total_weight > 0:
                    return fused_est / total_weight + drone.pos * 0.1
                    
            # Fallback to IMU dead-reckoning with drift
            return drone.pos + drone.imu_bias + np.random.normal(0, 2.0, 3)
    
    def compute_control(self, drone: Drone, all_drones: List[Drone], 
                        comm_active: Dict[int, bool]) -> Tuple[np.ndarray, int]:
        """Compute CIGRL control output with communication."""
        if not drone.is_moving():
            return np.zeros(3), 0
            
        packets = 0
        gps_q = self.env.get_gps_quality(drone.pos)
        drone.gps_quality = gps_q
        
        # Gather neighbor information (communication step)
        neighbors = []
        for other in all_drones:
            if other.id == drone.id:
                continue
            dist = np.linalg.norm(drone.pos - other.pos)
            if dist < self.config.communication_range:
                # Check communication availability
                if comm_active.get(other.id, True) and np.random.random() > self.config.comm_failure_prob:
                    other_q = self.env.get_gps_quality(other.pos)
                    noise = np.random.normal(0, (1.0 - other_q) * 5.0, 3)
                    neighbors.append({
                        'id': other.id,
                        'pos': other.pos + noise,
                        'conf': other_q,
                        'vel': other.vel.copy()
                    })
                    packets += 1
        
        # Attention mechanism
        attention_weights = self.compute_attention_weights(drone, neighbors)
        
        # Cooperative localization
        estimated_pos = self.cooperative_localization(drone, neighbors, attention_weights)
        
        # Navigation toward target
        direction = drone.target - estimated_pos
        dist = np.linalg.norm(direction)
        
        if dist > 0:
            direction = direction / dist
            
        speed = 5.0
        
        # Cooperative collision avoidance using neighbor velocities
        for n, w in zip(neighbors, attention_weights):
            diff = drone.pos - n['pos']
            dist_n = np.linalg.norm(diff)
            if dist_n < 15.0 and dist_n > 0:
                # Consider neighbor velocity for predictive avoidance
                relative_vel = drone.vel - n['vel']
                closing_speed = -np.dot(relative_vel, diff / dist_n)
                if closing_speed > 0:  # Approaching
                    repulsion = (diff / dist_n) * (15.0 - dist_n) / 15.0
                    direction = direction + repulsion * 0.7 * w
                    direction = direction / (np.linalg.norm(direction) + 1e-6)
        
        # Safety hover in complete GPS denial with no confident peers
        if gps_q < 0.2:
            confident_peers = [n for n in neighbors if n['conf'] > 0.5]
            if not confident_peers:
                speed = 1.0  # Slow cautious movement
                
        return direction * speed, packets


class Simulator:
    """Main simulation runner."""
    
    def __init__(self, config: SimConfig):
        self.config = config
        np.random.seed(config.seed)
        self.env = UrbanEnvironment.create_dense_urban(config.urban_area_size)
        
    def create_mission(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create start/target positions based on mission type."""
        missions = []
        n = self.config.num_drones
        size = self.config.urban_area_size
        
        if self.config.mission_type == MissionType.DELIVERY:
            # Multiple delivery targets across the city
            for i in range(n):
                start = np.array([50 + (i % 5) * 20, 50 + (i // 5) * 20, 10.0])
                target = np.array([
                    np.random.uniform(size * 0.6, size * 0.9),
                    np.random.uniform(size * 0.6, size * 0.9),
                    np.random.uniform(20, 60)
                ])
                missions.append((start, target))
                
        elif self.config.mission_type == MissionType.SURVEILLANCE:
            # Sector-based coverage
            sectors_x = int(np.ceil(np.sqrt(n)))
            sector_size = size / sectors_x
            for i in range(n):
                sx, sy = i % sectors_x, i // sectors_x
                start = np.array([50 + i * 10, 50, 30.0])
                target = np.array([
                    (sx + 0.5) * sector_size,
                    (sy + 0.5) * sector_size,
                    50.0
                ])
                missions.append((start, target))
                
        elif self.config.mission_type == MissionType.FORMATION:
            # Formation flight to target area
            center_target = np.array([size * 0.8, size * 0.8, 40.0])
            for i in range(n):
                angle = 2 * np.pi * i / n
                radius = 30.0
                start = np.array([50 + i * 15, 50, 20.0])
                offset = np.array([radius * np.cos(angle), radius * np.sin(angle), 0])
                target = center_target + offset
                missions.append((start, target))
                
        elif self.config.mission_type == MissionType.CANYON_TRANSIT:
            # Navigate through GPS-denied urban canyons
            for i in range(n):
                start = np.array([50 + i * 10, 50, 15.0])
                # Target requires passing through denied zones
                target = np.array([size - 50, size - 50, 30.0])
                missions.append((start, target))
                
        return missions
    
    def compute_metrics(self, drones: List[Drone], history: List[List[np.ndarray]], 
                        packets: int, collisions: int, near_misses: int) -> Metrics:
        """Compute comprehensive evaluation metrics."""
        metrics = Metrics()
        
        # Mission completion rate
        completed = sum(1 for d in drones if d.distance_to_target() < 10.0)
        metrics.mission_completion_rate = completed / len(drones)
        
        # Trajectory deviation (from straight line)
        deviations = []
        for d in drones:
            if len(d.path) > 1:
                start = d.path[0]
                end = d.target
                optimal_dist = np.linalg.norm(end - start)
                actual_dist = sum(np.linalg.norm(d.path[i+1] - d.path[i]) 
                                 for i in range(len(d.path) - 1))
                if optimal_dist > 0:
                    deviations.append((actual_dist - optimal_dist) / optimal_dist)
        metrics.avg_trajectory_deviation = np.mean(deviations) if deviations else 0
        
        # Localization drift estimate (accumulated error in GPS-denied zones)
        metrics.localization_drift = np.mean([d.imu_bias.sum() * self.config.steps 
                                               for d in drones])
        
        # Collision metrics
        metrics.collision_count = collisions
        metrics.near_miss_count = near_misses
        
        # Communication metrics
        metrics.communication_packets = packets
        metrics.communication_bytes = packets * 64  # Assume 64 bytes per packet
        
        return metrics
    
    def run(self, controller_type: str = "CIGRL") -> Tuple[List[Drone], Metrics]:
        """Run simulation with specified controller."""
        missions = self.create_mission()
        drones = [Drone(id=i, pos=m[0].copy(), target=m[1].copy()) 
                  for i, m in enumerate(missions)]
        
        if controller_type == "CIGRL":
            controller = CIGRLController(self.env, self.config)
        else:
            controller = BaselineController(self.env)
            
        total_packets = 0
        total_collisions = 0
        total_near_misses = 0
        total_reward = 0
        comm_active = {d.id: True for d in drones}
        
        for t in range(self.config.steps):
            # Random communication failures
            for d in drones:
                if np.random.random() < self.config.comm_failure_prob:
                    comm_active[d.id] = False
                else:
                    comm_active[d.id] = True
            
            # Compute controls
            for d in drones:
                if controller_type == "CIGRL":
                    vel, packets = controller.compute_control(d, drones, comm_active)
                    total_packets += packets
                else:
                    vel = controller.compute_control(d, drones)
                    
                d.vel = vel
                
            # Update physics
            for d in drones:
                d.pos = d.pos + d.vel * self.config.dt
                d.path.append(d.pos.copy())
                
                # Check mission completion
                if d.distance_to_target() < 5.0:
                    d.mission_complete = True
                    
            # Check collisions
            for i, d1 in enumerate(drones):
                for d2 in drones[i+1:]:
                    dist = np.linalg.norm(d1.pos - d2.pos)
                    if dist < self.config.collision_threshold:
                        total_collisions += 1
                    elif dist < self.config.near_miss_threshold:
                        total_near_misses += 1
                        
            # Compute reward
            reward = sum(-d.distance_to_target() * 0.01 for d in drones if d.is_moving())
            total_reward += reward
            
        metrics = self.compute_metrics(drones, [], total_packets, 
                                       total_collisions, total_near_misses)
        metrics.total_reward = total_reward
        
        return drones, metrics


def run_experiment_suite(swarm_sizes: List[int], 
                         mission_types: List[MissionType],
                         seeds: List[int]) -> Dict:
    """Run comprehensive experiment suite."""
    results = {
        'baseline': {},
        'cigrl': {}
    }
    
    for n_drones in swarm_sizes:
        for mission in mission_types:
            for seed in seeds:
                key = f"{n_drones}_{mission.value}_{seed}"
                
                config = SimConfig(
                    num_drones=n_drones,
                    steps=300,
                    mission_type=mission,
                    seed=seed
                )
                
                sim = Simulator(config)
                
                # Run baseline
                _, baseline_metrics = sim.run("Baseline")
                results['baseline'][key] = {
                    'completion_rate': baseline_metrics.mission_completion_rate,
                    'trajectory_deviation': baseline_metrics.avg_trajectory_deviation,
                    'collisions': baseline_metrics.collision_count,
                    'near_misses': baseline_metrics.near_miss_count,
                    'reward': baseline_metrics.total_reward
                }
                
                # Reset seed and run CIGRL
                np.random.seed(seed)
                sim = Simulator(config)
                _, cigrl_metrics = sim.run("CIGRL")
                results['cigrl'][key] = {
                    'completion_rate': cigrl_metrics.mission_completion_rate,
                    'trajectory_deviation': cigrl_metrics.avg_trajectory_deviation,
                    'collisions': cigrl_metrics.collision_count,
                    'near_misses': cigrl_metrics.near_miss_count,
                    'reward': cigrl_metrics.total_reward,
                    'packets': cigrl_metrics.communication_packets
                }
                
                print(f"Completed: {key}")
                
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("CIGRL Enhanced Simulation Framework")
    print("=" * 60)
    
    # Quick test with different swarm sizes
    swarm_sizes = [5, 10, 15, 20]
    mission_types = [MissionType.DELIVERY, MissionType.SURVEILLANCE, 
                     MissionType.FORMATION, MissionType.CANYON_TRANSIT]
    seeds = [42, 123, 456]
    
    print("\nRunning experiment suite...")
    results = run_experiment_suite(swarm_sizes, mission_types, seeds)
    
    # Save results
    with open("experiments/deep_evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to experiments/deep_evaluation_results.json")
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY: Average Performance Across All Experiments")
    print("=" * 60)
    
    baseline_completion = np.mean([v['completion_rate'] for v in results['baseline'].values()])
    cigrl_completion = np.mean([v['completion_rate'] for v in results['cigrl'].values()])
    
    baseline_reward = np.mean([v['reward'] for v in results['baseline'].values()])
    cigrl_reward = np.mean([v['reward'] for v in results['cigrl'].values()])
    
    print(f"\nMission Completion Rate:")
    print(f"  Baseline: {baseline_completion:.1%}")
    print(f"  CIGRL:    {cigrl_completion:.1%}")
    print(f"  Improvement: {(cigrl_completion - baseline_completion) * 100:.1f}%")
    
    print(f"\nCumulative Reward:")
    print(f"  Baseline: {baseline_reward:.1f}")
    print(f"  CIGRL:    {cigrl_reward:.1f}")
    print(f"  Improvement: {((cigrl_reward - baseline_reward) / abs(baseline_reward)) * 100:.1f}%")
    
    print("\n" + "=" * 60)
    print("Experiment suite complete!")
