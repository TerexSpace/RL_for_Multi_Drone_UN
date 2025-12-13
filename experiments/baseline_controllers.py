"""
Baseline Controllers for Comparison
Implements ORCA, A* with Potential Fields, and ablated MAPPO variants.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import heapq


@dataclass
class ORCAParams:
    """ORCA algorithm parameters."""
    tau: float = 5.0  # Time horizon
    max_speed: float = 5.0
    neighbor_dist: float = 50.0
    max_neighbors: int = 10


class ORCAController:
    """
    Optimal Reciprocal Collision Avoidance (ORCA)
    Based on van den Berg et al., 2011
    """
    
    def __init__(self, params: ORCAParams = None):
        self.params = params or ORCAParams()
        
    def compute_orca_velocity(self, drone_pos: np.ndarray, drone_vel: np.ndarray,
                               drone_target: np.ndarray, neighbors: List[Dict]) -> np.ndarray:
        """Compute ORCA-compliant velocity."""
        # Preferred velocity toward target
        to_target = drone_target - drone_pos
        dist_to_target = np.linalg.norm(to_target)
        
        if dist_to_target > 0:
            pref_vel = (to_target / dist_to_target) * min(self.params.max_speed, dist_to_target)
        else:
            pref_vel = np.zeros(3)
            
        if not neighbors:
            return pref_vel
            
        # Compute ORCA constraints from each neighbor
        orca_planes = []
        for n in neighbors:
            rel_pos = n['pos'] - drone_pos
            rel_vel = drone_vel - n['vel']
            dist = np.linalg.norm(rel_pos[:2])  # 2D distance
            
            combined_radius = 4.0  # Combined safety radius
            
            if dist < combined_radius:
                # Already in collision - compute escape velocity
                if dist > 0:
                    normal = -rel_pos[:2] / dist
                else:
                    normal = np.array([1.0, 0.0])
                orca_planes.append({'normal': normal, 'point': normal * combined_radius})
            else:
                # Compute velocity obstacle
                leg_dist = np.sqrt(dist**2 - combined_radius**2)
                
                # Left and right legs of velocity obstacle
                if dist > 0:
                    left_leg = np.array([
                        rel_pos[0] * leg_dist - rel_pos[1] * combined_radius,
                        rel_pos[0] * combined_radius + rel_pos[1] * leg_dist
                    ]) / (dist**2)
                    
                    right_leg = np.array([
                        rel_pos[0] * leg_dist + rel_pos[1] * combined_radius,
                        -rel_pos[0] * combined_radius + rel_pos[1] * leg_dist
                    ]) / (dist**2)
                    
                    # Determine which side of VO the relative velocity is on
                    if np.cross(rel_pos[:2], rel_vel[:2]) > 0:
                        normal = np.array([-left_leg[1], left_leg[0]])
                    else:
                        normal = np.array([right_leg[1], -right_leg[0]])
                        
                    if np.linalg.norm(normal) > 0:
                        normal = normal / np.linalg.norm(normal)
                        orca_planes.append({'normal': normal, 'point': rel_vel[:2] + normal * 0.5})
                        
        # Linear programming to find best velocity
        new_vel = self._solve_linear_program(pref_vel[:2], orca_planes)
        
        return np.array([new_vel[0], new_vel[1], pref_vel[2] if len(pref_vel) > 2 else 0])
    
    def _solve_linear_program(self, pref_vel: np.ndarray, 
                               planes: List[Dict]) -> np.ndarray:
        """Simplified LP solver for ORCA velocity."""
        result = pref_vel.copy()
        
        for plane in planes:
            normal = plane['normal']
            point = plane.get('point', np.zeros(2))
            
            # Project velocity onto valid half-plane
            dot = np.dot(result - point, normal)
            if dot < 0:
                result = result - dot * normal
                
        # Clamp to max speed
        speed = np.linalg.norm(result)
        if speed > self.params.max_speed:
            result = result / speed * self.params.max_speed
            
        return result


class AStarPotentialField:
    """
    A* Path Planning with Potential Field local adjustments.
    """
    
    def __init__(self, grid_size: float = 1000.0, resolution: float = 20.0):
        self.grid_size = grid_size
        self.resolution = resolution
        self.grid_dim = int(grid_size / resolution)
        self.obstacles = set()
        
    def world_to_grid(self, pos: np.ndarray) -> Tuple[int, int]:
        """Convert world coordinates to grid indices."""
        gx = int(np.clip(pos[0] / self.resolution, 0, self.grid_dim - 1))
        gy = int(np.clip(pos[1] / self.resolution, 0, self.grid_dim - 1))
        return (gx, gy)
    
    def grid_to_world(self, grid_pos: Tuple[int, int]) -> np.ndarray:
        """Convert grid indices to world coordinates."""
        return np.array([
            grid_pos[0] * self.resolution + self.resolution / 2,
            grid_pos[1] * self.resolution + self.resolution / 2,
            30.0  # Default altitude
        ])
    
    def add_obstacle(self, pos: np.ndarray, radius: float = 20.0):
        """Add obstacle to grid."""
        center = self.world_to_grid(pos)
        grid_radius = int(radius / self.resolution) + 1
        
        for dx in range(-grid_radius, grid_radius + 1):
            for dy in range(-grid_radius, grid_radius + 1):
                if dx*dx + dy*dy <= grid_radius*grid_radius:
                    gx, gy = center[0] + dx, center[1] + dy
                    if 0 <= gx < self.grid_dim and 0 <= gy < self.grid_dim:
                        self.obstacles.add((gx, gy))
                        
    def heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Euclidean distance heuristic."""
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    
    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get valid neighboring cells."""
        neighbors = []
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
            nx, ny = pos[0] + dx, pos[1] + dy
            if 0 <= nx < self.grid_dim and 0 <= ny < self.grid_dim:
                if (nx, ny) not in self.obstacles:
                    neighbors.append((nx, ny))
        return neighbors
    
    def plan_path(self, start: np.ndarray, goal: np.ndarray) -> List[np.ndarray]:
        """A* path planning."""
        start_grid = self.world_to_grid(start)
        goal_grid = self.world_to_grid(goal)
        
        if start_grid in self.obstacles or goal_grid in self.obstacles:
            return [goal]  # Fallback
            
        # A* algorithm
        open_set = [(0, start_grid)]
        came_from = {}
        g_score = {start_grid: 0}
        f_score = {start_grid: self.heuristic(start_grid, goal_grid)}
        
        while open_set:
            _, current = heapq.heappop(open_set)
            
            if current == goal_grid:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(self.grid_to_world(current))
                    current = came_from[current]
                path.reverse()
                return path if path else [goal]
                
            for neighbor in self.get_neighbors(current):
                tentative_g = g_score[current] + self.heuristic(current, neighbor)
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal_grid)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
                    
        return [goal]  # No path found
    
    def compute_potential_field(self, pos: np.ndarray, target: np.ndarray,
                                 obstacles: List[np.ndarray]) -> np.ndarray:
        """Compute potential field gradient."""
        # Attractive potential toward target
        to_target = target - pos
        dist_target = np.linalg.norm(to_target)
        
        if dist_target > 0:
            attract = to_target / dist_target
        else:
            attract = np.zeros(3)
            
        # Repulsive potential from obstacles
        repel = np.zeros(3)
        influence_dist = 30.0
        
        for obs in obstacles:
            to_obs = pos - obs
            dist_obs = np.linalg.norm(to_obs)
            
            if 0 < dist_obs < influence_dist:
                strength = (1.0 / dist_obs - 1.0 / influence_dist) / dist_obs
                repel += (to_obs / dist_obs) * strength
                
        # Combine
        gradient = attract + 0.5 * repel
        norm = np.linalg.norm(gradient)
        if norm > 0:
            gradient = gradient / norm
            
        return gradient * 5.0  # Max speed


class MAPPOAblation:
    """
    MAPPO variants for ablation study.
    Simulates learned policy behavior with/without attention and communication.
    """
    
    def __init__(self, use_attention: bool = True, use_communication: bool = True):
        self.use_attention = use_attention
        self.use_communication = use_communication
        
        # Simulated policy parameters (would be learned in full implementation)
        self.hidden_dim = 64
        np.random.seed(42)
        self.policy_weights = np.random.randn(6, 3) * 0.1  # Simple linear policy
        
    def compute_observation(self, drone_pos: np.ndarray, drone_vel: np.ndarray,
                            target: np.ndarray, neighbors: List[Dict]) -> np.ndarray:
        """Construct observation vector."""
        # Self state
        to_target = target - drone_pos
        dist = np.linalg.norm(to_target)
        if dist > 0:
            to_target_norm = to_target / dist
        else:
            to_target_norm = np.zeros(3)
            
        obs = np.concatenate([to_target_norm, [dist / 500.0], drone_vel / 5.0])
        
        if self.use_communication and neighbors:
            # Add neighbor information
            neighbor_features = []
            for n in neighbors[:3]:  # Top 3 neighbors
                rel_pos = n['pos'] - drone_pos
                neighbor_features.extend(rel_pos / 100.0)
                neighbor_features.append(n.get('conf', 1.0))
            while len(neighbor_features) < 12:
                neighbor_features.append(0.0)
            obs = np.concatenate([obs, neighbor_features])
            
        return obs[:6]  # Match policy input dim
    
    def compute_attention(self, drone_obs: np.ndarray, 
                          neighbor_obs: List[np.ndarray]) -> np.ndarray:
        """Simplified attention mechanism."""
        if not self.use_attention or not neighbor_obs:
            return np.ones(len(neighbor_obs)) / max(len(neighbor_obs), 1)
            
        # Query-key attention (simplified)
        query = drone_obs[:3]
        weights = []
        
        for n_obs in neighbor_obs:
            key = n_obs[:3]
            score = np.dot(query, key) / (np.linalg.norm(query) * np.linalg.norm(key) + 1e-6)
            weights.append(np.exp(score))
            
        weights = np.array(weights)
        return weights / (weights.sum() + 1e-6)
    
    def forward(self, obs: np.ndarray) -> np.ndarray:
        """Forward pass through policy network (simplified)."""
        action = np.dot(obs, self.policy_weights)
        
        # Normalize to max speed
        speed = np.linalg.norm(action)
        if speed > 5.0:
            action = action / speed * 5.0
            
        return action
    
    def compute_control(self, drone_pos: np.ndarray, drone_vel: np.ndarray,
                        target: np.ndarray, neighbors: List[Dict],
                        gps_quality: float) -> np.ndarray:
        """Compute control action using MAPPO variant."""
        obs = self.compute_observation(drone_pos, drone_vel, target, neighbors)
        
        if self.use_communication and neighbors:
            neighbor_obs = [
                self.compute_observation(n['pos'], n['vel'], target, [])
                for n in neighbors
            ]
            attention = self.compute_attention(obs, neighbor_obs)
            
            if self.use_attention:
                # Weighted aggregation
                weighted_obs = np.zeros_like(obs)
                for n_obs, w in zip(neighbor_obs, attention):
                    weighted_obs += w * n_obs
                obs = 0.7 * obs + 0.3 * weighted_obs
                
        action = self.forward(obs)
        
        # GPS quality affects confidence
        if gps_quality < 0.3 and not self.use_communication:
            # Without communication, panics in GPS denial
            action = action * gps_quality * 2
            
        return action


# Factory function for baseline controllers
def create_baseline(baseline_type: str, **kwargs):
    """Create baseline controller by type."""
    if baseline_type == "orca":
        return ORCAController(ORCAParams(**kwargs))
    elif baseline_type == "astar_potential":
        return AStarPotentialField(**kwargs)
    elif baseline_type == "mappo_no_attention":
        return MAPPOAblation(use_attention=False, use_communication=True)
    elif baseline_type == "mappo_no_comm":
        return MAPPOAblation(use_attention=True, use_communication=False)
    elif baseline_type == "mappo_basic":
        return MAPPOAblation(use_attention=False, use_communication=False)
    else:
        raise ValueError(f"Unknown baseline type: {baseline_type}")


if __name__ == "__main__":
    print("Testing baseline controllers...")
    
    # Test ORCA
    orca = ORCAController()
    pos = np.array([100.0, 100.0, 30.0])
    vel = np.array([1.0, 0.0, 0.0])
    target = np.array([500.0, 500.0, 30.0])
    neighbors = [
        {'pos': np.array([110.0, 100.0, 30.0]), 'vel': np.array([-1.0, 0.0, 0.0])}
    ]
    
    orca_vel = orca.compute_orca_velocity(pos, vel, target, neighbors)
    print(f"ORCA velocity: {orca_vel}")
    
    # Test A* + Potential Field
    astar = AStarPotentialField()
    astar.add_obstacle(np.array([300.0, 300.0, 30.0]), radius=50.0)
    path = astar.plan_path(pos, target)
    print(f"A* path length: {len(path)}")
    
    # Test MAPPO ablations
    for ablation in ["mappo_no_attention", "mappo_no_comm", "mappo_basic"]:
        controller = create_baseline(ablation)
        action = controller.compute_control(pos, vel, target, neighbors, 0.5)
        print(f"{ablation} action: {action}")
        
    print("All baseline tests passed!")
