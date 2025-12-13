"""
Drone Swarm Simulation

Multi-drone swarm with dynamics, sensing, and communication.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class DroneState:
    """State of a single drone."""
    position: np.ndarray  # (3,) x, y, z
    velocity: np.ndarray  # (3,) vx, vy, vz
    goal: np.ndarray      # (3,) target position
    covariance: np.ndarray  # (6, 6) position/velocity uncertainty
    gps_quality: float


class DroneSwarm:
    """
    Multi-drone swarm with dynamics and inter-drone communication.
    
    Each drone maintains:
    - Position and velocity state
    - Localization covariance matrix
    - GPS quality indicator
    - Communication with neighbors within range
    
    Parameters
    ----------
    n_drones : int
        Number of drones in swarm
    env : UrbanEnvironment, optional
        Environment for GPS quality lookup
    comm_range : float
        Communication range in meters
    max_speed : float
        Maximum drone speed in m/s
        
    Example
    -------
    >>> swarm = DroneSwarm(n_drones=5)
    >>> obs = swarm.reset()
    >>> actions = np.random.randn(5, 3) * 0.5
    >>> obs, rewards, dones, info = swarm.step(actions)
    """
    
    def __init__(
        self,
        n_drones: int = 5,
        env = None,
        comm_range: float = 150.0,
        max_speed: float = 10.0,
        dt: float = 0.1
    ):
        self.n_drones = n_drones
        self.env = env
        self.comm_range = comm_range
        self.max_speed = max_speed
        self.dt = dt
        
        self.drones: List[DroneState] = []
        self.step_count = 0
        self.max_steps = 1000
        
    def reset(self, seed: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Reset swarm to initial positions.
        
        Parameters
        ----------
        seed : int, optional
            Random seed
            
        Returns
        -------
        dict
            Observations with keys "states" and "covs"
        """
        if seed is not None:
            np.random.seed(seed)
            
        self.drones = []
        self.step_count = 0
        
        env_size = self.env.size if self.env else 1000
        
        for i in range(self.n_drones):
            # Spawn in different areas
            angle = 2 * np.pi * i / self.n_drones
            radius = env_size * 0.3
            x = env_size / 2 + radius * np.cos(angle)
            y = env_size / 2 + radius * np.sin(angle)
            z = 50.0
            
            # Random goal
            goal = np.array([
                np.random.uniform(100, env_size - 100),
                np.random.uniform(100, env_size - 100),
                50.0
            ])
            
            # Initial covariance
            cov = np.eye(6) * 0.5
            
            # Initial GPS quality
            gps = self.env.get_gps_quality(x, y, z) if self.env else 1.0
            
            self.drones.append(DroneState(
                position=np.array([x, y, z]),
                velocity=np.zeros(3),
                goal=goal,
                covariance=cov,
                gps_quality=gps
            ))
            
        return self._get_observations()
    
    def step(
        self, 
        actions: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, Dict]:
        """
        Step simulation forward.
        
        Parameters
        ----------
        actions : np.ndarray
            Velocity commands for each drone, shape (n_drones, 3)
            Values should be in [-1, 1], scaled by max_speed
            
        Returns
        -------
        observations : dict
            New observations
        rewards : np.ndarray
            Reward for each drone
        dones : np.ndarray
            Whether each drone reached its goal
        info : dict
            Additional information
        """
        actions = np.array(actions)
        assert actions.shape == (self.n_drones, 3)
        
        rewards = np.zeros(self.n_drones)
        dones = np.zeros(self.n_drones, dtype=bool)
        
        for i, (drone, action) in enumerate(zip(self.drones, actions)):
            # Scale action to velocity
            velocity = np.clip(action, -1, 1) * self.max_speed
            
            # Update position
            new_position = drone.position + velocity * self.dt
            
            # Check bounds
            env_size = self.env.size if self.env else 1000
            new_position = np.clip(new_position, [10, 10, 10], [env_size - 10, env_size - 10, 100])
            
            # Check collision
            collision = False
            if self.env:
                collision = self.env.check_collision(*new_position)
                
            if not collision:
                drone.position = new_position
                drone.velocity = velocity
                
            # Update GPS quality
            if self.env:
                drone.gps_quality = self.env.get_gps_quality(*drone.position)
            
            # Update covariance based on GPS quality
            self._update_covariance(drone)
            
            # Compute reward
            dist_to_goal = np.linalg.norm(drone.position - drone.goal)
            rewards[i] = -0.01 * dist_to_goal
            
            if collision:
                rewards[i] -= 10.0
                
            if dist_to_goal < 10.0:
                rewards[i] += 10.0
                dones[i] = True
                
        self.step_count += 1
        if self.step_count >= self.max_steps:
            dones[:] = True
            
        info = {
            "step": self.step_count,
            "completion_rate": np.mean(dones),
            "avg_gps_quality": np.mean([d.gps_quality for d in self.drones])
        }
        
        return self._get_observations(), rewards, dones, info
    
    def _update_covariance(self, drone: DroneState):
        """Update localization covariance based on GPS quality."""
        # Process noise growth
        Q = np.eye(6) * 0.01
        drone.covariance = drone.covariance + Q * self.dt
        
        # Measurement update if GPS available
        if drone.gps_quality > 0.1:
            # Kalman-like update (simplified)
            R_scale = 1.0 / max(drone.gps_quality, 0.01)
            K = drone.covariance[:3, :3] / (drone.covariance[:3, :3] + np.eye(3) * R_scale)
            drone.covariance[:3, :3] = (np.eye(3) - K) @ drone.covariance[:3, :3]
    
    def _get_observations(self) -> Dict[str, np.ndarray]:
        """Get observations for all drones."""
        states = []
        covs = []
        
        for drone in self.drones:
            # Observation: relative goal, velocity, GPS quality, neighbor count
            rel_goal = drone.goal - drone.position
            neighbors = self._count_neighbors(drone)
            
            obs = np.concatenate([
                rel_goal,                    # 3
                drone.velocity,              # 3
                [drone.gps_quality],         # 1
                [neighbors / self.n_drones], # 1
            ])
            
            # Pad to fixed size
            obs = np.pad(obs, (0, max(0, 32 - len(obs))))[:32]
            
            states.append(obs)
            covs.append(drone.covariance)
            
        return {
            "states": np.array(states),
            "covs": np.array(covs)
        }
    
    def _count_neighbors(self, drone: DroneState) -> int:
        """Count drones within communication range."""
        count = 0
        for other in self.drones:
            if other is not drone:
                dist = np.linalg.norm(drone.position - other.position)
                if dist < self.comm_range:
                    count += 1
        return count
    
    def get_positions(self) -> np.ndarray:
        """Get all drone positions."""
        return np.array([d.position for d in self.drones])
    
    def get_covariances(self) -> np.ndarray:
        """Get all drone covariance matrices."""
        return np.array([d.covariance for d in self.drones])
    
    def __repr__(self) -> str:
        return f"DroneSwarm(n_drones={self.n_drones}, comm_range={self.comm_range})"
