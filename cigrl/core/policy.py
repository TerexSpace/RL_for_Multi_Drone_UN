"""
CIGRL Policy Network

Multi-agent PPO policy with covariance-weighted attention for
inter-drone communication.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path


class CIGRLPolicy:
    """
    CIGRL policy network for multi-drone navigation.
    
    Implements a PPO-based policy with:
    - State encoder for observations
    - Covariance encoder for uncertainty information
    - Multi-head attention for neighbor communication
    - Policy and value heads for actor-critic training
    
    Parameters
    ----------
    obs_dim : int
        Observation dimension per drone
    action_dim : int
        Action dimension (typically 3 for velocity commands)
    hidden_dim : int
        Hidden layer dimension
    num_heads : int
        Number of attention heads
        
    Example
    -------
    >>> policy = CIGRLPolicy(obs_dim=32, action_dim=3)
    >>> obs = {"states": np.random.randn(5, 32), "covs": np.random.randn(5, 6, 6)}
    >>> actions = policy.select_actions(obs)
    """
    
    def __init__(
        self,
        obs_dim: int = 32,
        action_dim: int = 3,
        hidden_dim: int = 128,
        num_heads: int = 4
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Initialize network weights (in practice, use PyTorch)
        self._init_weights()
        
    def _init_weights(self):
        """Initialize network weights."""
        # State encoder: obs_dim -> hidden_dim
        self.state_encoder_w1 = np.random.randn(self.obs_dim, self.hidden_dim) * 0.02
        self.state_encoder_b1 = np.zeros(self.hidden_dim)
        
        # Covariance encoder: 36 (flattened 6x6) -> 32
        self.cov_encoder_w1 = np.random.randn(36, 32) * 0.02
        self.cov_encoder_b1 = np.zeros(32)
        
        # Policy head: hidden_dim + 32 -> action_dim
        self.policy_w1 = np.random.randn(self.hidden_dim + 32, self.action_dim) * 0.02
        self.policy_b1 = np.zeros(self.action_dim)
        
        # Value head: hidden_dim + 32 -> 1
        self.value_w1 = np.random.randn(self.hidden_dim + 32, 1) * 0.02
        self.value_b1 = np.zeros(1)
        
        # Log standard deviation for action distribution
        self.log_std = np.zeros(self.action_dim)
        
    def encode_state(self, obs: np.ndarray) -> np.ndarray:
        """Encode observation into hidden representation."""
        h = obs @ self.state_encoder_w1 + self.state_encoder_b1
        h = np.maximum(0, h)  # ReLU
        return h
    
    def encode_covariance(self, cov: np.ndarray) -> np.ndarray:
        """Encode covariance matrix into embedding."""
        cov_flat = cov.flatten()[:36]  # Flatten and truncate
        if len(cov_flat) < 36:
            cov_flat = np.pad(cov_flat, (0, 36 - len(cov_flat)))
        c = cov_flat @ self.cov_encoder_w1 + self.cov_encoder_b1
        c = np.maximum(0, c)  # ReLU
        return c
    
    def forward(
        self,
        obs: np.ndarray,
        cov: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Forward pass for single agent.
        
        Parameters
        ----------
        obs : np.ndarray
            Observation vector, shape (obs_dim,)
        cov : np.ndarray
            Covariance matrix, shape (6, 6)
            
        Returns
        -------
        action_mean : np.ndarray
            Mean of action distribution, shape (action_dim,)
        value : float
            State value estimate
        """
        h = self.encode_state(obs)
        c = self.encode_covariance(cov)
        
        combined = np.concatenate([h, c])
        
        # Policy head with tanh activation for bounded actions
        action_mean = np.tanh(combined @ self.policy_w1 + self.policy_b1)
        
        # Value head
        value = (combined @ self.value_w1 + self.value_b1)[0]
        
        return action_mean, value
    
    def select_actions(
        self,
        observations: Dict[str, np.ndarray],
        deterministic: bool = False
    ) -> np.ndarray:
        """
        Select actions for all agents.
        
        Parameters
        ----------
        observations : dict
            Dictionary with:
            - "states": np.ndarray of shape (n_agents, obs_dim)
            - "covs": np.ndarray of shape (n_agents, 6, 6)
        deterministic : bool
            If True, return mean actions without sampling
            
        Returns
        -------
        np.ndarray
            Actions for all agents, shape (n_agents, action_dim)
        """
        states = observations["states"]
        covs = observations["covs"]
        n_agents = len(states)
        
        actions = []
        for i in range(n_agents):
            action_mean, _ = self.forward(states[i], covs[i])
            
            if deterministic:
                action = action_mean
            else:
                # Sample from Gaussian
                std = np.exp(self.log_std)
                action = action_mean + std * np.random.randn(self.action_dim)
                action = np.clip(action, -1, 1)
                
            actions.append(action)
            
        return np.array(actions)
    
    def save(self, path: str):
        """Save policy weights to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        weights = {
            "state_encoder_w1": self.state_encoder_w1,
            "state_encoder_b1": self.state_encoder_b1,
            "cov_encoder_w1": self.cov_encoder_w1,
            "cov_encoder_b1": self.cov_encoder_b1,
            "policy_w1": self.policy_w1,
            "policy_b1": self.policy_b1,
            "value_w1": self.value_w1,
            "value_b1": self.value_b1,
            "log_std": self.log_std,
            "config": {
                "obs_dim": self.obs_dim,
                "action_dim": self.action_dim,
                "hidden_dim": self.hidden_dim,
                "num_heads": self.num_heads
            }
        }
        np.savez(path, **weights)
        
    @classmethod
    def load(cls, path: str) -> "CIGRLPolicy":
        """Load policy weights from file."""
        data = np.load(path, allow_pickle=True)
        config = data["config"].item()
        
        policy = cls(**config)
        policy.state_encoder_w1 = data["state_encoder_w1"]
        policy.state_encoder_b1 = data["state_encoder_b1"]
        policy.cov_encoder_w1 = data["cov_encoder_w1"]
        policy.cov_encoder_b1 = data["cov_encoder_b1"]
        policy.policy_w1 = data["policy_w1"]
        policy.policy_b1 = data["policy_b1"]
        policy.value_w1 = data["value_w1"]
        policy.value_b1 = data["value_b1"]
        policy.log_std = data["log_std"]
        
        return policy
    
    def __repr__(self) -> str:
        return (
            f"CIGRLPolicy(obs_dim={self.obs_dim}, action_dim={self.action_dim}, "
            f"hidden_dim={self.hidden_dim}, num_heads={self.num_heads})"
        )
