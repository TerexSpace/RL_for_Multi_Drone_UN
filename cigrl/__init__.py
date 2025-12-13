"""
CIGRL: Covariance-Integrated Graph Reinforcement Learning

A Python framework for multi-drone urban navigation with
covariance-weighted cooperative sensing.
"""

__version__ = "0.1.0"
__author__ = "Ayan Kemel, Luigi La Spada, Almas Ospanov"

from cigrl.core.attention import CovarianceAttention
from cigrl.core.policy import CIGRLPolicy
from cigrl.core.sensor_fusion import AdaptiveEKF
from cigrl.envs.urban import UrbanEnvironment
from cigrl.envs.drone import DroneSwarm

__all__ = [
    "CovarianceAttention",
    "CIGRLPolicy", 
    "AdaptiveEKF",
    "UrbanEnvironment",
    "DroneSwarm",
]
