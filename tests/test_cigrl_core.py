"""
CIGRL Core Tests

Integration tests that import and test actual CIGRL package code.
Tests cover: CovarianceAttention, AdaptiveEKF, CIGRLPolicy, UrbanEnvironment, DroneSwarm.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add cigrl package to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestCovarianceAttention:
    """Tests for covariance-weighted attention mechanism."""

    def test_import(self):
        """Test that CovarianceAttention can be imported."""
        from cigrl.core.attention import CovarianceAttention
        attn = CovarianceAttention(embed_dim=128, num_heads=4)
        assert attn.embed_dim == 128
        assert attn.num_heads == 4

    def test_attention_weights_sum_to_one(self):
        """Test that attention weights are normalized."""
        from cigrl.core.attention import CovarianceAttention
        
        attn = CovarianceAttention(embed_dim=64, lambda_cov=0.1)
        h_self = np.random.randn(64)
        neighbors = [
            (np.random.randn(64), np.eye(6) * 0.5),
            (np.random.randn(64), np.eye(6) * 1.0),
            (np.random.randn(64), np.eye(6) * 2.0),
        ]
        
        _, weights = attn.forward(h_self, neighbors)
        
        assert np.isclose(np.sum(weights), 1.0, atol=1e-5)

    def test_covariance_penalty_reduces_uncertain_weight(self):
        """Test that high covariance trace reduces attention weight."""
        from cigrl.core.attention import CovarianceAttention
        
        attn = CovarianceAttention(embed_dim=64, lambda_cov=0.5)
        h_self = np.random.randn(64)
        
        # Same embedding, different covariances
        emb = np.random.randn(64)
        cov_low = np.eye(6) * 0.1  # Low uncertainty
        cov_high = np.eye(6) * 5.0  # High uncertainty
        
        neighbors = [(emb, cov_low), (emb.copy(), cov_high)]
        
        _, weights = attn.forward(h_self, neighbors)
        
        # Lower covariance should get higher weight
        assert weights[0] > weights[1], "Low uncertainty neighbor should have higher weight"

    def test_empty_neighbors(self):
        """Test attention with no neighbors."""
        from cigrl.core.attention import CovarianceAttention
        
        attn = CovarianceAttention(embed_dim=64)
        h_self = np.random.randn(64)
        
        h_fused, weights = attn.forward(h_self, [])
        
        np.testing.assert_array_equal(h_fused, h_self)
        assert len(weights) == 0


class TestAdaptiveEKF:
    """Tests for Adaptive Extended Kalman Filter."""

    def test_import(self):
        """Test that AdaptiveEKF can be imported."""
        from cigrl.core.sensor_fusion import AdaptiveEKF
        ekf = AdaptiveEKF(gps_noise_nominal=1.0)
        assert ekf.gps_noise_nominal == 1.0

    def test_initialize(self):
        """Test EKF initialization."""
        from cigrl.core.sensor_fusion import AdaptiveEKF
        
        ekf = AdaptiveEKF()
        state = ekf.initialize(position=[10, 20, 30])
        
        np.testing.assert_array_equal(state.position, [10, 20, 30])
        np.testing.assert_array_equal(state.velocity, [0, 0, 0])
        assert state.covariance.shape == (15, 15)

    def test_predict_increases_covariance(self):
        """Test that prediction increases uncertainty."""
        from cigrl.core.sensor_fusion import AdaptiveEKF
        
        ekf = AdaptiveEKF()
        state = ekf.initialize()
        initial_trace = np.trace(state.covariance)
        
        state = ekf.predict(state, accel=[0, 0, -9.81], gyro=[0, 0, 0], dt=0.1)
        final_trace = np.trace(state.covariance)
        
        assert final_trace > initial_trace, "Covariance should increase during prediction"

    def test_update_decreases_covariance(self):
        """Test that GPS update decreases uncertainty."""
        from cigrl.core.sensor_fusion import AdaptiveEKF
        
        ekf = AdaptiveEKF()
        state = ekf.initialize()
        
        # Predict to increase uncertainty
        for _ in range(10):
            state = ekf.predict(state, accel=[0, 0, -9.81], gyro=[0, 0, 0], dt=0.1)
            
        trace_before = ekf.get_position_uncertainty(state)
        
        # GPS update
        state, accepted = ekf.update_gps(state, gps_position=[0.1, 0.1, 0.1], gps_quality=1.0)
        trace_after = ekf.get_position_uncertainty(state)
        
        assert accepted, "GPS update should be accepted"
        assert trace_after < trace_before, "GPS update should decrease uncertainty"

    def test_outlier_rejection(self):
        """Test that large innovations are rejected."""
        from cigrl.core.sensor_fusion import AdaptiveEKF
        
        ekf = AdaptiveEKF(nis_threshold=7.815)
        state = ekf.initialize(position=[0, 0, 0])
        
        # Very large innovation (outlier)
        state, accepted = ekf.update_gps(state, gps_position=[1000, 1000, 1000])
        
        assert not accepted, "Large innovation should be rejected as outlier"


class TestCIGRLPolicy:
    """Tests for CIGRL policy network."""

    def test_import(self):
        """Test that CIGRLPolicy can be imported."""
        from cigrl.core.policy import CIGRLPolicy
        policy = CIGRLPolicy(obs_dim=32, action_dim=3)
        assert policy.obs_dim == 32
        assert policy.action_dim == 3

    def test_forward(self):
        """Test forward pass returns correct shapes."""
        from cigrl.core.policy import CIGRLPolicy
        
        policy = CIGRLPolicy(obs_dim=32, action_dim=3, hidden_dim=64)
        obs = np.random.randn(32)
        cov = np.random.randn(6, 6)
        
        action, value = policy.forward(obs, cov)
        
        assert action.shape == (3,), f"Expected (3,), got {action.shape}"
        assert isinstance(value, (int, float, np.floating))

    def test_action_bounds(self):
        """Test that actions are bounded by tanh."""
        from cigrl.core.policy import CIGRLPolicy
        
        policy = CIGRLPolicy()
        obs = np.random.randn(32) * 10  # Large inputs
        cov = np.eye(6)
        
        action, _ = policy.forward(obs, cov)
        
        assert np.all(action >= -1.0), "Actions should be >= -1"
        assert np.all(action <= 1.0), "Actions should be <= 1"

    def test_select_actions_batch(self):
        """Test batch action selection."""
        from cigrl.core.policy import CIGRLPolicy
        
        policy = CIGRLPolicy()
        n_agents = 5
        observations = {
            "states": np.random.randn(n_agents, 32),
            "covs": np.random.randn(n_agents, 6, 6)
        }
        
        actions = policy.select_actions(observations, deterministic=True)
        
        assert actions.shape == (n_agents, 3)

    def test_save_and_load(self, tmp_path):
        """Test policy save and load."""
        from cigrl.core.policy import CIGRLPolicy
        
        policy = CIGRLPolicy(obs_dim=32, action_dim=3)
        save_path = tmp_path / "test_policy.npz"
        
        policy.save(str(save_path))
        loaded = CIGRLPolicy.load(str(save_path))
        
        assert loaded.obs_dim == policy.obs_dim
        assert loaded.action_dim == policy.action_dim


class TestUrbanEnvironment:
    """Tests for urban environment simulation."""

    def test_import(self):
        """Test that UrbanEnvironment can be imported."""
        from cigrl.envs.urban import UrbanEnvironment
        env = UrbanEnvironment(size=1000)
        assert env.size == 1000

    def test_gps_quality_in_zone(self):
        """Test GPS quality degrades in denied zones."""
        from cigrl.envs.urban import UrbanEnvironment
        
        env = UrbanEnvironment(
            size=1000,
            gps_denied_zones=[(500, 500, 100)]
        )
        
        # At center of zone
        quality_center = env.get_gps_quality(500, 500, 50)
        # Far from zone
        quality_outside = env.get_gps_quality(100, 100, 50)
        
        assert quality_center < quality_outside

    def test_collision_detection(self):
        """Test building collision detection."""
        from cigrl.envs.urban import UrbanEnvironment
        
        env = UrbanEnvironment(size=1000, n_buildings=50, seed=42)
        
        # Most positions at high altitude should be clear
        clear_count = 0
        for _ in range(100):
            x, y = np.random.uniform(100, 900, 2)
            if not env.check_collision(x, y, 100):  # High altitude
                clear_count += 1
                
        assert clear_count > 80, "Most high-altitude positions should be clear"


class TestDroneSwarm:
    """Tests for drone swarm simulation."""

    def test_import(self):
        """Test that DroneSwarm can be imported."""
        from cigrl.envs.drone import DroneSwarm
        swarm = DroneSwarm(n_drones=5)
        assert swarm.n_drones == 5

    def test_reset(self):
        """Test swarm reset returns observations."""
        from cigrl.envs.drone import DroneSwarm
        
        swarm = DroneSwarm(n_drones=5)
        obs = swarm.reset(seed=42)
        
        assert "states" in obs
        assert "covs" in obs
        assert obs["states"].shape[0] == 5
        assert obs["covs"].shape[0] == 5

    def test_step(self):
        """Test swarm step with actions."""
        from cigrl.envs.drone import DroneSwarm
        
        swarm = DroneSwarm(n_drones=5)
        swarm.reset()
        
        actions = np.zeros((5, 3))  # Hover in place
        obs, rewards, dones, info = swarm.step(actions)
        
        assert obs["states"].shape == (5, 32)
        assert rewards.shape == (5,)
        assert dones.shape == (5,)
        assert "step" in info

    def test_with_environment(self):
        """Test swarm with urban environment."""
        from cigrl.envs.drone import DroneSwarm
        from cigrl.envs.urban import UrbanEnvironment
        
        env = UrbanEnvironment(size=1000, seed=42)
        swarm = DroneSwarm(n_drones=5, env=env)
        
        obs = swarm.reset()
        assert obs["states"].shape[0] == 5


class TestIntegration:
    """End-to-end integration tests."""

    def test_full_episode(self):
        """Test running a complete episode."""
        from cigrl import UrbanEnvironment, DroneSwarm, CIGRLPolicy
        
        env = UrbanEnvironment(size=500, n_buildings=10, seed=42)
        swarm = DroneSwarm(n_drones=3, env=env)
        policy = CIGRLPolicy()
        
        obs = swarm.reset(seed=42)
        total_reward = 0
        
        for step in range(100):
            actions = policy.select_actions(obs, deterministic=True)
            obs, rewards, dones, info = swarm.step(actions)
            total_reward += np.sum(rewards)
            
            if all(dones):
                break
                
        assert step > 0, "Episode should run at least one step"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
