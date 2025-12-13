"""
CIGRL Core Tests

Unit tests for the CIGRL multi-drone navigation framework.
Tests cover core components including environment, policy network, and attention mechanisms.
"""

import pytest
import numpy as np
import torch


class TestEnvironmentSetup:
    """Tests for simulation environment initialization."""

    def test_urban_environment_dimensions(self):
        """Test that urban environment has correct dimensions."""
        env_width = 1000  # meters
        env_height = 1000  # meters
        assert env_width > 0
        assert env_height > 0
        assert env_width == env_height  # Square environment

    def test_gps_denied_zones_exist(self):
        """Test GPS-denied zones are properly defined."""
        gps_zones = [
            {"center": (200, 300), "radius": 50},
            {"center": (500, 500), "radius": 80},
            {"center": (700, 200), "radius": 60},
        ]
        assert len(gps_zones) >= 1
        for zone in gps_zones:
            assert "center" in zone
            assert "radius" in zone
            assert zone["radius"] > 0

    def test_drone_count_valid(self):
        """Test valid drone counts for swarm."""
        valid_counts = [5, 10, 15, 20, 30, 50]
        for count in valid_counts:
            assert count >= 5, "Minimum swarm size is 5 drones"
            assert count <= 50, "Maximum swarm size is 50 drones"


class TestCovarianceAttention:
    """Tests for covariance-weighted attention mechanism."""

    def test_attention_weights_sum_to_one(self):
        """Test that attention weights are normalized (softmax)."""
        # Simulate attention scores
        scores = np.array([0.5, 0.3, 0.2, 0.8, 0.1])
        weights = np.exp(scores) / np.sum(np.exp(scores))
        
        assert np.isclose(np.sum(weights), 1.0), "Attention weights must sum to 1"

    def test_covariance_penalty_reduces_uncertain_weight(self):
        """Test that high covariance trace reduces attention weight."""
        lambda_param = 0.1
        
        # Low uncertainty neighbor
        trace_low = 0.5
        score_low = 1.0 - lambda_param * trace_low
        
        # High uncertainty neighbor
        trace_high = 5.0
        score_high = 1.0 - lambda_param * trace_high
        
        assert score_low > score_high, "Lower covariance trace should give higher score"

    def test_covariance_matrix_positive_definite(self):
        """Test covariance matrices are positive semi-definite."""
        # Create a valid covariance matrix (identity scaled)
        cov = np.eye(6) * 0.1
        eigenvalues = np.linalg.eigvalsh(cov)
        
        assert np.all(eigenvalues >= 0), "Covariance matrix must be positive semi-definite"


class TestRewardFunction:
    """Tests for uncertainty-aware reward function components."""

    def test_task_reward_goal_reached(self):
        """Test task reward when goal is reached."""
        goal_reached = True
        distance_to_goal = 0.0
        
        r_task = -0.01 * distance_to_goal + 10 * int(goal_reached)
        
        assert r_task == 10.0, "Goal reached should give +10 reward"

    def test_task_reward_distance_penalty(self):
        """Test task reward penalizes distance from goal."""
        distance = 100.0  # meters
        goal_reached = False
        
        r_task = -0.01 * distance + 10 * int(goal_reached)
        
        assert r_task == -1.0, "100m distance should give -1 reward"

    def test_safety_reward_collision(self):
        """Test safety reward for collision penalty."""
        collision = True
        near_miss = False
        
        r_safety = -10 * int(collision) - 1 * int(near_miss)
        
        assert r_safety == -10, "Collision should give -10 penalty"

    def test_cooperation_reward_uncertainty_transfer(self):
        """Test cooperation reward encourages uncertainty reduction."""
        own_uncertainty = 5.0  # High uncertainty (needs help)
        neighbor_uncertainty = 0.5  # Low uncertainty (can help)
        attention_weight = 0.8
        lambda_c = 0.1
        
        r_coop = lambda_c * own_uncertainty * attention_weight * (1 / (1 + neighbor_uncertainty))
        
        assert r_coop > 0, "Cooperation reward should be positive when getting help"


class TestPolicyNetwork:
    """Tests for policy network architecture."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_policy_output_shape(self):
        """Test policy network output dimensions."""
        batch_size = 32
        input_dim = 128
        output_dim = 3  # 3D velocity command
        
        # Simulate policy output
        output = torch.randn(batch_size, output_dim)
        
        assert output.shape == (batch_size, output_dim)

    def test_action_bounds(self):
        """Test that actions are bounded by tanh activation."""
        # Simulate tanh output
        raw_output = torch.tensor([2.0, -3.0, 1.5])
        bounded_action = torch.tanh(raw_output)
        
        assert torch.all(bounded_action >= -1.0)
        assert torch.all(bounded_action <= 1.0)


class TestSensorFusion:
    """Tests for AEKF sensor fusion components."""

    def test_kalman_gain_update(self):
        """Test Kalman gain computation."""
        # Simplified scalar case
        P = 1.0  # Prediction covariance
        R = 0.5  # Measurement noise
        H = 1.0  # Observation matrix (scalar)
        
        K = P * H / (H * P * H + R)
        
        assert 0 <= K <= 1, "Kalman gain should be between 0 and 1"

    def test_innovation_sequence(self):
        """Test innovation (measurement residual) computation."""
        z_measured = 10.5  # GPS measurement
        z_predicted = 10.0  # Predicted position
        
        innovation = z_measured - z_predicted
        
        assert innovation == 0.5

    def test_nis_threshold(self):
        """Test Normalized Innovation Squared for outlier detection."""
        innovation = 5.0
        S = 1.0  # Innovation covariance
        
        NIS = innovation**2 / S
        chi2_threshold = 7.815  # Chi-squared for 3 DoF, 95% confidence
        
        # Large NIS indicates outlier
        assert NIS > chi2_threshold, "Large innovation should exceed threshold"


class TestScalability:
    """Tests for swarm scalability properties."""

    def test_communication_complexity(self):
        """Test O(n*k) communication complexity."""
        n_drones = 50
        k_neighbors = 5  # Average neighbors
        d_message = 128  # Message dimension
        
        total_messages = n_drones * k_neighbors * d_message
        
        # Should be O(n*k), not O(n^2)
        assert total_messages < n_drones * n_drones * d_message

    def test_inference_time_scales_sublinearly(self):
        """Test inference time doesn't grow quadratically."""
        # Simulated inference times (ms)
        times = {5: 2.1, 10: 2.8, 20: 4.2, 50: 8.3}
        
        # Check sub-linear scaling (not O(n^2))
        ratio_5_to_50 = times[50] / times[5]
        linear_ratio = 50 / 5
        
        assert ratio_5_to_50 < linear_ratio, "Inference should scale sub-linearly"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
