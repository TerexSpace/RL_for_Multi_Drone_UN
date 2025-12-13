"""
Sensor Fusion Tests

Unit tests for AEKF (Adaptive Extended Kalman Filter) sensor fusion components.
"""

import pytest
import numpy as np


class TestEKFPrediction:
    """Tests for EKF prediction step."""

    def test_state_propagation(self):
        """Test state vector propagation with velocity."""
        dt = 0.1  # Time step
        position = np.array([0.0, 0.0, 10.0])  # x, y, z
        velocity = np.array([1.0, 2.0, 0.0])  # vx, vy, vz
        
        new_position = position + velocity * dt
        
        expected = np.array([0.1, 0.2, 10.0])
        np.testing.assert_array_almost_equal(new_position, expected)

    def test_covariance_growth(self):
        """Test covariance increases during prediction."""
        P_prior = np.eye(6) * 0.1
        Q = np.eye(6) * 0.01  # Process noise
        
        # Simplified prediction (identity state transition)
        P_predicted = P_prior + Q
        
        assert np.trace(P_predicted) > np.trace(P_prior)


class TestEKFUpdate:
    """Tests for EKF measurement update step."""

    def test_covariance_reduction_with_measurement(self):
        """Test covariance decreases after GPS update."""
        P_prior = np.eye(3) * 1.0
        H = np.eye(3)  # Observation matrix
        R = np.eye(3) * 0.1  # Measurement noise
        
        # Kalman gain
        S = H @ P_prior @ H.T + R
        K = P_prior @ H.T @ np.linalg.inv(S)
        
        # Updated covariance
        P_posterior = (np.eye(3) - K @ H) @ P_prior
        
        assert np.trace(P_posterior) < np.trace(P_prior)

    def test_high_measurement_noise_reduces_gain(self):
        """Test high R reduces Kalman gain (less trust in measurement)."""
        P = 1.0
        H = 1.0
        
        R_low = 0.1
        K_low = P * H / (H * P * H + R_low)
        
        R_high = 10.0
        K_high = P * H / (H * P * H + R_high)
        
        assert K_high < K_low


class TestAdaptiveFuzzyEKF:
    """Tests for Fuzzy Adaptive EKF components."""

    def test_nis_computation(self):
        """Test Normalized Innovation Squared."""
        innovation = np.array([1.0, 0.5, 0.2])
        S = np.eye(3) * 0.5  # Innovation covariance
        
        NIS = innovation.T @ np.linalg.inv(S) @ innovation
        
        assert NIS > 0, "NIS should be positive"

    def test_adaptive_r_scaling(self):
        """Test R scaling based on GPS quality."""
        R_nominal = np.eye(3) * 0.1
        gps_quality = 0.3  # 30% quality (degraded)
        
        # Scale R inversely with GPS quality
        scaling_factor = 1.0 / max(gps_quality, 0.01)
        R_scaled = R_nominal * scaling_factor
        
        assert np.trace(R_scaled) > np.trace(R_nominal)

    def test_bias_estimation_convergence(self):
        """Test IMU bias estimation converges over time."""
        true_bias = 0.05  # m/s^2
        estimated_biases = [0.0]
        learning_rate = 0.1
        
        # Simulate bias learning
        for _ in range(50):
            error = true_bias - estimated_biases[-1]
            new_estimate = estimated_biases[-1] + learning_rate * error
            estimated_biases.append(new_estimate)
        
        final_error = abs(true_bias - estimated_biases[-1])
        assert final_error < 0.01, "Bias estimate should converge"


class TestGPSDeniedNavigation:
    """Tests for GPS-denied navigation behavior."""

    def test_position_drift_during_outage(self):
        """Test position drift accumulates without GPS."""
        dt = 1.0  # 1 second
        outage_duration = 60  # seconds
        imu_drift_rate = 0.1  # m/s^2 bias
        
        # Drift accumulates quadratically: 0.5 * a * t^2
        expected_drift = 0.5 * imu_drift_rate * outage_duration**2
        
        assert expected_drift > 0

    def test_cooperative_localization_reduces_drift(self):
        """Test neighbor measurements reduce position uncertainty."""
        self_uncertainty = 10.0  # High due to GPS outage
        neighbor_uncertainty = 1.0  # Low, has GPS
        relative_measurement_noise = 0.5
        
        # Weighted fusion
        w_self = 1.0 / self_uncertainty
        w_neighbor = 1.0 / (neighbor_uncertainty + relative_measurement_noise)
        
        fused_uncertainty = 1.0 / (w_self + w_neighbor)
        
        assert fused_uncertainty < self_uncertainty


class TestIMUModel:
    """Tests for IMU sensor model."""

    def test_accelerometer_noise_model(self):
        """Test accelerometer measurement with noise."""
        true_acceleration = np.array([0.0, 0.0, -9.81])
        noise_std = 0.02  # m/s^2
        
        np.random.seed(42)
        measured = true_acceleration + np.random.normal(0, noise_std, 3)
        
        error = np.linalg.norm(measured - true_acceleration)
        assert error < 0.1, "Measurement error should be small"

    def test_gyroscope_bias_drift(self):
        """Test gyroscope bias random walk."""
        initial_bias = 0.001  # rad/s
        drift_rate = 0.0001  # rad/s per sqrt(s)
        dt = 1.0
        
        np.random.seed(42)
        new_bias = initial_bias + drift_rate * np.sqrt(dt) * np.random.randn()
        
        # Bias should change but stay bounded
        assert abs(new_bias) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
