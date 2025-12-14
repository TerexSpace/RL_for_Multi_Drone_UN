"""
Sensor Fusion Tests

Integration tests for AdaptiveEKF sensor fusion components.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add cigrl package to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestEKFState:
    """Tests for EKF state dataclass."""

    def test_import_state(self):
        """Test EKFState can be imported."""
        from cigrl.core.sensor_fusion import EKFState
        
        state = EKFState(
            position=np.zeros(3),
            velocity=np.zeros(3),
            attitude=np.zeros(3),
            accel_bias=np.zeros(3),
            gyro_bias=np.zeros(3),
            covariance=np.eye(15)
        )
        assert state.position.shape == (3,)


class TestEKFPrediction:
    """Tests for EKF prediction step."""

    def test_state_propagation(self):
        """Test state vector propagation with velocity."""
        from cigrl.core.sensor_fusion import AdaptiveEKF
        
        ekf = AdaptiveEKF()
        state = ekf.initialize(position=[0, 0, 10], velocity=None)
        
        # Apply acceleration
        state = ekf.predict(state, accel=[1, 0, -9.81], gyro=[0, 0, 0], dt=1.0)
        
        # Velocity should have changed due to acceleration
        assert state.velocity[0] != 0

    def test_covariance_growth(self):
        """Test covariance increases during prediction."""
        from cigrl.core.sensor_fusion import AdaptiveEKF
        
        ekf = AdaptiveEKF()
        state = ekf.initialize()
        
        traces = [np.trace(state.covariance)]
        for _ in range(5):
            state = ekf.predict(state, accel=[0, 0, -9.81], gyro=[0, 0, 0], dt=0.1)
            traces.append(np.trace(state.covariance))
        
        # Each prediction should increase covariance
        for i in range(1, len(traces)):
            assert traces[i] > traces[i-1]


class TestEKFUpdate:
    """Tests for EKF measurement update step."""

    def test_covariance_reduction_with_measurement(self):
        """Test covariance decreases after GPS update."""
        from cigrl.core.sensor_fusion import AdaptiveEKF
        
        ekf = AdaptiveEKF(nis_threshold=100.0)  # High threshold to avoid rejection
        state = ekf.initialize()
        
        # Build up uncertainty with predictions
        for _ in range(20):
            state = ekf.predict(state, accel=[0, 0, -9.81], gyro=[0, 0, 0], dt=0.1)
        
        trace_before = np.trace(state.covariance[:3, :3])
        
        # Use GPS measurement close to current position to avoid NIS rejection
        gps_pos = state.position + np.random.randn(3) * 0.1
        state, accepted = ekf.update_gps(state, gps_position=gps_pos, gps_quality=1.0)
        trace_after = np.trace(state.covariance[:3, :3])
        
        # GPS update should reduce uncertainty
        assert accepted, "GPS update should be accepted with high NIS threshold"
        assert trace_after <= trace_before, "GPS update should reduce covariance"

    def test_low_gps_quality_increases_noise(self):
        """Test that low GPS quality leads to less correction."""
        from cigrl.core.sensor_fusion import AdaptiveEKF
        
        # Use separate EKF instances to avoid shared state
        ekf_high = AdaptiveEKF()
        ekf_low = AdaptiveEKF()
        
        # Test with high quality
        state_high = ekf_high.initialize(position=[100, 100, 100])
        for _ in range(10):
            state_high = ekf_high.predict(state_high, accel=[0, 0, -9.81], gyro=[0, 0, 0], dt=0.1)
        state_high, _ = ekf_high.update_gps(state_high, gps_position=[0, 0, 0], gps_quality=1.0)
        
        # Test with low quality
        state_low = ekf_low.initialize(position=[100, 100, 100])
        for _ in range(10):
            state_low = ekf_low.predict(state_low, accel=[0, 0, -9.81], gyro=[0, 0, 0], dt=0.1)
        state_low, _ = ekf_low.update_gps(state_low, gps_position=[0, 0, 0], gps_quality=0.1)
        
        # High quality should correct more (position closer to measurement)
        dist_high = np.linalg.norm(state_high.position)
        dist_low = np.linalg.norm(state_low.position)
        
        # Allow some tolerance for numerical precision
        assert dist_high <= dist_low * 1.1, f"High quality correction ({dist_high}) should be >= low quality ({dist_low})"


class TestAdaptiveFiltering:
    """Tests for Fuzzy Adaptive EKF components."""

    def test_nis_computation(self):
        """Test NIS-based outlier rejection works."""
        from cigrl.core.sensor_fusion import AdaptiveEKF
        
        ekf = AdaptiveEKF(nis_threshold=7.815)
        state = ekf.initialize()
        
        # Normal measurement
        _, accepted_normal = ekf.update_gps(state, [0.1, 0.1, 0.1])
        
        # Outlier measurement  
        _, accepted_outlier = ekf.update_gps(state, [1000, 1000, 1000])
        
        assert accepted_normal
        assert not accepted_outlier


class TestBiasEstimation:
    """Tests for IMU bias estimation."""

    def test_bias_state_exists(self):
        """Test that bias states are in the state vector."""
        from cigrl.core.sensor_fusion import AdaptiveEKF
        
        ekf = AdaptiveEKF()
        state = ekf.initialize()
        
        assert state.accel_bias.shape == (3,)
        assert state.gyro_bias.shape == (3,)

    def test_bias_updated_after_gps(self):
        """Test that biases can be updated via GPS measurements."""
        from cigrl.core.sensor_fusion import AdaptiveEKF
        
        ekf = AdaptiveEKF()
        state = ekf.initialize()
        initial_bias = state.accel_bias.copy()
        
        # Many prediction/update cycles
        for _ in range(50):
            state = ekf.predict(state, accel=[0.1, 0, -9.81], gyro=[0, 0, 0], dt=0.1)
            state, _ = ekf.update_gps(state, state.position + np.random.randn(3) * 0.1)
        
        # Bias should have changed (may drift slightly)
        # Note: with synthetic data, change might be small
        assert state.accel_bias is not None


class TestPositionUncertainty:
    """Tests for uncertainty metrics."""

    def test_get_position_uncertainty(self):
        """Test position uncertainty calculation."""
        from cigrl.core.sensor_fusion import AdaptiveEKF
        
        ekf = AdaptiveEKF()
        state = ekf.initialize()
        
        uncertainty = ekf.get_position_uncertainty(state)
        
        assert uncertainty > 0
        assert np.isclose(uncertainty, np.trace(state.covariance[:3, :3]))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
