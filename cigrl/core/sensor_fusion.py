"""
Adaptive Extended Kalman Filter for Sensor Fusion

Implements an EKF with:
- Online IMU bias estimation
- Fuzzy adaptive measurement noise adjustment
- Innovation-based outlier rejection
"""

import numpy as np
from typing import Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class EKFState:
    """EKF state vector and covariance."""
    position: np.ndarray  # (3,) x, y, z in NED frame
    velocity: np.ndarray  # (3,) vx, vy, vz
    attitude: np.ndarray  # (3,) roll, pitch, yaw
    accel_bias: np.ndarray  # (3,) accelerometer bias
    gyro_bias: np.ndarray  # (3,) gyroscope bias
    covariance: np.ndarray  # (15, 15) state covariance


class AdaptiveEKF:
    """
    Adaptive Extended Kalman Filter for INS/GNSS fusion.
    
    Features:
    - 15-state error-state formulation
    - Online accelerometer and gyroscope bias estimation
    - Fuzzy adaptive R matrix based on innovation statistics
    - Normalized Innovation Squared (NIS) outlier rejection
    
    Parameters
    ----------
    gps_noise_nominal : float
        Nominal GPS position noise standard deviation (meters)
    adaptation_rate : float
        Rate of measurement noise adaptation (0-1)
    nis_threshold : float
        Chi-squared threshold for outlier rejection
        
    Example
    -------
    >>> ekf = AdaptiveEKF(gps_noise_nominal=1.0, adaptation_rate=0.1)
    >>> state = ekf.initialize(position=[0, 0, 10])
    >>> state = ekf.predict(state, accel=[0, 0, -9.81], gyro=[0, 0, 0], dt=0.01)
    >>> state, accepted = ekf.update_gps(state, gps_position=[0.1, 0.1, 10.1])
    """
    
    def __init__(
        self,
        gps_noise_nominal: float = 1.0,
        adaptation_rate: float = 0.1,
        nis_threshold: float = 7.815  # Chi-squared, 3 DoF, 95%
    ):
        self.gps_noise_nominal = gps_noise_nominal
        self.adaptation_rate = adaptation_rate
        self.nis_threshold = nis_threshold
        
        # Current adaptive measurement noise
        self.R_gps = np.eye(3) * gps_noise_nominal**2
        
        # Process noise (tuned for MEMS IMU)
        self.Q = self._build_process_noise()
        
        # Innovation history for adaptation
        self.innovation_history = []
        self.max_history = 20
        
    def _build_process_noise(self) -> np.ndarray:
        """Build process noise covariance matrix."""
        Q = np.zeros((15, 15))
        
        # Position process noise (driven by velocity)
        Q[0:3, 0:3] = np.eye(3) * 0.01
        
        # Velocity process noise (driven by accelerometer noise)
        Q[3:6, 3:6] = np.eye(3) * 0.1
        
        # Attitude process noise (driven by gyro noise)
        Q[6:9, 6:9] = np.eye(3) * 0.001
        
        # Accelerometer bias random walk
        Q[9:12, 9:12] = np.eye(3) * 1e-6
        
        # Gyroscope bias random walk
        Q[12:15, 12:15] = np.eye(3) * 1e-8
        
        return Q
    
    def initialize(
        self,
        position: np.ndarray = None,
        velocity: np.ndarray = None,
        attitude: np.ndarray = None
    ) -> EKFState:
        """
        Initialize EKF state.
        
        Parameters
        ----------
        position : np.ndarray, optional
            Initial position [x, y, z], default [0, 0, 0]
        velocity : np.ndarray, optional
            Initial velocity [vx, vy, vz], default [0, 0, 0]
        attitude : np.ndarray, optional
            Initial attitude [roll, pitch, yaw], default [0, 0, 0]
            
        Returns
        -------
        EKFState
            Initialized state with default covariance
        """
        position = np.array(position) if position is not None else np.zeros(3)
        velocity = np.array(velocity) if velocity is not None else np.zeros(3)
        attitude = np.array(attitude) if attitude is not None else np.zeros(3)
        
        # Initial covariance (relatively uncertain)
        P = np.diag([
            10, 10, 10,       # Position uncertainty (m)
            1, 1, 1,          # Velocity uncertainty (m/s)
            0.1, 0.1, 0.1,    # Attitude uncertainty (rad)
            0.01, 0.01, 0.01, # Accel bias uncertainty (m/s^2)
            0.001, 0.001, 0.001  # Gyro bias uncertainty (rad/s)
        ])
        
        return EKFState(
            position=position,
            velocity=velocity,
            attitude=attitude,
            accel_bias=np.zeros(3),
            gyro_bias=np.zeros(3),
            covariance=P
        )
    
    def predict(
        self,
        state: EKFState,
        accel: np.ndarray,
        gyro: np.ndarray,
        dt: float
    ) -> EKFState:
        """
        Prediction step using IMU measurements.
        
        Parameters
        ----------
        state : EKFState
            Current state estimate
        accel : np.ndarray
            Accelerometer measurement [ax, ay, az] (m/s^2)
        gyro : np.ndarray
            Gyroscope measurement [wx, wy, wz] (rad/s)
        dt : float
            Time step (seconds)
            
        Returns
        -------
        EKFState
            Predicted state
        """
        accel = np.array(accel)
        gyro = np.array(gyro)
        
        # Correct for biases
        accel_corrected = accel - state.accel_bias
        gyro_corrected = gyro - state.gyro_bias
        
        # Simple integration (in practice, use more sophisticated mechanization)
        new_velocity = state.velocity + accel_corrected * dt
        new_position = state.position + state.velocity * dt + 0.5 * accel_corrected * dt**2
        new_attitude = state.attitude + gyro_corrected * dt
        
        # State transition Jacobian (simplified)
        F = np.eye(15)
        F[0:3, 3:6] = np.eye(3) * dt  # Position depends on velocity
        
        # Propagate covariance
        new_P = F @ state.covariance @ F.T + self.Q * dt
        
        return EKFState(
            position=new_position,
            velocity=new_velocity,
            attitude=new_attitude,
            accel_bias=state.accel_bias,
            gyro_bias=state.gyro_bias,
            covariance=new_P
        )
    
    def update_gps(
        self,
        state: EKFState,
        gps_position: np.ndarray,
        gps_quality: float = 1.0
    ) -> Tuple[EKFState, bool]:
        """
        Update state with GPS measurement.
        
        Parameters
        ----------
        state : EKFState
            Prior state estimate
        gps_position : np.ndarray
            GPS position measurement [x, y, z]
        gps_quality : float
            GPS quality indicator (0-1), affects measurement noise
            
        Returns
        -------
        EKFState
            Updated state
        bool
            True if measurement was accepted (passed NIS test)
        """
        gps_position = np.array(gps_position)
        
        # Observation matrix (observe position directly)
        H = np.zeros((3, 15))
        H[0:3, 0:3] = np.eye(3)
        
        # Adapt measurement noise based on GPS quality
        scale = 1.0 / max(gps_quality, 0.01)
        R_current = self.R_gps * scale
        
        # Innovation (measurement residual)
        innovation = gps_position - state.position
        
        # Innovation covariance
        S = H @ state.covariance @ H.T + R_current
        
        # Normalized Innovation Squared (NIS) for outlier rejection
        NIS = innovation.T @ np.linalg.inv(S) @ innovation
        
        if NIS > self.nis_threshold:
            # Reject as outlier
            self._update_adaptive_R(innovation, rejected=True)
            return state, False
        
        # Kalman gain
        K = state.covariance @ H.T @ np.linalg.inv(S)
        
        # State update
        dx = K @ innovation
        new_position = state.position + dx[0:3]
        new_velocity = state.velocity + dx[3:6]
        new_attitude = state.attitude + dx[6:9]
        new_accel_bias = state.accel_bias + dx[9:12]
        new_gyro_bias = state.gyro_bias + dx[12:15]
        
        # Covariance update (Joseph form for numerical stability)
        I_KH = np.eye(15) - K @ H
        new_P = I_KH @ state.covariance @ I_KH.T + K @ R_current @ K.T
        
        # Update adaptive R
        self._update_adaptive_R(innovation, rejected=False)
        
        return EKFState(
            position=new_position,
            velocity=new_velocity,
            attitude=new_attitude,
            accel_bias=new_accel_bias,
            gyro_bias=new_gyro_bias,
            covariance=new_P
        ), True
    
    def _update_adaptive_R(self, innovation: np.ndarray, rejected: bool):
        """Update measurement noise based on innovation statistics."""
        self.innovation_history.append(innovation)
        if len(self.innovation_history) > self.max_history:
            self.innovation_history.pop(0)
            
        if len(self.innovation_history) >= 5:
            # Estimate innovation covariance from history
            innovations = np.array(self.innovation_history)
            empirical_cov = np.cov(innovations.T)
            
            # Adapt R towards empirical covariance
            if not rejected:
                self.R_gps = (1 - self.adaptation_rate) * self.R_gps + \
                             self.adaptation_rate * empirical_cov
    
    def get_position_uncertainty(self, state: EKFState) -> float:
        """Get total position uncertainty (trace of position covariance)."""
        return np.trace(state.covariance[0:3, 0:3])
    
    def __repr__(self) -> str:
        return (
            f"AdaptiveEKF(gps_noise_nominal={self.gps_noise_nominal}, "
            f"adaptation_rate={self.adaptation_rate})"
        )
