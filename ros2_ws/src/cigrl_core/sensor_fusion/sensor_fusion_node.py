"""
CIGRL Sensor Fusion Module
Extended Kalman Filter based localization with GPS/IMU/Visual fusion.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, TwistStamped
from sensor_msgs.msg import Imu, NavSatFix
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32

import numpy as np
from typing import Optional
import threading


class ExtendedKalmanFilter:
    """
    Extended Kalman Filter for drone state estimation.
    State: [x, y, z, vx, vy, vz, roll, pitch, yaw]
    """
    
    def __init__(self):
        self.state_dim = 9
        
        # State vector
        self.x = np.zeros(self.state_dim)
        
        # State covariance
        self.P = np.eye(self.state_dim) * 0.1
        
        # Process noise
        self.Q = np.diag([0.01, 0.01, 0.01,   # Position noise
                         0.1, 0.1, 0.1,       # Velocity noise
                         0.001, 0.001, 0.001])  # Orientation noise
        
        # Measurement noise - GPS
        self.R_gps = np.diag([1.0, 1.0, 2.0])  # Higher vertical uncertainty
        
        # Measurement noise - IMU
        self.R_imu = np.diag([0.01, 0.01, 0.01,   # Acceleration
                              0.001, 0.001, 0.001])  # Angular velocity
        
        self.last_update_time = None
        
    def predict(self, dt: float, imu_accel: np.ndarray = None):
        """Prediction step using motion model."""
        if dt <= 0:
            return
            
        # State transition matrix (constant velocity model)
        F = np.eye(self.state_dim)
        F[0, 3] = dt  # x += vx * dt
        F[1, 4] = dt  # y += vy * dt
        F[2, 5] = dt  # z += vz * dt
        
        # Predict state
        self.x = F @ self.x
        
        # Include IMU acceleration if available
        if imu_accel is not None:
            # Transform body-frame acceleration to world frame
            # (simplified - ignoring rotation for now)
            self.x[3:6] += imu_accel * dt
            
        # Predict covariance
        self.P = F @ self.P @ F.T + self.Q * dt
        
    def update_gps(self, gps_position: np.ndarray, gps_quality: float):
        """GPS measurement update."""
        # Measurement matrix for position
        H = np.zeros((3, self.state_dim))
        H[0, 0] = 1  # x
        H[1, 1] = 1  # y
        H[2, 2] = 1  # z
        
        # Adjust measurement noise based on GPS quality
        R = self.R_gps / (gps_quality + 0.01)
        
        # Innovation
        z = gps_position
        y = z - H @ self.x
        
        # Innovation covariance
        S = H @ self.P @ H.T + R
        
        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Update state
        self.x = self.x + K @ y
        
        # Update covariance
        I = np.eye(self.state_dim)
        self.P = (I - K @ H) @ self.P
        
    def update_velocity(self, velocity: np.ndarray):
        """Velocity measurement update (from optical flow or other source)."""
        H = np.zeros((3, self.state_dim))
        H[0, 3] = 1  # vx
        H[1, 4] = 1  # vy
        H[2, 5] = 1  # vz
        
        R = np.diag([0.1, 0.1, 0.1])
        
        z = velocity
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        
        self.x = self.x + K @ y
        self.P = (np.eye(self.state_dim) - K @ H) @ self.P
        
    def update_orientation(self, orientation: np.ndarray):
        """Orientation update from IMU/AHRS."""
        H = np.zeros((3, self.state_dim))
        H[0, 6] = 1  # roll
        H[1, 7] = 1  # pitch
        H[2, 8] = 1  # yaw
        
        R = np.diag([0.001, 0.001, 0.01])  # Yaw has more uncertainty
        
        z = orientation
        y = z - H @ self.x
        
        # Handle yaw wraparound
        y[2] = np.arctan2(np.sin(y[2]), np.cos(y[2]))
        
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        
        self.x = self.x + K @ y
        self.P = (np.eye(self.state_dim) - K @ H) @ self.P
        
    def get_position(self) -> np.ndarray:
        return self.x[:3].copy()
    
    def get_velocity(self) -> np.ndarray:
        return self.x[3:6].copy()
    
    def get_orientation(self) -> np.ndarray:
        return self.x[6:9].copy()
    
    def get_position_covariance(self) -> np.ndarray:
        return self.P[:3, :3].copy()
    
    def get_confidence(self) -> float:
        """Return confidence score based on covariance."""
        trace = np.trace(self.P[:3, :3])
        return 1.0 / (1.0 + trace)


class SensorFusionNode(Node):
    """ROS2 node for multi-sensor fusion."""
    
    def __init__(self):
        super().__init__('sensor_fusion_node')
        
        # Parameters
        self.declare_parameter('drone_id', 0)
        self.declare_parameter('fusion_rate', 100.0)
        self.declare_parameter('gps_timeout', 2.0)
        
        self.drone_id = self.get_parameter('drone_id').value
        fusion_rate = self.get_parameter('fusion_rate').value
        self.gps_timeout = self.get_parameter('gps_timeout').value
        
        # EKF
        self.ekf = ExtendedKalmanFilter()
        self.ekf_lock = threading.Lock()
        
        # Sensor data storage
        self.last_imu_time = None
        self.last_gps_time = None
        self.last_imu_accel = np.zeros(3)
        self.gps_quality = 0.5
        
        # Subscribers
        self.imu_sub = self.create_subscription(
            Imu, 'imu', self.imu_callback, 10)
        self.gps_sub = self.create_subscription(
            NavSatFix, 'gps', self.gps_callback, 10)
        self.velocity_sub = self.create_subscription(
            TwistStamped, 'velocity_measured', self.velocity_callback, 10)
            
        # Publishers
        self.pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, 'pose_fused', 10)
        self.odom_pub = self.create_publisher(
            Odometry, 'odometry_fused', 10)
        self.confidence_pub = self.create_publisher(
            Float32, 'localization_confidence', 10)
            
        # Fusion timer
        self.fusion_timer = self.create_timer(
            1.0 / fusion_rate, self.fusion_callback)
            
        self.get_logger().info(f'Sensor Fusion Node initialized for drone {self.drone_id}')
        
    def imu_callback(self, msg: Imu):
        """Process IMU measurements."""
        current_time = self.get_clock().now().nanoseconds / 1e9
        
        with self.ekf_lock:
            # Store acceleration
            self.last_imu_accel = np.array([
                msg.linear_acceleration.x,
                msg.linear_acceleration.y,
                msg.linear_acceleration.z - 9.81  # Remove gravity
            ])
            
            # Predict step with IMU
            if self.last_imu_time is not None:
                dt = current_time - self.last_imu_time
                self.ekf.predict(dt, self.last_imu_accel)
                
            # Update orientation
            # Convert quaternion to euler (simplified)
            q = [msg.orientation.x, msg.orientation.y, 
                 msg.orientation.z, msg.orientation.w]
            roll = np.arctan2(2*(q[3]*q[0] + q[1]*q[2]), 1 - 2*(q[0]**2 + q[1]**2))
            pitch = np.arcsin(2*(q[3]*q[1] - q[2]*q[0]))
            yaw = np.arctan2(2*(q[3]*q[2] + q[0]*q[1]), 1 - 2*(q[1]**2 + q[2]**2))
            
            self.ekf.update_orientation(np.array([roll, pitch, yaw]))
            
            self.last_imu_time = current_time
            
    def gps_callback(self, msg: NavSatFix):
        """Process GPS measurements."""
        current_time = self.get_clock().now().nanoseconds / 1e9
        
        with self.ekf_lock:
            # Convert GPS to local coordinates (simplified - assumes flat earth near origin)
            # In real implementation, use proper projection
            gps_pos = np.array([msg.latitude, msg.longitude, msg.altitude])
            
            # Estimate quality from status and covariance
            if msg.status.status >= 0:  # Has fix
                if msg.position_covariance_type != NavSatFix.COVARIANCE_TYPE_UNKNOWN:
                    avg_cov = np.mean([msg.position_covariance[0], 
                                       msg.position_covariance[4],
                                       msg.position_covariance[8]])
                    self.gps_quality = 1.0 / (1.0 + avg_cov)
                else:
                    self.gps_quality = 0.5
            else:
                self.gps_quality = 0.1
                
            self.ekf.update_gps(gps_pos, self.gps_quality)
            self.last_gps_time = current_time
            
    def velocity_callback(self, msg: TwistStamped):
        """Process velocity measurements (e.g., from optical flow)."""
        with self.ekf_lock:
            velocity = np.array([
                msg.twist.linear.x,
                msg.twist.linear.y,
                msg.twist.linear.z
            ])
            self.ekf.update_velocity(velocity)
            
    def fusion_callback(self):
        """Main fusion loop - publish fused state."""
        current_time = self.get_clock().now()
        
        with self.ekf_lock:
            position = self.ekf.get_position()
            velocity = self.ekf.get_velocity()
            orientation = self.ekf.get_orientation()
            covariance = self.ekf.get_position_covariance()
            confidence = self.ekf.get_confidence()
            
        # Check GPS timeout
        if self.last_gps_time is not None:
            gps_age = current_time.nanoseconds / 1e9 - self.last_gps_time
            if gps_age > self.gps_timeout:
                confidence *= 0.5  # Reduce confidence without GPS
                
        # Publish fused pose
        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.stamp = current_time.to_msg()
        pose_msg.header.frame_id = 'map'
        pose_msg.pose.pose.position.x = float(position[0])
        pose_msg.pose.pose.position.y = float(position[1])
        pose_msg.pose.pose.position.z = float(position[2])
        
        # Convert euler to quaternion
        cy, sy = np.cos(orientation[2]/2), np.sin(orientation[2]/2)
        cp, sp = np.cos(orientation[1]/2), np.sin(orientation[1]/2)
        cr, sr = np.cos(orientation[0]/2), np.sin(orientation[0]/2)
        
        pose_msg.pose.pose.orientation.w = cy * cp * cr + sy * sp * sr
        pose_msg.pose.pose.orientation.x = cy * cp * sr - sy * sp * cr
        pose_msg.pose.pose.orientation.y = sy * cp * sr + cy * sp * cr
        pose_msg.pose.pose.orientation.z = sy * cp * cr - cy * sp * sr
        
        # Flatten covariance
        pose_msg.pose.covariance = [0.0] * 36
        pose_msg.pose.covariance[0] = covariance[0, 0]
        pose_msg.pose.covariance[7] = covariance[1, 1]
        pose_msg.pose.covariance[14] = covariance[2, 2]
        
        self.pose_pub.publish(pose_msg)
        
        # Publish odometry
        odom_msg = Odometry()
        odom_msg.header = pose_msg.header
        odom_msg.child_frame_id = f'drone_{self.drone_id}'
        odom_msg.pose = pose_msg.pose
        odom_msg.twist.twist.linear.x = float(velocity[0])
        odom_msg.twist.twist.linear.y = float(velocity[1])
        odom_msg.twist.twist.linear.z = float(velocity[2])
        
        self.odom_pub.publish(odom_msg)
        
        # Publish confidence
        conf_msg = Float32()
        conf_msg.data = float(confidence)
        self.confidence_pub.publish(conf_msg)


def main(args=None):
    rclpy.init(args=args)
    node = SensorFusionNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
