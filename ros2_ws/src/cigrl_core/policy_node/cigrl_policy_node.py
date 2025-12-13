"""
CIGRL RL Policy Node
PyTorch-based inference node for CIGRL multi-agent reinforcement learning policy.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import PoseStamped, TwistStamped, Point
from sensor_msgs.msg import Imu, NavSatFix
from std_msgs.msg import Float32MultiArray, Int32
from nav_msgs.msg import Path

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional
import threading
from dataclasses import dataclass


@dataclass
class DroneState:
    """State representation for a single drone."""
    position: np.ndarray
    velocity: np.ndarray
    orientation: np.ndarray
    gps_quality: float
    imu_data: np.ndarray
    target: np.ndarray


class AttentionModule(nn.Module):
    """Multi-head attention for neighbor information fusion."""
    
    def __init__(self, embed_dim: int = 64, num_heads: int = 4):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, 
                value: torch.Tensor) -> torch.Tensor:
        attn_output, _ = self.attention(query, key, value)
        return self.layer_norm(query + attn_output)


class CIGRLPolicy(nn.Module):
    """CIGRL policy network with attention-based communication."""
    
    def __init__(self, 
                 state_dim: int = 15,
                 action_dim: int = 3,
                 hidden_dim: int = 128,
                 neighbor_dim: int = 12,
                 max_neighbors: int = 5):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_neighbors = max_neighbors
        
        # Self-state encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Neighbor encoder
        self.neighbor_encoder = nn.Sequential(
            nn.Linear(neighbor_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Attention for neighbor fusion
        self.attention = AttentionModule(hidden_dim, num_heads=4)
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Bounded actions
        )
        
        # Value head (for training)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state: torch.Tensor, 
                neighbor_states: torch.Tensor) -> tuple:
        """
        Forward pass.
        
        Args:
            state: Self state [batch, state_dim]
            neighbor_states: Neighbor states [batch, max_neighbors, neighbor_dim]
            
        Returns:
            action: Action vector [batch, action_dim]
            value: State value [batch, 1]
        """
        # Encode self state
        self_encoding = self.state_encoder(state)  # [batch, hidden]
        
        # Encode neighbors
        batch_size = state.shape[0]
        neighbor_encodings = self.neighbor_encoder(
            neighbor_states.view(-1, neighbor_states.shape[-1])
        ).view(batch_size, self.max_neighbors, -1)  # [batch, neighbors, hidden]
        
        # Attention-based fusion
        query = self_encoding.unsqueeze(1)  # [batch, 1, hidden]
        fused = self.attention(query, neighbor_encodings, neighbor_encodings)
        fused = fused.squeeze(1)  # [batch, hidden]
        
        # Concatenate self and fused neighbor info
        combined = torch.cat([self_encoding, fused], dim=-1)  # [batch, hidden*2]
        
        # Output action and value
        action = self.policy_head(combined)
        value = self.value_head(combined)
        
        return action, value


class CIGRLPolicyNode(Node):
    """ROS2 node for CIGRL policy inference."""
    
    def __init__(self):
        super().__init__('cigrl_policy_node')
        
        # Parameters
        self.declare_parameter('drone_id', 0)
        self.declare_parameter('num_drones', 5)
        self.declare_parameter('model_path', '')
        self.declare_parameter('max_speed', 5.0)
        self.declare_parameter('control_rate', 20.0)
        
        self.drone_id = self.get_parameter('drone_id').value
        self.num_drones = self.get_parameter('num_drones').value
        self.max_speed = self.get_parameter('max_speed').value
        control_rate = self.get_parameter('control_rate').value
        
        # Initialize policy network
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy = CIGRLPolicy().to(self.device)
        self.policy.eval()
        
        # Load pretrained weights if provided
        model_path = self.get_parameter('model_path').value
        if model_path:
            self.load_model(model_path)
            
        # State storage
        self.self_state = DroneState(
            position=np.zeros(3),
            velocity=np.zeros(3),
            orientation=np.zeros(4),
            gps_quality=1.0,
            imu_data=np.zeros(6),
            target=np.zeros(3)
        )
        self.neighbor_states: Dict[int, DroneState] = {}
        self.state_lock = threading.Lock()
        
        # QoS profile for reliable communication
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        
        # Subscribers
        self.pose_sub = self.create_subscription(
            PoseStamped, 'pose', self.pose_callback, qos)
        self.velocity_sub = self.create_subscription(
            TwistStamped, 'velocity', self.velocity_callback, qos)
        self.imu_sub = self.create_subscription(
            Imu, 'imu', self.imu_callback, qos)
        self.gps_sub = self.create_subscription(
            NavSatFix, 'gps', self.gps_callback, qos)
        self.target_sub = self.create_subscription(
            Point, 'target', self.target_callback, qos)
            
        # Neighbor state subscribers
        for i in range(self.num_drones):
            if i != self.drone_id:
                self.create_subscription(
                    Float32MultiArray,
                    f'/drone_{i}/shared_state',
                    lambda msg, drone_id=i: self.neighbor_callback(msg, drone_id),
                    qos
                )
                
        # Publishers
        self.cmd_pub = self.create_publisher(TwistStamped, 'cmd_vel', qos)
        self.shared_state_pub = self.create_publisher(
            Float32MultiArray, 'shared_state', qos)
        self.path_pub = self.create_publisher(Path, 'planned_path', qos)
        
        # Control timer
        self.control_timer = self.create_timer(
            1.0 / control_rate, self.control_callback)
        
        # State sharing timer
        self.share_timer = self.create_timer(0.1, self.share_state_callback)
        
        self.get_logger().info(f'CIGRL Policy Node initialized for drone {self.drone_id}')
        
    def load_model(self, path: str):
        """Load pretrained model weights."""
        try:
            state_dict = torch.load(path, map_location=self.device)
            self.policy.load_state_dict(state_dict)
            self.get_logger().info(f'Loaded model from {path}')
        except Exception as e:
            self.get_logger().error(f'Failed to load model: {e}')
            
    def pose_callback(self, msg: PoseStamped):
        """Handle pose updates."""
        with self.state_lock:
            self.self_state.position = np.array([
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z
            ])
            self.self_state.orientation = np.array([
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
                msg.pose.orientation.w
            ])
            
    def velocity_callback(self, msg: TwistStamped):
        """Handle velocity updates."""
        with self.state_lock:
            self.self_state.velocity = np.array([
                msg.twist.linear.x,
                msg.twist.linear.y,
                msg.twist.linear.z
            ])
            
    def imu_callback(self, msg: Imu):
        """Handle IMU updates."""
        with self.state_lock:
            self.self_state.imu_data = np.array([
                msg.linear_acceleration.x,
                msg.linear_acceleration.y,
                msg.linear_acceleration.z,
                msg.angular_velocity.x,
                msg.angular_velocity.y,
                msg.angular_velocity.z
            ])
            
    def gps_callback(self, msg: NavSatFix):
        """Handle GPS updates and quality estimation."""
        with self.state_lock:
            # Estimate GPS quality from covariance
            if msg.position_covariance_type != NavSatFix.COVARIANCE_TYPE_UNKNOWN:
                avg_cov = np.mean(msg.position_covariance[:3])
                self.self_state.gps_quality = 1.0 / (1.0 + avg_cov)
            else:
                self.self_state.gps_quality = 0.5
                
    def target_callback(self, msg: Point):
        """Handle target waypoint updates."""
        with self.state_lock:
            self.self_state.target = np.array([msg.x, msg.y, msg.z])
            
    def neighbor_callback(self, msg: Float32MultiArray, drone_id: int):
        """Handle neighbor state broadcasts."""
        with self.state_lock:
            data = np.array(msg.data)
            if len(data) >= 10:
                self.neighbor_states[drone_id] = DroneState(
                    position=data[0:3],
                    velocity=data[3:6],
                    orientation=np.array([0, 0, 0, 1]),
                    gps_quality=data[6],
                    imu_data=np.zeros(6),
                    target=data[7:10]
                )
                
    def prepare_state_tensor(self) -> tuple:
        """Prepare state tensors for policy inference."""
        with self.state_lock:
            # Self state: position, velocity, target direction, GPS quality
            to_target = self.self_state.target - self.self_state.position
            dist = np.linalg.norm(to_target)
            if dist > 0:
                to_target = to_target / dist
                
            self_state = np.concatenate([
                self.self_state.position,
                self.self_state.velocity,
                to_target,
                [dist / 100.0],  # Normalized distance
                [self.self_state.gps_quality],
                self.self_state.imu_data[:3]
            ])
            
            # Neighbor states
            neighbor_list = []
            for n_id, n_state in list(self.neighbor_states.items())[:5]:
                rel_pos = n_state.position - self.self_state.position
                neighbor_list.append(np.concatenate([
                    rel_pos / 100.0,  # Normalized
                    n_state.velocity / self.max_speed,
                    [n_state.gps_quality],
                    n_state.target[:3] / 100.0
                ]))
                
            # Pad to max neighbors
            while len(neighbor_list) < 5:
                neighbor_list.append(np.zeros(12))
                
            neighbor_states = np.array(neighbor_list)
            
        # Convert to tensors
        self_tensor = torch.FloatTensor(self_state).unsqueeze(0).to(self.device)
        neighbor_tensor = torch.FloatTensor(neighbor_states).unsqueeze(0).to(self.device)
        
        return self_tensor, neighbor_tensor
        
    def control_callback(self):
        """Main control loop - run policy inference."""
        # Prepare inputs
        self_state, neighbor_states = self.prepare_state_tensor()
        
        # Run inference
        with torch.no_grad():
            action, _ = self.policy(self_state, neighbor_states)
            action = action.cpu().numpy().flatten()
            
        # Scale action to velocity command
        velocity = action * self.max_speed
        
        # Publish velocity command
        cmd = TwistStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = f'drone_{self.drone_id}'
        cmd.twist.linear.x = float(velocity[0])
        cmd.twist.linear.y = float(velocity[1])
        cmd.twist.linear.z = float(velocity[2])
        self.cmd_pub.publish(cmd)
        
    def share_state_callback(self):
        """Broadcast state to other drones."""
        with self.state_lock:
            state_data = np.concatenate([
                self.self_state.position,
                self.self_state.velocity,
                [self.self_state.gps_quality],
                self.self_state.target
            ])
            
        msg = Float32MultiArray()
        msg.data = state_data.tolist()
        self.shared_state_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = CIGRLPolicyNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
