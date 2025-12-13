"""
CIGRL Communication Interface
Handles inter-drone communication with attention-based message prioritization.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import Float32MultiArray, String
from geometry_msgs.msg import Point

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import threading
import json


@dataclass
class DroneMessage:
    """Message format for inter-drone communication."""
    sender_id: int
    timestamp: float
    position: np.ndarray
    velocity: np.ndarray
    gps_quality: float
    target: np.ndarray
    priority: float = 1.0
    message_type: str = "state"


@dataclass
class CommunicationStats:
    """Communication statistics for monitoring."""
    packets_sent: int = 0
    packets_received: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    dropped_packets: int = 0
    avg_latency_ms: float = 0.0


class MessageQueue:
    """Priority queue for outgoing messages."""
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.queue: List[DroneMessage] = []
        self.lock = threading.Lock()
        
    def enqueue(self, msg: DroneMessage):
        """Add message to queue with priority ordering."""
        with self.lock:
            self.queue.append(msg)
            self.queue.sort(key=lambda m: -m.priority)  # Higher priority first
            
            if len(self.queue) > self.max_size:
                self.queue = self.queue[:self.max_size]
                
    def dequeue(self) -> Optional[DroneMessage]:
        """Get highest priority message."""
        with self.lock:
            if self.queue:
                return self.queue.pop(0)
            return None
            
    def size(self) -> int:
        with self.lock:
            return len(self.queue)


class NeighborTracker:
    """Tracks neighbor drone states with timeout."""
    
    def __init__(self, timeout: float = 2.0):
        self.timeout = timeout
        self.neighbors: Dict[int, DroneMessage] = {}
        self.last_update: Dict[int, float] = {}
        self.lock = threading.Lock()
        
    def update(self, msg: DroneMessage, current_time: float):
        """Update neighbor state."""
        with self.lock:
            self.neighbors[msg.sender_id] = msg
            self.last_update[msg.sender_id] = current_time
            
    def get_active_neighbors(self, current_time: float) -> List[DroneMessage]:
        """Get neighbors that have been updated recently."""
        with self.lock:
            active = []
            expired = []
            
            for n_id, last_time in self.last_update.items():
                if current_time - last_time < self.timeout:
                    active.append(self.neighbors[n_id])
                else:
                    expired.append(n_id)
                    
            # Remove expired neighbors
            for n_id in expired:
                del self.neighbors[n_id]
                del self.last_update[n_id]
                
            return active
            
    def get_neighbor_count(self) -> int:
        with self.lock:
            return len(self.neighbors)


class CommunicationNode(Node):
    """ROS2 node for inter-drone communication."""
    
    def __init__(self):
        super().__init__('communication_node')
        
        # Parameters
        self.declare_parameter('drone_id', 0)
        self.declare_parameter('num_drones', 5)
        self.declare_parameter('comm_range', 150.0)
        self.declare_parameter('broadcast_rate', 10.0)
        self.declare_parameter('neighbor_timeout', 2.0)
        
        self.drone_id = self.get_parameter('drone_id').value
        self.num_drones = self.get_parameter('num_drones').value
        self.comm_range = self.get_parameter('comm_range').value
        broadcast_rate = self.get_parameter('broadcast_rate').value
        neighbor_timeout = self.get_parameter('neighbor_timeout').value
        
        # State
        self.my_position = np.zeros(3)
        self.my_velocity = np.zeros(3)
        self.my_gps_quality = 0.5
        self.my_target = np.zeros(3)
        self.state_lock = threading.Lock()
        
        # Communication components
        self.neighbor_tracker = NeighborTracker(neighbor_timeout)
        self.outbox = MessageQueue()
        self.stats = CommunicationStats()
        
        # QoS for reliable multicast
        qos_reliable = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )
        
        qos_best_effort = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT
        )
        
        # Publishers
        self.broadcast_pub = self.create_publisher(
            Float32MultiArray, 'drone_broadcast', qos_best_effort)
        self.stats_pub = self.create_publisher(
            String, 'comm_stats', qos_reliable)
        self.neighbors_pub = self.create_publisher(
            Float32MultiArray, 'active_neighbors', qos_reliable)
            
        # Subscribers for own state
        self.pose_sub = self.create_subscription(
            Float32MultiArray, 'pose_fused', self.pose_callback, 10)
        self.velocity_sub = self.create_subscription(
            Float32MultiArray, 'velocity', self.velocity_callback, 10)
        self.gps_quality_sub = self.create_subscription(
            Float32MultiArray, 'localization_confidence', self.gps_callback, 10)
        self.target_sub = self.create_subscription(
            Point, 'target', self.target_callback, 10)
            
        # Subscriber for other drones
        for i in range(self.num_drones):
            if i != self.drone_id:
                self.create_subscription(
                    Float32MultiArray,
                    f'/drone_{i}/drone_broadcast',
                    lambda msg, sender=i: self.receive_callback(msg, sender),
                    qos_best_effort
                )
                
        # Timers
        self.broadcast_timer = self.create_timer(
            1.0 / broadcast_rate, self.broadcast_callback)
        self.stats_timer = self.create_timer(1.0, self.stats_callback)
        
        self.get_logger().info(f'Communication Node initialized for drone {self.drone_id}')
        
    def pose_callback(self, msg: Float32MultiArray):
        """Update own position."""
        if len(msg.data) >= 3:
            with self.state_lock:
                self.my_position = np.array(msg.data[:3])
                
    def velocity_callback(self, msg: Float32MultiArray):
        """Update own velocity."""
        if len(msg.data) >= 3:
            with self.state_lock:
                self.my_velocity = np.array(msg.data[:3])
                
    def gps_callback(self, msg: Float32MultiArray):
        """Update GPS quality."""
        if len(msg.data) >= 1:
            with self.state_lock:
                self.my_gps_quality = msg.data[0]
                
    def target_callback(self, msg: Point):
        """Update target."""
        with self.state_lock:
            self.my_target = np.array([msg.x, msg.y, msg.z])
            
    def receive_callback(self, msg: Float32MultiArray, sender_id: int):
        """Handle received message from another drone."""
        current_time = self.get_clock().now().nanoseconds / 1e9
        
        if len(msg.data) < 11:
            return
            
        # Check if within communication range
        with self.state_lock:
            sender_pos = np.array(msg.data[1:4])
            distance = np.linalg.norm(sender_pos - self.my_position)
            
        if distance > self.comm_range:
            self.stats.dropped_packets += 1
            return
            
        # Parse message
        drone_msg = DroneMessage(
            sender_id=sender_id,
            timestamp=msg.data[0],
            position=np.array(msg.data[1:4]),
            velocity=np.array(msg.data[4:7]),
            gps_quality=msg.data[7],
            target=np.array(msg.data[8:11])
        )
        
        # Update neighbor tracker
        self.neighbor_tracker.update(drone_msg, current_time)
        
        # Stats
        self.stats.packets_received += 1
        self.stats.bytes_received += len(msg.data) * 4  # 4 bytes per float
        
        # Compute latency
        latency = (current_time - drone_msg.timestamp) * 1000
        self.stats.avg_latency_ms = 0.9 * self.stats.avg_latency_ms + 0.1 * latency
        
    def broadcast_callback(self):
        """Broadcast own state to other drones."""
        current_time = self.get_clock().now().nanoseconds / 1e9
        
        with self.state_lock:
            msg = Float32MultiArray()
            msg.data = [
                current_time,
                float(self.my_position[0]),
                float(self.my_position[1]),
                float(self.my_position[2]),
                float(self.my_velocity[0]),
                float(self.my_velocity[1]),
                float(self.my_velocity[2]),
                float(self.my_gps_quality),
                float(self.my_target[0]),
                float(self.my_target[1]),
                float(self.my_target[2])
            ]
            
        self.broadcast_pub.publish(msg)
        
        # Stats
        self.stats.packets_sent += 1
        self.stats.bytes_sent += len(msg.data) * 4
        
        # Publish active neighbors
        neighbors = self.neighbor_tracker.get_active_neighbors(current_time)
        neighbor_msg = Float32MultiArray()
        
        for n in neighbors:
            neighbor_msg.data.extend([
                float(n.sender_id),
                n.position[0], n.position[1], n.position[2],
                n.velocity[0], n.velocity[1], n.velocity[2],
                n.gps_quality
            ])
            
        self.neighbors_pub.publish(neighbor_msg)
        
    def stats_callback(self):
        """Publish communication statistics."""
        stats_dict = {
            'drone_id': self.drone_id,
            'packets_sent': self.stats.packets_sent,
            'packets_received': self.stats.packets_received,
            'bytes_sent': self.stats.bytes_sent,
            'bytes_received': self.stats.bytes_received,
            'dropped_packets': self.stats.dropped_packets,
            'avg_latency_ms': round(self.stats.avg_latency_ms, 2),
            'active_neighbors': self.neighbor_tracker.get_neighbor_count()
        }
        
        msg = String()
        msg.data = json.dumps(stats_dict)
        self.stats_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = CommunicationNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
