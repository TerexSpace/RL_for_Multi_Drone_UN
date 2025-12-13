"""
CIGRL Mission Planner
High-level mission planning and task allocation for multi-drone swarms.
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from geometry_msgs.msg import Point, PoseStamped
from std_msgs.msg import String, Int32, Float32MultiArray
from nav_msgs.msg import Path

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import threading
import heapq


class MissionState(Enum):
    """Mission execution states."""
    IDLE = "idle"
    PLANNING = "planning"
    EXECUTING = "executing"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskType(Enum):
    """Types of mission tasks."""
    GOTO = "goto"
    HOVER = "hover"
    SURVEY = "survey"
    DELIVERY = "delivery"
    FORMATION = "formation"
    RETURN_HOME = "return_home"


@dataclass
class Task:
    """Individual mission task."""
    task_id: str
    task_type: TaskType
    target: np.ndarray
    priority: int = 1
    deadline: Optional[float] = None
    assigned_drone: int = -1
    status: str = "pending"
    completion_time: float = 0.0
    params: Dict = field(default_factory=dict)


@dataclass
class DroneStatus:
    """Current status of a drone."""
    drone_id: int
    position: np.ndarray
    velocity: np.ndarray
    battery: float = 1.0
    is_available: bool = True
    current_task: Optional[str] = None


class TaskAllocator:
    """Optimal task allocation using Hungarian algorithm approximation."""
    
    def __init__(self):
        pass
        
    def compute_cost_matrix(self, drones: List[DroneStatus], 
                           tasks: List[Task]) -> np.ndarray:
        """Compute cost matrix for task allocation."""
        n_drones = len(drones)
        n_tasks = len(tasks)
        
        cost = np.full((n_drones, n_tasks), np.inf)
        
        for i, drone in enumerate(drones):
            if not drone.is_available:
                continue
                
            for j, task in enumerate(tasks):
                if task.status != "pending":
                    continue
                    
                # Distance cost
                dist = np.linalg.norm(task.target - drone.position)
                
                # Priority bonus (lower cost for higher priority)
                priority_factor = 1.0 / (task.priority + 1)
                
                # Battery consideration
                battery_factor = 1.0 / (drone.battery + 0.1)
                
                # Deadline urgency
                deadline_factor = 1.0
                if task.deadline:
                    time_to_deadline = task.deadline
                    if time_to_deadline < dist / 5.0:  # Might not make it
                        deadline_factor = 10.0
                        
                cost[i, j] = dist * priority_factor * battery_factor * deadline_factor
                
        return cost
    
    def allocate(self, drones: List[DroneStatus], 
                tasks: List[Task]) -> Dict[int, str]:
        """Greedy task allocation."""
        cost = self.compute_cost_matrix(drones, tasks)
        
        allocations = {}
        assigned_tasks = set()
        
        # Greedy assignment by minimum cost
        for _ in range(min(len(drones), len(tasks))):
            min_cost = np.inf
            best_drone, best_task = -1, -1
            
            for i in range(len(drones)):
                if i in allocations:
                    continue
                for j in range(len(tasks)):
                    if j in assigned_tasks:
                        continue
                    if cost[i, j] < min_cost:
                        min_cost = cost[i, j]
                        best_drone, best_task = i, j
                        
            if best_drone >= 0 and best_task >= 0 and min_cost < np.inf:
                allocations[drones[best_drone].drone_id] = tasks[best_task].task_id
                assigned_tasks.add(best_task)
                
        return allocations


class PathPlanner:
    """Simple A*-based path planner."""
    
    def __init__(self, grid_size: float = 1000.0, resolution: float = 10.0):
        self.grid_size = grid_size
        self.resolution = resolution
        self.obstacles: List[Tuple[np.ndarray, float]] = []  # (center, radius)
        
    def add_obstacle(self, center: np.ndarray, radius: float):
        """Add obstacle to the environment."""
        self.obstacles.append((center, radius))
        
    def is_valid(self, pos: np.ndarray) -> bool:
        """Check if position is valid (not in obstacle)."""
        for center, radius in self.obstacles:
            if np.linalg.norm(pos[:2] - center[:2]) < radius:
                return False
        return True
    
    def plan_path(self, start: np.ndarray, goal: np.ndarray) -> List[np.ndarray]:
        """Plan path from start to goal."""
        # Simple straight-line path with obstacle avoidance
        path = [start.copy()]
        
        direction = goal - start
        distance = np.linalg.norm(direction)
        
        if distance < self.resolution:
            path.append(goal.copy())
            return path
            
        direction = direction / distance
        num_steps = int(distance / self.resolution)
        
        for i in range(1, num_steps):
            point = start + direction * self.resolution * i
            
            if not self.is_valid(point):
                # Simple obstacle avoidance - go around
                perpendicular = np.array([-direction[1], direction[0], 0])
                point = point + perpendicular * 20.0
                
            path.append(point)
            
        path.append(goal.copy())
        return path


class MissionPlannerNode(Node):
    """ROS2 node for high-level mission planning."""
    
    def __init__(self):
        super().__init__('mission_planner_node')
        
        # Parameters
        self.declare_parameter('num_drones', 5)
        self.declare_parameter('planning_rate', 2.0)
        
        self.num_drones = self.get_parameter('num_drones').value
        planning_rate = self.get_parameter('planning_rate').value
        
        # State
        self.mission_state = MissionState.IDLE
        self.tasks: Dict[str, Task] = {}
        self.drone_status: Dict[int, DroneStatus] = {}
        self.task_allocator = TaskAllocator()
        self.path_planner = PathPlanner()
        self.state_lock = threading.Lock()
        
        # Initialize drone status
        for i in range(self.num_drones):
            self.drone_status[i] = DroneStatus(
                drone_id=i,
                position=np.zeros(3),
                velocity=np.zeros(3)
            )
            
        # Publishers
        self.mission_state_pub = self.create_publisher(
            String, 'mission_state', 10)
        self.task_allocation_pub = self.create_publisher(
            String, 'task_allocations', 10)
            
        # Per-drone publishers
        self.target_pubs = {}
        self.path_pubs = {}
        for i in range(self.num_drones):
            self.target_pubs[i] = self.create_publisher(
                Point, f'/drone_{i}/target', 10)
            self.path_pubs[i] = self.create_publisher(
                Path, f'/drone_{i}/planned_path', 10)
                
        # Subscribers
        for i in range(self.num_drones):
            self.create_subscription(
                Float32MultiArray,
                f'/drone_{i}/pose_fused',
                lambda msg, drone_id=i: self.drone_pose_callback(msg, drone_id),
                10
            )
            
        self.mission_sub = self.create_subscription(
            String, 'mission_command', self.mission_command_callback, 10)
            
        # Timer
        self.planning_timer = self.create_timer(
            1.0 / planning_rate, self.planning_callback)
            
        self.get_logger().info('Mission Planner Node initialized')
        
    def drone_pose_callback(self, msg: Float32MultiArray, drone_id: int):
        """Update drone position."""
        if len(msg.data) >= 3:
            with self.state_lock:
                self.drone_status[drone_id].position = np.array(msg.data[:3])
                if len(msg.data) >= 6:
                    self.drone_status[drone_id].velocity = np.array(msg.data[3:6])
                    
    def mission_command_callback(self, msg: String):
        """Handle mission commands."""
        try:
            cmd = json.loads(msg.data)
            command = cmd.get('command', '')
            
            if command == 'start':
                self.start_mission(cmd.get('tasks', []))
            elif command == 'pause':
                self.mission_state = MissionState.PAUSED
            elif command == 'resume':
                self.mission_state = MissionState.EXECUTING
            elif command == 'abort':
                self.abort_mission()
            elif command == 'add_task':
                self.add_task(cmd.get('task', {}))
                
        except json.JSONDecodeError:
            self.get_logger().error('Invalid mission command format')
            
    def start_mission(self, tasks: List[Dict]):
        """Start a new mission with given tasks."""
        with self.state_lock:
            self.tasks.clear()
            
            for t in tasks:
                task = Task(
                    task_id=t.get('id', f'task_{len(self.tasks)}'),
                    task_type=TaskType(t.get('type', 'goto')),
                    target=np.array(t.get('target', [0, 0, 30])),
                    priority=t.get('priority', 1),
                    deadline=t.get('deadline')
                )
                self.tasks[task.task_id] = task
                
            self.mission_state = MissionState.PLANNING
            
        self.get_logger().info(f'Started mission with {len(tasks)} tasks')
        
    def add_task(self, task_dict: Dict):
        """Add a single task to the mission."""
        with self.state_lock:
            task = Task(
                task_id=task_dict.get('id', f'task_{len(self.tasks)}'),
                task_type=TaskType(task_dict.get('type', 'goto')),
                target=np.array(task_dict.get('target', [0, 0, 30])),
                priority=task_dict.get('priority', 1),
                deadline=task_dict.get('deadline')
            )
            self.tasks[task.task_id] = task
            
    def abort_mission(self):
        """Abort current mission and return to home."""
        with self.state_lock:
            self.mission_state = MissionState.FAILED
            
            # Send all drones to home
            for i in range(self.num_drones):
                home = Point(x=50.0 + i * 10, y=50.0, z=30.0)
                self.target_pubs[i].publish(home)
                
    def planning_callback(self):
        """Main planning loop."""
        if self.mission_state == MissionState.IDLE:
            return
            
        if self.mission_state == MissionState.PAUSED:
            return
            
        with self.state_lock:
            # Get pending tasks
            pending_tasks = [t for t in self.tasks.values() if t.status == "pending"]
            
            # Get available drones
            available_drones = [d for d in self.drone_status.values() if d.is_available]
            
            if not pending_tasks:
                # Check if all tasks complete
                if all(t.status == "completed" for t in self.tasks.values()):
                    self.mission_state = MissionState.COMPLETED
                return
                
            # Allocate tasks
            allocations = self.task_allocator.allocate(available_drones, pending_tasks)
            
            # Execute allocations
            for drone_id, task_id in allocations.items():
                task = self.tasks[task_id]
                drone = self.drone_status[drone_id]
                
                task.assigned_drone = drone_id
                task.status = "executing"
                drone.is_available = False
                drone.current_task = task_id
                
                # Plan path
                path = self.path_planner.plan_path(drone.position, task.target)
                
                # Publish target
                target_msg = Point(
                    x=float(task.target[0]),
                    y=float(task.target[1]),
                    z=float(task.target[2])
                )
                self.target_pubs[drone_id].publish(target_msg)
                
                # Publish path
                path_msg = Path()
                path_msg.header.stamp = self.get_clock().now().to_msg()
                path_msg.header.frame_id = 'map'
                
                for p in path:
                    pose = PoseStamped()
                    pose.pose.position.x = float(p[0])
                    pose.pose.position.y = float(p[1])
                    pose.pose.position.z = float(p[2])
                    path_msg.poses.append(pose)
                    
                self.path_pubs[drone_id].publish(path_msg)
                
            # Check task completion
            for drone in self.drone_status.values():
                if drone.current_task:
                    task = self.tasks.get(drone.current_task)
                    if task and np.linalg.norm(drone.position - task.target) < 5.0:
                        task.status = "completed"
                        drone.is_available = True
                        drone.current_task = None
                        
            # Update mission state
            if self.mission_state == MissionState.PLANNING:
                self.mission_state = MissionState.EXECUTING
                
        # Publish state
        state_msg = String()
        state_msg.data = json.dumps({
            'state': self.mission_state.value,
            'active_tasks': sum(1 for t in self.tasks.values() if t.status == "executing"),
            'completed_tasks': sum(1 for t in self.tasks.values() if t.status == "completed"),
            'pending_tasks': sum(1 for t in self.tasks.values() if t.status == "pending")
        })
        self.mission_state_pub.publish(state_msg)


def main(args=None):
    rclpy.init(args=args)
    node = MissionPlannerNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
