"""
Urban Delivery Case Study
Demonstrates CIGRL's coordinated delivery with route deconfliction in GPS-challenged environments.
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Dict
import json
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Local Drone class to avoid import issues


@dataclass
class DeliveryTask:
    """Individual delivery task."""
    id: str
    pickup: np.ndarray
    dropoff: np.ndarray
    priority: str
    deadline_s: float
    assigned_drone: int = -1
    status: str = "pending"
    completion_time: float = 0.0


class DeliveryCaseStudy:
    """
    Case Study: Urban Delivery with Route Deconfliction
    
    Demonstrates:
    - Multi-drone coordinated delivery
    - Dynamic route deconfliction
    - GPS-denied zone navigation
    - Priority-based task assignment
    """
    
    def __init__(self, num_drones: int = 5):
        self.num_drones = num_drones
        self.depot = np.array([50, 50, 20])
        self.deliveries: List[DeliveryTask] = []
        self.drones: List[Drone] = []
        
        # GPS-denied zones (urban canyons)
        self.gps_denied_zones = [
            ((200, 200), (400, 400), 0.15),  # (start, end, quality)
            ((500, 100), (700, 300), 0.2),
            ((100, 500), (300, 700), 0.1),
        ]
        
    def setup_scenario(self):
        """Initialize delivery scenario."""
        # Create delivery tasks
        np.random.seed(42)
        
        delivery_points = [
            ("D1", [800, 700, 30], "high", 200),
            ("D2", [600, 200, 25], "medium", 300),
            ("D3", [300, 800, 35], "medium", 280),
            ("D4", [900, 400, 40], "low", 400),
            ("D5", [250, 350, 30], "high", 180),  # In GPS-denied zone
        ]
        
        for task_id, dropoff, priority, deadline in delivery_points:
            self.deliveries.append(DeliveryTask(
                id=task_id,
                pickup=self.depot.copy(),
                dropoff=np.array(dropoff, dtype=float),
                priority=priority,
                deadline_s=deadline
            ))
            
        # Initialize drones at depot
        for i in range(self.num_drones):
            offset = np.array([i * 5, 0, i * 2])
            self.drones.append(Drone(
                id=i,
                pos=self.depot + offset,
                target=self.depot.copy()
            ))
            
    def assign_tasks(self):
        """Priority-based task assignment."""
        # Sort by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        sorted_tasks = sorted(self.deliveries, 
                             key=lambda t: (priority_order[t.priority], t.deadline_s))
        
        available_drones = list(range(self.num_drones))
        
        for task in sorted_tasks:
            if not available_drones:
                break
            
            # Assign closest available drone
            best_drone = min(available_drones,
                           key=lambda d: np.linalg.norm(self.drones[d].pos - task.dropoff))
            
            task.assigned_drone = best_drone
            task.status = "assigned"
            self.drones[best_drone].target = task.dropoff.copy()
            available_drones.remove(best_drone)
            
    def check_route_conflicts(self) -> List[Tuple[int, int, float]]:
        """Detect potential route conflicts between drones."""
        conflicts = []
        
        for i, d1 in enumerate(self.drones):
            for j, d2 in enumerate(self.drones[i+1:], i+1):
                # Check if paths intersect
                # Simplified: check if targets are close
                target_dist = np.linalg.norm(d1.target - d2.target)
                current_dist = np.linalg.norm(d1.pos - d2.pos)
                
                if target_dist < 50 or current_dist < 20:
                    conflicts.append((i, j, min(target_dist, current_dist)))
                    
        return conflicts
    
    def resolve_conflicts(self, conflicts: List[Tuple[int, int, float]]):
        """Resolve route conflicts through altitude stratification."""
        altitude_layers = [25, 35, 45, 55, 65]
        
        for d1_id, d2_id, _ in conflicts:
            # Assign different altitudes
            layer1 = d1_id % len(altitude_layers)
            layer2 = (d2_id + 1) % len(altitude_layers)
            
            self.drones[d1_id].target[2] = altitude_layers[layer1]
            self.drones[d2_id].target[2] = altitude_layers[layer2]
            
    def get_gps_quality(self, pos: np.ndarray) -> float:
        """Check GPS quality at position."""
        for (start, end, quality) in self.gps_denied_zones:
            if start[0] <= pos[0] <= end[0] and start[1] <= pos[1] <= end[1]:
                return quality
        return 0.9
    
    def run_simulation(self, steps: int = 200, use_cigrl: bool = True) -> Dict:
        """Run delivery simulation."""
        self.setup_scenario()
        self.assign_tasks()
        
        dt = 0.5
        results = {
            'drone_paths': [[] for _ in self.drones],
            'task_completions': [],
            'conflicts_resolved': 0,
            'time_in_gps_denied': [0 for _ in self.drones]
        }
        
        for t in range(steps):
            # Check and resolve conflicts
            conflicts = self.check_route_conflicts()
            if conflicts:
                self.resolve_conflicts(conflicts)
                results['conflicts_resolved'] += len(conflicts)
            
            # Update drone positions
            for d in self.drones:
                gps_q = self.get_gps_quality(d.pos)
                
                if gps_q < 0.5:
                    results['time_in_gps_denied'][d.id] += dt
                
                # Navigation (CIGRL or baseline)
                to_target = d.target - d.pos
                dist = np.linalg.norm(to_target)
                
                if dist > 5:
                    direction = to_target / dist
                    
                    if use_cigrl and gps_q < 0.5:
                        # Cooperative localization - reduced noise
                        noise = np.random.normal(0, 2.0, 3)
                    else:
                        noise = np.random.normal(0, (1.0 - gps_q) * 10.0, 3)
                        
                    speed = 5.0 if gps_q > 0.3 else 2.0
                    d.vel = direction * speed + noise * 0.1
                    d.pos = d.pos + d.vel * dt
                    
                results['drone_paths'][d.id].append(d.pos.copy())
                
                # Check task completion
                for task in self.deliveries:
                    if task.assigned_drone == d.id and task.status == "assigned":
                        if np.linalg.norm(d.pos - task.dropoff) < 10:
                            task.status = "completed"
                            task.completion_time = t * dt
                            results['task_completions'].append({
                                'task_id': task.id,
                                'time': task.completion_time,
                                'deadline': task.deadline_s,
                                'on_time': task.completion_time <= task.deadline_s
                            })
                            
        # Summary metrics
        completed = sum(1 for t in self.deliveries if t.status == "completed")
        on_time = sum(1 for c in results['task_completions'] if c['on_time'])
        
        results['summary'] = {
            'completion_rate': completed / len(self.deliveries),
            'on_time_rate': on_time / max(completed, 1),
            'avg_completion_time': np.mean([c['time'] for c in results['task_completions']]) if results['task_completions'] else 0,
            'total_conflicts_resolved': results['conflicts_resolved'],
            'avg_time_gps_denied': np.mean(results['time_in_gps_denied'])
        }
        
        return results
    
    def visualize_results(self, results: Dict, filename: str = "delivery_case_study.png"):
        """Generate visualization."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Drone paths
        ax = axes[0]
        
        # Draw GPS-denied zones
        for (start, end, _) in self.gps_denied_zones:
            rect = plt.Rectangle(start, end[0]-start[0], end[1]-start[1],
                                color='red', alpha=0.2, label='GPS Denied')
            ax.add_patch(rect)
            
        # Draw paths
        colors = plt.cm.tab10(np.linspace(0, 1, len(results['drone_paths'])))
        for i, (path, color) in enumerate(zip(results['drone_paths'], colors)):
            if path:
                path_arr = np.array(path)
                ax.plot(path_arr[:, 0], path_arr[:, 1], color=color, 
                       linewidth=1.5, label=f'Drone {i}')
                ax.scatter(path_arr[0, 0], path_arr[0, 1], color=color, 
                          marker='o', s=50)
                ax.scatter(path_arr[-1, 0], path_arr[-1, 1], color=color, 
                          marker='x', s=100)
                
        # Draw delivery targets
        for task in self.deliveries:
            ax.scatter(task.dropoff[0], task.dropoff[1], color='green',
                      marker='s', s=100, zorder=10)
            ax.annotate(task.id, (task.dropoff[0] + 10, task.dropoff[1] + 10))
            
        ax.scatter([self.depot[0]], [self.depot[1]], color='blue',
                  marker='^', s=150, label='Depot', zorder=10)
        
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_title('Delivery Route Visualization')
        ax.set_xlim(-50, 1050)
        ax.set_ylim(-50, 1050)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)
        
        # Plot 2: Task completion timeline
        ax2 = axes[1]
        
        tasks_sorted = sorted(results['task_completions'], key=lambda x: x['time'])
        
        for i, task in enumerate(tasks_sorted):
            color = 'green' if task['on_time'] else 'red'
            ax2.barh(i, task['time'], color=color, alpha=0.7)
            ax2.axvline(task['deadline'], color='orange', linestyle='--', alpha=0.5)
            
        ax2.set_yticks(range(len(tasks_sorted)))
        ax2.set_yticklabels([t['task_id'] for t in tasks_sorted])
        ax2.set_xlabel('Time (seconds)')
        ax2.set_title('Delivery Completion Timeline')
        ax2.grid(True, axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        return filename


def run_case_study():
    """Execute the delivery case study."""
    print("=" * 60)
    print("CASE STUDY: Urban Delivery with Route Deconfliction")
    print("=" * 60)
    
    study = DeliveryCaseStudy(num_drones=5)
    
    # Run with CIGRL
    print("\nRunning CIGRL simulation...")
    cigrl_results = study.run_simulation(steps=200, use_cigrl=True)
    
    # Run baseline
    print("Running baseline simulation...")
    study_baseline = DeliveryCaseStudy(num_drones=5)
    baseline_results = study_baseline.run_simulation(steps=200, use_cigrl=False)
    
    # Compare results
    print("\n" + "=" * 60)
    print("RESULTS COMPARISON")
    print("=" * 60)
    
    print(f"\n{'Metric':<30} {'Baseline':<15} {'CIGRL':<15}")
    print("-" * 60)
    
    metrics = [
        ('completion_rate', 'Completion Rate'),
        ('on_time_rate', 'On-Time Rate'),
        ('avg_completion_time', 'Avg. Completion Time (s)'),
        ('avg_time_gps_denied', 'Avg. Time in GPS-Denied (s)')
    ]
    
    for key, name in metrics:
        base_val = baseline_results['summary'][key]
        cigrl_val = cigrl_results['summary'][key]
        if 'rate' in key.lower():
            print(f"{name:<30} {base_val:.1%}{'':<10} {cigrl_val:.1%}")
        else:
            print(f"{name:<30} {base_val:.1f}{'':<13} {cigrl_val:.1f}")
    
    # Visualize
    print("\nGenerating visualization...")
    study.visualize_results(cigrl_results, "case_studies/urban_delivery/results/delivery_case_study.png")
    
    # Save results
    with open("case_studies/urban_delivery/results/delivery_results.json", "w") as f:
        json.dump({
            'cigrl': cigrl_results['summary'],
            'baseline': baseline_results['summary']
        }, f, indent=2)
        
    print("\nCase study complete!")
    print("Results saved to case_studies/urban_delivery/results/")
    
    return cigrl_results, baseline_results


if __name__ == "__main__":
    run_case_study()
