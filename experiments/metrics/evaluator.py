"""
Evaluation Metrics Module
Comprehensive metrics computation for CIGRL evaluation.
"""

import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional
import json
import matplotlib.pyplot as plt
from pathlib import Path


@dataclass
class TrajectoryMetrics:
    """Metrics related to trajectory quality."""
    path_length: float = 0.0
    optimal_length: float = 0.0
    deviation_ratio: float = 0.0
    smoothness: float = 0.0  # Average acceleration magnitude
    jerk: float = 0.0  # Average jerk magnitude


@dataclass
class LocalizationMetrics:
    """Metrics for localization performance."""
    avg_position_error: float = 0.0
    max_position_error: float = 0.0
    drift_rate: float = 0.0  # meters per second
    gps_availability: float = 0.0  # Fraction of time with good GPS


@dataclass
class SafetyMetrics:
    """Safety-related metrics."""
    collision_count: int = 0
    near_miss_count: int = 0
    min_separation: float = float('inf')
    geofence_violations: int = 0


@dataclass
class CommunicationMetrics:
    """Communication efficiency metrics."""
    total_packets: int = 0
    total_bytes: int = 0
    avg_latency_ms: float = 0.0
    packet_loss_rate: float = 0.0
    bandwidth_utilization: float = 0.0


@dataclass
class MissionMetrics:
    """Mission-level metrics."""
    completion_rate: float = 0.0
    avg_time_to_completion: float = 0.0
    success_count: int = 0
    failure_count: int = 0
    timeout_count: int = 0


@dataclass
class ComprehensiveMetrics:
    """All metrics combined."""
    trajectory: TrajectoryMetrics = field(default_factory=TrajectoryMetrics)
    localization: LocalizationMetrics = field(default_factory=LocalizationMetrics)
    safety: SafetyMetrics = field(default_factory=SafetyMetrics)
    communication: CommunicationMetrics = field(default_factory=CommunicationMetrics)  
    mission: MissionMetrics = field(default_factory=MissionMetrics)
    total_reward: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            'trajectory': asdict(self.trajectory),
            'localization': asdict(self.localization),
            'safety': asdict(self.safety),
            'communication': asdict(self.communication),
            'mission': asdict(self.mission),
            'total_reward': self.total_reward
        }


class MetricsEvaluator:
    """Evaluates simulation runs and computes comprehensive metrics."""
    
    def __init__(self, dt: float = 0.5):
        self.dt = dt
        
    def compute_trajectory_metrics(self, paths: List[List[np.ndarray]], 
                                   targets: List[np.ndarray]) -> TrajectoryMetrics:
        """Compute trajectory quality metrics."""
        metrics = TrajectoryMetrics()
        
        path_lengths = []
        optimal_lengths = []
        smoothness_values = []
        
        for path, target in zip(paths, targets):
            if len(path) < 2:
                continue
                
            # Path length
            length = sum(np.linalg.norm(path[i+1] - path[i]) 
                        for i in range(len(path) - 1))
            path_lengths.append(length)
            
            # Optimal length (straight line)
            optimal = np.linalg.norm(target - path[0])
            optimal_lengths.append(optimal)
            
            # Smoothness (velocity changes)
            if len(path) >= 3:
                velocities = [(path[i+1] - path[i]) / self.dt 
                             for i in range(len(path) - 1)]
                accelerations = [(velocities[i+1] - velocities[i]) / self.dt 
                                for i in range(len(velocities) - 1)]
                smoothness = np.mean([np.linalg.norm(a) for a in accelerations])
                smoothness_values.append(smoothness)
                
        if path_lengths:
            metrics.path_length = np.mean(path_lengths)
            metrics.optimal_length = np.mean(optimal_lengths)
            
            if metrics.optimal_length > 0:
                metrics.deviation_ratio = (metrics.path_length - metrics.optimal_length) / metrics.optimal_length
                
        if smoothness_values:
            metrics.smoothness = np.mean(smoothness_values)
            
        return metrics
    
    def compute_localization_metrics(self, 
                                     true_positions: List[List[np.ndarray]],
                                     estimated_positions: List[List[np.ndarray]],
                                     gps_qualities: List[List[float]]) -> LocalizationMetrics:
        """Compute localization performance metrics."""
        metrics = LocalizationMetrics()
        
        all_errors = []
        drift_rates = []
        good_gps_fractions = []
        
        for true_path, est_path, gps_q in zip(true_positions, estimated_positions, gps_qualities):
            if len(true_path) != len(est_path):
                continue
                
            # Position errors
            errors = [np.linalg.norm(t - e) for t, e in zip(true_path, est_path)]
            all_errors.extend(errors)
            
            # Drift rate (error accumulation rate)
            if len(errors) > 1:
                drift = (errors[-1] - errors[0]) / (len(errors) * self.dt)
                drift_rates.append(max(0, drift))
                
            # GPS availability
            good_gps = sum(1 for q in gps_q if q > 0.5) / len(gps_q) if gps_q else 0
            good_gps_fractions.append(good_gps)
            
        if all_errors:
            metrics.avg_position_error = np.mean(all_errors)
            metrics.max_position_error = np.max(all_errors)
            
        if drift_rates:
            metrics.drift_rate = np.mean(drift_rates)
            
        if good_gps_fractions:
            metrics.gps_availability = np.mean(good_gps_fractions)
            
        return metrics
    
    def compute_safety_metrics(self, paths: List[List[np.ndarray]],
                               collision_threshold: float = 2.0,
                               near_miss_threshold: float = 5.0) -> SafetyMetrics:
        """Compute safety metrics from trajectory data."""
        metrics = SafetyMetrics()
        min_sep = float('inf')
        
        n_drones = len(paths)
        n_steps = len(paths[0]) if paths else 0
        
        for t in range(n_steps):
            for i in range(n_drones):
                for j in range(i + 1, n_drones):
                    if t < len(paths[i]) and t < len(paths[j]):
                        dist = np.linalg.norm(paths[i][t] - paths[j][t])
                        min_sep = min(min_sep, dist)
                        
                        if dist < collision_threshold:
                            metrics.collision_count += 1
                        elif dist < near_miss_threshold:
                            metrics.near_miss_count += 1
                            
        metrics.min_separation = min_sep if min_sep != float('inf') else 0.0
        return metrics
    
    def compute_communication_metrics(self, 
                                       packets: List[int],
                                       bytes_sent: List[int],
                                       latencies: List[float] = None) -> CommunicationMetrics:
        """Compute communication efficiency metrics."""
        metrics = CommunicationMetrics()
        
        metrics.total_packets = sum(packets)
        metrics.total_bytes = sum(bytes_sent)
        
        if latencies:
            metrics.avg_latency_ms = np.mean(latencies)
            
        return metrics
    
    def compute_mission_metrics(self, 
                                 final_positions: List[np.ndarray],
                                 targets: List[np.ndarray],
                                 completion_threshold: float = 10.0,
                                 completion_times: List[float] = None) -> MissionMetrics:
        """Compute mission-level metrics."""
        metrics = MissionMetrics()
        
        successes = []
        for pos, target in zip(final_positions, targets):
            dist = np.linalg.norm(pos - target)
            successes.append(dist < completion_threshold)
            
        metrics.success_count = sum(successes)
        metrics.failure_count = len(successes) - metrics.success_count
        metrics.completion_rate = metrics.success_count / len(successes) if successes else 0.0
        
        if completion_times:
            valid_times = [t for t, s in zip(completion_times, successes) if s and t > 0]
            if valid_times:
                metrics.avg_time_to_completion = np.mean(valid_times)
                
        return metrics


class ResultsVisualizer:
    """Generates publication-quality visualizations."""
    
    def __init__(self, output_dir: str = "experiments/results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Publication style
        plt.rcParams.update({
            'font.size': 12,
            'font.family': 'serif',
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'legend.fontsize': 11,
            'figure.figsize': (10, 6),
            'figure.dpi': 150
        })
        
    def plot_trajectory_comparison(self, 
                                    baseline_paths: List[List[np.ndarray]],
                                    cigrl_paths: List[List[np.ndarray]],
                                    targets: List[np.ndarray],
                                    gps_zones: List[Tuple] = None,
                                    filename: str = "trajectory_comparison.png"):
        """Plot trajectory comparison between baseline and CIGRL."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        for ax, paths, title in zip(axes, [baseline_paths, cigrl_paths], 
                                     ['Baseline (A* + Potential Fields)', 'CIGRL (Proposed)']):
            # Draw GPS-denied zones
            if gps_zones:
                for zone in gps_zones:
                    rect = plt.Rectangle((zone[0], zone[1]), 
                                         zone[2] - zone[0], zone[3] - zone[1],
                                         color='gray', alpha=0.3)
                    ax.add_patch(rect)
                    
            # Draw paths
            colors = plt.cm.viridis(np.linspace(0, 1, len(paths)))
            for path, color in zip(paths, colors):
                path_arr = np.array(path)
                ax.plot(path_arr[:, 0], path_arr[:, 1], color=color, alpha=0.7, linewidth=1.5)
                ax.scatter(path_arr[0, 0], path_arr[0, 1], color='green', s=50, marker='o', zorder=5)
                
            # Draw targets
            for target in targets:
                ax.scatter(target[0], target[1], color='red', s=100, marker='x', zorder=5)
                
            ax.set_xlabel('X (meters)')
            ax.set_ylabel('Y (meters)')
            ax.set_title(title)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=150, bbox_inches='tight')
        plt.close()
        
    def plot_scalability(self, 
                          swarm_sizes: List[int],
                          baseline_metrics: Dict[int, float],
                          cigrl_metrics: Dict[int, float],
                          metric_name: str = "Mission Completion Rate",
                          filename: str = "scalability.png"):
        """Plot performance vs swarm size."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        baseline_vals = [baseline_metrics.get(s, 0) for s in swarm_sizes]
        cigrl_vals = [cigrl_metrics.get(s, 0) for s in swarm_sizes]
        
        x = np.arange(len(swarm_sizes))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, baseline_vals, width, label='Baseline', color='#E74C3C', alpha=0.8)
        bars2 = ax.bar(x + width/2, cigrl_vals, width, label='CIGRL', color='#27AE60', alpha=0.8)
        
        ax.set_xlabel('Swarm Size (number of drones)')
        ax.set_ylabel(metric_name)
        ax.set_title(f'{metric_name} vs Swarm Size')
        ax.set_xticks(x)
        ax.set_xticklabels(swarm_sizes)
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)
        
        # Add value labels
        for bar in bars1 + bars2:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=150, bbox_inches='tight')
        plt.close()
        
    def plot_ablation_study(self,
                            ablation_results: Dict[str, Dict],
                            filename: str = "ablation_study.png"):
        """Plot ablation study results."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        variants = list(ablation_results.keys())
        
        # Completion rate
        completion_rates = [ablation_results[v].get('completion_rate', 0) for v in variants]
        axes[0].barh(variants, completion_rates, color=['#3498DB', '#E74C3C', '#F39C12', '#27AE60', '#9B59B6'])
        axes[0].set_xlabel('Mission Completion Rate')
        axes[0].set_title('Ablation Study: Mission Success')
        axes[0].set_xlim(0, 1)
        
        # Collision rate per 1000 steps
        collisions = [ablation_results[v].get('collisions', 0) for v in variants]
        axes[1].barh(variants, collisions, color=['#3498DB', '#E74C3C', '#F39C12', '#27AE60', '#9B59B6'])
        axes[1].set_xlabel('Collision Count')
        axes[1].set_title('Ablation Study: Safety')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=150, bbox_inches='tight')
        plt.close()
        
    def generate_summary_table(self, 
                                results: Dict[str, ComprehensiveMetrics],
                                filename: str = "results_table.json"):
        """Generate summary table of results."""
        table_data = {}
        
        for name, metrics in results.items():
            table_data[name] = metrics.to_dict()
            
        with open(self.output_dir / filename, 'w') as f:
            json.dump(table_data, f, indent=2)
            
        return table_data


if __name__ == "__main__":
    print("Testing metrics module...")
    
    # Create sample data
    evaluator = MetricsEvaluator()
    
    # Sample paths
    paths = [
        [np.array([0, 0, 30]) + np.array([i*5, i*5, 0]) for i in range(20)]
        for _ in range(5)
    ]
    targets = [np.array([100, 100, 30]) for _ in range(5)]
    
    # Compute metrics
    traj_metrics = evaluator.compute_trajectory_metrics(paths, targets)
    print(f"Trajectory deviation ratio: {traj_metrics.deviation_ratio:.3f}")
    
    safety_metrics = evaluator.compute_safety_metrics(paths)
    print(f"Collisions: {safety_metrics.collision_count}")
    print(f"Min separation: {safety_metrics.min_separation:.2f}m")
    
    mission_metrics = evaluator.compute_mission_metrics(
        [p[-1] for p in paths], targets
    )
    print(f"Completion rate: {mission_metrics.completion_rate:.1%}")
    
    print("\nMetrics module tests passed!")
