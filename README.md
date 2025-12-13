# CIGRL: Covariance-Integrated Graph Reinforcement Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![ROS2 Humble](https://img.shields.io/badge/ROS2-Humble-blue)](https://docs.ros.org/en/humble/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Tests](https://github.com/TerexSpace/RL_for_Multi_Drone_UN/actions/workflows/ci.yml/badge.svg)](https://github.com/TerexSpace/RL_for_Multi_Drone_UN/actions)

**Multi-Drone Urban Navigation with Attention-based Cooperative Sensing**

CIGRL is a Python framework for coordinated multi-drone navigation in GPS-challenged urban environments. When GPS signals degrade, drones share localization covariance matrices through attention-weighted graph neural networks, enabling robust cooperative position estimation.

## Features

- 🤖 **Multi-Agent PPO** with covariance-weighted attention communication
- 📡 **Adaptive Extended Kalman Filter** with online bias estimation
- 🏙️ **Urban environment simulation** with configurable GPS-denied zones
- 🔄 **ROS2 Humble integration** for real-world deployment

## Installation

### Quick Install (Python only)

```bash
# Clone repository
git clone https://github.com/TerexSpace/RL_for_Multi_Drone_UN.git
cd RL_for_Multi_Drone_UN

# Install package
pip install -e .

# Or with development dependencies
pip install -e ".[dev]"
```

### Docker (Recommended for full stack)

```bash
docker build -t cigrl:latest -f docker/Dockerfile .
docker run -it --gpus all cigrl:latest
```

### ROS2 Installation

```bash
# Prerequisites: ROS2 Humble installed
# See: https://docs.ros.org/en/humble/Installation.html

# Install ROS2 dependencies
cd ros2_ws
rosdep install --from-paths src --ignore-src -r -y

# Build
colcon build --symlink-install
source install/setup.bash
```

## Quick Start

### Minimal Example (No ROS2 Required)

```python
from cigrl import DroneSwarm, UrbanEnvironment, CIGRLPolicy

# Create environment with GPS-denied zones
env = UrbanEnvironment(
    size=1000,  # 1km x 1km
    n_buildings=30,
    gps_denied_zones=[(500, 500, 100)]  # center_x, center_y, radius
)

# Initialize swarm
swarm = DroneSwarm(n_drones=5, env=env)

# Load pretrained policy
policy = CIGRLPolicy.load("models/surveillance_5drones.pt")

# Run episode
obs = swarm.reset()
for step in range(1000):
    actions = policy.select_actions(obs)
    obs, rewards, dones, info = swarm.step(actions)
    if all(dones):
        break

print(f"Mission completion: {info['completion_rate']:.1%}")
```

### ROS2 Simulation

```bash
# Launch 5-drone swarm
ros2 launch cigrl_core cigrl_swarm.launch.py num_drones:=5

# In another terminal: monitor status
ros2 topic echo /swarm/status
```

### Training

```bash
# Train new policy
python -m cigrl.scripts.train --config configs/training.yaml --num-drones 5

# Evaluate trained policy
python -m cigrl.scripts.evaluate --model models/my_policy.pt
```

## Architecture

```
cigrl/
├── cigrl/                    # Python package
│   ├── core/                 # Core algorithms
│   │   ├── attention.py      # Covariance-weighted attention
│   │   ├── policy.py         # MAPPO policy network
│   │   └── sensor_fusion.py  # Adaptive EKF
│   ├── envs/                 # Simulation environments
│   │   ├── urban.py          # Urban canyon simulation
│   │   └── drone.py          # Drone dynamics
│   └── scripts/              # CLI tools
├── ros2_ws/src/cigrl_core/   # ROS2 packages
├── tests/                    # Test suite
├── configs/                  # Configuration files
└── models/                   # Pretrained weights
```

## Key Components

### Covariance-Weighted Attention

When GPS quality degrades, drones share position estimates weighted by localization confidence:

```python
# Attention weights incorporate uncertainty
alpha_ij = softmax(
    dot(W_q @ h_i, W_k @ h_j) / sqrt(d)
    - lambda * trace(Sigma_j)  # Penalize uncertain neighbors
)
```

### Adaptive Extended Kalman Filter

The sensor fusion module monitors innovation statistics and adjusts measurement noise:

```python
from cigrl.core.sensor_fusion import AdaptiveEKF

ekf = AdaptiveEKF(
    state_dim=15,  # pos, vel, attitude, accel_bias, gyro_bias
    gps_noise_nominal=0.5,
    adaptation_rate=0.1
)
```

## Mission Types

| Mission        | Description                    | Command                  |
| -------------- | ------------------------------ | ------------------------ |
| Surveillance   | Sector coverage with handovers | `--mission surveillance` |
| Delivery       | Multi-point routing            | `--mission delivery`     |
| Formation      | V-formation flight             | `--mission formation`    |
| Canyon Transit | GPS-denied corridor            | `--mission canyon`       |

## Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=cigrl --cov-report=html
```

## Citation

If you use CIGRL in your research, please cite:

```bibtex
@article{cigrl2024,
  title={CIGRL: A Python Framework for Multi-Drone Urban Navigation
         with Covariance-Weighted Cooperative Sensing},
  author={Kemel, Ayan and La Spada, Luigi and Ospanov, Almas},
  journal={Journal of Open Source Software},
  year={2024},
  doi={10.21105/joss.XXXXX}
}
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:

- Reporting issues
- Submitting pull requests
- Code style requirements

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- ROS2 Humble Hawksbill
- PyTorch for deep learning
- OpenAI Gym / Gymnasium for RL interfaces
