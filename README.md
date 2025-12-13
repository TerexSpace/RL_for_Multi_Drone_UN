# CIGRL: Covariance-driven Intelligent Graph Reinforcement Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![ROS2 Humble](https://img.shields.io/badge/ROS2-Humble-blue)](https://docs.ros.org/en/humble/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)

**Multi-Drone Urban Navigation with Attention-based Cooperative Sensing**

CIGRL is a reinforcement learning framework for coordinated multi-drone navigation in GPS-challenged urban environments. It features:

- 🤖 **Multi-Agent PPO** with attention-based communication
- 📡 **Cooperative localization** in GPS-denied zones
- 🏙️ **Urban environment** simulation with GNSS degradation
- 🔄 **ROS2 integration** for real-world deployment

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/<user>/CIGRL.git
cd CIGRL

# Docker (recommended)
docker build -t cigrl:latest -f docker/Dockerfile .
docker run -it --gpus all cigrl:latest

# Or local installation
pip install -r requirements.txt
cd ros2_ws && colcon build
```

### Run Simulation

```bash
# Python simulation
python experiments/cigrl_enhanced_simulation.py

# ROS2 launch
source ros2_ws/install/setup.bash
ros2 launch cigrl_core cigrl_swarm.launch.py num_drones:=5
```

### Training

```bash
python scripts/train.py --config configs/training.yaml
```

## Architecture

```
CIGRL/
├── ros2_ws/src/
│   ├── cigrl_core/           # Core ROS2 packages
│   │   ├── policy_node/      # RL policy inference
│   │   ├── sensor_fusion/    # EKF localization
│   │   ├── communication/    # Inter-drone messaging
│   │   └── mission_planner/  # Task allocation
│   └── cigrl_sim/            # Simulation bridges
├── experiments/              # Experimental scenarios
├── case_studies/             # Validation case studies
├── models/                   # Pretrained weights
├── configs/                  # Configuration files
└── docs/                     # Documentation
```

## Key Features

### 1. Attention-Based Cooperative Sensing

When GPS quality degrades, drones share position estimates with neighbors. An attention mechanism weighs neighbor information based on their GPS confidence:

```python
# Simplified attention computation
weights = softmax([neighbor.gps_quality ** 2 for neighbor in neighbors])
fused_position = sum(w * n.position for w, n in zip(weights, neighbors))
```

### 2. Multi-Mission Support

| Mission Type   | Description                          |
| -------------- | ------------------------------------ |
| Surveillance   | Sector-based coverage with handovers |
| Delivery       | Route deconfliction with priority    |
| Formation      | Dynamic formation control            |
| Canyon Transit | GPS-denied corridor navigation       |

### 3. Comprehensive Metrics

- Mission completion rate
- Trajectory deviation
- Localization drift
- Collision/near-miss rates
- Communication efficiency

## Experiments

Run the full experimental suite:

```bash
python experiments/cigrl_enhanced_simulation.py
```

Results are saved to `experiments/deep_evaluation_results.json`.

## Case Studies

### Urban Delivery

```bash
python case_studies/urban_delivery/scenario.py
```

### GPS-Denied Navigation

```bash
python case_studies/gps_denied_navigation/relay_node_protocol.py
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

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- ROS2 Humble Hawksbill
- PyTorch for deep learning
- AirSim/Gazebo for simulation
