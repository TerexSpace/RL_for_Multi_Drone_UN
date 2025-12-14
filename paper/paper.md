---
title: "CIGRL: A Python Framework for Multi-Drone Urban Navigation with Covariance-Weighted Cooperative Sensing"
tags:
  - Python
  - multi-agent reinforcement learning
  - graph neural networks
  - drone navigation
  - sensor fusion
  - ROS2
  - GPS-denied navigation
authors:
  - name: Ayan Kemel
    orcid: 0009-0008-2074-0942
    affiliation: "1, 3"
  - name: Luigi La Spada
    orcid: 0000-0003-4923-1320
    affiliation: 2
  - name: Almas Ospanov
    orcid: 0009-0004-3834-130X
    corresponding: true
    affiliation: 3
affiliations:
  - name: L.N. Gumilyov Eurasian National University, Astana, Kazakhstan
    index: 1
  - name: Edinburgh Napier University, Edinburgh, United Kingdom
    index: 2
  - name: Astana IT University, Astana, Kazakhstan
    index: 3
date: 13 December 2024
bibliography: paper.bib
doi: 10.5281/zenodo.17926274
---

# Summary

CIGRL (Covariance-Integrated Graph Reinforcement Learning) is a Python framework for coordinating multi-drone navigation in GPS-challenged urban environments. The software provides modular components for multi-agent reinforcement learning with attention-based communication, adaptive sensor fusion using Extended Kalman Filters, and cooperative localization through graph neural networks. CIGRL integrates with ROS2 Humble for deployment on physical drone platforms and includes simulation bridges for AirSim and Gazebo.

The framework is structured around four core modules: (1) `cigrl_core` for policy inference and attention-based message passing, (2) `sensor_fusion` implementing Adaptive Extended Kalman Filtering with online bias estimation, (3) `communication` for inter-drone state sharing, and (4) `mission_planner` for task allocation. Users can train policies using the included MAPPO implementation, evaluate performance in urban canyon simulations, and deploy trained models on embedded hardware via ROS2 nodes.

# Statement of Need

Existing multi-agent reinforcement learning libraries such as PettingZoo [@terry2021pettingzoo] and VMAS [@bettini2024vmas] provide general-purpose environments but lack integrated sensor fusion for GPS-denied navigation scenarios. Drone simulation platforms like AirSim [@shah2018airsim] offer high-fidelity physics but do not include cooperative localization algorithms. ROS2 localization packages such as `robot_localization` [@moore2014robot] implement Kalman filtering but are designed for single-robot use without uncertainty sharing between agents.

CIGRL addresses this gap by providing:

- **Covariance-weighted attention**: Drones share localization covariance matrices, enabling principled uncertainty fusion where well-localized agents contribute more to neighbors' position estimates
- **Adaptive sensor fusion**: The Fuzzy Adaptive EKF monitors innovation statistics and dynamically adjusts measurement noise covariance when GPS signals degrade [@wang2023fuzzy]
- **Modular ROS2 architecture**: Each component runs as an independent node, allowing researchers to replace individual modules without modifying the full stack
- **Pretrained models and examples**: Ready-to-use policies for surveillance, delivery, formation, and canyon transit missions

Target users include robotics researchers studying multi-agent coordination, UAV developers building GPS-resilient navigation systems, and engineers deploying drone swarms for urban applications. The framework has been tested on DJI Tello EDU quadrotors with NVIDIA Jetson Nano compute modules.

# Acknowledgements

We acknowledge support from L.N. Gumilyov Eurasian National University, Edinburgh Napier University, and Astana IT University. We thank the PyTorch, ROS2, and OpenAI Gym communities for foundational tools.

# References
