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
---

# Summary

CIGRL (Covariance-Integrated Graph Reinforcement Learning) is a Python framework for coordinating multi-drone navigation in GPS-challenged urban environments. The software enables swarms of 5-50 drones to navigate reliably through "urban canyons" where GPS signals are frequently obstructed by buildings. CIGRL addresses this challenge by integrating attention-based cooperative sensing with multi-agent reinforcement learning, allowing drones to share localization uncertainty through graph neural networks.

The framework provides: (1) a graph neural network architecture where drones share localization covariance matrices to enable cooperative position estimation, (2) an Adaptive Extended Kalman Filter with intelligent online bias estimation for sensor fusion, (3) uncertainty-aware reward shaping for multi-agent policy gradient training, and (4) a complete ROS2 Humble software stack for real-world deployment.

# Statement of Need

Autonomous multi-drone operations in urban environments face critical challenges including GPS signal degradation, dynamic obstacle avoidance, and inter-drone coordination. Standard navigation systems using low-cost IMUs suffer from rapid position drift when satellite data is lost [@groves2013principles]. While multi-agent reinforcement learning approaches like QMIX [@rashid2020monotonic], CommNet [@sukhbaatar2016learning], and MAPPO [@yu2022surprising] enable coordination, they assume perfect state information and do not propagate localization uncertainty through agent communications [@zhu2022survey].

CIGRL fills this gap by providing an open-source framework that:

- **Propagates uncertainty**: Drones share full covariance matrices, enabling principled uncertainty fusion via attention mechanisms weighted by inverse covariance trace
- **Adapts to sensor degradation**: The Fuzzy Adaptive Extended Kalman Filter monitors innovation statistics and dynamically tunes measurement noise covariance [@wang2023fuzzy]
- **Scales efficiently**: Graph-based message passing with $O(n \cdot k)$ complexity enables deployment on swarms up to 50 drones
- **Deploys in production**: Complete ROS2 implementation with Docker containerization and pretrained models

Target users include robotics researchers studying multi-agent coordination, UAV developers building GPS-resilient navigation systems, and engineers deploying drone swarms for urban logistics, surveillance, or emergency response. The framework has been validated through extensive simulation experiments and hardware testing on a 5-drone testbed using DJI Tello EDU quadrotors with NVIDIA Jetson Nano compute modules.

CIGRL achieves statistically significant improvements over baselines: 144.7% higher mission completion versus non-cooperative methods ($p<0.001$), 81% reduction in position drift during 60-second GPS outages, and real-time inference at 119 Hz on embedded hardware.

# Acknowledgements

We acknowledge the support of the L.N. Gumilyov Eurasian National University, Edinburgh Napier University, and Astana IT University. We thank the PyTorch and ROS2 communities for their foundational open-source tools.

# References
