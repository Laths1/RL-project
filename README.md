# Reinforcement Learning Project: DQN and PPO Algorithm Comparison

## Project Overview

This repository contains the implementation and analysis of Deep Q-Network (DQN) and Proximal Policy Optimization (PPO) reinforcement learning algorithms for complex environment tasks. The project compares baseline implementations against enhanced versions with various architectural improvements.

## Repository Structure
├── assignment/
│ ├── baseline/
│ │ ├── implementation.py # Baseline model implementations
│ │ ├── trained_weights/ # Pre-trained model weights
│ │ └── stats.json # Performance statistics
│ ├── improved/
│ │ ├── implementation.py # Enhanced model implementations
│ │ ├── trained_weights/ # Pre-trained model weights
│ │ └── stats.json # Performance statistics
│ ├── performance_stats.py # Game performance evaluation script
│ └── stats_plot.py # Visualization and plotting utilities
├── results/
│ ├── training_rewards/ # Reward logs from training episodes
│ └── figures/ # Generated plots and visualizations
└── README.md

## Implementation Details

### Baseline Models
- **DQN Baseline**: Standard Deep Q-Network implementation
- **PPO Baseline**: Standard Proximal Policy Optimization implementation

### Enhanced Models
The improved implementations feature several key enhancements:

**DQN Improvements:**
- Frame stacking for temporal awareness
- Biologically-inspired memory management
- Advanced experience replay sampling
- Training stabilization techniques

**PPO Improvements:**
- Residual connections in CNN architecture
- Deep residual networks for feature extraction
- Enhanced gradient flow and training stability
- Temporal processing via frame stacking

## Usage Instructions
All the models were trained on the HPC cluster. To train the models, create a .sh file and train on the cluster.

