# Reinforcement Learning Project: DQN and PPO Algorithm Comparison

## Project Overview

This repository contains the implementation and analysis of Deep Q-Network (DQN) and Proximal Policy Optimization (PPO) reinforcement learning algorithms for complex environment tasks. The project compares baseline implementations against enhanced versions with various architectural improvements.

## Repository Structure
he assignment folder includes two subfolders, baseline and improved which include the implementation of the baseline models and the final imrpoved models. These files also include the weights after training which are used to run the game and generate the achievement stats used in the paper(performance_stats.py & stats_plot.py). These folders also include the stats json files. The rewards for each implementation is saved which is used to plot the training rewards over 5000 episodes.

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

