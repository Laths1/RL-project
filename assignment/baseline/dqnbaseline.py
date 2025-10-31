import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import crafter
import random
import matplotlib.pyplot as plt
from collections import deque
from shimmy import GymV21CompatibilityV0
from gym.envs.registration import register
import pandas as pd
# Fix numpy bool8 issue
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
#########
# DQN
#########
# ---------------- CNN ----------------
class CNN(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, action_dim)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            # Convert to uint8 then float32 for normalization
            x = torch.tensor(x, dtype=torch.uint8).float()
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        if x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2)
        x = x.to(device) / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# ---------------- Memory ----------------
class MemoryBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, transition):
        # Store state as uint8
        state, action, reward, next_state, done = transition
        self.buffer.append((state.astype(np.uint8), action, reward, next_state.astype(np.uint8), done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions_batch, rewards_batch, next_states, dones_batch = zip(*batch)

        # Convert to tensors, loading as float32
        states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
        actions_batch = torch.tensor(actions_batch, dtype=torch.long).unsqueeze(1).to(device)
        rewards_batch = torch.tensor(rewards_batch, dtype=torch.float32).to(device)
        dones_batch = torch.tensor(dones_batch, dtype=torch.float32).to(device)

        return states, actions_batch, rewards_batch, next_states, dones_batch

    def __len__(self):
        return len(self.buffer)

# ---------------- DQN Agent ----------------
class DQN:
    def __init__(self, env, alpha, epsilon, discount, episodes, targetSync, memoryBuffer):
        self.env = env
        self.actions = env.action_space.n
        self.alpha = alpha
        self.epsilon = epsilon
        self.discount = discount
        self.episodes = episodes
        self.targetSync = targetSync
        self.memoryBuffer = memoryBuffer

        self.policy_net = CNN(self.actions).to(device)
        self.target_net = CNN(self.actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.alpha)

    def train(self):
        memory = MemoryBuffer(self.memoryBuffer)
        rewards_history = []

        for episode in range(self.episodes):
            if episode % self.targetSync == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            # Reset environment and convert to np.uint8
            state, _ = self.env.reset()  
            state = np.array(state, dtype=np.uint8)
            episode_reward = 0
            terminated = False
            truncated = False

            while not (terminated or truncated):
                # Epsilon-greedy action
                if np.random.rand() < self.epsilon:
                    action = self.env.action_space.sample()
                else:
                    with torch.no_grad():
                        state_t = torch.tensor(state, dtype=torch.uint8).float().unsqueeze(0).to(device)  # add batch dim
                        q_values = self.policy_net(state_t)
                        action = q_values.argmax().item()

                # Step environment
                next_state, reward, terminated, truncated, info = self.env.step(action)
                next_state = np.array(next_state, dtype=np.uint8)
                episode_reward += reward

                # Store transition
                memory.push((state, action, reward, next_state, terminated or truncated))
                state = next_state

                # Learn from batch
                if len(memory) >= 128:
                    states, actions_batch, rewards_batch, next_states, dones_batch = memory.sample(128)

                    q_values_batch = self.policy_net(states).gather(1, actions_batch).squeeze(1)

                    with torch.no_grad():
                        max_next_q = self.target_net(next_states).max(1)[0]
                        targets = rewards_batch + self.discount * max_next_q * (1 - dones_batch)

                    loss = F.mse_loss(q_values_batch, targets)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            # Decay epsilon
            self.epsilon = max(0.01, self.epsilon * 0.995)

            rewards_history.append(episode_reward)
            print(f"Episode {episode+1}, Reward: {episode_reward:.2f}, Epsilon: {self.epsilon:.3f}")

        # Save model
        torch.save(self.policy_net.state_dict(), "CrafterDQNbase.pt")

        # Plot rewards
        try:
            window = 50
            if len(rewards_history) >= window:
                moving_avg = np.convolve(rewards_history, np.ones(window)/window, mode='valid')
                plt.figure()
                plt.plot(moving_avg)
                plt.xlabel("Episodes")
                plt.ylabel("Rewards (moving avg)")
                plt.title("DQN Training Rewards")
                plt.savefig("Crafter_rewardsDQNbase.png")
                plt.close()
        except Exception:
            pass

        self.env.close()
        return rewards_history

# Define variables directly
outdir = 'logdir/crafter_reward-dqn/0'

register(id='CrafterNoReward-v1',entry_point=crafter.Env)

env = gym.make('CrafterNoReward-v1')  
env = crafter.Recorder(
  env, './path/to/logdir',
  save_stats=True,
  save_video=False,
  save_episode=False,
)
env = GymV21CompatibilityV0(env=env)

# agents
agent = DQN(env, alpha=0.05, epsilon=1, discount=0.99,episodes=5000, targetSync=5, memoryBuffer=5_000)

DQN_rewards = []

# agent training
for _ in range(5):
    DQN_rewards.append(agent.train())

# average results
DQN_rewards = np.mean(DQN_rewards, axis=0)

# save to csv
np.savetxt("DQN_rewards.csv", DQN_rewards, delimiter=",")

