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
# --- Global Frame Stacking Constant ---
STACK_SIZE = 4

# ---------------- Frame Stacking Utility ----------------
class FrameStacker:
    """Manages the stacking of image frames for state representation."""
    def __init__(self, history_len, frame_shape):
        self.history_len = history_len
        self.frame_shape = frame_shape # (H, W, C)
        # Use a deque to store the last N frames
        self.stack = deque(maxlen=history_len)

    def reset(self, initial_frame):
        """Reset the stack with the initial frame, duplicated N times."""
        self.stack.clear()
        # Initial frame shape is (H, W, 3). Ensure it's np.uint8.
        for _ in range(self.history_len):
            self.stack.append(initial_frame)
        return self.get_stack()

    def push(self, frame):
        """Add a new frame to the stack."""
        self.stack.append(frame)

    def get_stack(self):
        """Return the current stacked state as a single NumPy array (H, W, C*N)."""
        # Stack along the channel dimension (last axis for HWC format)
        return np.concatenate(list(self.stack), axis=-1)

# ---------------- CNN ----------------
class CNN(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        # FIX: Input channels are now 3 * STACK_SIZE = 12
        self.conv1 = nn.Conv2d(3 * STACK_SIZE, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)

        # The output size of conv3 needs to be calculated for the Crafter environment state.
        # Assuming Crafter's default observation is (64, 64, 3) -> stack is (64, 64, 12)
        # 64 -> (64-8)/4 + 1 = 15 -> (15-4)/2 + 1 = 6.5 -> 6 -> (6-3)/1 + 1 = 4
        # (64x64 input) -> conv1 (15x15) -> conv2 (6x6) -> conv3 (4x4)
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, action_dim)

    def forward(self, x):
        # Input state 'x' is assumed to be the stacked state (B, H, W, C*N) or (H, W, C*N)

        if isinstance(x, np.ndarray):
            # Convert to uint8 then float32 for normalization
            x = torch.tensor(x, dtype=torch.uint8).float()

        if len(x.shape) == 3:
            x = x.unsqueeze(0) # Add batch dimension (1, H, W, C*N)

        # Permute from (B, H, W, C*N) to (B, C*N, H, W)
        # C*N is 12 here
        if x.shape[-1] == 3 * STACK_SIZE:
            x = x.permute(0, 3, 1, 2)

        x = x.to(device) / 255.0 # Normalization (handled here for convenience)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# ---------------- Memory ----------------
class MemoryBuffer:
    def __init__(self, capacity, uncertainty_dist, age_dist, decay_rate):
        self.capacity = capacity
        self.uncertainty_dist = uncertainty_dist
        self.age_dist = age_dist
        self.decay_rate = decay_rate
        self.buffer = []
        self.priorities = []
        self.ages = []  

    def push(self, transition, priority):
        """Push transition with initial priority and age 0"""
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
            self.priorities.pop(0)
            self.ages.pop(0)
        self.buffer.append(transition)
        self.priorities.append(priority + 1e-6)
        self.ages.append(0)

    def increment_age(self):
        self.ages = [a + 1 for a in self.ages]

    def decay_uncertainty_by_age(self):
        self.priorities = [
            p * np.exp(-self.decay_rate * a) for p, a in zip(self.priorities, self.ages)
        ]

    def sample(self, batch_size):
        n = len(self.buffer)
        if n == 0:
            return [], [], []

        self.decay_uncertainty_by_age()

        # Compute number of samples for each category
        n_priority = int(batch_size * self.uncertainty_dist)
        n_age = int(batch_size * self.age_dist)
        n_random = batch_size - n_priority - n_age

        indices = set()

        # --- Priority-based sampling ---
        if n_priority > 0:
            priorities = np.array(self.priorities, dtype=np.float64)
            sum_p = np.sum(priorities)
            if sum_p == 0:
                priority_indices = np.random.choice(n, n_priority, replace=False)
            else:
                probs = priorities / sum_p
                priority_indices = np.random.choice(n, n_priority, replace=False, p=probs)
            indices.update(priority_indices)

        # --- Age-based sampling (older samples first) ---
        if n_age > 0:
            age_indices = np.argsort(-np.array(self.ages))[:n_age]
            indices.update(age_indices)

        # --- Random sampling for the rest ---
        if n_random > 0:
            remaining_candidates = list(set(range(n)) - indices)
            if len(remaining_candidates) > 0:
                rand_indices = np.random.choice(remaining_candidates, min(n_random, len(remaining_candidates)), replace=False)
                indices.update(rand_indices)

        indices = list(indices)
        if len(indices) < batch_size:
            remaining = list(set(range(n)) - set(indices))
            if remaining:
                extra = np.random.choice(remaining, min(batch_size - len(indices), len(remaining)), replace=False)
                indices.extend(extra)

        # Truncate to batch_size if oversampled due to set operation filling
        indices = indices[:batch_size]

        batch = [self.buffer[i] for i in indices]
        ages = [self.ages[i] for i in indices]

        return batch, indices, ages

    def update_priority_and_reset_age(self, idx, new_priority):
        self.priorities[idx] = new_priority + 1e-6
        self.ages[idx] = 0 

    def __len__(self):
        return len(self.buffer)

# ---------------- DQN Agent ----------------
class DQN:
    def __init__(self, env, alpha, discount, episodes, targetSync, memoryBuffer, age_update, uncertainty_dist, age_dist, mem_sample, uncertainty_decay, epsilon):
        self.actions = env.action_space.n
        self.env = env
        self.alpha = alpha
        self.gamma = discount
        self.episodes = episodes
        self.targetSync = targetSync
        self.memoryBuffer = memoryBuffer
        self.age_update = age_update
        self.age_dist = age_dist
        self.uncertainty_dist = uncertainty_dist
        self.mem_sample = mem_sample
        self.uncertainty_decay = uncertainty_decay
        self.epsilon = epsilon

        self.policy_net = CNN(self.actions).to(device)
        self.target_net = CNN(self.actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.alpha)
        self.memory = MemoryBuffer(self.memoryBuffer, self.uncertainty_dist, self.age_dist, self.uncertainty_decay)

        initial_frame, _ = self.env.reset()
        self.frame_stacker = FrameStacker(STACK_SIZE, initial_frame.shape)

    def select_action(self, state_stack, epsilon):
        """State_stack is the concatenated array (H, W, C*N)"""
        if random.random() < epsilon:
            return self.env.action_space.sample()

        with torch.no_grad():
            q_values = self.policy_net(state_stack)
            return q_values.argmax(dim=1).item()

    def update(self):
        if len(self.memory) < self.mem_sample:
            return 0.0, None, None, None # Return consistent number of values

        batch, indices, ages = self.memory.sample(self.mem_sample)
        if not batch: 
            return 0.0, None, None, None

        # Unpack, ensuring correct numpy/tensor dtype conversion
        # The states_np/next_states_np are now the STACKED frames
        states_np, actions_np, rewards_np, next_states_np, dones_np = zip(*[t[:5] for t in batch])

        # Convert to Tensors (Policy_net forward method handles permute/normalize)
        states = torch.tensor(np.stack(states_np), dtype=torch.uint8, device=device)
        next_states = torch.tensor(np.stack(next_states_np), dtype=torch.uint8, device=device)
        actions = torch.tensor(actions_np, dtype=torch.long, device=device).unsqueeze(1)
        rewards_batch = torch.tensor(rewards_np, dtype=torch.float32, device=device).unsqueeze(1)
        # Done mask should be float to participate in multiplication
        dones = torch.tensor(dones_np, dtype=torch.float32, device=device).unsqueeze(1)


        # Compute current Q: Q(s, a; theta_online)
        # CNN.forward takes (B, H, W, C*N) -> (B, C*N, H, W) internally
        q_values = self.policy_net(states).gather(1, actions)

        # Double DQN target (no_grad block)
        with torch.no_grad():
            next_q_online = self.policy_net(next_states)
            next_actions = next_q_online.argmax(1, keepdim=True)
            next_q_target = self.target_net(next_states)
            next_q_values = next_q_target.gather(1, next_actions)
            target_q = rewards_batch + self.gamma * (1.0 - dones) * next_q_values

            # --- The Uncertainty Metric: Absolute TD-Error ---
            td_errors = torch.abs(target_q - q_values).squeeze().cpu().numpy()

        loss = F.mse_loss(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        return loss.item(), td_errors, indices, ages

    def train(self):
        rewards_history = []
        losses = []
        steps = 0

        for episode in range(self.episodes):

            # Reset environment and get initial frame
            frame, _ = self.env.reset()
            frame = np.array(frame, dtype=np.uint8)

            # Reset and get the initial stacked state
            stacked_state = self.frame_stacker.reset(frame)

            episode_reward = 0
            terminated = False
            truncated = False

            while not (terminated or truncated):
                steps += 1

                # Target network synchronization
                if steps % self.targetSync == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())

                # Epsilon-greedy action selection
                # Pass the stacked state
                action = self.select_action(stacked_state, self.epsilon)
                next_frame, reward, terminated_env, truncated_env, _ = self.env.step(action)
                reward = np.clip(reward, -1.0, 1.0)
                # Convert next frame to np.uint8
                next_frame = np.array(next_frame, dtype=np.uint8)

                # Update the frame stacker with the new frame
                self.frame_stacker.push(next_frame)
                next_stacked_state = self.frame_stacker.get_stack()

                episode_reward += reward

                # FIX: Set initial priority
                initial_priority = 1.0

                # Store transition: Stacked state, action, reward, next stacked state, done
                transition = (stacked_state, action, reward, next_stacked_state, terminated_env or truncated_env)
                # FIX: Use initial_priority
                self.memory.push(transition, priority=initial_priority)

                # Update current state for next loop iteration
                stacked_state = next_stacked_state

                if len(self.memory) >= self.mem_sample:
                    # FIX: Get new return values
                    loss_item, td_errors, indices, ages = self.update()

                    if td_errors is not None:
                        losses.append(loss_item)

                        # --- Custom Age/Uncertainty Update Logic ---
                        for idx, age_val, td_err in zip(indices, ages, td_errors):
                            if age_val >= self.age_update:
                                # Update priority using the TD-Error (our uncertainty metric)
                                self.memory.update_priority_and_reset_age(idx, td_err)

                if terminated_env or truncated_env:
                    break

            # Decay epsilon
            self.epsilon = max(0.01, self.epsilon * 0.995)

            # Increment ages (done once per episode)
            self.memory.increment_age()

            rewards_history.append(episode_reward)
            print(f"Episode {episode+1}, Reward: {episode_reward:.2f}, Epsilon: {self.epsilon:.3f}")

        # Save model
        torch.save(self.policy_net.state_dict(), "CrafterImprovedDQN.pt")

        # Plot rewards
        try:
            window = 50
            if len(rewards_history) >= window:
                moving_avg = np.convolve(rewards_history, np.ones(window)/window, mode='valid')
                # Use a dummy plt if not available
                # import matplotlib.pyplot as plt
                plt.figure()
                plt.plot(moving_avg)
                plt.xlabel("Episodes")
                plt.ylabel("Rewards (moving avg)")
                plt.title("DQN Training Rewards")
                plt.savefig("Crafter_rewardsImprovedDQN.png")
                plt.close()
        except Exception:
            pass

        self.env.close()
        return rewards_history, losses

# Define variables directly
outdir = 'logdir/crafter_reward-ppo/0'

register(id='CrafterNoReward-v1',entry_point=crafter.Env)

env = gym.make('CrafterNoReward-v1')  
env = crafter.Recorder(
  env, './path/to/logdir',
  save_stats=True,
  save_video=False,
  save_episode=False,
)
env = GymV21CompatibilityV0(env=env)

# # agents
agent = DQN(env, alpha=0.001, discount=0.99, episodes=5_000, targetSync=1_000, memoryBuffer=100_000, age_update=10, uncertainty_dist=0.5, age_dist=0.3, mem_sample=32, uncertainty_decay=0.05, epsilon=1)

DQN_rewards = []

# agent training
for _ in range(5):
    rewards, _ = agent.train()
    DQN_rewards.append(rewards)

# average results
DQN_rewards = np.mean(DQN_rewards, axis=0)

# save to csv
np.savetxt("ImprovedDQN_rewards2.csv", DQN_rewards, delimiter=",")

