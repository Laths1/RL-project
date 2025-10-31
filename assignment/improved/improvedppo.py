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
STACK_SIZE = 4
# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

###################
# Improved Residual Block with BatchNorm
###################
class ImprovedResidualBlock(nn.Module):
    """Enhanced residual block with batch normalization for deeper networks"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # Main path
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                              stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)

        # Skip connection
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.skip(x)

        # Main path with pre-activation (BN -> ReLU -> Conv)
        out = self.bn1(x)
        out = F.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = F.relu(out)
        out = self.conv2(out)

        out += identity
        return out

###################
# Frame Stacker 
###################
class FrameStacker:
    def __init__(self, stack_size, frame_shape):
        self.stack_size = stack_size
        self.frames = deque(maxlen=stack_size)
        self.frame_shape = frame_shape

    def reset(self, frame):
        """Reset the stack with the first frame repeated"""
        self.frames.clear()
        for _ in range(self.stack_size):
            self.frames.append(frame)
        return self.get_stack()

    def push(self, frame):
        """Push a new frame to the stack"""
        self.frames.append(frame)

    def get_stack(self):
        """Get the current stack as numpy array (H, W, C*N)"""
        return np.concatenate(list(self.frames), axis=2)

###################
# Much Deeper CNN Feature Extractor
###################
class DeepCNNFeature(nn.Module):
    """Much deeper CNN encoder with residual blocks"""
    def __init__(self, feature_dim=512, input_shape=(84, 84)):
        super().__init__()

        # Initial convolution (more channels to start)
        self.conv1 = nn.Conv2d(3 * STACK_SIZE, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Deep residual blocks organized in stages
        self.stage1 = self._make_stage(64, 64, 3, stride=1)           # 64x21x21
        self.stage2 = self._make_stage(64, 128, 4, stride=2)          # 128x11x11
        self.stage3 = self._make_stage(128, 256, 6, stride=2)         # 256x6x6
        self.stage4 = self._make_stage(256, 512, 3, stride=2)         # 512x3x3

        # Global average pooling instead of flattening for better stability
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Calculate feature size dynamically
        self.feature_size = self._calculate_feature_size(input_shape)
        print(f"Calculated feature size: {self.feature_size}")

        # Final fully connected layer
        self.fc = nn.Linear(512, feature_dim)  # After stage4 we have 512 channels

    def _make_stage(self, in_channels, out_channels, num_blocks, stride):
        """Create a stage with multiple residual blocks"""
        layers = []

        # First block in stage may need downsampling
        layers.append(ImprovedResidualBlock(in_channels, out_channels, stride))

        # Subsequent blocks
        for _ in range(1, num_blocks):
            layers.append(ImprovedResidualBlock(out_channels, out_channels, stride=1))

        return nn.Sequential(*layers)

    def _calculate_feature_size(self, input_shape):
        """Calculate the feature size after convolutional layers"""
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3 * STACK_SIZE, input_shape[0], input_shape[1])

            # Forward pass through all layers
            x = self.conv1(dummy_input)
            x = self.bn1(x)
            x = F.relu(x)
            x = self.maxpool(x)

            x = self.stage1(x)
            x = self.stage2(x)
            x = self.stage3(x)
            x = self.stage4(x)

            x = self.avgpool(x)
            return x.view(1, -1).size(1)

    def forward(self, x):
        # Input handling (same as before)
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.uint8).float()
        if x.dim() == 3:
            x = x.unsqueeze(0)

        # Handle channel-first vs channel-last input
        if len(x.shape) == 4:
            if x.shape[1] == 3 * STACK_SIZE:
                pass
            elif x.shape[3] == 3 * STACK_SIZE:
                x = x.permute(0, 3, 1, 2)

        x = x.to(device) / 255.0

        # Forward pass through deep network
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc(x))
        return x

###################
# Enhanced PPO Agent with Deeper Network
###################
class DeepPPOAgent:
    def __init__(self, env, lr=1e-4, gamma=0.99, lam=0.95, clip_eps=0.2,
                 epochs=4, minibatch_size=64, batch_size=2048, feature_dim=512,
                 ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5):
        self.env = env
        self.action_dim = env.action_space.n
        self.lr = lr
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.minibatch_size = minibatch_size
        self.batch_size = batch_size
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

        # Get input shape from environment
        initial_frame, _ = self.env.reset()
        input_shape = initial_frame.shape[:2]

        # Use the much deeper encoder
        self.encoder = DeepCNNFeature(feature_dim, input_shape=input_shape).to(device)

        # Policy and value heads
        self.actor = nn.Linear(feature_dim, self.action_dim).to(device)
        self.critic = nn.Linear(feature_dim, 1).to(device)

        # Optimizer with weight decay for regularization
        self.optimizer = torch.optim.AdamW(
            list(self.encoder.parameters()) +
            list(self.actor.parameters()) +
            list(self.critic.parameters()),
            lr=self.lr, weight_decay=1e-4
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=1000
        )

        # Initialize Frame Stacking
        self.frame_stacker = FrameStacker(STACK_SIZE, initial_frame.shape)

        # Training diagnostics
        self.gradient_norms = []

    def select_action(self, stacked_state):
        """Return action, log_prob, value for a single STACKED state"""
        state_t = torch.tensor(np.array(stacked_state, dtype=np.uint8), dtype=torch.float32)

        if state_t.dim() == 3:
            state_t = state_t.unsqueeze(0)

        with torch.no_grad():
            features = self.encoder(state_t)
            logits = self.actor(features)
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            value = self.critic(features).squeeze(-1)

        return action.item(), log_prob.item(), value.item()

    def compute_gae(self, rewards, values, dones, last_value):
        values = np.append(values, last_value)
        adv = np.zeros_like(rewards, dtype=np.float32)
        lastgaelam = 0
        for t in reversed(range(len(rewards))):
            nonterminal = 1.0 - float(dones[t])
            delta = rewards[t] + self.gamma * values[t + 1] * nonterminal - values[t]
            adv[t] = lastgaelam = delta + self.gamma * self.lam * nonterminal * lastgaelam
        returns = adv + values[:-1]
        return adv, returns

    def update(self, rollout):
        states = rollout['states']
        actions = rollout['actions']
        old_log_probs = rollout['log_probs']
        returns = rollout['returns']
        advantages = rollout['advantages']

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        N = len(states)
        indices = np.arange(N)

        actor_losses = []
        critic_losses = []
        entropy_losses = []

        for _ in range(self.epochs):
            np.random.shuffle(indices)
            for start in range(0, N, self.minibatch_size):
                mb_idx = indices[start:start + self.minibatch_size]

                mb_states = np.stack(states[mb_idx]).astype(np.float32)
                mb_states = torch.tensor(mb_states).to(device)

                mb_actions = torch.tensor(actions[mb_idx], dtype=torch.long, device=device)
                mb_old_log_probs = torch.tensor(old_log_probs[mb_idx], dtype=torch.float32, device=device)
                mb_returns = torch.tensor(returns[mb_idx], dtype=torch.float32, device=device)
                mb_advantages = torch.tensor(advantages[mb_idx], dtype=torch.float32, device=device)

                # Forward pass
                features = self.encoder(mb_states)
                logits = self.actor(features)

                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)

                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                # Policy loss (clipped)
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                values = self.critic(features).squeeze(-1)
                value_loss = F.mse_loss(values, mb_returns)

                # Total loss
                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

                # Backward pass with gradient monitoring
                self.optimizer.zero_grad()
                loss.backward()

                # Gradient clipping and monitoring
                total_norm = nn.utils.clip_grad_norm_(
                    list(self.encoder.parameters()) +
                    list(self.actor.parameters()) +
                    list(self.critic.parameters()),
                    self.max_grad_norm
                )
                self.gradient_norms.append(total_norm.item())

                self.optimizer.step()

                actor_losses.append(policy_loss.item())
                critic_losses.append(value_loss.item())
                entropy_losses.append(entropy.item())

        # Step learning rate scheduler
        self.scheduler.step()

        return np.mean(actor_losses), np.mean(critic_losses), np.mean(entropy_losses)

    def train(self, num_episodes=1000, max_steps_per_episode=1000, render=False, save_models=True):
        rewards_history = []
        losses = {"actor": [], "critic": [], "entropy": []}

        # Initial reset
        frame, _ = self.env.reset()
        frame = np.array(frame, dtype=np.uint8)
        stacked_state = self.frame_stacker.reset(frame)

        episode_reward = 0.0

        # Rollout buffers
        states_buf = []
        actions_buf = []
        logp_buf = []
        rewards_buf = []
        dones_buf = []
        values_buf = []

        for ep in range(num_episodes):
            frame, _ = self.env.reset()
            frame = np.array(frame, dtype=np.uint8)
            stacked_state = self.frame_stacker.reset(frame)

            episode_reward = 0.0
            done = False
            truncated = False
            steps = 0

            while not (done or truncated) and steps < max_steps_per_episode:
                steps += 1

                action, logp, value = self.select_action(stacked_state)

                next_frame, reward, done_env, truncated_env, _ = self.env.step(action)
                next_frame = np.array(next_frame, dtype=np.uint8)

                self.frame_stacker.push(next_frame)
                next_stacked_state = self.frame_stacker.get_stack()

                episode_reward += reward

                # Store experience
                states_buf.append(stacked_state.astype(np.uint8))
                actions_buf.append(action)
                logp_buf.append(logp)
                rewards_buf.append(reward)
                dones_buf.append(done_env or truncated_env)
                values_buf.append(value)

                # Update current state
                stacked_state = next_stacked_state
                done, truncated = done_env, truncated_env

                # Update when enough steps collected or episode ends
                if len(states_buf) >= self.batch_size or (done or truncated):
                    # Compute last value for bootstrap
                    if done or truncated:
                        last_value = 0.0
                    else:
                        with torch.no_grad():
                            s_t = torch.tensor(np.array(stacked_state, dtype=np.uint8), dtype=torch.float32)
                            if s_t.dim() == 3:
                                s_t = s_t.unsqueeze(0)
                            last_value = self.critic(self.encoder(s_t)).squeeze(-1).item()

                    values_np = np.array(values_buf, dtype=np.float32)
                    rewards_np = np.array(rewards_buf, dtype=np.float32)
                    dones_np = np.array(dones_buf, dtype=np.float32)

                    advantages, returns = self.compute_gae(rewards_np, values_np, dones_np, last_value)

                    rollout = {
                        'states': np.array(states_buf, dtype=np.uint8),
                        'actions': np.array(actions_buf, dtype=np.int64),
                        'log_probs': np.array(logp_buf, dtype=np.float32),
                        'returns': returns.astype(np.float32),
                        'advantages': advantages.astype(np.float32)
                    }

                    a_loss, v_loss, e_loss = self.update(rollout)
                    losses['actor'].append(a_loss)
                    losses['critic'].append(v_loss)
                    losses['entropy'].append(e_loss)

                    # Clear buffers
                    states_buf = []
                    actions_buf = []
                    logp_buf = []
                    rewards_buf = []
                    dones_buf = []
                    values_buf = []

                if render:
                    try:
                        self.env.render()
                    except Exception:
                        pass

            rewards_history.append(episode_reward)
            current_lr = self.scheduler.get_last_lr()[0]
            avg_grad_norm = np.mean(self.gradient_norms[-10:]) if self.gradient_norms else 0

            print(f"[DeepPPO] Episode {ep+1}/{num_episodes} "
                  f"Reward: {episode_reward:.2f} Steps: {steps} "
                  f"LR: {current_lr:.2e} GradNorm: {avg_grad_norm:.2f}")

        # Save models if requested
        if save_models:
            torch.save(self.encoder.state_dict(), 'deep_ppo_encoder.pth')
            torch.save(self.actor.state_dict(), 'deep_ppo_actor.pth')
            torch.save(self.critic.state_dict(), 'deep_ppo_critic.pth')
            print("Models saved!")

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

# agents
ppo_agent = DeepPPOAgent(env=env)

PPO_rewards = []

# agent training
for _ in range(5):
    ppo_rewards, _ = ppo_agent.train(num_episodes=5000,max_steps_per_episode=1000,render=False,save_models=True)
    PPO_rewards.append(ppo_rewards)

# average results
PPO_rewards = np.mean(PPO_rewards, axis=0)

# save to csv
np.savetxt("ImprovedPPO_rewards2.csv", PPO_rewards, delimiter=",")

episodes = np.arange(len(PPO_rewards))
