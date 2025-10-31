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

###################
#PPO
###################
class CNNFeature(nn.Module):
    """CNN encoder that returns a feature vector (512 dims)."""
    def __init__(self, feature_dim=512):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        self.fc = nn.Linear(64 * 4 * 4, feature_dim)

    def forward(self, x):
        # Accepts torch tensor shaped (B,H,W,3) or (B,3,H,W)
        if isinstance(x, np.ndarray):
             # Convert to uint8 then float32 for normalization
            x = torch.tensor(x, dtype=torch.uint8).float()
        if x.dim() == 3:
            x = x.unsqueeze(0)
        # if channels last convert to channels first
        if x.shape[-1] == 3 and x.dim() == 4:
            x = x.permute(0, 3, 1, 2)
        x = x.to(device) / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc(x))
        return x

class PPOAgent:
    def __init__(self, env, lr=2.5e-4, gamma=0.99, lam=0.95, clip_eps=0.2,
                 epochs=4, minibatch_size=64, batch_size=2048, feature_dim=512,
                 ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5):
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

        # Shared encoder
        self.encoder = CNNFeature(feature_dim).to(device)
        # policy (logits) and value heads
        self.actor = nn.Linear(feature_dim, self.action_dim).to(device)
        self.critic = nn.Linear(feature_dim, 1).to(device)

        # optim
        self.optimizer = torch.optim.Adam(list(self.encoder.parameters()) +
                                          list(self.actor.parameters()) +
                                          list(self.critic.parameters()), lr=self.lr)

    def select_action(self, state):
        """Return action, log_prob, value for a single state (numpy input accepted)."""
        # Convert state to uint8 then float32 for inference
        state_t = torch.tensor(np.array(state, dtype=np.uint8), dtype=torch.float32)
        if state_t.dim() == 3:
            state_t = state_t.unsqueeze(0)
        if state_t.shape[-1] == 3 and state_t.dim() == 4:
            state_t = state_t.permute(0, 3, 1, 2)
        with torch.no_grad():
            features = self.encoder(state_t.to(device) / 255.0)
            logits = self.actor(features)
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            value = self.critic(features).squeeze(-1)
        return action.item(), log_prob.item(), value.item()

    def compute_gae(self, rewards, values, dones, last_value):
        """Compute advantages using GAE. rewards, values are lists/arrays."""
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
        """
        rollout: dict with keys states, actions, log_probs, returns, advantages
        Each is a numpy array with length N (batch_size)
        """
        states = rollout['states']
        actions = rollout['actions']
        old_log_probs = rollout['log_probs']
        returns = rollout['returns']
        advantages = rollout['advantages']

        # normalize advantages
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

                # Properly stack image states and ensure float32 dtype for network input
                mb_states = np.stack(states[mb_idx]).astype(np.float32)
                mb_states = torch.tensor(mb_states)

                if mb_states.shape[-1] == 3 and mb_states.dim() == 4:
                    mb_states = mb_states.permute(0, 3, 1, 2)
                mb_states = mb_states.to(device) / 255.0

                mb_actions = torch.tensor(actions[mb_idx], dtype=torch.long, device=device)
                mb_old_log_probs = torch.tensor(old_log_probs[mb_idx], dtype=torch.float32, device=device)
                mb_returns = torch.tensor(returns[mb_idx], dtype=torch.float32, device=device)
                mb_advantages = torch.tensor(advantages[mb_idx], dtype=torch.float32, device=device)


                features = self.encoder(mb_states)
                logits = self.actor(features)
                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)

                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                # policy loss (clipped)
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # value loss
                values = self.critic(features).squeeze(-1)
                value_loss = F.mse_loss(values, mb_returns)

                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(self.encoder.parameters()) +
                                         list(self.actor.parameters()) +
                                         list(self.critic.parameters()), self.max_grad_norm)
                self.optimizer.step()

                actor_losses.append(policy_loss.item())
                critic_losses.append(value_loss.item())
                entropy_losses.append(entropy.item())

        return np.mean(actor_losses), np.mean(critic_losses), np.mean(entropy_losses)

    def train(self, num_episodes=1000, max_steps_per_episode=1000, render=False, save_models=True):
        rewards_history = []
        losses = {"actor": [], "critic": [], "entropy": []}

        state, _ = self.env.reset()
        episode_reward = 0.0
        ep_len = 0

        # rollout buffers
        states_buf = []
        actions_buf = []
        logp_buf = []
        rewards_buf = []
        dones_buf = []
        values_buf = []

        for ep in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0.0
            done = False
            truncated = False
            steps = 0

            while not (done or truncated) and steps < max_steps_per_episode:
                steps += 1
                # select action
                action, logp, value = self.select_action(state)

                next_state, reward, done, truncated, _ = self.env.step(action)
                episode_reward += reward

                # store (store state as uint8 numpy array)
                states_buf.append(state.astype(np.uint8))
                actions_buf.append(action)
                logp_buf.append(logp)
                rewards_buf.append(reward)
                dones_buf.append(done or truncated)
                values_buf.append(value)

                state = next_state

                # when enough steps collected, or episode ends, do update
                if len(states_buf) >= self.batch_size or (done or truncated):
                    # compute last value for bootstrap
                    if done or truncated:
                        last_value = 0.0
                    else:
                        # estimate last value from critic
                        with torch.no_grad():
                            # Convert state to uint8 then float32 for inference
                            s_t = torch.tensor(np.array(state, dtype=np.uint8), dtype=torch.float32)
                            if s_t.dim() == 3:
                                s_t = s_t.unsqueeze(0)
                            if s_t.shape[-1] == 3 and s_t.dim() == 4:
                                s_t = s_t.permute(0, 3, 1, 2)
                            last_value = self.critic(self.encoder(s_t.to(device) / 255.0)).squeeze(-1).item()

                    values_np = np.array(values_buf, dtype=np.float32)
                    rewards_np = np.array(rewards_buf, dtype=np.float32)
                    dones_np = np.array(dones_buf, dtype=np.float32)

                    advantages, returns = self.compute_gae(rewards_np, values_np, dones_np, last_value)

                    rollout = {
                        'states': np.array(states_buf, dtype=np.uint8), # Store as uint8 numpy array
                        'actions': np.array(actions_buf, dtype=np.int64),
                        'log_probs': np.array(logp_buf, dtype=np.float32),
                        'returns': returns.astype(np.float32),
                        'advantages': advantages.astype(np.float32)
                    }

                    a_loss, v_loss, e_loss = self.update(rollout)
                    losses['actor'].append(a_loss)
                    losses['critic'].append(v_loss)
                    losses['entropy'].append(e_loss)

                    # clear buffers
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
            print(f"[PPO] Episode {ep+1}/{num_episodes} Reward: {episode_reward:.2f} Steps: {steps}")

        if save_models:
            torch.save({
                'encoder': self.encoder.state_dict(),
                'actor': self.actor.state_dict(),
                'critic': self.critic.state_dict(),
            }, "Crafter_PPObase.pt")

        # plot rewards moving average
        try:
            window = 50
            if len(rewards_history) >= window:
                moving_avg = np.convolve(rewards_history, np.ones(window)/window, mode='valid')
                plt.figure()
                plt.plot(moving_avg)
                plt.xlabel("Episodes")
                plt.ylabel("Rewards (moving avg)")
                plt.title("PPO Training Rewards")
                plt.savefig("Crafter_rewardsPPObase.png")
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

# agents

ppo_agent = PPOAgent(env=env,lr=2.5e-4,gamma=0.99,lam=0.95,clip_eps=0.2,epochs=4,minibatch_size=64,batch_size=2048,feature_dim=512,ent_coef=0.0,vf_coef=0.5)

PPO_rewards = []

# agent training

for _ in range(5):
    ppo_rewards, _ = ppo_agent.train(num_episodes=5000,max_steps_per_episode=1000,render=False,save_models=True)
    PPO_rewards.append(ppo_rewards)

# average results
PPO_rewards = np.mean(PPO_rewards, axis=0)

# save to csv
np.savetxt("PPO_rewards.csv", PPO_rewards, delimiter=",")

