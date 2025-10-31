import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import crafter
from collections import deque
from shimmy import GymV21CompatibilityV0
from gym.envs.registration import register
import os
from pathlib import Path
import shutil
import random

# Fix numpy bool8 issue
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Global Frame Stacking Constant
STACK_SIZE = 4

# Frame Stacking Utility
class FrameStacker:
    """Manages the stacking of image frames for state representation."""
    def __init__(self, history_len, frame_shape):
        self.history_len = history_len
        self.frame_shape = frame_shape
        self.stack = deque(maxlen=history_len)

    def reset(self, initial_frame):
        """Reset the stack with the initial frame, duplicated N times."""
        self.stack.clear()
        for _ in range(self.history_len):
            self.stack.append(initial_frame)
        return self.get_stack()

    def push(self, frame):
        """Add a new frame to the stack."""
        self.stack.append(frame)

    def get_stack(self):
        """Return the current stacked state as a single NumPy array (H, W, C*N)."""
        return np.concatenate(list(self.stack), axis=-1)

# Flexible CNN that can handle both 3 and 12 input channels
class FlexibleCNN(nn.Module):
    def __init__(self, action_dim, in_channels=3):
        super().__init__()
        self.in_channels = in_channels
        self.conv1 = nn.Conv2d(in_channels, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, action_dim)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.uint8).float()

        if len(x.shape) == 3:
            x = x.unsqueeze(0)

        # Handle different input formats
        if x.shape[-1] == self.in_channels:
            x = x.permute(0, 3, 1, 2)
        elif len(x.shape) == 4 and x.shape[1] == self.in_channels:
            # Already in correct format
            pass
        elif len(x.shape) == 4 and x.shape[1] > self.in_channels:
            # If we have more channels than expected, take the last ones
            x = x[:, -self.in_channels:, :, :]

        x = x.to(device) / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# Flexible CNN Feature Extractor for PPO
class FlexibleCNNFeature(nn.Module):
    def __init__(self, feature_dim=512, in_channels=3):
        super().__init__()
        self.in_channels = in_channels
        self.conv1 = nn.Conv2d(in_channels, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        self.fc = nn.Linear(64 * 4 * 4, feature_dim)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.uint8).float()
        if x.dim() == 3:
            x = x.unsqueeze(0)

        # Handle different input formats
        if x.shape[-1] == self.in_channels and x.dim() == 4:
            x = x.permute(0, 3, 1, 2)
        elif len(x.shape) == 4 and x.shape[1] == self.in_channels:
            # Already in correct format
            pass
        elif len(x.shape) == 4 and x.shape[1] > self.in_channels:
            # If we have more channels than expected, take the last ones
            x = x[:, -self.in_channels:, :, :]

        x = x.to(device) / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc(x))
        return x

def detect_input_channels(model_path):
    """Detect whether a model expects 3 or 12 input channels"""
    try:
        checkpoint = torch.load(model_path, map_location=device)
        
        # For DQN models (direct state_dict)
        if 'conv1.weight' in checkpoint:
            conv1_weight = checkpoint['conv1.weight']
            in_channels = conv1_weight.shape[1]
            print(f"Detected {in_channels} input channels for {model_path}")
            return in_channels
        
        # For PPO models (checkpoint dict with encoder)
        elif 'encoder' in checkpoint:
            encoder_weights = checkpoint['encoder']
            if 'conv1.weight' in encoder_weights:
                conv1_weight = encoder_weights['conv1.weight']
                in_channels = conv1_weight.shape[1]
                print(f"Detected {in_channels} input channels for {model_path}")
                return in_channels
        
        # Default to 3 if we can't detect
        print(f"Could not detect input channels for {model_path}, defaulting to 3")
        return 3
        
    except Exception as e:
        print(f"Error detecting input channels for {model_path}: {e}")
        return 3

class PPOAgent:
    def __init__(self, env, feature_dim=512, in_channels=3):
        self.env = env
        self.action_dim = env.action_space.n
        self.in_channels = in_channels
        self.uses_frame_stacking = (in_channels == 12)
        
        self.encoder = FlexibleCNNFeature(feature_dim, in_channels).to(device)
        self.actor = nn.Linear(feature_dim, self.action_dim).to(device)
        self.critic = nn.Linear(feature_dim, 1).to(device)
        
        # Initialize frame stacker only if needed
        if self.uses_frame_stacking:
            initial_frame, _ = self.env.reset()
            self.frame_stacker = FrameStacker(STACK_SIZE, initial_frame.shape)

    def load_models(self, model_path):
        """Load trained PPO model from combined checkpoint file"""
        print(f"Loading PPO model from: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        print(f"PPO model loaded successfully! Uses frame stacking: {self.uses_frame_stacking}")

    def select_action(self, state):
        """Select action using the trained policy"""
        # Handle frame stacking if needed
        if self.uses_frame_stacking and len(state.shape) == 3 and state.shape[-1] == 3:
            # Single frame but model expects stacked frames - this shouldn't happen
            print("Warning: Model expects stacked frames but got single frame")
            current_frame = state
        elif not self.uses_frame_stacking and len(state.shape) == 3 and state.shape[-1] == 12:
            # Stacked frames but model expects single frame - take current frame
            current_frame = state[..., -3:]
        else:
            current_frame = state
            
        state_t = torch.tensor(np.array(current_frame, dtype=np.uint8), dtype=torch.float32)
        if state_t.dim() == 3:
            state_t = state_t.unsqueeze(0)

        with torch.no_grad():
            features = self.encoder(state_t.to(device) / 255.0)
            logits = self.actor(features)
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
        return action.item()

class DQNAgent:
    def __init__(self, env, in_channels=3):
        self.env = env
        self.actions = env.action_space.n
        self.in_channels = in_channels
        self.uses_frame_stacking = (in_channels == 12)
        self.policy_net = FlexibleCNN(self.actions, in_channels).to(device)
        
        # Initialize frame stacker only if needed
        if self.uses_frame_stacking:
            initial_frame, _ = self.env.reset()
            self.frame_stacker = FrameStacker(STACK_SIZE, initial_frame.shape)

    def load_model(self, model_path):
        """Load trained DQN model"""
        print(f"Loading DQN model from: {model_path}")
        self.policy_net.load_state_dict(torch.load(model_path, map_location=device))
        print(f"DQN model loaded successfully! Uses frame stacking: {self.uses_frame_stacking}")

    def select_action(self, state, epsilon=0.01):
        """Select action using the trained DQN policy"""
        if random.random() < epsilon:
            return self.env.action_space.sample()

        # Handle frame stacking if needed
        if self.uses_frame_stacking and len(state.shape) == 3 and state.shape[-1] == 3:
            # Single frame but model expects stacked frames - this shouldn't happen
            print("Warning: Model expects stacked frames but got single frame")
            current_frame = state
        elif not self.uses_frame_stacking and len(state.shape) == 3 and state.shape[-1] == 12:
            # Stacked frames but model expects single frame - take current frame
            current_frame = state[..., -3:]
        else:
            current_frame = state

        state_t = torch.tensor(np.array(current_frame, dtype=np.uint8), dtype=torch.float32)
        if state_t.dim() == 3:
            state_t = state_t.unsqueeze(0)

        with torch.no_grad():
            q_values = self.policy_net(state_t)
            return q_values.argmax(dim=1).item()

def setup_evaluation_env(logdir):
    """Set up environment with Crafter Recorder"""
    register(id='CrafterNoReward-v1', entry_point=crafter.Env)

    env = gym.make('CrafterNoReward-v1')
    env = crafter.Recorder(
        env,
        logdir,
        save_stats=True,
        save_video=False,
        save_episode=False,
    )
    env = GymV21CompatibilityV0(env=env)
    return env

def run_evaluation(agent, agent_type, num_episodes=10, max_steps=1000, logdir="./eval_logs"):
    """
    Run evaluation using Crafter Recorder to automatically save stats
    """
    # Create fresh log directory
    if os.path.exists(logdir):
        shutil.rmtree(logdir)
    Path(logdir).mkdir(parents=True, exist_ok=True)

    # Set up environment with recorder
    env = setup_evaluation_env(logdir)

    print(f"Evaluating {agent_type} for {num_episodes} episodes...")

    for episode in range(num_episodes):
        print(f"Episode {episode + 1}/{num_episodes}")

        # Reset environment
        frame, _ = env.reset()
        frame = np.array(frame, dtype=np.uint8)
        
        # Handle frame stacking if the agent uses it
        if hasattr(agent, 'uses_frame_stacking') and agent.uses_frame_stacking:
            state = agent.frame_stacker.reset(frame)
        else:
            state = frame

        episode_reward = 0
        steps = 0
        terminated = False
        truncated = False

        while not (terminated or truncated) and steps < max_steps:
            steps += 1

            # Select action based on agent type
            if agent_type in ["DQN Baseline", "DQN Improved"]:
                action = agent.select_action(state, epsilon=0.01)
            else:  # PPO
                action = agent.select_action(state)

            # Take action
            next_frame, reward, terminated_env, truncated_env, info = env.step(action)
            next_frame = np.array(next_frame, dtype=np.uint8)

            # Update state (with frame stacking if needed)
            if hasattr(agent, 'uses_frame_stacking') and agent.uses_frame_stacking:
                agent.frame_stacker.push(next_frame)
                next_state = agent.frame_stacker.get_stack()
            else:
                next_state = next_frame

            episode_reward += reward
            state = next_state
            terminated, truncated = terminated_env, truncated_env

            if terminated or truncated:
                break

        print(f"  Reward: {episode_reward:.2f}, Steps: {steps}")

    env.close()

    # Collect and process the stats that Crafter Recorder saved
    return collect_crafter_stats(logdir, agent_type)

def collect_crafter_stats(logdir, agent_type):
    """
    Collect and process the stats.jsonl files created by Crafter Recorder
    """
    stats_files = list(Path(logdir).glob("**/stats.jsonl"))

    if not stats_files:
        print(f"No stats.jsonl files found in {logdir}")
        return None

    all_stats = []
    for stats_file in stats_files:
        try:
            episodes = []
            with open(stats_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    if line.strip():
                        try:
                            episode_data = json.loads(line.strip())
                            episode_data['agent_type'] = agent_type
                            episodes.append(episode_data)
                        except json.JSONDecodeError as e:
                            print(f"Warning: Could not parse line {line_num} in {stats_file}: {e}")
                            continue
            
            all_stats.extend(episodes)
            print(f"Loaded {len(episodes)} episodes from {stats_file}")
            
        except Exception as e:
            print(f"Error reading {stats_file}: {e}")

    return all_stats

def evaluate_agent(agent, agent_type, model_path, num_episodes=10, max_steps=1000, output_dir="./eval_results"):
    """
    Main evaluation function that uses Crafter Recorder for stats
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load model
    if agent_type in ["DQN Baseline", "DQN Improved"]:
        agent.load_model(model_path)
    else:  # PPO agents
        agent.load_models(model_path)

    # Run evaluation with Crafter Recorder
    logdir = f"./temp_{agent_type.replace(' ', '_')}_logs"
    stats = run_evaluation(agent, agent_type, num_episodes, max_steps, logdir)

    if stats:
        # Save processed stats
        stats_filename = f"{output_dir}/{agent_type.replace(' ', '_')}_crafter_stats.json"
        with open(stats_filename, 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"Saved Crafter stats to {stats_filename}")

        # Calculate summary statistics
        if stats and len(stats) > 0:
            # Extract relevant metrics from Crafter stats
            achievements = []
            rewards = []
            steps = []

            for episode_stats in stats:
                if 'achievements' in episode_stats:
                    achievements.append(episode_stats['achievements'])
                if 'reward' in episode_stats:
                    rewards.append(episode_stats['reward'])
                if 'length' in episode_stats:
                    steps.append(episode_stats['length'])

            summary = {
                "agent_type": agent_type,
                "num_episodes": len(stats),
                "mean_reward": np.mean(rewards) if rewards else 0,
                "std_reward": np.std(rewards) if rewards else 0,
                "mean_steps": np.mean(steps) if steps else 0,
            }

            # Save summary
            summary_filename = f"{output_dir}/{agent_type.replace(' ', '_')}_summary.json"
            with open(summary_filename, 'w') as f:
                json.dump(summary, f, indent=2)

            print(f"\n{agent_type} Summary from Crafter Stats:")
            print(f"  Episodes: {summary['num_episodes']}")
            print(f"  Mean Reward: {summary['mean_reward']:.2f} ± {summary['std_reward']:.2f}")
            print(f"  Mean Steps: {summary['mean_steps']:.2f}")

            return stats, summary

    return None, None

def main():
    """Main function to evaluate all four models using Crafter Recorder"""
    # Model paths using your specific file names
    model_paths = {
        'DQN Baseline': "CrafterDQNbase.pt",
        'DQN Improved': "CrafterImprovedDQN.pt", 
        'PPO Baseline': "Crafter_PPObase.pt",
        'PPO Improved': "Crafter_PPO.pt"
    }

    # Check which models exist and detect their input channels
    available_models = {}
    input_channels = {}
    
    for agent_type, model_path in model_paths.items():
        if os.path.exists(model_path):
            channels = detect_input_channels(model_path)
            available_models[agent_type] = model_path
            input_channels[agent_type] = channels
            print(f"✓ Found {agent_type}: {model_path} ({channels} input channels)")
        else:
            print(f"✗ Missing {agent_type}: {model_path}")

    if not available_models:
        print("No model files found! Please ensure the weight files are in the current directory.")
        return

    # Evaluation parameters
    num_episodes = 10
    max_steps = 1000
    output_dir = "./four_model_evaluation_results"

    # Create a temporary environment for agent initialization
    temp_env = setup_evaluation_env("./temp_init")

    # Evaluate each available model
    all_results = {}
    
    for agent_type, model_path in available_models.items():
        print("\n" + "="*60)
        print(f"Evaluating {agent_type} with {model_path}")
        print(f"Input channels: {input_channels[agent_type]}")
        print("="*60)

        if "DQN" in agent_type:
            agent = DQNAgent(temp_env, in_channels=input_channels[agent_type])
        else:  # PPO agents
            agent = PPOAgent(temp_env, in_channels=input_channels[agent_type])

        stats, summary = evaluate_agent(
            agent, agent_type, model_path, num_episodes, max_steps, output_dir
        )
        
        if stats and summary:
            all_results[agent_type] = {
                'stats': stats,
                'summary': summary,
                'input_channels': input_channels[agent_type]
            }

    temp_env.close()
    
    print(f"\nEvaluation complete! Results saved to {output_dir}")
    
    # Print final comparison
    if all_results:
        print("\n" + "="*60)
        print("FINAL COMPARISON SUMMARY")
        print("="*60)
        
        for agent_type, results in all_results.items():
            summary = results['summary']
            channels = results['input_channels']
            print(f"{agent_type} ({channels} channels):")
            print(f"  Mean Reward: {summary['mean_reward']:.2f} ± {summary['std_reward']:.2f}")
            print(f"  Mean Steps: {summary['mean_steps']:.2f}")
            print()

if __name__ == "__main__":
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    main()