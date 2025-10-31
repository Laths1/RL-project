import json
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import pandas as pd
import seaborn as sns

def load_stats_from_evaluation_results():
    """Load stats from the four_model_evaluation_results directory"""
    results_dir = Path("./four_model_evaluation_results")
    
    if not results_dir.exists():
        print(f"Results directory {results_dir} not found!")
        return {}
    
    all_stats = {}
    
    # Look for all the JSON stats files
    stats_files = list(results_dir.glob("*_crafter_stats.json"))
    
    for stats_file in stats_files:
        # Extract agent type from filename
        filename = stats_file.stem
        if 'DQN_Baseline' in filename:
            agent_type = 'DQN Baseline'
        elif 'DQN_Improved' in filename:
            agent_type = 'DQN Improved'
        elif 'PPO_Baseline' in filename:
            agent_type = 'PPO Baseline'
        elif 'PPO_Improved' in filename:
            agent_type = 'PPO Improved'
        else:
            continue
        
        try:
            with open(stats_file, 'r') as f:
                stats_data = json.load(f)
                all_stats[agent_type] = stats_data
                print(f"Loaded {len(stats_data)} episodes for {agent_type}")
        except Exception as e:
            print(f"Error loading {stats_file}: {e}")
    
    return all_stats

def analyze_achievements(stats_data):
    """Analyze achievement data from stats"""
    if not stats_data:
        return {}, []

    # Get achievement keys from first episode of first agent
    first_agent = list(stats_data.keys())[0]
    if stats_data[first_agent]:
        achievement_keys = [key for key in list(stats_data[first_agent][0].keys()) if key.startswith('achievement_')]
    else:
        achievement_keys = []

    results = {}
    for agent_type, episodes in stats_data.items():
        if not episodes:
            continue

        agent_results = {
            'episode_lengths': [],
            'total_rewards': [],
            'achievements': {key: [] for key in achievement_keys},
            'achievement_counts_per_episode': []
        }

        for episode in episodes:
            agent_results['episode_lengths'].append(episode['length'])
            agent_results['total_rewards'].append(episode['reward'])

            # Count achievements in this episode
            episode_achievements = 0
            for achievement in achievement_keys:
                count = episode.get(achievement, 0)
                agent_results['achievements'][achievement].append(count)
                if count > 0:
                    episode_achievements += 1
            agent_results['achievement_counts_per_episode'].append(episode_achievements)

        results[agent_type] = agent_results

    return results, achievement_keys

def plot_comparison_summary(analysis_results, output_file="four_model_comparison_summary.png"):
    """Create a comprehensive comparison plot for four models"""
    if not analysis_results:
        print("No data to plot!")
        return None

    # Set style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Crafter Environment: Four Model Performance Comparison', fontsize=16, fontweight='bold')

    # Define the desired order of agents
    desired_order = ['DQN Baseline', 'DQN Improved', 'PPO Baseline', 'PPO Improved']
    agents = [agent for agent in desired_order if agent in analysis_results]
    
    if not agents:
        agents = list(analysis_results.keys())
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red

    # Plot 1: Average Rewards
    rewards_data = [analysis_results[agent]['total_rewards'] for agent in agents]
    box_plot = axes[0, 0].boxplot(rewards_data, labels=agents, patch_artist=True)
    
    # Color the boxes
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    axes[0, 0].set_title('Total Rewards per Episode', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('Reward', fontsize=12)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].tick_params(axis='x', rotation=45)

    # Plot 2: Episode Lengths
    length_data = [analysis_results[agent]['episode_lengths'] for agent in agents]
    box_plot = axes[0, 1].boxplot(length_data, labels=agents, patch_artist=True)
    
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    axes[0, 1].set_title('Episode Lengths', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('Steps', fontsize=12)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].tick_params(axis='x', rotation=45)

    # Plot 3: Achievement Counts
    achievement_data = [analysis_results[agent]['achievement_counts_per_episode'] for agent in agents]
    box_plot = axes[1, 0].boxplot(achievement_data, labels=agents, patch_artist=True)
    
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    axes[1, 0].set_title('Number of Unique Achievements per Episode', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('Achievement Count', fontsize=12)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].tick_params(axis='x', rotation=45)

    # Plot 4: Mean performance bars
    mean_rewards = [np.mean(analysis_results[agent]['total_rewards']) for agent in agents]
    mean_achievements = [np.mean(analysis_results[agent]['achievement_counts_per_episode']) for agent in agents]

    x = np.arange(len(agents))
    width = 0.35

    bars1 = axes[1, 1].bar(x - width/2, mean_rewards, width, label='Mean Reward', 
                   color=colors, alpha=0.8, edgecolor='black')
    bars2 = axes[1, 1].bar(x + width/2, mean_achievements, width, label='Mean Achievements', 
                   color=colors, alpha=0.6, edgecolor='black', hatch='//')
    
    # Add value labels on bars
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        axes[1, 1].text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 0.1, 
                       f'{mean_rewards[i]:.1f}', ha='center', va='bottom', fontsize=9)
        axes[1, 1].text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 0.1, 
                       f'{mean_achievements[i]:.1f}', ha='center', va='bottom', fontsize=9)
    
    axes[1, 1].set_title('Mean Performance Metrics', fontsize=14, fontweight='bold')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(agents, rotation=45)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()

    return fig

def plot_achievement_breakdown(analysis_results, achievement_keys, output_file="four_model_achievement_breakdown.png"):
    """Plot detailed achievement breakdown for four models"""
    if not analysis_results:
        return None

    # Define the desired order of agents
    desired_order = ['DQN Baseline', 'DQN Improved', 'PPO Baseline', 'PPO Improved']
    agents = [agent for agent in desired_order if agent in analysis_results]
    
    if not agents:
        agents = list(analysis_results.keys())
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    # Calculate mean achievement rates
    achievement_rates = {}
    for agent in agents:
        rates = {}
        for achievement in achievement_keys:
            rates[achievement] = np.mean(analysis_results[agent]['achievements'][achievement])
        achievement_rates[agent] = rates

    # Create readable achievement names
    achievement_names = [key.replace('achievement_', '').replace('_', ' ').title() for key in achievement_keys]

    # Plot
    plt.style.use('seaborn-v0_8')
    fig, ax = plt.subplots(figsize=(18, 10))

    x = np.arange(len(achievement_keys))
    width = 0.8 / len(agents)

    for i, agent in enumerate(agents):
        rates = [achievement_rates[agent][key] for key in achievement_keys]
        ax.bar(x + i * width - width * (len(agents) - 1) / 2, rates, width, 
               label=agent, color=colors[i], alpha=0.8, edgecolor='black')

    ax.set_xlabel('Achievements', fontsize=12)
    ax.set_ylabel('Average Count per Episode', fontsize=12)
    ax.set_title('Average Achievement Performance: Four Model Comparison', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(achievement_names, rotation=45, ha='right')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()

    return fig

def plot_success_rates(analysis_results, achievement_keys, output_file="four_model_success_rates.png"):
    """Plot success rates for four models"""
    if not analysis_results:
        return None

    # Define the desired order of agents
    desired_order = ['DQN Baseline', 'DQN Improved', 'PPO Baseline', 'PPO Improved']
    agents = [agent for agent in desired_order if agent in analysis_results]
    
    if not agents:
        agents = list(analysis_results.keys())
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    # Calculate success rates
    success_rates = {}
    for agent in agents:
        rates = {}
        for achievement in achievement_keys:
            counts = analysis_results[agent]['achievements'][achievement]
            success_rate = np.mean([1 if count > 0 else 0 for count in counts]) * 100
            rates[achievement] = success_rate
        success_rates[agent] = rates

    # Create readable achievement names
    achievement_names = [key.replace('achievement_', '').replace('_', ' ').title() for key in achievement_keys]

    # Plot
    plt.style.use('seaborn-v0_8')
    fig, ax = plt.subplots(figsize=(18, 8))

    x = np.arange(len(achievement_keys))
    width = 0.8 / len(agents)

    for i, agent in enumerate(agents):
        rates = [success_rates[agent][key] for key in achievement_keys]
        ax.bar(x + i * width - width * (len(agents) - 1) / 2, rates, width, 
               label=agent, color=colors[i], alpha=0.8, edgecolor='black')

    ax.set_xlabel('Achievements', fontsize=12)
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title('Achievement Success Rates: Four Model Comparison\n(Percentage of Episodes Where Achievement Was Completed)', 
                 fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(achievement_names, rotation=45, ha='right')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()

    return fig

def plot_reward_progression(analysis_results, output_file="four_model_reward_progression.png"):
    """Plot reward progression across episodes"""
    if not analysis_results:
        return None

    plt.style.use('seaborn-v0_8')
    fig, ax = plt.subplots(figsize=(12, 8))

    # Define the desired order of agents
    desired_order = ['DQN Baseline', 'DQN Improved', 'PPO Baseline', 'PPO Improved']
    agents = [agent for agent in desired_order if agent in analysis_results]
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['o', 's', '^', 'D']
    
    for i, agent in enumerate(agents):
        rewards = analysis_results[agent]['total_rewards']
        episodes = range(1, len(rewards) + 1)
        ax.plot(episodes, rewards, marker=markers[i], markersize=4, 
                label=agent, color=colors[i], alpha=0.8, linewidth=1.5)
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Reward', fontsize=12)
    ax.set_title('Reward Progression Across Episodes', fontsize=16, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def plot_algorithm_family_comparison(analysis_results, output_file="algorithm_family_comparison.png"):
    """Plot comparison between DQN and PPO families"""
    if not analysis_results:
        return None

    # Group by algorithm family
    dqn_agents = [agent for agent in analysis_results.keys() if 'DQN' in agent]
    ppo_agents = [agent for agent in analysis_results.keys() if 'PPO' in agent]
    
    if not dqn_agents or not ppo_agents:
        print("Need both DQN and PPO agents for family comparison")
        return None

    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('DQN vs PPO Algorithm Family Comparison', fontsize=16, fontweight='bold')

    # Colors for DQN and PPO families
    dqn_colors = ['#1f77b4', '#aec7e8']  # Blues
    ppo_colors = ['#2ca02c', '#98df8a']  # Greens

    # Plot 1: Mean rewards comparison
    dqn_rewards = [np.mean(analysis_results[agent]['total_rewards']) for agent in dqn_agents]
    ppo_rewards = [np.mean(analysis_results[agent]['total_rewards']) for agent in ppo_agents]
    
    dqn_labels = [agent.replace('DQN ', '') for agent in dqn_agents]
    ppo_labels = [agent.replace('PPO ', '') for agent in ppo_agents]
    
    # DQN bars
    for i, reward in enumerate(dqn_rewards):
        axes[0].bar(i - 0.2, reward, 0.4, color=dqn_colors[i], alpha=0.8, 
                   label=f'DQN {dqn_labels[i]}', edgecolor='black')
    
    # PPO bars
    for i, reward in enumerate(ppo_rewards):
        axes[0].bar(i + 0.2, reward, 0.4, color=ppo_colors[i], alpha=0.8,
                   label=f'PPO {ppo_labels[i]}', edgecolor='black')
    
    axes[0].set_title('Mean Rewards by Algorithm Family', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Mean Reward', fontsize=12)
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(['Baseline', 'Improved'])
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Mean achievements comparison
    dqn_achievements = [np.mean(analysis_results[agent]['achievement_counts_per_episode']) for agent in dqn_agents]
    ppo_achievements = [np.mean(analysis_results[agent]['achievement_counts_per_episode']) for agent in ppo_agents]
    
    # DQN bars
    for i, achievement in enumerate(dqn_achievements):
        axes[1].bar(i - 0.2, achievement, 0.4, color=dqn_colors[i], alpha=0.8, 
                   label=f'DQN {dqn_labels[i]}', edgecolor='black')
    
    # PPO bars
    for i, achievement in enumerate(ppo_achievements):
        axes[1].bar(i + 0.2, achievement, 0.4, color=ppo_colors[i], alpha=0.8,
                   label=f'PPO {ppo_labels[i]}', edgecolor='black')
    
    axes[1].set_title('Mean Achievements by Algorithm Family', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Mean Achievements per Episode', fontsize=12)
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(['Baseline', 'Improved'])
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()

    return fig

def generate_summary_table(analysis_results, achievement_keys):
    """Generate a summary table of statistics"""
    if not analysis_results:
        return pd.DataFrame()

    # Define the desired order
    desired_order = ['DQN Baseline', 'DQN Improved', 'PPO Baseline', 'PPO Improved']
    agents = [agent for agent in desired_order if agent in analysis_results]
    
    if not agents:
        agents = list(analysis_results.keys())

    summary_data = []

    for agent in agents:
        data = analysis_results[agent]

        # Calculate achievement success rates
        achievement_success = {}
        for achievement in achievement_keys:
            success_rate = np.mean([1 if count > 0 else 0 for count in data['achievements'][achievement]]) * 100
            achievement_success[achievement.replace('achievement_', '')] = f"{success_rate:.1f}%"

        summary_data.append({
            'Agent': agent,
            'Mean Reward': f"{np.mean(data['total_rewards']):.2f} ± {np.std(data['total_rewards']):.2f}",
            'Mean Episode Length': f"{np.mean(data['episode_lengths']):.1f} ± {np.std(data['episode_lengths']):.1f}",
            'Mean Achievements per Episode': f"{np.mean(data['achievement_counts_per_episode']):.2f} ± {np.std(data['achievement_counts_per_episode']):.2f}",
            'Total Episodes': len(data['total_rewards']),
            **achievement_success
        })

    df = pd.DataFrame(summary_data)
    return df

def main():
    """Main function to analyze and plot four model comparison"""
    print("Loading evaluation results...")
    all_stats = load_stats_from_evaluation_results()
    
    if not all_stats:
        print("No evaluation results found! Please run the evaluation first.")
        print("Looking in: ./four_model_evaluation_results/")
        return
    
    print(f"Found results for: {list(all_stats.keys())}")
    
    # Analyze the data
    print("Analyzing achievement data...")
    analysis_results, achievement_keys = analyze_achievements(all_stats)
    
    if not analysis_results:
        print("No data to analyze!")
        return
    
    # Generate summary table
    print("Generating summary table...")
    summary_df = generate_summary_table(analysis_results, achievement_keys)
    print("\n" + "="*80)
    print("FOUR MODEL PERFORMANCE SUMMARY")
    print("="*80)
    print(summary_df.to_string(index=False))
    
    # Save summary table
    summary_df.to_csv("four_model_performance_summary.csv", index=False)
    print("\nSaved summary table to 'four_model_performance_summary.csv'")
    
    # Create plots
    print("\nGenerating comparison plots...")
    
    # Main comparison plot
    plot_comparison_summary(analysis_results)
    print("Saved comparison summary to 'four_model_comparison_summary.png'")
    
    # Achievement breakdown
    plot_achievement_breakdown(analysis_results, achievement_keys)
    print("Saved achievement breakdown to 'four_model_achievement_breakdown.png'")
    
    # Success rates
    plot_success_rates(analysis_results, achievement_keys)
    print("Saved success rates to 'four_model_success_rates.png'")
    
    # Reward progression
    plot_reward_progression(analysis_results)
    print("Saved reward progression to 'four_model_reward_progression.png'")
    
    # Algorithm family comparison
    plot_algorithm_family_comparison(analysis_results)
    print("Saved algorithm family comparison to 'algorithm_family_comparison.png'")
    
    # Print key insights
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    
    for agent in ['DQN Baseline', 'DQN Improved', 'PPO Baseline', 'PPO Improved']:
        if agent in analysis_results:
            data = analysis_results[agent]
            mean_reward = np.mean(data['total_rewards'])
            mean_achievements = np.mean(data['achievement_counts_per_episode'])
            
            # Find top achievements
            achievement_means = {}
            for achievement in achievement_keys:
                achievement_means[achievement] = np.mean(data['achievements'][achievement])
            
            top_achievements = sorted(achievement_means.items(), key=lambda x: x[1], reverse=True)[:3]
            
            print(f"\n{agent}:")
            print(f"  • Average Reward: {mean_reward:.2f}")
            print(f"  • Average Unique Achievements per Episode: {mean_achievements:.2f}")
            print(f"  • Top 3 Achievements:")
            for achievement, rate in top_achievements:
                achievement_name = achievement.replace('achievement_', '').replace('_', ' ').title()
                print(f"    - {achievement_name}: {rate:.2f} per episode")

if __name__ == "__main__":
    main()