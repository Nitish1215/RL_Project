# visualize_results.py - Visualize training results from CSV

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_training_results(csv_path: str, save_dir: str = "results", show: bool = True):
    """
    Create comprehensive visualization of training results.
    
    Args:
        csv_path: Path to training_metrics.csv
        save_dir: Directory to save plots
        show: Whether to display plots
    """
    # Load data
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    
    print(f"Loaded {len(df)} episodes from {csv_path}")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Episode Rewards
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(df['episode'], df['reward'], alpha=0.3, color='blue', label='Episode Reward')
    ax1.plot(df['episode'], df['avg_reward'], color='darkblue', linewidth=2, label='Rolling Average')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Episode Rewards')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Success Rate
    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(df['episode'], df['success_rate'], color='green', linewidth=2)
    ax2.fill_between(df['episode'], 0, df['success_rate'], alpha=0.3, color='green')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Success Rate (%)')
    ax2.set_title('Success Rate (Rolling 100 Episodes)')
    ax2.set_ylim([0, 105])
    ax2.grid(True, alpha=0.3)
    
    # 3. Episode Length
    ax3 = plt.subplot(3, 3, 3)
    ax3.plot(df['episode'], df['length'], alpha=0.3, color='orange', label='Episode Length')
    ax3.plot(df['episode'], df['avg_length'], color='darkorange', linewidth=2, label='Rolling Average')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Steps')
    ax3.set_title('Episode Length')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Average Loss
    ax4 = plt.subplot(3, 3, 4)
    ax4.plot(df['episode'], df['avg_loss'], color='red', linewidth=1.5)
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Loss')
    ax4.set_title('Training Loss (Rolling 100 Steps)')
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    
    # 5. Epsilon Decay
    ax5 = plt.subplot(3, 3, 5)
    ax5.plot(df['episode'], df['epsilon'], color='purple', linewidth=2)
    ax5.set_xlabel('Episode')
    ax5.set_ylabel('Epsilon')
    ax5.set_title('Exploration Rate')
    ax5.grid(True, alpha=0.3)
    
    # 6. Replay Buffer Size
    ax6 = plt.subplot(3, 3, 6)
    ax6.plot(df['episode'], df['replay_size'], color='brown', linewidth=1.5)
    ax6.set_xlabel('Episode')
    ax6.set_ylabel('Buffer Size')
    ax6.set_title('Replay Buffer Size')
    ax6.grid(True, alpha=0.3)
    
    # 7. Reward Distribution
    ax7 = plt.subplot(3, 3, 7)
    ax7.hist(df['reward'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    ax7.axvline(df['reward'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["reward"].mean():.2f}')
    ax7.axvline(df['reward'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {df["reward"].median():.2f}')
    ax7.set_xlabel('Reward')
    ax7.set_ylabel('Frequency')
    ax7.set_title('Reward Distribution')
    ax7.legend()
    ax7.grid(True, alpha=0.3, axis='y')
    
    # 8. Success vs Episode (scatter)
    ax8 = plt.subplot(3, 3, 8)
    successes = df[df['success'] == True]
    failures = df[df['success'] == False]
    ax8.scatter(failures['episode'], failures['reward'], alpha=0.3, s=10, color='red', label='Failed')
    ax8.scatter(successes['episode'], successes['reward'], alpha=0.5, s=10, color='green', label='Success')
    ax8.set_xlabel('Episode')
    ax8.set_ylabel('Reward')
    ax8.set_title('Success/Failure Scatter')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # 9. Cumulative Success Count
    ax9 = plt.subplot(3, 3, 9)
    cumulative_success = df['success'].cumsum()
    ax9.plot(df['episode'], cumulative_success, color='darkgreen', linewidth=2)
    ax9.fill_between(df['episode'], 0, cumulative_success, alpha=0.3, color='green')
    ax9.set_xlabel('Episode')
    ax9.set_ylabel('Total Successes')
    ax9.set_title('Cumulative Successes')
    ax9.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'detailed_training_analysis.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved detailed analysis plot to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    # Print statistics
    print("\n" + "="*60)
    print("TRAINING STATISTICS")
    print("="*60)
    print(f"Total Episodes:        {len(df)}")
    print(f"Total Successes:       {df['success'].sum()}")
    print(f"Final Success Rate:    {df['success_rate'].iloc[-1]:.2f}%")
    print(f"Mean Reward:           {df['reward'].mean():.2f} ± {df['reward'].std():.2f}")
    print(f"Final Avg Reward:      {df['avg_reward'].iloc[-1]:.2f}")
    print(f"Best Avg Reward:       {df['avg_reward'].max():.2f} (Episode {df['avg_reward'].idxmax() + 1})")
    print(f"Mean Episode Length:   {df['length'].mean():.1f} ± {df['length'].std():.1f}")
    print(f"Final Epsilon:         {df['epsilon'].iloc[-1]:.4f}")
    print(f"Final Replay Size:     {df['replay_size'].iloc[-1]}")
    print("="*60)
    
    # Learning milestones
    print("\nLEARNING MILESTONES")
    print("-"*60)
    
    # First success
    first_success = df[df['success'] == True]
    if len(first_success) > 0:
        print(f"First Success:         Episode {first_success.iloc[0]['episode']}")
    
    # 50% success rate
    high_success = df[df['success_rate'] >= 50.0]
    if len(high_success) > 0:
        print(f"50% Success Rate:      Episode {high_success.iloc[0]['episode']}")
    
    # Best performance window
    if len(df) >= 100:
        best_window_idx = df['avg_reward'].idxmax()
        best_window_start = max(0, best_window_idx - 50)
        best_window_end = min(len(df), best_window_idx + 50)
        best_window = df.iloc[best_window_start:best_window_end]
        print(f"Best Performance:      Episodes {best_window_start}-{best_window_end}")
        print(f"  Avg Reward:          {best_window['reward'].mean():.2f}")
        print(f"  Success Rate:        {best_window['success'].mean() * 100:.1f}%")
    
    print("-"*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize training results")
    parser.add_argument("--csv", type=str, default="results/training_metrics.csv",
                       help="Path to training metrics CSV file")
    parser.add_argument("--save-dir", type=str, default="results",
                       help="Directory to save plots")
    parser.add_argument("--no-show", action="store_true",
                       help="Don't display plots (just save)")
    
    args = parser.parse_args()
    
    plot_training_results(
        csv_path=args.csv,
        save_dir=args.save_dir,
        show=not args.no_show
    )
