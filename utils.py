# utils.py - Utility functions for logging, metrics, and visualization

import os
import csv
import logging
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from typing import Dict, List, Optional, Any


def setup_logger(name: str, log_file: Optional[str] = None, level=logging.INFO) -> logging.Logger:
    """Setup logger with both file and console handlers"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers = []
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class MetricsTracker:
    """Track and save training metrics"""
    
    def __init__(self, save_path: Optional[str] = None, window_size: int = 100):
        self.save_path = save_path
        self.window_size = window_size
        
        # Metrics storage
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.episode_successes: List[bool] = []
        self.losses: List[float] = []
        self.epsilon_values: List[float] = []
        self.replay_sizes: List[int] = []
        
        # Moving averages
        self.reward_window = deque(maxlen=window_size)
        self.length_window = deque(maxlen=window_size)
        self.success_window = deque(maxlen=window_size)
        
        # Initialize CSV if save path provided
        if save_path:
            self._init_csv()
    
    def _init_csv(self):
        """Initialize CSV file with headers"""
        os.makedirs(os.path.dirname(self.save_path) if os.path.dirname(self.save_path) else '.', exist_ok=True)
        with open(self.save_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'episode', 'reward', 'length', 'success', 'epsilon', 
                'replay_size', 'avg_reward', 'avg_length', 'success_rate', 'avg_loss'
            ])
    
    def add_episode(self, episode: int, reward: float, length: int, 
                   success: bool, epsilon: float, replay_size: int, 
                   recent_loss: Optional[float] = None):
        """Add episode metrics"""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.episode_successes.append(success)
        self.epsilon_values.append(epsilon)
        self.replay_sizes.append(replay_size)
        
        self.reward_window.append(reward)
        self.length_window.append(length)
        self.success_window.append(float(success))
        
        # Calculate averages
        avg_reward = np.mean(self.reward_window)
        avg_length = np.mean(self.length_window)
        success_rate = np.mean(self.success_window) * 100
        avg_loss = np.mean(self.losses[-100:]) if self.losses else 0.0
        
        # Save to CSV
        if self.save_path:
            with open(self.save_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    episode, reward, length, success, epsilon, 
                    replay_size, avg_reward, avg_length, success_rate, avg_loss
                ])
        
        return {
            'avg_reward': avg_reward,
            'avg_length': avg_length,
            'success_rate': success_rate,
            'avg_loss': avg_loss
        }
    
    def add_loss(self, loss: float):
        """Add training loss"""
        self.losses.append(loss)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics"""
        return {
            'total_episodes': len(self.episode_rewards),
            'avg_reward': np.mean(self.reward_window) if self.reward_window else 0.0,
            'avg_length': np.mean(self.length_window) if self.length_window else 0.0,
            'success_rate': np.mean(self.success_window) * 100 if self.success_window else 0.0,
            'avg_loss': np.mean(self.losses[-100:]) if self.losses else 0.0,
            'best_reward': max(self.episode_rewards) if self.episode_rewards else 0.0,
            'total_successes': sum(self.episode_successes),
        }
    
    def plot_training_curves(self, save_path: str = 'training_curves.png'):
        """Plot training curves"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Rewards
        axes[0, 0].plot(self.episode_rewards, alpha=0.3, label='Episode Reward')
        if len(self.episode_rewards) >= self.window_size:
            smoothed = np.convolve(self.episode_rewards, 
                                  np.ones(self.window_size)/self.window_size, 
                                  mode='valid')
            axes[0, 0].plot(range(self.window_size-1, len(self.episode_rewards)), 
                           smoothed, label=f'{self.window_size}-Episode Average')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Success Rate
        if len(self.episode_successes) >= 10:
            success_rate = []
            for i in range(9, len(self.episode_successes)):
                rate = np.mean(self.episode_successes[max(0, i-self.window_size+1):i+1]) * 100
                success_rate.append(rate)
            axes[0, 1].plot(range(10, len(self.episode_successes)+1), success_rate)
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Success Rate (%)')
            axes[0, 1].set_title(f'Success Rate (Rolling {self.window_size}-Episode Window)')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Episode Length
        axes[1, 0].plot(self.episode_lengths, alpha=0.3, label='Episode Length')
        if len(self.episode_lengths) >= self.window_size:
            smoothed = np.convolve(self.episode_lengths, 
                                  np.ones(self.window_size)/self.window_size, 
                                  mode='valid')
            axes[1, 0].plot(range(self.window_size-1, len(self.episode_lengths)), 
                           smoothed, label=f'{self.window_size}-Episode Average')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Steps')
        axes[1, 0].set_title('Episode Length')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Loss
        if self.losses:
            axes[1, 1].plot(self.losses, alpha=0.2)
            if len(self.losses) >= 100:
                smoothed = np.convolve(self.losses, np.ones(100)/100, mode='valid')
                axes[1, 1].plot(range(99, len(self.losses)), smoothed, label='100-Step Average')
            axes[1, 1].set_xlabel('Training Step')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].set_title('Training Loss')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def save_checkpoint(agent, episode: int, metrics: MetricsTracker, 
                   epsilon: float, filepath: str):
    """Save complete training checkpoint"""
    import torch
    
    checkpoint = {
        'episode': episode,
        'policy_state_dict': agent.policy.state_dict(),
        'target_state_dict': agent.target.state_dict(),
        'optimizer_state_dict': agent.optim.state_dict(),
        'epsilon': epsilon,
        'replay_buffer': list(agent.replay),
        'steps': agent.steps,
        'metrics': {
            'episode_rewards': metrics.episode_rewards,
            'episode_lengths': metrics.episode_lengths,
            'episode_successes': metrics.episode_successes,
            'losses': metrics.losses,
            'epsilon_values': metrics.epsilon_values,
            'replay_sizes': metrics.replay_sizes,
        }
    }
    
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    torch.save(checkpoint, filepath)


def load_checkpoint(agent, filepath: str, load_replay: bool = True):
    """Load training checkpoint"""
    import torch
    from collections import deque
    
    checkpoint = torch.load(filepath, map_location=agent.device)
    
    agent.policy.load_state_dict(checkpoint['policy_state_dict'])
    agent.target.load_state_dict(checkpoint['target_state_dict'])
    agent.optim.load_state_dict(checkpoint['optimizer_state_dict'])
    agent.steps = checkpoint.get('steps', 0)
    
    if load_replay and 'replay_buffer' in checkpoint:
        agent.replay = deque(checkpoint['replay_buffer'], maxlen=agent.replay.maxlen)
    
    metrics_data = checkpoint.get('metrics', {})
    
    return {
        'episode': checkpoint['episode'],
        'epsilon': checkpoint['epsilon'],
        'metrics': metrics_data
    }
