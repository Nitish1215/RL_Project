# evaluate.py - Evaluation utilities

import numpy as np
import torch
from typing import Dict, Optional
from grid_env import ComplexGridEnv as GridEnv
from dqn_agent import DQNAgent


def evaluate_agent(agent: DQNAgent, n_episodes: int = 10, 
                   grid_size: int = 10, obstacle_prob: float = 0.3,
                   n_moving: int = 3, max_steps: int = 200,
                   epsilon: float = 0.0, seed: Optional[int] = None,
                   render: bool = False, verbose: bool = False) -> Dict:
    """
    Evaluate agent performance on test episodes.
    
    Args:
        agent: DQN agent to evaluate
        n_episodes: Number of evaluation episodes
        grid_size: Grid environment size
        obstacle_prob: Probability of obstacles
        n_moving: Number of moving obstacles
        max_steps: Maximum steps per episode
        epsilon: Exploration rate (0.0 for greedy)
        seed: Random seed for reproducibility
        render: Whether to render episodes
        verbose: Whether to print episode details
    
    Returns:
        Dictionary with evaluation metrics
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    # Create evaluation environment
    env = GridEnv(size=grid_size, obstacle_prob=obstacle_prob, 
                  n_moving=n_moving, max_steps=max_steps, seed=seed)
    
    # Metrics
    episode_rewards = []
    episode_lengths = []
    successes = []
    
    agent.policy.eval()  # Set to evaluation mode
    
    for ep in range(n_episodes):
        state = env.reset()
        done = False
        episode_reward = 0.0
        steps = 0
        
        while not done:
            action = agent.act(state, eps=epsilon)
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            steps += 1
            state = next_state
            
            if render:
                env.render(show=True)
        
        # Check if goal was reached
        success = (env.agent == env.goal)
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        successes.append(success)
        
        if verbose:
            print(f"  Eval Episode {ep+1}/{n_episodes}: Reward={episode_reward:.2f}, "
                  f"Steps={steps}, Success={success}")
        
        if render:
            env.close_renderer()
    
    agent.policy.train()  # Set back to training mode
    
    results = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths),
        'success_rate': np.mean(successes) * 100,
        'min_reward': np.min(episode_rewards),
        'max_reward': np.max(episode_rewards),
        'total_successes': sum(successes),
    }
    
    return results


def run_evaluation_demo(model_path: str, n_episodes: int = 5, 
                       grid_size: int = 10, render: bool = True):
    """
    Run evaluation demo with a saved model.
    
    Args:
        model_path: Path to saved model checkpoint
        n_episodes: Number of episodes to run
        grid_size: Grid size for evaluation
        render: Whether to render episodes
    """
    # Load model
    env = GridEnv(size=grid_size, obstacle_prob=0.3, max_steps=200)
    obs_dim = env.observation_space_dim
    n_actions = env.action_space_n
    
    agent = DQNAgent(obs_dim, n_actions)
    agent.policy.load_state_dict(torch.load(model_path, map_location=agent.device))
    
    print(f"Loaded model from {model_path}")
    print(f"Running {n_episodes} evaluation episodes...\n")
    
    results = evaluate_agent(
        agent, 
        n_episodes=n_episodes,
        grid_size=grid_size,
        render=render,
        verbose=True
    )
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Mean Reward:     {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"Mean Length:     {results['mean_length']:.1f} ± {results['std_length']:.1f}")
    print(f"Success Rate:    {results['success_rate']:.1f}%")
    print(f"Total Successes: {results['total_successes']}/{n_episodes}")
    print(f"Reward Range:    [{results['min_reward']:.2f}, {results['max_reward']:.2f}]")
    print("="*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate trained DQN agent")
    parser.add_argument("--model-path", type=str, required=True, 
                       help="Path to saved model checkpoint")
    parser.add_argument("--episodes", type=int, default=10,
                       help="Number of evaluation episodes")
    parser.add_argument("--grid-size", type=int, default=10,
                       help="Grid size")
    parser.add_argument("--render", action="store_true",
                       help="Render episodes")
    parser.add_argument("--epsilon", type=float, default=0.0,
                       help="Exploration epsilon (0.0 for greedy)")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed")
    
    args = parser.parse_args()
    
    run_evaluation_demo(
        model_path=args.model_path,
        n_episodes=args.episodes,
        grid_size=args.grid_size,
        render=args.render
    )
