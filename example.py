import argparse
import torch
import numpy as np
import time  # Add this import at the top if not already present
from grid_env import ComplexGridEnv as GridEnv
from dqn_agent import DQNAgent

def evaluate_model(model_path, episodes=10, grid_size=10, obstacle_prob=0.5, seed=0, render=False, epsilon=0.05):
    """
    Evaluate the model on the environment.

    Args:
        model_path (str): Path to the model to load.
        episodes (int): Number of episodes to evaluate.
        grid_size (int): Size of the grid environment.
        obstacle_prob (float): Probability of obstacles in the grid.
        seed (int): Random seed for reproducibility.
        render (bool): Whether to render the environment.
        epsilon (float): Epsilon value for epsilon-greedy policy during evaluation.
    """
    # Set random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Initialize environment
    env = GridEnv(size=grid_size, obstacle_prob=obstacle_prob, max_steps=100, seed=seed)
    obs_dim = env.observation_space_dim
    n_actions = env.action_space_n

    # Initialize agent
    agent = DQNAgent(obs_dim, n_actions, device=None)
    agent.policy.load_state_dict(torch.load(model_path))
    agent.policy.eval()

    for ep in range(1, episodes + 1):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.act(state, eps=epsilon)  # Epsilon-greedy action selection
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            state = next_state

            if render:
                env.render()

            time.sleep(0.1)  # Add a delay of 0.1 seconds to slow down the step process
            if done:
                break

        print(f"Episode {ep}: Total Reward: {total_reward}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to the trained model.")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to evaluate.")
    parser.add_argument("--grid-size", type=int, default=10, help="Size of the grid environment.")
    parser.add_argument("--obstacle-prob", type=float, default=0.5, help="Probability of obstacles in the grid.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    parser.add_argument("--render", action="store_true", help="Render the environment during evaluation.")
    parser.add_argument("--epsilon", type=float, default=0.05, help="Epsilon value for epsilon-greedy policy during evaluation.")
    args = parser.parse_args()

    evaluate_model(
        model_path=args.model_path,
        episodes=args.episodes,
        grid_size=args.grid_size,
        obstacle_prob=args.obstacle_prob,
        seed=args.seed,
        render=args.render,
        epsilon=args.epsilon
    )
