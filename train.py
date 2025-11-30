# train.py - Improved training script with proper metrics tracking
import os
import time
import random
import argparse
import numpy as np
import torch

from grid_env import ComplexGridEnv as GridEnv
from dqn_agent import DQNAgent
from config import Config
from utils import setup_logger, MetricsTracker, save_checkpoint, load_checkpoint
from evaluate import evaluate_agent


def train(config: Config = None, resume_from: str = None):
    """
    Train DQN agent with comprehensive metrics tracking and evaluation.
    
    Args:
        config: Configuration object
        resume_from: Path to checkpoint to resume training from
    """
    if config is None:
        config = Config()
    
    # Setup directories
    for dir_path in [config.training.MODEL_DIR, config.training.FRAME_DIR, 
                     config.training.LOG_DIR, config.training.RESULTS_DIR]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Setup logging
    log_file = os.path.join(config.training.LOG_DIR, 'training.log')
    logger = setup_logger('training', log_file)
    
    # Set random seeds
    random.seed(config.training.SEED)
    np.random.seed(config.training.SEED)
    torch.manual_seed(config.training.SEED)
    
    # Create environment and agent
    env = GridEnv(
        size=config.env.GRID_SIZE,
        obstacle_prob=config.env.OBSTACLE_PROB,
        n_moving=config.env.N_MOVING_OBSTACLES,
        local_view=config.env.LOCAL_VIEW_RADIUS,
        max_steps=config.env.MAX_STEPS,
        seed=config.training.SEED
    )
    
    obs_dim = env.observation_space_dim
    n_actions = env.action_space_n
    
    # Create agent with config parameters
    agent = DQNAgent(
        obs_dim=obs_dim,
        n_actions=n_actions,
        lr=config.dqn.LEARNING_RATE,
        gamma=config.dqn.GAMMA,
        hidden_layers=config.dqn.HIDDEN_LAYERS,
        use_dueling=config.dqn.USE_DUELING,
        replay_size=config.dqn.REPLAY_BUFFER_SIZE,
        batch_size=config.dqn.BATCH_SIZE,
        min_replay=config.dqn.MIN_REPLAY_SIZE,
        target_update=config.dqn.TARGET_UPDATE_FREQUENCY,
        gradient_clip=config.dqn.GRADIENT_CLIP,
        device=config.dqn.DEVICE
    )
    
    logger.info("="*60)
    logger.info("TRAINING CONFIGURATION")
    logger.info("="*60)
    logger.info(f"Device: {agent.device}")
    logger.info(f"Observation dim: {obs_dim}, Action space: {n_actions}")
    logger.info(f"Network: {'Dueling DQN' if config.dqn.USE_DUELING else 'Standard DQN'}")
    logger.info(f"Hidden layers: {config.dqn.HIDDEN_LAYERS}")
    logger.info(f"Replay buffer: {config.dqn.REPLAY_BUFFER_SIZE}, Min replay: {config.dqn.MIN_REPLAY_SIZE}")
    logger.info(f"Batch size: {config.dqn.BATCH_SIZE}, Target update: {config.dqn.TARGET_UPDATE_FREQUENCY}")
    logger.info(f"Episodes: {config.training.NUM_EPISODES}")
    logger.info("="*60)
    
    # Initialize metrics tracker
    metrics_path = os.path.join(config.training.RESULTS_DIR, config.training.METRICS_FILE)
    metrics = MetricsTracker(save_path=metrics_path if config.training.SAVE_METRICS else None)
    
    # Exploration parameters
    epsilon = config.dqn.EPSILON_START
    start_episode = 1
    
    # Resume from checkpoint if specified
    if resume_from:
        logger.info(f"Resuming training from {resume_from}")
        checkpoint_data = load_checkpoint(agent, resume_from, load_replay=True)
        start_episode = checkpoint_data['episode'] + 1
        epsilon = checkpoint_data['epsilon']
        
        # Restore metrics if available
        if 'metrics' in checkpoint_data:
            metrics_data = checkpoint_data['metrics']
            metrics.episode_rewards = metrics_data.get('episode_rewards', [])
            metrics.episode_lengths = metrics_data.get('episode_lengths', [])
            metrics.episode_successes = metrics_data.get('episode_successes', [])
            metrics.losses = metrics_data.get('losses', [])
        
        logger.info(f"Resumed from episode {start_episode}, epsilon={epsilon:.3f}")
    
    # Best model tracking (using actual rewards now!)
    best_avg_reward = -float('inf')
    
    # Training loop
    logger.info("Starting training...")
    training_start_time = time.time()
    
    try:
        for episode in range(start_episode, config.training.NUM_EPISODES + 1):
            # Create dynamic environment for this episode
            if config.env.CURRICULUM_ENABLED:
                # Simple curriculum: gradually increase difficulty
                progress = (episode - 1) / config.training.NUM_EPISODES
                dynamic_size = int(config.env.GRID_SIZE + progress * 5)
                dynamic_obstacle_prob = config.env.OBSTACLE_PROB + progress * 0.1
                dynamic_n_moving = config.env.N_MOVING_OBSTACLES + int(progress * 3)
            else:
                # Random variation
                size_offset = random.randint(*config.env.DYNAMIC_SIZE_RANGE)
                dynamic_size = config.env.GRID_SIZE + size_offset
                
                obstacle_offset = random.uniform(*config.env.DYNAMIC_OBSTACLE_RANGE)
                dynamic_obstacle_prob = config.env.OBSTACLE_PROB + obstacle_offset
                
                dynamic_n_moving = random.randint(*config.env.DYNAMIC_MOVING_RANGE)
            
            env = GridEnv(
                size=dynamic_size,
                obstacle_prob=dynamic_obstacle_prob,
                n_moving=dynamic_n_moving,
                max_steps=config.env.MAX_STEPS,
                seed=config.training.SEED + episode
            )
            
            # Episode execution
            state = env.reset()
            done = False
            episode_reward = 0.0
            episode_steps = 0
            
            # Frame saving setup if needed
            save_dir = None
            do_render = (config.training.RENDER_FREQUENCY > 0 and 
                        episode % config.training.RENDER_FREQUENCY == 0)
            if do_render and config.training.SAVE_FRAMES:
                save_dir = os.path.join(config.training.FRAME_DIR, f"ep{episode:04d}")
                os.makedirs(save_dir, exist_ok=True)
            
            # Episode loop
            while not done:
                action = agent.act(state, epsilon)
                next_state, reward, done, _ = env.step(action)
                
                # STABILITY FIX: Clip rewards to reduce variance and prevent Q-value explosion
                # This helps stabilize learning by bounding the range of TD targets
                clipped_reward = np.clip(reward, config.reward.REWARD_CLIP_MIN, config.reward.REWARD_CLIP_MAX)
                
                # Store transition with clipped reward
                agent.store(state, action, clipped_reward, next_state, done)
                
                # Train agent
                loss = agent.train_step()
                if loss is not None:
                    metrics.add_loss(loss)
                
                state = next_state
                episode_reward += reward
                episode_steps += 1
                
                # Rendering
                if do_render:
                    env.render(show=True)
                    if config.training.SAVE_FRAMES and save_dir:
                        fname = os.path.join(save_dir, f"step{episode_steps:04d}.png")
                        env.render(show=False, save_path=fname)
                    time.sleep(0.005)
            
            # Check if episode was successful
            success = (env.agent == env.goal)
            
            # Update metrics
            replay_size = len(agent.replay)
            stats = metrics.add_episode(
                episode=episode,
                reward=episode_reward,
                length=episode_steps,
                success=success,
                epsilon=epsilon,
                replay_size=replay_size
            )
            
            # Decay epsilon
            epsilon = max(config.dqn.EPSILON_END, epsilon * config.dqn.EPSILON_DECAY)
            
            # Logging
            if episode % config.training.LOG_FREQUENCY == 0:
                logger.info(
                    f"Ep {episode:04d} | Reward: {episode_reward:7.2f} | "
                    f"Steps: {episode_steps:3d} | Success: {success} | "
                    f"Replay: {replay_size:5d} | ε: {epsilon:.3f}"
                )
            
            # Detailed summary
            if episode % config.training.SUMMARY_FREQUENCY == 0:
                logger.info("-" * 60)
                logger.info(f"Summary at Episode {episode}:")
                logger.info(f"  Avg Reward (last 100):  {stats['avg_reward']:.2f}")
                logger.info(f"  Avg Length (last 100):  {stats['avg_length']:.1f}")
                logger.info(f"  Success Rate (last 100): {stats['success_rate']:.1f}%")
                logger.info(f"  Avg Loss (last 100):    {stats['avg_loss']:.4f}")
                logger.info(f"  Epsilon: {epsilon:.3f}")
                
                # STABILITY FIX: Log Q-value statistics to detect divergence
                q_stats = agent.get_q_stats(window=100)
                if q_stats:
                    logger.info(f"  Q-values (last 100): Mean={q_stats['q_mean']:.2f}, "
                              f"Max={q_stats['q_max']:.2f}, Min={q_stats['q_min']:.2f}, "
                              f"Std={q_stats['q_std']:.2f}")
                    # Warning if Q-values are exploding
                    if abs(q_stats['q_mean']) > 1000 or abs(q_stats['q_max']) > 5000:
                        logger.warning("  ⚠️  Q-VALUES MAY BE DIVERGING! Consider reducing learning rate.")
                
                logger.info("-" * 60)
            
            # Evaluation
            if config.training.EVAL_FREQUENCY > 0 and episode % config.training.EVAL_FREQUENCY == 0:
                logger.info("Running evaluation...")
                eval_results = evaluate_agent(
                    agent,
                    n_episodes=config.training.EVAL_EPISODES,
                    grid_size=config.env.GRID_SIZE,
                    obstacle_prob=config.env.OBSTACLE_PROB,
                    epsilon=config.training.EVAL_EPSILON,
                    seed=config.training.SEED + 10000
                )
                logger.info(f"  Eval Mean Reward: {eval_results['mean_reward']:.2f}")
                logger.info(f"  Eval Success Rate: {eval_results['success_rate']:.1f}%")
            
            # Save best model (FIXED: using actual average reward!)
            if episode % config.training.BEST_MODEL_CHECK_FREQUENCY == 0:
                recent_rewards = metrics.episode_rewards[-100:]  # Last 100 episodes
                avg_reward = np.mean(recent_rewards) if recent_rewards else -float('inf')
                
                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    best_path = os.path.join(config.training.MODEL_DIR, "dqn_best.pt")
                    torch.save(agent.policy.state_dict(), best_path)
                    logger.info(f"  ★ New best model! Avg reward: {best_avg_reward:.2f} → {best_path}")
            
            # Periodic checkpoints
            if episode % config.training.CHECKPOINT_FREQUENCY == 0:
                checkpoint_path = os.path.join(config.training.MODEL_DIR, f"checkpoint_ep{episode}.pt")
                save_checkpoint(agent, episode, metrics, epsilon, checkpoint_path)
                logger.info(f"  Checkpoint saved: {checkpoint_path}")
            
            # Close renderer if opened
            if do_render:
                env.close_renderer()
    
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
    
    finally:
        try:
            env.close_renderer()
        except:
            pass
        
        # Final save
        final_path = os.path.join(config.training.MODEL_DIR, "dqn_final.pt")
        torch.save(agent.policy.state_dict(), final_path)
        
        final_checkpoint = os.path.join(config.training.MODEL_DIR, "checkpoint_final.pt")
        save_checkpoint(agent, episode, metrics, epsilon, final_checkpoint)
        
        logger.info(f"\nFinal model saved: {final_path}")
        logger.info(f"Final checkpoint saved: {final_checkpoint}")
        
        # Training summary
        training_time = time.time() - training_start_time
        logger.info("\n" + "="*60)
        logger.info("TRAINING COMPLETED")
        logger.info("="*60)
        logger.info(f"Total episodes: {len(metrics.episode_rewards)}")
        logger.info(f"Training time: {training_time/3600:.2f} hours")
        logger.info(f"Best avg reward: {best_avg_reward:.2f}")
        
        final_stats = metrics.get_stats()
        logger.info(f"Final avg reward: {final_stats['avg_reward']:.2f}")
        logger.info(f"Final success rate: {final_stats['success_rate']:.1f}%")
        logger.info(f"Total successes: {final_stats['total_successes']}")
        logger.info("="*60)
        
        # Plot training curves
        if config.training.SAVE_METRICS:
            plot_path = os.path.join(config.training.RESULTS_DIR, "training_curves.png")
            metrics.plot_training_curves(plot_path)
            logger.info(f"Training curves saved: {plot_path}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN agent on Grid Navigation")
    parser.add_argument("--episodes", type=int, default=None,
                       help="Number of training episodes (default: from config)")
    parser.add_argument("--grid-size", type=int, default=None,
                       help="Grid size (default: from config)")
    parser.add_argument("--obstacle-prob", type=float, default=None,
                       help="Obstacle probability (default: from config)")
    parser.add_argument("--render-every", type=int, default=None,
                       help="Render every N episodes (default: from config)")
    parser.add_argument("--save-frames", action="store_true",
                       help="Save rendered frames")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed (default: from config)")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume training from")
    parser.add_argument("--no-dueling", action="store_true",
                       help="Disable Dueling DQN architecture")
    parser.add_argument("--eval-freq", type=int, default=None,
                       help="Evaluation frequency in episodes")
    
    args = parser.parse_args()
    
    # Load config and override with command line args
    config = Config()
    
    if args.episodes is not None:
        config.training.NUM_EPISODES = args.episodes
    if args.grid_size is not None:
        config.env.GRID_SIZE = args.grid_size
    if args.obstacle_prob is not None:
        config.env.OBSTACLE_PROB = args.obstacle_prob
    if args.render_every is not None:
        config.training.RENDER_FREQUENCY = args.render_every
    if args.save_frames:
        config.training.SAVE_FRAMES = True
    if args.seed is not None:
        config.training.SEED = args.seed
    if args.no_dueling:
        config.dqn.USE_DUELING = False
    if args.eval_freq is not None:
        config.training.EVAL_FREQUENCY = args.eval_freq
    
    # Run training
    train(config=config, resume_from=args.resume)

