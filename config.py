# config.py - Centralized configuration for RL training

class EnvironmentConfig:
    """Grid environment configuration"""
    GRID_SIZE = 10
    OBSTACLE_PROB = 0.3
    N_MOVING_OBSTACLES = 3
    LOCAL_VIEW_RADIUS = 2
    MAX_STEPS = 200
    CONNECTIVITY_TRIES = 50
    
    # Curriculum learning
    CURRICULUM_ENABLED = True
    CURRICULUM_START_SIZE = 8
    CURRICULUM_MAX_SIZE = 15
    CURRICULUM_SIZE_INCREMENT = 1
    CURRICULUM_OBSTACLE_INCREMENT = 0.05
    CURRICULUM_MOVING_INCREMENT = 1
    
    # Dynamic environment during training
    DYNAMIC_SIZE_RANGE = (0, 6)  # (min_offset, max_offset) from base size
    DYNAMIC_OBSTACLE_RANGE = (0.0, 0.15)  # (min_offset, max_offset) from base prob
    DYNAMIC_MOVING_RANGE = (3, 8)  # (min, max) moving obstacles


class RewardConfig:
    """Reward shaping constants"""
    STEP_PENALTY = -0.05
    GOAL_REWARD = 20.0
    COLLISION_PENALTY = -1.0
    PROGRESS_BONUS = 0.5
    PROGRESS_PENALTY = -0.1
    OSCILLATION_PENALTY = -2.0  # ANTI-OSCILLATION FIX: Increased from -0.5 to strongly discourage revisiting positions
    STUCK_PENALTY = -1.5  # NEW: Penalty for staying in same small area for too long
    
    # STABILITY FIX: Reward clipping to reduce variance and stabilize Q-learning
    REWARD_CLIP_MIN = -10.0  # Prevents extreme negative rewards from destabilizing training
    REWARD_CLIP_MAX = 25.0   # Prevents extreme positive rewards from causing Q-value explosion


class DQNConfig:
    """DQN agent hyperparameters"""
    # Network architecture
    HIDDEN_LAYERS = [256, 256]  # Deeper network
    USE_DUELING = True  # Use Dueling DQN architecture
    
    # Training
    LEARNING_RATE = 1e-4  # STABILITY FIX: Reduced from 5e-4 to prevent Q-value divergence
    GAMMA = 0.99
    BATCH_SIZE = 128  # STABILITY FIX: Increased from 64 for more stable gradient estimates
    REPLAY_BUFFER_SIZE = 100000  # STABILITY FIX: Increased from 50k for better experience diversity
    MIN_REPLAY_SIZE = 2000  # STABILITY FIX: Increased to ensure sufficient sampling diversity
    TARGET_UPDATE_FREQUENCY = 1000  # STABILITY FIX: Increased from 500 to reduce overestimation bias
    GRADIENT_CLIP = 10.0  # Already good - prevents exploding gradients
    
    # Exploration
    # STABILITY FIX: Epsilon-greedy exploration schedule
    # Formula: epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY) each episode
    EPSILON_START = 1.0
    EPSILON_END = 0.1  # STABILITY FIX: Increased from 0.05 to maintain exploration
                       # Higher minimum prevents policy collapse from lack of exploration
    EPSILON_DECAY = 0.9985  # STABILITY FIX: Slower decay from 0.995
                            # Reaches ~0.37 after 500 eps, ~0.14 after 1000 eps
                            # Prevents premature convergence to suboptimal policy
    
    # Device
    DEVICE = None  # None means auto-detect


class TrainingConfig:
    """Training loop configuration"""
    NUM_EPISODES = 1000
    SEED = 42
    
    # Checkpointing
    CHECKPOINT_FREQUENCY = 100
    BEST_MODEL_CHECK_FREQUENCY = 10
    
    # Evaluation
    EVAL_FREQUENCY = 50
    EVAL_EPISODES = 10
    EVAL_EPSILON = 0.0  # Greedy evaluation
    
    # Rendering
    RENDER_FREQUENCY = 0  # 0 means no rendering during training
    SAVE_FRAMES = False
    
    # Logging
    LOG_FREQUENCY = 1  # Log every N episodes
    SUMMARY_FREQUENCY = 10  # Detailed summary every N episodes
    SAVE_METRICS = True
    METRICS_FILE = "training_metrics.csv"
    
    # Directories
    MODEL_DIR = "models"
    FRAME_DIR = "frames"
    LOG_DIR = "logs"
    RESULTS_DIR = "results"


class Config:
    """Master configuration class"""
    env = EnvironmentConfig()
    reward = RewardConfig()
    dqn = DQNConfig()
    training = TrainingConfig()
