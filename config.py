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
    OSCILLATION_PENALTY = -0.5


class DQNConfig:
    """DQN agent hyperparameters"""
    # Network architecture
    HIDDEN_LAYERS = [256, 256]  # Deeper network
    USE_DUELING = True  # Use Dueling DQN architecture
    
    # Training
    LEARNING_RATE = 5e-4
    GAMMA = 0.99
    BATCH_SIZE = 64
    REPLAY_BUFFER_SIZE = 50000
    MIN_REPLAY_SIZE = 1000
    TARGET_UPDATE_FREQUENCY = 500
    GRADIENT_CLIP = 10.0
    
    # Exploration
    EPSILON_START = 1.0
    EPSILON_END = 0.05
    EPSILON_DECAY = 0.995
    
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
