# Deep Q-Network for Grid Navigation

## 📝 Introduction

This project implements a state-of-the-art **Deep Q-Network (DQN)** agent that learns to navigate complex grid environments with static and dynamic obstacles. The agent must find an optimal path from its starting position to a goal while avoiding collisions and minimizing travel time.

The implementation combines several advanced reinforcement learning techniques including **Double DQN** to reduce overestimation bias and **Dueling DQN** architecture for better value estimation. The agent learns entirely through trial and error, receiving rewards for progress and penalties for inefficient behavior.

**Key Highlights:**
- Learns navigation strategy from raw observations (no hand-coded rules)
- Handles dynamic environments with moving obstacles
- Achieves 70-90% success rate after ~1000 episodes
- Fully configurable with comprehensive monitoring and evaluation tools

---

## 🔄 Workflow Diagram

```mermaid
graph TD
    A[Start Training] --> B[Initialize Environment]
    B --> C[Initialize DQN Agent]
    C --> D{Episode Loop}
    
    D --> E[Reset Environment]
    E --> F[Get Initial State]
    F --> G{Episode Done?}
    
    G -->|No| H[Select Action<br/>ε-greedy policy]
    H --> I[Execute Action<br/>in Environment]
    I --> J[Receive Reward<br/>& Next State]
    J --> K[Store Transition<br/>in Replay Buffer]
    K --> L[Sample Batch<br/>from Replay Buffer]
    L --> M[Compute TD Target<br/>Double DQN]
    M --> N[Calculate Loss<br/>MSE Q-values]
    N --> O[Backpropagate<br/>& Update Weights]
    O --> P[Update Target Network<br/>every N steps]
    P --> G
    
    G -->|Yes| Q[Log Metrics]
    Q --> R{Evaluation<br/>Episode?}
    R -->|Yes| S[Evaluate Agent<br/>Greedy Policy]
    S --> T[Save Best Model]
    R -->|No| T
    T --> U{More Episodes?}
    U -->|Yes| D
    U -->|No| V[Save Final Model]
    V --> W[Generate Training Curves]
    W --> X[End Training]
    
    style A fill:#90EE90
    style X fill:#FFB6C1
    style M fill:#87CEEB
    style N fill:#87CEEB
    style O fill:#87CEEB
    style S fill:#FFD700
```

---

## 🎯 Features

- **Dueling DQN Architecture**: Improved value estimation with separate value and advantage streams
- **Double DQN**: Reduces overestimation bias in Q-learning
- **Experience Replay**: Breaks temporal correlation in training data
- **Target Network**: Stabilizes training by fixing Q-targets
- **Comprehensive Metrics Tracking**: CSV logging, success rates, training curves
- **Proper Checkpointing**: Resume training with full state restoration
- **Curriculum Learning**: Gradually increase environment complexity
- **Periodic Evaluation**: Test agent performance during training
- **Visualization**: Training curves and episode rendering

## 📁 Project Structure

```
RL_Project/
├── config.py           # Centralized configuration
├── dqn_agent.py       # DQN agent with Dueling architecture
├── grid_env.py        # Complex grid environment
├── train.py           # Training script with metrics tracking
├── evaluate.py        # Evaluation utilities
├── utils.py           # Logging, metrics, and checkpointing
├── example.py         # Evaluation demo script
├── requirements.txt   # Python dependencies
├── models/            # Saved model checkpoints
├── logs/              # Training logs
├── results/           # Metrics and training curves
└── frames/            # Rendered episode frames
```

---

## 🚀 Setup & Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-capable GPU for faster training

### Installation Steps

1. **Clone the repository**
```bash
git clone https://github.com/Madan2248c/RL_Project.git
cd RL_Project
```

2. **Create a virtual environment (recommended)**
```bash
# Using venv
python -m venv rl_env
source rl_env/bin/activate  # On Windows: rl_env\Scripts\activate

# Or using conda
conda create -n rl_env python=3.10
conda activate rl_env
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Verify installation**
```bash
python -c "import torch; import numpy; import matplotlib; print('All dependencies installed successfully!')"
```

### Requirements
The project requires the following packages:
- `torch>=2.0.0` - Deep learning framework
- `numpy>=1.24.0` - Numerical computations
- `matplotlib>=3.7.0` - Visualization
- `pandas>=2.0.0` - Data analysis

---

## 🎮 How to Run

### Quick Start (Default Settings)

```bash
# Start training with default configuration
python train.py
```

This will:
- Train for 1000 episodes
- Use Dueling DQN with [256, 256] hidden layers
- Save best model to `models/dqn_best.pt`
- Generate metrics in `results/training_metrics.csv`
- Create training curves in `results/training_curves.png`
- Log to `logs/training.log`

### Training with Custom Parameters

```bash
# Train for 2000 episodes on larger grid
python train.py --episodes 2000 --grid-size 15 --obstacle-prob 0.4

# Enable periodic evaluation every 50 episodes
python train.py --eval-freq 50

# Train with rendering (slower but visual)
python train.py --render-every 100

# Use standard DQN instead of Dueling
python train.py --no-dueling

# Set custom random seed for reproducibility
python train.py --seed 123
```

### Resume Training from Checkpoint

```bash
# Resume from a saved checkpoint
python train.py --resume models/checkpoint_ep500.pt

# Continue training with different settings
python train.py --resume models/checkpoint_ep500.pt --episodes 2000
```

### Evaluate Trained Model

```bash
# Evaluate with rendering
python evaluate.py --model-path models/dqn_best.pt --episodes 20 --render

# Evaluate without rendering (faster)
python evaluate.py --model-path models/dqn_best.pt --episodes 100

# Evaluate with some exploration
python evaluate.py --model-path models/dqn_best.pt --epsilon 0.1 --render
```

### Visualize Training Results

```bash
# Generate comprehensive analysis plots
python visualize_results.py

# Save without displaying
python visualize_results.py --no-show

# Specify custom CSV path
python visualize_results.py --csv results/training_metrics.csv
```

### Command-Line Arguments Reference

**Training (`train.py`):**
- `--episodes N` - Number of training episodes (default: 1000)
- `--grid-size N` - Grid environment size (default: 10)
- `--obstacle-prob P` - Static obstacle probability (default: 0.3)
- `--render-every N` - Render every N episodes (default: 0)
- `--save-frames` - Save rendered frames to disk
- `--seed N` - Random seed (default: 42)
- `--resume PATH` - Resume from checkpoint
- `--no-dueling` - Disable Dueling DQN
- `--eval-freq N` - Evaluate every N episodes (default: 50)

**Evaluation (`evaluate.py`):**
- `--model-path PATH` - Path to saved model (required)
- `--episodes N` - Number of evaluation episodes (default: 10)
- `--grid-size N` - Grid size (default: 10)
- `--render` - Render episodes
- `--epsilon E` - Exploration rate (default: 0.0)
- `--seed N` - Random seed

---

## 🧠 Core Logic & RL Concepts

### Reinforcement Learning Framework

This project implements a **Markov Decision Process (MDP)** for grid navigation, defined by the tuple $(S, A, P, R, \gamma)$:

#### 1. State Space ($S$)

The agent observes a **33-dimensional continuous state vector**:

$$s_t = [p_x, p_y, g_x, g_y, l_0, l_1, l_2, l_3, v_{00}, v_{01}, ..., v_{44}]$$

Where:
- **Agent Position** (2D, normalized): $p_x, p_y \in [0, 1]$
  - $p_x = \frac{x}{size-1}$, $p_y = \frac{y}{size-1}$
  
- **Goal Vector** (2D, normalized direction):
  - $g_x = \frac{goal_x - agent_x}{\max(1, size-1)}$
  - $g_y = \frac{goal_y - agent_y}{\max(1, size-1)}$
  - Provides relative direction and distance to goal
  
- **LIDAR Sensors** (4D, normalized distances):
  - $l_i$ = normalized distance to obstacle in direction $i$ (up, right, down, left)
  - $l_i = \frac{distance_i}{size-1} \in [0, 1]$
  - Provides obstacle proximity information
  
- **Local View** (25D for radius=2):
  - $(2r+1)^2 = 25$ binary occupancy values
  - Grid cells around agent: 1=obstacle, 0=free
  - Provides detailed local environment information

**Total state dimension**: $2 + 2 + 4 + 25 = 33$

#### 2. Action Space ($A$)

Discrete action space with 5 actions:
$$A = \{0, 1, 2, 3, 4\}$$

- **0**: Move Up ($\Delta x = -1, \Delta y = 0$)
- **1**: Move Right ($\Delta x = 0, \Delta y = 1$)
- **2**: Move Down ($\Delta x = 1, \Delta y = 0$)
- **3**: Move Left ($\Delta x = 0, \Delta y = -1$)
- **4**: Stay ($\Delta x = 0, \Delta y = 0$)

#### 3. Reward Function ($R$)

The reward function is carefully shaped to encourage efficient navigation:

$$R(s_t, a_t, s_{t+1}) = r_{step} + r_{goal} + r_{collision} + r_{progress} + r_{oscillation}$$

Where:
- **Step Penalty**: $r_{step} = -0.05$ 
  - Encourages faster solutions
  
- **Goal Reward**: $r_{goal} = \begin{cases} +20.0 & \text{if goal reached} \\ 0 & \text{otherwise} \end{cases}$
  
- **Collision Penalty**: $r_{collision} = \begin{cases} -1.0 & \text{if collision} \\ 0 & \text{otherwise} \end{cases}$
  - Collision does NOT terminate episode (agent continues)
  
- **Progress Bonus/Penalty**:
  $$r_{progress} = \begin{cases} 
    +0.5 & \text{if } d(s_{t+1}) < d(s_t) \\
    -0.1 & \text{if } d(s_{t+1}) > d(s_t) \\
    0 & \text{otherwise}
  \end{cases}$$
  where $d(s) = |agent_x - goal_x| + |agent_y - goal_y|$ (Manhattan distance)
  
- **Oscillation Penalty**: $r_{oscillation} = \begin{cases} -0.5 & \text{if position in last 5 steps} \\ 0 & \text{otherwise} \end{cases}$
  - Prevents agent from oscillating between positions

#### 4. Transition Dynamics ($P$)

Deterministic transitions: $P(s' | s, a) = 1$ for valid moves

The environment includes:
- **Static obstacles**: Fixed positions (probability $p_{obs} = 0.3$)
- **Dynamic obstacles**: $N_{moving} = 3$ obstacles that move randomly each step
- **Boundary constraints**: Grid boundaries are impassable
- **Collision handling**: Agent stays in place on collision

#### 5. Discount Factor ($\gamma$)

$$\gamma = 0.99$$

High discount factor encourages long-term planning.

---

### Deep Q-Network (DQN) Algorithm

#### Core Concept: Q-Learning

The agent learns an **action-value function** $Q(s, a)$ that estimates expected cumulative discounted reward:

$$Q^\pi(s, a) = \mathbb{E}_\pi \left[ \sum_{t=0}^{\infty} \gamma^t r_t \mid s_0=s, a_0=a \right]$$

The **optimal Q-function** satisfies the Bellman optimality equation:

$$Q^*(s, a) = \mathbb{E}_{s'} \left[ r + \gamma \max_{a'} Q^*(s', a') \mid s, a \right]$$

#### Neural Network Approximation

Since the state space is continuous and high-dimensional, we use a **deep neural network** $Q(s, a; \theta)$ to approximate $Q^*(s, a)$:

$$Q(s, a; \theta) \approx Q^*(s, a)$$

where $\theta$ represents the network parameters (weights and biases).

#### 1. Experience Replay

**Problem**: Sequential data has high temporal correlation, leading to unstable training.

**Solution**: Store transitions in a **replay buffer** $\mathcal{D}$ and sample randomly.

**Replay Buffer**: $\mathcal{D} = \{(s_t, a_t, r_t, s_{t+1}, done_t)\}$ with capacity 50,000

**Benefits**:
- Breaks temporal correlation
- Enables multiple updates from same experience
- Improves sample efficiency

#### 2. Target Network

**Problem**: Using same network for both prediction and target causes instability.

**Solution**: Maintain a separate **target network** $Q(s, a; \theta^-)$ updated periodically.

**Update Rule**:
$$\theta^- \leftarrow \theta \quad \text{every } N=500 \text{ steps}$$

#### 3. Double DQN

**Problem**: Standard DQN overestimates Q-values due to max operator bias.

**Standard DQN Target**:
$$y_t = r_t + \gamma \max_{a'} Q(s_{t+1}, a'; \theta^-)$$

**Double DQN Target** (implemented):
$$y_t = r_t + \gamma Q(s_{t+1}, \arg\max_{a'} Q(s_{t+1}, a'; \theta), \theta^-)$$

**Key Difference**: 
- Action selection: Use **online network** $\theta$
- Action evaluation: Use **target network** $\theta^-$
- Reduces overestimation bias

#### 4. Loss Function

**Mean Squared Error (MSE)** between predicted and target Q-values:

$$\mathcal{L}(\theta) = \mathbb{E}_{(s,a,r,s',d) \sim \mathcal{D}} \left[ \left( Q(s, a; \theta) - y \right)^2 \right]$$

where:
$$y = \begin{cases}
r & \text{if episode terminated (} d=1 \text{)} \\
r + \gamma Q(s', \arg\max_{a'} Q(s', a'; \theta), \theta^-) & \text{otherwise}
\end{cases}$$

#### 5. Training Algorithm

**Pseudocode**:
```
Initialize replay buffer D with capacity N
Initialize Q-network with random weights θ
Initialize target network θ⁻ = θ
For episode = 1 to M:
    Reset environment, get initial state s
    For t = 1 to T:
        # Action selection (ε-greedy)
        With probability ε: select random action a
        Otherwise: a = argmax_a' Q(s, a'; θ)
        
        # Execute action
        Execute a, observe reward r and next state s'
        
        # Store transition
        Store (s, a, r, s', done) in D
        
        # Training step (if enough data)
        If |D| >= min_replay_size:
            Sample mini-batch of B=64 transitions from D
            
            For each transition (sᵢ, aᵢ, rᵢ, s'ᵢ, dᵢ):
                # Compute target (Double DQN)
                a* = argmax_a' Q(s'ᵢ, a'; θ)
                yᵢ = rᵢ + (1 - dᵢ) * γ * Q(s'ᵢ, a*; θ⁻)
            
            # Gradient descent
            Loss = (1/B) * Σᵢ (Q(sᵢ, aᵢ; θ) - yᵢ)²
            θ ← θ - α * ∇_θ Loss
            
            # Gradient clipping
            Clip gradients to max_norm = 10.0
            
            # Target network update (every 500 steps)
            If training_step % 500 == 0:
                θ⁻ ← θ
        
        s ← s'
    
    # Decay exploration
    ε ← max(ε_min, ε * decay_rate)
```

---

### Network Architecture

#### Standard DQN Architecture

```
Input (33) → Linear(256) → ReLU → Linear(256) → ReLU → Output(5)
```

**Mathematical Formulation**:
$$h_1 = \text{ReLU}(W_1 s + b_1)$$
$$h_2 = \text{ReLU}(W_2 h_1 + b_2)$$
$$Q(s, \cdot; \theta) = W_3 h_2 + b_3$$

where $\theta = \{W_1, b_1, W_2, b_2, W_3, b_3\}$

#### Dueling DQN Architecture (Default)

**Key Insight**: Decompose Q-function into value and advantage:

$$Q(s, a) = V(s) + A(s, a)$$

where:
- $V(s)$: **State value function** - how good is this state regardless of action
- $A(s, a)$: **Advantage function** - how much better is action $a$ compared to average

**Network Structure**:
```
Input (33)
    ↓
Linear(256) → ReLU  [Shared Feature Layer]
    ↓
    ├─→ Value Stream:      Linear(256) → ReLU → Linear(1)    → V(s)
    └─→ Advantage Stream:  Linear(256) → ReLU → Linear(5)    → A(s,a)
    
Q(s,a) = V(s) + (A(s,a) - mean_a A(s,a))
```

**Mathematical Formulation**:

1. **Shared Features**:
   $$f(s) = \text{ReLU}(W_f s + b_f)$$

2. **Value Stream**:
   $$V(s) = W_v \text{ReLU}(W_{v1} f(s) + b_{v1}) + b_v$$

3. **Advantage Stream**:
   $$A(s, a) = W_a \text{ReLU}(W_{a1} f(s) + b_{a1}) + b_a$$

4. **Q-value Aggregation** (with mean subtraction for identifiability):
   $$Q(s, a; \theta) = V(s) + \left( A(s, a) - \frac{1}{|A|} \sum_{a'} A(s, a') \right)$$

**Why Mean Subtraction?**
Without it, the decomposition is not unique. Mean subtraction forces:
- $V(s)$ to represent true state value
- $A(s, a)$ to represent advantage relative to average action

**Benefits**:
- Better learning of state values (useful in states where action choice doesn't matter much)
- More stable training
- Faster convergence
- Better performance in practice

---

### Exploration Strategy: ε-Greedy

**Policy**:
$$\pi(a|s) = \begin{cases}
\arg\max_a Q(s, a; \theta) & \text{with probability } 1-\epsilon \\
\text{random action} & \text{with probability } \epsilon
\end{cases}$$

**Epsilon Decay Schedule**:
$$\epsilon_t = \max(\epsilon_{min}, \epsilon_{t-1} \cdot \lambda)$$

where:
- $\epsilon_0 = 1.0$ (start with full exploration)
- $\epsilon_{min} = 0.05$ (minimum exploration)
- $\lambda = 0.995$ (decay rate per episode)

**Decay Over Time**:
- Episode 1: $\epsilon \approx 1.0$ (100% random)
- Episode 100: $\epsilon \approx 0.606$
- Episode 300: $\epsilon \approx 0.223$
- Episode 500: $\epsilon \approx 0.082$
- Episode 1000: $\epsilon \approx 0.05$ (reaches minimum)

---

### Optimization Details

#### Optimizer: Adam

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) \nabla_\theta \mathcal{L}$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) (\nabla_\theta \mathcal{L})^2$$
$$\theta_t = \theta_{t-1} - \alpha \frac{m_t}{\sqrt{v_t} + \epsilon}$$

**Parameters**:
- Learning rate: $\alpha = 5 \times 10^{-4}$
- $\beta_1 = 0.9$, $\beta_2 = 0.999$
- $\epsilon = 10^{-8}$

#### Gradient Clipping

Prevent exploding gradients:
$$\nabla\theta \leftarrow \begin{cases}
\nabla\theta & \text{if } ||\nabla\theta|| \leq 10.0 \\
\frac{10.0 \cdot \nabla\theta}{||\nabla\theta||} & \text{otherwise}
\end{cases}$$

#### Batch Training

- **Batch size**: $B = 64$
- **Min replay size**: 1000 transitions before training starts
- **Update frequency**: Every step (if enough data in buffer)

---

### Training Procedure

#### Episode Structure

1. **Environment Reset**: Create new grid with random obstacles
2. **Dynamic Curriculum**: Vary grid size and complexity per episode
3. **Step Loop**: Until goal reached or max steps (200)
4. **Metrics Logging**: Track rewards, successes, losses
5. **Periodic Evaluation**: Test on fixed environments every 50 episodes
6. **Checkpointing**: Save full state every 100 episodes

#### Curriculum Learning (Optional)

Gradually increase difficulty over training:

$$\text{grid\_size}(t) = \text{base\_size} + \left\lfloor \frac{t}{T} \cdot 5 \right\rfloor$$
$$\text{obstacle\_prob}(t) = \text{base\_prob} + \frac{t}{T} \cdot 0.1$$
$$\text{n\_moving}(t) = \text{base\_moving} + \left\lfloor \frac{t}{T} \cdot 3 \right\rfloor$$

where $t$ is current episode and $T$ is total episodes.

**Progression Example** (1000 episodes):
- Episode 1: Grid 10×10, 30% obstacles, 3 moving
- Episode 500: Grid 12×12, 35% obstacles, 4-5 moving
- Episode 1000: Grid 15×15, 40% obstacles, 6 moving

---

### Key Hyperparameters Summary

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Episodes | 1000 | Total training episodes |
| Replay Buffer | 50,000 | Experience storage capacity |
| Batch Size | 64 | Mini-batch size for training |
| Learning Rate | 5e-4 | Adam optimizer step size |
| Gamma (γ) | 0.99 | Discount factor |
| Epsilon Start | 1.0 | Initial exploration rate |
| Epsilon End | 0.05 | Minimum exploration rate |
| Epsilon Decay | 0.995 | Decay rate per episode |
| Target Update | 500 | Steps between target network updates |
| Hidden Layers | [256, 256] | Network architecture |
| Gradient Clip | 10.0 | Max gradient norm |
| Min Replay | 1000 | Minimum transitions before training |

---

## 📁 Project Structure

### Training

**Basic training:**
```bash
python train.py
```

**Custom training with arguments:**
```bash
python train.py --episodes 2000 --grid-size 12 --obstacle-prob 0.4 --eval-freq 50
```

**Resume training from checkpoint:**
```bash
python train.py --resume models/checkpoint_ep500.pt
```

**All training arguments:**
- `--episodes`: Number of training episodes (default: 1000)
- `--grid-size`: Grid environment size (default: 10)
- `--obstacle-prob`: Probability of static obstacles (default: 0.3)
- `--render-every`: Render every N episodes (default: 0, no rendering)
- `--save-frames`: Save rendered frames to disk
- `--seed`: Random seed for reproducibility (default: 42)
- `--resume`: Path to checkpoint to resume from
- `--no-dueling`: Disable Dueling DQN architecture
- `--eval-freq`: Evaluation frequency in episodes (default: 50)

### Evaluation

**Evaluate trained model:**
```bash
python evaluate.py --model-path models/dqn_best.pt --episodes 20 --render
```

**Using the example script:**
```bash
python example.py --model-path models/dqn_best.pt --episodes 10 --render
```

**Evaluation arguments:**
- `--model-path`: Path to saved model (required)
- `--episodes`: Number of evaluation episodes (default: 10)
- `--grid-size`: Grid size for evaluation (default: 10)
- `--render`: Render episodes
- `--epsilon`: Exploration rate during evaluation (default: 0.0)
- `--seed`: Random seed

## ⚙️ Configuration

All hyperparameters can be modified in `config.py`:

### Environment Settings
```python
class EnvironmentConfig:
    GRID_SIZE = 10
    OBSTACLE_PROB = 0.3
    N_MOVING_OBSTACLES = 3
    MAX_STEPS = 200
    CURRICULUM_ENABLED = True
```

### DQN Hyperparameters
```python
class DQNConfig:
    HIDDEN_LAYERS = [256, 256]  # Network architecture
    USE_DUELING = True
    LEARNING_RATE = 5e-4
    GAMMA = 0.99
    BATCH_SIZE = 64
    REPLAY_BUFFER_SIZE = 50000
    EPSILON_START = 1.0
    EPSILON_END = 0.05
```

### Training Settings
```python
class TrainingConfig:
    NUM_EPISODES = 1000
    CHECKPOINT_FREQUENCY = 100
    EVAL_FREQUENCY = 50
    EVAL_EPISODES = 10
    SAVE_METRICS = True
```

## 📊 Outputs & Results

### Saved Models
- `models/dqn_best.pt` - Best model based on rolling average reward
- `models/dqn_final.pt` - Final model after training completion
- `models/checkpoint_epXXX.pt` - Periodic full checkpoints with complete training state

### Metrics & Analysis
- `results/training_metrics.csv` - Episode-by-episode metrics (reward, length, success, loss, etc.)
- `results/training_curves.png` - Auto-generated 4-panel training visualization
- `results/detailed_training_analysis.png` - Comprehensive 9-panel analysis (from visualize_results.py)

### Logs
- `logs/training.log` - Detailed training logs with timestamps

### Expected Performance
With default settings (1000 episodes, grid size 10):
- **First Success**: Episodes 50-100
- **50% Success Rate**: Episodes 300-500
- **Final Success Rate**: 70-90%
- **Average Reward**: 10-15 (final 100 episodes)
- **Training Time**: 30-60 minutes (CPU), 15-30 minutes (GPU)

---

## 📈 Performance Tracking

The training script tracks:
- Episode rewards and lengths
- Success rate (goal reached)
- Training loss
- Replay buffer size
- Exploration epsilon

Metrics are:
- Logged to console and file
- Saved to CSV for analysis
- Plotted in training curves
- Used for best model selection

## 🔄 Advanced Features

### Curriculum Learning
Gradually increases environment difficulty:
- Grid size: 10 → 15
- Obstacle density: 0.3 → 0.4
- Moving obstacles: 3 → 6

Enable/disable in `config.py`:
```python
config.env.CURRICULUM_ENABLED = True
```

### Checkpointing & Resume
Full training state is saved including:
- Policy and target network weights
- Optimizer state
- Replay buffer
- Training metrics
- Epsilon value

Resume training:
```bash
python train.py --resume models/checkpoint_ep500.pt
```

### Periodic Evaluation
Agent is evaluated on fixed test environments during training:
```python
config.training.EVAL_FREQUENCY = 50  # Evaluate every 50 episodes
config.training.EVAL_EPISODES = 10   # Run 10 test episodes
```

## 🐛 Troubleshooting

**Training is slow:**
- Reduce replay buffer size
- Decrease batch size
- Reduce network size in `config.py`

**Agent not learning:**
- Check that `MIN_REPLAY_SIZE` is reached
- Verify reward signals are meaningful
- Increase exploration (higher `EPSILON_START`, slower `EPSILON_DECAY`)
- Try standard DQN: `--no-dueling`

**Out of memory:**
- Reduce `REPLAY_BUFFER_SIZE`
- Use smaller `HIDDEN_LAYERS`
- Decrease `BATCH_SIZE`

**Low success rate:**
- Disable curriculum learning initially
- Reduce environment complexity (smaller grid, fewer obstacles)
- Train for more episodes
- Adjust reward shaping in `config.py`

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@misc{dqn_grid_navigation,
  author = {Your Name},
  title = {Deep Q-Network for Grid Navigation},
  year = {2025},
  url = {https://github.com/yourusername/RL_Project}
}
```

## 📄 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact [your email].
