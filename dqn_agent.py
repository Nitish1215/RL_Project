# dqn_agent.py
"""
Stable DQN Agent Implementation with Multiple Stability Improvements

STABILITY FEATURES IMPLEMENTED:
================================

1. DOUBLE DQN (train_step method):
   - Problem: Vanilla DQN overestimates Q-values due to max operator bias
   - Solution: Use policy network to SELECT actions, target network to EVALUATE
   - Impact: Prevents Q-value explosion and more stable learning

2. TARGET NETWORK with PERIODIC UPDATES:
   - Problem: Training on moving targets causes instability
   - Solution: Separate target network updated every N steps
   - Impact: Reduces correlation between Q-values and targets

3. GRADIENT CLIPPING:
   - Problem: Large gradients can cause training instability
   - Solution: Clip gradient norms to max value (default: 10.0)
   - Impact: Prevents exploding gradients and weight updates

4. EXPERIENCE REPLAY:
   - Problem: Sequential data is highly correlated
   - Solution: Uniform random sampling from replay buffer
   - Impact: Breaks temporal correlation, more stable updates

5. DUELING ARCHITECTURE (optional):
   - Problem: Some states don't require action differentiation
   - Solution: Separate value and advantage streams
   - Impact: Better value estimation, faster learning

6. Q-VALUE MONITORING:
   - Problem: Q-value divergence can go undetected
   - Solution: Track mean, max, min, std of Q-values
   - Impact: Early detection of training instability

ADDITIONAL STABILITY RECOMMENDATIONS:
- Use reward clipping in training loop (see config.py)
- Use lower learning rate (1e-4 recommended)
- Use larger replay buffer (100k+ for complex environments)
- Use slower epsilon decay (0.9985 vs 0.995)
- Keep epsilon_min >= 0.1 to maintain exploration
"""
import random, numpy as np, torch
import torch.nn as nn, torch.optim as optim
from collections import deque
from typing import List, Optional, Tuple


class MLP(nn.Module):
    """Standard MLP network"""
    def __init__(self, in_dim: int, out_dim: int, hidden_layers: List[int] = [128, 128]):
        super().__init__()
        layers = []
        prev_dim = in_dim
        
        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, out_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


class DuelingMLP(nn.Module):
    """Dueling DQN network architecture"""
    def __init__(self, in_dim: int, out_dim: int, hidden_layers: List[int] = [128, 128]):
        super().__init__()
        
        # Shared feature layers
        feature_layers = []
        prev_dim = in_dim
        for hidden_dim in hidden_layers[:-1]:  # All but last layer
            feature_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim
        self.features = nn.Sequential(*feature_layers)
        
        # Value stream
        final_hidden = hidden_layers[-1] if hidden_layers else 128
        self.value_stream = nn.Sequential(
            nn.Linear(prev_dim, final_hidden),
            nn.ReLU(),
            nn.Linear(final_hidden, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(prev_dim, final_hidden),
            nn.ReLU(),
            nn.Linear(final_hidden, out_dim)
        )
    
    def forward(self, x):
        features = self.features(x)
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values


class DQNAgent:
    def __init__(self, obs_dim: int, n_actions: int, device: Optional[str] = None, 
                 lr: float = 5e-4, gamma: float = 0.99, 
                 hidden_layers: List[int] = [256, 256], use_dueling: bool = True,
                 replay_size: int = 50000, batch_size: int = 64, 
                 min_replay: int = 1000, target_update: int = 500,
                 gradient_clip: float = 10.0):
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.min_replay = min_replay
        self.target_update = target_update
        self.gradient_clip = gradient_clip
        self.steps = 0

        # Choose network architecture
        NetworkClass = DuelingMLP if use_dueling else MLP
        self.policy = NetworkClass(obs_dim, n_actions, hidden_layers).to(self.device)
        self.target = NetworkClass(obs_dim, n_actions, hidden_layers).to(self.device)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()  # Target network always in eval mode

        self.optim = optim.Adam(self.policy.parameters(), lr=lr)
        self.replay = deque(maxlen=replay_size)
        self.loss_fn = nn.MSELoss()
        
        # STABILITY FIX: Track Q-values to detect divergence
        self.q_value_stats = {'mean': [], 'max': [], 'min': [], 'std': []}

    def act(self, obs: np.ndarray, eps: float = 0.1) -> int:
        if random.random() < eps:
            return random.randrange(self.n_actions)
        t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q = self.policy(t)
        return int(q.argmax().item())

    def store(self, s: np.ndarray, a: int, r: float, s2: np.ndarray, done: bool):
        self.replay.append((s, a, r, s2, done))

    def sample_batch(self) -> Tuple[torch.Tensor, ...]:
        batch = random.sample(self.replay, self.batch_size)
        s, a, r, s2, d = zip(*batch)
        s = torch.tensor(np.stack(s), dtype=torch.float32, device=self.device)
        a = torch.tensor(a, dtype=torch.int64, device=self.device).unsqueeze(1)
        r = torch.tensor(r, dtype=torch.float32, device=self.device).unsqueeze(1)
        s2 = torch.tensor(np.stack(s2), dtype=torch.float32, device=self.device)
        d = torch.tensor(d, dtype=torch.float32, device=self.device).unsqueeze(1)
        return s, a, r, s2, d

    def train_step(self) -> Optional[float]:
        """Train on a batch with Double DQN and return loss.
        
        STABILITY FEATURES:
        - Double DQN prevents Q-value overestimation
        - Gradient clipping prevents exploding gradients
        - Q-value statistics track potential divergence
        - Periodic target network updates reduce moving target problem
        """
        if len(self.replay) < max(self.min_replay, self.batch_size):
            return None
        
        s, a, r, s2, d = self.sample_batch()
        
        # Current Q values
        q_vals = self.policy(s).gather(1, a)
        
        # STABILITY FIX: Double DQN prevents overestimation bias
        # Use policy network to SELECT action, target network to EVALUATE it
        with torch.no_grad():
            next_actions = self.policy(s2).argmax(dim=1, keepdim=True)
            q_next = self.target(s2).gather(1, next_actions)
            q_target = r + (1.0 - d) * (self.gamma * q_next)
        
        loss = self.loss_fn(q_vals, q_target)
        
        self.optim.zero_grad()
        loss.backward()
        # STABILITY FIX: Gradient clipping prevents exploding gradients
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.gradient_clip)
        self.optim.step()
        
        # STABILITY FIX: Track Q-value statistics to detect divergence
        with torch.no_grad():
            q_mean = q_vals.mean().item()
            q_max = q_vals.max().item()
            q_min = q_vals.min().item()
            q_std = q_vals.std().item()
            self.q_value_stats['mean'].append(q_mean)
            self.q_value_stats['max'].append(q_max)
            self.q_value_stats['min'].append(q_min)
            self.q_value_stats['std'].append(q_std)
        
        self.steps += 1
        # STABILITY FIX: Periodic target update reduces moving target problem
        if self.steps % self.target_update == 0:
            self.target.load_state_dict(self.policy.state_dict())
        
        return loss.item()
    
    def get_q_stats(self, window=100):
        """Get recent Q-value statistics for monitoring."""
        if not self.q_value_stats['mean']:
            return None
        recent = min(window, len(self.q_value_stats['mean']))
        return {
            'q_mean': np.mean(self.q_value_stats['mean'][-recent:]),
            'q_max': np.mean(self.q_value_stats['max'][-recent:]),
            'q_min': np.mean(self.q_value_stats['min'][-recent:]),
            'q_std': np.mean(self.q_value_stats['std'][-recent:])
        }

