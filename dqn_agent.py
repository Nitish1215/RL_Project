# dqn_agent.py
import random, numpy as np, torch
import torch.nn as nn, torch.optim as optim
from collections import deque

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )
    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, obs_dim, n_actions, device=None, lr=5e-4, gamma=0.99):
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.gamma = gamma

        self.policy = MLP(obs_dim, n_actions).to(self.device)
        self.target = MLP(obs_dim, n_actions).to(self.device)
        self.target.load_state_dict(self.policy.state_dict())

        self.optim = optim.Adam(self.policy.parameters(), lr=lr)
        self.replay = deque(maxlen=20000)
        self.batch_size = 64
        self.min_replay = 200
        self.steps = 0
        self.target_update = 500
        self.loss_fn = nn.MSELoss()

    def act(self, obs, eps=0.1):
        if random.random() < eps:
            return random.randrange(self.n_actions)
        t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q = self.policy(t)
        return int(q.argmax().item())

    def store(self, s,a,r,s2,done):
        self.replay.append((s,a,r,s2,done))

    def sample_batch(self):
        batch = random.sample(self.replay, self.batch_size)
        s,a,r,s2,d = zip(*batch)
        s = torch.tensor(np.stack(s), dtype=torch.float32, device=self.device)
        a = torch.tensor(a, dtype=torch.int64, device=self.device).unsqueeze(1)
        r = torch.tensor(r, dtype=torch.float32, device=self.device).unsqueeze(1)
        s2 = torch.tensor(np.stack(s2), dtype=torch.float32, device=self.device)
        d = torch.tensor(d, dtype=torch.float32, device=self.device).unsqueeze(1)
        return s,a,r,s2,d

    def train_step(self):
        if len(self.replay) < max(self.min_replay, self.batch_size):
            return None
        s,a,r,s2,d = self.sample_batch()
        # current Q
        q_vals = self.policy(s).gather(1,a)
        # Double DQN target: use policy to pick argmax, target to evaluate
        with torch.no_grad():
            next_actions = self.policy(s2).argmax(dim=1, keepdim=True)
            q_next = self.target(s2).gather(1, next_actions)
            q_target = r + (1.0 - d) * (self.gamma * q_next)
        loss = self.loss_fn(q_vals, q_target)
        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 10.0)
        self.optim.step()
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target.load_state_dict(self.policy.state_dict())
        return loss.item()
