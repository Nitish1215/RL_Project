# complex_env.py
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque

class ComplexGridEnv:
    """
    Complex Grid Environment:
    - size: grid size (size x size)
    - static obstacles (obstacle_prob)
    - moving obstacles: N dynamic obstacles that move randomly each step
    - observation: [agent_pos_norm(2), goal_vec_norm(2), local_view(flattened (2r+1)^2), lidar(4)]
    - collisions: penalty but NOT terminal
    - episodes end on reaching goal or after max_steps
    """

    HEADINGS = [(-1,0),(0,1),(1,0),(0,-1)]  # up,right,down,left

    def __init__(self, size=12, obstacle_prob=0.18, n_moving=3, local_view=2, max_steps=200, connectivity_tries=50, seed=None):
        if seed is not None:
            random.seed(seed); np.random.seed(seed)
        self.size = size
        self.obstacle_prob = obstacle_prob
        self.n_moving = n_moving
        self.local_view = local_view
        self.max_steps = max_steps
        self.connectivity_tries = connectivity_tries
        self._fig = None
        self._ax = None
        # ANTI-OSCILLATION FIX: Track more positions to detect stuck behavior
        self.prev_positions = deque(maxlen=10)  # Increased from 5 to 10 for better detection
        self.position_counts = {}  # Count visits to each position
        self.reset()

    # ----------------------
    # connectivity check (BFS)
    # ----------------------
    def _is_connected(self, start, goal, grid):
        if start == goal:
            return True
        q = deque([start])
        seen = set([start])
        while q:
            x,y = q.popleft()
            for dx,dy in ComplexGridEnv.HEADINGS:
                nx,ny = x+dx, y+dy
                if 0 <= nx < self.size and 0 <= ny < self.size and grid[nx,ny] == 0:
                    if (nx,ny) == goal:
                        return True
                    if (nx,ny) not in seen:
                        seen.add((nx,ny)); q.append((nx,ny))
        return False

    def reset(self):
        # build static obstacles and ensure connectivity between start and goal
        for _ in range(self.connectivity_tries):
            grid = np.zeros((self.size, self.size), dtype=np.uint8)
            mask = np.random.rand(self.size, self.size) < self.obstacle_prob
            grid[mask] = 1
            # ensure outer border free
            grid[0,:]=0; grid[-1,:]=0; grid[:,0]=0; grid[:,-1]=0
            free = list(zip(*np.where(grid==0)))
            if len(free) < 2:
                continue
            agent_pos = random.choice(free)
            goal_pos = random.choice(free)
            if agent_pos == goal_pos:
                continue
            if self._is_connected(agent_pos, goal_pos, grid):
                self.grid = grid
                self.agent = agent_pos
                self.goal = goal_pos
                break
        else:
            # fallback: empty grid
            self.grid = np.zeros((self.size, self.size), dtype=np.uint8)
            free = list(zip(*np.where(self.grid==0)))
            self.agent = random.choice(free)
            self.goal = random.choice(free)
            while self.agent == self.goal:
                self.goal = random.choice(free)

        # place moving obstacles on free cells (not agent/goal)
        free_cells = [c for c in zip(*np.where(self.grid==0)) if c != self.agent and c != self.goal]
        random.shuffle(free_cells)
        self.moving = []
        for i in range(min(self.n_moving, len(free_cells))):
            self.moving.append(free_cells[i])

        self.steps = 0
        self.done = False
        self._prev_dist = self._manhattan(self.agent, self.goal)
        self.prev_positions.clear()  # Clear oscillation tracking
        self.position_counts.clear()  # Clear visit counts
        return self._get_obs()

    # ----------------------
    # utilities
    # ----------------------
    def _manhattan(self, a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])

    def _in_bounds(self, p):
        x,y = p
        return 0 <= x < self.size and 0 <= y < self.size

    def _is_free(self, p, consider_moving=True):
        x,y = p
        if not self._in_bounds(p): return False
        if self.grid[x,y] == 1: return False
        if consider_moving and p in self.moving: return False
        return True

    # local occupancy view around agent (flattened)
    def _local_view(self):
        r = self.local_view
        x,y = self.agent
        view = np.ones((2*r+1, 2*r+1), dtype=np.uint8)  # default obstacle/out-of-bounds
        for dx in range(-r, r+1):
            for dy in range(-r, r+1):
                nx,ny = x+dx, y+dy
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    # treat moving obstacles as obstacles too
                    view[dx+r, dy+r] = 1 if (self.grid[nx,ny]==1 or (nx,ny) in self.moving) else 0
        return view.flatten().astype(np.float32)

    # lidar: distance to obstacle in 4 directions (normalized)
    def _lidar(self):
        dists = []
        x,y = self.agent
        for dx,dy in ComplexGridEnv.HEADINGS:
            dist = 0
            nx,ny = x+dx, y+dy
            while 0 <= nx < self.size and 0 <= ny < self.size and self.grid[nx,ny]==0 and (nx,ny) not in self.moving:
                dist += 1
                nx += dx; ny += dy
            # normalize by size
            dists.append(dist / (self.size-1))
        return np.array(dists, dtype=np.float32)

    def _get_obs(self):
        x,y = self.agent
        # normalized pos
        pos = np.array([x/(self.size-1), y/(self.size-1)], dtype=np.float32)
        # goal vector normalized
        gx,gy = self.goal
        vec = np.array([(gx - x)/max(1,self.size-1), (gy - y)/max(1,self.size-1)], dtype=np.float32)
        local = self._local_view()
        lidar = self._lidar()
        obs = np.concatenate([pos, vec, lidar, local])
        return obs

    # ----------------------
    # step dynamics
    # ----------------------
    def step(self, action):
        """
        action: 0=up,1=right,2=down,3=left,4=stay
        Returns: obs, reward, done, info
        """
        if self.done:
            return self._get_obs(), 0.0, True, {}

        self.steps += 1
        # move moving obstacles first (random moves)
        self._move_dynamic()

        # apply agent action
        reward = -0.05  # small time penalty
        done = False
        if action == 4:
            # ANTI-OSCILLATION FIX: Extra penalty for staying in place
            reward -= 0.5  # Discourage staying action
        else:
            dx,dy = ComplexGridEnv.HEADINGS[action]
            new = (self.agent[0]+dx, self.agent[1]+dy)
            # collision if out-of-bounds, hitting static or moving obstacle
            if not self._in_bounds(new) or self.grid[new]==1 or new in self.moving:
                reward -= 1.0  # collision penalty but continue
            else:
                self.agent = new

        # ANTI-OSCILLATION FIX: Strong penalties for oscillating and getting stuck
        # Track position visits
        self.position_counts[self.agent] = self.position_counts.get(self.agent, 0) + 1
        
        # Escalating penalty for revisiting positions
        if self.agent in self.prev_positions:
            # Count how many times this position appears in recent history
            recent_visits = list(self.prev_positions).count(self.agent)
            reward -= 2.0 * (1 + recent_visits)  # Escalating penalty: -2, -4, -6, etc.
        
        # Additional penalty if stuck in small area (visiting same position too often)
        if self.position_counts[self.agent] > 3:
            reward -= 1.5 * (self.position_counts[self.agent] - 3)  # Extra penalty for repeated visits
        
        self.prev_positions.append(self.agent)

        # goal check
        if self.agent == self.goal:
            done = True
            reward += 20.0

        # progress bonus (encourage decreasing manhattan distance)
        dist_now = self._manhattan(self.agent, self.goal)
        if dist_now < self._prev_dist:
            reward += 0.5
        elif dist_now > self._prev_dist:
            reward -= 0.1
        self._prev_dist = dist_now

        if self.steps >= self.max_steps:
            done = True

        return self._get_obs(), float(reward), bool(done), {}

    def _move_dynamic(self):
        # each moving obstacle tries to move to a random adjacent free cell (not into agent or goal)
        new_positions = []
        occupied = set(self.moving)
        occupied.add(self.agent)
        occupied.add(self.goal)
        for pos in self.moving:
            choices = []
            for dx,dy in ComplexGridEnv.HEADINGS + [(0,0)]:
                nx,ny = pos[0]+dx, pos[1]+dy
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    if self.grid[nx,ny]==0 and (nx,ny) not in occupied:
                        choices.append((nx,ny))
            if choices:
                nxt = random.choice(choices)
                new_positions.append(nxt)
                occupied.add(nxt)
            else:
                new_positions.append(pos)
                occupied.add(pos)
        self.moving = new_positions

    # ----------------------
    # Rendering
    # ----------------------
    def render(self, show=True, save_path=None, scale=32):
        import matplotlib.pyplot as plt
        try:
            plt.ion()
        except Exception:
            pass

        grid_rgb = np.ones((self.size, self.size, 3), dtype=np.float32)
        # static obstacles dark
        grid_rgb[self.grid==1] = np.array([0.12,0.12,0.12])
        # goal green
        gx,gy = self.goal
        grid_rgb[gx,gy] = np.array([0.2,0.8,0.2])
        # moving obstacles red
        for (mx,my) in self.moving:
            grid_rgb[mx,my] = np.array([0.9,0.2,0.2])
        # agent blue
        ax,ay = self.agent
        grid_rgb[ax,ay] = np.array([0.2,0.4,0.9])

        axfig = self._ax; fig = self._fig
        if fig is None or axfig is None:
            fig, axfig = plt.subplots(figsize=(self.size*scale/100, self.size*scale/100))
            self._fig = fig; self._ax = axfig

        axfig.clear()
        axfig.imshow(grid_rgb, interpolation='nearest', origin='upper')
        axfig.set_xticks([]); axfig.set_yticks([])
        axfig.set_xlim(-0.5, self.size-0.5)
        axfig.set_ylim(self.size-0.5, -0.5)
        axfig.set_title(f"Step {self.steps}")
        axfig.set_aspect('equal')

        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=120)
        if show:
            try:
                fig.canvas.draw()
                fig.canvas.flush_events()
            except Exception:
                plt.pause(0.001)

    def close_renderer(self):
        import matplotlib.pyplot as plt
        if self._fig is not None:
            try:
                plt.close(self._fig)
            except Exception:
                pass
            self._fig = None
            self._ax = None

    @property
    def observation_space_dim(self):
        # pos(2) + goalvec(2) + lidar(4) + local_view ((2r+1)^2)
        return 2 + 2 + 4 + (2*self.local_view+1)**2

    @property
    def action_space_n(self):
        return 5
