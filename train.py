# train.py  (diagnostic + training)
import os, time, random, argparse
import numpy as np
import torch

from grid_env import ComplexGridEnv as GridEnv        # your fixed grid_env.py
from dqn_agent import DQNAgent      # your DQN agent (with train_step returning loss)

def train(episodes=600, grid_size=10, obstacle_prob=0.3,
          render_every=0, save_frames=False, seed=0):
    # reproducibility
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    # create env and agent
    env = GridEnv(size=grid_size, obstacle_prob=obstacle_prob, max_steps=80, seed=seed)
    obs_dim = env.observation_space_dim
    n_actions = env.action_space_n

    # create agent with smaller warmup thresholds to ensure training starts quickly
    agent = DQNAgent(obs_dim, n_actions, lr=5e-4, device=None)
    # tune agent internals if present (safe-guard)
    try:
        # lower min_replay and batch size to start training quickly
        agent.min_replay = min(100, getattr(agent, "min_replay", 100))
        agent.batch_size = min(32, getattr(agent, "batch_size", 32))
        agent.target_update = getattr(agent, "target_update", 500)
    except Exception:
        pass

    print(f"[INFO] Device: {agent.device}")
    print(f"[INFO] obs_dim={obs_dim}, n_actions={n_actions}")
    print(f"[INFO] agent.batch_size={agent.batch_size}, agent.min_replay={agent.min_replay}, target_update={agent.target_update}")

    eps = 1.0; eps_min = 0.05; eps_decay = 0.995
    os.makedirs("models", exist_ok=True)
    os.makedirs("frames", exist_ok=True)

    replay_sizes = []
    losses = []
    best_avg = -1e9
    train_calls = 0

    try:
        for ep in range(1, episodes+1):
            # Increase grid size and complexity for training
            dynamic_grid_size = random.randint(grid_size, grid_size + 6)  # Larger grid
            dynamic_obstacle_prob = random.uniform(obstacle_prob, obstacle_prob + 0.15)  # Higher obstacle probability
            dynamic_n_moving = random.randint(5, 8)  # More moving obstacles
            env = GridEnv(size=dynamic_grid_size, obstacle_prob=dynamic_obstacle_prob, n_moving=dynamic_n_moving, max_steps=300, seed=seed)

            s = env.reset()
            done = False
            ep_reward = 0.0
            step = 0

            # frame saving directory if requested
            save_dir = None
            do_render = (render_every > 0) and (ep % render_every == 0)
            if do_render and save_frames:
                save_dir = os.path.join("frames", f"ep{ep:04d}")
                os.makedirs(save_dir, exist_ok=True)

            while not done:
                a = agent.act(s, eps)
                s2, r, done, _ = env.step(a)

                # store & train every step
                agent.store(s, a, r, s2, done)

                # call train_step and capture loss
                loss = agent.train_step()
                if loss is not None:
                    train_calls += 1
                    losses.append(loss)

                s = s2
                ep_reward += r
                step += 1

                # render/save if needed (non-blocking)
                if do_render:
                    env.render(show=True)
                    if save_frames and save_dir:
                        fname = os.path.join(save_dir, f"step{step:04d}.png")
                        env.render(show=False, save_path=fname)
                    # small pause so rendering doesn't starve cpu (adjustable)
                    time.sleep(0.005)

            replay_size = len(getattr(agent, "replay", []))
            replay_sizes.append(replay_size)
            eps = max(eps_min, eps * eps_decay)

            avg50 = np.mean(replay_sizes[-50:]) if len(replay_sizes) >= 1 else replay_size
            avg_loss = np.mean(losses[-100:]) if len(losses) > 0 else float("nan")
            recent_reward_avg = None
            # compute rolling reward avg (last 50)
            try:
                # we don't store all rewards list globally; print ep reward and avg of last 10 via sliding window (simple)
                pass
            except Exception:
                pass

            # Print helpful diagnostic every episode and a richer summary every 10 episodes
            if ep % 1 == 0:
                print(f"Ep {ep:03d} | steps {step:02d} | ep_reward {ep_reward:6.2f} | replay {replay_size:04d} | train_calls {train_calls} | last_loss {losses[-1]:.4f}" if losses else f"Ep {ep:03d} | steps {step:02d} | ep_reward {ep_reward:6.2f} | replay {replay_size:04d} | train_calls {train_calls} | last_loss None")

            if ep % 10 == 0:
                print(f"--- summary @ep {ep:03d} avg_lastloss {avg_loss:.4f} eps {eps:.3f} ---")

            # save best by training loss or reward heuristics if you prefer
            # here we save periodic checkpoints
            if ep % 200 == 0:
                torch.save(agent.policy.state_dict(), f"models/dqn_ep{ep}.pt")
                print(f"[INFO] saved checkpoint models/dqn_ep{ep}.pt")

            # Save the best model based on average reward
            if ep % 10 == 0:
                recent_rewards = replay_sizes[-10:]  # Use the last 10 episodes' rewards
                avg_reward = np.mean(recent_rewards) if recent_rewards else -float('inf')
                if avg_reward > best_avg:
                    best_avg = avg_reward
                    torch.save(agent.policy.state_dict(), "models/dqn_best.pt")
                    print(f"[INFO] New best model saved with avg reward {best_avg:.2f} at models/dqn_best.pt")

            # close renderer for this episode if opened
            if do_render:
                env.close_renderer()

    finally:
        try:
            env.close_renderer()
        except Exception:
            pass

    # final save
    torch.save(agent.policy.state_dict(), "models/dqn_final.pt")
    print("[INFO] Training finished. Total train calls:", train_calls)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=600)
    parser.add_argument("--grid-size", type=int, default=10)
    parser.add_argument("--obstacle-prob", type=float, default=0.5)
    parser.add_argument("--render-every", type=int, default=100)
    parser.add_argument("--save-frames", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    train(episodes=args.episodes,
          grid_size=args.grid_size,
          obstacle_prob=args.obstacle_prob,
          render_every=args.render_every,
          save_frames=args.save_frames,
          seed=args.seed)
