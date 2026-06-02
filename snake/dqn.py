"""Deep Q-Network — the value-based, off-policy contrast to PPO.

Q(s,a) over the 3 relative actions, trained with a replay buffer, a target
network, ε-greedy exploration, and the Double-DQN target. Shares the env,
checkpoints, reward shaping, and eval harness with the policy-gradient methods.
"""
from __future__ import annotations

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from snake.ppo import clip_grad_norm


class QNetwork(nn.Module):
    """CNN → Q-value per action (3). Same backbone as the actor-critic, one head."""
    def __init__(self, H: int, W: int, n_actions: int = 3):
        super().__init__()
        self.H, self.W = H, W
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.trunk = nn.Linear(64 * H * W, 512)
        self.q = nn.Linear(512, n_actions)

    def __call__(self, x: mx.array) -> mx.array:
        x = nn.relu(self.conv1(x))
        x = nn.relu(self.conv2(x))
        x = nn.relu(self.conv3(x))
        x = x.reshape(x.shape[0], -1)
        x = nn.relu(self.trunk(x))
        return self.q(x)

    def q_values(self, obs_np: np.ndarray) -> np.ndarray:
        out = self(mx.array(obs_np))
        mx.eval(out)
        return np.array(out)


class ReplayBuffer:
    def __init__(self, capacity: int, H: int, W: int):
        self.cap = capacity
        self.obs = np.zeros((capacity, H, W, 3), dtype=np.float32)
        self.next_obs = np.zeros((capacity, H, W, 3), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.idx = 0
        self.full = False

    def add_batch(self, obs, actions, rewards, next_obs, dones):
        n = obs.shape[0]
        for i in range(n):
            j = self.idx
            self.obs[j] = obs[i]
            self.next_obs[j] = next_obs[i]
            self.actions[j] = actions[i]
            self.rewards[j] = rewards[i]
            self.dones[j] = dones[i]
            self.idx = (self.idx + 1) % self.cap
            if self.idx == 0:
                self.full = True

    def __len__(self):
        return self.cap if self.full else self.idx

    def sample(self, batch_size: int):
        n = len(self)
        idx = np.random.randint(0, n, size=batch_size)
        return (self.obs[idx], self.actions[idx], self.rewards[idx],
                self.next_obs[idx], self.dones[idx])


class DQNTrainer:
    def __init__(self, H: int, W: int, cfg: dict):
        self.cfg = cfg
        self.model = QNetwork(H, W)
        self.target = QNetwork(H, W)
        self.target.update(self.model.parameters())   # snapshot (functional updates keep it frozen)
        mx.eval(self.model.parameters(), self.target.parameters())
        self.optimizer = optim.Adam(learning_rate=cfg["lr"])
        self.gamma = cfg["gamma"]

    def act_epsilon(self, obs_np: np.ndarray, eps: float) -> np.ndarray:
        n = obs_np.shape[0]
        q = self.model.q_values(obs_np)
        greedy = q.argmax(axis=-1).astype(np.int32)
        rand = np.random.randint(0, q.shape[-1], size=n).astype(np.int32)
        explore = np.random.random(n) < eps
        return np.where(explore, rand, greedy)

    def update(self, batch) -> float:
        obs, actions, rewards, next_obs, dones = batch
        obs_mx = mx.array(obs)
        next_mx = mx.array(next_obs)
        act_mx = mx.array(actions)
        rew_mx = mx.array(rewards)
        done_mx = mx.array(dones)
        B = obs.shape[0]
        idx = mx.arange(B)

        # Double DQN target: action from online net, value from target net.
        q_next_online = self.model(next_mx)
        next_act = mx.argmax(q_next_online, axis=-1)
        q_next_target = self.target(next_mx)
        next_q = q_next_target[idx, next_act]
        y = rew_mx + self.gamma * (1.0 - done_mx) * next_q
        y = mx.stop_gradient(y)

        def loss_fn(model):
            q = model(obs_mx)
            qa = q[idx, act_mx]
            return mx.mean((qa - y) ** 2)

        loss, grads = nn.value_and_grad(self.model, loss_fn)(self.model)
        grads = clip_grad_norm(grads, self.cfg["max_grad_norm"])
        self.optimizer.update(self.model, grads)
        mx.eval(self.model.parameters(), self.optimizer.state)
        return float(loss)

    def sync_target(self):
        self.target.update(self.model.parameters())
        mx.eval(self.target.parameters())
