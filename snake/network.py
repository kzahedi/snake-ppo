import numpy as np
import mlx.core as mx
import mlx.nn as nn


class ActorCritic(nn.Module):
    def __init__(self, H: int, W: int, n_actions: int = 3):
        super().__init__()
        self.H = H
        self.W = W
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.trunk = nn.Linear(64 * H * W, 512)
        self.actor = nn.Linear(512, n_actions)
        self.value = nn.Linear(512, 1)

    def __call__(self, x: mx.array):
        # x: (B, H, W, 3) — MLX Conv2d expects NHWC
        x = nn.relu(self.conv1(x))
        x = nn.relu(self.conv2(x))
        x = nn.relu(self.conv3(x))
        x = x.reshape(x.shape[0], -1)
        x = nn.relu(self.trunk(x))
        probs = mx.softmax(self.actor(x), axis=-1)
        val = self.value(x).squeeze(-1)
        return probs, val

    def activations(self, obs_np: np.ndarray) -> dict:
        """Introspection: run forward pass capturing intermediate activations.

        obs_np: (1, H, W, 3). Returns numpy activations for visualisation.
        """
        x = mx.array(obs_np)
        a1 = nn.relu(self.conv1(x))
        a2 = nn.relu(self.conv2(a1))
        a3 = nn.relu(self.conv3(a2))
        flat = a3.reshape(a3.shape[0], -1)
        trunk = nn.relu(self.trunk(flat))
        probs = mx.softmax(self.actor(trunk), axis=-1)
        val = self.value(trunk).squeeze(-1)
        mx.eval(a1, a2, a3, trunk, probs, val)
        return {
            "conv1": np.array(a1)[0],    # (H, W, 32)
            "conv2": np.array(a2)[0],    # (H, W, 64)
            "conv3": np.array(a3)[0],    # (H, W, 64)
            "trunk": np.array(trunk)[0],  # (512,)
            "probs": np.array(probs)[0],  # (3,)
            "value": float(np.array(val)[0]),
        }

    def select_action(self, obs_np: np.ndarray):
        """Rollout-time action selection. Returns numpy arrays."""
        x = mx.array(obs_np)
        probs, values = self(x)
        mx.eval(probs, values)
        probs_np = np.array(probs)
        values_np = np.array(values)

        B = probs_np.shape[0]
        # Vectorised categorical sample
        cumprobs = np.cumsum(probs_np, axis=-1)
        rand = np.random.random((B, 1))
        actions = (rand < cumprobs).argmax(axis=-1).astype(np.int32)
        log_probs = np.log(probs_np[np.arange(B), actions] + 1e-8).astype(np.float32)
        return actions, log_probs, values_np

    def evaluate(self, obs: mx.array, actions: mx.array):
        """PPO update — all MLX arrays."""
        probs, values = self(obs)
        B = probs.shape[0]
        idx = mx.arange(B)
        log_probs = mx.log(probs[idx, actions] + 1e-8)
        entropy = -mx.sum(probs * mx.log(probs + 1e-8), axis=-1)
        return log_probs, values, entropy

    def value_grid(self, state: dict) -> np.ndarray:
        """Evaluate V(s) for each cell — synthetic states for heatmap."""
        H, W = self.H, self.W
        body = state["body"]
        food = state["food"]
        # Build base obs (body+food channels, no head channel)
        base = np.zeros((H, W, 3), dtype=np.float32)
        for r, c in body:
            base[r, c, 0] = 1.0
        base[food[0], food[1], 1] = 1.0

        # Create H*W synthetic observations with head placed at each cell
        batch = np.tile(base[None], (H * W, 1, 1, 1))
        for i, (r, c) in enumerate(np.ndindex(H, W)):
            batch[i, r, c, 2] = 1.0

        x = mx.array(batch)
        _, vals = self(x)
        mx.eval(vals)
        return np.array(vals).reshape(H, W)
