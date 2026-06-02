import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map


def clip_grad_norm(grads, max_norm: float):
    leaves = [v for _, v in tree_flatten(grads) if isinstance(v, mx.array)]
    if not leaves:
        return grads
    total_sq = mx.sum(mx.stack([mx.sum(g ** 2) for g in leaves]))
    total_norm = mx.sqrt(total_sq)
    mx.eval(total_norm)
    coef = float(max_norm / (float(total_norm) + 1e-6))
    coef = min(coef, 1.0)
    return tree_map(lambda g: g * coef if isinstance(g, mx.array) else g, grads)


class RolloutBuffer:
    def __init__(self, T: int, N: int, H: int, W: int):
        self.T = T
        self.N = N
        self.obs = np.zeros((T, N, H, W, 3), dtype=np.float32)
        self.actions = np.zeros((T, N), dtype=np.int32)
        self.rewards = np.zeros((T, N), dtype=np.float32)
        self.dones = np.zeros((T, N), dtype=np.float32)
        self.log_probs = np.zeros((T, N), dtype=np.float32)
        self.values = np.zeros((T, N), dtype=np.float32)
        self.advantages = np.zeros((T, N), dtype=np.float32)
        self.returns = np.zeros((T, N), dtype=np.float32)
        self.t = 0

    def reset(self):
        self.t = 0

    def add(self, obs, actions, rewards, dones, log_probs, values):
        self.obs[self.t] = obs
        self.actions[self.t] = actions
        self.rewards[self.t] = rewards
        self.dones[self.t] = dones.astype(np.float32)
        self.log_probs[self.t] = log_probs
        self.values[self.t] = values
        self.t += 1

    def compute_gae(self, next_values: np.ndarray, gamma: float, gae_lambda: float):
        T = self.t
        advantages = np.zeros((T, self.N), dtype=np.float32)
        last_gae = np.zeros(self.N, dtype=np.float32)

        for t in reversed(range(T)):
            non_terminal = 1.0 - self.dones[t]
            next_val = next_values if t == T - 1 else self.values[t + 1]
            delta = self.rewards[t] + gamma * next_val * non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * non_terminal * last_gae
            advantages[t] = last_gae

        self.advantages = advantages
        self.returns = advantages + self.values[:T]

    def get_batches(self, batch_size: int):
        B = self.t * self.N
        idx = np.random.permutation(B)
        obs = self.obs[:self.t].reshape(B, *self.obs.shape[2:])
        actions = self.actions[:self.t].reshape(B)
        old_lp = self.log_probs[:self.t].reshape(B)
        adv = self.advantages[:self.t].reshape(B)
        ret = self.returns[:self.t].reshape(B)

        for start in range(0, B, batch_size):
            bi = idx[start:start + batch_size]
            yield {
                "obs": obs[bi],
                "actions": actions[bi],
                "old_log_probs": old_lp[bi],
                "advantages": adv[bi],
                "returns": ret[bi],
            }


class PPOTrainer:
    def __init__(self, model, cfg: dict):
        self.model = model
        self.cfg = cfg
        self.optimizer = optim.Adam(learning_rate=cfg["lr"])

    def update(self, buffer: RolloutBuffer, step: int) -> dict:
        cfg = self.cfg
        algo = cfg.get("algo", "ppo")   # "ppo" (clipped) or "a2c" (vanilla PG)
        all_policy_loss = []
        all_value_loss = []
        all_entropy = []
        all_kl = []

        for _ in range(cfg["ppo_epochs"]):
            for batch in buffer.get_batches(cfg["mini_batch_size"]):
                obs_mx = mx.array(batch["obs"])
                act_mx = mx.array(batch["actions"])
                old_lp_mx = mx.array(batch["old_log_probs"])
                adv_mx = mx.array(batch["advantages"])
                ret_mx = mx.array(batch["returns"])

                # Normalise advantages per mini-batch
                adv_norm = (adv_mx - adv_mx.mean()) / (adv_mx.std() + 1e-8)

                def loss_fn(model):
                    lp, vals, ent = model.evaluate(obs_mx, act_mx)
                    if algo == "a2c":
                        # Vanilla policy gradient (no clipping) — the A2C ablation.
                        policy_loss = mx.mean(-adv_norm * lp)
                    else:
                        ratio = mx.exp(lp - old_lp_mx)
                        loss1 = -adv_norm * ratio
                        loss2 = -adv_norm * mx.clip(ratio, 1 - cfg["clip_eps"], 1 + cfg["clip_eps"])
                        policy_loss = mx.mean(mx.maximum(loss1, loss2))
                    value_loss = mx.mean((vals - ret_mx) ** 2)
                    entropy_mean = mx.mean(ent)
                    total = (policy_loss
                             + cfg["value_coef"] * value_loss
                             - cfg["entropy_coef"] * entropy_mean)
                    return total

                loss_grad_fn = nn.value_and_grad(self.model, loss_fn)
                loss, grads = loss_grad_fn(self.model)
                grads = clip_grad_norm(grads, cfg["max_grad_norm"])
                self.optimizer.update(self.model, grads)
                mx.eval(self.model.parameters(), self.optimizer.state)

                # Metrics (second cheap forward, no grad)
                lp, vals, ent = self.model.evaluate(obs_mx, act_mx)
                mx.eval(lp, vals, ent)
                entropy_val = float(mx.mean(ent))
                kl_val = float(mx.mean(old_lp_mx - lp))
                all_policy_loss.append(float(loss))
                all_value_loss.append(float(mx.mean((vals - ret_mx) ** 2)))
                all_entropy.append(entropy_val)
                all_kl.append(kl_val)

        metrics = {
            "policy_loss": float(np.mean(all_policy_loss)),
            "value_loss": float(np.mean(all_value_loss)),
            "mean_entropy": float(np.mean(all_entropy)),
            "approx_kl": float(np.mean(all_kl)),
        }

        # Entropy floor warning
        floor = cfg.get("entropy_floor", 0.05)
        threshold = cfg.get("entropy_floor_step_threshold", 10_000_000)
        if metrics["mean_entropy"] < floor and step < threshold:
            metrics["entropy_warning"] = (
                f"entropy {metrics['mean_entropy']:.4f} below floor {floor} at step {step}"
            )

        return metrics
