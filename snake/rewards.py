"""Shared reward shaping, so every approach (PPO, A2C, DQN, ...) optimises the
exact same objective. The env returns the raw +1 eat / -1 die signal; this adds
the config-gated shaping terms used for learning."""
from __future__ import annotations

import numpy as np


def shaped_reward(env, rewards: np.ndarray, cfg: dict) -> np.ndarray:
    shaped = rewards + cfg.get("shaping_coef", 0.0) * env.last_shaping
    lrc = cfg.get("length_reward_coef", 0.0)
    if lrc:
        ate = (rewards == 1.0).astype(np.float32)
        shaped = shaped + lrc * ate * (env.lengths() / float(env.H * env.W))
    sp = cfg.get("step_penalty", 0.0)
    if sp:
        shaped = shaped - sp
    wb = cfg.get("win_bonus", 0.0)
    if wb:
        shaped = shaped + wb * env.last_won.astype(np.float32)
    return shaped.astype(np.float32)
