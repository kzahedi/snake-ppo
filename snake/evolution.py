"""Neuroevolution — gradient-free policy search. Evolve a population of policy
networks by mutation + selection; fitness = how full the board gets. No
backprop, no advantage, no reward gradient — pure black-box optimisation.

The policy reuses ActorCritic (only its actor head / argmax is used), so the
best individual saves as a normal checkpoint and works with the eval harness.
"""
from __future__ import annotations

import numpy as np
import mlx.core as mx
from mlx.utils import tree_map

from snake.env import VectorizedSnakeEnv
from snake.network import ActorCritic


def perturb(params, sigma: float):
    """Add Gaussian noise to every weight tensor."""
    return tree_map(lambda a: a + sigma * mx.random.normal(a.shape), params)


def evaluate_individual(model, params, H, W, episodes, win_bonus=5.0, seed=0,
                        survival_coef=0.005, proximity_coef=2.0):
    """Set the model's weights to `params`, play greedy episodes, return fitness
    plus stats. Fitness rewards, in priority order:
      - apples eaten  (mean_len − 3, the real objective),
      - getting CLOSE to the food   (proximity — pulls evolution toward food,
        not just survival; without this it learns to loop and never eats),
      - a tiny bit of survival       (bootstraps off random policies).
    """
    model.update(params)
    cells = H * W
    stall = 2 * cells
    maxd = (H - 1) + (W - 1)
    lengths, steps_list, prox_list, wins = [], [], [], 0
    for e in range(episodes):
        np.random.seed(seed * 1000 + e)
        env = VectorizedSnakeEnv(H, W, 1, auto_reset=False)
        obs = env.observation()
        ssf = 0
        t = 0
        min_d = maxd
        for t in range(cells * cells):
            hr, hc = env.bodies[0][0]
            fr, fc = int(env.food[0][0]), int(env.food[0][1])
            min_d = min(min_d, abs(hr - fr) + abs(hc - fc))
            probs, _ = model(mx.array(obs))
            mx.eval(probs)
            a = int(np.array(probs).argmax(axis=-1)[0])
            obs, r, d = env.step(np.array([a], dtype=np.int32))
            ssf = 0 if r[0] == 1.0 else ssf + 1
            if not env.alive[0]:
                wins += 1 if env.death_cause[0] == "win" else 0
                break
            if ssf >= stall:
                break
        lengths.append(len(env.bodies[0]))
        steps_list.append(t + 1)
        prox_list.append(1.0 - min_d / maxd)
    mean_len = float(np.mean(lengths))
    win_rate = wins / episodes
    fitness = (mean_len
               + proximity_coef * float(np.mean(prox_list))
               + survival_coef * float(np.mean(steps_list))
               + win_bonus * win_rate)
    return fitness, mean_len, win_rate
