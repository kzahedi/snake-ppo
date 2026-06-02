"""Shared evaluation harness — run ANY agent over N greedy episodes on the
same environment and report fill% / solve-rate / efficiency, so different
approaches can be compared on equal footing.

Usage:
  python -m snake.eval --grid 8 --episodes 50 \
      --ppo runs/solve --baselines hamiltonian,greedy-astar,flood-fill
"""
from __future__ import annotations

import argparse

import numpy as np

from snake.env import VectorizedSnakeEnv


def evaluate(agent, H, W, episodes=50, max_steps=None, seed=0):
    """Run `agent` (object with .act(env)->int) for `episodes` greedy games.

    Returns dict: solve_rate, mean_fill, max_fill, mean_len, mean_steps, deaths.
    """
    if max_steps is None:
        max_steps = H * W * H * W
    np.random.seed(seed)
    stall = 2 * H * W
    lengths, steps_list, reasons = [], [], []
    for _ in range(episodes):
        env = VectorizedSnakeEnv(H, W, 1, auto_reset=False)
        obs = env.observation()
        ssf = 0
        t = 0
        for t in range(max_steps):
            a = agent.act(env) if hasattr(agent, "act") else agent(env, obs)
            obs, r, d = env.step(np.array([a], dtype=np.int32))
            ssf = 0 if r[0] == 1.0 else ssf + 1
            if not env.alive[0]:
                reasons.append(env.death_cause[0] or "crash")
                break
            if ssf >= stall:
                reasons.append("stalled")
                break
        else:
            reasons.append("cap")
        lengths.append(len(env.bodies[0]))
        steps_list.append(t + 1)
    cells = H * W
    lengths = np.array(lengths)
    return {
        "episodes": episodes,
        "solve_rate": float(np.mean([1.0 if r == "win" else 0.0 for r in reasons])),
        "mean_fill": float(np.mean(lengths) / cells),
        "max_fill": float(np.max(lengths) / cells),
        "mean_len": float(np.mean(lengths)),
        "mean_steps": float(np.mean(steps_list)),
    }


class _NetAgent:
    """Greedy wrapper around a trained ActorCritic checkpoint."""
    def __init__(self, run_dir, H, W, checkpoint="latest"):
        import mlx.core as mx
        from snake.network import ActorCritic
        from snake.checkpoint import CheckpointManager
        self.mx = mx
        self.model = ActorCritic(H, W)
        self.step = CheckpointManager(run_dir).load_weights_into(self.model, checkpoint)
        self.name = f"ppo({run_dir.rstrip('/').split('/')[-1]})"

    def act(self, env):
        mx = self.mx
        probs, _ = self.model(mx.array(env.observation()))
        mx.eval(probs)
        return int(np.array(probs).argmax(axis=-1)[0])


class _QNetAgent:
    """Greedy wrapper around a trained DQN Q-network checkpoint."""
    def __init__(self, run_dir, H, W, checkpoint="latest"):
        from snake.dqn import QNetwork
        from snake.checkpoint import CheckpointManager
        self.model = QNetwork(H, W)
        self.step = CheckpointManager(run_dir).load_weights_into(self.model, checkpoint)
        self.name = f"dqn({run_dir.rstrip('/').split('/')[-1]})"

    def act(self, env):
        q = self.model.q_values(env.observation())
        return int(q.argmax(axis=-1)[0])


def _make_baseline(name, H, W):
    from snake.baselines import HamiltonianAgent, GreedyAStarAgent, FloodFillAgent
    table = {a.name: a for a in (HamiltonianAgent, GreedyAStarAgent, FloodFillAgent)}
    if name not in table:
        raise ValueError(f"unknown baseline '{name}'; choose from {list(table)}")
    return table[name](H, W)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--grid", type=int, default=8)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--ppo", action="append", default=[],
                   help="run dir of a trained PPO policy (repeatable)")
    p.add_argument("--dqn", action="append", default=[],
                   help="run dir of a trained DQN policy (repeatable)")
    p.add_argument("--baselines", default="hamiltonian,greedy-astar,flood-fill")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    H = W = args.grid

    agents = []
    for rd in args.ppo:
        agents.append(_NetAgent(rd, H, W))
    for rd in args.dqn:
        agents.append(_QNetAgent(rd, H, W))
    for name in [b for b in args.baselines.split(",") if b]:
        agents.append(_make_baseline(name, H, W))

    print(f"\nEvaluating {len(agents)} agents on {H}×{W}, {args.episodes} episodes each\n")
    header = f"{'agent':<22}{'solve%':>8}{'mean fill':>11}{'max fill':>10}{'mean len':>10}{'mean steps':>12}"
    print(header)
    print("-" * len(header))
    rows = []
    for ag in agents:
        m = evaluate(ag, H, W, episodes=args.episodes, seed=args.seed)
        rows.append((ag.name, m))
        print(f"{ag.name:<22}{100*m['solve_rate']:>7.0f}%{100*m['mean_fill']:>10.0f}%"
              f"{100*m['max_fill']:>9.0f}%{m['mean_len']:>10.1f}{m['mean_steps']:>12.0f}")
    print()
    return rows


if __name__ == "__main__":
    main()
