"""DQN training loop. `python -m snake.train_dqn --config configs/dqn.json`

Shares the env, reward shaping, checkpoint format, thermal guard, and metrics
schema with the PPO trainer so results are directly comparable.
"""
from __future__ import annotations

import argparse
import json
import shutil
import signal
import time
from datetime import datetime
from pathlib import Path

import mlx.core as mx
import numpy as np
from tqdm import tqdm

from snake.checkpoint import CheckpointManager
from snake.config import load_config
from snake.dqn import DQNTrainer, ReplayBuffer
from snake.env import VectorizedSnakeEnv
from snake.rewards import shaped_reward
from snake.thermal import ThermalGuard


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--run-dir", default=None)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config)
    H = W = cfg["grid_size"]
    N = cfg["num_envs"]
    total_steps = cfg["total_steps"]

    # DQN hyperparameters (with sensible defaults)
    cap = cfg.get("replay_capacity", 100_000)
    batch = cfg.get("batch_size", 512)
    updates_per_iter = cfg.get("updates_per_iter", 1)
    target_sync = cfg.get("target_sync_steps", 2000)
    learn_start = cfg.get("learning_starts", 5000)
    eps_start = cfg.get("eps_start", 1.0)
    eps_end = cfg.get("eps_end", 0.05)
    eps_decay = cfg.get("eps_decay_steps", 1_000_000)

    run_dir = Path(args.run_dir) if args.run_dir else Path("runs") / datetime.now().strftime("dqn_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    if not args.resume:
        shutil.copy(args.config, run_dir / "config.json")
    metrics_file = open(run_dir / "metrics.jsonl", "a")

    trainer = DQNTrainer(H, W, cfg)
    ckpt = CheckpointManager(run_dir)
    step = 0
    if args.resume:
        try:
            step = ckpt.load_weights_into(trainer.model, "latest")
            trainer.sync_target()
            print(f"resumed from step {step:,}")
        except Exception as e:
            print(f"resume failed: {e} — fresh start")

    env = VectorizedSnakeEnv(H, W, N)
    env.compute_shaping = cfg.get("shaping_coef", 0.0) > 0
    replay = ReplayBuffer(cap, H, W)
    obs = env.observation()

    thermal = ThermalGuard(enabled=cfg.get("thermal_guard", True),
                           check_every=cfg.get("thermal_check_every", 25),
                           cooldown_s=cfg.get("thermal_cooldown_s", 30),
                           pause_limit=cfg.get("thermal_pause_limit", 90))

    _stop = [False]
    signal.signal(signal.SIGINT, lambda s, f: _stop.__setitem__(0, True))

    ep_rewards, ep_lengths, ep_wins = [], [], []
    ep_ret = np.zeros(N, dtype=np.float32)
    ep_len = np.zeros(N, dtype=np.int32)
    next_ckpt = (step // cfg["checkpoint_every"] + 1) * cfg["checkpoint_every"]
    best_winrate = 0.0
    last_loss = 0.0
    mean_r = 0.0
    t_start = time.time()
    t_last = t_start

    print(f"DQN  grid={H}×{W}  envs={N}  total={total_steps:,}  run={run_dir}")
    pbar = tqdm(total=total_steps, initial=step, unit="step", unit_scale=True,
                dynamic_ncols=True, smoothing=0.1)

    it = 0
    while step < total_steps:
        eps = max(eps_end, eps_start - (eps_start - eps_end) * step / eps_decay)
        actions = trainer.act_epsilon(obs, eps)
        next_obs, rewards, dones = env.step(actions)
        shaped = shaped_reward(env, rewards, cfg)
        won = env.last_won

        ep_ret += rewards
        ep_len += 1
        for i in range(N):
            if dones[i]:
                ep_rewards.append(float(ep_ret[i]))
                ep_lengths.append(int(ep_len[i]))
                ep_wins.append(1 if won[i] else 0)
                ep_ret[i] = 0.0
                ep_len[i] = 0

        replay.add_batch(obs, actions, shaped, next_obs, dones)
        obs = next_obs
        step += N
        it += 1

        if len(replay) >= learn_start:
            for _ in range(updates_per_iter):
                last_loss = trainer.update(replay.sample(batch))
        if step % target_sync < N:
            trainer.sync_target()

        # Logging once per ~rollout-sized chunk
        if it % 16 == 0:
            mean_r = float(np.mean(ep_rewards)) if ep_rewards else 0.0
            mean_l = float(np.mean(ep_lengths)) if ep_lengths else 0.0
            wr = float(np.mean(ep_wins)) if ep_wins else 0.0
            best_winrate = max(best_winrate, wr)
            ep_rewards.clear(); ep_lengths.clear(); ep_wins.clear()
            now = time.time()
            sps = int(N * 16 / max(now - t_last, 1e-6)); t_last = now
            metrics_file.write(json.dumps({
                "step": step, "wall_time": now - t_start, "mean_reward": mean_r,
                "mean_ep_length": mean_l, "mean_entropy": 0.0, "policy_loss": last_loss,
                "value_loss": last_loss, "approx_kl": 0.0, "win_rate": wr, "eps": eps,
            }) + "\n")
            metrics_file.flush()
            pbar.update(N * 16)
            pbar.set_postfix_str(f"R {mean_r:5.2f}  len {mean_l:5.1f}  eps {eps:.2f}  "
                                 f"loss {last_loss:.3f}  win {100*wr:.0f}%  {sps:,} sps")

        if step >= next_ckpt or _stop[0]:
            ckpt.save(step, trainer.model, trainer.optimizer, {"mean_reward": mean_r})
            next_ckpt += cfg["checkpoint_every"]
            if _stop[0]:
                pbar.write(f"interrupted — checkpointed at {step:,}")
                break
        thermal.check(it, pbar.write)

    pbar.close(); metrics_file.close()
    print(f"done. best win-rate {100*best_winrate:.0f}%  run {run_dir}")


if __name__ == "__main__":
    main()
