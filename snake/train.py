from __future__ import annotations

import argparse
import json
import shutil
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np

from snake.checkpoint import CheckpointManager
from snake.config import load_config
from snake.env import VectorizedSnakeEnv
from snake.network import ActorCritic
from snake.ppo import PPOTrainer, RolloutBuffer
from snake.recorder import VideoExporter


def _warmup(model, H, W, N):
    dummy = np.zeros((N, H, W, 3), dtype=np.float32)
    for _ in range(3):
        actions, lp, vals = model.select_action(dummy)
    print("MLX warmup done (JIT compiled)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-video", action="store_true",
                        help="Skip per-checkpoint video export + timelapse (faster, weights only)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    H = W = cfg["grid_size"]
    N = cfg["num_envs"]
    T = cfg["steps_per_rollout"]
    total_steps = cfg["total_steps"]

    # Set up run directory
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path("runs") / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "videos").mkdir(exist_ok=True)

    config_path = run_dir / "config.json"
    if not args.resume:
        shutil.copy(args.config, config_path)

    metrics_path = run_dir / "metrics.jsonl"
    metrics_file = open(metrics_path, "a")

    # Model + trainer
    model = ActorCritic(H, W)
    trainer = PPOTrainer(model, cfg)
    ckpt = CheckpointManager(run_dir)
    exporter = VideoExporter(
        grid_size=H,
        resolution=cfg.get("render_resolution", 800),
        fps=30,
        max_steps=H * W * 4,
    )

    step = 0
    if args.resume:
        try:
            step = ckpt.load_weights_into(model, "latest")
            print(f"Resumed from step {step:,}")
        except Exception as e:
            print(f"Resume failed: {e} — starting fresh")

    # Warm up JIT
    _warmup(model, H, W, N)

    env = VectorizedSnakeEnv(H, W, N)
    shaping_coef = cfg.get("shaping_coef", 0.0)
    env.compute_shaping = shaping_coef > 0
    obs = env.observation()
    buf = RolloutBuffer(T, N, H, W)
    if env.compute_shaping:
        print(f"Free-space connectivity shaping ON (coef={shaping_coef})")

    # SIGINT handler — finish rollout then checkpoint
    _interrupted = [False]
    def _handle_sigint(sig, frame):
        print("\nInterrupted — will checkpoint after this rollout.")
        _interrupted[0] = True
    signal.signal(signal.SIGINT, _handle_sigint)

    next_ckpt = (step // cfg["checkpoint_every"] + 1) * cfg["checkpoint_every"]
    t_start = time.time()
    t_last = t_start
    ep_rewards: list[float] = []
    ep_lengths: list[int] = []
    ep_len_counters = np.zeros(N, dtype=np.int32)
    ep_return_counters = np.zeros(N, dtype=np.float32)

    print(f"Training  grid={H}×{W}  envs={N}  total={total_steps:,}  run={run_dir}")

    while step < total_steps:
        buf.reset()

        for t in range(T):
            actions, log_probs, values = model.select_action(obs)
            next_obs, rewards, dones = env.step(actions)

            # Raw reward drives the interpretable metric (apples eaten);
            # shaped reward (raw + connectivity bonus) drives learning.
            shaped = rewards + shaping_coef * env.last_shaping

            ep_len_counters += 1
            ep_return_counters += rewards
            for i in range(N):
                if dones[i]:
                    # Full episode return (sum of raw rewards = food_eaten - 1)
                    ep_rewards.append(float(ep_return_counters[i]))
                    ep_lengths.append(int(ep_len_counters[i]))
                    ep_len_counters[i] = 0
                    ep_return_counters[i] = 0.0

            buf.add(obs, actions, shaped, dones, log_probs, values)
            obs = next_obs

        step += T * N

        # Bootstrap value for last obs
        _, _, last_vals = model.select_action(obs)
        buf.compute_gae(last_vals, cfg["gamma"], cfg["gae_lambda"])

        update_metrics = trainer.update(buf, step)

        # Aggregate episode stats (episode return = food_eaten - 1)
        mean_reward = float(np.mean(ep_rewards)) if ep_rewards else 0.0
        mean_ep_len = float(np.mean(ep_lengths)) if ep_lengths else 0.0
        ep_rewards.clear()
        ep_lengths.clear()

        now = time.time()
        sps = int(T * N / max(now - t_last, 1e-6))
        t_last = now

        # Metrics record
        record = {
            "step": step,
            "wall_time": now - t_start,
            "mean_reward": mean_reward,
            "mean_ep_length": mean_ep_len,
            "mean_entropy": update_metrics["mean_entropy"],
            "policy_loss": update_metrics["policy_loss"],
            "value_loss": update_metrics["value_loss"],
            "approx_kl": update_metrics["approx_kl"],
        }
        metrics_file.write(json.dumps(record) + "\n")
        metrics_file.flush()

        print(
            f"step={step:>10,}  sps={sps:>6,}  "
            f"reward={mean_reward:>6.3f}  ep_len={mean_ep_len:>6.1f}  "
            f"entropy={update_metrics['mean_entropy']:>5.3f}",
            flush=True,
        )

        if "entropy_warning" in update_metrics:
            print(f"  [WARN] {update_metrics['entropy_warning']}")

        # Checkpoint
        if step >= next_ckpt or _interrupted[0]:
            all_metrics = {**record, **update_metrics}
            ckpt.save(step, model, trainer.optimizer, all_metrics)
            print(f"  → checkpoint saved at step {step:,}")

            if not args.no_video:
                try:
                    exporter.export_checkpoint(model, step, run_dir,
                                               keep_videos=cfg.get("keep_videos", True))
                    print(f"  → episode video saved")
                except Exception as e:
                    print(f"  [WARN] video export failed: {e}")

            next_ckpt += cfg["checkpoint_every"]

            if _interrupted[0]:
                break

    metrics_file.close()

    # Timelapse
    if not args.no_video:
        print("Assembling timelapse...")
        tl = exporter.assemble_timelapse(run_dir, fps=24,
                                          keep_videos=cfg.get("keep_videos", True))
        if tl:
            print(f"Timelapse: {tl}")

    print(f"Done. Run dir: {run_dir}")


if __name__ == "__main__":
    main()
