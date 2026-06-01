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
from tqdm import tqdm

from snake.checkpoint import CheckpointManager
from snake.config import load_config
from snake.env import VectorizedSnakeEnv
from snake.network import ActorCritic
from snake.ppo import PPOTrainer, RolloutBuffer
from snake.recorder import VideoExporter


# ANSI colours (disabled when output is not a terminal)
class _C:
    on = sys.stdout.isatty()
    CYAN = "\033[96m" if on else ""
    ORANGE = "\033[38;5;208m" if on else ""
    PURPLE = "\033[95m" if on else ""
    GREY = "\033[90m" if on else ""
    GREEN = "\033[92m" if on else ""
    BOLD = "\033[1m" if on else ""
    DIM = "\033[2m" if on else ""
    R = "\033[0m" if on else ""


def _banner(cfg, run_dir, total_iters, device, shaping_coef, preview_every):
    H = W = cfg["grid_size"]
    N = cfg["num_envs"]
    T = cfg["steps_per_rollout"]
    c = _C
    rows = [
        ("grid", f"{H} × {W}"),
        ("parallel envs", f"{N}"),
        ("total steps", f"{cfg['total_steps']:,}  ({total_iters:,} iterations)"),
        ("rollout", f"{T} steps × {N} envs = {T*N:,}/update"),
        ("optim", f"lr {cfg['lr']:.1e}   γ {cfg['gamma']}   λ {cfg['gae_lambda']}"),
        ("ppo", f"clip {cfg['clip_eps']}   epochs {cfg['ppo_epochs']}   "
                f"ent {cfg['entropy_coef']}"),
        ("shaping", f"{shaping_coef}  (free-space connectivity)"
                    if shaping_coef > 0 else "off"),
        ("preview", f"every {preview_every:,} iters → preview.mp4"
                    if preview_every else "off"),
        ("device", str(device)),
        ("run dir", str(run_dir)),
    ]
    inner = 56                      # visible chars between the side borders
    line = "─" * inner

    def emit(visible: str, colored: str):
        """Print one bordered row, padding by the *visible* length."""
        pad = inner - len(visible)
        print(f"{c.CYAN}│{c.R}{colored}{' ' * max(pad, 0)}{c.CYAN}│{c.R}")

    print(f"{c.CYAN}╭{line}╮{c.R}")
    # title (emoji renders 2 cells wide but counts as 1 in len → pad one less)
    emit(" 🐍 SNAKE PPO" + " ", f" {c.BOLD}🐍 SNAKE PPO{c.R} ")
    print(f"{c.CYAN}├{line}┤{c.R}")
    for k, v in rows:
        visible = f" {k:<14} {v}"
        colored = f" {c.GREY}{k:<14}{c.R} {v}"
        emit(visible, colored)
    print(f"{c.CYAN}╰{line}╯{c.R}")


def _warmup(model, H, W, N):
    print(f"{_C.DIM}⚙  warming up MLX (JIT compile)…{_C.R}", end=" ", flush=True)
    dummy = np.zeros((N, H, W, 3), dtype=np.float32)
    for _ in range(3):
        actions, lp, vals = model.select_action(dummy)
    print(f"{_C.GREEN}done{_C.R}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-video", action="store_true",
                        help="Skip per-checkpoint video export + timelapse (faster, weights only)")
    parser.add_argument("--no-preview", action="store_true",
                        help="Disable the rolling preview.mp4 (rendered every N iterations)")
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
        # Generous hard cap so a full board solve fits; episodes actually end on
        # death or on the 2×cells no-food stall check inside _render_episode.
        max_steps=H * W * H * W,
    )

    step = 0
    if args.resume:
        try:
            step = ckpt.load_weights_into(model, "latest")
            print(f"{_C.GREEN}resumed from step {step:,}{_C.R}")
        except Exception as e:
            print(f"resume failed: {e} — starting fresh")

    # Warm up JIT
    _warmup(model, H, W, N)

    env = VectorizedSnakeEnv(H, W, N)
    shaping_coef = cfg.get("shaping_coef", 0.0)
    env.compute_shaping = shaping_coef > 0
    obs = env.observation()
    buf = RolloutBuffer(T, N, H, W)

    # Iteration accounting + rolling-preview cadence.
    # An "iteration" is one rollout + PPO update (T*N env steps). Round the
    # total to a whole number of iterations so the bar lands exactly on 100%.
    steps_per_iter = T * N
    total_iters = max(1, round(total_steps / steps_per_iter))
    total_steps = total_iters * steps_per_iter
    if args.no_preview:
        preview_every = 0
    else:
        # Every 1000 iters for long runs; otherwise ~10 previews across the run.
        preview_every = 1000 if total_iters >= 10_000 else max(1, total_iters // 10)

    _banner(cfg, run_dir, total_iters, mx.default_device(), shaping_coef, preview_every)

    # SIGINT handler — finish rollout then checkpoint
    _interrupted = [False]
    def _handle_sigint(sig, frame):
        _interrupted[0] = True
    signal.signal(signal.SIGINT, _handle_sigint)

    next_ckpt = (step // cfg["checkpoint_every"] + 1) * cfg["checkpoint_every"]
    t_start = time.time()
    t_last = t_start
    it = step // steps_per_iter   # iteration counter (survives --resume)
    ep_rewards: list[float] = []
    ep_lengths: list[int] = []
    ep_len_counters = np.zeros(N, dtype=np.int32)
    ep_return_counters = np.zeros(N, dtype=np.float32)
    best_reward = float("-inf")

    pbar = tqdm(total=total_steps, initial=step, unit="step", unit_scale=True,
                dynamic_ncols=True, smoothing=0.1,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                           "[{elapsed}<{remaining}, {rate_fmt}{postfix}]")

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
        it += 1

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

        # Live progress bar
        pbar.update(T * N)
        c = _C
        pbar.set_postfix_str(
            f"{c.GREY}it{c.R} {it:,}/{total_iters:,}  "
            f"{c.ORANGE}R{c.R} {mean_reward:6.2f}  "
            f"{c.CYAN}len{c.R} {mean_ep_len:5.1f}  "
            f"{c.PURPLE}H{c.R} {update_metrics['mean_entropy']:.3f}  "
            f"{c.GREY}{sps:,} sps{c.R}"
        )

        if mean_reward > best_reward:
            best_reward = mean_reward

        if "entropy_warning" in update_metrics:
            pbar.write(f"  {_C.GREY}[warn] {update_metrics['entropy_warning']}{_C.R}")

        # Rolling preview video (overwrites preview.mp4)
        if preview_every and it % preview_every == 0:
            try:
                info = exporter.export_preview(model, run_dir)
                fill = 100.0 * info["length"] / (H * W)
                pbar.write(
                    f"  {_C.CYAN}▸ preview{_C.R} {_C.GREY}iter {it:,}: "
                    f"ended by {info['reason']} at length {info['length']} "
                    f"({fill:.0f}% fill), {info['apples']} apples, "
                    f"{info['steps']} steps{_C.R}")
            except Exception as e:
                pbar.write(f"  {_C.GREY}[warn] preview failed: {e}{_C.R}")

        # Checkpoint
        if step >= next_ckpt or _interrupted[0]:
            all_metrics = {**record, **update_metrics}
            ckpt.save(step, model, trainer.optimizer, all_metrics)
            pbar.write(f"  {_C.GREEN}✓ checkpoint{_C.R} {_C.GREY}step {step:,}"
                       f"  (R={mean_reward:.2f}, best={best_reward:.2f}){_C.R}")

            if not args.no_video:
                try:
                    exporter.export_checkpoint(model, step, run_dir,
                                               keep_videos=cfg.get("keep_videos", True))
                    pbar.write(f"  {_C.GREEN}✓ checkpoint videos saved{_C.R}")
                except Exception as e:
                    pbar.write(f"  {_C.GREY}[warn] video export failed: {e}{_C.R}")

            next_ckpt += cfg["checkpoint_every"]

            if _interrupted[0]:
                pbar.write(f"\n{_C.ORANGE}⏹  interrupted — checkpointed at "
                           f"step {step:,}{_C.R}")
                break

    pbar.close()
    metrics_file.close()

    # Timelapse
    if not args.no_video:
        print(f"{_C.DIM}assembling timelapse…{_C.R}")
        tl = exporter.assemble_timelapse(run_dir, fps=24,
                                          keep_videos=cfg.get("keep_videos", True))
        if tl:
            print(f"{_C.GREEN}✓ timelapse:{_C.R} {tl}")

    c = _C
    print(f"\n{c.GREEN}{c.BOLD}✓ done{c.R}  "
          f"{c.GREY}best reward {best_reward:.2f}  ·  {it:,} iterations{c.R}")
    print(f"  run dir:  {c.CYAN}{run_dir}{c.R}")
    print(f"  watch:    {c.GREY}python -m snake.watch --run {run_dir} --loop --policy{c.R}")


if __name__ == "__main__":
    main()
