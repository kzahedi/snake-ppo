## Why

The Mac Mini M4's GPU and Neural Engine are underutilised for personal learning experiments. Snake is an ideal testbed: simple rules, rich strategic depth, and a difficulty curve that scales cleanly from 8×8 to 32×32 grids. Training a PPO agent overnight — with beautiful moderngl rendering and per-checkpoint video exports — creates both a hardware benchmark and a visually meditative artifact showing the full arc of policy learning.

## What Changes

- New standalone Python project `snake/` implementing a complete RL training + rendering + video pipeline
- PPO trainer built entirely in MLX, running natively on M4 GPU via Metal
- Vectorized Snake environment (256 parallel envs in numpy)
- moderngl offscreen renderer for headless checkpoint recording
- moderngl windowed renderer for live policy watching
- Per-checkpoint episode video export + end-of-training timelapse assembly via ffmpeg
- Config-driven grid sizes: 8×8 (quick smoke test), 16×16 (medium), 32×32 (overnight)

## Capabilities

### New Capabilities

- `snake-env`: Vectorized Snake game environment — grid state, step logic, reward, reset, 256 parallel instances
- `ppo-trainer`: PPO with GAE, entropy bonus, clipped surrogate objective, CNN actor-critic, runs on MLX/M4 GPU
- `moderngl-renderer`: Offscreen and windowed renderer — snake body gradient, food glow shader, value heatmap overlay
- `checkpoint-manager`: Save/load model weights and training metadata at configurable step intervals
- `video-exporter`: Capture rendered frames per checkpoint, assemble timelapse via ffmpeg/imageio
- `training-pipeline`: Unified entry point orchestrating env stepping, PPO updates, checkpointing, and recording
- `watch-mode`: Live windowed replay of any saved checkpoint episode

### Modified Capabilities

## Impact

- New project directory: `/Volumes/Eregion/projects/snake/snake/`
- New dependencies: `mlx`, `moderngl`, `pygame` (window backend for moderngl), `imageio[ffmpeg]`, `numpy`
- Output artefacts written to `runs/TIMESTAMP/` — no shared infrastructure touched
- No existing code modified; entirely greenfield
