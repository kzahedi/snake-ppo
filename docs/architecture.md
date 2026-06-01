# Architecture

## Design principles

1. **Apple-native compute.** All neural-network math runs on the M-series GPU
   through MLX (Metal). No CUDA, no PyTorch. The environment is plain NumPy on
   the CPU — grid logic doesn't need the GPU.
2. **Simple over clever.** This is an experimentation testbed, not a production
   RL library. The PPO loop is written out explicitly so it can be read and
   tweaked, rather than hidden behind a framework.
3. **Visualisation is first-class.** Rendering, video export, and live policy
   introspection are part of the project, not afterthoughts.

## Module map

```
snake/
├── config.py        Config loading + validation (required vs defaulted fields)
├── env.py           VectorizedSnakeEnv — N parallel games in NumPy
├── network.py       ActorCritic — CNN policy/value net in MLX
├── ppo.py           RolloutBuffer (GAE) + PPOTrainer (clipped update)
├── checkpoint.py    CheckpointManager — save/load weights + metadata
├── renderer.py      SnakeRenderer — moderngl GLSL renderer (offscreen + window)
├── recorder.py      VideoExporter — per-checkpoint videos, preview, timelapse
├── plots.py         MetricsPlot — live training-curve panel (matplotlib)
├── policy_panel.py  PolicyPanel — live action probs / feature maps / net diagram
├── train.py         Training entry point (python -m snake.train)
└── watch.py         Live watch entry point (python -m snake.watch)
```

Each module has a single responsibility and depends only on those below it:

```
  train.py ──▶ config, env, network, ppo, checkpoint, recorder
  watch.py ──▶ config, env, network, checkpoint, renderer, plots, policy_panel
  recorder ──▶ env, renderer, network
  ppo      ──▶ network
  network  ──▶ (mlx only)
  env      ──▶ (numpy only)
```

## The training data flow

One **iteration** = one rollout + one PPO update = `steps_per_rollout × num_envs`
environment steps (default 128 × 256 = 32,768).

```
  for t in 1..T (steps_per_rollout):
      actions, logp, V = model.select_action(obs)     # GPU, sampled
      next_obs, reward, done = env.step(actions)       # CPU, 256 games
      buffer.add(obs, actions, reward, done, logp, V)
      obs = next_obs

  last_V = model.value(obs)                            # bootstrap
  buffer.compute_gae(last_V, γ, λ)                     # advantages
  trainer.update(buffer)                               # K epochs, GPU

  every checkpoint_every steps:  save weights, (export videos)
  every N iterations:            overwrite preview.mp4
```

Key point: **CPU env-stepping and GPU forward passes alternate** within the
inner loop — they are not overlapped. The environment step (and, when enabled,
the flood-fill shaping) is the main CPU cost; the PPO update is the main GPU cost.

## The watch data flow

```
  load checkpoint weights into ActorCritic
  env = VectorizedSnakeEnv(N=1, auto_reset=False)   # so deaths are visible
  loop:
      action = argmax(policy(obs))                  # greedy, not sampled
      obs, reward, done = env.step(action)
      if crashed:  render red death frame, then reset
      renderer.show(state)                          # GL window
      (optional) policy_panel.update(activations)   # mpl window
      (optional) plotter.refresh()                  # mpl window
```

## Why these choices

The reasoning behind MLX, relative actions, the CNN representation, moderngl,
256 parallel envs, and step-interval checkpointing is documented in the OpenSpec
design record at [`openspec/changes/snake-ppo/design.md`](../openspec/changes/snake-ppo/design.md).
The segment-age observation channel and connectivity reward shaping were added
later; see [environment.md](environment.md).

## Run outputs

Every run writes to `runs/<timestamp>/` (or `--run-dir`):

```
runs/<name>/
├── config.json        copy of the config used
├── metrics.jsonl      one JSON record per PPO update
├── preview.mp4        rolling preview of the current policy (overwritten)
├── checkpoints/
│   └── step_<NNN>/    weights.npz + meta.json
├── videos/            per-checkpoint step_*.mp4 (+ _heatmap) unless --no-video
└── timelapse.mp4      stitched at the end unless --no-video
```
