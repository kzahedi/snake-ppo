# Snake PPO — Documentation

Full documentation for the Snake PPO project: a from-scratch PPO reinforcement
learning agent that learns to play Snake on the Apple GPU via MLX, with GLSL
rendering, video export, and live policy visualisation.

## Contents

| Doc | What it covers |
|-----|----------------|
| [architecture.md](architecture.md) | System overview, data flow, module map |
| [environment.md](environment.md) | The Snake env: state, observation, actions, reward, collisions, shaping |
| [algorithm.md](algorithm.md) | PPO: actor-critic network, GAE, clipped objective, the update loop |
| [training.md](training.md) | Training pipeline, CLI, checkpoints, metrics, the rolling preview |
| [visualization.md](visualization.md) | moderngl renderer, watch mode, policy panel, plots, video export |
| [configuration.md](configuration.md) | Every config field, with ranges and the shipped presets |

## Quick orientation

```
        ┌──────────────────────────────────────────────────────┐
        │                    TRAINING                          │
        │                                                      │
        │  VectorizedSnakeEnv ──obs──▶ ActorCritic (MLX/GPU)   │
        │   (256 envs, NumPy)  ◀─actions─┘     │               │
        │         │                            │               │
        │         └──────▶ RolloutBuffer ──▶ PPOTrainer        │
        │                       │  GAE        (clip + GPU)     │
        │                       ▼                              │
        │              CheckpointManager + VideoExporter       │
        └──────────────────────────────────────────────────────┘
                                  │ weights
        ┌─────────────────────────▼────────────────────────────┐
        │                    WATCHING                          │
        │  checkpoint ──▶ ActorCritic ──▶ SnakeRenderer (GL)   │
        │                      │          + PolicyPanel        │
        │                      └────────▶ + MetricsPlot        │
        └──────────────────────────────────────────────────────┘
```

## The 30-second model

- The snake sees a `(H, W, 3)` grid: **body-age**, **food**, **head**.
- A CNN maps that to **3 relative actions** (turn-left / straight / turn-right)
  plus a value estimate.
- PPO trains it from a sparse reward: **+1** per apple, **−1** on death.
- It runs entirely on the Apple GPU through MLX; the environment is vectorised
  NumPy on the CPU (256 games in lockstep).

See [environment.md](environment.md) and [algorithm.md](algorithm.md) for the details.
