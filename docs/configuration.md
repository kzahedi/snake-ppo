# Configuration

Runs are driven entirely by a JSON config (`configs/*.json`), validated on
startup by `snake/config.py`. Required fields must be present and correctly
typed; optional fields fall back to defaults.

## Required fields

| Field | Type | Meaning |
|-------|------|---------|
| `grid_size` | int | Board is `grid_size × grid_size` |
| `num_envs` | int | Parallel games (rollout batch width) |
| `total_steps` | int | Total env-steps to train (rounded to whole iterations) |
| `steps_per_rollout` | int | Steps collected per env before each PPO update (`T`) |
| `ppo_epochs` | int | Passes over each rollout per update (`K`) |
| `mini_batch_size` | int | Samples per gradient step |
| `lr` | float | Adam learning rate |
| `gamma` | float | Discount γ — planning horizon |
| `gae_lambda` | float | GAE λ — advantage bias/variance |
| `clip_eps` | float | PPO clip range ε |
| `entropy_coef` | float | Entropy bonus weight |
| `value_coef` | float | Value-loss weight |
| `max_grad_norm` | float | Gradient clipping (global norm) |
| `checkpoint_every` | int | Save a checkpoint every N **steps** |

## Optional fields (with defaults)

| Field | Default | Meaning |
|-------|---------|---------|
| `render_resolution` | `800` | Pixel size of rendered frames/videos |
| `keep_videos` | `true` | Keep per-checkpoint videos after the timelapse |
| `entropy_floor` | `0.05` | Warn if entropy drops below this early |
| `entropy_floor_step_threshold` | `10_000_000` | …before this many steps |
| `shaping_coef` | `0.0` | Free-space connectivity shaping weight; `0` = off (no flood-fill) |

## One iteration

```
  steps_per_iter = steps_per_rollout × num_envs
  total_iters    = round(total_steps / steps_per_iter)
```

For `fill.json`: 128 × 256 = 32,768 steps/iter; 80M ÷ 32,768 ≈ 2,441 iterations.

## Shipped presets

| Config | Grid | Envs | Steps | γ | Shaping | Purpose |
|--------|------|------|-------|---|---------|---------|
| `quick.json` | 8×8 | 64 | 1M | 0.99 | off | smoke test (~3 min) |
| `medium.json` | 16×16 | 256 | 20M | 0.99 | off | scaling check (~1 h) |
| `overnight.json` | 32×32 | 256 | 200M | 0.99 | off | large-grid overnight |
| `fill.json` | 8×8 | 256 | 80M | **0.997** | off* | push toward high board fill |

\* `fill.json` ships with `shaping_coef: 0.0`. The connectivity shaping was
disabled after it halved throughput for little early benefit; re-enable by
setting `shaping_coef` (e.g. `0.05`) if the segment-age channel plateaus below
the fill target. See [../IDEAS.md](../IDEAS.md).

## Tuning guidance

- **Want higher fill?** First lever is more `total_steps`. `gamma` already at
  0.997 for the long horizon. Then consider re-enabling `shaping_coef`.
- **Want faster iteration?** Smaller `grid_size`, fewer `num_envs`, shaping off.
- **Training unstable / entropy collapses early?** Lower `lr`, raise
  `entropy_coef`, or lower `clip_eps`. Watch `approx_kl` in `metrics.jsonl`.
- **Bigger board (16×16, 32×32)?** Episodes get longer, so `R` and `len` climb
  more slowly; expect to need proportionally more `total_steps`.

## Creating a config

Copy a preset and edit:

```bash
cp configs/fill.json configs/my_run.json
# edit fields, then:
python -m snake.train --config configs/my_run.json --no-video
```
