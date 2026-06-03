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
| `algo` | `"ppo"` | `"ppo"` (clipped surrogate) or `"a2c"` (vanilla policy gradient) |
| `render_resolution` | `800` | Pixel size of rendered frames/videos |
| `keep_videos` | `true` | Keep per-checkpoint videos after the timelapse |
| `entropy_floor` | `0.05` | Warn if entropy drops below this early |
| `entropy_floor_step_threshold` | `10_000_000` | …before this many steps |
| `shaping_coef` | `0.0` | Free-space connectivity shaping weight; `0` = off (no flood-fill) |
| `length_reward_coef` | `0.0` | Apple bonus scaled by current fill — rewards **length** (`0` = off) |
| `step_penalty` | `0.0` | Per-step cost — rewards **growth rate** / efficiency (`0` = off) |
| `win_bonus` | `0.0` | Terminal bonus for filling the whole board — rewards **solving** (`0` = off) |
| `thermal_guard` | `true` | Pause training when the CPU thermally throttles (macOS) |
| `thermal_check_every` | `25` | Iterations between thermal checks |
| `thermal_cooldown_s` | `30` | Seconds to pause when hot before re-checking |
| `thermal_pause_limit` | `90` | Pause when `CPU_Speed_Limit` drops below this (100 = unthrottled) |

The shaping terms (`shaping_coef`, `length_reward_coef`, `step_penalty`,
`win_bonus`) feed the **shaped** reward used for learning only; the `R` metric
stays the raw apple count. Raising `gamma` toward 1.0 (e.g. `0.999`) makes total
apples — i.e. final length — the objective rather than *fast* scoring. See
`configs/length.json` / `configs/solve.json`.

### DQN-specific (`train_dqn`, in `configs/dqn.json`)

| Field | Meaning |
|-------|---------|
| `replay_capacity` | Replay buffer size (transitions) |
| `batch_size` | Mini-batch per gradient step |
| `updates_per_iter` | Gradient updates per rollout iteration |
| `target_sync_steps` | Env-steps between target-network syncs |
| `learning_starts` | Env-steps of pure collection before training |
| `eps_start` / `eps_end` / `eps_decay_steps` | ε-greedy schedule |

### Neuroevolution-specific (`train_evo`, in `configs/evo.json`)

| Field | Meaning |
|-------|---------|
| `population` / `elite` | Population size and number of elites kept |
| `sigma` / `init_sigma` | Mutation noise (per generation / initial spread) |
| `episodes_per_eval` | Episodes per individual when measuring fitness |
| `generations` | Number of generations |

(DQN/evo configs also include the standard required fields with dummy values so
they pass the shared config validator; those fields are ignored by those trainers.)

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
| `length.json` | 8×8 | 256 | 80M | **0.999** | off | length-objective experiment (length + growth-rate reward, higher entropy) |
| `solve.json` | 8×8 | 256 | 100M | **0.999** | off | full objective: length + growth + **win bonus** (best PPO; `runs/solve`) |
| `a2c.json` | 8×8 | 256 | 20M | 0.999 | off | A2C comparison run (`algo: a2c`) |
| `dqn.json` | 8×8 | 64 | 20M | 0.999 | off | DQN comparison run (use `train_dqn`) |
| `evo.json` | 8×8 | — | — | — | off | neuroevolution comparison (use `train_evo`; generation-based) |

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
