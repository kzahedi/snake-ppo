# Training

`snake/train.py` — `python -m snake.train`.

## Command

```bash
python -m snake.train --config configs/fill.json [options]
```

| Flag | Effect |
|------|--------|
| `--config PATH` | **required** — the JSON config to run |
| `--run-dir PATH` | output directory (default `runs/<timestamp>/`) |
| `--resume` | continue from the latest checkpoint in `--run-dir`; appends to `metrics.jsonl` |
| `--no-video` | skip per-checkpoint video archive + final timelapse (faster) |
| `--no-preview` | disable the rolling `preview.mp4` |

## What happens on start

1. Config is loaded and validated (missing/typo'd fields fail loudly).
2. The run directory is created; the config is copied to `config.json`.
3. The model + optimiser are built; `--resume` restores weights and step count.
4. **MLX warm-up**: 3 dummy forward passes trigger JIT compilation so the
   throughput timer isn't polluted by first-call compile cost.
5. A banner prints the resolved configuration; the tqdm bar starts.

## The loop

One **iteration** = one rollout (`steps_per_rollout × num_envs` env-steps) + one
PPO update. `total_steps` is rounded to a whole number of iterations so the
progress bar lands exactly on 100%.

```
  collect rollout  →  GAE  →  PPO update  →  log metrics  →
  (every checkpoint_every steps)  save weights [+ videos]  →
  (every N iterations)            overwrite preview.mp4
```

## Progress display

The live tqdm bar:

```
 38%|███▊      | 30.5M/80M [42:11<68:22, 5.5kstep/s, it 931/2441  R 12.40  len 188.2  H 0.21  5,512 sps]
```

| Field | Meaning |
|-------|---------|
| `it 931/2441` | iteration (PPO update) count |
| `R 12.40` | mean **episode return** = apples − 1 → body length ≈ `R + 4` |
| `len 188.2` | mean **steps survived** per episode (not body size) |
| `H 0.21` | mean policy entropy (falls as the policy converges) |
| `5,512 sps` | environment steps per second |

`R`, `len`, and `H` are averaged over episodes that **finished during the last
iteration**, using the **sampled** (exploratory) policy — see
[../docs/algorithm.md](algorithm.md) for why greedy play is usually a bit better.
To convert `R` to fill %: `(R + 4) / (H·W)`.

## Checkpoints

Every `checkpoint_every` **steps**, weights + metadata are saved to
`runs/<name>/checkpoints/step_<NNN>/` (`weights.npz` + `meta.json`). With videos
enabled, a clean + heatmap episode video is also rendered per checkpoint.

## Rolling preview

A single `preview.mp4` of the **current** policy is re-rendered every N
iterations and overwritten in place — a lightweight progress view that works
even with `--no-video`. Cadence:

```
  N = 1000              if total_iters ≥ 10,000
  N = total_iters // 10  otherwise   (≈10 previews across the run)
```

Disable with `--no-preview`.

## Metrics log

`metrics.jsonl` gets one JSON line per update with: `step`, `wall_time`,
`mean_reward`, `mean_ep_length`, `mean_entropy`, `policy_loss`, `value_loss`,
`approx_kl`. This is what `watch --plots` reads to draw live curves.

## Interrupting & resuming

`Ctrl+C` finishes the current rollout, saves a checkpoint, and exits cleanly.
Resume with:

```bash
python -m snake.train --config configs/fill.json --run-dir runs/<name> --resume
```

## Throughput notes

- ~6,000 steps/sec on an M4 for 8×8 with shaping **off**.
- ~3,400 steps/sec with the (currently disabled) flood-fill shaping **on**.
- The CPU env-step and GPU forward pass alternate (not overlapped); the env loop
  is single-core NumPy/Python. Multiprocessing the envs was ruled out for tiny
  8×8 boards (IPC overhead). See [../IDEAS.md](../IDEAS.md) for performance ideas.
