# Project Status — pick up here

_Snapshot of where the Snake-PPO project stands, so work can resume cleanly._
_Last updated: 2026-06-04._

## TL;DR

- **PPO solves 8×8 Snake** — fills the entire board (~99% mean fill, full-board
  solves), discovered from a `+1 eat / −1 die` reward + length/win shaping. No
  hand-coded cycle, no safety shield.
- **A full multi-approach comparison is done and published** (PPO vs A2C vs DQN
  vs neuroevolution vs Hamiltonian / greedy-A* / flood-fill) — chart + behaviour
  GIF + results table in the README.
- **Nothing is training right now.** The GPU is free. One run is *paused and
  resumable* (the PPO solve fine-tune, see below).

## Headline comparison (8×8, 50 eval episodes; A2C/DQN/evo retrained & tuned)

| Agent | Type | Mean fill | Solve rate |
|-------|------|-----------|-----------|
| **PPO + safety shield** | learned + flood-fill guard | **99%** | **64%** |
| Hamiltonian | hand-coded cycle | 79% | **78%** |
| **PPO** | learned, clipped policy-gradient | **99%** | 50% |
| DQN | learned, value-based (tuned) | 66% | 0% |
| flood-fill | hand-coded | 41% | 0% |
| greedy-A* | hand-coded | 36% | 0% |
| A2C | learned, no clipping (tuned) | 20% | 0% |
| Neuroevolution | gradient-free | 5% | 0% |

## Runs on disk

| Run dir | What it is | State | Best |
|---------|-----------|-------|------|
| `runs/solve` | **PPO fine-tune** (win-state objective) | **paused @ 92M steps — resumable** | ~100% fill, 77% sampled solve-rate |
| `runs/20260601_213205` | original PPO length run (the showcase GIF source) | complete (80M) | ~100% fill |
| `runs/a2c` | A2C comparison run (tuned) | complete (30M) | ~20% fill (collapses late) |
| `runs/dqn` | DQN comparison run (tuned) | complete (30M) | 66% fill |
| `runs/evo` | neuroevolution comparison run | complete (600 gens) | 5% fill |
| `runs/verify`, `runs/reward_test`, `runs/2026060*` | early dev/test runs | superseded | — |

The two PPO checkpoints (`runs/solve`, `runs/20260601_213205`) are the keepers.

## Resume / re-run commands

```bash
conda activate snake

# Resume the PPO solve fine-tune from step 92M (push solve-rate higher)
python -m snake.train --config configs/solve.json --run-dir runs/solve --no-video --resume

# Watch the best policy play (with the policy panel / value heatmap)
python -m snake.watch --run runs/solve --loop --policy --heatmap

# Re-run the whole approach comparison (A2C → DQN → evo → chart+video)
bash scripts/run_comparison.sh

# Compare any set of trained agents (chart + behaviour video)
python -m snake.compare --grid 8 --episodes 40 \
    --ppo runs/solve:PPO --dqn runs/dqn:DQN --ppo runs/a2c:A2C --ppo runs/evo:Evolution

# Just the numbers
python -m snake.eval --grid 8 --episodes 40 --ppo runs/solve:PPO --dqn runs/dqn:DQN
```

## What's implemented

- **Approaches:** PPO (`ppo.py`), A2C (`algo: a2c` flag), DQN (`dqn.py` +
  `train_dqn.py`), neuroevolution (`evolution.py` + `train_evo.py`), and three
  algorithmic baselines (`baselines.py`: Hamiltonian, greedy-A*, flood-fill).
- **Shared infra:** env (`env.py`), reward shaping (`rewards.py`), checkpoints
  (`checkpoint.py`), GLSL renderer with wall border (`renderer.py`), video +
  preview (`recorder.py`), live panels (`plots.py`, `policy_panel.py`), thermal
  guard (`thermal.py`), eval + comparison harness (`eval.py`, `compare.py`).

## Open threads / next steps

See [IDEAS.md](IDEAS.md) for the full backlog. Most likely next moves:
1. **Push PPO+shield higher** — `shield.py` lifted the solve rate 50% → 64%; a
   deeper-lookahead shield (it currently trusts eating moves) could approach
   ~100%. (PPO refinement on `runs/solve` itself saturated at ~50% greedy.)
2. **DQN/A2C tuned** (done): DQN 36%→66% fill (slower ε-decay, γ 0.997, bigger
   replay); A2C 5%→20% (lower lr, more entropy — but still collapses late). Could
   push DQN further (n-step returns, dueling, prioritized replay) or stabilise
   A2C more. Neuroevolution stayed ~5% even at 600 gens — gradient-free is just
   sample-starved here.
3. **Scale the comparison to 16×16 / 32×32** — does PPO still solve? (the open
   research question).

Note: the PPO solve fine-tune (`runs/solve`) reached ~128M steps but the
solve-rate **saturated at ~50% greedy** — refinement gave diminishing returns,
so the shield (lever #1) is the productive path, not more PPO training.
