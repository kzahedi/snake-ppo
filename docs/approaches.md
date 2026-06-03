# Approaches & Comparison

The project implements several agents on one shared environment + evaluation
harness, so they can be compared head-to-head. PPO is the deep dive; the others
exist for comparison.

## The agents

| Agent | Module | Paradigm | Trains via |
|-------|--------|----------|-----------|
| **PPO** | `ppo.py`, `network.py` | on-policy policy-gradient, clipped surrogate | `python -m snake.train` |
| **A2C** | `ppo.py` (`algo: "a2c"`) | on-policy, vanilla policy gradient (no clip) | `python -m snake.train` |
| **DQN** | `dqn.py`, `train_dqn.py` | off-policy value-based (replay, target net, Double-DQN, ε-greedy) | `python -m snake.train_dqn` |
| **Neuroevolution** | `evolution.py`, `train_evo.py` | gradient-free (μ,λ) evolution strategy | `python -m snake.train_evo` |
| **Hamiltonian** | `baselines.py` | hand-coded cycle follower | no training |
| **greedy-A\*** | `baselines.py` | shortest-path-to-food | no training |
| **flood-fill** | `baselines.py` | safe-greedy (vetoes space-stranding moves) | no training |

All learning methods optimise the **same shaped reward** (`rewards.py`):
`+1` eat, `−1` die, `+win_bonus` for filling the board, plus a fill-scaled apple
bonus and a small per-step cost. The algorithmic baselines share the env's
relative action space.

## Results (8×8, 40 eval episodes; learned methods at an equal 20M-step budget)

| Agent | Mean fill | Solve rate |
|-------|-----------|-----------|
| **PPO + safety shield** | **99%** | **64%** |
| Hamiltonian | 79% | **78%** |
| **PPO** | **99%** | 50% |
| flood-fill | 41% | 0% |
| DQN | 36% | 0% |
| greedy-A* | 36% | 0% |
| A2C | 5% | 0% |
| Neuroevolution | 5% | 0% |

**Safety shield** (`shield.py`, run with `--shielded <run_dir>`): an
inference-time wrapper — the policy ranks the moves, the shield takes the
highest-ranked one that keeps the snake's tail reachable (so it can never trap
itself), trusting the policy on eating/winning moves. Adds **no training** and
lifts PPO's solve rate 50% → 64%, the best of the learned agents.

### Reading the results

- **PPO** reaches the highest fill of *any* agent (above the hand-coded
  Hamiltonian) and is the only learned method that completes the board.
- **DQN** learns to eat (~20 apples) but never solves at this budget.
- **A2C** = PPO minus clipping → collapses (entropy → 0) and barely learns. The
  clean ablation: PPO's clip is what makes it stable.
- **Neuroevolution** is far less sample-efficient; at this budget it only learns
  to survive briefly. (Fitness = fill + food-proximity + survival + win.)
- **Hamiltonian** is the algorithmic optimal reference (high solve-rate; lower
  *mean* fill because random starts occasionally misalign and it wanders early).

## Running the comparison

```bash
# Full pipeline: train A2C → DQN → neuroevolution → regenerate chart + video
bash scripts/run_comparison.sh

# Compare already-trained agents (writes assets/comparison.png + behaviours.mp4)
python -m snake.compare --grid 8 --episodes 40 \
    --ppo runs/solve:PPO --ppo runs/a2c:A2C --dqn runs/dqn:DQN --ppo runs/evo:Evolution

# Numbers only
python -m snake.eval --grid 8 --episodes 40 --ppo runs/solve:PPO --dqn runs/dqn:DQN
```

`compare`/`eval` take `--ppo PATH[:Label]` (ActorCritic checkpoints: PPO, A2C,
neuroevolution all save this format) and `--dqn PATH[:Label]` (Q-networks).
`--baselines` selects the hand-coded agents.

## Caveats

- The 20M-step budget is modest; DQN and A2C could do better with more steps /
  tuning. The comparison is "equal budget", not "each to convergence".
- The Hamiltonian baseline doesn't safely path onto its cycle from a random
  start, so ~20% of its episodes die early — its true ceiling is higher.
