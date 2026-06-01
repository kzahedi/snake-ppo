# Ideas & Backlog

A living scratchpad for things we might try. Not commitments — just captured
thinking so nothing gets lost. Add freely; prune when done or abandoned.

Status key: 💡 idea · 🔬 worth experimenting · ⏳ deferred · ✅ done · ❌ ruled out

---

## Reaching higher board fill (the 99% goal)

- 🔬 **Optimized connectivity shaping** — bring back the flood-fill self-trap
  penalty, but cheap: only compute once the snake is long enough to fragment
  the board (it's a no-op while short), and/or JIT the BFS with numba, and/or
  use `scipy.ndimage.label`. We turned the naive version off because it halved
  throughput and did nothing early. Revisit if age-channel + γ plateaus.
- ⏳ **Hybrid: RL + safety shield** — learned policy proposes a move; a
  flood-fill safety layer vetoes any move that would strand the tail, falling
  back to a safe move. Reliably reaches ~99–100% fill. Less "pure learning",
  so deferred unless pure RL stalls.
- 💡 **Curriculum** — start on a tiny grid, grow it as the policy improves, so
  late-game (long-snake) skills transfer instead of being learned from scratch.
- 💡 **Reward shaping alternatives** — distance-to-tail reward, survival bonus,
  or penalty for `reachable_free < body_length` (a guaranteed future trap).
- 💡 **Bigger / deeper network** once 8×8 is solved, before scaling the grid.

## Performance

- ✅ Cache connectivity Φ across steps (compute flood-fill once, not twice).
- ❌ Multiprocessing env workers (SubprocVecEnv) — IPC overhead cancels the gain
  for tiny 8×8 envs. Reconsider only for large grids (32×32+).
- 💡 **Vectorize the env step** in NumPy (drop the per-env Python loop). Hard
  because of variable-length deque bodies, but the biggest single-core win.
- 💡 **numba JIT** the hot paths (env step + flood-fill) — native speed, one dep.
- 💡 Overlap CPU env-stepping with GPU work (currently serial per timestep).

## Visualization & UX

- ⏳ **Direct fill% readout** in the progress bar: `fill 50% (best 53%)` from
  actual body length at death (mean + max), instead of mental `R + 4` math.
- 💡 `--preview-every N` flag to control preview cadence directly.
- 💡 Note in `--plots` when `metrics.jsonl` stops growing ("training finished,
  showing final curves") so a static plot isn't mistaken for a bug.
- 💡 Side-by-side video: clean render + value heatmap + policy panel in one frame.
- 💡 Render the Hamiltonian-style path the converged snake traces (trail overlay).

## Experiments & research

- 🔬 **Does the age channel alone crack it?** — the current `fill.json` run.
  Baseline before re-adding shaping. (in progress)
- 💡 **Policy comparison** — pit PPO vs DQN vs neuroevolution (NEAT) on the same
  env; compare fill %, sample efficiency, and how "meditative" each looks.
- 💡 **Ablations** — age-channel on/off, relative vs absolute actions, γ sweep,
  to see which choices actually matter for fill.
- 💡 **Large grids** — 16×16 and 32×32 overnight; watch the policy rediscover
  sweeping Hamiltonian loops from reward alone.

## Hardware exploration (the original motivation)

- 💡 **CoreML / ANE export** — train on GPU, export the policy to CoreML and run
  inference on the Neural Engine. Was a non-goal for training, but a fun way to
  actually touch the ANE. Measure inference latency/power vs GPU.
- 💡 Benchmark MLX vs PyTorch-MPS on the same PPO loop.

---

_Last touched: 2026-06-01_
