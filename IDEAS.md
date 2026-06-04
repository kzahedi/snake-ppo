# Ideas & Backlog

A living scratchpad for things we might try. Not commitments — just captured
thinking so nothing gets lost. Add freely; prune when done or abandoned.

Status key: 💡 idea · 🔬 worth experimenting · ⏳ deferred · ✅ done · ❌ ruled out

---

## Reaching higher board fill (the 99% goal) — ✅ ACHIEVED on 8×8

PPO solves the 8×8 board (~99% mean fill, full-board solves) via the segment-age
channel + γ=0.999 + length/win reward shaping. The connectivity shaping and
safety shield turned out to be **unnecessary** for 8×8. Remaining threads:

- ⏳ **Push the solve-rate higher** — `runs/solve` is paused at 92M / 77% solve;
  resume to climb further (`--resume`).
- ✅ **Hybrid: RL + safety shield** — built (`shield.py`): inference-time
  flood-fill guard that vetoes self-trapping moves. Lifts PPO 50% → 64% solve
  with no training (`--shielded` in eval/compare).
- 🔬 **Deeper-lookahead shield** — current shield trusts eating moves (the leak);
  a 1–2 step lookahead before trusting an eat could push solve-rate toward ~100%.
- 🔬 **Optimized connectivity shaping** — still an option (numba-JIT the BFS,
  skip when short) if larger grids need the anti-trap signal. Off for 8×8.
- 💡 **Curriculum** — start small, grow the grid; likely needed for 32×32.
- 💡 **Bigger / deeper network** for larger grids.

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

- ✅ **Does the age channel alone crack it?** — yes; PPO solved 8×8 with it.
- ✅ **Policy comparison** — PPO vs A2C vs DQN vs neuroevolution + 3 algorithmic
  baselines, on a shared env/eval harness. Done; chart + behaviour video + table
  in the README. See [docs/approaches.md](docs/approaches.md).
  Result: PPO 99% fill ≫ DQN 37% ≫ A2C/evo 5% (all at 20M steps); Hamiltonian
  80% solve.
- 🔬 **Fairer DQN / A2C** — they got an equal 20M-step budget; give them more
  steps and tuning (A2C: stronger entropy reg / lower lr to avoid collapse; DQN:
  longer ε decay, prioritized replay) for a more flattering comparison.
- 💡 **Ablations** — age-channel on/off, relative vs absolute actions, γ sweep.
- 🔬 **Large grids** — 16×16 / 32×32: does PPO still solve? The open question.
  Likely needs curriculum + bigger net. `medium.json` / `overnight.json` exist.

## Visualization & UX (updated)

- ✅ Visible wall border around the play field.
- ✅ Previews end on a real death/win (not a step cap); win-rate in the bar.
- ✅ Comparison chart + side-by-side behaviour video (`compare.py`).
- ⏳ **Direct fill% readout** in the live bar (still `R + 4` mentally for PPO).
- 💡 `--preview-every N`; static-plots note in `--plots`.

## Hardware exploration (the original motivation)

- 💡 **CoreML / ANE export** — train on GPU, export the policy to CoreML and run
  inference on the Neural Engine. Was a non-goal for training, but a fun way to
  actually touch the ANE. Measure inference latency/power vs GPU.
- 💡 Benchmark MLX vs PyTorch-MPS on the same PPO loop.

---

_Last touched: 2026-06-03 — see [STATE.md](STATE.md) for current project state & resume commands._
