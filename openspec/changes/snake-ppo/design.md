## Context

Greenfield project on Mac Mini M4 (32 GB unified memory, 40-core GPU, Metal 4). No existing codebase — all decisions are unconstrained by legacy. The goal is both a working RL system and a visually compelling artefact: policy learning made visible through beautiful rendering and timelapse video.

MLX and PyTorch (with MPS) are both installed. ffmpeg, imageio, and numpy are available. pygame and moderngl need installing.

## Goals / Non-Goals

**Goals:**
- Train PPO agent on Snake to convergence overnight on M4 GPU
- Render episodes beautifully with moderngl (offscreen training, windowed watch)
- Produce per-checkpoint .mp4 episodes and a final timelapse video
- Keep the codebase simple enough to experiment with (not a production library)

**Non-Goals:**
- CoreML / ANE export (training on GPU is sufficient; ANE is inference-only and adds complexity)
- Multi-agent or self-play variants
- Web UI or remote training dashboard
- Support for non-Apple platforms

## Decisions

### D1: MLX over PyTorch/MPS for neural network ops

MLX runs natively on Apple Silicon unified memory with no CPU↔GPU copies. Its NumPy-like API makes custom RL loops natural to write. PyTorch has a richer RL ecosystem (CleanRL, Stable Baselines) but MPS support is still maturing and the abstraction adds friction for custom PPO.

**Chosen:** MLX for all tensor ops. NumPy for environment stepping (pure CPU, no GPU needed for grid logic).

**Alternative considered:** PyTorch + MPS — ruled out because MLX is the stated goal and the RL loop we're writing is simple enough not to need SB3.

### D2: Relative action space (3 actions) over absolute (4 directions)

Relative actions (turn left, go straight, turn right) make U-turns physically impossible — the agent can never instantly reverse into itself. This eliminates a large class of trivial deaths early in training, accelerating convergence. Policies that emerge tend to be smoother and more meditative: long arcs rather than erratic pivots.

**Chosen:** 3 relative actions.

**Alternative considered:** 4 absolute directions — simpler state encoding but slower convergence and more chaotic early-training behavior.

### D3: CNN over feature-engineered MLP

A CNN operating on the raw grid (3 channels: body mask, food mask, head mask) learns spatial representations without hand-crafting features. The value function heatmap that emerges from a CNN is spatially meaningful — it shows which cells the agent considers safe or desirable — which is directly useful for the overlay visualisation.

**Chosen:** 3 conv layers (32→64→64 filters, 3×3 kernel, same padding) → flatten → 512-unit dense trunk → actor head (softmax, 3 outputs) + value head (scalar).

**Alternative considered:** MLP on hand-crafted features (distances, danger flags) — faster convergence but no spatial heatmap and less interesting learned representations.

### D4: moderngl for rendering over pygame

moderngl gives GPU-accelerated fragment shaders: the food glow, body gradient, and value heatmap overlay are all implemented as GLSL shaders running on the same M4 GPU used for training. pygame rectangles cannot produce the same visual quality. moderngl supports both offscreen (standalone context) and windowed (pygame backend) modes from the same code path.

**Chosen:** moderngl with pygame as the window backend for watch mode; standalone context for headless training.

**Alternative considered:** Pure pygame — fast to set up but limited to software blitting; no shader effects.

### D5: 256 parallel environments in numpy

PPO is on-policy and benefits from large, diverse rollout batches. 256 parallel environments (stepped in vectorized numpy) produce 32,768 samples per rollout (256 envs × 128 steps). The environment step is pure Python/numpy — no GPU needed — and runs fast enough that it does not bottleneck the GPU-side PPO update. Starting more envs beyond 256 yields diminishing returns without further optimisation.

**Chosen:** 256 envs, 128 steps per rollout = 32,768 samples/update.

### D6: Checkpoint + video at fixed step intervals

Rather than episode-count-based checkpointing (variable time between saves), checkpoint every N environment steps. This gives uniform spacing in the timelapse video regardless of how episode lengths change during training. Every checkpoint triggers: save weights, run 1 greedy evaluation episode, render it to video.

**Chosen:** Checkpoint every 500k env steps. For a 200M-step overnight run this produces ~400 checkpoints and ~400 video files.

## Risks / Trade-offs

- **moderngl offscreen on macOS headless**: `moderngl.create_standalone_context()` requires a display server or virtual framebuffer on some macOS versions. The watch mode window (pygame backend) confirms a working display; offscreen training rendering may need `EGL` or a dummy framebuffer if run without a session. **Mitigation**: test offscreen rendering first in a quick smoke-test run before committing to overnight.

- **MLX graph compilation overhead**: MLX uses lazy evaluation and JIT-compiles the compute graph. The first few training steps are slow while the graph compiles. **Mitigation**: warm up with a few dummy steps before starting the training timer.

- **Video file accumulation**: 400 checkpoint videos × ~10 MB each = ~4 GB for an overnight run on 32×32. **Mitigation**: configurable `keep_videos` flag; timelapse assembly reads all files then optionally deletes intermediates.

- **PPO instability on large grids**: PPO can collapse (entropy → 0 early) if the clip range or learning rate is too aggressive. **Mitigation**: log entropy at every update; add a configurable entropy floor that triggers an LR reset if entropy drops below threshold before step 10M.

## Migration Plan

Greenfield — no migration required. To start:
```
pip install mlx moderngl pygame imageio[ffmpeg]
python -m snake.train --config configs/quick.json   # smoke test
python -m snake.train --config configs/overnight.json
```

## Open Questions

- Should the value heatmap render pass use the actor's value head (V(s) for each cell evaluated independently) or a pre-computed spatial value grid? Independent evaluation is correct but expensive for large grids; a spatial approximation may suffice for visualisation.
- Is a separate `configs/medium.json` (16×16) worth running before overnight, or jump straight to 32×32 after the quick smoke test?
