## 1. Project Setup

- [x] 1.1 Create project structure: `snake/` package with `__init__.py`, `configs/`, `runs/` directories
- [x] 1.2 Install dependencies: `pip install mlx moderngl pygame imageio[ffmpeg]`
- [x] 1.3 Create `configs/quick.json` (8×8, 1M steps), `configs/medium.json` (16×16, 20M steps), `configs/overnight.json` (32×32, 200M steps)
- [x] 1.4 Add config schema validation in `snake/config.py` with clear error messages for missing/invalid fields

## 2. Snake Environment

- [x] 2.1 Implement `snake/env.py` — `VectorizedSnakeEnv` class with numpy-based batch state: N grids, N snake body deques, N directions, N food positions
- [x] 2.2 Implement relative action mapping (0=left, 1=straight, 2=right) with U-turn prevention
- [x] 2.3 Implement step logic: move head, check collision (wall + self), check food, grow body, place new food, auto-reset dead envs
- [x] 2.4 Implement `observation()` method producing float32 arrays of shape (N, H, W, 3) — body/food/head channels
- [x] 2.5 Write smoke test: step 256 envs for 1000 steps, verify observation shapes, reward range, done flags trigger resets

## 3. CNN Actor-Critic Network

- [x] 3.1 Implement `snake/network.py` — `ActorCritic` class in MLX with 3 conv layers (32→64→64, 3×3, same), 512-unit trunk, actor head (softmax, 3), value head (scalar)
- [x] 3.2 Verify network runs on MLX GPU device — check `mx.default_device()` returns GPU
- [x] 3.3 Test forward pass: batch of (4, 8, 8, 3) observations → actor shape (4, 3), value shape (4,)
- [x] 3.4 Implement `select_action(obs)` returning actions, log_probs, values; and `evaluate(obs, actions)` returning log_probs, values, entropy for PPO update

## 4. PPO Trainer

- [x] 4.1 Implement `snake/ppo.py` — `RolloutBuffer` class accumulating (obs, action, reward, done, log_prob, value) for T steps × N envs
- [x] 4.2 Implement GAE advantage computation: iterate backwards through rollout, compute δ = r + γ·V(s') − V(s), accumulate with λ discount
- [x] 4.3 Implement PPO update: compute ratio, clipped surrogate loss, value loss, entropy bonus; sum as total loss; apply gradient clip (max norm 0.5)
- [x] 4.4 Implement K-epoch mini-batch loop with rollout shuffling; hold old log_probs and values fixed across epochs
- [x] 4.5 Implement entropy floor warning: log warning to metrics if entropy < 0.05 before step 10M
- [x] 4.6 Test single PPO update: verify loss decreases on a trivial synthetic batch, verify parameter values change

## 5. Checkpoint Manager

- [x] 5.1 Implement `snake/checkpoint.py` — `CheckpointManager` class with `save(step, model, optimizer, metrics)` writing `weights.npz`, `optimizer.npz`, `meta.json` to `checkpoints/step_{step:010d}/`
- [x] 5.2 Implement `load(step_or_"latest")` restoring weights, optimizer state, returning step counter; raise `CheckpointNotFoundError` if missing
- [x] 5.3 Implement `list()` returning sorted list of `{step, wall_time, mean_reward, mean_ep_length, mean_entropy}` dicts
- [x] 5.4 Test round-trip: save checkpoint, reload, verify all parameter values match before/after

## 6. moderngl Renderer

- [x] 6.1 Implement `snake/renderer.py` — `SnakeRenderer` class initializing either offscreen (`create_standalone_context`) or windowed (pygame backend) based on `mode` parameter
- [x] 6.2 Write GLSL vertex/fragment shaders for snake body: quad mesh per segment, vertex colour interpolated head→tail along body index
- [x] 6.3 Write GLSL fragment shader for food: radial glow with `sin(time)` pulse uniform, configurable colour
- [x] 6.4 Implement background render: dark fill with subtle grid lines (thin quads or lines primitive)
- [x] 6.5 Implement `render_frame(state, time=0.0)` returning numpy uint8 (H_px, W_px, 3) in offscreen mode
- [x] 6.6 Implement `show(state, time=0.0)` for windowed mode: render to screen, swap buffers, poll pygame events
- [x] 6.7 Implement value heatmap overlay: accept (H, W) float array, render coloured quads at configurable opacity blended over game render
- [x] 6.8 Implement `should_quit()` checking pygame QUIT event and Q keypress
- [x] 6.9 Smoke test offscreen rendering: render a frame of a known state, verify output shape and that head pixel is brighter than tail pixel

## 7. Video Exporter

- [x] 7.1 Implement `snake/recorder.py` — `VideoExporter` class with `record_episode(model, env_config, step, run_dir)` that runs one greedy episode and saves frames
- [x] 7.2 Implement frame-to-video writer using `imageio.get_writer` with ffmpeg backend, H.264/CRF 18, 30 fps
- [x] 7.3 Implement dual-variant export: call `record_episode` twice — once with heatmap=False, once with heatmap=True — saving `step_XXXXK.mp4` and `step_XXXXK_heatmap.mp4`
- [x] 7.4 Implement `assemble_timelapse(run_dir, fps=24)` reading first frame of each checkpoint video in step order and writing `timelapse.mp4`
- [x] 7.5 Implement `keep_videos=False` cleanup: delete `step_*.mp4` after timelapse assembly
- [x] 7.6 Test: export a 50-step episode video, verify file exists, is valid mp4, and is QuickTime-playable

## 8. Training Pipeline

- [x] 8.1 Implement `snake/train.py` — `__main__` entry point parsing `--config`, `--run-dir`, `--resume` args; create timestamped run directory; copy config
- [x] 8.2 Implement main training loop: collect rollout (T steps × N envs) → GAE → PPO update → log metrics → checkpoint check
- [x] 8.3 Implement JSONL metrics logging: one record per PPO update with all 8 required fields
- [x] 8.4 Implement stdout progress line: format `step=  sps=  reward=  ep_len=  entropy=` after each update
- [x] 8.5 Implement `--resume` flag: detect existing run dir, load latest checkpoint, append to existing metrics.jsonl
- [x] 8.6 Implement SIGINT handler: finish current rollout, save checkpoint, print checkpoint path, exit cleanly
- [x] 8.7 Wire checkpoint manager: call `save()` and `record_episode()` every `checkpoint_every` steps
- [x] 8.8 MLX warm-up: run 3 dummy forward+backward passes before starting the training timer to trigger JIT compilation
- [x] 8.9 End-to-end smoke test: run `configs/quick.json` for 50k steps, verify run dir structure, one checkpoint, one video pair exist

## 9. Watch Mode

- [x] 9.1 Implement `snake/watch.py` — `__main__` entry point parsing `--run`, `--checkpoint`, `--heatmap`, `--fps`, `--loop`
- [x] 9.2 Load checkpoint, construct single-instance env (N=1), run greedy episode (argmax over actor output)
- [x] 9.3 Implement game loop: step env → render frame → `renderer.show()` → sleep to hit target fps
- [x] 9.4 Implement keyboard controls: Q/window-close → exit; SPACE → pause/unpause; UP/DOWN arrows → ±5 fps
- [x] 9.5 Implement `--loop` mode: restart episode automatically after done=True
- [x] 9.6 Implement `--heatmap` overlay: evaluate value head on current obs, pass value grid to renderer each step
- [x] 9.7 Manual test: load a trained checkpoint, open watch window, verify snake plays smoothly at default fps

## 10. Integration and Overnight Run

- [x] 10.1 Run `configs/quick.json` end-to-end: verify env, network, PPO, checkpoint, video all work in concert
- [ ] 10.2 Run `configs/medium.json` for 1 hour: check that reward increases over time, entropy decays gradually
- [ ] 10.3 Launch `configs/overnight.json` (32×32, 200M steps): let run overnight, verify stable training (no NaN, no entropy collapse)
- [ ] 10.4 Assemble timelapse from overnight run, review policy evolution from chaos to meditative spirals
- [ ] 10.5 Watch mode test on overnight checkpoint: load step 100M checkpoint, verify fluent meditative play in window
