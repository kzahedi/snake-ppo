# Visualization

Three layers: the **renderer** (GLSL game graphics), the **watch** mode (live
windowed play + controls), and two optional **matplotlib panels** (training
curves, policy internals). Plus offline **video export**.

## The renderer — `snake/renderer.py`

`SnakeRenderer` uses **moderngl** with GLSL shaders, in two modes:

- `mode="offscreen"` — a standalone context renders to a framebuffer and returns
  a NumPy `(res, res, 3)` uint8 frame. Used for video export; no window, works
  headless during training.
- `mode="window"` — a pygame-backed window for live watching. On macOS it
  requests an OpenGL 3.3 **Core profile** (required for moderngl ≥3.3 shaders).

### Shaders

| Element | Shader behaviour |
|---------|-----------------|
| Wall | the framebuffer clears to a stone-purple wall colour; the play field is drawn inset, so the boundary the snake dies against is visible |
| Snake body | per-segment quads, vertex colour interpolated **head → tail** (bright cyan → dark blue) |
| Food | point quad with a radial **glow** that pulses via a `sin(time)` uniform |
| Value heatmap | per-cell quads coloured by V(s) (cool→warm), blended over the game at ~40% opacity |
| Death | body recoloured **red** + dark-red field for one frame on a crash (a win keeps the normal colours) |

`render_frame(state, time, value_grid, dead)` → NumPy frame (offscreen).
`show(state, time, value_grid, dead)` → draws to the window and swaps buffers.

## Watch mode — `snake/watch.py`

```bash
python -m snake.watch --run runs/<name> [options]
```

| Flag | Effect |
|------|--------|
| `--checkpoint STEP\|latest` | which checkpoint to load (default `latest`) |
| `--loop` | restart automatically after death / max-steps |
| `--heatmap` | overlay the value-function heatmap |
| `--policy` | open the live policy panel (see below) |
| `--plots` | open the live training-curve panel |
| `--fps N` | playback speed (default 10 steps/sec) |
| `--max-steps N` | cap per episode (default `grid²·4`) |

It loads the checkpoint, runs the **greedy** policy (argmax) in a single env
(`auto_reset=False` so crashes are shown), and renders each step.

### Crash visibility & loop detection

- A real crash freezes on a **red death frame**, prints `[crash] died after N
  steps`, then resets (with `--loop`).
- If the snake goes `H·W` steps without eating it's looping forever; the watcher
  prints `[loop] …` and resets.

### Controls

Work from **whichever window has focus** — the game window (pygame) *or* a
matplotlib panel (matplotlib key events), since panels steal focus on macOS:

| Key | Action |
|-----|--------|
| `Q` / close | quit |
| `SPACE` | pause / resume |
| `↑` / `↓` | speed ±5 fps |

## Policy panel — `snake/policy_panel.py` (`--policy`)

A matplotlib window showing **what's driving the snake**, updated every step:

```
┌─────────────────────────────────────────┐
│  Action probabilities (policy head)  V(s)│  ← bars: P(left/straight/right),
│  LEFT     ████████████  0.65  ◀ chosen   │     chosen one highlighted
├──────────────────┬──────────────────────┤
│  conv1 (16 maps) │  conv2 (16 maps)      │  ← magma feature-map montages,
├──────────────────┼──────────────────────┤     live activations
│  conv3 (16 maps) │  network diagram      │  ← architecture lit by
│                  │  in→c1→c2→c3→trunk    │     activation intensity
└──────────────────┴──────────────────────┘
```

Driven by `ActorCritic.activations(obs)`, which re-runs the forward pass and
returns the intermediate conv activations, trunk, action probs, and value.

## Training-curve panel — `snake/plots.py` (`--plots`)

Reads `metrics.jsonl` and draws reward / episode-length / entropy curves,
refreshing every 2 seconds. It only animates while **training is actively
writing** to that file (run training and watch concurrently). On a finished run
the curves are static — there's no new data, not a bug.

## Video export — `snake/recorder.py`

`VideoExporter` renders greedy episodes offscreen and encodes H.264/CRF-18 mp4
(QuickTime-compatible) via imageio-ffmpeg:

- `export_checkpoint(model, step, run_dir)` — clean + heatmap variants named by
  step. Gated by `--no-video`.
- `export_preview(model, run_dir)` — single rolling `preview.mp4`, overwritten.
- `assemble_timelapse(run_dir)` — stitches one frame per checkpoint video into a
  `timelapse.mp4` at the end of training.

## Requirements

`moderngl`, `pygame` (window backend), `imageio[ffmpeg]`, `matplotlib`, and
`ffmpeg` on PATH. All installed via `requirements.txt`.
