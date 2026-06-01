from __future__ import annotations

import argparse
import time

import mlx.core as mx
import numpy as np

from snake.checkpoint import CheckpointManager, CheckpointNotFoundError
from snake.env import VectorizedSnakeEnv
from snake.network import ActorCritic
from snake.renderer import SnakeRenderer


def main():
    parser = argparse.ArgumentParser(description="Watch a trained snake policy")
    parser.add_argument("--run", required=True, help="Run directory")
    parser.add_argument("--checkpoint", default="latest",
                        help="Step number or 'latest'")
    parser.add_argument("--heatmap", action="store_true",
                        help="Show value heatmap overlay")
    parser.add_argument("--fps", type=int, default=10,
                        help="Game steps per second")
    parser.add_argument("--loop", action="store_true",
                        help="Restart episode after death or max-steps")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Max steps per episode before restart (default: grid_size^2 * 4)")
    parser.add_argument("--plots", action="store_true",
                        help="Show live training curves from metrics.jsonl")
    parser.add_argument("--policy", action="store_true",
                        help="Show live policy panel: action probs, conv feature maps, network diagram")
    args = parser.parse_args()

    # Load config from run dir
    import json
    from pathlib import Path
    cfg_path = Path(args.run) / "config.json"
    with open(cfg_path) as f:
        cfg = json.load(f)

    H = W = cfg["grid_size"]
    resolution = cfg.get("render_resolution", 800)

    # Load model weights
    ckpt_step = args.checkpoint if args.checkpoint == "latest" else int(args.checkpoint)
    mgr = CheckpointManager(args.run)
    model = ActorCritic(H, W)
    step = mgr.load_weights_into(model, ckpt_step)
    print(f"Loaded checkpoint: step {step:,}")

    plotter = None
    if args.plots:
        from snake.plots import MetricsPlot
        plotter = MetricsPlot(args.run, refresh_every=2.0)
        plotter.refresh(force=True)

    policy_panel = None
    if args.policy:
        from snake.policy_panel import PolicyPanel
        policy_panel = PolicyPanel(H, W)

    renderer = SnakeRenderer(H, W, resolution=resolution, mode="window")
    env = VectorizedSnakeEnv(H, W, N=1, auto_reset=False)
    obs = env.observation()

    t = 0.0
    max_steps = args.max_steps if args.max_steps is not None else H * W * 4
    ep_step = 0
    steps_since_food = 0
    loop_threshold = H * W  # can't go H*W steps without eating unless looping

    pygame = renderer._pygame

    # Shared control state. Keys are handled from whichever window has focus —
    # the pygame game window OR a matplotlib panel (which steals focus on macOS).
    ctrl = {"fps": args.fps, "step_interval": 1.0 / args.fps,
            "paused": False, "quit": False}

    def handle_key(name: str):
        if name in ("q", "escape"):
            ctrl["quit"] = True
        elif name in ("space", " "):
            ctrl["paused"] = not ctrl["paused"]
            print("Paused" if ctrl["paused"] else "Resumed")
        elif name == "up":
            ctrl["fps"] = min(ctrl["fps"] + 5, 120)
            ctrl["step_interval"] = 1.0 / ctrl["fps"]
            print(f"Speed: {ctrl['fps']} fps")
        elif name == "down":
            ctrl["fps"] = max(ctrl["fps"] - 5, 1)
            ctrl["step_interval"] = 1.0 / ctrl["fps"]
            print(f"Speed: {ctrl['fps']} fps")

    # pygame keycode -> name, so both backends feed the same handler
    _pg_names = {pygame.K_q: "q", pygame.K_ESCAPE: "escape",
                 pygame.K_SPACE: "space", pygame.K_UP: "up", pygame.K_DOWN: "down"}

    # Register matplotlib key handlers so keys work when a panel has focus
    for panel in (plotter, policy_panel):
        if panel is not None:
            panel.fig.canvas.mpl_connect("key_press_event", lambda e: handle_key(e.key))

    print("Controls: Q=quit  SPACE=pause  UP/DOWN=speed  "
          "(focus the game OR a panel window)")

    running = True
    while running:
        frame_start = time.time()

        if not ctrl["paused"]:
            # Greedy action
            x = mx.array(obs)
            probs, _ = model(x)
            mx.eval(probs)
            actions = np.array(probs).argmax(axis=-1).astype(np.int32)

            # Policy panel reflects the obs that produced this action
            if policy_panel:
                policy_panel.update(model.activations(obs))

            obs, rewards, dones = env.step(actions)
            t += ctrl["step_interval"]
            ep_step += 1

            if rewards[0] == 1.0:
                steps_since_food = 0
            else:
                steps_since_food += 1

            crashed = not env.alive[0]            # hit wall or itself
            looping = steps_since_food >= loop_threshold
            timed_out = ep_step >= max_steps

            if crashed:
                # Show the death frame (frozen, red) so the failure is visible
                state = env.get_state(0)
                vgrid = model.value_grid(state) if args.heatmap else None
                renderer.show(state, time=t, value_grid=vgrid, dead=True)
                print(f"  [crash] died after {ep_step} steps")

            if crashed or looping or timed_out:
                if looping:
                    print(f"  [loop] no food in {steps_since_food} steps — resetting")
                if args.loop:
                    obs = env.reset()
                    t = 0.0
                    ep_step = 0
                    steps_since_food = 0
                else:
                    running = False
                continue

        # Render
        state = env.get_state(0)
        vgrid = model.value_grid(state) if args.heatmap else None
        renderer.show(state, time=t, value_grid=vgrid)

        # Keys from the pygame window
        for key in renderer.last_keys():
            name = _pg_names.get(key)
            if name:
                handle_key(name)

        if ctrl["quit"] or renderer.should_quit():
            break

        if plotter:
            plotter.refresh()

        # Maintain target fps
        elapsed = time.time() - frame_start
        sleep = ctrl["step_interval"] - elapsed
        if sleep > 0:
            time.sleep(sleep)

    renderer.close()
    if plotter:
        plotter.close()
    if policy_panel:
        policy_panel.close()
    print("Goodbye.")


if __name__ == "__main__":
    main()
