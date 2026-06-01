from __future__ import annotations

import re
import time
from pathlib import Path

import imageio
import numpy as np

from snake.env import VectorizedSnakeEnv
from snake.renderer import SnakeRenderer


class VideoExporter:
    def __init__(self, grid_size: int, resolution: int = 800,
                 fps: int = 30, max_steps: int = 2000):
        self.grid_size = grid_size
        self.resolution = resolution
        self.fps = fps
        self.max_steps = max_steps

    def export_checkpoint(self, model, step: int, run_dir, keep_videos: bool = True):
        """Render one greedy episode (clean + heatmap variants) at this checkpoint."""
        videos_dir = Path(run_dir) / "videos"
        videos_dir.mkdir(parents=True, exist_ok=True)

        clean_path = videos_dir / f"step_{step:010d}.mp4"
        heat_path = videos_dir / f"step_{step:010d}_heatmap.mp4"

        self._render_episode(model, str(clean_path), heatmap=False)
        self._render_episode(model, str(heat_path), heatmap=True)

        return str(clean_path), str(heat_path)

    def export_preview(self, model, run_dir):
        """Render one greedy episode of the CURRENT policy to a single rolling
        preview.mp4 (overwritten each call) — a lightweight live progress view.
        Returns the episode summary dict (reason, length, apples, steps)."""
        out_path = Path(run_dir) / "preview.mp4"
        info = self._render_episode(model, str(out_path), heatmap=False)
        info["path"] = str(out_path)
        return info

    def assemble_timelapse(self, run_dir, fps: int = 24, keep_videos: bool = True):
        """Stitch first frame of each checkpoint video into a timelapse."""
        videos_dir = Path(run_dir) / "videos"
        out_path = Path(run_dir) / "timelapse.mp4"

        # Find all clean checkpoint videos in step order
        pattern = re.compile(r"step_(\d{10})\.mp4$")
        checkpoint_vids = sorted(
            (p for p in videos_dir.glob("step_*.mp4")
             if not p.name.endswith("_heatmap.mp4")),
            key=lambda p: int(pattern.search(p.name).group(1))
        )

        if not checkpoint_vids:
            return None

        frames = []
        for vid_path in checkpoint_vids:
            reader = imageio.get_reader(str(vid_path))
            frame = reader.get_data(0)
            frames.append(frame)
            reader.close()

        writer = imageio.get_writer(str(out_path), fps=fps,
                                    codec="libx264", quality=8,
                                    macro_block_size=1,
                                    output_params=["-crf", "18"])
        for frame in frames:
            writer.append_data(frame)
        writer.close()

        if not keep_videos:
            for p in checkpoint_vids:
                p.unlink(missing_ok=True)
            for p in videos_dir.glob("step_*_heatmap.mp4"):
                p.unlink(missing_ok=True)

        return str(out_path)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _render_episode(self, model, out_path: str, heatmap: bool = False) -> dict:
        """Render one greedy episode. Plays until the snake dies (collision) or
        stalls — defined as going `2 × cells` steps without eating, long enough
        to let it solve the board but short enough to cut off endless loops.

        Returns {"reason", "length", "apples", "steps"}.
        """
        import mlx.core as mx
        H = W = self.grid_size
        env = VectorizedSnakeEnv(H, W, N=1, auto_reset=False)
        renderer = SnakeRenderer(H, W, resolution=self.resolution, mode="offscreen")
        obs = env.observation()
        t = 0.0

        loop_threshold = 2 * H * W       # steps allowed between apples
        steps_since_food = 0
        apples = 0
        reason = "cap"
        step_i = 0

        writer = imageio.get_writer(out_path, fps=self.fps,
                                    codec="libx264", quality=8,
                                    macro_block_size=1,
                                    output_params=["-crf", "18"])

        for step_i in range(self.max_steps):
            x = mx.array(obs)
            probs, _ = model(x)
            mx.eval(probs)
            actions = np.array(probs).argmax(axis=-1).astype(np.int32)

            state = env.get_state(0)
            vgrid = model.value_grid(state) if heatmap else None
            writer.append_data(renderer.render_frame(state, time=t, value_grid=vgrid))

            obs, rewards, dones = env.step(actions)
            t += 1.0 / self.fps

            if rewards[0] == 1.0:
                apples += 1
                steps_since_food = 0
            else:
                steps_since_food += 1

            if not env.alive[0]:                      # collision death
                reason = env.death_cause[0] or "crash"
                dstate = env.get_state(0)
                dgrid = model.value_grid(dstate) if heatmap else None
                for _ in range(max(1, self.fps // 3)):  # hold the red death frame
                    writer.append_data(renderer.render_frame(dstate, time=t,
                                                             value_grid=dgrid, dead=True))
                break

            if steps_since_food >= loop_threshold:    # stalled / looping
                reason = "stalled"
                break
        else:
            reason = "cap"

        writer.close()
        renderer.close()
        return {"reason": reason, "length": len(env.bodies[0]),
                "apples": apples, "steps": step_i + 1}
