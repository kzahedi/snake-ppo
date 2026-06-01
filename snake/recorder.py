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

    def _render_episode(self, model, out_path: str, heatmap: bool = False):
        H = W = self.grid_size
        env = VectorizedSnakeEnv(H, W, N=1)
        renderer = SnakeRenderer(H, W, resolution=self.resolution, mode="offscreen")
        obs = env.observation()
        t = 0.0

        writer = imageio.get_writer(out_path, fps=self.fps,
                                    codec="libx264", quality=8,
                                    output_params=["-crf", "18"])

        for step_i in range(self.max_steps):
            import mlx.core as mx
            x = mx.array(obs)
            probs, _ = model(x)
            mx.eval(probs)
            actions = np.array(probs).argmax(axis=-1).astype(np.int32)

            state = env.get_state(0)
            vgrid = model.value_grid(state) if heatmap else None
            frame = renderer.render_frame(state, time=t, value_grid=vgrid)
            writer.append_data(frame)

            obs, rewards, dones = env.step(actions)
            t += 1.0 / self.fps
            if dones[0]:
                break

        writer.close()
        renderer.close()
