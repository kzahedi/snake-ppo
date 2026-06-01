from __future__ import annotations

import json
import time
from pathlib import Path
import mlx.core as mx


class CheckpointNotFoundError(Exception):
    pass


class CheckpointManager:
    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self._start_time = time.time()

    def save(self, step: int, model, optimizer, metrics: dict):
        d = self.ckpt_dir / f"step_{step:010d}"
        d.mkdir(exist_ok=True)

        model.save_weights(str(d / "weights.npz"))

        meta = {
            "step": step,
            "wall_time_seconds": time.time() - self._start_time,
            "mean_reward": float(metrics.get("mean_reward", 0.0)),
            "mean_ep_length": float(metrics.get("mean_ep_length", 0.0)),
            "mean_entropy": float(metrics.get("mean_entropy", 0.0)),
        }
        (d / "meta.json").write_text(json.dumps(meta, indent=2))

    def load(self, step_or_latest):
        if step_or_latest == "latest":
            steps = self._available_steps()
            if not steps:
                raise CheckpointNotFoundError(f"No checkpoints in {self.ckpt_dir}")
            step = steps[-1]
        else:
            step = int(step_or_latest)

        d = self.ckpt_dir / f"step_{step:010d}"
        if not d.exists():
            raise CheckpointNotFoundError(f"Checkpoint step {step} not found at {d}")
        return step, str(d / "weights.npz")

    def load_weights_into(self, model, step_or_latest) -> int:
        step, weights_path = self.load(step_or_latest)
        model.load_weights(weights_path)
        return step

    def list(self) -> list:
        result = []
        for step in self._available_steps():
            d = self.ckpt_dir / f"step_{step:010d}"
            meta_path = d / "meta.json"
            meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
            meta["step"] = step
            result.append(meta)
        return result

    def _available_steps(self) -> list:
        steps = []
        for p in self.ckpt_dir.iterdir():
            if p.is_dir() and p.name.startswith("step_"):
                try:
                    steps.append(int(p.name.split("_")[1]))
                except (IndexError, ValueError):
                    pass
        return sorted(steps)
