from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("MacOSX")  # native macOS backend, doesn't block
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

_BG = "#0d0d14"
_PANEL = "#13131f"
_CYAN = "#00ffcc"
_ORANGE = "#ff6b35"
_PURPLE = "#9b72ff"
_GREY = "#666688"
_TEXT = "#ccccdd"


def _style_ax(ax, title: str):
    ax.set_facecolor(_PANEL)
    ax.set_title(title, color=_TEXT, fontsize=9, pad=4)
    ax.tick_params(colors=_GREY, labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor(_PANEL)
    ax.grid(True, color="#22223a", linewidth=0.5, alpha=0.7)
    ax.xaxis.label.set_color(_GREY)
    ax.yaxis.label.set_color(_GREY)


class MetricsPlot:
    """Live-updating training curves read from metrics.jsonl."""

    def __init__(self, run_dir: str, refresh_every: float = 2.0):
        self.metrics_path = Path(run_dir) / "metrics.jsonl"
        self.refresh_every = refresh_every
        self._last_refresh = 0.0

        plt.ion()
        self.fig = plt.figure(figsize=(7, 6), facecolor=_BG)
        self.fig.canvas.manager.set_window_title("Snake PPO — Training Curves")

        gs = gridspec.GridSpec(3, 1, figure=self.fig, hspace=0.55,
                               top=0.93, bottom=0.08, left=0.12, right=0.97)

        self.ax_reward = self.fig.add_subplot(gs[0])
        self.ax_eplen  = self.fig.add_subplot(gs[1])
        self.ax_ent    = self.fig.add_subplot(gs[2])

        _style_ax(self.ax_reward, "Mean Episode Reward")
        _style_ax(self.ax_eplen,  "Mean Episode Length")
        _style_ax(self.ax_ent,    "Policy Entropy")

        self.ax_ent.set_xlabel("Training Steps", color=_GREY, fontsize=8)

        (self.line_reward,) = self.ax_reward.plot([], [], color=_CYAN,  lw=1.4)
        (self.line_eplen,)  = self.ax_eplen.plot([],  [], color=_ORANGE, lw=1.4)
        (self.line_ent,)    = self.ax_ent.plot([],    [], color=_PURPLE, lw=1.4)

        self.fig.suptitle("Training History", color=_TEXT, fontsize=11, y=0.98)
        self.fig.canvas.draw()
        plt.pause(0.01)

    def refresh(self, force: bool = False):
        import time
        now = time.time()
        if not force and (now - self._last_refresh) < self.refresh_every:
            return
        self._last_refresh = now

        records = self._load()
        if not records:
            return

        steps   = np.array([r["step"]            for r in records])
        rewards = np.array([r["mean_reward"]      for r in records])
        eplens  = np.array([r["mean_ep_length"]   for r in records])
        entropy = np.array([r["mean_entropy"]     for r in records])

        # Smooth with rolling mean (window = 5% of data, min 1)
        w = max(1, len(steps) // 20)
        rewards_s = _smooth(rewards, w)
        eplens_s  = _smooth(eplens, w)
        entropy_s = _smooth(entropy, w)

        self.line_reward.set_data(steps, rewards_s)
        self.line_eplen.set_data(steps, eplens_s)
        self.line_ent.set_data(steps, entropy_s)

        for ax, data in ((self.ax_reward, rewards_s),
                         (self.ax_eplen,  eplens_s),
                         (self.ax_ent,    entropy_s)):
            ax.relim()
            ax.autoscale_view()
            # Annotate last value
            ax.set_title(ax.get_title().split("—")[0].rstrip() +
                         f"  —  {data[-1]:.3f}", color=_TEXT, fontsize=9, pad=4)

        # Mark current checkpoint step with a vertical line
        last_step = int(steps[-1])
        for ax in (self.ax_reward, self.ax_eplen, self.ax_ent):
            for line in ax.lines[1:]:  # remove old vlines
                line.remove()
            ax.axvline(last_step, color=_GREY, lw=0.6, ls="--", alpha=0.5)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def _load(self) -> list[dict]:
        if not self.metrics_path.exists():
            return []
        records = []
        try:
            for line in self.metrics_path.read_text().splitlines():
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        except (json.JSONDecodeError, OSError):
            pass
        return records

    def close(self):
        plt.close(self.fig)


def _smooth(arr: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="same")
