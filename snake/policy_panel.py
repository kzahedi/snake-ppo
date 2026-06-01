from __future__ import annotations

import numpy as np

import matplotlib
matplotlib.use("MacOSX")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

_BG = "#0d0d14"
_PANEL = "#13131f"
_TEXT = "#ccccdd"
_GREY = "#666688"
_CYAN = "#00ffcc"
_ORANGE = "#ff6b35"
_PURPLE = "#9b72ff"

_ACTIONS = ["LEFT", "STRAIGHT", "RIGHT"]


def _montage(act_hwc: np.ndarray, n: int) -> np.ndarray:
    """Tile the first n channels of an (H, W, C) activation into a square grid."""
    H, W, C = act_hwc.shape
    n = min(n, C)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    canvas = np.zeros((rows * H, cols * W), dtype=np.float32)
    for k in range(n):
        r, c = divmod(k, cols)
        tile = act_hwc[:, :, k]
        canvas[r*H:(r+1)*H, c*W:(c+1)*W] = tile
    return canvas


def _style(ax, title):
    ax.set_facecolor(_PANEL)
    ax.set_title(title, color=_TEXT, fontsize=8, pad=3)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor("#2a2a44")


class PolicyPanel:
    """Live visualisation of the policy network driving the snake."""

    def __init__(self, H: int, W: int, channels_per_layer: int = 16):
        self.H = H
        self.W = W
        self.k = channels_per_layer

        plt.ion()
        self.fig = plt.figure(figsize=(7.5, 8.5), facecolor=_BG)
        self.fig.canvas.manager.set_window_title("Snake PPO — Policy Inside the Network")
        gs = gridspec.GridSpec(3, 2, figure=self.fig, hspace=0.4, wspace=0.25,
                               top=0.92, bottom=0.05, left=0.08, right=0.96,
                               height_ratios=[0.8, 1.1, 1.1])

        # Row 0: action probabilities (full width)
        self.ax_probs = self.fig.add_subplot(gs[0, :])
        self._style_probs()

        # Row 1: conv1 | conv2
        self.ax_c1 = self.fig.add_subplot(gs[1, 0])
        self.ax_c2 = self.fig.add_subplot(gs[1, 1])
        _style(self.ax_c1, f"conv1 — first {self.k} of 32 filters")
        _style(self.ax_c2, f"conv2 — first {self.k} of 64 filters")

        # Row 2: conv3 | network diagram
        self.ax_c3 = self.fig.add_subplot(gs[2, 0])
        self.ax_net = self.fig.add_subplot(gs[2, 1])
        _style(self.ax_c3, f"conv3 — first {self.k} of 64 filters")
        _style(self.ax_net, "network — activation flow")

        # imshow handles (created on first update)
        self._im1 = self._im2 = self._im3 = None
        self._net_boxes = None

        self.fig.suptitle("What's driving the snake", color=_TEXT, fontsize=12, y=0.97)
        self.fig.canvas.draw()
        plt.pause(0.01)

    def _style_probs(self):
        ax = self.ax_probs
        ax.set_facecolor(_PANEL)
        ax.set_title("Action probabilities (policy head)", color=_TEXT, fontsize=9, pad=4)
        self._bars = ax.barh(_ACTIONS, [0.33, 0.33, 0.33],
                             color=[_PURPLE, _CYAN, _ORANGE])
        ax.set_xlim(0, 1)
        ax.tick_params(colors=_GREY, labelsize=8)
        for s in ax.spines.values():
            s.set_edgecolor(_PANEL)
        self._prob_texts = [
            ax.text(0.02, i, "", va="center", color=_BG, fontsize=8, fontweight="bold")
            for i in range(3)
        ]

    def _build_net(self, act):
        """Draw the architecture once; store boxes for recolouring."""
        ax = self.ax_net
        ax.set_xlim(0, 10); ax.set_ylim(0, 10)
        layers = [
            ("input", 0.5, 5, 1.2),
            ("conv1", 2.2, 5, 1.2),
            ("conv2", 3.9, 5, 1.2),
            ("conv3", 5.6, 5, 1.2),
            ("trunk", 7.3, 5, 1.6),
        ]
        self._net_boxes = {}
        for name, x, y, h in layers:
            box = FancyBboxPatch((x, y - h/2), 1.0, h,
                                 boxstyle="round,pad=0.05",
                                 linewidth=1, edgecolor="#3a3a5a",
                                 facecolor=_PANEL, mutation_scale=4)
            ax.add_patch(box)
            ax.text(x + 0.5, y - h/2 - 0.5, name, ha="center",
                    color=_GREY, fontsize=7)
            self._net_boxes[name] = box
        # heads
        for name, y in (("actor", 6.8), ("value", 3.2)):
            box = FancyBboxPatch((9.0, y - 0.5), 0.9, 1.0,
                                 boxstyle="round,pad=0.05",
                                 linewidth=1, edgecolor="#3a3a5a",
                                 facecolor=_PANEL, mutation_scale=4)
            ax.add_patch(box)
            ax.text(9.45, y - 1.0, name, ha="center", color=_GREY, fontsize=7)
            self._net_boxes[name] = box
        # arrows
        xs = [1.5, 3.2, 4.9, 6.6, 8.3]
        for x in xs[:-1]:
            ax.add_patch(FancyArrowPatch((x, 5), (x + 0.7, 5),
                         arrowstyle="->", color="#3a3a5a", mutation_scale=8))
        ax.add_patch(FancyArrowPatch((8.3, 5), (9.0, 6.8),
                     arrowstyle="->", color="#3a3a5a", mutation_scale=8))
        ax.add_patch(FancyArrowPatch((8.3, 5), (9.0, 3.2),
                     arrowstyle="->", color="#3a3a5a", mutation_scale=8))

    def update(self, act: dict):
        # --- action probabilities ---
        probs = act["probs"]
        chosen = int(np.argmax(probs))
        for i, (bar, p, txt) in enumerate(zip(self._bars, probs, self._prob_texts)):
            bar.set_width(float(p))
            bar.set_alpha(1.0 if i == chosen else 0.45)
            label = f"{p:.2f}" + ("  ◀ chosen" if i == chosen else "")
            txt.set_text(label)
        self.ax_probs.set_title(
            f"Action probabilities (policy head)   V(s) = {act['value']:.2f}",
            color=_TEXT, fontsize=9, pad=4)

        # --- conv feature maps ---
        for ax, im_attr, key in ((self.ax_c1, "_im1", "conv1"),
                                 (self.ax_c2, "_im2", "conv2"),
                                 (self.ax_c3, "_im3", "conv3")):
            montage = _montage(act[key], self.k)
            im = getattr(self, im_attr)
            if im is None:
                im = ax.imshow(montage, cmap="magma", interpolation="nearest")
                setattr(self, im_attr, im)
            else:
                im.set_data(montage)
                im.set_clim(montage.min(), max(montage.max(), 1e-6))

        # --- network diagram ---
        if self._net_boxes is None:
            self._build_net(act)
        intensities = {
            "input": 0.6,
            "conv1": float(np.mean(act["conv1"])),
            "conv2": float(np.mean(act["conv2"])),
            "conv3": float(np.mean(act["conv3"])),
            "trunk": float(np.mean(act["trunk"])),
            "actor": float(np.max(probs)),
            "value": float(min(abs(act["value"]) / 5.0, 1.0)),
        }
        # normalise conv/trunk intensities to [0,1] with a soft scale
        for name, box in self._net_boxes.items():
            v = intensities[name]
            v = float(np.clip(v / (0.5 if name in ("conv1","conv2","conv3","trunk") else 1.0), 0, 1))
            # interpolate panel -> cyan by activation
            col = (0.0 + v*0.0, 0.07 + v*0.93, 0.10 + v*0.70)
            box.set_facecolor(col)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def close(self):
        plt.close(self.fig)
