"""Compare approaches visually: a performance bar chart + a side-by-side
behaviour video of each agent playing.

  python -m snake.compare --grid 8 --episodes 40 --ppo runs/solve
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from snake.env import VectorizedSnakeEnv
from snake.eval import (evaluate, _NetAgent, _QNetAgent, _ShieldedNetAgent,
                        _make_baseline, _split_label)


# ---------------------------------------------------------------------------
# Episode frame capture (any agent)
# ---------------------------------------------------------------------------

def capture_episode(agent, H, W, resolution=280, max_steps=None, seed=0):
    """Run one greedy episode, returning (frames list, info dict)."""
    from snake.renderer import SnakeRenderer
    if max_steps is None:
        max_steps = H * W * H * W
    np.random.seed(seed)
    env = VectorizedSnakeEnv(H, W, 1, auto_reset=False)
    rend = SnakeRenderer(H, W, resolution=resolution, mode="offscreen")
    obs = env.observation()
    frames = []
    stall = 2 * H * W
    ssf = 0
    reason = "cap"
    for _ in range(max_steps):
        frames.append(rend.render_frame(env.get_state(0)))
        a = agent.act(env)
        obs, r, d = env.step(np.array([a], dtype=np.int32))
        ssf = 0 if r[0] == 1.0 else ssf + 1
        if not env.alive[0]:
            reason = env.death_cause[0] or "crash"
            dead = env.get_state(0)
            for _ in range(8):  # hold the final frame briefly
                frames.append(rend.render_frame(dead, dead=(reason != "win")))
            break
        if ssf >= stall:
            reason = "stalled"
            break
    rend.close()
    return frames, {"reason": reason, "length": len(env.bodies[0]), "frames": len(frames)}


# ---------------------------------------------------------------------------
# Bar chart
# ---------------------------------------------------------------------------

def bar_chart(rows, H, W, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [n for n, _ in rows]
    solve = [100 * m["solve_rate"] for _, m in rows]
    fill = [100 * m["mean_fill"] for _, m in rows]
    x = np.arange(len(names))
    w = 0.38
    fig, ax = plt.subplots(figsize=(1.6 * len(names) + 2, 5), facecolor="#0d0d14")
    ax.set_facecolor("#13131f")
    b1 = ax.bar(x - w/2, fill, w, label="mean fill %", color="#00ffcc")
    b2 = ax.bar(x + w/2, solve, w, label="solve rate %", color="#ff6b35")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+1, f"{b.get_height():.0f}",
                    ha="center", va="bottom", color="#ccccdd", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(names, color="#ccccdd", fontsize=9)
    ax.set_ylim(0, 105); ax.set_ylabel("%", color="#ccccdd")
    ax.set_title(f"Snake approaches on {H}×{W}", color="#ccccdd", fontsize=12)
    ax.tick_params(colors="#666688")
    for s in ax.spines.values():
        s.set_edgecolor("#2a2a44")
    ax.legend(facecolor="#13131f", edgecolor="#2a2a44", labelcolor="#ccccdd")
    ax.grid(axis="y", color="#22223a", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, facecolor="#0d0d14")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Side-by-side behaviour video
# ---------------------------------------------------------------------------

def behaviour_video(agents, H, W, out_path, cell=260, fps=20, seed=0):
    import imageio
    from PIL import Image, ImageDraw
    clips, infos = [], []
    for ag in agents:
        f, info = capture_episode(ag, H, W, resolution=cell, seed=seed)
        clips.append(f); infos.append(info)
    n = len(agents)
    cols = 2 if n <= 4 else 3
    rows_n = int(np.ceil(n / cols))
    maxlen = max(len(c) for c in clips)
    bar_h = 26

    def labeled(frame, text):
        img = Image.fromarray(frame)
        out = Image.new("RGB", (cell, cell + bar_h), (18, 18, 30))
        d = ImageDraw.Draw(out)
        d.text((8, 6), text, fill=(220, 220, 235))
        out.paste(img, (0, bar_h))
        return np.asarray(out)

    labels = [f"{ag.name}" for ag in agents]
    writer = imageio.get_writer(out_path, fps=fps, codec="libx264",
                                quality=8, macro_block_size=1,
                                output_params=["-crf", "20"])
    for t in range(maxlen):
        cells = []
        for i in range(n):
            fr = clips[i][min(t, len(clips[i]) - 1)]   # freeze finished clips
            tag = labels[i]
            if t >= len(clips[i]) - 1:
                tag += f"  [{infos[i]['reason']} · {infos[i]['length']}/{H*W}]"
            cells.append(labeled(fr, tag))
        # pad to a full grid
        while len(cells) < cols * rows_n:
            cells.append(np.zeros_like(cells[0]))
        grid = np.vstack([np.hstack(cells[r*cols:(r+1)*cols]) for r in range(rows_n)])
        writer.append_data(grid)
    writer.close()
    return infos


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--grid", type=int, default=8)
    p.add_argument("--episodes", type=int, default=40)
    p.add_argument("--ppo", action="append", default=[])
    p.add_argument("--dqn", action="append", default=[])
    p.add_argument("--shielded", action="append", default=[])
    p.add_argument("--baselines", default="hamiltonian,greedy-astar,flood-fill")
    p.add_argument("--out-dir", default="assets")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-video", action="store_true")
    args = p.parse_args()
    H = W = args.grid
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    agents = []
    for rd in args.ppo:
        path, label = _split_label(rd)
        agents.append(_NetAgent(path, H, W, label=label))
    for rd in args.dqn:
        path, label = _split_label(rd)
        agents.append(_QNetAgent(path, H, W, label=label))
    for rd in args.shielded:
        path, label = _split_label(rd)
        agents.append(_ShieldedNetAgent(path, H, W, label=label))
    agents += [_make_baseline(b, H, W) for b in args.baselines.split(",") if b]

    print(f"Evaluating {len(agents)} agents ({args.episodes} eps each)…")
    rows = []
    for ag in agents:
        m = evaluate(ag, H, W, episodes=args.episodes, seed=args.seed)
        rows.append((ag.name, m))
        print(f"  {ag.name:<22} solve {100*m['solve_rate']:>3.0f}%  fill {100*m['mean_fill']:>3.0f}%")

    chart = f"{args.out_dir}/comparison.png"
    bar_chart(rows, H, W, chart)
    print(f"chart → {chart}")

    if not args.no_video:
        vid = f"{args.out_dir}/behaviours.mp4"
        infos = behaviour_video(agents, H, W, vid, seed=args.seed)
        print(f"video → {vid}")
        for ag, info in zip(agents, infos):
            print(f"  {ag.name:<22} {info['reason']} at {info['length']}/{H*W}")


if __name__ == "__main__":
    main()
