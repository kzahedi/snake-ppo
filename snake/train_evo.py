"""Neuroevolution training loop. `python -m snake.train_evo --config configs/evo.json`

A (μ, λ)-style evolution strategy: each generation evaluate the population,
keep the elites, and refill by mutating elites. The best individual is saved as
an ActorCritic checkpoint, directly comparable in the eval harness.
"""
from __future__ import annotations

import argparse
import json
import shutil
import signal
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import mlx.core as mx
from tqdm import tqdm

from snake.checkpoint import CheckpointManager
from snake.config import load_config
from snake.evolution import perturb, evaluate_individual
from snake.network import ActorCritic
from snake.thermal import ThermalGuard


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--run-dir", default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    H = W = cfg["grid_size"]
    P = cfg.get("population", 24)
    E = cfg.get("elite", 6)
    sigma = cfg.get("sigma", 0.02)
    init_sigma = cfg.get("init_sigma", 0.5)
    eps = cfg.get("episodes_per_eval", 3)
    gens = cfg.get("generations", 300)
    win_bonus = cfg.get("win_bonus", 5.0)

    run_dir = Path(args.run_dir) if args.run_dir else Path("runs") / datetime.now().strftime("evo_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    shutil.copy(args.config, run_dir / "config.json")
    metrics_file = open(run_dir / "metrics.jsonl", "a")

    model = ActorCritic(H, W)               # hosts each individual's weights during eval
    ckpt = CheckpointManager(run_dir)
    base = model.parameters()
    population = [perturb(base, init_sigma) for _ in range(P)]
    for pa in population:
        mx.eval(pa)

    thermal = ThermalGuard(enabled=cfg.get("thermal_guard", True),
                           check_every=cfg.get("thermal_check_every", 5),
                           cooldown_s=cfg.get("thermal_cooldown_s", 30),
                           pause_limit=cfg.get("thermal_pause_limit", 90))
    _stop = [False]
    signal.signal(signal.SIGINT, lambda s, f: _stop.__setitem__(0, True))

    best_fit = -1e9
    best_fill = 0.0
    cells = H * W
    t_start = time.time()
    print(f"Neuroevolution  grid={H}×{W}  pop={P}  elite={E}  σ={sigma}  gens={gens}  run={run_dir}")
    pbar = tqdm(total=gens, unit="gen", dynamic_ncols=True)

    for g in range(gens):
        results = [evaluate_individual(model, pa, H, W, eps, win_bonus, seed=g)
                   for pa in population]
        fits = np.array([r[0] for r in results])
        order = np.argsort(-fits)
        elites = [population[i] for i in order[:E]]
        gen_best = results[order[0]]
        mean_fill = float(np.mean([r[1] for r in results])) / cells
        best_fill = gen_best[1] / cells
        win_rate = gen_best[2]

        if gen_best[0] > best_fit:
            best_fit = gen_best[0]
            best_fill = best_fill if gen_best[1] / cells < best_fill else gen_best[1] / cells
            model.update(elites[0])
            ckpt.save((g + 1) * P * eps, model, None,
                      {"mean_reward": gen_best[1] - 4, "mean_ep_length": gen_best[1]})

        metrics_file.write(json.dumps({
            "step": (g + 1) * P * eps, "generation": g + 1,
            "wall_time": time.time() - t_start,
            "mean_reward": best_fill * cells - 4, "mean_ep_length": gen_best[1],
            "mean_entropy": 0.0, "policy_loss": 0.0, "value_loss": 0.0,
            "approx_kl": 0.0, "win_rate": win_rate,
            "best_fill": best_fill, "pop_mean_fill": mean_fill,
        }) + "\n")
        metrics_file.flush()

        pbar.update(1)
        pbar.set_postfix_str(f"best fill {100*best_fill:.0f}%  pop-mean {100*mean_fill:.0f}%  "
                             f"win {100*win_rate:.0f}%  best-fit {best_fit:.1f}")

        # next generation: elites + mutated elites
        nextpop = list(elites)
        while len(nextpop) < P:
            parent = elites[np.random.randint(E)]
            nextpop.append(perturb(parent, sigma))
        for pa in nextpop:
            mx.eval(pa)
        population = nextpop

        thermal.check(g, pbar.write)
        if _stop[0]:
            pbar.write(f"interrupted at generation {g+1}")
            break

    pbar.close(); metrics_file.close()
    print(f"done. best board fill {100*best_fill:.0f}%  (best fitness {best_fit:.1f})  run {run_dir}")


if __name__ == "__main__":
    main()
