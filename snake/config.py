import json

REQUIRED: dict[str, type] = {
    "grid_size": int,
    "num_envs": int,
    "total_steps": int,
    "steps_per_rollout": int,
    "ppo_epochs": int,
    "mini_batch_size": int,
    "lr": float,
    "gamma": float,
    "gae_lambda": float,
    "clip_eps": float,
    "entropy_coef": float,
    "value_coef": float,
    "max_grad_norm": float,
    "checkpoint_every": int,
}

DEFAULTS: dict = {
    "render_resolution": 800,
    "keep_videos": True,
    "entropy_floor": 0.05,
    "entropy_floor_step_threshold": 10_000_000,
    "shaping_coef": 0.0,   # 0 = off; >0 enables free-space connectivity shaping
    "length_reward_coef": 0.0,  # 0 = off; >0 scales an apple bonus by current fill (rewards LENGTH)
    "step_penalty": 0.0,        # 0 = off; >0 small per-step cost (rewards GROWTH RATE / efficiency)
    "win_bonus": 0.0,           # 0 = off; >0 terminal bonus for filling the whole board (rewards SOLVING)
    # Thermal guard (macOS): pause training when the CPU is throttling (hot)
    "thermal_guard": True,
    "thermal_check_every": 25,   # iterations between thermal checks
    "thermal_cooldown_s": 30,    # seconds to pause when hot, before re-checking
    "thermal_pause_limit": 90,   # pause when CPU_Speed_Limit drops below this (100 = unthrottled)
}


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = json.load(f)

    errors = []
    for key, expected in REQUIRED.items():
        if key not in cfg:
            errors.append(f"missing required field: '{key}'")
        elif expected is int and not isinstance(cfg[key], int):
            errors.append(f"'{key}' must be int, got {type(cfg[key]).__name__}")
        elif expected is float and not isinstance(cfg[key], (int, float)):
            errors.append(f"'{key}' must be numeric, got {type(cfg[key]).__name__}")

    if errors:
        raise ValueError("Config validation failed:\n" + "\n".join(f"  - {e}" for e in errors))

    for key, default in DEFAULTS.items():
        cfg.setdefault(key, default)

    # Normalise int fields that JSON may emit as float
    for key in ("grid_size", "num_envs", "total_steps", "steps_per_rollout",
                "ppo_epochs", "mini_batch_size", "checkpoint_every"):
        cfg[key] = int(cfg[key])

    return cfg
