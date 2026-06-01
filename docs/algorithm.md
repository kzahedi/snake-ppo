# The Algorithm — PPO

Proximal Policy Optimization, written from scratch in MLX. Two files:
`snake/network.py` (the model) and `snake/ppo.py` (the buffer + update).

## The network — `ActorCritic`

A CNN with a shared trunk and two heads:

```
  input  (B, H, W, 3)          channels: body-age, food, head
    │
    ├─ Conv2d 3→32,  3×3, pad 1  → ReLU
    ├─ Conv2d 32→64, 3×3, pad 1  → ReLU
    ├─ Conv2d 64→64, 3×3, pad 1  → ReLU
    │
    └─ flatten → Linear(64·H·W → 512) → ReLU      (shared trunk)
                         │
              ┌──────────┴──────────┐
         actor head             value head
       Linear(512→3)          Linear(512→1)
          softmax                 scalar
```

- **`same` padding** keeps spatial dimensions, so the value head can produce a
  spatially meaningful estimate (used for the heatmap overlay).
- **Actor** outputs a probability over the 3 relative actions.
- **Value** estimates the expected return from the current state.

All parameters are MLX arrays on the GPU (`mx.default_device()` → `gpu`).

### Key methods

| Method | Used by | Returns |
|--------|---------|---------|
| `__call__(x)` | everywhere | `(probs, value)` |
| `select_action(obs)` | rollout (training) | `(actions, log_probs, values)` — actions **sampled** from `probs` |
| `evaluate(obs, actions)` | PPO update | `(log_probs, values, entropy)` |
| `activations(obs)` | policy panel | dict of conv1/2/3 maps, trunk, probs, value |
| `value_grid(state)` | heatmap | `(H, W)` value of head-at-each-cell |

Note: training **samples** actions (exploration); watch/eval take **argmax**
(greedy), which is usually a bit better than the training reward suggests.

## The rollout buffer — `RolloutBuffer`

Stores one rollout of `T = steps_per_rollout` steps × `N = num_envs`:
`obs, actions, rewards, dones, log_probs, values`.

### Generalized Advantage Estimation (GAE)

`compute_gae(last_values, γ, λ)` walks backwards through the rollout:

```
  δ_t       = r_t + γ · V(s_{t+1}) · (1 − done_t) − V(s_t)
  A_t       = δ_t + γ · λ · (1 − done_t) · A_{t+1}
  return_t  = A_t + V(s_t)
```

- **γ (gamma)** — discount; how far ahead the agent plans. Snake needs a long
  horizon (a trap is set many steps before the death), so `fill.json` uses
  **0.997** rather than the usual 0.99.
- **λ (gae_lambda)** — bias/variance trade-off for the advantage estimate (0.95).

Advantages are normalised to zero mean / unit variance per mini-batch before the
policy loss.

## The PPO update — `PPOTrainer.update`

For **`ppo_epochs`** passes over the rollout, in shuffled mini-batches of
`mini_batch_size`:

```
  ratio        = exp(new_log_prob − old_log_prob)
  unclipped    = ratio · A
  clipped      = clip(ratio, 1−ε, 1+ε) · A
  policy_loss  = −mean(min(unclipped, clipped))
  value_loss   =  mean((V − return)²)
  entropy      =  mean(−Σ p·log p)
  total_loss   = policy_loss + c_v·value_loss − c_e·entropy
```

- **clip ε (clip_eps, 0.2)** — caps how far the policy can move per update; the
  core PPO stability mechanism.
- **c_v (value_coef, 0.5)** — weight on the value-function loss.
- **c_e (entropy_coef, 0.01)** — bonus for keeping the policy uncertain; prevents
  premature convergence. The decaying `H` you see in the progress bar is this
  entropy falling as the policy sharpens.

Old `log_prob`/`value` are computed once before the epochs and held fixed as the
reference point.

### Optimiser & gradient clipping

- **Adam** (`mlx.optimizers.Adam`) at learning rate `lr`.
- Gradients are clipped to global norm `max_grad_norm` (0.5) via
  `clip_grad_norm` before each step.

### Entropy floor (observability)

If mean entropy drops below `entropy_floor` before `entropy_floor_step_threshold`
steps, a warning is logged (it does **not** intervene) — an early-warning that
the policy may be collapsing into a local optimum.

## Reported metrics per update

`policy_loss`, `value_loss`, `mean_entropy`, and `approx_kl` (the approximate KL
divergence between the old and new policy — a health check that updates aren't
too aggressive). These, plus reward/length, are written to `metrics.jsonl`.

## What was tuned for high fill

- **Segment-age observation channel** (see [environment.md](environment.md)) —
  gives the planner the information it needs.
- **γ = 0.997** — long horizon for trap-avoidance.
- **Connectivity reward shaping** — available but currently off; see
  [environment.md](environment.md) and [../IDEAS.md](../IDEAS.md).
