# The Snake Environment

`snake/env.py` — `VectorizedSnakeEnv`. N independent Snake games stepped in
lockstep with NumPy.

```python
env = VectorizedSnakeEnv(H, W, N, auto_reset=True)
obs = env.observation()              # (N, H, W, 3) float32
obs, rewards, dones = env.step(actions)   # actions: (N,) int in {0,1,2}
```

## Grid and snake state

Each of the N games holds:
- a **body** — a `deque` of `(row, col)` cells, head at index 0, tail at the end
- a **body set** — the same cells as a `set` for O(1) collision checks
- a **direction** — `0=UP, 1=RIGHT, 2=DOWN, 3=LEFT`
- a **food** cell `(row, col)`
- an **alive** flag (used in `auto_reset=False` mode)

Snakes start at length 3, at a random position ≥2 cells from the edges, facing
a random direction, with food placed on a random free cell.

## Observation — `(H, W, 3)` float32

Three stacked binary/scalar maps fed to the CNN:

| Channel | Name | Values |
|---------|------|--------|
| 0 | **body-age** | `(n − j) / n` for the segment at index `j` (head `j=0`) — head = 1.0, tail ≈ 1/len |
| 1 | **food** | 1.0 at the apple cell, 0 elsewhere |
| 2 | **head** | 1.0 at the head cell only |

### Why body-age, not a binary mask

The age channel encodes **time-until-each-cell-is-vacated**: the head stays
longest (1.0), the tail vacates next step (≈1/len). This is the single most
important signal for safe late-game play — it lets the policy reason about
"that cell frees up in 3 steps, so I can route through it" without which
near-Hamiltonian play is impossible. A flat binary body mask hides this.

```
  binary mask            age gradient (head=1.0, tail≈0)
  ■ ■ ■ ■                .9 .8 .7 .6
  ■ · · ■        →       1.0  ·  · .5
  ■ · · ·                 ·   ·  · .4
```

## Actions — 3 relative moves

| Action | Meaning |
|--------|---------|
| 0 | turn left (counter-clockwise) |
| 1 | go straight |
| 2 | turn right (clockwise) |

New direction = `(current_direction + offset) mod 4`, with offsets
`[-1, 0, +1]`. Because moves are **relative**, a 180° U-turn is physically
impossible — the snake can never instantly reverse into itself. This removes a
large class of trivial early deaths and yields smoother trajectories.

## Reward

| Event | Reward |
|-------|--------|
| Eat an apple | **+1.0** (snake grows by 1, new apple spawns on a random free cell) |
| Die (wall or self) | **−1.0** |
| Any other step | **0.0** |

The **episode return** (what the `R` metric reports) is the sum over a life:
```
  return = apples − 1
  body length at death = 3 + apples = R + 4
  fill % (8×8) = (R + 4) / 64
```

## Collision rules

- **Wall**: moving off the grid is fatal.
- **Self**: moving into a body cell is fatal — *except* moving into the current
  **tail** cell when **not eating**, because the tail vacates that cell on the
  same step (the classic "chase your tail" survival move). Moving into the tail
  while eating *is* fatal (the tail stays, since the snake grows).

```
  safe (tail vacates)        fatal (mid-body)        fatal (tail, but eating)
  · → T                      · → ■                   · → T  (+apple, tail stays)
  H ■ ■                      H ■ ■                    H ■ ■
```

## Reset behaviour — `auto_reset`

- `auto_reset=True` (default, **training**): a game that dies is reset
  immediately *inside* `step()`, so the returned `obs` is already a fresh game.
  Essential for keeping all N envs stepping in lockstep.
- `auto_reset=False` (**watch mode**): a dead game freezes on its death frame
  (`alive[i] = False`) until the caller calls `reset_dead()`. This is what lets
  the watch window render the crash (red flash) instead of silently teleporting.

## Connectivity reward shaping (optional, off by default)

When `compute_shaping` is enabled (config `shaping_coef > 0`), each step the env
computes **Φ = free cells reachable from the head / total free cells** via a
flood fill, and stores `last_shaping[i] = Φ_after − Φ_before`:

```
  Φ = 1.0  → all free space is one region the snake can still reach (safe)
  Φ < 1.0  → the snake has sealed off free cells it can never get back to
```

The training loop adds `shaping_coef × last_shaping` to the reward used for
learning, while the **raw** reward still drives the apples-eaten metric.

**Currently disabled** (`shaping_coef: 0.0` in `fill.json`) because the
flood-fill roughly halved throughput, did nothing until the snake was long
enough to fragment the board, and was unvalidated. The machinery remains in
place (with cached Φ across steps, so it's computed once per step not twice) for
if the segment-age channel alone plateaus below the fill target. See
[../IDEAS.md](../IDEAS.md).

## Rendering hooks

`get_state(i)` returns `{"body": [...], "food": (r, c)}` for a single env, which
the renderer and video exporter consume.
