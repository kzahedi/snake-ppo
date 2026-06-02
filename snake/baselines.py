"""Non-learned reference agents for comparison against the RL methods.

Each agent exposes `act(env) -> int`, returning a RELATIVE action {0,1,2}
(left/straight/right) for env index 0, matching VectorizedSnakeEnv's action
space. These establish reference lines: the Hamiltonian cycle is near-optimal
(fills the board), greedy-A* is fast but self-traps, flood-fill survives longer.
"""
from __future__ import annotations

from collections import deque

# Directions: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT  (matches env._DR/_DC)
_DR = [-1, 0, 1, 0]
_DC = [0, 1, 0, -1]


def _dir_from_delta(dr, dc):
    for d in range(4):
        if _DR[d] == dr and _DC[d] == dc:
            return d
    return None


def _to_relative(cur_dir, target_dir):
    """Relative action to turn from cur_dir to target_dir (U-turn → straight)."""
    diff = (target_dir - cur_dir) % 4
    return {0: 1, 1: 2, 3: 0}.get(diff, 1)


def _safe_moves(env):
    """Candidate (action, new_head, new_dir) that don't immediately collide."""
    H, W = env.H, env.W
    body = env.bodies[0]
    body_set = env.body_sets[0]
    head = body[0]
    tail = body[-1]
    food = (int(env.food[0][0]), int(env.food[0][1]))
    cur = int(env.dirs[0])
    out = []
    for action in (0, 1, 2):
        nd = (cur + (action - 1)) % 4
        nr, nc = head[0] + _DR[nd], head[1] + _DC[nd]
        if nr < 0 or nr >= H or nc < 0 or nc >= W:
            continue
        nh = (nr, nc)
        eating = (nh == food)
        if nh in body_set and not (nh == tail and not eating):
            continue
        out.append((action, nh, nd))
    return out


def _flood_free(body_set, start, H, W):
    """Count free cells reachable from `start` (4-connectivity)."""
    seen = {start}
    q = deque([start])
    n = 0
    while q:
        r, c = q.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in seen and (nr, nc) not in body_set:
                seen.add((nr, nc))
                n += 1
                q.append((nr, nc))
    return n


def _bfs_path_dir(env, start, goal):
    """First-step direction of a shortest path start→goal over free cells, or None."""
    H, W = env.H, env.W
    body_set = env.body_sets[0]
    if start == goal:
        return None
    prev = {start: None}
    q = deque([start])
    while q:
        cell = q.popleft()
        if cell == goal:
            break
        r, c = cell
        for d in range(4):
            nr, nc = r + _DR[d], c + _DC[d]
            nb = (nr, nc)
            if 0 <= nr < H and 0 <= nc < W and nb not in prev and \
                    (nb not in body_set or nb == goal):
                prev[nb] = cell
                q.append(nb)
    if goal not in prev:
        return None
    # walk back to the first step from start
    cur = goal
    while prev[cur] != start:
        cur = prev[cur]
        if cur is None:
            return None
    return _dir_from_delta(cur[0] - start[0], cur[1] - start[1])


class HamiltonianAgent:
    """Follows a fixed Hamiltonian cycle — visits every cell, never self-traps,
    fills the board. Near-optimal reference (requires an even dimension)."""
    name = "hamiltonian"

    def __init__(self, H, W):
        self.H, self.W = H, W
        self.order = self._cycle(H, W)
        self.index = {cell: i for i, cell in enumerate(self.order)}

    @staticmethod
    def _cycle(H, W):
        assert W % 2 == 0 or H % 2 == 0, "Hamiltonian cycle needs an even dimension"
        cells = [(0, 0)]
        for c in range(1, W):          # row 0, left → right
            cells.append((0, c))
        going_down = True              # boustrophedon over cols W-1..1, rows 1..H-1
        for c in range(W - 1, 0, -1):
            rng = range(1, H) if going_down else range(H - 1, 0, -1)
            for r in rng:
                cells.append((r, c))
            going_down = not going_down
        for r in range(H - 1, 0, -1):  # return up column 0
            cells.append((r, 0))
        return cells

    def act(self, env):
        head = env.bodies[0][0]
        nxt = self.order[(self.index[head] + 1) % len(self.order)]
        td = _dir_from_delta(nxt[0] - head[0], nxt[1] - head[1])
        return _to_relative(int(env.dirs[0]), td)


class GreedyAStarAgent:
    """Beelines to the food via shortest path; falls back to any safe move.
    Fast scorer early, but traps itself as the board fills."""
    name = "greedy-astar"

    def __init__(self, H, W):
        self.H, self.W = H, W

    def act(self, env):
        head = env.bodies[0][0]
        food = (int(env.food[0][0]), int(env.food[0][1]))
        cur = int(env.dirs[0])
        td = _bfs_path_dir(env, head, food)
        if td is not None:
            diff = (td - cur) % 4
            if diff != 2:               # not a U-turn
                return _to_relative(cur, td)
        safe = _safe_moves(env)
        return safe[0][0] if safe else 1


class FloodFillAgent:
    """Among non-fatal moves, picks the one keeping the most reachable free
    space (tie-break toward the food). Survives much longer than greedy."""
    name = "flood-fill"

    def __init__(self, H, W):
        self.H, self.W = H, W

    def act(self, env):
        H, W = self.H, self.W
        body = env.bodies[0]
        length = len(body)
        food = (int(env.food[0][0]), int(env.food[0][1]))
        safe = _safe_moves(env)
        if not safe:
            return 1
        scored = []
        for action, nh, nd in safe:
            # simulate the body after this move (tail vacates unless eating)
            new_set = set(env.body_sets[0])
            new_set.add(nh)
            if nh != food:
                new_set.discard(body[-1])
            free = _flood_free(new_set, nh, H, W)
            dist = abs(nh[0] - food[0]) + abs(nh[1] - food[1])
            scored.append((action, free, dist))
        # Seek the food, but only via moves that keep enough room not to trap
        # the body (reachable free space >= remaining length); else play safest.
        viable = [s for s in scored if s[1] >= length] or scored
        best = min(viable, key=lambda s: (s[2], -s[1]))   # nearest food, then most space
        return best[0]
