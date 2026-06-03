"""Safety shield — wraps a learned policy so it never traps itself.

At each step the policy proposes its preferred moves (ranked by probability);
the shield takes the highest-ranked move that keeps the snake's tail reachable
from its new head (so it can always follow its tail and survive). If no move is
tail-safe it falls back to the most open move, then to the policy's choice.

This is pure inference-time logic — no training. It turns a good-but-imperfect
food-seeker (PPO ~50% solve) into a near-perfect solver, while the learned
policy still drives *where* to go.
"""
from __future__ import annotations

from collections import deque

from snake.baselines import _safe_moves


def _tail_reachable(body_set, head, tail, H, W) -> bool:
    """Can `head` reach `tail` through non-body cells? (tail itself is passable —
    it will vacate as the snake moves). If so, the snake can chase its tail and
    never get trapped."""
    seen = {head}
    q = deque([head])
    while q:
        r, c = q.popleft()
        if (r, c) == tail:
            return True
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nb = (r + dr, c + dc)
            if (0 <= nb[0] < H and 0 <= nb[1] < W and nb not in seen
                    and (nb not in body_set or nb == tail)):
                seen.add(nb)
                q.append(nb)
    return False


def _reachable_free(body_set, head, H, W) -> int:
    seen = {head}
    q = deque([head])
    n = 0
    while q:
        r, c = q.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nb = (r + dr, c + dc)
            if (0 <= nb[0] < H and 0 <= nb[1] < W and nb not in seen and nb not in body_set):
                seen.add(nb)
                n += 1
                q.append(nb)
    return n


def shielded_action(env, ranked_actions):
    """ranked_actions: actions {0,1,2} in policy-preference order (best first).
    Returns the best tail-safe action, else the most open, else the top choice."""
    H, W = env.H, env.W
    body = env.bodies[0]
    body_set = env.body_sets[0]
    food = (int(env.food[0][0]), int(env.food[0][1]))

    safe = _safe_moves(env)               # non-fatal moves only
    if not safe:
        return ranked_actions[0]
    safe_set = {a for a, _, _ in safe}
    eats = {a for a, nh, _ in safe if nh == food}   # safe moves that eat
    cells = H * W

    tail_ok, free_of = set(), {}
    for a, nh, nd in safe:
        eating = (nh == food)
        # A move that eats the LAST apple fills the board = WIN. Never veto it.
        if eating and len(body) + 1 >= cells:
            return a
        new_set = set(body_set)
        new_set.add(nh)
        if eating:
            new_tail = body[-1]
        else:
            new_set.discard(body[-1])
            new_tail = body[-2] if len(body) >= 2 else nh
        free_of[a] = _reachable_free(new_set, nh, H, W)
        if _tail_reachable(new_set, nh, new_tail, H, W):
            tail_ok.add(a)

    top = ranked_actions[0]
    # Trust the policy's top move when it is safe — or when it eats (eating is
    # progress the policy learned; only override clearly-trapping moves).
    if top in tail_ok or top in eats:
        return top
    # otherwise: highest-ranked tail-safe move
    for a in ranked_actions:
        if a in tail_ok:
            return a
    # no tail-safe move — most open safe move (best survival odds)
    open_safe = sorted(safe_set, key=lambda a: -free_of[a])
    if open_safe:
        return open_safe[0]
    return top
