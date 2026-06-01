from collections import deque
import numpy as np

# Directions: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
_DR = np.array([-1, 0, 1, 0], dtype=np.int32)
_DC = np.array([0, 1, 0, -1], dtype=np.int32)

# Relative action: 0=turn-left, 1=straight, 2=turn-right
# New direction = (current + offset) % 4
_REL_OFFSET = np.array([-1, 0, 1], dtype=np.int32)


class VectorizedSnakeEnv:
    def __init__(self, H: int, W: int, N: int, auto_reset: bool = True):
        self.H = H
        self.W = W
        self.N = N
        # auto_reset=True: dead envs reset immediately inside step() (training).
        # auto_reset=False: dead envs freeze on the death frame until reset_dead()
        # is called by the caller (watch mode, so the crash is visible).
        self.auto_reset = auto_reset

        self.bodies: list[deque] = []   # head at index 0
        self.body_sets: list[set] = []
        self.dirs = np.zeros(N, dtype=np.int32)
        self.food = np.zeros((N, 2), dtype=np.int32)
        self.alive = np.ones(N, dtype=np.bool_)

        self._init_all()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        self._init_all()
        self.alive[:] = True
        return self.observation()

    def reset_dead(self):
        """Revive and respawn any frozen-dead envs (auto_reset=False mode)."""
        for i in range(self.N):
            if not self.alive[i]:
                self._reset_single(i)
                self.alive[i] = True

    def _die(self, i: int):
        """Handle a fatal move for env i."""
        if self.auto_reset:
            self._reset_single(i)
        else:
            self.alive[i] = False

    def step(self, actions: np.ndarray):
        """actions: int array (N,) in {0,1,2} — left/straight/right."""
        rewards = np.zeros(self.N, dtype=np.float32)
        dones = np.zeros(self.N, dtype=np.bool_)

        new_dirs = (self.dirs + _REL_OFFSET[actions]) % 4

        for i in range(self.N):
            # Skip frozen-dead envs (auto_reset=False mode)
            if not self.alive[i]:
                dones[i] = True
                continue

            d = new_dirs[i]
            head = self.bodies[i][0]
            nr = head[0] + _DR[d]
            nc = head[1] + _DC[d]
            new_head = (nr, nc)

            # Wall collision
            if nr < 0 or nr >= self.H or nc < 0 or nc >= self.W:
                rewards[i] = -1.0
                dones[i] = True
                self._die(i)
                continue

            eating = (nr == self.food[i, 0] and nc == self.food[i, 1])
            tail = self.bodies[i][-1]

            # Self collision. Moving into the tail's current cell is SAFE when
            # not eating (the tail vacates that cell this step); fatal otherwise.
            if new_head in self.body_sets[i]:
                if new_head != tail or eating:
                    rewards[i] = -1.0
                    dones[i] = True
                    self._die(i)
                    continue

            # Move
            self.bodies[i].appendleft(new_head)
            self.body_sets[i].add(new_head)
            self.dirs[i] = d

            if eating:
                rewards[i] = 1.0
                self._place_food(i)
            else:
                removed = self.bodies[i].pop()
                self.body_sets[i].discard(removed)

        return self.observation(), rewards, dones

    def observation(self) -> np.ndarray:
        """Returns float32 array (N, H, W, 3): body/food/head channels."""
        obs = np.zeros((self.N, self.H, self.W, 3), dtype=np.float32)
        for i in range(self.N):
            for r, c in self.bodies[i]:
                obs[i, r, c, 0] = 1.0
            obs[i, self.food[i, 0], self.food[i, 1], 1] = 1.0
            hr, hc = self.bodies[i][0]
            obs[i, hr, hc, 2] = 1.0
        return obs

    def get_state(self, i: int) -> dict:
        """Single-env state for rendering."""
        return {
            "body": list(self.bodies[i]),
            "food": (int(self.food[i, 0]), int(self.food[i, 1])),
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _init_all(self):
        self.bodies = []
        self.body_sets = []
        for i in range(self.N):
            self.bodies.append(deque())
            self.body_sets.append(set())
            self._reset_single(i)

    def _reset_single(self, i: int):
        H, W = self.H, self.W
        # Random starting position (away from edges so initial body fits)
        margin = 2
        r = np.random.randint(margin, H - margin)
        c = np.random.randint(margin, W - margin)
        d = np.random.randint(4)

        # Build initial body of length 3
        body = deque()
        body_set = set()
        for k in range(3):
            br = r - k * _DR[d]
            bc = c - k * _DC[d]
            br = max(0, min(H - 1, br))
            bc = max(0, min(W - 1, bc))
            pos = (br, bc)
            if pos not in body_set:
                body.append(pos)
                body_set.add(pos)

        self.bodies[i] = body
        self.body_sets[i] = body_set
        self.dirs[i] = d
        self._place_food(i)

    def _place_food(self, i: int):
        H, W = self.H, self.W
        occupied = self.body_sets[i]
        # Rejection sample
        for _ in range(H * W * 4):
            r = np.random.randint(H)
            c = np.random.randint(W)
            if (r, c) not in occupied:
                self.food[i] = [r, c]
                return
        # Fallback: find first free cell
        for r in range(H):
            for c in range(W):
                if (r, c) not in occupied:
                    self.food[i] = [r, c]
                    return
