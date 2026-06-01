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
        self.death_cause = [None] * N   # "wall" | "self" | None (per env)

        # Reward shaping: when enabled, last_shaping[i] holds the per-step change
        # in free-space connectivity Φ (fraction of free cells reachable from the
        # head). Positive = opened space, negative = fragmented/trapped space.
        self.compute_shaping = False
        self.last_shaping = np.zeros(N, dtype=np.float32)
        # Cached connectivity Φ per env: the post-move Φ of step t is the
        # pre-move Φ of step t+1, so we only flood-fill once per step.
        self._conn = np.full(N, np.nan, dtype=np.float32)

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
        self.last_shaping[:] = 0.0

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

            if self.compute_shaping:
                phi_before = self._conn[i]
                if np.isnan(phi_before):
                    phi_before = self._connectivity(self.body_sets[i], head)
            else:
                phi_before = 0.0

            # Wall collision
            if nr < 0 or nr >= self.H or nc < 0 or nc >= self.W:
                rewards[i] = -1.0
                dones[i] = True
                self.death_cause[i] = "wall"
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
                    self.death_cause[i] = "self"
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

            if self.compute_shaping:
                phi_after = self._connectivity(self.body_sets[i], new_head)
                self.last_shaping[i] = phi_after - phi_before
                self._conn[i] = phi_after

        return self.observation(), rewards, dones

    def observation(self) -> np.ndarray:
        """Returns float32 array (N, H, W, 3): body-age / food / head channels.

        Channel 0 encodes segment age as time-until-vacated, normalised to
        (0, 1]: head = 1.0 (stays longest), tail ≈ 1/len (vacates next step).
        This lets the policy reason about which cells free up soon — the key
        signal for safe late-game (near-Hamiltonian) play.
        """
        obs = np.zeros((self.N, self.H, self.W, 3), dtype=np.float32)
        for i in range(self.N):
            body = self.bodies[i]
            n = len(body)
            for j, (r, c) in enumerate(body):
                # j=0 is head; cell vacates in (n - j) steps
                obs[i, r, c, 0] = (n - j) / n
            obs[i, self.food[i, 0], self.food[i, 1], 1] = 1.0
            hr, hc = body[0]
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
        self._conn[i] = np.nan   # invalidate cached connectivity
        self.death_cause[i] = None
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

    def _connectivity(self, body_set: set, head: tuple) -> float:
        """Fraction of free cells reachable from the head via 4-connectivity.

        1.0 = all free space is one region the snake can still reach (safe);
        < 1.0 = the snake has fragmented the board, sealing off free cells
        it can no longer get to (a trap in the making).
        """
        H, W = self.H, self.W
        total_free = H * W - len(body_set)
        if total_free <= 0:
            return 1.0
        seen = {head}
        queue = deque([head])
        reached = 0
        while queue:
            r, c = queue.popleft()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if (0 <= nr < H and 0 <= nc < W
                        and (nr, nc) not in seen and (nr, nc) not in body_set):
                    seen.add((nr, nc))
                    reached += 1
                    queue.append((nr, nc))
        return reached / total_free
