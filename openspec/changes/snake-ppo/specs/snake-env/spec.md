## ADDED Requirements

### Requirement: Vectorized parallel environments
The environment SHALL support N simultaneous Snake game instances (default N=256) stepped in lockstep using numpy array operations. Each instance SHALL maintain its own grid state, snake body, direction, and food position. Stepping all instances simultaneously SHALL return batched observations, rewards, and done flags as numpy arrays of shape (N, ...).

#### Scenario: Batch step returns correct shapes
- **WHEN** `env.step(actions)` is called with an integer array of shape (N,)
- **THEN** returned observations have shape (N, H, W, 3), rewards shape (N,), dones shape (N,)

#### Scenario: Dead environments auto-reset
- **WHEN** any environment instance reaches a done state (wall collision or self-collision)
- **THEN** that instance is reset immediately and its next observation reflects a fresh game, while other instances continue uninterrupted

### Requirement: Three-channel grid observation
Each environment instance SHALL produce a float32 grid observation of shape (H, W, 3) where channel 0 is the snake body mask (1.0 at occupied cells, 0.0 elsewhere), channel 1 is the food mask (1.0 at food cell), and channel 2 is the head mask (1.0 at head cell only).

#### Scenario: Channels are mutually consistent
- **WHEN** an observation is produced
- **THEN** channel 2 (head) is a strict subset of channel 0 (body), and channel 1 (food) has exactly one non-zero cell

#### Scenario: Observation range
- **WHEN** any observation is returned
- **THEN** all values are in [0.0, 1.0]

### Requirement: Relative action space
The environment SHALL accept actions in {0, 1, 2} representing turn-left, go-straight, and turn-right relative to the snake's current heading. The environment SHALL reject physically impossible moves by treating a U-turn action identically to straight.

#### Scenario: U-turn prevention
- **WHEN** the snake is heading north and action 0 (which would resolve to south given a particular relative mapping) is applied
- **THEN** the snake continues straight rather than reversing

#### Scenario: Valid relative actions produce correct absolute motion
- **WHEN** the snake is heading east and action 0 (turn left) is applied
- **THEN** the snake moves north on the next step

### Requirement: Configurable grid size
The environment SHALL accept a grid size parameter H×W at construction. SHALL support at minimum 5×5 (minimum viable) through 64×64. The same environment class SHALL work for all supported sizes without modification.

#### Scenario: Small grid initialization
- **WHEN** environment is constructed with grid_size=8
- **THEN** observations have shape (N, 8, 8, 3) and the snake starts with length 3 in a valid position

#### Scenario: Large grid initialization
- **WHEN** environment is constructed with grid_size=32
- **THEN** observations have shape (N, 32, 32, 3) and food is placed in a free cell

### Requirement: Reward signal
The environment SHALL return reward +1.0 when the snake eats food, reward -1.0 on death, and reward 0.0 on all other steps. Food SHALL be replaced at a random free cell immediately after being eaten.

#### Scenario: Food reward on eating
- **WHEN** the snake's head moves onto the food cell
- **THEN** reward is +1.0, snake length increases by 1, new food appears at a free cell

#### Scenario: Death penalty on collision
- **WHEN** the snake's head moves into a wall or its own body
- **THEN** reward is -1.0 and done is True for that instance
