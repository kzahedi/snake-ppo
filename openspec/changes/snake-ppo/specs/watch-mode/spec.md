## ADDED Requirements

### Requirement: Watch mode entry point
Watch mode SHALL be invokable as `python -m snake.watch --run <run_dir> [--checkpoint <step|latest>] [--heatmap] [--fps <n>] [--loop]`. It SHALL open a moderngl windowed renderer and play one evaluation episode using the greedy policy from the specified checkpoint.

#### Scenario: Watch mode opens a window
- **WHEN** `python -m snake.watch --run runs/20240601_220000 --checkpoint latest` is executed
- **THEN** a window opens and the snake game plays in real time

#### Scenario: Default checkpoint is latest
- **WHEN** `--checkpoint` is not specified
- **THEN** the most recent checkpoint is loaded automatically

### Requirement: Greedy policy evaluation
Watch mode SHALL run the policy in greedy mode (argmax over actor output, no sampling). The episode SHALL play at configurable speed (default 10 steps/second, adjustable via `--fps`). Watch mode SHALL NOT modify any run artefacts.

#### Scenario: Greedy policy runs without randomness
- **WHEN** the same checkpoint and starting seed are used twice
- **THEN** the episode trajectory is identical both times

#### Scenario: Speed is configurable
- **WHEN** `--fps 30` is specified
- **THEN** game steps advance at approximately 30 steps per second

### Requirement: Optional value heatmap overlay in window
When `--heatmap` is passed, the windowed renderer SHALL display the value heatmap overlay. The heatmap SHALL update each step based on the current observation passed through the value head.

#### Scenario: Heatmap updates every step
- **WHEN** `--heatmap` is active and the snake moves
- **THEN** the heatmap colours change to reflect the new observation's value estimates

### Requirement: Loop mode for continuous playback
When `--loop` is passed, watch mode SHALL restart a new episode automatically after each episode ends, continuing until the window is closed.

#### Scenario: Episode restarts after death
- **WHEN** `--loop` is active and the snake dies
- **THEN** a new episode begins within one second without user interaction

### Requirement: Window controls
The watch mode window SHALL respond to: Q or window close → exit; SPACE → pause/unpause episode playback; UP/DOWN arrows → increase/decrease playback speed by 5 steps/second.

#### Scenario: Spacebar pauses playback
- **WHEN** the user presses SPACE during playback
- **THEN** the snake stops moving and the window remains open and responsive

#### Scenario: Arrow keys adjust speed
- **WHEN** the user presses UP arrow
- **THEN** the effective steps-per-second increases by 5
