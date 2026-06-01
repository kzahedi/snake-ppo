## ADDED Requirements

### Requirement: Save checkpoint at fixed step intervals
The checkpoint manager SHALL save a checkpoint every N environment steps (configurable, default 500,000). Each checkpoint SHALL consist of: serialized MLX model weights (`.npz` format via `mx.savez`), optimizer state, current total environment steps, current training metrics (mean reward, mean episode length, mean entropy from the last update), and the run config. Checkpoints SHALL be written to `{run_dir}/checkpoints/step_{step:010d}/`.

#### Scenario: Checkpoint directory is created
- **WHEN** a checkpoint is saved at step 500,000
- **THEN** the directory `checkpoints/step_0000500000/` exists and contains `weights.npz`, `optimizer.npz`, and `meta.json`

#### Scenario: Checkpoint interval is respected
- **WHEN** training runs for 1,500,000 steps with checkpoint_every=500,000
- **THEN** exactly 3 checkpoint directories exist

### Requirement: Load checkpoint for resumption
The checkpoint manager SHALL support loading a checkpoint by step number or by "latest" keyword. Loading SHALL restore model weights, optimizer state, and step counter so training can resume from the exact saved state.

#### Scenario: Load latest checkpoint
- **WHEN** `manager.load("latest")` is called and checkpoints exist
- **THEN** the checkpoint with the highest step number is loaded and the returned step count matches the saved value

#### Scenario: Load by step number
- **WHEN** `manager.load(step=500000)` is called and that checkpoint exists
- **THEN** model weights and optimizer state are restored from that specific checkpoint

#### Scenario: Load from empty run raises error
- **WHEN** `manager.load("latest")` is called and no checkpoints exist
- **THEN** a clear `CheckpointNotFoundError` is raised

### Requirement: List available checkpoints
The checkpoint manager SHALL provide a `list()` method that returns a sorted list of available checkpoint step numbers and their metadata (timestamp, mean reward, mean episode length).

#### Scenario: List returns sorted checkpoints
- **WHEN** checkpoints exist at steps 500k, 1M, and 1.5M
- **THEN** `manager.list()` returns them in ascending step order with metadata dicts

### Requirement: Metadata JSON per checkpoint
Each checkpoint SHALL include a `meta.json` file containing: `step` (int), `wall_time_seconds` (float, elapsed since run start), `mean_reward` (float), `mean_ep_length` (float), `mean_entropy` (float), `config` (dict, copy of run config).

#### Scenario: Meta file is valid JSON
- **WHEN** a checkpoint is saved
- **THEN** `meta.json` is parseable and contains all required keys with correct types
