## ADDED Requirements

### Requirement: Single entry point command
The training pipeline SHALL be invokable as `python -m snake.train --config <path> [--run-dir <path>] [--resume]`. All hyperparameters, grid size, number of environments, checkpoint interval, and video settings SHALL be read from the JSON config file. Command-line flags SHALL be limited to run directory and resume flag.

#### Scenario: Training starts from config
- **WHEN** `python -m snake.train --config configs/overnight.json` is run
- **THEN** a timestamped run directory is created, training begins, and progress is logged to stdout

#### Scenario: Config file controls all hyperparameters
- **WHEN** the config specifies `grid_size: 32` and `num_envs: 256`
- **THEN** the environment is constructed with those parameters without any code changes

### Requirement: Run directory with timestamped name
Each training run SHALL create a directory at `runs/YYYYMMDD_HHMMSS/` containing: `config.json` (copy of the run config), `metrics.jsonl` (one JSON line per PPO update), `checkpoints/` subdirectory, and `videos/` subdirectory.

#### Scenario: Run directory is created on start
- **WHEN** training begins
- **THEN** the run directory and all subdirectories exist before the first environment step

#### Scenario: Config is preserved in run directory
- **WHEN** training begins
- **THEN** `{run_dir}/config.json` is an exact copy of the config file used

### Requirement: Metrics logging to JSONL
The pipeline SHALL append one JSON record per PPO update to `metrics.jsonl`. Each record SHALL contain: `step` (total env steps), `wall_time` (seconds since run start), `mean_reward` (mean episode reward over last rollout), `mean_ep_length` (mean episode length), `mean_entropy` (mean policy entropy), `policy_loss`, `value_loss`, `approx_kl` (approximate KL divergence between old and new policy).

#### Scenario: Metrics file grows during training
- **WHEN** training runs for 10 PPO updates
- **THEN** `metrics.jsonl` contains exactly 10 lines, each valid JSON

#### Scenario: All required fields present
- **WHEN** a metrics record is written
- **THEN** all 8 required fields are present with numeric values

### Requirement: Stdout progress display
The pipeline SHALL print a one-line progress update to stdout after each PPO update. The line SHALL include: step count, steps-per-second throughput, mean reward, mean episode length, and mean entropy.

#### Scenario: Progress line format
- **WHEN** a PPO update completes
- **THEN** a line matching `step=XXXXXXXX  sps=XXXX  reward=X.XX  ep_len=XX.X  entropy=X.XX` is printed

### Requirement: Resume from latest checkpoint
When `--resume` is passed, the pipeline SHALL load the latest checkpoint from the existing run directory, restore model weights, optimizer state, and step counter, and continue training from that step.

#### Scenario: Resume restores step count
- **WHEN** training is resumed after a checkpoint at step 1,000,000
- **THEN** the step counter starts at 1,000,000 and metrics.jsonl is appended (not overwritten)

### Requirement: Graceful interrupt handling
The pipeline SHALL handle SIGINT (Ctrl+C) by completing the current rollout, saving a checkpoint, and exiting cleanly. Partial rollout data SHALL not be discarded without saving.

#### Scenario: Ctrl+C triggers checkpoint save
- **WHEN** the user sends SIGINT during training
- **THEN** a checkpoint is saved before the process exits and the exit message confirms the checkpoint path

### Requirement: Config schema validation
The pipeline SHALL validate the config file on startup and raise a clear error with the offending field name if required fields are missing or have invalid types.

#### Scenario: Missing required field raises error
- **WHEN** a config file omits `grid_size`
- **THEN** the pipeline exits with an error message identifying `grid_size` as missing before any training begins
