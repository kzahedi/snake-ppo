## ADDED Requirements

### Requirement: CNN actor-critic network
The policy network SHALL consist of a shared CNN backbone (3 convolutional layers: 32, 64, 64 filters, 3×3 kernel, same padding, ReLU activations) followed by a shared dense trunk (512 units, ReLU), an actor head (linear → softmax, 3 outputs), and a value head (linear → scalar). All parameters SHALL be MLX arrays stored on the GPU device.

#### Scenario: Forward pass output shapes
- **WHEN** a batch of observations of shape (B, H, W, 3) is passed through the network
- **THEN** actor output has shape (B, 3) summing to 1.0 per row, value output has shape (B,)

#### Scenario: Network runs on MLX GPU
- **WHEN** the network is constructed
- **THEN** `mx.default_device()` returns a GPU device and parameter arrays reside on it

### Requirement: PPO clipped surrogate objective
The trainer SHALL implement PPO with a clipped surrogate objective. The policy loss SHALL be `−min(r·A, clip(r, 1−ε, 1+ε)·A)` where r is the probability ratio between new and old policy, A is the advantage estimate, and ε is the clip range (default 0.2). The total loss SHALL be `policy_loss + c_v · value_loss − c_e · entropy` where `c_v=0.5` and `c_e` is the configurable entropy coefficient.

#### Scenario: Clipping prevents large policy updates
- **WHEN** the probability ratio r deviates beyond [1−ε, 1+ε] for some sample
- **THEN** that sample's gradient contribution is clipped and does not drive a larger update than the unclipped bound

#### Scenario: Entropy bonus is included in loss
- **WHEN** a PPO update step is performed
- **THEN** the logged entropy loss is non-zero and has the correct sign (reduces total loss when entropy is high)

### Requirement: Generalized Advantage Estimation (GAE)
The trainer SHALL compute advantages using GAE with configurable γ (discount factor, default 0.99) and λ (GAE lambda, default 0.95). Advantages SHALL be normalized to zero mean and unit variance per mini-batch before computing the policy loss.

#### Scenario: GAE advantage shape
- **WHEN** advantages are computed from a rollout of T steps × N environments
- **THEN** advantage tensor has shape (T, N) before flattening for mini-batch updates

#### Scenario: Advantage normalization
- **WHEN** a mini-batch is sampled from the rollout buffer
- **THEN** the advantages in that mini-batch have mean ≈ 0 and std ≈ 1

### Requirement: Multiple PPO epochs per rollout
The trainer SHALL perform K epochs (default 4) of mini-batch updates over each collected rollout. Each epoch SHALL randomly shuffle the rollout data and split into mini-batches of configurable size (default 4096 samples). Old log probabilities and values SHALL be computed once before all epochs begin and held fixed as the reference.

#### Scenario: K epochs are executed per rollout
- **WHEN** one rollout collection completes
- **THEN** exactly K gradient update passes over the rollout data are performed before the next rollout begins

### Requirement: Configurable entropy coefficient with floor
The trainer SHALL support a configurable entropy coefficient (default 0.01). The trainer SHALL log the mean policy entropy at every update. If entropy drops below a configurable floor (default 0.05 nats) before a configurable step threshold (default 10M env steps), the trainer SHALL emit a warning log but SHALL NOT automatically reset the learning rate (entropy floor is observability only, not control).

#### Scenario: Entropy logging
- **WHEN** a PPO update completes
- **THEN** mean policy entropy is written to the metrics log

#### Scenario: Entropy floor warning
- **WHEN** mean entropy falls below the floor before the step threshold
- **THEN** a warning line is written to the metrics log with current step and entropy value

### Requirement: Adam optimizer on MLX
The trainer SHALL use the Adam optimizer from `mlx.optimizers` with configurable learning rate (default 3e-4). Gradient clipping by global norm SHALL be applied with a configurable max norm (default 0.5) before each optimizer step.

#### Scenario: Optimizer step updates all parameters
- **WHEN** a gradient update is applied
- **THEN** all network parameters change (verified by comparing parameter checksums before and after)
