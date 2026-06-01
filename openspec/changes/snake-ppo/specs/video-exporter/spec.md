## ADDED Requirements

### Requirement: Per-checkpoint episode video
At each checkpoint, the video exporter SHALL render one complete evaluation episode (greedy policy, no exploration noise) to a `.mp4` file. The video SHALL be saved at `{run_dir}/videos/step_{step:010d}.mp4`. Rendering SHALL use the offscreen moderngl renderer at configurable resolution (default 800×800) and frame rate (default 30 fps). Each game step SHALL produce exactly one frame (no interpolation required for checkpoint videos).

#### Scenario: Video file is created at checkpoint
- **WHEN** a checkpoint is saved at step 500,000
- **THEN** `videos/step_0000500000.mp4` exists and is a valid mp4 with at least 1 frame

#### Scenario: Episode plays to completion
- **WHEN** the video is produced
- **THEN** the final frame shows either the death state or the maximum episode length reached

### Requirement: Dual render variants per checkpoint
The exporter SHALL produce two video variants per checkpoint: a clean game render (`step_XXXXK.mp4`) and a value heatmap overlay render (`step_XXXXK_heatmap.mp4`). Both SHALL cover the same episode trajectory. Heatmap rendering uses the value head evaluated on each observed state.

#### Scenario: Both variants are created
- **WHEN** a checkpoint video is exported
- **THEN** both `step_0000500000.mp4` and `step_0000500000_heatmap.mp4` exist in the videos directory

#### Scenario: Heatmap variant visually differs from clean render
- **WHEN** both variants are rendered for the same episode
- **THEN** corresponding frames differ in pixel content (overlay is applied)

### Requirement: End-of-training timelapse assembly
After training completes (or when explicitly invoked), the exporter SHALL assemble all per-checkpoint clean videos into a single timelapse `.mp4` at `{run_dir}/timelapse.mp4`. Each checkpoint contributes one representative frame (the first frame of its episode video) to the timelapse, played at 24 fps.

#### Scenario: Timelapse contains one frame per checkpoint
- **WHEN** training ran for 200M steps with checkpoint_every=500k (≈400 checkpoints)
- **THEN** `timelapse.mp4` has approximately 400 frames at 24 fps (≈17 seconds)

#### Scenario: Timelapse is assembled in step order
- **WHEN** checkpoint videos exist at steps 500k, 1M, 1.5M
- **THEN** the timelapse plays in ascending step order

### Requirement: Video codec and quality
All videos SHALL be encoded with H.264 codec (libx264) at CRF 18 (high quality) using imageio-ffmpeg. Videos SHALL be playable on macOS QuickTime without additional codecs.

#### Scenario: Output is QuickTime-compatible
- **WHEN** a video is exported
- **THEN** the file can be opened by macOS QuickTime Player

### Requirement: Optional intermediate video cleanup
The exporter SHALL support a configurable `keep_videos` flag (default True). When set to False, per-checkpoint video files SHALL be deleted after the timelapse is assembled, retaining only `timelapse.mp4`.

#### Scenario: Intermediate files deleted when keep_videos=False
- **WHEN** training completes with `keep_videos=False`
- **THEN** `timelapse.mp4` exists but individual `step_*.mp4` files do not
