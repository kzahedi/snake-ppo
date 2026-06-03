#!/bin/bash
# Sequential comparison pipeline: train A2C, then DQN, then neuroevolution,
# then regenerate the comparison chart + behaviour video across all approaches
# (including the already-trained PPO at runs/solve). Each stage writes a marker
# file so the monitor can tell which stage is running.
set -e
cd "$(dirname "$0")/.."
RUN="conda run -n snake python -m"
mkdir -p runs

echo "STAGE a2c"   > runs/.pipeline_stage
$RUN snake.train     --config configs/a2c.json --run-dir runs/a2c --no-video

echo "STAGE dqn"   > runs/.pipeline_stage
$RUN snake.train_dqn --config configs/dqn.json --run-dir runs/dqn

echo "STAGE evo"   > runs/.pipeline_stage
$RUN snake.train_evo --config configs/evo.json --run-dir runs/evo

echo "STAGE compare" > runs/.pipeline_stage
$RUN snake.compare --grid 8 --episodes 50 \
    --ppo "runs/solve:PPO" \
    --shielded "runs/solve:PPO+shield" \
    --ppo "runs/a2c:A2C" \
    --dqn "runs/dqn:DQN" \
    --ppo "runs/evo:Evolution"

echo "STAGE done"  > runs/.pipeline_stage
echo "PIPELINE COMPLETE"
