#!/usr/bin/env bash
# Run Stratus in local mode (single workspace camera, no arm cam)
# Usage: ./scripts/run_local.sh [extra args]
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:$PYTHONPATH"
conda activate rebot
python scripts/run.py \
  --camera 0 \
  --cam-width 1920 \
  --cam-height 1080 \
  --model models/yolov8s-world.pt \
  --conf 0.15 \
  --gripper-id 7 \
  "$@"