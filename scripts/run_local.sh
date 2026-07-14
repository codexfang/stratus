#!/usr/bin/env bash
# Run Stratus in local mode (no AWS)
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:$PYTHONPATH"
conda activate rebot
python -m stratus.pipeline.engine
