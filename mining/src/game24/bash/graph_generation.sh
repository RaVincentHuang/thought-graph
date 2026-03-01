#!/bin/bash
EXP_NAME="exp1"
BENCHMARK_MODEL="qwen"
CLUSTER_ID=9
CONFIG_PATH="src/game24/bash/config.json"
MODE="dfs"

python src/game24/failed.py --exp_name "$EXP_NAME" --benchmark_model "$BENCHMARK_MODEL" --cluster_id "$CLUSTER_ID" --config $CONFIG_PATH --mode "$MODE"
