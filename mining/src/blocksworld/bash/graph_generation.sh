#!/bin/bash
EXP_NAME="exp1"
BENCHMARK_MODEL="gpt4o"
CLUSTER_ID=9
CONFIG_PATH="src/blocksworld/bash/config.json"
MODE="dfs"

python src/blocksworld/failed.py --exp_name "$EXP_NAME" --benchmark_model "$BENCHMARK_MODEL" --cluster_id "$CLUSTER_ID" --config $CONFIG_PATH --mode "$MODE"
