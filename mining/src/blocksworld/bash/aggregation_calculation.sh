#!/bin/bash
EXP_NAME="exp1"
BENCHMARK_MODEL="gpt4o"
CLUSTER_ID=9
CONFIG_PATH="src/blocksworld/bash/config.json"
MODE="bfs"
STEP=4

python src/blocksworld/failed.py --exp_name "$EXP_NAME" --benchmark_model "$BENCHMARK_MODEL" --cluster_id "$CLUSTER_ID" --config $CONFIG_PATH --mode "$MODE"

python src/blocksworld/graph/split_graph.py --exp_name "$EXP_NAME" --benchmark_model "$BENCHMARK_MODEL" --cluster_id "$CLUSTER_ID" --config $CONFIG_PATH --mode "$MODE"

python src/blocksworld/graph/pattern_match/aggregation_cal.py --exp_name "$EXP_NAME" --benchmark_model "$BENCHMARK_MODEL" --cluster_id "$CLUSTER_ID" --config $CONFIG_PATH --step $STEP --mode $MODE
