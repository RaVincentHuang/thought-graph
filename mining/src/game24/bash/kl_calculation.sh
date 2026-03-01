#!/bin/bash
EXP_NAME="exp1"
BENCHMARK_MODEL="qwen"
CLUSTER_ID=9
CONFIG_PATH="src/game24/bash/config.json"
MODE="bfs"

python src/game24/failed.py --exp_name "$EXP_NAME" --benchmark_model "$BENCHMARK_MODEL" --cluster_id "$CLUSTER_ID" --config $CONFIG_PATH --mode "$MODE"

python src/game24/graph/split_graph.py --exp_name "$EXP_NAME" --benchmark_model "$BENCHMARK_MODEL" --cluster_id "$CLUSTER_ID" --config $CONFIG_PATH --mode "$MODE"

python src/game24/graph/pattern_match/kl_cal.py --exp_name "$EXP_NAME" --benchmark_model "$BENCHMARK_MODEL" --cluster_num $CLUSTER_ID --config $CONFIG_PATH --mode "$MODE"