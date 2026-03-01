#!/bin/bash
EXP_NAME="exp1"
BENCHMARK_MODEL="gpt35"
CLUSTER_ID=8
CONFIG_PATH="src/gsm8k/bash/config.json"
MODE="bfs"

python src/gsm8k/failed.py --exp_name "$EXP_NAME" --benchmark_model "$BENCHMARK_MODEL" --cluster_id "$CLUSTER_ID" --config $CONFIG_PATH --mode "$MODE"

python src/gsm8k/graph/split_graph.py --exp_name "$EXP_NAME" --benchmark_model "$BENCHMARK_MODEL" --cluster_id "$CLUSTER_ID" --config $CONFIG_PATH --mode "$MODE"

python src/gsm8k/graph/pattern_match/kl_cal.py --exp_name "$EXP_NAME" --benchmark_model "$BENCHMARK_MODEL" --cluster_num $CLUSTER_ID --config $CONFIG_PATH --mode "$MODE"