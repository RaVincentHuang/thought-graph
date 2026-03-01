#!/bin/bash
EXP_NAME="exp1"
BENCHMARK_MODEL="qwen"
CONFIG_PATH="src/game24/bash/config.json"
NUM_CLUSTERS_MIN=9
NUM_CLUSTERS_MAX=9

python src/game24/sucess.py --exp_name "$EXP_NAME" --benchmark_model "$BENCHMARK_MODEL" --config $CONFIG_PATH --num_clusters_min "$NUM_CLUSTERS_MIN" --num_clusters_max "$NUM_CLUSTERS_MAX"