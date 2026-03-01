#!/bin/bash
EXP_NAME="exp1"
BENCHMARK_MODEL="gpt35"
CONFIG_PATH="src/gsm8k/bash/config.json"
NUM_CLUSTERS_MIN=8
NUM_CLUSTERS_MAX=8

python src/gsm8k/sucess.py --exp_name "$EXP_NAME" --benchmark_model "$BENCHMARK_MODEL" --config $CONFIG_PATH --num_clusters_min "$NUM_CLUSTERS_MIN" --num_clusters_max "$NUM_CLUSTERS_MAX"