#!/bin/bash
EXP_NAME="exp1"
BENCHMARK_MODEL="qwen"
CONFIG_PATH="src/game24/bash/config.json"

python src/game24/graph/pattern_match/ctrs_cal.py --exp_name "$EXP_NAME" --benchmark_model "$BENCHMARK_MODEL" --config $CONFIG_PATH