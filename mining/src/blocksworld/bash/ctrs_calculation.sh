#!/bin/bash
EXP_NAME="exp1"
BENCHMARK_MODEL="gpt4o"
CONFIG_PATH="src/blocksworld/bash/config.json"
STEP=4

python src/blocksworld/graph/pattern_match/ctrs_cal.py --exp_name "$EXP_NAME" --benchmark_model "$BENCHMARK_MODEL" --config $CONFIG_PATH --step $STEP