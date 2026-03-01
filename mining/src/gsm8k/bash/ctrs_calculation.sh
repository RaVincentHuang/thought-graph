#!/bin/bash
EXP_NAME="exp1"
BENCHMARK_MODEL="gpt35"
CONFIG_PATH="src/gsm8k/bash/config.json"

python src/gsm8k/graph/pattern_match/ctrs_cal.py --exp_name "$EXP_NAME" --benchmark_model "$BENCHMARK_MODEL" --config $CONFIG_PATH