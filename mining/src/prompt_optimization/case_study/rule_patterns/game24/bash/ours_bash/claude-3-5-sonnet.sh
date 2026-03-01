#!/bin/bash
export OPENAI_BASE_URL="xxx"
export OPENAI_API_KEY="sk-xxx"

python case_study/rule_patterns/game24/inference.py \
    --mode ours \
    --base_lm claude-3-5-sonnet-20240620 \
    --temperature 0.5 \
    --log_dir case_study/rule_patterns/game24/logs/ours/claude