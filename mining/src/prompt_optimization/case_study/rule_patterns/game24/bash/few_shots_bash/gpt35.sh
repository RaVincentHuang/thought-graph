#!/bin/bash
export OPENAI_BASE_URL="xxx"
export OPENAI_API_KEY="sk-xxx"

python case_study/rule_patterns/game24/inference.py \
    --mode few_shots \
    --base_lm gpt-3.5-turbo \
    --temperature 0.5 \
    --log_dir case_study/rule_patterns/game24/logs/few_shots/gpt35