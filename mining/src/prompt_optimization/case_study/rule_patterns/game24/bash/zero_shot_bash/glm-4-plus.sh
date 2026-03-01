#!/bin/bash
export OPENAI_BASE_URL="xxx"
export OPENAI_API_KEY="sk-xxx"

python case_study/rule_patterns/game24/inference.py \
    --mode zero_shot \
    --base_lm glm-4-plus \
    --temperature 0.5 \
    --log_dir case_study/rule_patterns/game24/logs/zero_shot/glm4