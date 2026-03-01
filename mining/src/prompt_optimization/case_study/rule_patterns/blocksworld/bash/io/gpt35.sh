#!/bin/bash
export OPENAI_BASE_URL="xxx"
export OPENAI_API_KEY="sk-xxx"
export PYTHONPATH="./:$PYTHONPATH"
export VAL="LLMs-Planning/planner_tools/VAL"

timestamp=$(date +"%Y%m%d_%H%M%S")
log_dir="case_study/rule_patterns/blocksworld/logs/io/gpt35_${timestamp}"

python case_study/rule_patterns/blocksworld/cot_inference.py \
    --mode_type io \
    --base_lm gpt-3.5-turbo \
    --data_path 'case_study/rule_patterns/blocksworld/data/split_v1/split_v1_step_4_data.json' \
    --log_dir "${log_dir}" \
    --prompt_path "case_study/rule_patterns/blocksworld/prompts/pool_prompt_v2_step_4.json" \
    --temperature 0.8