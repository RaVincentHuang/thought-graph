#!/bin/bash
export base_lm="gpt-3.5-turbo"
export OPENAI_API_BASE="xxx"
export OPENAI_API_KEY="sk-xxx"
export PYTHONPATH="./:$PYTHONPATH"

for i in {1..3}
do
    log_dir="case_study/multipath/gsm8k/data/cot/gpt35/log_run_${i}"
    echo "Starting run $i, log directory: ${log_dir}"
    
    python case_study/multipath/gsm8k/inference.py --base_lm $base_lm --temperature 0.01 --log_dir ${log_dir} --mode "cot"
    
    echo "Run $i completed"
done