#!/bin/bash
export base_lm="hf"
export PYTHONPATH="./:$PYTHONPATH"

CUDA_VISIBLE_DEVICES=0 

for i in {1..3}
do
    log_dir="case_study/multipath/gsm8k/data/ours/glm4/log_run_${i}"
    echo "Starting run $i, log directory: ${log_dir}"
    
    python -m torch.distributed.run --nproc_per_node 1 case_study/multipath/gsm8k/inference.py --base_lm $base_lm --temperature 0.01 --log_dir ${log_dir} --model_dir models/glm-4-9b --mode "ours"
    
    echo "Run $i completed"
done