export base_lm="hf"
export PYTHONPATH="./:$PYTHONPATH"

CUDA_VISIBLE_DEVICES=0 

for i in {1..3}
do
    log_dir="case_study/multipath/gsm8k/data/cot/qwen/log_run_${i}"
    echo "Starting run $i, log directory: ${log_dir}"
    
    python -m torch.distributed.run --nproc_per_node 1 examples/CoT/gsm8k/inference.py --base_lm $base_lm --temperature 0.01 --log_dir ${log_dir} --model_dir models/qwen-1.5-7B --mode "cot"
    
    echo "Run $i completed"
done