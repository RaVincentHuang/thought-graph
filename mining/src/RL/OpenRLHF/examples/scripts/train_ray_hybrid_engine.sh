set -x
ray start --head  --port=6379 --dashboard-host=127.0.0.1 --dashboard-port=8265 --num-gpus 2
TIME=
Model_Path =
save_path =
data_path =
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"working_dir": "/data/coding/OpenRLHF"}' \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 2 \
   --reward_num_nodes 1 \
   --reward_num_gpus_per_node 2 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 2 \
   --vllm_num_engines 2 \
   --vllm_tensor_parallel_size 1 \
   --colocate_all_models \
   --vllm_gpu_memory_utilization 0.6 \
   --advantage_estimator group_norm \
   --pretrain $Model_Path \
   --save_path $save_path \
   --ckpt_path /data/coding/model_save/ckpt \
   --save_hf_ckpt \
   --remote_rm_url \
   --micro_train_batch_size 4 \
   --train_batch_size 32 \
   --micro_rollout_batch_size 8 \
   --rollout_batch_size 32\
   --n_samples_per_prompt 8 \
   --init_kl_coef 1e-3 \
   --gamma 1.0 \
   --use_kl_loss \
   --kl_estimator k2 \
   --max_epochs 5 \
   --prompt_max_len 1024 \
   --max_samples 100000 \
   --generate_max_len 1024 \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --critic_learning_rate 9e-6 \
   --prompt_data $data_path \
   --input_key context_messages \
   --apply_chat_template \
   --normalize_reward \
   --gradient_checkpointing \
   --packing_samples \
   --vllm_sync_backend nccl \
   --enforce_eager \
   --vllm_enable_sleep \
   --deepspeed_enable_sleep

# You could also try
#   --use_kl_loss \
#   --kl_estimator k3 | k2 \

# also supports --advantage_estimator rloo | reinforce_baseline group_norm dr_grpo
