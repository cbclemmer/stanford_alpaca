#!/bin/bash

#SBATCH -p GPU-shared
#SBATCH -t 2:00:00
#SBATCH --gres=gpu:v100-32:2

if [ ! -e "hf_llama/7B" ]; then
  echo "Error: Must have huggingface converted llama model in hf_llama/7B"
  exit 1
fi

if [ ! -e "fine_tune_data.json" ]; then
  echo "Error: Must have fine tune data in fine_tune_data.json"
  exit 1
fi

export OMP_NUM_THREADS=1
module load AI/pytorch_23.02-1.13.1-py3

export PATH="/opt/packages/cuda/v11.7.1/bin:$PATH"
export LD_LIBRARY_PATH="/opt/packages/cuda/v11.7.1/lib64:$LD_LIBRARY_PATH"

pip install -r requirements.txt
pip install deepspeed
pip install accelerate
# pip freeze

torchrun --nproc_per_node=2 --master_port=3456 train.py \
    --model_name_or_path hf_llama/7B \
    --data_path fine_tune_data.json \
    --output_dir fine_tuned_model \
    --fp16 True \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --deepspeed "./configs/default_offload_opt_param.json"

echo "DONE"
