#!/bin/bash

#SBATCH -p GPU-shared
#SBATCH -t 60:00
#SBATCH --gres=gpu:v100-32:4

export OMP_NUM_THREADS=1
module load AI/pytorch_23.02-1.13.1-py3

# pip install -r requirements.txt

torchrun --nproc_per_node=4 --master_port=3456 train.py \
    --model_name_or_path hf_llama \
    --data_path llama_label_phrases.json \
    --output_dir label_model \
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
    --fsdp "full_shard auto_wrap offload" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer'

echo "DONE"
