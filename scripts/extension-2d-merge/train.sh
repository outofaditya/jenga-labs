#!/bin/bash
# Atom I5: train a LoRA adapter on the 2D sparsity model (Jenga token
# sparsity composed with LongLoRA shifted attention) with post hoc
# broadcast merging enabled from step 0. 2400 steps. Output writes to
# checkpoints/peft_model_2d_merge so the I4 adapter is preserved.
set -e
cd "$(dirname "$0")/../.."

model=${1:-"llama2"}
max_length=${2:-8192}
device=${3:-"a6000"}

mkdir -p logs/extensions/token_merging_2d checkpoints/peft_model_2d_merge
if [[ "${PYTORCH_CUDA_ALLOC_CONF}" != *"expandable_segments:True"* ]]; then
    export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
fi

log_file="logs/extensions/token_merging_2d/train-${model}-${max_length}-${device}.log"

python src/experiment/extension_2d_merge/train_2d_merge.py \
    --model_name_or_path "checkpoints/${model}" \
    --predictor_path checkpoints/predictor \
    --output_dir checkpoints/peft_model_2d_merge \
    --max_steps 2400 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --save_strategy "steps" \
    --save_steps 1200 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --weight_decay 0.0 \
    --warmup_steps 20 \
    --lr_scheduler_type constant_with_warmup \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --bf16 \
    --model_max_length "${max_length}" \
    --flash_attention True \
    --pool_size 64 \
    --thresh 0.4 \
    --gradient_checkpoint False \
    --logging_steps 20 \
    > "${log_file}" 2>&1
