#!/bin/bash
# Atom I3 driver: compare Jenga baseline vs Jenga + Token Merging on the same
# Llama 2 7B + Jenga adapter at 8K context. Inference only.
set -e
cd "$(dirname "$0")/../.."
mkdir -p logs/extensions/token_merging output_figures/extensions/token_merging

python src/experiment/extension_token_merging/measure_merge.py \
    --base_model checkpoints/llama2 \
    --peft_model checkpoints/peft_model/rp/8k/jenga \
    --predictor_path checkpoints/predictor \
    --seq_len 8192 \
    --n_docs 4
