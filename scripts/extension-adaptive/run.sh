#!/bin/bash
# Atom I1 driver: sweep dynamic_threshold_lambda across four values and capture
# per layer entropy vs token retention. Inference only.
set -e
cd "$(dirname "$0")/../.."
mkdir -p logs/extensions/adaptive_thresholds output_figures/improvement/adaptive_thresholds

python src/experiment/extension_adaptive/measure_retention.py \
    --base_model checkpoints/llama2 \
    --peft_model checkpoints/peft_model/rp/8k/jenga \
    --predictor_path checkpoints/predictor \
    --seq_len 8192 \
    --n_docs 4 \
    --lams 0.0 0.05 0.1 0.2

python src/experiment/extension_adaptive/plot_scatter.py
