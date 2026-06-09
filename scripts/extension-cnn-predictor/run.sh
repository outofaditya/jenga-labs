#!/bin/bash
# Atom I2 driver: train MLP and CNN attention predictors head to head on a
# fixed RedPajama ground truth cache extracted from Llama 2 7B.
set -e
cd "$(dirname "$0")/../.."
mkdir -p logs/extensions/cnn_predictor output_figures/extensions/cnn_predictor

python src/experiment/extension_cnn_predictor/train_both.py \
    --model_path "${MODEL_PATH:-checkpoints/opt-1.3b}" \
    --seq_len "${SEQ_LEN:-2048}" \
    --n_docs "${N_DOCS:-4}" \
    --epochs "${EPOCHS:-200}" \
    --lr "${LR:-1e-3}" \
    --seeds 0 1 2

python src/experiment/extension_cnn_predictor/plot_loss.py
