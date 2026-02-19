#!/bin/bash

# Script to run RL fine-tuning on a pre-trained model

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <experiment_name> [gpu_id]"
    echo "Example: $0 baseline_saxena 0"
    exit 1
fi

EXP_NAME=$1
GPU_ID=${2:-0}

# Paths
DATA_DIR="/Users/manangarg/Downloads/bert_gradient_classifier/data"  # Adjust if needed
CHECKPOINT_PATH="experiments/${EXP_NAME}/best_model.pt"

if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint not found at $CHECKPOINT_PATH"
    echo "Please train the supervised model first using run.sh or train_sequence.py"
    exit 1
fi

echo "Starting RL Fine-tuning for $EXP_NAME on GPU $GPU_ID..."

CUDA_VISIBLE_DEVICES=$GPU_ID python train_rl.py \
    --checkpoint-path "$CHECKPOINT_PATH" \
    --train-path "$DATA_DIR" \
    --val-path "$DATA_DIR" \
    --test-path "$DATA_DIR" \
    --epochs 5 \
    --lr 1e-5 \
    --reward-metric f1_macro \
    --output-dir "experiments_rl"

echo "RL Fine-tuning completed. Results in experiments_rl/${EXP_NAME}_rl"

